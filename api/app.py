
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = OUTPUT_DIR / "catboost_unweighted_model.cbm"

NUMERIC_FEATURES = [
    "days_since_purchase",
    "claim_amount_inr",
    "customer_prior_claims",
    "partner_tenure_days",
    "claim_to_product_price_ratio",
    "list_price_inr",
    "warranty_months",
    "claim_month",
    "claim_weekday",
    "is_low_value_claim",
    "is_after_policy_change",
    "partner_prior_claim_count"
]

CATEGORICAL_FEATURES = [
    "photo_attached",
    "source",
    "partner_type",
    "city",
    "family"
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# LOAD MODEL AND REFERENCE DATA
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = CatBoostClassifier()
model.load_model(str(MODEL_PATH))

partners = pd.read_csv(DATA_DIR / "partners.csv")
products = pd.read_csv(DATA_DIR / "products.csv")
historical_claims = pd.read_csv(DATA_DIR / "train.csv")

partners["onboarded_date"] = pd.to_datetime(
    partners["onboarded_date"], errors="coerce"
)

historical_claims = historical_claims[
    historical_claims["is_fraud"].isin([0, 1])
].copy()

historical_claims["submitted_at"] = pd.to_datetime(
    historical_claims["submitted_at"], errors="coerce"
)

historical_claims["claim_id"] = (
    historical_claims["claim_id"]
    .astype(str).str.strip().str.upper()
)

historical_claims = (
    historical_claims
    .sort_values("submitted_at")
    .drop_duplicates("claim_id", keep="first")
)

print("CatBoost model loaded.")
print("Partner records:", len(partners))
print("Product records:", len(products))
print("Historical claims:", len(historical_claims))


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Kestrel Warranty Claim Risk API",
    description=(
        "Fraud-risk scoring for warranty claims using "
        "a locally hosted CatBoost model."
    ),
    version="1.0.0"
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class ClaimRequest(BaseModel):

    claim_id: str = Field(..., min_length=1)
    submitted_at: datetime

    partner_id: str = Field(..., min_length=1)
    sku: str = Field(..., min_length=1)

    product_serial: str = ""

    days_since_purchase: float = Field(..., ge=0)
    claim_amount_inr: float = Field(..., gt=0)

    customer_prior_claims: int = Field(..., ge=0)

    photo_attached: str
    source: str

# ============================================================
# WEB INTERFACE
# ============================================================

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE_DIR / "api" / "index.html")
# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_category(value: Any) -> str:
    if pd.isna(value):
        return "unknown"

    return str(value).strip().lower()


def get_partner_history_count(
    partner_id: str,
    submitted_at: pd.Timestamp
) -> int:
    """
    Count labelled historical claims from this partner
    submitted before the incoming claim.

    Same-timestamp ordering is not reconstructed here;
    this is a practical inference-time approximation.
    """

    previous = historical_claims[
        (historical_claims["partner_id"].astype(str) == partner_id)
        & (historical_claims["submitted_at"] < submitted_at)
    ]

    return int(len(previous))


def build_features(claim: ClaimRequest) -> pd.DataFrame:

    partner_id = str(claim.partner_id).strip()
    sku = str(claim.sku).strip()

    partner_match = partners[
        partners["partner_id"].astype(str) == partner_id
    ]

    product_match = products[
        products["sku"].astype(str) == sku
    ]

    if partner_match.empty:
        raise HTTPException(
            status_code=404,
            detail="Partner ID not found in reference data."
        )

    if product_match.empty:
        raise HTTPException(
            status_code=404,
            detail="Product SKU not found in reference data."
        )

    partner = partner_match.iloc[0]
    product = product_match.iloc[0]

    submitted_at = pd.Timestamp(claim.submitted_at)

    if submitted_at.tzinfo is not None:
        submitted_at = submitted_at.tz_localize(None)

    onboarded_date = partner["onboarded_date"]

    if pd.isna(onboarded_date):
        partner_tenure_days = np.nan
    else:
        partner_tenure_days = (
            submitted_at - onboarded_date
        ).days

    list_price = pd.to_numeric(
        product["list_price_inr"], errors="coerce"
    )

    if pd.isna(list_price) or list_price == 0:
        price_ratio = np.nan
    else:
        price_ratio = claim.claim_amount_inr / list_price

    features = {
        "days_since_purchase": claim.days_since_purchase,
        "claim_amount_inr": claim.claim_amount_inr,
        "customer_prior_claims": claim.customer_prior_claims,
        "partner_tenure_days": partner_tenure_days,
        "claim_to_product_price_ratio": price_ratio,
        "list_price_inr": list_price,
        "warranty_months": pd.to_numeric(
            product["warranty_months"], errors="coerce"
        ),
        "claim_month": submitted_at.month,
        "claim_weekday": submitted_at.weekday(),
        "is_low_value_claim": int(
            claim.claim_amount_inr < 2000
        ),
        "is_after_policy_change": int(
            submitted_at >= pd.Timestamp("2026-05-01")
        ),
        "partner_prior_claim_count": get_partner_history_count(
            partner_id, submitted_at
        ),
        "photo_attached": normalize_category(
            claim.photo_attached
        ),
        "source": normalize_category(claim.source),
        "partner_type": normalize_category(
            partner["partner_type"]
        ),
        "city": normalize_category(partner["city"]),
        "family": normalize_category(product["family"])
    }

    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


# ============================================================
# HUMAN-READABLE REVIEW SIGNALS
# ============================================================

def generate_review_signals(claim: ClaimRequest) -> list[str]:

    reasons = []

    if claim.claim_amount_inr >= 5000:
        reasons.append(
            "The requested claim amount is relatively high."
        )

    if claim.days_since_purchase <= 90:
        reasons.append(
            "The claim was submitted within 90 days of purchase."
        )

    if claim.customer_prior_claims >= 2:
        reasons.append(
            "The customer has submitted multiple previous claims."
        )

    if normalize_category(claim.photo_attached) in {
        "no", "n", "false", "0"
    }:
        reasons.append(
            "No photo attachment was indicated."
        )

    if not reasons:
        reasons.append(
            "No specific rule-based review signal was triggered."
        )

    return reasons


def assign_risk_level(score: float) -> str:
    """
    Provisional operational bands, not calibrated
    probabilities or validated fraud decision thresholds.
    """

    if score >= 0.03:
        return "High"

    if score >= 0.01:
        return "Medium"

    return "Low"


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "CatBoost",
        "model_loaded": True
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(claim: ClaimRequest):

    try:
        X = build_features(claim)

        score = float(
            model.predict_proba(X)[0, 1]
        )

        return {
            "claim_id": claim.claim_id.strip().upper(),
            "fraud_risk_score": round(score, 6),
            "risk_level": assign_risk_level(score),
            "review_signals": generate_review_signals(claim),
            "recommendation": (
                "Consider manual review based on the score "
                "and available claim information."
            ),
            "disclaimer": (
                "This is a model-generated risk assessment, "
                "not a confirmed fraud determination. "
                "Risk bands are provisional."
            )
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )
