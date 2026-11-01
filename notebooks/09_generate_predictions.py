
# ============================================================
# KESTREL HOME WARRANTY CLAIM REVIEW
# FINAL TEST PREDICTION GENERATION
# Model: Unweighted CatBoost baseline
# ============================================================

import pandas as pd
import numpy as np

from pathlib import Path
from catboost import CatBoostClassifier

# ============================================================
# STEP 1: PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = OUTPUT_DIR / "catboost_unweighted_model.cbm"
OUTPUT_PATH = BASE_DIR / "predictions.csv"

# ============================================================
# STEP 2: LOAD DATA
# ============================================================

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test_unlabelled.csv")

partners = pd.read_csv(DATA_DIR / "partners.csv")
products = pd.read_csv(DATA_DIR / "products.csv")

print("Original training shape:", train.shape)
print("Original test shape:", test.shape)

# Keep original test order
test["_original_order"] = np.arange(len(test))

# ============================================================
# STEP 3: LABEL CLEANING
# ============================================================

train = train[train["is_fraud"].isin([0, 1])].copy()
train["is_fraud"] = train["is_fraud"].astype(int)

for df in [train, test]:
    df["claim_id"] = (
        df["claim_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

# ============================================================
# STEP 4: DATE AND SERIAL CLEANING
# ============================================================

for df in [train, test]:

    df["submitted_at"] = pd.to_datetime(
        df["submitted_at"],
        errors="coerce"
    )

    df["claim_month"] = df["submitted_at"].dt.month
    df["claim_weekday"] = df["submitted_at"].dt.weekday

    df["normalized_serial"] = (
        df["product_serial"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )

# ============================================================
# STEP 5: MERGE PARTNER AND PRODUCT DATA
# ============================================================

partners["onboarded_date"] = pd.to_datetime(
    partners["onboarded_date"],
    errors="coerce"
)

train = train.merge(
    partners,
    on="partner_id",
    how="left",
    validate="many_to_one"
)

test = test.merge(
    partners,
    on="partner_id",
    how="left",
    validate="many_to_one"
)

train = train.merge(
    products,
    on="sku",
    how="left",
    validate="many_to_one"
)

test = test.merge(
    products,
    on="sku",
    how="left",
    validate="many_to_one"
)

# ============================================================
# STEP 6: BUSINESS FEATURES
# ============================================================

for df in [train, test]:

    df["partner_tenure_days"] = (
        df["submitted_at"] - df["onboarded_date"]
    ).dt.days

    df["claim_to_product_price_ratio"] = (
        df["claim_amount_inr"] / df["list_price_inr"]
    )

    df["is_low_value_claim"] = (
        df["claim_amount_inr"] < 2000
    ).astype(int)

    df["is_after_policy_change"] = (
        df["submitted_at"] >= pd.Timestamp("2026-05-01")
    ).astype(int)

# ============================================================
# STEP 7: CATEGORICAL STANDARDIZATION
# ============================================================

categorical_features = [
    "photo_attached",
    "source",
    "partner_type",
    "city",
    "family"
]

for df in [train, test]:

    for col in categorical_features:

        df[col] = (
            df[col]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

# ============================================================
# STEP 8: REMOVE TRAINING DUPLICATES
# ============================================================

train = (
    train.sort_values("submitted_at")
    .drop_duplicates(subset="claim_id", keep="first")
    .copy()
    .reset_index(drop=True)
)

test = test.reset_index(drop=True)

# ============================================================
# STEP 9: CHRONOLOGICAL PARTNER HISTORY
# ============================================================

train["dataset"] = "train"
test["dataset"] = "test"

history = pd.concat(
    [train, test],
    ignore_index=True,
    sort=False
)

history = history.sort_values(
    "submitted_at",
    kind="mergesort"
).copy()

history["partner_prior_claim_count"] = (
    history.groupby("partner_id").cumcount()
)

train = (
    history[history["dataset"] == "train"]
    .drop(columns="dataset")
    .reset_index(drop=True)
)

test = (
    history[history["dataset"] == "test"]
    .drop(columns="dataset")
    .reset_index(drop=True)
)

# ============================================================
# STEP 10: MODEL FEATURES
# ============================================================

numeric_features = [
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

feature_columns = numeric_features + categorical_features

X_test = test[feature_columns].copy()

for col in numeric_features:
    X_test[col] = pd.to_numeric(
        X_test[col],
        errors="coerce"
    )

print("\nTest feature shape:", X_test.shape)

# ============================================================
# STEP 11: LOAD SAVED CATBOOST MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Saved model not found: {MODEL_PATH}"
    )

model = CatBoostClassifier()
model.load_model(str(MODEL_PATH))

print("\nSaved CatBoost model loaded successfully.")

# ============================================================
# STEP 12: GENERATE FRAUD PROBABILITY SCORES
# ============================================================

scores = model.predict_proba(X_test)[:, 1]

# ============================================================
# STEP 13: CREATE FINAL SUBMISSION FILE
# ============================================================

predictions = pd.DataFrame({
    "claim_id": test["claim_id"],
    "score": scores
})

predictions["_original_order"] = test["_original_order"]

predictions = (
    predictions
    .sort_values("_original_order")
    .drop(columns="_original_order")
    .reset_index(drop=True)
)

predictions.to_csv(OUTPUT_PATH, index=False)

# ============================================================
# STEP 14: VALIDATION CHECKS
# ============================================================

print("\n========== FINAL PREDICTION CHECKS ==========")

print("Predictions generated:", len(predictions))
print("Unique claim IDs:", predictions["claim_id"].nunique())
print("Missing values:", predictions.isnull().sum().sum())

print("Minimum score:", predictions["score"].min())
print("Maximum score:", predictions["score"].max())
print("Mean score:", predictions["score"].mean())

print("\nOutput columns:", predictions.columns.tolist())

assert len(predictions) == len(test), (
    "Prediction count does not match test data."
)

assert predictions["claim_id"].is_unique, (
    "Duplicate claim IDs found in predictions."
)

assert predictions["score"].between(0, 1).all(), (
    "Scores must be between 0 and 1."
)

assert predictions.isnull().sum().sum() == 0, (
    "Missing values found in final predictions."
)

assert predictions.columns.tolist() == ["claim_id", "score"]

print("\nFirst five predictions:")
print(predictions.head())

print("\nFinal predictions saved to:", OUTPUT_PATH)
print("All prediction checks passed!")
