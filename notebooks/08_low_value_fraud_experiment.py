
# ============================================================
# KESTREL HOME WARRANTY CLAIM REVIEW
# LOW-VALUE FRAUD DETECTION EXPERIMENT
# ============================================================

import pandas as pd
import numpy as np

from pathlib import Path
from catboost import CatBoostClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)

# ============================================================
# 1. PATHS AND DATA LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA_DIR / "train.csv")
partners = pd.read_csv(DATA_DIR / "partners.csv")
products = pd.read_csv(DATA_DIR / "products.csv")

print("Original training shape:", train.shape)

# ============================================================
# 2. LABEL CLEANING
# ============================================================

train = train[train["is_fraud"].isin([0, 1])].copy()
train["is_fraud"] = train["is_fraud"].astype(int)

train["claim_id"] = (
    train["claim_id"].astype(str).str.strip().str.upper()
)

# ============================================================
# 3. DATE AND SERIAL CLEANING
# ============================================================

train["submitted_at"] = pd.to_datetime(
    train["submitted_at"], errors="coerce"
)

train["claim_month"] = train["submitted_at"].dt.month
train["claim_weekday"] = train["submitted_at"].dt.weekday

train["normalized_serial"] = (
    train["product_serial"]
    .fillna("")
    .astype(str)
    .str.upper()
    .str.replace(r"[^A-Z0-9]", "", regex=True)
)

# ============================================================
# 4. REFERENCE DATA MERGING
# ============================================================

partners["onboarded_date"] = pd.to_datetime(
    partners["onboarded_date"], errors="coerce"
)

train = train.merge(
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

# ============================================================
# 5. BUSINESS FEATURES
# ============================================================

train["partner_tenure_days"] = (
    train["submitted_at"] - train["onboarded_date"]
).dt.days

train["claim_to_product_price_ratio"] = (
    train["claim_amount_inr"] / train["list_price_inr"]
)

train["is_low_value_claim"] = (
    train["claim_amount_inr"] < 2000
).astype(int)

train["is_after_policy_change"] = (
    train["submitted_at"] >= pd.Timestamp("2026-05-01")
).astype(int)

# ============================================================
# 6. CATEGORICAL STANDARDIZATION
# ============================================================

categorical_features = [
    "photo_attached",
    "source",
    "partner_type",
    "city",
    "family"
]

for col in categorical_features:
    train[col] = (
        train[col]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

# ============================================================
# 7. DEDUPLICATION
# ============================================================

train = (
    train.sort_values("submitted_at")
    .drop_duplicates(subset="claim_id", keep="first")
    .copy()
    .reset_index(drop=True)
)

print("\nUnique training claims:", len(train))

# ============================================================
# 8. PARTNER HISTORY
# ============================================================

train["dataset"] = "train"

history = train.sort_values(
    "submitted_at", kind="mergesort"
).copy()

history["partner_prior_claim_count"] = (
    history.groupby("partner_id").cumcount()
)

train = history.drop(columns="dataset").reset_index(drop=True)

# ============================================================
# 9. TEMPORAL TRAIN / VALIDATION SPLIT
# ============================================================

validation_start = pd.Timestamp("2026-04-01")
validation_end = pd.Timestamp("2026-07-01")

train_data = train[
    (train["submitted_at"] < validation_start) &
    (train["source"] == "crm")
].copy()

validation_data = train[
    (train["submitted_at"] >= validation_start) &
    (train["submitted_at"] < validation_end)
].copy()

# ============================================================
# 10. FEATURE SELECTION
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

# Convert numeric columns
for col in numeric_features:
    train_data[col] = pd.to_numeric(
        train_data[col], errors="coerce"
    )

    validation_data[col] = pd.to_numeric(
        validation_data[col], errors="coerce"
    )

# ============================================================
# 11. CREATE LOW-VALUE TRAINING DATA
# ============================================================

low_train = train_data[
    train_data["claim_amount_inr"] < 2000
].copy()

low_validation = validation_data[
    validation_data["claim_amount_inr"] < 2000
].copy()

print("\n========== LOW-VALUE DATA SUMMARY ==========")

print("Low-value training claims:", len(low_train))
print("Low-value training fraud:", low_train["is_fraud"].sum())

print("Low-value validation claims:", len(low_validation))
print("Low-value validation fraud:", low_validation["is_fraud"].sum())

# ============================================================
# 12. CHECK CLASS COUNTS
# ============================================================

if low_train["is_fraud"].nunique() < 2:
    raise ValueError(
        "Low-value training data does not contain both classes."
    )

if low_validation["is_fraud"].nunique() < 2:
    raise ValueError(
        "Low-value validation data does not contain both classes."
    )

# ============================================================
# 13. PREPARE FEATURES
# ============================================================

X_train = low_train[feature_columns].copy()
y_train = low_train["is_fraud"].copy()

X_val = low_validation[feature_columns].copy()
y_val = low_validation["is_fraud"].copy()

cat_features_indices = [
    X_train.columns.get_loc(col)
    for col in categorical_features
]

# ============================================================
# 14. TRAIN LOW-VALUE CATBOOST
# ============================================================

low_model = CatBoostClassifier(
    iterations=700,
    depth=4,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="AUC",
    l2_leaf_reg=5,
    random_seed=42,
    verbose=100,
    allow_writing_files=False
)

print("\nTraining low-value CatBoost model...")

low_model.fit(
    X_train,
    y_train,
    cat_features=cat_features_indices,
    eval_set=(X_val, y_val),
    verbose=100,
    early_stopping_rounds=100
)

print("Best low-value iteration:", low_model.get_best_iteration())

# ============================================================
# 15. LOW-VALUE MODEL PREDICTIONS
# ============================================================

low_scores = low_model.predict_proba(X_val)[:, 1]

low_pr_auc = average_precision_score(y_val, low_scores)
low_roc_auc = roc_auc_score(y_val, low_scores)

print("\n========== LOW-VALUE MODEL METRICS ==========")
print("PR-AUC:", round(low_pr_auc, 6))
print("ROC-AUC:", round(low_roc_auc, 6))

# ============================================================
# 16. LOAD EXISTING BASELINE PREDICTIONS
# ============================================================

baseline = pd.read_csv(
    OUTPUT_DIR / "catboost_validation_predictions.csv"
)

baseline["claim_id"] = baseline["claim_id"].astype(str)
low_validation["claim_id"] = low_validation["claim_id"].astype(str)

low_output = low_validation[
    ["claim_id", "submitted_at", "is_fraud", "claim_amount_inr"]
].copy()

low_output["low_model_score"] = low_scores

baseline = baseline.rename(
    columns={"catboost_score": "baseline_score"}
)

# Merge low-value predictions into the baseline
combined = baseline.merge(
    low_output[["claim_id", "low_model_score"]],
    on="claim_id",
    how="left",
    validate="one_to_one"
)

# ============================================================
# 17. CREATE THREE SCORING STRATEGIES
# ============================================================

combined["baseline_score"] = pd.to_numeric(
    combined["baseline_score"], errors="coerce"
)

combined["low_model_score"] = pd.to_numeric(
    combined["low_model_score"], errors="coerce"
)

combined["month"] = (
    pd.to_datetime(combined["submitted_at"])
    .dt.to_period("M")
)

combined["low_value"] = (
    combined["claim_amount_inr"] < 2000
)

# Baseline scores for every claim
combined["score_baseline"] = combined["baseline_score"]

# Low-value model is used only for low-value claims.
# Other claims retain their baseline scores.
combined["score_combined"] = np.where(
    combined["low_value"],
    combined["low_model_score"],
    combined["baseline_score"]
)

# Low-value-only model:
# Only low-value claims are eligible for review.
combined["score_low_only"] = np.where(
    combined["low_value"],
    combined["low_model_score"],
    -1
)

# ============================================================
# 18. TOP-40 MONTHLY EVALUATION
# ============================================================

score_columns = {
    "Baseline CatBoost": "score_baseline",
    "Low-value only": "score_low_only",
    "Combined scoring": "score_combined"
}

monthly_results = []

for month, month_data in combined.groupby("month"):

    total_fraud = int(month_data["is_fraud"].sum())

    for strategy_name, score_column in score_columns.items():

        selected = month_data.nlargest(
            40, score_column
        )

        detected = int(selected["is_fraud"].sum())

        monthly_results.append({
            "month": str(month),
            "strategy": strategy_name,
            "claims_reviewed": len(selected),
            "total_fraud": total_fraud,
            "fraud_detected": detected,
            "fraud_missed": total_fraud - detected,
            "precision_percent": round(
                detected / len(selected) * 100, 2
            ) if len(selected) else 0,
            "recall_percent": round(
                detected / total_fraud * 100, 2
            ) if total_fraud else 0
        })

monthly_results = pd.DataFrame(monthly_results)

print("\n========== MONTHLY RESULTS ==========")
print(monthly_results.to_string(index=False))

# ============================================================
# 19. OVERALL RESULTS
# ============================================================

overall = (
    monthly_results.groupby("strategy")
    .agg(
        claims_reviewed=("claims_reviewed", "sum"),
        fraud_detected=("fraud_detected", "sum"),
        fraud_missed=("fraud_missed", "sum"),
        total_fraud=("total_fraud", "sum")
    )
    .reset_index()
)

overall["precision_percent"] = (
    overall["fraud_detected"] /
    overall["claims_reviewed"] * 100
).round(2)

overall["recall_percent"] = (
    overall["fraud_detected"] /
    overall["total_fraud"] * 100
).round(2)

print("\n========== OVERALL COMPARISON ==========")
print(overall.to_string(index=False))

# ============================================================
# 20. SAVE RESULTS
# ============================================================

monthly_results.to_csv(
    OUTPUT_DIR / "low_value_monthly_results.csv",
    index=False
)

overall.to_csv(
    OUTPUT_DIR / "low_value_overall_results.csv",
    index=False
)

combined.to_csv(
    OUTPUT_DIR / "low_value_combined_scores.csv",
    index=False
)

print("\n==========================================")
print("Low-value fraud experiment completed.")
print("Results saved in:", OUTPUT_DIR)
print("==========================================")
