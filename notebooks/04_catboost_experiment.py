

# ============================================================
# KESTREL HOME WARRANTY CLAIM REVIEW
# CATBOOST FRAUD DETECTION EXPERIMENT
# Experiment: Without Class Weighting
# ============================================================

import pandas as pd
import numpy as np

from pathlib import Path
from catboost import CatBoostClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score
)

# ============================================================
# STEP 1: PATHS AND DATA LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test_unlabelled.csv")

partners = pd.read_csv(DATA_DIR / "partners.csv")
products = pd.read_csv(DATA_DIR / "products.csv")

print("Original training shape:", train.shape)
print("Test shape:", test.shape)

# ============================================================
# STEP 2: LABEL CLEANING
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
# STEP 3: DATE AND SERIAL CLEANING
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
# STEP 4: REFERENCE DATA MERGING
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
# STEP 5: BUSINESS FEATURES
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
# STEP 6: CATEGORICAL STANDARDIZATION
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
# STEP 7: REMOVE DUPLICATE CLAIMS
# ============================================================

train = (
    train.sort_values("submitted_at")
    .drop_duplicates(subset="claim_id", keep="first")
    .copy()
    .reset_index(drop=True)
)

test = test.reset_index(drop=True)

print("\nUnique training claims:", len(train))
print("Unique test claims:", len(test))

# ============================================================
# STEP 8: CHRONOLOGICAL PARTNER HISTORY
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
# STEP 9: CHRONOLOGICAL TRAIN / VALIDATION SPLIT
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

print("\nTraining samples:", len(train_data))
print("Training fraud cases:", train_data["is_fraud"].sum())

print("\nValidation samples:", len(validation_data))
print("Validation fraud cases:", validation_data["is_fraud"].sum())

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

categorical_features = [
    "photo_attached",
    "source",
    "partner_type",
    "city",
    "family"
]

feature_columns = numeric_features + categorical_features

X_train = train_data[feature_columns].copy()
y_train = train_data["is_fraud"].copy()

X_val = validation_data[feature_columns].copy()
y_val = validation_data["is_fraud"].copy()

for col in numeric_features:

    X_train[col] = pd.to_numeric(
        X_train[col], errors="coerce"
    )

    X_val[col] = pd.to_numeric(
        X_val[col], errors="coerce"
    )

cat_features_indices = [
    X_train.columns.get_loc(col)
    for col in categorical_features
]

print("\nCategorical feature indices:", cat_features_indices)

# ============================================================
# STEP 11: CHECK DATA BEFORE TRAINING
# ============================================================

positive_count = int(y_train.sum())
negative_count = int((y_train == 0).sum())

if positive_count == 0:
    raise ValueError("No fraud cases found in training data.")

if y_val.nunique() < 2:
    raise ValueError("Validation data must contain both classes.")

print("\nTraining genuine cases:", negative_count)
print("Training fraud cases:", positive_count)
print("Training fraud prevalence:", round(y_train.mean(), 4))

# ============================================================
# STEP 12: TRAIN CATBOOST WITHOUT CLASS WEIGHTS
# ============================================================

model = CatBoostClassifier(
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

print("\nTraining CatBoost model without class weights...")

model.fit(
    X_train,
    y_train,
    cat_features=cat_features_indices,
    eval_set=(X_val, y_val),
    verbose=100
)

print("\nBest iteration:", model.get_best_iteration())

# ============================================================
# STEP 13: VALIDATION PREDICTIONS
# ============================================================

# IMPORTANT:
# Generate probability scores BEFORE using them anywhere else.

val_scores = model.predict_proba(X_val)[:, 1]

val_predictions = (val_scores >= 0.5).astype(int)

pr_auc = average_precision_score(y_val, val_scores)
roc_auc = roc_auc_score(y_val, val_scores)

precision = precision_score(
    y_val,
    val_predictions,
    zero_division=0
)

recall = recall_score(
    y_val,
    val_predictions,
    zero_division=0
)

# Save CatBoost predictions for the next experiment

catboost_validation = validation_data[
    ["claim_id", "submitted_at", "is_fraud", "claim_amount_inr"]
].copy()

catboost_validation["catboost_score"] = val_scores

catboost_validation.to_csv(
    OUTPUT_DIR / "catboost_validation_predictions.csv",
    index=False
)

print("\nCatBoost validation predictions saved!")

print("\n========== CATBOOST RESULTS ==========")
print("PR-AUC:", round(pr_auc, 6))
print("ROC-AUC:", round(roc_auc, 6))
print("Precision:", round(precision, 6))
print("Recall:", round(recall, 6))

# ============================================================
# STEP 14: MONTHLY TOP-40 EVALUATION
# ============================================================

validation_output = validation_data[
    ["claim_id", "submitted_at", "is_fraud"]
].copy()

validation_output["score"] = val_scores

validation_output["month"] = (
    validation_output["submitted_at"]
    .dt.to_period("M")
    .astype(str)
)

capacity_results = []

for month, group in validation_output.groupby("month"):

    selected = group.nlargest(40, "score")

    fraud_found = int(selected["is_fraud"].sum())
    total_fraud = int(group["is_fraud"].sum())

    capacity_results.append({
        "month": month,
        "claims_reviewed": len(selected),
        "fraud_found": fraud_found,
        "total_fraud": total_fraud,
        "precision_at_40": (
            fraud_found / len(selected)
            if len(selected) else 0
        ),
        "recall_at_40": (
            fraud_found / total_fraud
            if total_fraud else 0
        )
    })

capacity_df = pd.DataFrame(capacity_results)

print("\n========== MONTHLY TOP-40 RESULTS ==========")
print(capacity_df.to_string(index=False))

# ============================================================
# STEP 14A: ERROR ANALYSIS
# ============================================================

print("\n========== CATBOOST ERROR ANALYSIS ==========")

diagnostic_columns = [
    "claim_id",
    "submitted_at",
    "is_fraud",
    "days_since_purchase",
    "claim_amount_inr",
    "customer_prior_claims",
    "partner_tenure_days",
    "claim_to_product_price_ratio",
    "partner_type",
    "family",
    "photo_attached",
    "source"
]

diagnostic = validation_data[diagnostic_columns].copy()

diagnostic["score"] = val_scores

diagnostic["month"] = (
    diagnostic["submitted_at"]
    .dt.to_period("M")
    .astype(str)
)

diagnostic["monthly_rank"] = (
    diagnostic.groupby("month")["score"]
    .rank(method="first", ascending=False)
    .astype(int)
)

# Actual fraud cases and their ranks

fraud_cases = diagnostic[
    diagnostic["is_fraud"] == 1
].sort_values(["month", "monthly_rank"])

print("\nACTUAL FRAUD CASES AND THEIR MODEL RANKS:")

print(
    fraud_cases[
        ["month", "score", "monthly_rank"]
    ].to_string(index=False)
)

# Genuine claims selected in top 40

false_positives = diagnostic[
    (diagnostic["is_fraud"] == 0) &
    (diagnostic["monthly_rank"] <= 40)
].sort_values(
    ["month", "score"],
    ascending=[True, False]
)

print("\nTOP 10 GENUINE CLAIMS SELECTED PER MONTH:")

for month in sorted(false_positives["month"].unique()):

    print(f"\nMONTH: {month}")

    month_data = false_positives[
        false_positives["month"] == month
    ].head(10)

    print(
        month_data[
            [
                "score",
                "days_since_purchase",
                "claim_amount_inr",
                "customer_prior_claims",
                "partner_tenure_days",
                "claim_to_product_price_ratio",
                "partner_type",
                "family"
            ]
        ].to_string(index=False)
    )

diagnostic.to_csv(
    OUTPUT_DIR / "catboost_error_analysis.csv",
    index=False
)

print("\nError analysis saved successfully!")

# ============================================================
# STEP 14B: MAY FRAUD VS GENUINE COMPARISON
# ============================================================

may_data = diagnostic[
    diagnostic["month"] == "2026-05"
].copy()

may_fraud = may_data[
    may_data["is_fraud"] == 1
]

may_selected_genuine = may_data[
    (may_data["is_fraud"] == 0) &
    (may_data["monthly_rank"] <= 40)
]

features = [
    "days_since_purchase",
    "claim_amount_inr",
    "customer_prior_claims",
    "partner_tenure_days",
    "claim_to_product_price_ratio"
]

print("\n========== MAY FRAUD VS SELECTED GENUINE ==========")

comparison = pd.DataFrame({
    "Missed_Fraud_Mean": may_fraud[features].mean(),
    "Selected_Genuine_Mean": may_selected_genuine[features].mean(),
    "Missed_Fraud_Median": may_fraud[features].median(),
    "Selected_Genuine_Median": may_selected_genuine[features].median()
})

print(comparison.round(2).to_string())

print("\nPARTNER TYPE DISTRIBUTION")

print("\nMissed fraud:")
print(may_fraud["partner_type"].value_counts())

print("\nSelected genuine:")
print(may_selected_genuine["partner_type"].value_counts())

print("\nPRODUCT FAMILY DISTRIBUTION")

print("\nMissed fraud:")
print(may_fraud["family"].value_counts())

print("\nSelected genuine:")
print(may_selected_genuine["family"].value_counts())

# ============================================================
# STEP 15: TRAINING VS VALIDATION FRAUD PATTERNS
# ============================================================

print("\n========== TRAINING VS VALIDATION FRAUD PATTERNS ==========")

train_analysis = train_data[
    ["claim_amount_inr", "days_since_purchase", "is_fraud"]
].copy()

validation_analysis = validation_data[
    ["claim_amount_inr", "days_since_purchase", "is_fraud"]
].copy()

train_analysis["period"] = "Training"
validation_analysis["period"] = "Validation"

combined_analysis = pd.concat(
    [train_analysis, validation_analysis],
    ignore_index=True
)

# Fraud by claim amount

combined_analysis["amount_group"] = pd.cut(
    combined_analysis["claim_amount_inr"],
    bins=[0, 2000, 5000, float("inf")],
    labels=["Below 2000", "2000-5000", "Above 5000"],
    include_lowest=True
)

amount_analysis = (
    combined_analysis
    .groupby(
        ["period", "amount_group"],
        observed=False
    )
    .agg(
        total_claims=("is_fraud", "count"),
        fraud_cases=("is_fraud", "sum"),
        fraud_rate=("is_fraud", "mean")
    )
    .reset_index()
)

amount_analysis["fraud_rate_percent"] = (
    amount_analysis["fraud_rate"] * 100
)

print("\nFRAUD RATE BY CLAIM AMOUNT:")
print(amount_analysis.to_string(index=False))

# Fraud by purchase age

combined_analysis["purchase_age_group"] = pd.cut(
    combined_analysis["days_since_purchase"],
    bins=[0, 90, 180, 365, 720],
    labels=[
        "0-90 days",
        "91-180 days",
        "181-365 days",
        "366-720 days"
    ],
    include_lowest=True
)

purchase_age_analysis = (
    combined_analysis
    .groupby(
        ["period", "purchase_age_group"],
        observed=False
    )
    .agg(
        total_claims=("is_fraud", "count"),
        fraud_cases=("is_fraud", "sum"),
        fraud_rate=("is_fraud", "mean")
    )
    .reset_index()
)

purchase_age_analysis["fraud_rate_percent"] = (
    purchase_age_analysis["fraud_rate"] * 100
)

print("\nFRAUD RATE BY PURCHASE AGE:")
print(purchase_age_analysis.to_string(index=False))

amount_analysis.to_csv(
    OUTPUT_DIR / "fraud_by_claim_amount.csv",
    index=False
)

purchase_age_analysis.to_csv(
    OUTPUT_DIR / "fraud_by_purchase_age.csv",
    index=False
)

print("\nTraining vs validation analysis saved successfully!")

# ============================================================
# STEP 16: MONTHLY LOW-VALUE FRAUD ANALYSIS
# ============================================================

print("\n========== MONTHLY LOW-VALUE FRAUD ANALYSIS ==========")

monthly_data = validation_data[
    ["submitted_at", "claim_amount_inr", "is_fraud"]
].copy()

monthly_data["month"] = (
    monthly_data["submitted_at"]
    .dt.to_period("M")
    .astype(str)
)

monthly_data["amount_group"] = np.where(
    monthly_data["claim_amount_inr"] < 2000,
    "Below 2000",
    "2000 and above"
)

monthly_analysis = (
    monthly_data
    .groupby(["month", "amount_group"])
    .agg(
        total_claims=("is_fraud", "count"),
        fraud_cases=("is_fraud", "sum"),
        fraud_rate=("is_fraud", "mean")
    )
    .reset_index()
)

monthly_analysis["fraud_rate_percent"] = (
    monthly_analysis["fraud_rate"] * 100
)

print(monthly_analysis.to_string(index=False))

monthly_analysis.to_csv(
    OUTPUT_DIR / "monthly_low_value_fraud_analysis.csv",
    index=False
)

print("\nMonthly analysis saved successfully!")

# ============================================================
# STEP 17: SAVE MODEL METRICS AND OUTPUTS
# ============================================================

metrics_df = pd.DataFrame([{
    "model": "CatBoost_Unweighted",
    "pr_auc": pr_auc,
    "roc_auc": roc_auc,
    "precision": precision,
    "recall": recall,
    "training_samples": len(train_data),
    "training_fraud_cases": positive_count,
    "validation_samples": len(validation_data),
    "validation_fraud_cases": int(y_val.sum())
}])

metrics_df.to_csv(
    OUTPUT_DIR / "catboost_unweighted_metrics.csv",
    index=False
)

capacity_df.to_csv(
    OUTPUT_DIR / "catboost_unweighted_capacity.csv",
    index=False
)

validation_output.to_csv(
    OUTPUT_DIR / "catboost_unweighted_validation_predictions.csv",
    index=False
)

model.save_model(
    OUTPUT_DIR / "catboost_unweighted_model.cbm"
)

feature_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

feature_importance.to_csv(
    OUTPUT_DIR / "catboost_unweighted_feature_importance.csv",
    index=False
)

print("\nSaved unweighted CatBoost experiment outputs to:", OUTPUT_DIR)
print("Experiment completed successfully.")
