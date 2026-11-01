
# ============================================================
# KESTREL HOME WARRANTY CLAIM REVIEW
# DATA CLEANING, FEATURE ENGINEERING AND MODEL EXPERIMENTS
# ============================================================

import re
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score
)

# ============================================================
# STEP 1: PROJECT PATHS AND DATA LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test_unlabelled.csv")
partners = pd.read_csv(DATA_DIR / "partners.csv")
products = pd.read_csv(DATA_DIR / "products.csv")

print("\n--- ORIGINAL DATASET SHAPES ---")
print("Train:", train.shape)
print("Test:", test.shape)

train_clean = train.copy()
test_clean = test.copy()

# ============================================================
# STEP 2: DATA QUALITY CHECKS
# ============================================================

print("\n--- MISSING VALUES ---")
print("\nTraining:")
print(train_clean.isnull().sum())

print("\nTesting:")
print(test_clean.isnull().sum())

print("\n--- FRAUD LABEL STATUS ---")
print(train_clean["is_fraud"].value_counts(dropna=False))

print("\n--- DUPLICATE CLAIM ID ANALYSIS ---")

print(
    "Training duplicate rows:",
    train_clean.duplicated("claim_id", keep=False).sum()
)

print(
    "Unique duplicated claim IDs:",
    train_clean.loc[
        train_clean.duplicated("claim_id", keep=False),
        "claim_id"
    ].nunique()
)

print(
    "Testing duplicate rows:",
    test_clean.duplicated("claim_id", keep=False).sum()
)

# ============================================================
# STEP 3: LABEL CLEANING AND CLAIM ID STANDARDIZATION
# ============================================================

print("\n--- LABEL CLEANING ---")

train_labelled = train_clean[
    train_clean["is_fraud"].isin([0, 1])
].copy()

train_labelled["is_fraud"] = (
    train_labelled["is_fraud"].astype(int)
)

train_labelled["claim_id"] = (
    train_labelled["claim_id"]
    .astype(str).str.strip().str.upper()
)

test_clean["claim_id"] = (
    test_clean["claim_id"]
    .astype(str).str.strip().str.upper()
)

print("Original training rows:", len(train_clean))
print("Labelled training rows:", len(train_labelled))
print("Undecided rows excluded:", train_clean["is_fraud"].isna().sum())

print("\nLabel distribution:")
print(train_labelled["is_fraud"].value_counts())

# ============================================================
# STEP 4: DATE AND SERIAL NUMBER CLEANING
# ============================================================

print("\n--- DATE CLEANING ---")

for df in [train_labelled, test_clean]:

    df["submitted_at"] = pd.to_datetime(
        df["submitted_at"],
        errors="coerce"
    )

    df["claim_year"] = df["submitted_at"].dt.year
    df["claim_month"] = df["submitted_at"].dt.month
    df["claim_day"] = df["submitted_at"].dt.day
    df["claim_weekday"] = df["submitted_at"].dt.weekday

    df["normalized_serial"] = (
        df["product_serial"]
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )

print(
    "Training missing dates:",
    train_labelled["submitted_at"].isna().sum()
)

print(
    "Testing missing dates:",
    test_clean["submitted_at"].isna().sum()
)

print("\nSerial number example:")
print(train_labelled[
    ["product_serial", "normalized_serial"]
].head())

# ============================================================
# STEP 5: REFERENCE TABLE CHECKS AND MERGING
# ============================================================

print("\n--- REFERENCE TABLE CHECKS ---")

print("Duplicate partner IDs:", partners["partner_id"].duplicated().sum())
print("Duplicate product SKUs:", products["sku"].duplicated().sum())

partners["onboarded_date"] = pd.to_datetime(
    partners["onboarded_date"],
    errors="coerce"
)

train_clean = train_labelled.merge(
    partners,
    on="partner_id",
    how="left",
    validate="many_to_one"
)

test_clean = test_clean.merge(
    partners,
    on="partner_id",
    how="left",
    validate="many_to_one"
)

train_clean = train_clean.merge(
    products,
    on="sku",
    how="left",
    validate="many_to_one"
)

test_clean = test_clean.merge(
    products,
    on="sku",
    how="left",
    validate="many_to_one"
)

print("Training shape after merging:", train_clean.shape)
print("Testing shape after merging:", test_clean.shape)

print(
    "Missing partner information:",
    train_clean["partner_type"].isna().sum(),
    test_clean["partner_type"].isna().sum()
)

print(
    "Missing product information:",
    train_clean["family"].isna().sum(),
    test_clean["family"].isna().sum()
)

# ============================================================
# STEP 6: BUSINESS FEATURE ENGINEERING
# ============================================================

print("\n--- BUSINESS FEATURE ENGINEERING ---")

for df in [train_clean, test_clean]:

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

# Serial counts are retained for data exploration only.
# They are not used as model features because full-dataset
# counts can introduce future-information leakage.

print(
    train_clean[
        [
            "partner_tenure_days",
            "claim_to_product_price_ratio",
            "is_low_value_claim",
            "is_after_policy_change"
        ]
    ].head()
)

# ============================================================
# STEP 7: CATEGORICAL FEATURE STANDARDIZATION
# ============================================================

print("\n--- CATEGORICAL FEATURE PREPARATION ---")

categorical_columns = [
    "photo_attached",
    "partner_inspected",
    "source",
    "partner_type",
    "city",
    "family"
]

for df in [train_clean, test_clean]:

    for column in categorical_columns:

        df[column] = (
            df[column]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

for column in categorical_columns:
    print(column, sorted(train_clean[column].unique()))

# ============================================================
# STEP 8: REMOVE DUPLICATE CLAIM SUBMISSIONS
# ============================================================

print("\n--- MODELLING DATASET PREPARATION ---")

train_clean = train_clean.sort_values("submitted_at")

model_train = train_clean.drop_duplicates(
    subset="claim_id",
    keep="first"
).copy()

model_train = model_train.reset_index(drop=True)
test_clean = test_clean.reset_index(drop=True)

# ============================================================
# STEP 8A: CHRONOLOGICAL HISTORICAL FEATURES
# ============================================================

print("\n--- CHRONOLOGICAL HISTORICAL FEATURES ---")

# Combine training and test claims to build a shared timeline.
# No fraud labels are used to calculate these counts.

history_data = pd.concat(
    [
        model_train.assign(dataset="train"),
        test_clean.assign(dataset="test")
    ],
    ignore_index=True,
    sort=False
)

# Sort chronologically. A stable sort preserves row order
# when multiple claims have the same timestamp.
history_data = history_data.sort_values(
    "submitted_at",
    kind="mergesort"
).copy()

# Number of earlier claims submitted by the same partner.
history_data["partner_prior_claim_count"] = (
    history_data.groupby("partner_id").cumcount()
)

# Number of earlier claims associated with the same normalized serial.
history_data["serial_prior_claim_count"] = (
    history_data.groupby("normalized_serial").cumcount()
)

# Restore the original training and test datasets.
model_train = (
    history_data[history_data["dataset"] == "train"]
    .drop(columns=["dataset"])
    .sort_index()
    .reset_index(drop=True)
)

test_clean = (
    history_data[history_data["dataset"] == "test"]
    .drop(columns=["dataset"])
    .sort_index()
    .reset_index(drop=True)
)

print("Historical features created:")
print(
    model_train[
        [
            "partner_id",
            "partner_prior_claim_count",
            "normalized_serial",
            "serial_prior_claim_count"
        ]
    ].head(10)
)

print("\nTraining historical feature summary:")
print(
    model_train[
        ["partner_prior_claim_count", "serial_prior_claim_count"]
    ].describe()
)


print("Labelled training rows:", len(train_clean))
print("Unique training claims:", len(model_train))
print("Testing rows:", len(test_clean))
print("Testing unique claims:", test_clean["claim_id"].nunique())

print("\nLabel distribution:")
print(model_train["is_fraud"].value_counts())

print(
    "Fraud percentage:",
    round(model_train["is_fraud"].mean() * 100, 3)
)

# ============================================================
# STEP 9: TEMPORAL ANALYSIS
# ============================================================

print("\n--- TEMPORAL DATA ANALYSIS ---")

monthly_data = (
    model_train
    .groupby(model_train["submitted_at"].dt.to_period("M"))
    .agg(
        total_claims=("claim_id", "count"),
        fraud_claims=("is_fraud", "sum")
    )
)

monthly_data["fraud_rate_percent"] = (
    monthly_data["fraud_claims"]
    / monthly_data["total_claims"] * 100
)

print(monthly_data)

print("\nTraining date range:")
print(model_train["submitted_at"].min())
print(model_train["submitted_at"].max())

print("\nTest date range:")
print(test_clean["submitted_at"].min())
print(test_clean["submitted_at"].max())

# ============================================================
# STEP 10: CHRONOLOGICAL TRAIN-VALIDATION SPLIT
# ============================================================

print("\n--- CHRONOLOGICAL TRAIN-VALIDATION SPLIT ---")

validation_start = pd.Timestamp("2026-04-01")

train_data = model_train[
    (model_train["submitted_at"] < validation_start) &
    (model_train["source"] == "crm")
].copy()

validation_data = model_train[
    model_train["submitted_at"] >= validation_start
].copy()

print("Training samples:", len(train_data))
print("Validation samples:", len(validation_data))

print("\nTraining label distribution:")
print(train_data["is_fraud"].value_counts())

print("\nValidation label distribution:")
print(validation_data["is_fraud"].value_counts())

print(
    "Training fraud rate:",
    round(train_data["is_fraud"].mean() * 100, 3),
    "%"
)

print(
    "Validation fraud rate:",
    round(validation_data["is_fraud"].mean() * 100, 3),
    "%"
)

# ============================================================
# STEP 11: CLEAN CLAIM DESCRIPTIONS
# ============================================================

print("\n--- CLAIM DESCRIPTION CLEANING ---")

def clean_claim_description(description):

    text = str(description).lower().strip()

    markers = [
        r"\[note for ai",
        r"\bops note\b",
        r"\breviewer tools\b",
        r"\bfor automated analysis\b",
        r"\bfor ai\b",
        r"\bautomated review\b"
    ]

    for marker in markers:

        text = re.split(
            marker,
            text,
            maxsplit=1
        )[0]

    return text.strip(" .;,-")


for df in [model_train, test_clean]:

    df["clean_description"] = (
        df["claim_description"]
        .fillna("")
        .apply(clean_claim_description)
    )

# Recreate the split after creating the text feature.
train_data = model_train[
    (model_train["submitted_at"] < validation_start) &
    (model_train["source"] == "crm")
].copy()
print("\n--- CRM-ONLY EXPERIMENT ---")
print("Training source distribution:")
print(train_data["source"].value_counts())

print("\nTraining samples:", len(train_data))
print("Training fraud cases:", train_data["is_fraud"].sum())

print("\nValidation samples:", len(validation_data))
print("Validation fraud cases:", validation_data["is_fraud"].sum())

validation_data = model_train[
    model_train["submitted_at"] >= validation_start
].copy()

print("\nOriginal and cleaned descriptions:")

print(
    model_train[
        ["claim_description", "clean_description", "is_fraud"]
    ].head(10).to_string(index=False)
)

print("\nSuspicious descriptions after cleaning:")

suspicious_rows = model_train[
    model_train["claim_description"].str.lower().str.contains(
        r"note for ai|ops note|reviewer tools|automated analysis",
        regex=True,
        na=False
    )
]

print(
    suspicious_rows[
        ["clean_description", "is_fraud"]
    ].to_string(index=False)
)

# ============================================================
# STEP 12: MODEL FEATURE SELECTION
# ============================================================

print("\n--- MODEL FEATURE SELECTION ---")


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
    "partner_prior_claim_count",
    "serial_prior_claim_count"
]


# partner_inspected is intentionally excluded because
# the score is intended to be generated at claim submission.

categorical_features = [
    "photo_attached",
    "source",
    "partner_type",
    "city",
    "family"
]

feature_columns = (
    numeric_features
    + categorical_features
    + ["clean_description"]
)

X_train = train_data[feature_columns]
y_train = train_data["is_fraud"]

X_val = validation_data[feature_columns]
y_val = validation_data["is_fraud"]

print("Numerical features:", numeric_features)
print("Categorical features:", categorical_features)
print("Text feature: clean_description")

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)

# ============================================================
# STEP 13: PREPROCESSING PIPELINE
# ============================================================

print("\n--- PREPROCESSING PIPELINE ---")

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        ),
        (
            "text",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=100
            ),
            "clean_description"
        )
    ]
)

# ============================================================
# STEP 14: MODEL TRAINING AND COMPARISON
# ============================================================

print("\n--- MODEL TRAINING ---")


# Calculate class imbalance using training data only
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print(f"\nXGBoost scale_pos_weight: {scale_pos_weight:.2f}")

models = {

    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
}


results = []
trained_models = {}
validation_predictions = {}

for model_name, classifier in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    pipeline.fit(X_train, y_train)

    trained_models[model_name] = pipeline

    val_probabilities = pipeline.predict_proba(X_val)[:, 1]

    validation_predictions[model_name] = val_probabilities

    pr_auc = average_precision_score(
        y_val,
        val_probabilities
    )

    roc_auc = roc_auc_score(
        y_val,
        val_probabilities
    )

    val_predictions = (
        val_probabilities >= 0.5
    ).astype(int)

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

    results.append({
        "Model": model_name,
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "Precision": precision,
        "Recall": recall
    })

    print("PR-AUC:", round(pr_auc, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))

results_df = pd.DataFrame(results)

print("\n--- MODEL COMPARISON ---")
print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

# ============================================================
# STEP 14A: MODEL RANKING DIAGNOSTICS
# ============================================================

print("\n--- MODEL RANKING DIAGNOSTICS ---")

diagnostic_results = []

for model_name, probabilities in validation_predictions.items():

    diagnostic_data = validation_data[
        ["submitted_at", "is_fraud"]
    ].copy()

    diagnostic_data["score"] = probabilities

    # Compare average scores for genuine and fraud claims
    fraud_mean = diagnostic_data.loc[
        diagnostic_data["is_fraud"] == 1, "score"
    ].mean()

    genuine_mean = diagnostic_data.loc[
        diagnostic_data["is_fraud"] == 0, "score"
    ].mean()

    normal_auc = roc_auc_score(
        diagnostic_data["is_fraud"],
        diagnostic_data["score"]
    )

    reversed_auc = roc_auc_score(
        diagnostic_data["is_fraud"],
        1 - diagnostic_data["score"]
    )

    print(f"\nModel: {model_name}")
    print("Average fraud score:", round(fraud_mean, 5))
    print("Average genuine score:", round(genuine_mean, 5))
    print("Original ROC-AUC:", round(normal_auc, 4))
    print("Reversed ROC-AUC:", round(reversed_auc, 4))

    for month, group in diagnostic_data.groupby(
        diagnostic_data["submitted_at"].dt.to_period("M")
    ):

        if group["is_fraud"].nunique() == 2:

            month_auc = roc_auc_score(
                group["is_fraud"],
                group["score"]
            )

            month_pr_auc = average_precision_score(
                group["is_fraud"],
                group["score"]
            )

            print(
                f"{month}: "
                f"ROC-AUC={month_auc:.4f}, "
                f"PR-AUC={month_pr_auc:.4f}"
            )

    diagnostic_results.append({
        "Model": model_name,
        "Average Fraud Score": fraud_mean,
        "Average Genuine Score": genuine_mean,
        "Original ROC-AUC": normal_auc,
        "Reversed ROC-AUC": reversed_auc
    })

diagnostic_df = pd.DataFrame(diagnostic_results)

diagnostic_df.to_csv(
    OUTPUT_DIR / "ranking_diagnostics.csv",
    index=False
)

print("\n--- DIAGNOSTIC SUMMARY ---")
print(diagnostic_df.to_string(index=False))

# ============================================================
# STEP 14B: FRAUD FEATURE COMPARISON
# ============================================================

print("\n--- FRAUD FEATURE COMPARISON ---")

comparison_data = model_train.copy()

comparison_data["period"] = np.where(
    comparison_data["submitted_at"] < validation_start,
    "Training",
    "Validation"
)

numeric_comparison = [
    "claim_amount_inr",
    "days_since_purchase",
    "customer_prior_claims",
    "partner_tenure_days",
    "claim_to_product_price_ratio"
]

for column in numeric_comparison:

    print(f"\nFEATURE: {column}")

    summary = (
        comparison_data
        .groupby(["period", "is_fraud"])[column]
        .agg(["count", "mean", "median"])
        .round(2)
    )

    print(summary)

categorical_comparison = [
    "photo_attached",
    "source",
    "partner_type",
    "family"
]

for column in categorical_comparison:

    print(f"\nFEATURE: {column}")

    summary = (
        comparison_data
        .groupby(["period", column])["is_fraud"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )

    summary["fraud_rate_percent"] = (
        summary["mean"] * 100
    )

    print(
        summary.drop(columns="mean").round(2).to_string(index=False)
    )

# Save numeric and categorical summaries separately

numeric_rows = []

for column in numeric_comparison:

    summary = (
        comparison_data
        .groupby(["period", "is_fraud"])[column]
        .agg(["count", "mean", "median"])
        .reset_index()
    )

    summary["feature"] = column
    numeric_rows.append(summary)

numeric_summary = pd.concat(
    numeric_rows,
    ignore_index=True
)

numeric_summary.to_csv(
    OUTPUT_DIR / "numeric_feature_comparison.csv",
    index=False
)

categorical_rows = []

for column in categorical_comparison:

    summary = (
        comparison_data
        .groupby(["period", column])["is_fraud"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )

    summary["fraud_rate_percent"] = summary["mean"] * 100
    summary["feature"] = column

    categorical_rows.append(summary)

categorical_summary = pd.concat(
    categorical_rows,
    ignore_index=True
)

categorical_summary.to_csv(
    OUTPUT_DIR / "categorical_feature_comparison.csv",
    index=False
)

print("\nFeature comparison files saved successfully.")

# ============================================================
# STEP 14C: XGBOOST FEATURE IMPORTANCE
# ============================================================

print("\n--- XGBOOST FEATURE IMPORTANCE ---")

xgb_pipeline = trained_models["XGBoost"]

fitted_preprocessor = xgb_pipeline.named_steps["preprocessor"]
xgb_classifier = xgb_pipeline.named_steps["classifier"]

feature_names = fitted_preprocessor.get_feature_names_out()
importance_values = xgb_classifier.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\nTop 20 important features:")
print(importance_df.head(20).to_string(index=False))

importance_df.to_csv(
    OUTPUT_DIR / "xgboost_feature_importance.csv",
    index=False
)

print("\nFeature importance saved.")


# ============================================================
# STEP 15: INVESTIGATION CAPACITY EVALUATION
# ============================================================

print("\n--- TOP 40 CLAIMS PER MONTH ---")

all_capacity_results = []

for model_name, val_probabilities in validation_predictions.items():

    validation_results = validation_data[
        ["claim_id", "submitted_at", "is_fraud"]
    ].copy()

    validation_results["fraud_score"] = val_probabilities

    validation_results["month"] = (
        validation_results["submitted_at"].dt.to_period("M")
    )

    print(f"\nModel: {model_name}")

    total_selected = 0
    total_fraud_found = 0
    total_actual_fraud = 0

    for month, monthly_claims in validation_results.groupby("month"):

        top_40 = monthly_claims.nlargest(
            min(40, len(monthly_claims)),
            "fraud_score"
        )

        fraud_found = int(top_40["is_fraud"].sum())
        actual_fraud = int(monthly_claims["is_fraud"].sum())

        precision_at_40 = (
            fraud_found / len(top_40)
            if len(top_40) else 0
        )

        recall_at_40 = (
            fraud_found / actual_fraud
            if actual_fraud else 0
        )

        total_selected += len(top_40)
        total_fraud_found += fraud_found
        total_actual_fraud += actual_fraud

        print(f"\nMonth: {month}")
        print("Claims selected:", len(top_40))
        print("Actual fraud cases:", actual_fraud)
        print("Fraud cases found:", fraud_found)
        print("Precision@40:", round(precision_at_40, 4))
        print("Recall@40:", round(recall_at_40, 4))

        all_capacity_results.append({
            "Model": model_name,
            "Month": str(month),
            "Claims Reviewed": len(top_40),
            "Actual Fraud": actual_fraud,
            "Fraud Found": fraud_found,
            "Precision@40": precision_at_40,
            "Recall@40": recall_at_40
        })

    print("\nTotal claims reviewed:", total_selected)
    print("Total fraud cases found:", total_fraud_found)
    print("Total actual fraud cases:", total_actual_fraud)

    overall_recall = (
        total_fraud_found / total_actual_fraud
        if total_actual_fraud else 0
    )

    print("Overall recall:", round(overall_recall, 4))

capacity_df = pd.DataFrame(all_capacity_results)

capacity_df.to_csv(
    OUTPUT_DIR / "capacity_evaluation.csv",
    index=False
)

# ============================================================
# STEP 16: SAVE VALIDATION SCORES
# ============================================================

print("\n--- SAVING VALIDATION PREDICTIONS ---")

validation_output = validation_data[
    ["claim_id", "submitted_at", "is_fraud"]
].copy()

for model_name, probabilities in validation_predictions.items():

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
    )

    validation_output[f"{safe_name}_score"] = probabilities

validation_output.to_csv(
    OUTPUT_DIR / "validation_predictions.csv",
    index=False
)

print("Saved validation_predictions.csv")

# ============================================================
# STEP 17: EXPERIMENT SUMMARY
# ============================================================

print("\n--- EXPERIMENT SUMMARY ---")

print("Training claims:", len(train_data))
print("Validation claims:", len(validation_data))
print("Training fraud cases:", int(y_train.sum()))
print("Validation fraud cases:", int(y_val.sum()))

print("\nValidation fraud prevalence:")
print(round(y_val.mean(), 4))

print("\nModel comparison:")
print(results_df.to_string(index=False))

print("\nCapacity evaluation:")
print(capacity_df.to_string(index=False))

print("\nAll experiments completed.")

print("Output directory:", OUTPUT_DIR)
