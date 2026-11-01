
import pandas as pd
from pathlib import Path

# ==========================================
# 1. LOAD DATA
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pred = pd.read_csv(
    OUTPUT_DIR / "catboost_validation_predictions.csv"
)

train = pd.read_csv(DATA_DIR / "train.csv")
partners = pd.read_csv(DATA_DIR / "partners.csv")
products = pd.read_csv(DATA_DIR / "products.csv")

# Standardize IDs
for frame in [pred, train, partners]:
    frame["partner_id"] = frame["partner_id"].astype(str) if "partner_id" in frame.columns else None

pred["claim_id"] = pred["claim_id"].astype(str)
train["claim_id"] = train["claim_id"].astype(str)

# Parse dates
pred["submitted_at"] = pd.to_datetime(pred["submitted_at"])
train["submitted_at"] = pd.to_datetime(train["submitted_at"])
partners["onboarded_date"] = pd.to_datetime(partners["onboarded_date"])

pred["month"] = pred["submitted_at"].dt.to_period("M")

# ==========================================
# 2. PREPARE ORIGINAL CLAIM FEATURES
# ==========================================

# Keep earliest record for each claim ID
train = train.sort_values("submitted_at")
train = train.drop_duplicates("claim_id", keep="first")

# Merge partner details, including onboarding date
partner_columns = [
    "partner_id",
    "partner_type",
    "onboarded_date"
]

claims = train.merge(
    partners[partner_columns],
    on="partner_id",
    how="left",
    suffixes=("", "_partner")
)

# Calculate partner tenure at claim submission
claims["partner_tenure_days"] = (
    claims["submitted_at"] - claims["onboarded_date"]
).dt.days

# Merge product details
product_columns = ["sku", "family"]

claims = claims.merge(
    products[product_columns],
    on="sku",
    how="left"
)

# ==========================================
# 3. MERGE PREDICTIONS WITH CLAIM FEATURES
# ==========================================

df = pred.merge(
    claims,
    on="claim_id",
    how="left",
    suffixes=("", "_original")
)

# Check matching
print("\n========== DATA MATCHING CHECK ==========")
print("Validation prediction rows:", len(pred))
print("Matched claim rows:", df["days_since_purchase"].notna().sum())
print("Unmatched claim rows:", df["days_since_purchase"].isna().sum())

# Use the original label only if prediction output does not already contain it
if "is_fraud_original" in df.columns:
    df["is_fraud"] = df["is_fraud"].fillna(df["is_fraud_original"])

df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce")
df = df[df["is_fraud"].isin([0, 1])].copy()
df["is_fraud"] = df["is_fraud"].astype(int)

# ==========================================
# 4. MONTHLY RANK AND DETECTION STATUS
# ==========================================

df["monthly_rank"] = (
    df.groupby("month")["catboost_score"]
    .rank(method="first", ascending=False)
)

df["status"] = "Genuine"

df.loc[
    (df["is_fraud"] == 1) & (df["monthly_rank"] <= 40),
    "status"
] = "Detected Fraud"

df.loc[
    (df["is_fraud"] == 1) & (df["monthly_rank"] > 40),
    "status"
] = "Missed Fraud"

# ==========================================
# 5. CLAIM AMOUNT ANALYSIS
# ==========================================

df["amount_band"] = pd.cut(
    df["claim_amount_inr"],
    bins=[0, 2000, 5000, float("inf")],
    labels=["Below 2000", "2000-5000", "Above 5000"],
    right=False
)

print("\n========== FRAUD BY CLAIM AMOUNT ==========")

amount_analysis = pd.crosstab(
    df["amount_band"],
    df["status"]
)

print(amount_analysis)

# ==========================================

# ==========================================
# 6. DETECTED VS MISSED FRAUD COMPARISON
# ==========================================

numeric_features = [
    "claim_amount_inr",
    "days_since_purchase",
    "customer_prior_claims",
    "partner_tenure_days"
]

categorical_features = [
    "partner_type",
    "family"
]

fraud_cases = df[df["is_fraud"] == 1].copy()

# Numeric feature comparison
available_numeric = [
    col for col in numeric_features if col in fraud_cases.columns
]

print("\n========== NUMERIC FEATURE COMPARISON ==========")

numeric_comparison = (
    fraud_cases.groupby("status")[available_numeric]
    .agg(["count", "mean", "median"])
)

print(numeric_comparison.to_string())

# Categorical feature comparison
available_categorical = [
    col for col in categorical_features if col in fraud_cases.columns
]

print("\n========== CATEGORICAL FEATURE COMPARISON ==========")

for feature in available_categorical:
    print(f"\n--- {feature.upper()} ---")

    categorical_comparison = pd.crosstab(
        fraud_cases["status"],
        fraud_cases[feature],
        margins=True
    )

    print(categorical_comparison.to_string())


# ==========================================
# 7. MONTHLY FRAUD RATE BY AMOUNT BAND
# ==========================================

print("\n========== MONTHLY FRAUD RATE BY AMOUNT ==========")

monthly_amount = (
    df.groupby(["month", "amount_band"], observed=False)
    .agg(
        total_claims=("is_fraud", "size"),
        fraud_cases=("is_fraud", "sum")
    )
    .reset_index()
)

monthly_amount["fraud_rate_percent"] = (
    monthly_amount["fraud_cases"]
    / monthly_amount["total_claims"].replace(0, pd.NA)
    * 100
)

print(monthly_amount.to_string(index=False))

# ==========================================
# 8. SAVE AGGREGATED RESULTS
# ==========================================

amount_analysis.to_csv(
    OUTPUT_DIR / "fraud_by_amount_band.csv"
)

monthly_amount.to_csv(
    OUTPUT_DIR / "monthly_fraud_by_amount_band.csv",
    index=False
)

numeric_comparison.to_csv(
    OUTPUT_DIR / "detected_vs_missed_numeric_features.csv"
)

print("\n==========================================")
print("Missed-fraud investigation completed.")
print("Aggregated results saved in:", OUTPUT_DIR)
print("==========================================")
