
import pandas as pd

# Load historical claims and reference files
train = pd.read_csv("data/raw/train.csv")
partners = pd.read_csv("data/raw/partners.csv")
products = pd.read_csv("data/raw/products.csv")

# Keep only claims with confirmed investigation outcomes
df = train[train["is_fraud"].isin([0, 1])].copy()

# Merge partner and product information
df = df.merge(partners, on="partner_id", how="left")
df = df.merge(products, on="sku", how="left")

# Convert submission date and create month column
df["submitted_at"] = pd.to_datetime(df["submitted_at"], errors="coerce")
df["month"] = df["submitted_at"].dt.to_period("M").astype(str)

# Focus on April, May and June 2026
months = ["2026-04", "2026-05", "2026-06"]
df = df[df["month"].isin(months)].copy()

print("\nMONTHLY CLAIM COUNTS")
print(df.groupby("month").agg(
    total_claims=("claim_id", "count"),
    fraud_cases=("is_fraud", "sum")
))

# Compare fraud rates across important categories
columns = [
    "photo_attached",
    "partner_inspected",
    "partner_type",
    "family",
    "source",
    "customer_prior_claims"
]

for col in columns:
    print(f"\n{'=' * 55}")
    print(f"FRAUD ANALYSIS BY {col.upper()}")
    print("=" * 55)

    result = (
        df.groupby(["month", col], dropna=False)
        .agg(
            total_claims=("claim_id", "count"),
            fraud_cases=("is_fraud", "sum")
        )
        .reset_index()
    )

    result["fraud_rate_pct"] = (
        result["fraud_cases"] / result["total_claims"] * 100
    ).round(2)

    print(result.to_string(index=False))

# Compare claim amounts
print("\nFRAUD ANALYSIS BY CLAIM AMOUNT")

df["amount_group"] = pd.cut(
    df["claim_amount_inr"],
    bins=[0, 1000, 2000, 5000, float("inf")],
    labels=["Below 1000", "1000-2000", "2000-5000", "Above 5000"]
)

amount_result = (
    df.groupby(["month", "amount_group"], observed=False)
    .agg(
        total_claims=("claim_id", "count"),
        fraud_cases=("is_fraud", "sum")
    )
    .reset_index()
)

amount_result["fraud_rate_pct"] = (
    amount_result["fraud_cases"] /
    amount_result["total_claims"].replace(0, pd.NA) * 100
).round(2)

print(amount_result.to_string(index=False))
