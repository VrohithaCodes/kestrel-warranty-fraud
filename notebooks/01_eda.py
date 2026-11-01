
# ==========================================
# Kestrel Warranty Fraud Detection
# Step 1: Exploratory Data Analysis (EDA)
# Part 1: Loading and Understanding Data
# ==========================================

import pandas as pd
import numpy as np
from pathlib import Path

# 1. Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

# 2. Load datasets
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test_unlabelled.csv")
partners = pd.read_csv(DATA_DIR / "partners.csv")
products = pd.read_csv(DATA_DIR / "products.csv")

# 3. Display dataset dimensions
print("\n--- DATASET SHAPES ---")
print("Training data:", train.shape)
print("Testing data:", test.shape)
print("Partners data:", partners.shape)
print("Products data:", products.shape)

# 4. Display training dataset columns
print("\n--- COLUMN NAMES ---")
print(train.columns.tolist())

# 5. Display first five training records
print("\n--- FIRST FIVE RECORDS ---")
print(train.head())

# 6. Display data types
print("\n--- DATA TYPES ---")
print(train.dtypes)

# 7. Display statistical summary
print("\n--- STATISTICAL SUMMARY ---")
print(train.describe(include="all").T)




# ==========================================
# PART 2: DATA QUALITY ANALYSIS
# ==========================================

# 1. Check missing values
print("\n--- MISSING VALUES ---")

missing = train.isnull().sum()

print(missing[missing > 0])

# 2. Check duplicate claim IDs
print("\n--- DUPLICATE CLAIM IDS ---")

duplicate_count = train["claim_id"].duplicated().sum()

print("Duplicate records:", duplicate_count)

# 3. Check fraud label distribution
print("\n--- FRAUD LABEL DISTRIBUTION ---")

print(train["is_fraud"].value_counts(dropna=False))

# 4. Calculate fraud percentage among decided claims
labelled = train.dropna(subset=["is_fraud"])

fraud_percentage = (
    labelled["is_fraud"].mean() * 100
)

print("\n--- FRAUD RATE ---")

print(f"Fraud percentage: {fraud_percentage:.2f}%")

# 5. Check dataset date range
print("\n--- DATE RANGE ---")

train["submitted_at"] = pd.to_datetime(
    train["submitted_at"]
)

print("First claim:", train["submitted_at"].min())
print("Last claim:", train["submitted_at"].max())


# ==========================================
# PART 3: FRAUD DISTRIBUTION VISUALIZATION
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Exclude undecided claims for label analysis
labelled = train.dropna(subset=["is_fraud"]).copy()

labelled["fraud_status"] = labelled["is_fraud"].map({
    0.0: "Genuine",
    1.0: "Fraud"
})

# Plot fraud distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=labelled,
    x="fraud_status",
    hue="fraud_status",
    order=["Genuine", "Fraud"],
    palette=["seagreen", "tomato"],
    legend=False
)

plt.title("Fraud vs Genuine Claims")
plt.xlabel("Claim Status")
plt.ylabel("Number of Claims")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fraud_distribution.png",
    dpi=300
)

plt.show()

print("\nFraud distribution graph saved successfully!")



# ==========================================
# PART 4: CLAIM AMOUNT ANALYSIS
# ==========================================

# 1. Calculate claim amount statistics
print("\n--- CLAIM AMOUNT ANALYSIS ---")

amount_summary = labelled.groupby("fraud_status")[
    "claim_amount_inr"
].agg(["count", "mean", "median", "min", "max"])

print(amount_summary)

# 2. Visualize claim amounts by fraud status
plt.figure(figsize=(9, 5))

sns.boxplot(
    data=labelled,
    x="fraud_status",
    y="claim_amount_inr",
    hue="fraud_status",
    order=["Genuine", "Fraud"],
    legend=False
)

plt.title("Claim Amount by Fraud Status")
plt.xlabel("Claim Status")
plt.ylabel("Claim Amount (INR)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "claim_amount_analysis.png",
    dpi=300
)

plt.show()

# 3. Compare claim amount distributions
plt.figure(figsize=(9, 5))

sns.histplot(
    data=labelled,
    x="claim_amount_inr",
    hue="fraud_status",
    bins=40,
    element="step",
    stat="density",
    common_norm=False
)

plt.title("Distribution of Claim Amounts")
plt.xlabel("Claim Amount (INR)")
plt.ylabel("Density")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "claim_amount_distribution.png",
    dpi=300
)

plt.show()

print("\nClaim amount analysis completed!")



# ==========================================
# PART 5: CUSTOMER CLAIM HISTORY ANALYSIS
# ==========================================

print("\n--- CUSTOMER CLAIM HISTORY ANALYSIS ---")

# Group claims by previous claim count
customer_analysis = labelled.groupby(
    "customer_prior_claims"
).agg(
    total_claims=("is_fraud", "count"),
    fraud_cases=("is_fraud", "sum")
)

# Calculate fraud percentage
customer_analysis["fraud_rate"] = (
    customer_analysis["fraud_cases"]
    / customer_analysis["total_claims"]
) * 100

print(customer_analysis)

# Visualize fraud rate by previous claims
plt.figure(figsize=(9, 5))

sns.barplot(
    data=customer_analysis.reset_index(),
    x="customer_prior_claims",
    y="fraud_rate",
    hue="customer_prior_claims",
    palette="viridis",
    legend=False
)

plt.title("Fraud Rate by Customer Previous Claims")
plt.xlabel("Number of Previous Claims")
plt.ylabel("Fraud Rate (%)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "customer_claim_history.png",
    dpi=300
)

plt.show()

print("\nCustomer claim history analysis completed!")



# ==========================================
# PART 6: PRODUCT-WISE FRAUD ANALYSIS
# ==========================================

print("\n--- PRODUCT-WISE FRAUD ANALYSIS ---")

# Merge training claims with product information
product_data = labelled.merge(
    products,
    on="sku",
    how="left"
)

# Group claims by product family
product_analysis = product_data.groupby(
    "family"
).agg(
    total_claims=("is_fraud", "count"),
    fraud_cases=("is_fraud", "sum"),
    average_claim_amount=("claim_amount_inr", "mean")
)

# Calculate fraud rate
product_analysis["fraud_rate"] = (
    product_analysis["fraud_cases"]
    / product_analysis["total_claims"]
) * 100

print(product_analysis.sort_values(
    by="fraud_rate",
    ascending=False
))

# Visualize product-wise fraud rates
plt.figure(figsize=(11, 6))

sns.barplot(
    data=product_analysis.reset_index(),
    x="family",
    y="fraud_rate",
    hue="family",
    palette="magma",
    legend=False
)

plt.title("Fraud Rate by Product Family")
plt.xlabel("Product Family")
plt.ylabel("Fraud Rate (%)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "product_fraud_analysis.png",
    dpi=300
)

plt.show()

print("\nProduct analysis completed!")

# ==========================================
# PART 7: PARTNER-WISE FRAUD ANALYSIS
# ==========================================

print("\n--- PARTNER-WISE FRAUD ANALYSIS ---")

# Merge training claims with partner information
partner_data = labelled.merge(
    partners,
    on="partner_id",
    how="left"
)

# Group claims by partner type
partner_analysis = partner_data.groupby(
    "partner_type"
).agg(
    total_claims=("is_fraud", "count"),
    fraud_cases=("is_fraud", "sum"),
    average_claim_amount=("claim_amount_inr", "mean")
)

# Calculate fraud rate
partner_analysis["fraud_rate"] = (
    partner_analysis["fraud_cases"]
    / partner_analysis["total_claims"]
) * 100

print(partner_analysis.sort_values(
    by="fraud_rate",
    ascending=False
))

# Visualize partner-wise fraud rates
plt.figure(figsize=(9, 5))

sns.barplot(
    data=partner_analysis.reset_index(),
    x="partner_type",
    y="fraud_rate",
    hue="partner_type",
    palette="viridis",
    legend=False
)

plt.title("Fraud Rate by Partner Type")
plt.xlabel("Partner Type")
plt.ylabel("Fraud Rate (%)")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "partner_fraud_analysis.png",
    dpi=300
)

plt.show()

print("\nPartner analysis completed!")



# ==========================================
# PART 8: PARTNER TENURE ANALYSIS
# ==========================================

print("\n--- PARTNER TENURE ANALYSIS ---")

# Merge partner onboarding information with claims
tenure_data = labelled.merge(
    partners[["partner_id", "onboarded_date"]],
    on="partner_id",
    how="left"
)

# Convert dates
tenure_data["submitted_at"] = pd.to_datetime(
    tenure_data["submitted_at"]
)

tenure_data["onboarded_date"] = pd.to_datetime(
    tenure_data["onboarded_date"]
)

# Calculate partner experience in days at claim submission
tenure_data["partner_tenure_days"] = (
    tenure_data["submitted_at"] -
    tenure_data["onboarded_date"]
).dt.days

# Create experience groups
tenure_data["partner_experience"] = pd.cut(
    tenure_data["partner_tenure_days"],
    bins=[-1, 90, 180, 365, float("inf")],
    labels=["0-3 months", "3-6 months",
            "6-12 months", "Over 1 year"]
)

# Analyse fraud rates by experience
tenure_analysis = tenure_data.groupby(
    "partner_experience",
    observed=True
).agg(
    total_claims=("is_fraud", "count"),
    fraud_cases=("is_fraud", "sum")
)

tenure_analysis["fraud_rate"] = (
    tenure_analysis["fraud_cases"] /
    tenure_analysis["total_claims"]
) * 100

print(tenure_analysis)

print("\nPartner tenure analysis completed!")



# ==========================================
# PART 9: PARTNER-LEVEL FRAUD CONCENTRATION
# ==========================================

print("\n--- PARTNER-LEVEL FRAUD CONCENTRATION ---")

individual_partner_analysis = labelled.groupby(
    "partner_id"
).agg(
    total_claims=("is_fraud", "count"),
    fraud_cases=("is_fraud", "sum")
)

individual_partner_analysis["fraud_rate"] = (
    individual_partner_analysis["fraud_cases"]
    / individual_partner_analysis["total_claims"]
) * 100

# Consider partners with at least 20 labelled claims
eligible_partners = individual_partner_analysis[
    individual_partner_analysis["total_claims"] >= 20
].copy()

print("\nPartners with at least 20 claims:")
print(
    eligible_partners.sort_values(
        by="fraud_rate",
        ascending=False
    ).head(15)
)

print("\nTotal unique partners:", len(individual_partner_analysis))
print("Partners with at least 20 claims:", len(eligible_partners))

print("\nPartner concentration analysis completed!")

# ==========================================
# PART 10: TIME-BASED FRAUD TREND ANALYSIS
# ==========================================

print("\n--- TIME-BASED FRAUD TREND ANALYSIS ---")

# Exclude undecided claims
time_data = labelled.copy()

# Avoid counting repeated claim IDs more than once
time_data = (
    time_data.sort_values("submitted_at")
    .drop_duplicates(subset="claim_id", keep="first")
)

# Extract month and year
time_data["month"] = time_data["submitted_at"].dt.to_period("M")

# Calculate monthly claims and fraud rates
monthly_analysis = time_data.groupby("month").agg(
    total_claims=("is_fraud", "count"),
    fraud_cases=("is_fraud", "sum")
)

monthly_analysis["fraud_rate"] = (
    monthly_analysis["fraud_cases"]
    / monthly_analysis["total_claims"]
) * 100

print("\nMonthly fraud analysis:")
print(monthly_analysis)

# Compare before and after May 1, 2026
time_data["policy_period"] = time_data["submitted_at"].apply(
    lambda date: "Before May 2026"
    if date < pd.Timestamp("2026-05-01")
    else "From May 2026"
)

policy_trend = time_data.groupby("policy_period").agg(
    total_claims=("is_fraud", "count"),
    fraud_cases=("is_fraud", "sum")
)

policy_trend["fraud_rate"] = (
    policy_trend["fraud_cases"]
    / policy_trend["total_claims"]
) * 100

print("\nFraud rate before and after May 1, 2026:")
print(policy_trend)

# Plot monthly fraud rate
plt.figure(figsize=(12, 6))

plt.plot(
    monthly_analysis.index.astype(str),
    monthly_analysis["fraud_rate"],
    marker="o"
)

plt.axvline(
    x=list(monthly_analysis.index.astype(str)).index("2026-05"),
    linestyle="--",
    label="May 2026 policy change"
)

plt.title("Monthly Fraud Rate Trend")
plt.xlabel("Month")
plt.ylabel("Fraud Rate (%)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.savefig(
    BASE_DIR / "outputs" / "monthly_fraud_trend.png"
)

plt.close()

print("\nTime-based fraud analysis completed!")

# ==========================================
# PART 11: POLICY-BASED CLAIM ANALYSIS
# ==========================================

print("\n--- POLICY-BASED CLAIM ANALYSIS ---")

# Use labelled claims and remove repeated claim IDs
policy_data = labelled.copy()

policy_data = (
    policy_data.sort_values("submitted_at")
    .drop_duplicates(subset="claim_id", keep="first")
)

# Identify policy period
policy_data["policy_period"] = np.where(
    policy_data["submitted_at"] < pd.Timestamp("2026-05-01"),
    "Before May 1, 2026",
    "From May 1, 2026"
)

# Group claims by policy period and amount
policy_data["claim_amount_group"] = np.where(
    policy_data["claim_amount_inr"] < 2000,
    "Below ₹2,000",
    "₹2,000 or above"
)

# Compare inspection patterns
inspection_analysis = policy_data.groupby(
    ["policy_period", "claim_amount_group"]
).agg(
    total_claims=("claim_id", "count"),
    inspected_claims=(
        "partner_inspected",
        lambda x: (x == "Y").sum()
    ),
    fraud_cases=("is_fraud", "sum")
)

inspection_analysis["inspection_rate"] = (
    inspection_analysis["inspected_claims"]
    / inspection_analysis["total_claims"]
) * 100

inspection_analysis["fraud_rate"] = (
    inspection_analysis["fraud_cases"]
    / inspection_analysis["total_claims"]
) * 100

print("\nInspection and fraud analysis:")
print(inspection_analysis)
print(inspection_analysis.to_string())

print("\nPolicy-based analysis completed!")

# ==========================================
# PART 12: DUPLICATE CLAIM AND SERIAL ANALYSIS
# ==========================================

print("\n--- DUPLICATE CLAIM AND SERIAL ANALYSIS ---")

# Work with the complete training dataset
duplicate_data = train.copy()

# 1. Duplicate claim IDs
duplicate_claims = duplicate_data[
    duplicate_data.duplicated(
        subset="claim_id",
        keep=False
    )
].sort_values("claim_id")

print("\nDuplicate claim ID summary:")
print("Total rows involved:", len(duplicate_claims))
print("Unique duplicated claim IDs:",
      duplicate_claims["claim_id"].nunique())

# 2. Normalize product serial numbers
duplicate_data["normalized_serial"] = (
    duplicate_data["product_serial"]
    .astype(str)
    .str.upper()
    .str.replace(r"[^A-Z0-9]", "", regex=True)
)

# 3. Find repeated serial numbers
serial_counts = duplicate_data.groupby(
    "normalized_serial"
).agg(
    total_claims=("claim_id", "count"),
    unique_claims=("claim_id", "nunique"),
    fraud_cases=("is_fraud", "sum")
)

repeated_serials = serial_counts[
    serial_counts["unique_claims"] > 1
].copy()

print("\nRepeated product serial summary:")
print("Unique serial numbers:",
      duplicate_data["normalized_serial"].nunique())

print("Repeated serial numbers:",
      len(repeated_serials))

print("Claims involving repeated serials:",
      repeated_serials["total_claims"].sum())

print("\nTop repeated serial numbers:")
print(
    repeated_serials.sort_values(
        "total_claims",
        ascending=False
    ).head(10)
)

print("\nDuplicate and serial analysis completed!")