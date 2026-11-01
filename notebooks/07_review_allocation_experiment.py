
import pandas as pd
from pathlib import Path

# ==========================================
# 1. LOAD VALIDATION PREDICTIONS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"

df = pd.read_csv(
    OUTPUT_DIR / "catboost_validation_predictions.csv"
)

df["submitted_at"] = pd.to_datetime(df["submitted_at"])
df["month"] = df["submitted_at"].dt.to_period("M")

df["is_fraud"] = df["is_fraud"].astype(int)

REVIEW_CAPACITY = 40
LOW_VALUE_LIMIT = 2000

print("\n========== DATASET SUMMARY ==========")
print("Validation claims:", len(df))
print("Fraud cases:", df["is_fraud"].sum())
print("Months:", sorted(df["month"].astype(str).unique()))


# ==========================================
# 2. DEFINE REVIEW STRATEGIES
# ==========================================

def select_top_claims(data, count):
    """Select highest-scoring claims within a group."""
    return data.nlargest(count, "catboost_score")


def strategy_a(month_data):
    # Current approach: top 40 overall
    return select_top_claims(month_data, REVIEW_CAPACITY)


def strategy_b(month_data):
    # 20 low-value + 20 remaining claims
    low_value = month_data[
        month_data["claim_amount_inr"] < LOW_VALUE_LIMIT
    ]

    other = month_data[
        month_data["claim_amount_inr"] >= LOW_VALUE_LIMIT
    ]

    selected = pd.concat([
        select_top_claims(low_value, 20),
        select_top_claims(other, 20)
    ])

    return selected.drop_duplicates("claim_id")


def strategy_c(month_data):
    # 30 low-value + 10 remaining claims
    low_value = month_data[
        month_data["claim_amount_inr"] < LOW_VALUE_LIMIT
    ]

    other = month_data[
        month_data["claim_amount_inr"] >= LOW_VALUE_LIMIT
    ]

    selected = pd.concat([
        select_top_claims(low_value, 30),
        select_top_claims(other, 10)
    ])

    return selected.drop_duplicates("claim_id")


strategies = {
    "Strategy A - Top 40 Overall": strategy_a,
    "Strategy B - 20 Low + 20 Other": strategy_b,
    "Strategy C - 30 Low + 10 Other": strategy_c
}


# ==========================================
# 3. EVALUATE EACH STRATEGY MONTHLY
# ==========================================

monthly_results = []

for month, month_data in df.groupby("month"):

    total_fraud = int(month_data["is_fraud"].sum())

    for strategy_name, strategy_function in strategies.items():

        selected = strategy_function(month_data)

        detected_fraud = int(selected["is_fraud"].sum())
        selected_count = len(selected)

        precision = (
            detected_fraud / selected_count * 100
            if selected_count else 0
        )

        recall = (
            detected_fraud / total_fraud * 100
            if total_fraud else 0
        )

        monthly_results.append({
            "month": str(month),
            "strategy": strategy_name,
            "claims_reviewed": selected_count,
            "total_fraud": total_fraud,
            "fraud_detected": detected_fraud,
            "fraud_missed": total_fraud - detected_fraud,
            "precision_percent": round(precision, 2),
            "recall_percent": round(recall, 2)
        })


monthly_results = pd.DataFrame(monthly_results)


# ==========================================
# 4. DISPLAY MONTHLY RESULTS
# ==========================================

print("\n========== MONTHLY STRATEGY COMPARISON ==========")

print(
    monthly_results.to_string(index=False)
)


# ==========================================
# 5. CALCULATE OVERALL RESULTS
# ==========================================

overall_results = (
    monthly_results.groupby("strategy")
    .agg(
        total_claims_reviewed=("claims_reviewed", "sum"),
        total_fraud_detected=("fraud_detected", "sum"),
        total_fraud_missed=("fraud_missed", "sum")
    )
    .reset_index()
)

overall_results["total_fraud"] = (
    overall_results["total_fraud_detected"]
    + overall_results["total_fraud_missed"]
)

overall_results["precision_percent"] = (
    overall_results["total_fraud_detected"]
    / overall_results["total_claims_reviewed"] * 100
).round(2)

overall_results["recall_percent"] = (
    overall_results["total_fraud_detected"]
    / overall_results["total_fraud"] * 100
).round(2)


# ==========================================
# 6. DISPLAY OVERALL COMPARISON
# ==========================================

print("\n========== OVERALL STRATEGY COMPARISON ==========")

print(overall_results.to_string(index=False))


# ==========================================
# 7. SAVE RESULTS
# ==========================================

monthly_results.to_csv(
    OUTPUT_DIR / "review_strategy_monthly_results.csv",
    index=False
)

overall_results.to_csv(
    OUTPUT_DIR / "review_strategy_overall_results.csv",
    index=False
)

print("\n==========================================")
print("Review allocation experiment completed.")
print("Results saved in:", OUTPUT_DIR)
print("==========================================")
