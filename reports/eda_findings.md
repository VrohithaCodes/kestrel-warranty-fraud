# Exploratory Data Analysis (EDA) — Kestrel Warranty Claims

## 1. Dataset Overview

The objective of this analysis is to understand historical warranty claim patterns, identify potential fraud indicators, and assess data quality before developing a fraud risk scoring model.

| Dataset | Rows | Columns |
|---|---:|---:|
| Training data | 12,029 | 14 |
| Unlabelled test data | 2,252 | 13 |
| Partner reference data | 380 | 4 |
| Product reference data | 21 | 4 |

The training data covers claims submitted between **April 1, 2025 and June 30, 2026**.

The training dataset contains 215 claims with undecided fraud labels. These records should not be treated as confirmed genuine claims during model training.

## 2. Fraud Label Distribution

**Observation:** The labelled dataset is highly imbalanced.

- Genuine claims: 11,669
- Fraudulent claims: 145
- Undecided claims: 215
- Fraud rate among labelled claims: approximately 1.23%

**Interpretation:** Fraudulent claims represent a very small proportion of the labelled data.

**Model implication:** Accuracy alone is not an appropriate measure of model performance. Evaluation should also consider precision, recall, PR-AUC, and the number of fraud cases identified within the investigation team's review capacity.

## 3. Claim Amount Analysis

**Observation:** Fraudulent claims have higher average and median claim amounts than genuine claims.

| Claim category | Number of claims | Average amount | Median amount |
|---|---:|---:|---:|
| Fraudulent | 145 | ₹4,449 | ₹2,202 |
| Genuine | 11,669 | ₹2,594 | ₹1,428 |

**Interpretation:** Claim amount may be a useful indicator of fraud risk. However, the distributions overlap, so a high claim amount does not automatically mean a claim is fraudulent.

**Model implication:** Include claim amount as a numerical feature, but combine it with other claim characteristics rather than using a fixed amount threshold as the only fraud detection rule.

## 4. Customer Claim History Analysis

**Observation:** The observed fraud rate generally increases with the number of previous customer claims.

| Previous claims | Total claims | Fraud cases | Fraud rate |
|---:|---:|---:|---:|
| 0 | 8,693 | 66 | 0.76% |
| 1 | 2,646 | 45 | 1.70% |
| 2 | 421 | 23 | 5.46% |
| 3 | 46 | 7 | 15.22% |
| 4 | 6 | 2 | 33.33% |
| 5 | 1 | 1 | 100% |
| 6 | 1 | 1 | 100% |

**Interpretation:** Customers with more previous claims show higher observed fraud rates in this dataset. However, the groups with four or more previous claims contain very few records, making their percentages unstable.

**Model implication:** Customer claim history can be considered a useful risk feature. Small groups should not be interpreted as conclusive evidence of fraud.

## 5. Product-wise Fraud Analysis

**Observation:** Fraud rates vary across product families.

| Product family | Total claims | Fraud rate |
|---|---:|---:|
| Robot Vacuum | 1,706 | 1.58% |
| Water Purifier | 1,630 | 1.47% |
| Induction Cooktop | 1,694 | 1.24% |
| Mixer Grinder | 1,704 | 1.23% |
| Air Fryer | 1,678 | 1.19% |
| Ceiling Fan | 1,701 | 0.94% |
| Room Heater | 1,701 | 0.94% |

**Interpretation:** Robot Vacuum and Water Purifier claims have somewhat higher observed fraud rates. The differences are relatively small and do not establish that any product family is inherently more associated with fraud.

**Model implication:** Product family may be included as a categorical feature and evaluated alongside other claim characteristics.

## 6. Data Quality Observations

The initial analysis identified the following issues:

- **Missing inspector notes:** 3,237 records have no inspector note. This field may be unavailable when an inspection has not taken place.
- **Undecided labels:** 215 records have missing fraud outcomes. These should be excluded from supervised model training unless their outcomes are later confirmed.
- **Duplicate claim IDs:** 681 duplicate claim ID records were identified. The assignment notes that partner resubmissions can produce duplicate claim numbers, so these records require careful handling.
- **Legacy data:** The assignment notes that older Zoho records could contain zero values for unresolved cases because the system could not store blank outcomes. Historical zero labels therefore require caution.

**Model implication:** Data cleaning and validation are necessary before training. Duplicate claims must be handled carefully to avoid data leakage, and potentially unreliable historical labels should be considered during validation.

## 7. Initial Conclusions

The EDA suggests several potentially useful fraud risk indicators:

1. Claim amount.
2. Customer's previous claim count.
3. Product family.
4. Additional claim and partner characteristics to be investigated in subsequent analysis.

The dataset has a significant class imbalance and data quality limitations. Therefore, the model should be evaluated using appropriate fraud detection metrics and a time-based validation strategy.

**Important:** These findings describe associations in the available historical data. They do not establish that any single feature causes fraudulent behaviour. Further validation is required before using these observations to make claim decisions.

## 6. Partner-wise Fraud Analysis

**Observation:** Fraud rates vary across partner types.

| Partner type | Total claims | Fraud rate |
|---|---:|---:|
| Franchise | 4,057 | 1.97% |
| Authorised service centre | 5,417 | 0.98% |
| Freelance technician | 2,340 | 0.51% |

**Interpretation:** Franchise claims have the highest observed fraud rate, followed by authorised service centres and freelance technicians. However, these differences do not establish that a particular partner category causes fraudulent claims.

**Model implication:** Partner type may be considered as a categorical feature. The model should assess individual claim risk rather than automatically treating all claims from a particular partner category as suspicious.
Partner-Level Fraud Concentration: Partner-level analysis identified variation in observed fraud rates across partners. Among partners with at least 20 labelled claims, some showed substantially higher rates than the overall dataset fraud rate of 1.23%. The highest observed rate was 35% (14 fraud cases among 40 claims). These patterns may help prioritize investigations, but they should not be treated as proof of fraud. Duplicate claim IDs and repeated submissions must also be considered before using partner-level statistics in the model.


Policy-Based Claim Analysis: After May 1, 2026, claims below ₹2,000 had an observed fraud rate of 3.23%, compared with 0.54% before the policy change. For claims of ₹2,000 or above, the observed fraud rate decreased from 1.93% to 0.30%. This shift suggests that low-value claims deserve closer monitoring under the auto-approval policy. However, the results are observational and do not establish that the policy caused the change.


Inspection policy compliance
Period and claim amount	Inspection rate
Before May — below ₹2,000	92.92%
Before May — ₹2,000 or above	93.11%
From May — below ₹2,000	0.00%
From May — ₹2,000 or above	91.99%

Policy pattern confirmed
From May 2026, none of the 1,085 labelled claims below ₹2,000 were inspected. Claims of ₹2,000 or above continued to have a high inspection rate.

Inspection Policy Analysis: Following the May 1, 2026 policy change, all 1,085 labelled claims below ₹2,000 in the analysed dataset were recorded as not inspected. Claims of ₹2,000 or above had an inspection rate of approximately 91.99%. The low-value group also showed a higher observed fraud rate of 3.23%. This suggests that low-value auto-approved claims deserve risk-based monitoring. The analysis does not establish that lack of inspection caused fraud.

### 12. Duplicate Claim and Product Serial Analysis

The training dataset contains 681 duplicated claim IDs, involving 1,362 rows. These records may represent partner resubmissions rather than independent claims. Duplicate records must therefore be considered during model validation to prevent data leakage.

Product serial numbers were normalized by converting them to uppercase and removing spaces and special characters. The analysis identified 7,049 unique normalized serial numbers, of which 3,073 appeared across multiple claims, involving 7,818 rows.

Repeated serial numbers may indicate multiple claims for the same product, resubmissions, data-entry inconsistencies, or potentially suspicious activity. However, repeated serial numbers alone are not sufficient evidence of fraud.

**Modeling implications:**
- Avoid placing duplicate claim IDs across training and validation sets.
- Normalize serial numbers consistently.
- Consider repeated serial numbers as a risk feature.
- Avoid automatically classifying repeated serial numbers as fraudulent.