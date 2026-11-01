# Kestrel Home Warranty Claim Review
## AI-Powered Fraud Risk Scoring and Investigation Decision Support

## 1. Project Overview

This project focuses on developing a machine learning-based fraud risk scoring system for home warranty claims.

The objective is to help investigation teams prioritize claims for manual review by identifying potentially suspicious patterns in claim data.

The system generates a risk score for each claim, allowing investigators to focus their limited review capacity on higher-priority cases.

**Important:** This is a decision-support system. It does not automatically reject claims or confirm fraudulent activity.

## 2. Business Problem

Kestrel processes approximately 750 warranty claims per month, while its investigation team can manually review only 40 claims per month.

The challenge is to identify potentially fraudulent claims within this limited investigation capacity while minimizing unnecessary delays for genuine customers.

This project explores how machine learning can support claim prioritization.

## 3. Project Objectives

- Analyze historical warranty claim data.
- Identify data quality issues and suspicious patterns.
- Perform data cleaning and feature engineering.
- Experiment with multiple machine learning algorithms.
- Generate fraud risk scores for unlabelled claims.
- Build a REST API for claim-level predictions.
- Develop a web dashboard to interact with the prediction service.
- Evaluate model performance under a limited investigation capacity.

## 4. Technology Stack

| Category | Tools |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, CatBoost |
| Text Processing | TF-IDF |
| Data Visualization | Matplotlib, Seaborn |
| Backend | FastAPI |
| API Server | Uvicorn |
| Frontend | HTML, CSS, JavaScript |

## 5. Project Workflow

### Step 1: Exploratory Data Analysis

- Analyzed labelled and unlabelled claim datasets.
- Examined fraud distribution and class imbalance.
- Investigated duplicate claim submissions.
- Studied claim amounts, partner information, product information, and historical patterns.
- Examined policy-related changes and data quality issues.

### Step 2: Data Cleaning and Feature Engineering

- Handled duplicate claim identifiers.
- Excluded unresolved labels from supervised training.
- Parsed date fields and normalized serial numbers.
- Integrated partner and product reference information.
- Created features such as partner tenure, claim-to-product-price ratio, and policy-period indicators.
- Cleaned claim descriptions and removed suspicious instruction-like text.
- Avoided using post-submission information as model input.

### Step 3: Model Experimentation

Evaluated the following algorithms:

- Logistic Regression
- Random Forest
- XGBoost
- CatBoost

A chronological validation strategy was used to evaluate the models on later claims.

CatBoost was selected for the prediction service based on the experiments conducted.

### Step 4: Prediction Generation

Generated fraud risk scores for all records in the unlabelled test dataset.

The prediction output contains:

| Column | Description |
|---|---|
| `claim_id` | Claim identifier |
| `score` | Model-generated fraud risk score |

Higher scores indicate greater relative model-assessed risk.

## 6. Model Evaluation

The exploratory validation covered April–June 2026.

Under a simulated investigation capacity of 40 claims per month:

| Evaluation Metric | Result |
|---|---:|
| Claims reviewed | 120 |
| Fraud cases identified | 6 |
| Fraud cases missed | 39 |
| Precision | 5% |
| Recall | 13.33% |

### Key Observation

The model identified some fraudulent claims within the limited review capacity, but it also missed a substantial number of fraudulent cases.

The results indicate that the current model requires further improvement before operational deployment.

**Validation limitation:** The validation period was also used for model early stopping. Therefore, these results are exploratory and should not be treated as an independent estimate of future performance.

## 7. API Development

A FastAPI application was developed to provide claim-level predictions.

| Endpoint | Description |
|---|---|
| `GET /health` | Checks API and model status |
| `POST /predict` | Accepts claim details and returns a risk assessment |
| `/` | Opens the prediction dashboard |

The prediction response includes:

- Fraud risk score.
- Provisional risk category.
- Human-readable review signals.
- Investigation recommendation.

Risk categories are provisional and do not represent confirmed fraud probabilities.

## 8. Web Dashboard

A lightweight web interface was developed using HTML, CSS, and JavaScript.

The dashboard allows users to:

1. Enter individual claim details.
2. Submit the claim to the FastAPI endpoint.
3. Receive a model-generated risk score.
4. View the provisional risk category.
5. Review recommendation signals.

## 9. Repository Structure

```text
kestrel-warranty-fraud/
│
├── api/
│   ├── app.py
│   └── index.html
│
├── notebooks/
│   ├── 01_eda.py
│   ├── 02_feature_engineering.py
│   ├── 02_feature_engineering_backup.py
│   ├── 03_monthly_fraud_analysis.py
│   ├── 04_catboost_experiment.py
│   ├── 05_policy_aware_strategy.py
│   ├── 06_missed_fraud_analysis.py
│   ├── 07_review_allocation_experiment.py
│   ├── 08_low_value_fraud_experiment.py
│   └── 09_generate_predictions.py
│
├── reports/
│   └── eda_findings.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 10. Installation and Execution

### Prerequisites

- Python 3.10 or later
- pip
- Authorized access to the private assignment data and trained model

### Clone the repository

```bash
git clone https://github.com/VrohithaCodes/kestrel-warranty-fraud.git
cd kestrel-warranty-fraud
```

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure private assets

The prediction API requires authorized local copies of:

- `data/raw/train.csv`
- `data/raw/partners.csv`
- `data/raw/products.csv`
- `outputs/catboost_unweighted_model.cbm`

Obtain these files through the approved private assignment handover. Place them in their expected local directories.

### Start the API

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Open the dashboard:

`http://127.0.0.1:8000/`

API documentation:

`http://127.0.0.1:8000/docs`

## 11. Limitations and Future Improvements

- Address severe class imbalance.
- Improve the quality of historical fraud labels.
- Investigate duplicate and inconsistent claim records.
- Perform independent time-based validation.
- Improve fraud detection among low-value claims.
- Calibrate risk scores.
- Evaluate investigation thresholds using business costs.
- Monitor model performance and data drift.
- Avoid automated claim rejection without sufficient evidence.

## 12. Confidentiality

This public repository contains source code and non-sensitive documentation only.

Original customer datasets, partner records, claim-level predictions, trained model files, screen recordings, and private business documents are excluded from public GitHub uploads and must remain in approved private storage.

---

**Project Focus:** Machine Learning | Fraud Risk Scoring | Explainable Decision Support | FastAPI | Applied AI
### Dashboard Preview

The dashboard provides a simple interface for entering warranty claim
information and viewing the model-generated fraud risk assessment.

![Warranty Claim Review Dashboard](docs/images/dashboard_prediction.png)

The interface displays:
- Claim information
- Fraud risk score
- Provisional risk category
- Review signals
- Investigation recommendation