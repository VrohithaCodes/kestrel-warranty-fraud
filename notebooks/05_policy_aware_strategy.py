# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


# ============================================================
# 2. DEFINE PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. LOAD DATASETS
# ============================================================

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test_unlabelled.csv")

partners = pd.read_csv(DATA_DIR / "partners.csv")
products = pd.read_csv(DATA_DIR / "products.csv")


print("Training dataset:", train.shape)
print("Test dataset:", test.shape)
print("Partners dataset:", partners.shape)
print("Products dataset:", products.shape)

# ============================================================
# 4. INSPECT PREVIOUS VALIDATION PREDICTIONS
# ============================================================

validation_file = OUTPUT_DIR / "validation_predictions.csv"

if validation_file.exists():
    validation_predictions = pd.read_csv(validation_file)

    print("\nValidation predictions shape:")
    print(validation_predictions.shape)

    print("\nAvailable columns:")
    print(validation_predictions.columns.tolist())

    print("\nFirst five rows:")
    print(validation_predictions.head())

else:
    print("validation_predictions.csv not found!")