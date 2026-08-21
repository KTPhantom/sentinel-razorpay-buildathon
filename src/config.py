"""
Project-wide constants — paths, seeds, thresholds.
Everything random references RANDOM_SEED from here so results stay reproducible.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KAGGLE_DATA_PATH = DATA_DIR / "creditcard.csv"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_SEED: int = 42

TRAIN_RATIO: float = 0.70
VALIDATION_RATIO: float = 0.15
TEST_RATIO: float = 0.15

KAGGLE_EXPECTED_COLUMNS: list[str] = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Class",
]
KAGGLE_EXPECTED_ROW_COUNT: int = 284_807
KAGGLE_FRAUD_CLASS: int = 1

TOP_N_SHAP_FEATURES: int = 3

# Rule engine thresholds
RULE_VELOCITY_THRESHOLD: int = 3
RULE_GEO_DISTANCE_KM: float = 500.0
RULE_AMOUNT_RATIO: float = 5.0
RULE_NEW_DEVICE_AMOUNT_RATIO: float = 2.0
RULE_ODD_HOUR_AMOUNT_RATIO: float = 3.0

# Synthetic data defaults
SYNTHETIC_NORMAL_COUNT: int = 2_000
SYNTHETIC_ANOMALY_COUNT: int = 100
