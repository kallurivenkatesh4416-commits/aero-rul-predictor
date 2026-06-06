"""
Project configuration file.

This file stores common project paths and settings.
Keeping paths in one place makes the project easier to maintain.
"""

from pathlib import Path


# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# NASA C-MAPSS extracted data directory
CMAPSS_DATA_DIR = RAW_DATA_DIR / "cmapss" / "CMAPSSData"

# Model directory
MODEL_DIR = PROJECT_ROOT / "models"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"

# Raw NASA C-MAPSS file names
TRAIN_FILE_NAME = "train_FD001.txt"
TEST_FILE_NAME = "test_FD001.txt"
RUL_FILE_NAME = "RUL_FD001.txt"

# Full raw file paths
TRAIN_FILE_PATH = CMAPSS_DATA_DIR / TRAIN_FILE_NAME
TEST_FILE_PATH = CMAPSS_DATA_DIR / TEST_FILE_NAME
RUL_FILE_PATH = CMAPSS_DATA_DIR / RUL_FILE_NAME

# Saved model path
MODEL_PATH = MODEL_DIR / "xgboost_rul_model.joblib"

# Random seed for reproducibility
RANDOM_STATE = 42

# Maximum RUL cap
# In predictive maintenance, very high RUL values are usually capped
# to make the target more stable for model training.
MAX_RUL = 125

# Final tuned XGBoost hyperparameters
# These were selected from the experimentation notebook using
# GroupKFold cross-validation to avoid engine-level data leakage.
XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 700,
    "learning_rate": 0.01,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}