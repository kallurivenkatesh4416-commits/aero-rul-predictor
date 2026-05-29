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

# Model directory
MODEL_DIR = PROJECT_ROOT / "models"

print(MODEL_DIR)