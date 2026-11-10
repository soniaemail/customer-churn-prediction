"""Central configuration: paths, constants and logging.

Every other module imports its paths and constants from here so that the
project has a single source of truth.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = ROOT_DIR / "data"
MODELS_DIR: Final[Path] = ROOT_DIR / "models"
REPORTS_DIR: Final[Path] = ROOT_DIR / "reports"
LOGS_DIR: Final[Path] = ROOT_DIR / "logs"
ASSETS_DIR: Final[Path] = ROOT_DIR / "assets"
LOTTIE_DIR: Final[Path] = ASSETS_DIR / "lottie"

RAW_DATA_PATH: Final[Path] = DATA_DIR / "telco_customer_churn.csv"
MODEL_PATH: Final[Path] = MODELS_DIR / "churn_model.pkl"
PREPROCESSOR_PATH: Final[Path] = MODELS_DIR / "preprocessor.pkl"

METRICS_PATH: Final[Path] = REPORTS_DIR / "metrics.json"
ROC_PATH: Final[Path] = REPORTS_DIR / "roc_curves.json"
METADATA_PATH: Final[Path] = REPORTS_DIR / "metadata.json"
SHAP_PATH: Final[Path] = REPORTS_DIR / "shap_values.joblib"
BASELINE_PATH: Final[Path] = REPORTS_DIR / "baseline_stats.json"
PREDICTION_LOG_PATH: Final[Path] = LOGS_DIR / "predictions.jsonl"

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
# Public mirrors of the IBM "Telco Customer Churn" sample dataset.
DATASET_URLS: Final[tuple[str, ...]] = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv",
)
EXPECTED_ROWS: Final[int] = 7043

TARGET: Final[str] = "Churn"
ID_COLUMN: Final[str] = "customerID"

# Raw feature groups (before feature engineering)
NUMERIC_FEATURES: Final[list[str]] = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FEATURES: Final[list[str]] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]
MULTI_FEATURES: Final[list[str]] = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]
RAW_FEATURES: Final[list[str]] = NUMERIC_FEATURES + BINARY_FEATURES + MULTI_FEATURES

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.20
CV_FOLDS: Final[int] = 5
SELECTION_METRIC: Final[str] = "roc_auc"  # metric used to pick the best model
N_SEARCH_ITER: Final[int] = 25  # RandomizedSearchCV iterations

# Risk tiers used by the app (probability thresholds)
LOW_RISK_MAX: Final[float] = 0.30
HIGH_RISK_MIN: Final[float] = 0.60


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str) -> logging.Logger:
    """Return a logger writing to the console and to ``logs/app.log``.

    The function is idempotent: calling it several times with the same name
    never duplicates handlers (important with Streamlit's script re-runs).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOGS_DIR / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:  # read-only file system (some PaaS): console logging only
        logger.warning("File logging disabled: %s is not writable", LOGS_DIR)

    logger.propagate = False
    return logger


def ensure_directories() -> None:
    """Create every output directory used by the project."""
    for directory in (DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR, LOTTIE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
