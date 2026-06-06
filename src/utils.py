"""
utils.py
--------
Shared utilities: logging setup, path resolution, and constants.
"""

import logging
import sys
from pathlib import Path


# ── Project root (one level above /src) ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR     = PROJECT_ROOT / "models"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGS   = PROJECT_ROOT / "results" / "figures"
RESULTS_PREDS  = PROJECT_ROOT / "results" / "predictions"

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ── Feature lists ──────────────────────────────────────────────────────────────
NUMERICAL_FEATURES = [
    "price",
    "total_time_spent",
    "session_length",
    "interaction_count",
    "view_count",
    "click_count",
    "wishlist_count",
    "add_to_cart_count",
    "avg_time_per_interaction",
    "cart_to_view_ratio",
    "click_to_view_ratio",
    "has_cart_action",
    "has_wishlist_action",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

CATEGORICAL_FEATURES = [
    "category",
    "brand",
    "channel",
    "device_type",
    "region",
    "traffic_source",
]

TARGET = "is_conversion"

# Columns excluded to prevent leakage or memorisation
LEAKAGE_COLUMNS = [
    "is_conversion",
    "purchase_count",
    "drop_off_flag",
    "session_id",
    "user_id",
    "product_id",
    "timestamp_utc",
    "user_action",
    "user_conversion_rate",
    "user_total_purchases",
]


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger that writes to stdout.

    Parameters
    ----------
    name : str
        Logger name (use __name__ in calling module).
    level : int
        Logging level (default INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dirs() -> None:
    """Create all required project directories if they do not already exist."""
    for path in [DATA_RAW, DATA_PROCESSED, MODELS_DIR,
                 RESULTS_TABLES, RESULTS_FIGS, RESULTS_PREDS]:
        path.mkdir(parents=True, exist_ok=True)
