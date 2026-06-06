"""
data_preprocessing.py
----------------------
Loads the raw event-level CSV, validates it, and writes a cleaned
event-level file to data/processed/.

Run directly:
    python src/data_preprocessing.py
"""

from __future__ import annotations

import pandas as pd

from utils import (
    DATA_RAW,
    DATA_PROCESSED,
    get_logger,
    ensure_dirs,
)

logger = get_logger(__name__)

RAW_FILE       = DATA_RAW / "retail_user_behavior_100k.csv"
CLEAN_FILE     = DATA_PROCESSED / "retail_clean_event_level.csv"

REQUIRED_COLUMNS = [
    "session_id", "user_id", "product_id", "timestamp_utc",
    "user_action", "category", "brand", "channel", "device_type",
    "region", "traffic_source", "price", "time_spent_sec",
    "session_length", "interaction_count", "is_conversion", "drop_off_flag",
]

EXPECTED_ACTIONS = {"view", "click", "wishlist", "add_to_cart", "purchase", "drop"}


# ── Public API ─────────────────────────────────────────────────────────────────

def load_raw(path: str | None = None) -> pd.DataFrame:
    """
    Load the raw event-level dataset.

    Parameters
    ----------
    path : str or None
        Path to the raw CSV. Defaults to data/raw/retail_user_behavior_100k.csv.

    Returns
    -------
    pd.DataFrame
    """
    src = path or RAW_FILE
    logger.info("Loading raw data from %s", src)
    df = pd.read_csv(src)
    logger.info("Raw dataset loaded — %d rows, %d columns", *df.shape)
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run basic structural checks on the raw DataFrame.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    unknown_actions = set(df["user_action"].unique()) - EXPECTED_ACTIONS - {""}
    if unknown_actions:
        logger.warning("Unexpected user_action values: %s", unknown_actions)

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning rules to the raw event-level DataFrame.

    Steps
    -----
    1. Drop exact duplicate rows.
    2. Strip whitespace from string columns.
    3. Coerce numeric columns; drop rows where price/time_spent_sec are non-positive.
    4. Standardise categorical casing.

    Returns
    -------
    pd.DataFrame – cleaned event-level data.
    """
    n_before = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows", n_before - len(df))

    # Strip string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # Coerce and filter numerics
    for col in ["price", "time_spent_sec"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df[df["price"] > 0]
    df = df[df["time_spent_sec"] >= 0]
    logger.info("Removed %d rows with invalid numeric values", before - len(df))

    # Standardise categorical casing
    for col in ["category", "brand", "channel", "device_type", "region", "traffic_source"]:
        df[col] = df[col].str.lower().str.strip()

    df = df.reset_index(drop=True)
    logger.info("Cleaned dataset — %d rows remain", len(df))
    return df


def run_pipeline(raw_path: str | None = None) -> pd.DataFrame:
    """
    Full preprocessing pipeline: load → validate → clean → save.

    Returns
    -------
    pd.DataFrame – cleaned event-level data.
    """
    ensure_dirs()
    df = load_raw(raw_path)
    df = validate(df)
    df = clean(df)
    df.to_csv(CLEAN_FILE, index=False)
    logger.info("Saved cleaned data to %s", CLEAN_FILE)
    return df


# ── CLI entry-point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
