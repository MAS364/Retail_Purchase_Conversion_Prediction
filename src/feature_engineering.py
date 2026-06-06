"""
feature_engineering.py
-----------------------
Converts cleaned event-level data into a session-level modelling dataset.

Transformations
---------------
1. Group events by session_id.
2. Pivot user_action counts into per-session feature columns.
3. Derive intent ratios and binary intent flags.
4. Extract temporal features from the first event timestamp.
5. Remove leakage-prone columns.

Run directly:
    python src/feature_engineering.py
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from utils import (
    DATA_PROCESSED,
    RESULTS_TABLES,
    TARGET,
    LEAKAGE_COLUMNS,
    get_logger,
    ensure_dirs,
)

logger = get_logger(__name__)

CLEAN_FILE   = DATA_PROCESSED / "retail_clean_event_level.csv"
SESSION_FILE = DATA_PROCESSED / "retail_session_level.csv"


# ── Feature construction helpers ───────────────────────────────────────────────

def _build_action_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot user_action events into per-session count columns."""
    counts = pd.crosstab(df["session_id"], df["user_action"])

    for action in ["view", "click", "wishlist", "add_to_cart", "purchase", "drop"]:
        if action not in counts.columns:
            counts[action] = 0

    counts = counts.rename(columns={
        "view":        "view_count",
        "click":       "click_count",
        "wishlist":    "wishlist_count",
        "add_to_cart": "add_to_cart_count",
        "purchase":    "purchase_count",
        "drop":        "drop_count",
    }).reset_index()

    return counts


def _aggregate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level rows to one row per session."""
    session_df = df.groupby("session_id").agg(
        user_id          =("user_id",          "first"),
        product_id       =("product_id",       "first"),
        timestamp_utc    =("timestamp_utc",    "first"),
        category         =("category",         "first"),
        brand            =("brand",            "first"),
        channel          =("channel",          "first"),
        device_type      =("device_type",      "first"),
        region           =("region",           "first"),
        traffic_source   =("traffic_source",   "first"),
        price            =("price",            "mean"),
        total_time_spent =("time_spent_sec",   "sum"),
        session_length   =("session_length",   "max"),
        interaction_count=("interaction_count","max"),
        is_conversion    =("is_conversion",    "max"),
        drop_off_flag    =("drop_off_flag",    "max"),
    ).reset_index()
    return session_df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day_of_week, month, is_weekend from timestamp_utc."""
    ts = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    df["hour"]        = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek          # 0 = Monday
    df["month"]       = ts.dt.month
    df["is_weekend"]  = ts.dt.dayofweek.isin([5, 6]).astype(int)
    return df


def _add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute intent-ratio and binary intent features."""
    df["cart_to_view_ratio"] = np.where(
        df["view_count"] > 0,
        df["add_to_cart_count"] / df["view_count"],
        0.0,
    )
    df["click_to_view_ratio"] = np.where(
        df["view_count"] > 0,
        df["click_count"] / df["view_count"],
        0.0,
    )
    df["avg_time_per_interaction"] = np.where(
        df["interaction_count"] > 0,
        df["total_time_spent"] / df["interaction_count"],
        0.0,
    )
    df["has_cart_action"]     = (df["add_to_cart_count"] > 0).astype(int)
    df["has_wishlist_action"] = (df["wishlist_count"] > 0).astype(int)
    return df


# ── Public API ─────────────────────────────────────────────────────────────────

def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a cleaned event-level DataFrame into a session-level feature set.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned event-level data (output of data_preprocessing.run_pipeline).

    Returns
    -------
    pd.DataFrame
        One row per session with all engineered features.
    """
    logger.info("Building session-level features from %d event rows", len(df))

    action_counts = _build_action_counts(df)
    session_df    = _aggregate_sessions(df)

    session_df = session_df.merge(action_counts, on="session_id", how="left")

    count_cols = ["view_count", "click_count", "wishlist_count",
                  "add_to_cart_count", "purchase_count", "drop_count"]
    session_df[count_cols] = session_df[count_cols].fillna(0).astype(int)

    session_df = _add_temporal_features(session_df)
    session_df = _add_ratio_features(session_df)

    logger.info("Session-level dataset — %d sessions, %d features",
                len(session_df), session_df.shape[1])
    return session_df


def run_pipeline() -> pd.DataFrame:
    """
    Load cleaned event data, engineer features, and save the session dataset.

    Returns
    -------
    pd.DataFrame
    """
    ensure_dirs()
    df = pd.read_csv(CLEAN_FILE)
    logger.info("Loaded cleaned event data from %s", CLEAN_FILE)

    session_df = build_session_features(df)
    session_df.to_csv(SESSION_FILE, index=False)
    logger.info("Saved session-level dataset to %s", SESSION_FILE)

    # Save a feature engineering summary
    summary = pd.DataFrame([
        {"Feature": "view_count",              "Type": "Behavioural count", "Source": "user_action pivot"},
        {"Feature": "click_count",             "Type": "Behavioural count", "Source": "user_action pivot"},
        {"Feature": "wishlist_count",          "Type": "Behavioural count", "Source": "user_action pivot"},
        {"Feature": "add_to_cart_count",       "Type": "Behavioural count", "Source": "user_action pivot"},
        {"Feature": "total_time_spent",        "Type": "Engagement",        "Source": "sum of time_spent_sec"},
        {"Feature": "avg_time_per_interaction","Type": "Engagement ratio",  "Source": "total_time_spent / interaction_count"},
        {"Feature": "cart_to_view_ratio",      "Type": "Intent ratio",      "Source": "add_to_cart_count / view_count"},
        {"Feature": "click_to_view_ratio",     "Type": "Intent ratio",      "Source": "click_count / view_count"},
        {"Feature": "has_cart_action",         "Type": "Binary intent",     "Source": "add_to_cart_count > 0"},
        {"Feature": "has_wishlist_action",     "Type": "Binary intent",     "Source": "wishlist_count > 0"},
        {"Feature": "hour",                    "Type": "Temporal",          "Source": "timestamp_utc"},
        {"Feature": "day_of_week",             "Type": "Temporal",          "Source": "timestamp_utc"},
        {"Feature": "month",                   "Type": "Temporal",          "Source": "timestamp_utc"},
        {"Feature": "is_weekend",              "Type": "Temporal",          "Source": "timestamp_utc"},
    ])
    summary.to_csv(RESULTS_TABLES / "feature_engineering_summary.csv", index=False)
    logger.info("Saved feature engineering summary")

    return session_df


# ── CLI entry-point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
