"""
train.py
--------
Train, tune, and persist all classification pipelines.

Each pipeline bundles preprocessing (StandardScaler + OneHotEncoder) with
the model so the saved artefact is fully self-contained and prediction-ready.

Run directly:
    python src/train.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import (
    CATEGORICAL_FEATURES,
    DATA_PROCESSED,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    RESULTS_TABLES,
    TARGET,
    TEST_SIZE,
    get_logger,
    ensure_dirs,
)

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

SESSION_FILE = DATA_PROCESSED / "retail_session_level.csv"
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ── Preprocessor factory ───────────────────────────────────────────────────────

def _make_preprocessor() -> ColumnTransformer:
    """
    Build a fresh ColumnTransformer for each pipeline.

    Using a fresh instance ensures each GridSearchCV fit is independent.
    """
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(),  NUMERICAL_FEATURES),
            ("cat", encoder,           CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# ── Model definitions ──────────────────────────────────────────────────────────

def _model_configs() -> list[dict]:
    """
    Return a list of (name, pipeline, param_grid) configurations.

    All hyperparameter grids were informed by the original notebook experiments.
    """
    return [
        {
            "name": "decision_tree",
            "pipeline": Pipeline([
                ("preprocessor", _make_preprocessor()),
                ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]),
            "params": {
                "model__max_depth":        [3, 5, 7, 10, None],
                "model__min_samples_split":[2, 10, 20],
                "model__min_samples_leaf": [1, 5, 10],
                "model__class_weight":     [None, "balanced"],
            },
        },
        {
            "name": "naive_bayes",
            "pipeline": Pipeline([
                ("preprocessor", _make_preprocessor()),
                ("model", GaussianNB()),
            ]),
            "params": {
                "model__var_smoothing": np.logspace(-12, -6, 7),
            },
        },
        {
            "name": "svm",
            "pipeline": Pipeline([
                ("preprocessor", _make_preprocessor()),
                ("model", SVC(probability=True, random_state=RANDOM_STATE)),
            ]),
            "params": {
                "model__C":          [0.1, 1, 10],
                "model__kernel":     ["rbf", "linear"],
                "model__class_weight":[None, "balanced"],
            },
        },
        {
            "name": "mlp",
            "pipeline": Pipeline([
                ("preprocessor", _make_preprocessor()),
                ("model", MLPClassifier(
                    max_iter=500,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                )),
            ]),
            "params": {
                "model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "model__activation":         ["relu", "tanh"],
                "model__alpha":              [0.0001, 0.001],
            },
        },
    ]


# ── Training helpers ───────────────────────────────────────────────────────────

def _train_one(
    config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> tuple[GridSearchCV, dict]:
    """
    Fit a GridSearchCV for one model config.

    Returns
    -------
    search : GridSearchCV (best estimator is the fitted pipeline)
    summary : dict of training metadata
    """
    logger.info("Training %s …", config["name"])
    t0 = time.time()

    search = GridSearchCV(
        estimator  = config["pipeline"],
        param_grid = config["params"],
        cv         = cv,
        scoring    = "roc_auc",
        n_jobs     = -1,
        refit      = True,
    )
    search.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 1)

    logger.info(
        "%s — best CV ROC-AUC: %.4f | time: %ss | params: %s",
        config["name"],
        search.best_score_,
        elapsed,
        search.best_params_,
    )

    summary = {
        "model_name":        config["name"],
        "best_cv_roc_auc":   round(search.best_score_, 4),
        "training_time_sec": elapsed,
        "best_params":       str(search.best_params_),
    }
    return search, summary


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the session-level dataset and return a stratified train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(SESSION_FILE)
    logger.info("Loaded session data — %d rows", len(df))

    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )
    logger.info(
        "Split — train: %d | test: %d | conversion rate train: %.2f%% | test: %.2f%%",
        len(y_train), len(y_test),
        y_train.mean() * 100, y_test.mean() * 100,
    )

    # Persist splits for downstream reproducibility
    X_train.to_csv(DATA_PROCESSED / "X_train.csv", index=False)
    X_test.to_csv( DATA_PROCESSED / "X_test.csv",  index=False)
    y_train.to_csv(DATA_PROCESSED / "y_train.csv", index=False)
    y_test.to_csv( DATA_PROCESSED / "y_test.csv",  index=False)

    # Persist feature metadata
    joblib.dump(
        {"numerical_features": NUMERICAL_FEATURES,
         "categorical_features": CATEGORICAL_FEATURES,
         "all_features": ALL_FEATURES},
        MODELS_DIR / "feature_metadata.joblib",
    )
    logger.info("Saved train/test splits and feature metadata")
    return X_train, X_test, y_train, y_test


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """
    End-to-end training: load data → tune models → save pipelines.
    """
    ensure_dirs()

    X_train, X_test, y_train, y_test = load_data()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    summaries: list[dict] = []

    for config in _model_configs():
        search, summary = _train_one(config, X_train, y_train, cv)
        best_pipeline   = search.best_estimator_

        save_path = MODELS_DIR / f"{config['name']}_pipeline.joblib"
        joblib.dump(best_pipeline, save_path)
        logger.info("Saved %s → %s", config["name"], save_path)

        summaries.append(summary)

    pd.DataFrame(summaries).to_csv(
        RESULTS_TABLES / "training_summary.csv", index=False
    )
    logger.info("Training complete. Summary saved to results/tables/training_summary.csv")


# ── CLI entry-point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
