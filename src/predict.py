"""
predict.py
----------
Reusable prediction script.

Loads a saved pipeline, reads unseen session-level data from a CSV,
and writes predictions + probabilities to results/predictions/.

Usage
-----
# Default — uses MLP (best model) and prompts for input CSV:
    python src/predict.py --input data/processed/X_test.csv

# Specify a different model:
    python src/predict.py --input new_sessions.csv --model decision_tree

# Save to a custom path:
    python src/predict.py --input new_sessions.csv --output my_preds.csv

Output CSV format
-----------------
session_index, prediction, probability
0, 1, 0.87
1, 0, 0.12
…
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from utils import (
    MODELS_DIR,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    RESULTS_PREDS,
    get_logger,
    ensure_dirs,
)

logger = get_logger(__name__)

MODEL_MAP = {
    "decision_tree": "decision_tree_pipeline.joblib",
    "naive_bayes":   "naive_bayes_pipeline.joblib",
    "svm":           "svm_pipeline.joblib",
    "mlp":           "mlp_pipeline.joblib",
}

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ── Core prediction logic ──────────────────────────────────────────────────────

def load_pipeline(model_name: str = "mlp"):
    """
    Load a saved sklearn Pipeline from the models directory.

    Parameters
    ----------
    model_name : str
        One of: decision_tree, naive_bayes, svm, mlp.

    Returns
    -------
    sklearn Pipeline
    """
    if model_name not in MODEL_MAP:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_MAP.keys())}"
        )
    path = MODELS_DIR / MODEL_MAP[model_name]
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Run 'python src/train.py' first."
        )
    pipeline = joblib.load(path)
    logger.info("Loaded pipeline: %s (%s)", model_name, path)
    return pipeline


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check that the input DataFrame contains the required feature columns
    and select only those needed by the pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data (may contain extra or identifier columns).

    Returns
    -------
    pd.DataFrame – with only pipeline input features.
    """
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required feature columns:\n{missing}\n\n"
            f"Expected columns: {ALL_FEATURES}"
        )
    return df[ALL_FEATURES]


def generate_predictions(
    pipeline,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the pipeline and return a DataFrame of predictions and probabilities.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    X : pd.DataFrame – feature matrix (validated).

    Returns
    -------
    pd.DataFrame with columns:
        - session_index : int
        - prediction    : int (0 or 1)
        - probability   : float (probability of class 1)
    """
    logger.info("Generating predictions for %d sessions …", len(X))

    predictions = pipeline.predict(X)

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)[:, 1]
    else:
        # SVM without predict_proba fallback (shouldn't happen with probability=True)
        logger.warning("Pipeline does not support predict_proba; probability set to NaN.")
        probabilities = [None] * len(predictions)

    output = pd.DataFrame({
        "session_index": range(len(predictions)),
        "prediction":    predictions,
        "probability":   [round(p, 4) if p is not None else None for p in probabilities],
    })

    conversion_rate = predictions.mean() * 100
    logger.info(
        "Predictions generated — %d sessions | predicted conversion rate: %.2f%%",
        len(output), conversion_rate,
    )
    return output


def predict_from_csv(
    input_path: str,
    model_name: str = "mlp",
    output_path: str | None = None,
    id_column: str | None = None,
) -> pd.DataFrame:
    """
    End-to-end prediction from a CSV file.

    Parameters
    ----------
    input_path : str
        Path to a CSV file containing session-level features.
    model_name : str
        Which model pipeline to use (default: 'mlp').
    output_path : str or None
        Where to save the prediction CSV.
        Defaults to results/predictions/predictions_<model_name>.csv.
    id_column : str or None
        Optional column name (e.g. 'session_id') to include in output.

    Returns
    -------
    pd.DataFrame – predictions DataFrame.
    """
    ensure_dirs()

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_df = pd.read_csv(input_path)
    logger.info("Loaded input — %d rows from %s", len(raw_df), input_path)

    X = validate_input(raw_df)

    pipeline = load_pipeline(model_name)

    result_df = generate_predictions(pipeline, X)

    # Optionally attach an ID column for traceability
    if id_column and id_column in raw_df.columns:
        result_df.insert(0, id_column, raw_df[id_column].values)

    # Save output
    out = output_path or RESULTS_PREDS / f"predictions_{model_name}.csv"
    result_df.to_csv(out, index=False)
    logger.info("Predictions saved to %s", out)

    return result_df


# ── CLI entry-point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate purchase-conversion predictions from session-level data."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV with session-level features.",
    )
    parser.add_argument(
        "--model", default="mlp",
        choices=list(MODEL_MAP.keys()),
        help="Which trained pipeline to use (default: mlp).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save the prediction CSV (optional).",
    )
    parser.add_argument(
        "--id-column", default=None,
        help="Column name to include in output as a row identifier (e.g. session_id).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = predict_from_csv(
        input_path  = args.input,
        model_name  = args.model,
        output_path = args.output,
        id_column   = args.id_column,
    )
    print(df.head(10).to_string(index=False))
