"""
evaluate.py
-----------
Load all saved pipelines, run evaluation on the held-out test set,
and write figures + tables to results/.

Outputs
-------
results/tables/model_comparison.csv
results/figures/confusion_matrix_<model>.png
results/figures/roc_curves.png
results/figures/feature_importance_decision_tree.png

Run directly:
    python src/evaluate.py
"""

from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from utils import (
    DATA_PROCESSED,
    MODELS_DIR,
    RESULTS_FIGS,
    RESULTS_TABLES,
    get_logger,
    ensure_dirs,
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

logger = get_logger(__name__)

MODEL_NAMES = {
    "Decision Tree":     "decision_tree_pipeline.joblib",
    "Naive Bayes":       "naive_bayes_pipeline.joblib",
    "SVM":               "svm_pipeline.joblib",
    "MLP Neural Network":"mlp_pipeline.joblib",
}


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _compute_metrics(name: str, pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )
    roc_auc = round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None

    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   roc_auc,
    }


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _plot_confusion_matrix(name: str, pipeline, X_test, y_test) -> None:
    y_pred = pipeline.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    disp   = ConfusionMatrixDisplay(cm, display_labels=["No Purchase", "Purchase"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    slug = name.lower().replace(" ", "_")
    fig.savefig(RESULTS_FIGS / f"confusion_matrix_{slug}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix: %s", slug)


def _plot_roc_curves(pipelines: dict, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, pipeline in pipelines.items():
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc          = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(RESULTS_FIGS / "roc_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curve comparison")


def _plot_feature_importance(pipeline, save_path) -> None:
    """Plot feature importances for tree-based models."""
    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        num_names = preprocessor.transformers_[0][2]
        cat_encoder = preprocessor.transformers_[1][1]
        cat_names = list(cat_encoder.get_feature_names_out(
            preprocessor.transformers_[1][2]
        ))
        feature_names = num_names + cat_names

        importances = model.feature_importances_
        if len(importances) != len(feature_names):
            logger.warning("Feature name/importance length mismatch; skipping plot.")
            return

        fi_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(20)
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=fi_df, x="importance", y="feature", ax=ax, palette="viridis")
        ax.set_title("Top-20 Feature Importances — Decision Tree")
        ax.set_xlabel("Importance")
        ax.set_ylabel("")
        plt.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved feature importance plot")

        fi_df.to_csv(RESULTS_TABLES / "feature_importance_decision_tree.csv", index=False)
    except Exception as exc:
        logger.warning("Could not plot feature importance: %s", exc)


def _plot_model_comparison(comparison_df: pd.DataFrame) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    plot_df = comparison_df.copy()
    for m in metrics:
        plot_df[m] = pd.to_numeric(plot_df[m], errors="coerce")

    plot_df.set_index("Model")[metrics].plot(kind="bar", figsize=(12, 6))
    plt.title("Model Performance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(RESULTS_FIGS / "model_comparison_bar.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved model comparison bar chart")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_pipeline() -> pd.DataFrame:
    """
    Load test data, evaluate all models, save results.

    Returns
    -------
    pd.DataFrame – model comparison table.
    """
    ensure_dirs()

    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze()
    logger.info("Test set loaded — %d rows", len(y_test))

    pipelines: dict = {}
    for name, filename in MODEL_NAMES.items():
        path = MODELS_DIR / filename
        if not path.exists():
            logger.warning("Model not found, skipping: %s", path)
            continue
        pipelines[name] = joblib.load(path)
        logger.info("Loaded %s", name)

    if not pipelines:
        raise FileNotFoundError("No trained models found. Run train.py first.")

    # Metrics table
    rows = [_compute_metrics(name, pipeline, X_test, y_test)
            for name, pipeline in pipelines.items()]
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(RESULTS_TABLES / "model_comparison.csv", index=False)
    logger.info("Saved model comparison table")

    # Per-model classification reports
    reports = {}
    for name, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_test)
        reports[name] = classification_report(
            y_test, y_pred,
            target_names=["No Purchase", "Purchase"],
            zero_division=0,
        )
        logger.info("\n%s\n%s", name, reports[name])

    # Plots
    _plot_roc_curves(pipelines, X_test, y_test)
    _plot_model_comparison(comparison_df)
    for name, pipeline in pipelines.items():
        _plot_confusion_matrix(name, pipeline, X_test, y_test)

    # Feature importance for Decision Tree
    if "Decision Tree" in pipelines:
        _plot_feature_importance(
            pipelines["Decision Tree"],
            RESULTS_FIGS / "feature_importance_decision_tree.png",
        )

    # Best model summary
    best_row = comparison_df.loc[comparison_df["ROC-AUC"].idxmax()]
    logger.info(
        "Best model by ROC-AUC: %s (%.4f)",
        best_row["Model"], best_row["ROC-AUC"],
    )

    return comparison_df


# ── CLI entry-point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
