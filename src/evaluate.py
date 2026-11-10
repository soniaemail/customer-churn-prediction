"""Model evaluation: metrics, ROC curves and comparison tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import METRICS_PATH, ROC_PATH, get_logger

logger = get_logger(__name__)

METRIC_COLUMNS: list[str] = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
    "roc_auc": "ROC AUC",
}


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute the five classification metrics used across the project.

    Args:
        y_true: Ground-truth labels (0/1).
        y_proba: Predicted probability of the positive class (churn).
        threshold: Decision threshold used to derive hard predictions.

    Returns:
        Dict with accuracy, precision, recall, f1 and roc_auc.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def compute_confusion(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> list[list[int]]:
    """Return the 2x2 confusion matrix ``[[TN, FP], [FN, TP]]`` as nested lists."""
    y_pred = (y_proba >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist()


def compute_roc_points(
    y_true: np.ndarray, y_proba: np.ndarray, max_points: int = 200
) -> dict[str, list[float]]:
    """Compute a (down-sampled) ROC curve that is cheap to store as JSON."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    if len(fpr) > max_points:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return {"fpr": np.round(fpr, 5).tolist(), "tpr": np.round(tpr, 5).tolist()}


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find the decision threshold that maximises the F1 score.

    Churn is imbalanced (~27% positives): the default 0.5 cut-off misses about
    half of the churners. Lowering the threshold trades some precision for a
    much better recall, which is usually the right call when a retention
    offer is cheap compared with a lost customer.

    Must be called on *out-of-fold training predictions*, never on the test
    set, so that the reported test metrics stay unbiased.
    """
    thresholds = np.arange(0.10, 0.91, 0.01)
    scores = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
    return float(np.round(thresholds[int(np.argmax(scores))], 2))


def build_comparison_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Turn the per-model results into a tidy comparison table.

    Args:
        results: ``{model_name: {"test": {...}, "cv_roc_auc": float, ...}}``.

    Returns:
        DataFrame indexed by model, sorted by cross-validated ROC AUC.
    """
    rows = []
    for name, result in results.items():
        row: dict[str, Any] = {"model": name}
        row.update({metric: result["test"][metric] for metric in METRIC_COLUMNS})
        row["cv_roc_auc"] = result["cv_roc_auc"]
        row["cv_roc_auc_std"] = result["cv_roc_auc_std"]
        row["fit_seconds"] = result.get("fit_seconds", float("nan"))
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")
    return table.sort_values("cv_roc_auc", ascending=False)


def select_best_model(table: pd.DataFrame, metric: str = "cv_roc_auc") -> str:
    """Pick the best model name according to ``metric``.

    The default criterion is the *cross-validated* ROC AUC computed on the
    training set only: choosing a model on test-set scores would leak the
    test set into model selection and make the reported metrics optimistic.
    """
    best = str(table[metric].idxmax())
    logger.info("Best model by %s: %s (%.4f)", metric, best, table.loc[best, metric])
    return best


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Write ``payload`` as pretty-printed JSON, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON file produced by :func:`save_json`."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_comparison_table(path: Path = METRICS_PATH) -> pd.DataFrame:
    """Reload the comparison table saved by the training pipeline."""
    return build_comparison_table(load_json(path))


def load_roc_curves(path: Path = ROC_PATH) -> dict[str, dict[str, list[float]]]:
    """Reload the ROC curves saved by the training pipeline."""
    return load_json(path)


if __name__ == "__main__":
    # `python -m src.evaluate` prints the latest comparison table
    pd.set_option("display.float_format", "{:.4f}".format)
    print(load_comparison_table())
