"""Lightweight model monitoring.

Two production concerns are covered without any external service:

1. **Prediction logging** - every real-time prediction is appended to
   ``logs/predictions.jsonl`` (inputs, score, model version, timestamp).
2. **Data drift** - the Population Stability Index (PSI) compares the
   distribution of live inputs/scores with the training baseline saved in
   ``reports/baseline_stats.json``.

PSI reading guide: < 0.10 stable, 0.10-0.25 moderate drift, > 0.25 significant
drift (time to investigate and probably retrain).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import BASELINE_PATH, PREDICTION_LOG_PATH, get_logger

logger = get_logger(__name__)

MONITORED_NUMERIC: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]
MONITORED_CATEGORICAL: list[str] = ["Contract", "InternetService", "PaymentMethod"]
SCORE_COLUMN: str = "churn_probability"
PSI_MODERATE: float = 0.10
PSI_SIGNIFICANT: float = 0.25
MIN_SAMPLES_FOR_DRIFT: int = 30
_EPSILON: float = 1e-4  # avoids log(0) for empty bins


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
def build_baseline(X_train: pd.DataFrame, train_scores: np.ndarray) -> dict[str, Any]:
    """Summarise the training distribution of monitored features and scores.

    Numeric features are bucketed into deciles, categorical features keep
    their category frequencies, scores use ten equal-width bins.
    """
    baseline: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "n_samples": int(len(X_train)),
        "numeric": {},
        "categorical": {},
    }
    for column in MONITORED_NUMERIC:
        values = X_train[column].dropna().to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        inner = edges[1:-1]  # outer bins are open-ended
        counts = np.bincount(np.digitize(values, inner), minlength=len(inner) + 1)
        baseline["numeric"][column] = {
            "edges": inner.tolist(),
            "expected": (counts / counts.sum()).tolist(),
        }
    for column in MONITORED_CATEGORICAL:
        baseline["categorical"][column] = (
            X_train[column].value_counts(normalize=True).to_dict()
        )

    score_edges = np.linspace(0.1, 0.9, 9)
    counts = np.bincount(np.digitize(train_scores, score_edges), minlength=10)
    baseline["score"] = {
        "edges": score_edges.tolist(),
        "expected": (counts / counts.sum()).tolist(),
        "mean": float(np.mean(train_scores)),
    }
    return baseline


def save_baseline(baseline: dict[str, Any], path: Path = BASELINE_PATH) -> None:
    """Persist the baseline next to the other training reports."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, Any] | None:
    """Load the baseline; returns ``None`` when the model was never trained."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Prediction log
# --------------------------------------------------------------------------- #
def log_prediction(
    features: dict[str, Any],
    probability: float,
    model_name: str,
    path: Path = PREDICTION_LOG_PATH,
) -> bool:
    """Append one prediction to the JSON-lines log.

    Logging must never break the user-facing prediction, so I/O errors are
    swallowed and reported through the application logger.

    Returns:
        ``True`` when the record was written.
    """
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model_name,
        SCORE_COLUMN: round(float(probability), 6),
        **features,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        return True
    except OSError as exc:
        logger.error("Could not log prediction: %s", exc)
        return False


def load_prediction_log(path: Path = PREDICTION_LOG_PATH) -> pd.DataFrame:
    """Read the prediction log; malformed lines are skipped, not fatal."""
    if not path.exists():
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed log line %d", line_number)
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
def population_stability_index(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI between two discrete distributions defined on the same bins."""
    expected = np.clip(np.asarray(expected, dtype=float), _EPSILON, None)
    actual = np.clip(np.asarray(actual, dtype=float), _EPSILON, None)
    expected, actual = expected / expected.sum(), actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_status(psi: float) -> str:
    """Translate a PSI value into a traffic-light status."""
    if psi < PSI_MODERATE:
        return "Stable"
    if psi < PSI_SIGNIFICANT:
        return "Moderate drift"
    return "Significant drift"


def compute_drift_report(live: pd.DataFrame, baseline: dict[str, Any]) -> pd.DataFrame:
    """Compare live data with the training baseline.

    Args:
        live: Prediction log (see :func:`load_prediction_log`).
        baseline: Output of :func:`build_baseline`.

    Returns:
        One row per monitored variable with its PSI and status. Empty when
        there is no live data yet.
    """
    if live.empty:
        return pd.DataFrame(columns=["variable", "kind", "psi", "status"])

    rows: list[dict[str, Any]] = []

    def _binned_psi(values: pd.Series, spec: dict[str, Any]) -> float:
        data = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
        counts = np.bincount(
            np.digitize(data, np.asarray(spec["edges"])), minlength=len(spec["expected"])
        )
        return population_stability_index(spec["expected"], counts / max(counts.sum(), 1))

    for column, spec in baseline["numeric"].items():
        if column in live:
            rows.append({"variable": column, "kind": "numeric", "psi": _binned_psi(live[column], spec)})

    for column, expected in baseline["categorical"].items():
        if column in live:
            actual = live[column].value_counts(normalize=True)
            categories = sorted(set(expected) | set(actual.index))
            psi = population_stability_index(
                np.array([expected.get(c, 0.0) for c in categories]),
                np.array([actual.get(c, 0.0) for c in categories]),
            )
            rows.append({"variable": column, "kind": "categorical", "psi": psi})

    if SCORE_COLUMN in live:
        rows.append(
            {
                "variable": "Model score",
                "kind": "prediction",
                "psi": _binned_psi(live[SCORE_COLUMN], baseline["score"]),
            }
        )

    report = pd.DataFrame(rows)
    report["status"] = report["psi"].map(drift_status)
    return report.sort_values("psi", ascending=False).reset_index(drop=True)
