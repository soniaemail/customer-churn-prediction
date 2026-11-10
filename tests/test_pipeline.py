"""Unit tests for preprocessing, evaluation and monitoring.

They run on a small synthetic dataset so that CI needs no network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import RAW_FEATURES, TARGET
from src.evaluate import compute_metrics, find_best_threshold
from src.explain import prettify_feature_name, split_feature_name
from src.monitoring import build_baseline, compute_drift_report, population_stability_index
from src.preprocessing import FeatureEngineer, build_preprocessor, clean_data, encode_target


@pytest.fixture()
def raw_frame() -> pd.DataFrame:
    """Synthetic data with the schema (and quirks) of the Telco dataset."""
    rng = np.random.default_rng(0)
    n = 300
    internet = rng.choice(["DSL", "Fiber optic", "No"], n)
    addon = lambda: np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n))  # noqa: E731
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(18, 120, n).round(2)
    total = (tenure * monthly).round(2).astype(str)
    total[tenure == 0] = " "  # blank TotalCharges for brand-new customers
    return pd.DataFrame(
        {
            "customerID": [f"ID-{i}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": internet,
            "OnlineSecurity": addon(),
            "OnlineBackup": addon(),
            "DeviceProtection": addon(),
            "TechSupport": addon(),
            "StreamingTV": addon(),
            "StreamingMovies": addon(),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(["Electronic check", "Mailed check"], n),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        }
    )


def test_clean_data_fixes_types(raw_frame: pd.DataFrame) -> None:
    cleaned = clean_data(raw_frame)
    assert "customerID" not in cleaned.columns
    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])
    assert set(cleaned["SeniorCitizen"].unique()) <= {"Yes", "No"}
    assert len(cleaned) == len(raw_frame)


def test_encode_target() -> None:
    encoded, encoder = encode_target(pd.Series(["No", "Yes", "No"]))
    assert encoded.tolist() == [0, 1, 0]
    assert list(encoder.classes_) == ["No", "Yes"]


def test_feature_engineering(raw_frame: pd.DataFrame) -> None:
    engineered = FeatureEngineer().fit_transform(clean_data(raw_frame)[RAW_FEATURES])
    assert {"AvgMonthlySpend", "ChargeDelta", "NumServices", "TenureGroup"} <= set(engineered.columns)
    assert engineered["TotalCharges"].notna().all()
    assert engineered["NumServices"].between(0, 8).all()
    assert np.isfinite(engineered["AvgMonthlySpend"]).all()  # no division by zero at tenure 0


def test_feature_engineering_rejects_missing_columns(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Missing input columns"):
        FeatureEngineer().fit_transform(clean_data(raw_frame).drop(columns=["tenure"]))


def test_preprocessor_output(raw_frame: pd.DataFrame) -> None:
    cleaned = clean_data(raw_frame)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(cleaned[RAW_FEATURES])
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == len(cleaned)
    assert not transformed.isna().any().any()
    assert abs(transformed["tenure"].mean()) < 1e-6  # standardised
    assert "Contract_Two year" in transformed.columns  # one-hot encoded


def test_preprocessor_handles_unseen_category(raw_frame: pd.DataFrame) -> None:
    cleaned = clean_data(raw_frame)
    preprocessor = build_preprocessor().fit(cleaned[RAW_FEATURES])
    row = cleaned[RAW_FEATURES].head(1).copy()
    row["PaymentMethod"] = "Crypto wallet"
    transformed = preprocessor.transform(row)
    assert transformed.filter(like="PaymentMethod_").to_numpy().sum() == 0


def test_metrics_and_threshold() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.35, 0.4, 0.8, 0.3, 0.45, 0.05])
    metrics = compute_metrics(y_true, y_proba)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1 / 3)
    threshold = find_best_threshold(y_true, y_proba)
    assert compute_metrics(y_true, y_proba, threshold)["f1"] == pytest.approx(1.0)


def test_feature_names() -> None:
    assert split_feature_name("Contract_Two year") == ("Contract", "Two year")
    assert split_feature_name("tenure") == ("tenure", None)
    assert prettify_feature_name("Contract_Two year") == "Contract = Two year"


def test_psi_and_drift_report(raw_frame: pd.DataFrame) -> None:
    assert population_stability_index(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == pytest.approx(0)
    assert population_stability_index(np.array([0.9, 0.1]), np.array([0.1, 0.9])) > 0.25

    cleaned = clean_data(raw_frame)
    scores = np.random.default_rng(1).uniform(0, 1, len(cleaned))
    baseline = build_baseline(cleaned[RAW_FEATURES], scores)

    same = cleaned[RAW_FEATURES].assign(churn_probability=scores)
    assert (compute_drift_report(same, baseline)["psi"] < 0.1).all()

    shifted = same.assign(tenure=same["tenure"] * 0 + 70, Contract="Two year")
    report = compute_drift_report(shifted, baseline).set_index("variable")
    assert report.loc["tenure", "status"] == "Significant drift"
    assert report.loc["Contract", "status"] == "Significant drift"
    assert TARGET not in report.index
