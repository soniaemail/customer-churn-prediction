"""SHAP-based explainability.

Global view  : which features drive churn across the customer base?
Local view   : why is *this* customer predicted at risk?

SHAP values are expressed in the model's raw output space (log-odds for
linear and boosted models): a positive value pushes the prediction towards
churn, a negative value towards retention.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend: works in Streamlit, Docker and CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.config import MULTI_FEATURES, RANDOM_STATE, get_logger  # noqa: E402
from src.preprocessing import ENGINEERED_CATEGORICAL  # noqa: E402

logger = get_logger(__name__)

PRETTY_NAMES: dict[str, str] = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges ($)",
    "TotalCharges": "Total charges ($)",
    "AvgMonthlySpend": "Avg. monthly spend ($)",
    "ChargeDelta": "Bill vs. historical avg ($)",
    "NumServices": "Number of services",
    "SeniorCitizen": "Senior citizen",
    "PaperlessBilling": "Paperless billing",
    "PhoneService": "Phone service",
    "gender": "Gender (1 = male)",
    "InternetService": "Internet service",
    "PaymentMethod": "Payment method",
    "TenureGroup": "Tenure stage",
    "MultipleLines": "Multiple lines",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
}

# Business reading of the main churn drivers, shown in the app
BUSINESS_INSIGHTS: dict[str, dict[str, str]] = {
    "tenure": {
        "title": "Tenure",
        "insight": "Churn is concentrated in the first 12 months. Once customers pass "
        "the first year, the risk drops sharply.",
        "action": "Invest in onboarding: welcome calls, a 90-day check-in and early-life "
        "loyalty perks.",
    },
    "Contract": {
        "title": "Contract type",
        "insight": "Month-to-month customers can leave at any time and churn far more "
        "than customers on one- or two-year contracts.",
        "action": "Offer a discount or a free add-on in exchange for a 12-month commitment.",
    },
    "MonthlyCharges": {
        "title": "Monthly charges",
        "insight": "High monthly bills increase the risk, especially for fiber-optic "
        "customers who do not perceive enough value.",
        "action": "Proactively propose a right-sized plan or a bundle to price-sensitive "
        "customers.",
    },
    "TotalCharges": {
        "title": "Total charges",
        "insight": "A high cumulative spend reflects a long, valuable relationship: these "
        "customers churn less, and losing one costs more.",
        "action": "Protect high-lifetime-value customers with a dedicated retention desk.",
    },
    "InternetService": {
        "title": "Internet service",
        "insight": "Fiber-optic subscribers churn more than DSL subscribers: higher price "
        "and stronger competition on this segment.",
        "action": "Audit fiber quality of service and benchmark its pricing.",
    },
    "PaymentMethod": {
        "title": "Payment method",
        "insight": "Customers paying by electronic check churn the most; automatic "
        "payments are associated with loyalty.",
        "action": "Incentivise the switch to automatic payment (e.g. a one-off credit).",
    },
    "TechSupport": {
        "title": "Tech support & security",
        "insight": "Customers without tech support or online security leave more often: "
        "these add-ons make the service stickier.",
        "action": "Bundle a free trial of tech support for at-risk customers.",
    },
}


# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #
def split_feature_name(feature: str) -> tuple[str, str | None]:
    """Split a transformed column name into ``(raw feature, category)``.

    >>> split_feature_name("Contract_Two year")
    ('Contract', 'Two year')
    >>> split_feature_name("tenure")
    ('tenure', None)
    """
    for column in MULTI_FEATURES + ENGINEERED_CATEGORICAL:
        prefix = f"{column}_"
        if feature.startswith(prefix):
            return column, feature[len(prefix) :]
    return feature, None


def prettify_feature_name(feature: str) -> str:
    """Human-readable label for a transformed feature name."""
    base, category = split_feature_name(feature)
    if category is not None:
        return f"{PRETTY_NAMES.get(base, base)} = {category}"
    return PRETTY_NAMES.get(base, base)


# --------------------------------------------------------------------------- #
# SHAP computation
# --------------------------------------------------------------------------- #
def build_explainer(model: Any, background: pd.DataFrame) -> shap.Explainer:
    """Create the most efficient SHAP explainer for ``model``.

    ``shap.Explainer`` dispatches to ``LinearExplainer`` / ``TreeExplainer``
    when it recognises the model. If that fails (version incompatibilities
    do happen between SHAP and boosting libraries) we fall back to the
    model-agnostic permutation explainer on ``predict_proba``.

    Args:
        model: Fitted classifier.
        background: Sample of *transformed* training rows.
    """
    try:
        explainer = shap.Explainer(model, background)
        explainer(background.iloc[:2])  # smoke test
        logger.info("SHAP explainer: %s", type(explainer).__name__)
        return explainer
    except Exception as exc:  # noqa: BLE001 - any failure triggers the fallback
        logger.warning("Native SHAP explainer failed (%s); using permutation fallback", exc)

        def predict_churn(data: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(data, columns=background.columns)
            return model.predict_proba(frame)[:, 1]

        masker = shap.maskers.Independent(background, max_samples=100)
        return shap.Explainer(predict_churn, masker, seed=RANDOM_STATE)


def compute_shap_values(
    explainer: shap.Explainer,
    X: pd.DataFrame,
    display_data: pd.DataFrame | None = None,
) -> shap.Explanation:
    """Compute SHAP values for the churn class.

    Args:
        explainer: Explainer from :func:`build_explainer`.
        X: Transformed feature rows.
        display_data: Optional human-readable version of ``X`` (un-scaled
            numeric values) shown next to the bars of waterfall plots.

    Returns:
        A 2-D ``shap.Explanation`` (rows x features) with pretty feature names.
    """
    explanation = explainer(X)
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)

    # Some explainers (e.g. RandomForest) return one slice per class
    if values.ndim == 3:
        values = values[:, :, 1]
        base_values = base_values[:, 1] if base_values.ndim == 2 else base_values
    if base_values.ndim == 0:
        base_values = np.repeat(base_values, len(X))

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=X.to_numpy(),
        display_data=None if display_data is None else display_data.to_numpy(),
        feature_names=[prettify_feature_name(c) for c in X.columns],
    )


def global_importance(explanation: shap.Explanation, raw_columns: list[str]) -> pd.DataFrame:
    """Global feature importance = mean absolute SHAP value.

    Returns two levels of detail in one frame: each transformed column, plus
    its parent raw feature so that one-hot columns can be aggregated
    (``Contract_*`` -> ``Contract``).
    """
    importance = np.abs(explanation.values).mean(axis=0)
    frame = pd.DataFrame(
        {
            "feature": explanation.feature_names,
            "raw_feature": [split_feature_name(c)[0] for c in raw_columns],
            "importance": importance,
        }
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def grouped_importance(importance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate importance per raw feature (sums one-hot columns)."""
    grouped = (
        importance.groupby("raw_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    grouped["label"] = grouped["raw_feature"].map(lambda c: PRETTY_NAMES.get(c, c))
    return grouped


def top_drivers(explanation_row: shap.Explanation, top_n: int = 5) -> pd.DataFrame:
    """Rank the features that most influenced a single prediction.

    Returns:
        DataFrame with ``feature``, ``value``, ``shap`` and ``direction``
        (``increases`` / ``decreases`` the churn risk).
    """
    values = np.asarray(explanation_row.values)
    shown = (
        explanation_row.display_data
        if explanation_row.display_data is not None
        else explanation_row.data
    )
    order = np.argsort(-np.abs(values))[:top_n]
    return pd.DataFrame(
        {
            "feature": [explanation_row.feature_names[i] for i in order],
            "value": [shown[i] for i in order],
            "shap": values[order],
            "direction": ["increases" if values[i] > 0 else "decreases" for i in order],
        }
    )


# --------------------------------------------------------------------------- #
# Plots (matplotlib figures, ready for ``st.pyplot``)
# --------------------------------------------------------------------------- #
def summary_plot(explanation: shap.Explanation, max_display: int = 15) -> Figure:
    """SHAP beeswarm summary plot: impact and direction of every feature."""
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    figure = plt.gcf()
    figure.set_size_inches(9, 6.5)
    figure.tight_layout()
    return figure


def waterfall_plot(explanation_row: shap.Explanation, max_display: int = 10) -> Figure:
    """SHAP waterfall plot explaining one prediction step by step."""
    plt.figure()
    shap.plots.waterfall(explanation_row, max_display=max_display, show=False)
    figure = plt.gcf()
    figure.set_size_inches(9, 5.5)
    figure.tight_layout()
    return figure
