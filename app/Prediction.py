"""🤖 Prediction - real-time churn scoring of a single customer."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import ui
from src.config import HIGH_RISK_MIN, LOW_RISK_MAX, RAW_FEATURES
from src.explain import (
    BUSINESS_INSIGHTS,
    compute_shap_values,
    split_feature_name,
    top_drivers,
    waterfall_plot,
)
from src.monitoring import log_prediction
from src.preprocessing import SERVICE_COLUMNS, numeric_display_frame

SENIOR_AGE = 65  # the dataset only knows "senior citizen": derived from the age input
CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["Fiber optic", "DSL", "No"]
PAYMENTS = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]


def build_customer(inputs: dict[str, Any]) -> pd.DataFrame:
    """Turn the form inputs into a one-row dataframe with the training schema.

    Business rules keep the row consistent with the dataset: add-ons require
    an internet subscription, multiple lines require a phone line, and the
    total charges are estimated from tenure x monthly charges.
    """
    row: dict[str, Any] = dict(inputs)
    row["SeniorCitizen"] = "Yes" if row.pop("age") >= SENIOR_AGE else "No"
    row["TotalCharges"] = float(row["tenure"]) * float(row["MonthlyCharges"])
    if row["InternetService"] == "No":
        row.update({column: "No internet service" for column in SERVICE_COLUMNS})
    if row["PhoneService"] == "No":
        row["MultipleLines"] = "No phone service"
    return pd.DataFrame([row])[RAW_FEATURES]


def gauge(probability: float, level: dict[str, str]) -> go.Figure:
    """Modern risk gauge with the three risk bands."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 46, "color": ui.INK}, "valueformat": ".0f"},
            gauge={
                "axis": {"range": [0, 100], "tickvals": [0, LOW_RISK_MAX * 100, HIGH_RISK_MIN * 100, 100],
                         "ticksuffix": "%", "tickcolor": ui.INK_MUTED},
                "bar": {"color": level["color"], "thickness": 0.28},
                "bgcolor": "#fff",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, LOW_RISK_MAX * 100], "color": ui.RISK_LEVELS["low"]["bg"]},
                    {"range": [LOW_RISK_MAX * 100, HIGH_RISK_MIN * 100], "color": ui.RISK_LEVELS["medium"]["bg"]},
                    {"range": [HIGH_RISK_MIN * 100, 100], "color": ui.RISK_LEVELS["high"]["bg"]},
                ],
            },
        )
    )
    fig.update_layout(height=270, margin={"l": 30, "r": 30, "t": 30, "b": 10},
                      paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter, sans-serif"})
    return fig


def recommended_actions(drivers: pd.DataFrame, raw_columns: dict[str, str]) -> list[str]:
    """Retention actions matching the features that increase this customer's risk."""
    actions: list[str] = []
    for feature in drivers.loc[drivers["direction"] == "increases", "feature"]:
        base = raw_columns.get(feature, feature)
        insight = BUSINESS_INSIGHTS.get(base) or (
            BUSINESS_INSIGHTS["TechSupport"] if base in SERVICE_COLUMNS else None
        )
        if insight and insight["action"] not in actions:
            actions.append(insight["action"])
    return actions[:3]


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
header, animation = st.columns([3, 1], vertical_alignment="center")
with header:
    ui.page_header("Churn Prediction", "Describe a customer and get their churn risk in real time.")
with animation:
    ui.show_lottie("robot", height=170, fallback="🤖")

bundle = ui.get_model_bundle()
metadata = ui.get_metadata()

# ----------------------------------------------------------------------------
# Form
# ----------------------------------------------------------------------------
with st.form("prediction-form"):
    st.markdown("**Customer profile**")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", min_value=18, max_value=100, value=42, step=1,
                          help=f"Customers aged {SENIOR_AGE}+ are treated as senior citizens.")
    contract = c2.selectbox("Contract", CONTRACTS)
    payment = c3.selectbox("Payment method", PAYMENTS)

    c4, c5, c6 = st.columns(3)
    internet = c4.selectbox("Internet service", INTERNET)
    tenure = c5.slider("Tenure (months)", min_value=0, max_value=72, value=6)
    monthly = c6.slider("Monthly charges ($)", min_value=18.0, max_value=120.0, value=85.0, step=0.5)

    with st.expander("Advanced profile (optional)"):
        a1, a2, a3, a4 = st.columns(4)
        gender = a1.selectbox("Gender", ["Female", "Male"])
        partner = a2.selectbox("Partner", ["No", "Yes"])
        dependents = a3.selectbox("Dependents", ["No", "Yes"])
        paperless = a4.selectbox("Paperless billing", ["Yes", "No"])
        b1, b2, b3, b4 = st.columns(4)
        phone = b1.selectbox("Phone service", ["Yes", "No"])
        multiple = b2.selectbox("Multiple lines", ["No", "Yes"])
        security = b3.selectbox("Online security", ["No", "Yes"])
        backup = b4.selectbox("Online backup", ["No", "Yes"])
        d1, d2, d3, d4 = st.columns(4)
        protection = d1.selectbox("Device protection", ["No", "Yes"])
        support = d2.selectbox("Tech support", ["No", "Yes"])
        tv = d3.selectbox("Streaming TV", ["No", "Yes"])
        movies = d4.selectbox("Streaming movies", ["No", "Yes"])
        st.caption("Add-on services are ignored when the customer has no internet service.")

    submitted = st.form_submit_button("🔮 Predict Churn", type="primary", width="stretch")

if submitted:
    inputs = {
        "age": age, "Contract": contract, "PaymentMethod": payment, "InternetService": internet,
        "tenure": tenure, "MonthlyCharges": monthly, "gender": gender, "Partner": partner,
        "Dependents": dependents, "PaperlessBilling": paperless, "PhoneService": phone,
        "MultipleLines": multiple, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
        "StreamingMovies": movies,
    }
    try:
        customer = build_customer(inputs)
        transformed = bundle["preprocessor"].transform(customer)
        probability = float(bundle["model"].predict_proba(transformed)[0, 1])
    except Exception as exc:  # noqa: BLE001 - show a clear message instead of a traceback
        ui.logger.exception("Prediction failed")
        st.error(f"The prediction could not be computed: {exc}")
        st.stop()

    log_prediction(customer.iloc[0].to_dict(), probability, metadata["best_model"])
    ui.logger.info("Prediction served: p=%.3f contract=%s tenure=%s", probability, contract, tenure)
    # Keep the result across re-runs (e.g. when a widget elsewhere changes)
    st.session_state["prediction"] = {
        "customer": customer, "transformed": transformed, "probability": probability,
    }

result = st.session_state.get("prediction")
if result is None:
    st.info("Fill in the form and click **Predict Churn** to score a customer.", icon="👆")
    st.stop()

# ----------------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------------
probability = result["probability"]
level = ui.risk_level(probability)
flagged = probability >= metadata["decision_threshold"]

ui.section("Prediction result")
score_col, gauge_col, anim_col = st.columns([1.15, 1.5, 0.8], vertical_alignment="center")
with score_col:
    st.markdown(
        f"""
        <div class="risk-box" style="background:{level['bg']}">
            <div class="risk-caption">Risk Score</div>
            <div class="risk-score" style="color:{level['color']}">{probability:.0%}</div>
            <div class="risk-badge" style="color:{level['color']}">{level['icon']} {level['label']}</div>
            <div class="kpi-help" style="margin-top:.8rem;color:{ui.INK_SECONDARY}">
                Probability that this customer leaves.<br>
                {"⚠️ Above" if flagged else "✓ Below"} the retention threshold
                ({metadata['decision_threshold']:.0%}).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with gauge_col:
    st.plotly_chart(gauge(probability, level), config=ui.PLOTLY_CONFIG, theme=None, key="gauge")
with anim_col:
    ui.show_lottie("success", height=150, loop=False, fallback="✅",
                   key=f"success-{probability:.6f}")
    st.caption("Prediction complete and logged for monitoring.")

# ----------------------------------------------------------------------------
# Explanation
# ----------------------------------------------------------------------------
ui.section("Why this score?")
explainer = ui.get_explainer()
if explainer is None:
    st.warning("Explanations are unavailable (SHAP artefacts missing). Run `python -m src.train`.")
    st.stop()

try:
    transformed = result["transformed"]
    explanation = compute_shap_values(
        explainer, transformed, numeric_display_frame(bundle["preprocessor"], transformed)
    )[0]
    drivers = top_drivers(explanation, top_n=6)
except Exception as exc:  # noqa: BLE001
    ui.logger.exception("SHAP explanation failed")
    st.warning(f"The explanation could not be computed: {exc}")
    st.stop()

raw_by_pretty = {
    pretty: split_feature_name(column)[0]
    for pretty, column in zip(explanation.feature_names, transformed.columns, strict=False)
}

plot_col, text_col = st.columns([1.5, 1])
with plot_col:
    with st.container(border=True):
        st.pyplot(waterfall_plot(explanation), clear_figure=True)
        st.caption(
            "SHAP waterfall: starting from the average customer, red bars push the risk up and "
            "blue bars pull it down (model output in log-odds)."
        )
with text_col:
    st.markdown("**Main factors for this customer**")
    for driver in drivers.itertuples():
        arrow = "🔺" if driver.direction == "increases" else "🔻"
        value = f" ({driver.value:g})" if " = " not in driver.feature else (
            "" if driver.value == 1 else " (not the case)")
        st.markdown(f"{arrow} **{driver.feature}**{value} — {driver.direction} the risk")

    actions = recommended_actions(drivers, raw_by_pretty)
    if actions and probability >= LOW_RISK_MAX:
        st.markdown("**Recommended retention actions**")
        for action in actions:
            st.markdown(f"- {action}")
    elif probability < LOW_RISK_MAX:
        st.success("Low risk: no retention action needed. A good candidate for upsell.", icon="🟢")
