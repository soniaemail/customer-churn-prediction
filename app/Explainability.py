"""🔍 Explainability - what drives churn, globally and customer by customer."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import shap
import streamlit as st

from app import ui
from src.explain import (
    BUSINESS_INSIGHTS,
    global_importance,
    grouped_importance,
    summary_plot,
    top_drivers,
    waterfall_plot,
)

ui.page_header("Explainable AI", "Open the black box: which factors push customers to leave, and why.")

bundle = ui.get_shap_bundle()
if bundle is None:
    st.warning("SHAP artefacts are missing. Run `python -m src.train` to generate them.")
    st.stop()

explanation = shap.Explanation(
    values=bundle["values"],
    base_values=bundle["base_values"],
    data=bundle["data"],
    display_data=bundle["display_data"],
    feature_names=bundle["feature_names"],
)
importance = global_importance(explanation, bundle["columns"])
grouped = grouped_importance(importance)
metadata = ui.get_metadata()

st.caption(
    f"SHAP values computed for **{metadata['best_model']}** on {len(explanation):,} customers of "
    "the test set. A SHAP value measures how much a feature moves one prediction away from "
    "the average prediction."
)

tab_global, tab_local, tab_business = st.tabs(
    ["🌍 Global drivers", "👤 Single customer", "💼 Business insights"]
)

# ----------------------------------------------------------------------------
# Global
# ----------------------------------------------------------------------------
with tab_global:
    left, right = st.columns([1, 1.25])
    with left:
        top = grouped.head(10).iloc[::-1]
        bar = go.Figure(
            go.Bar(
                x=top["importance"], y=top["label"], orientation="h",
                marker={"color": ui.RETAINED, "cornerradius": 4},
                hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.3f}<extra></extra>",
            )
        )
        bar.update_xaxes(title_text="Mean |SHAP value|", showgrid=True, gridcolor="#EEF2F7")
        bar.update_yaxes(showgrid=False, ticksuffix="  ")
        ui.show_chart(ui.style_figure(bar, "Feature importance — top 10", height=470, legend=False),
                      key="importance")
    with right:
        with st.container(border=True):
            st.markdown("**SHAP summary plot**")
            st.pyplot(summary_plot(explanation, max_display=14), clear_figure=True)
    st.info(
        "**How to read the summary plot** — each dot is a customer. Its horizontal position is "
        "the impact on the churn score (right = more risk); its colour is the feature value "
        "(red = high, blue = low). Example: red dots for *Tenure* sit on the left, so a long "
        "tenure lowers the risk.",
        icon="📖",
    )
    st.caption(
        "Importance of categorical features is summed over their categories. Correlated "
        "features (monthly charges, internet service, number of services) share the credit "
        "between them, so read them as a group rather than one by one."
    )
    with st.expander("Detailed importance per encoded feature"):
        st.dataframe(
            importance[["feature", "importance"]].rename(
                columns={"feature": "Feature", "importance": "Mean |SHAP|"}
            ),
            hide_index=True,
            column_config={"Mean |SHAP|": st.column_config.ProgressColumn(
                format="%.3f", min_value=0, max_value=float(importance["importance"].max()))},
        )

# ----------------------------------------------------------------------------
# Local
# ----------------------------------------------------------------------------
with tab_local:
    probabilities = np.asarray(bundle["probabilities"])
    order = np.argsort(-probabilities)  # customers sorted by decreasing risk
    pick_col, info_col = st.columns([1, 2], vertical_alignment="bottom")
    rank = pick_col.number_input(
        "Customer rank (1 = highest risk)", min_value=1, max_value=len(order), value=1, step=1
    )
    index = int(order[rank - 1])
    probability = float(probabilities[index])
    level = ui.risk_level(probability)
    info_col.markdown(
        f"**Churn probability: {probability:.0%}** &nbsp; {level['icon']} {level['label']}"
    )

    left, right = st.columns([1.5, 1])
    with left:
        with st.container(border=True):
            st.pyplot(waterfall_plot(explanation[index]), clear_figure=True)
    with right:
        st.markdown("**Why this customer?**")
        for driver in top_drivers(explanation[index], top_n=5).itertuples():
            arrow = "🔺" if driver.direction == "increases" else "🔻"
            st.markdown(f"{arrow} **{driver.feature}** — {driver.direction} the risk")
        st.markdown("**Customer profile**")
        profile = bundle["raw_sample"].iloc[index]
        st.dataframe(
            profile[["tenure", "Contract", "MonthlyCharges", "TotalCharges", "InternetService",
                     "PaymentMethod", "TechSupport"]].astype(str).rename("Value"),
            height=282,
        )

# ----------------------------------------------------------------------------
# Business
# ----------------------------------------------------------------------------
with tab_business:
    st.markdown("From model signals to retention levers — the top factors, in business terms.")
    ranking = {feature: position + 1 for position, feature in enumerate(grouped["raw_feature"])}
    keys = ["tenure", "Contract", "MonthlyCharges", "TotalCharges", "InternetService", "PaymentMethod"]
    for row_start in range(0, len(keys), 3):
        for column, key in zip(st.columns(3), keys[row_start : row_start + 3], strict=False):
            insight = BUSINESS_INSIGHTS[key]
            with column:
                ui.info_card(
                    insight["title"],
                    f"{insight['insight']} ➜ {insight['action']}",
                    tag=f"Importance rank #{ranking.get(key, '–')}",
                )
        st.write("")
