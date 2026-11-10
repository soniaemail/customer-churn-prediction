"""🏠 Home - landing page with the value proposition and headline numbers."""

from __future__ import annotations

import streamlit as st

from app import ui

data = ui.get_data()
metadata = ui.get_metadata()
metrics = metadata["final_metrics"]
threshold_metrics = metadata["threshold_metrics"]
churn_rate = (data["Churn"] == "Yes").mean()

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
hero, animation = st.columns([1.7, 1], vertical_alignment="center")
with hero:
    st.markdown(
        """
        <div class="hero">
            <span class="eyebrow">MACHINE LEARNING · CUSTOMER RETENTION</span>
            <h1>Know which customers will leave — before they do.</h1>
            <p>An end-to-end churn intelligence platform: it scores every customer,
            explains the reasons behind each prediction and points retention teams
            to the accounts worth saving.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with animation:
    ui.show_lottie("ai", height=300, fallback="🧠")

# ----------------------------------------------------------------------------
# Headline KPIs
# ----------------------------------------------------------------------------
st.write("")
col1, col2, col3, col4 = st.columns(4)
with col1:
    ui.kpi_card("Customers analysed", f"{len(data):,}", "Telco Customer Churn dataset", "👥")
with col2:
    ui.kpi_card("Observed churn rate", f"{churn_rate:.1%}", "About 1 customer in 4 leaves", "📉")
with col3:
    ui.kpi_card("Model ROC AUC", f"{metrics['roc_auc']:.3f}", f"{metadata['best_model']} · test set", "🎯")
with col4:
    ui.kpi_card(
        "Churners detected",
        f"{threshold_metrics['recall']:.0%}",
        f"Recall at the {metadata['decision_threshold']:.2f} decision threshold",
        "🛟",
    )

# ----------------------------------------------------------------------------
# What the platform does
# ----------------------------------------------------------------------------
ui.section("What you can do here")
features = [
    ("📈", "Dashboard", "Explore the customer base: churn by contract, tenure and payment "
     "method, with interactive filters.", "Dashboard.py", "Explore the data"),
    ("🤖", "Prediction", "Score a customer in real time and get a risk level, a gauge and the "
     "reasons behind the score.", "Prediction.py", "Predict churn"),
    ("📊", "Model Comparison", "Five algorithms benchmarked on accuracy, precision, recall, F1 "
     "and ROC AUC.", "Model_Comparison.py", "Compare models"),
    ("🔍", "Explainability", "SHAP values reveal the drivers of churn and turn them into "
     "retention actions.", "Explainability.py", "Understand why"),
]
for column, (emoji, title, body, page, cta) in zip(st.columns(4), features, strict=False):
    with column:
        ui.info_card(title, body, emoji=emoji)
        st.page_link(f"app/{page}", label=f"{cta} →")

# ----------------------------------------------------------------------------
# How it works
# ----------------------------------------------------------------------------
ui.section("How it works")
steps = [
    ("1 · Data", "7,043 customers and 20 attributes: profile, subscribed services, contract "
     "and billing."),
    ("2 · Features", "Cleaning, business features (services count, bill evolution, tenure "
     "stage), scaling and encoding in one scikit-learn pipeline."),
    ("3 · Models", "Logistic Regression, Random Forest, Gradient Boosting, XGBoost and "
     "LightGBM compared with 5-fold cross-validation; the winner is tuned."),
    ("4 · Decisions", "A probability, a risk tier and a SHAP explanation for each customer, "
     "plus monitoring of live predictions."),
]
for column, (title, body) in zip(st.columns(4), steps, strict=False):
    with column:
        ui.info_card(title, body)
