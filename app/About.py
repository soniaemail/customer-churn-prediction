"""ℹ️ About Project - context, methodology, architecture and tech stack."""

from __future__ import annotations

import streamlit as st

from app import ui

ui.page_header("About the Project", "The business problem, the method and the engineering behind the app.")
metadata = ui.get_metadata()

left, right = st.columns([1.4, 1])
with left:
    st.markdown(
        """
        #### 🎯 Business context
        Acquiring a new customer costs several times more than keeping an existing one.
        A telecom operator wants to **identify customers likely to leave** so that the
        retention team can act *before* the cancellation — with the right offer, for the
        right customer.

        #### 🧪 Methodology
        1. **Exploration** — data quality checks and churn analysis by segment.
        2. **Preprocessing** — imputation, label / one-hot encoding, scaling and business
           features inside a single `ColumnTransformer` pipeline.
        3. **Modelling** — five algorithms compared with stratified 5-fold cross-validation.
        4. **Optimisation** — `RandomizedSearchCV` on the best model, then a decision
           threshold tuned for the business objective.
        5. **Explainability** — SHAP values, globally and per customer.
        6. **Delivery** — Streamlit app, Docker image, CI pipeline and drift monitoring.
        """
    )
with right:
    ui.info_card(
        "Model card",
        f"{metadata['best_model']} · trained on {metadata['n_train']:,} customers, evaluated on "
        f"{metadata['n_test']:,}. ROC AUC {metadata['final_metrics']['roc_auc']:.3f}, recall "
        f"{metadata['threshold_metrics']['recall']:.0%} at threshold "
        f"{metadata['decision_threshold']:.2f}.",
        emoji="🧠",
        tag=f"scikit-learn {metadata['sklearn_version']}",
    )
    st.write("")
    ui.info_card(
        "Dataset",
        "IBM Telco Customer Churn: 7,043 customers of a fictional telecom company, "
        "20 attributes and a churn label (26.5% positives).",
        emoji="🗂️",
        tag="Public sample dataset",
    )

ui.section("Architecture")
st.code(
    """
data/telco_customer_churn.csv
        │  src/data_loader.py      download + schema validation
        ▼
src/preprocessing.py               cleaning → FeatureEngineer → ColumnTransformer
        ▼
src/train.py                       5 models × 5-fold CV → best model → RandomizedSearchCV
        │                          src/evaluate.py  metrics, ROC, threshold
        │                          src/explain.py   SHAP values
        ▼
models/churn_model.pkl  +  models/preprocessor.pkl  +  reports/*.json
        ▼
main.py → app/*.py                 Streamlit multipage app
        ▼
src/monitoring.py                  prediction log + PSI drift report
    """,
    language="text",
)

ui.section("Tech stack")
stack = [
    ("🐍", "Data", "Python 3.11 · Pandas · NumPy"),
    ("🤖", "Machine Learning", "scikit-learn · XGBoost · LightGBM · SHAP"),
    ("📊", "Visualisation", "Plotly · Matplotlib · Seaborn · Lottie"),
    ("🚀", "Delivery", "Streamlit · Docker · GitHub Actions"),
]
for column, (emoji, title, body) in zip(st.columns(4), stack, strict=False):
    with column:
        ui.info_card(title, body, emoji=emoji)

ui.section("Limits and next steps")
st.markdown(
    """
    - The dataset is a **static snapshot**: no dates, so seasonality and time-to-churn cannot be modelled.
    - Scores are correlations, not causes: the impact of a retention action should be validated with an **A/B test**.
    - Next steps: probability calibration, uplift modelling to target *persuadable* customers,
      and an API endpoint (FastAPI) for CRM integration.
    """
)

ui.section("Author")
author = ui.AUTHOR
links = [f"[GitHub]({author['github']})"]
if author.get("linkedin"):
    links.append(f"[LinkedIn]({author['linkedin']})")
st.markdown(f"**{author['name']}** — {author['role']}  \n" + " · ".join(links))
