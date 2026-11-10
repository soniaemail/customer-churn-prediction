"""Entry point of the Streamlit application.

Run with::

    streamlit run main.py

``main.py`` owns everything that is global to the app (page config, CSS,
sidebar, navigation, first-launch training) and then hands over to the page
selected in the sidebar. Each page lives in its own module under ``app/``.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app import ui
from src.config import ASSETS_DIR, get_logger

logger = get_logger("app")
APP_DIR = Path(__file__).parent / "app"

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()

logo = ASSETS_DIR / "images" / "logo.svg"
if logo.exists():
    st.logo(str(logo), size="large")

# Train on first launch (fresh clone / cloud deployment), then build the nav
if not ui.ensure_artifacts():
    st.stop()

pages = {
    "Overview": [
        st.Page(APP_DIR / "Home.py", title="Home", icon="🏠", default=True),
        st.Page(APP_DIR / "Dashboard.py", title="Dashboard", icon="📈"),
    ],
    "Machine Learning": [
        st.Page(APP_DIR / "Prediction.py", title="Prediction", icon="🤖"),
        st.Page(APP_DIR / "Model_Comparison.py", title="Model Comparison", icon="📊"),
        st.Page(APP_DIR / "Explainability.py", title="Explainability", icon="🔍"),
        st.Page(APP_DIR / "Monitoring.py", title="Monitoring", icon="🩺"),
    ],
    "Project": [
        st.Page(APP_DIR / "About.py", title="About Project", icon="ℹ️"),
    ],
}
current_page = st.navigation(pages, expanded=True)

with st.sidebar:
    metadata = ui.get_metadata()
    st.markdown(
        f"""
        <div class="card" style="margin-top:1rem">
            <div class="kpi-label">Model in production</div>
            <div style="font-weight:800;color:{ui.INK};font-size:1.05rem;margin-top:.2rem">
                {metadata["best_model"]}</div>
            <div class="kpi-help">ROC AUC {metadata["final_metrics"]["roc_auc"]:.3f}
                · trained {metadata["trained_at"][:10]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

try:
    current_page.run()
except Exception:  # noqa: BLE001 - last-resort guard: log and show a friendly message
    logger.exception("Unhandled error on page %s", current_page.title)
    st.error(
        "Something went wrong while rendering this page. The error has been written to "
        "`logs/app.log`."
    )

ui.footer()
