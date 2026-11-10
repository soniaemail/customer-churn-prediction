"""Shared UI toolkit for the Streamlit pages.

Everything visual that is reused across pages lives here: the design
tokens, the global CSS, KPI cards, the Plotly theme, Lottie loading and the
cached loaders for data and model artefacts.
"""

from __future__ import annotations

import json
from html import escape
from typing import Any

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    HIGH_RISK_MIN,
    LOTTIE_DIR,
    LOW_RISK_MAX,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    ROC_PATH,
    SHAP_PATH,
    get_logger,
)
from src.data_loader import load_raw_data
from src.evaluate import build_comparison_table, load_json
from src.preprocessing import clean_data

logger = get_logger("app")

# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #
# Brand (UI chrome): blue / violet / white
BLUE = "#2563EB"
VIOLET = "#7C3AED"
INDIGO = "#4F46E5"
INK = "#0F172A"
INK_SECONDARY = "#475569"
INK_MUTED = "#94A3B8"
BORDER = "#E2E8F0"
SURFACE = "#FFFFFF"

# Data colours. Blue and violet are too close to tell apart for colour-blind
# readers, so charts pair the brand blue with a contrasting orange (palette
# checked for CVD separation); the order is fixed and never cycled.
RETAINED = "#2A78D6"
CHURNED = "#EB6834"
CHURN_COLORS: dict[str, str] = {"No": RETAINED, "Yes": CHURNED}
CHURN_LABELS: dict[str, str] = {"No": "Retained", "Yes": "Churned"}
SERIES_COLORS: list[str] = ["#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4"]

# Status colours (always shown with an icon + label, never colour alone)
RISK_LEVELS: dict[str, dict[str, str]] = {
    "low": {"label": "Low Risk", "icon": "🟢", "color": "#15803D", "bg": "#DCFCE7"},
    "medium": {"label": "Medium Risk", "icon": "🟠", "color": "#B45309", "bg": "#FEF3C7"},
    "high": {"label": "High Risk", "icon": "🔴", "color": "#B91C1C", "bg": "#FEE2E2"},
}

# Shown on the About page - replace with your own details
AUTHOR: dict[str, str] = {
    "name": "Your Name",
    "role": "Data Scientist · Machine Learning Engineer",
    "github": "https://github.com/your-username",
    "linkedin": "https://www.linkedin.com/in/your-profile",
}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, button, input, select, textarea {{
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #F5F3FF 0%, #EEF2FF 55%, #FFFFFF 100%);
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebarNav"] a {{ border-radius: 10px; margin: 2px 0; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{
    background: linear-gradient(90deg, rgba(37,99,235,.14), rgba(124,58,237,.14));
}}
[data-testid="stSidebarNav"] a[aria-current="page"] span {{ color: {INDIGO}; font-weight: 700; }}

/* ---------- Hero ---------- */
.hero {{
    background: linear-gradient(120deg, {BLUE} 0%, {INDIGO} 45%, {VIOLET} 100%);
    border-radius: 24px; padding: 2.6rem 2.4rem; color: #fff;
    box-shadow: 0 20px 45px -20px rgba(79,70,229,.55);
}}
.hero h1 {{ color: #fff; font-size: 2.5rem; font-weight: 800; line-height: 1.12;
           margin: 0 0 .7rem 0; padding: 0; letter-spacing: -.02em; }}
.hero p {{ color: rgba(255,255,255,.92); font-size: 1.08rem; margin: 0; max-width: 46rem; }}
.hero .eyebrow {{ display: inline-block; background: rgba(255,255,255,.18);
    border: 1px solid rgba(255,255,255,.35); border-radius: 999px; padding: .25rem .8rem;
    font-size: .78rem; font-weight: 600; letter-spacing: .04em; margin-bottom: 1rem; }}

/* ---------- Page header ---------- */
.page-title {{ font-size: 2rem; font-weight: 800; letter-spacing: -.02em; margin: 0;
    background: linear-gradient(90deg, {BLUE}, {VIOLET}); -webkit-background-clip: text;
    background-clip: text; color: transparent; display: inline-block; }}
.page-subtitle {{ color: {INK_SECONDARY}; font-size: 1.02rem; margin: .25rem 0 1.2rem 0; }}

/* ---------- KPI cards ---------- */
.kpi-card {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 18px;
    padding: 1.1rem 1.25rem; height: 100%; position: relative; overflow: hidden;
    box-shadow: 0 8px 24px -16px rgba(15,23,42,.25); transition: transform .15s ease, box-shadow .15s ease;
}}
.kpi-card:hover {{ transform: translateY(-2px); box-shadow: 0 14px 30px -16px rgba(79,70,229,.45); }}
.kpi-card::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 5px;
    background: linear-gradient(180deg, {BLUE}, {VIOLET}); }}
.kpi-label {{ color: {INK_SECONDARY}; font-size: .8rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: .06em; padding-right: 2.2rem; }}
.kpi-value {{ color: {INK}; font-size: 1.9rem; font-weight: 800; line-height: 1.2; margin-top: .2rem;
    letter-spacing: -.02em; }}
.kpi-help {{ color: {INK_MUTED}; font-size: .82rem; margin-top: .15rem; }}
.kpi-icon {{ position: absolute; right: 1rem; top: .9rem; font-size: 1.5rem; opacity: .9; }}

/* ---------- Generic cards ---------- */
.card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 18px;
    padding: 1.2rem 1.3rem; height: 100%; box-shadow: 0 8px 24px -18px rgba(15,23,42,.25); }}
.card h4 {{ margin: 0 0 .35rem 0; padding: 0; font-size: 1.05rem; color: {INK}; }}
.card p {{ color: {INK_SECONDARY}; font-size: .93rem; margin: 0; }}
.card .tag {{ display: inline-block; margin-top: .7rem; font-size: .78rem; font-weight: 600;
    color: {INDIGO}; background: #EEF2FF; border-radius: 999px; padding: .2rem .65rem; }}
.card .emoji {{ font-size: 1.6rem; display: block; margin-bottom: .4rem; }}

/* ---------- Risk result ---------- */
.risk-box {{ border-radius: 20px; padding: 1.4rem 1.5rem; border: 1px solid {BORDER}; }}
.risk-score {{ font-size: 3.2rem; font-weight: 800; letter-spacing: -.03em; line-height: 1; }}
.risk-caption {{ color: {INK_SECONDARY}; font-size: .85rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: .06em; margin-bottom: .35rem; }}
.risk-badge {{ display: inline-block; margin-top: .8rem; border-radius: 999px; padding: .35rem .95rem;
    font-weight: 700; font-size: 1rem; background: #fff; }}

/* ---------- Buttons / form ---------- */
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {{
    background: linear-gradient(90deg, {BLUE}, {VIOLET}); border: 0; color: #fff;
    font-weight: 700; border-radius: 12px; padding: .6rem 1.2rem;
    box-shadow: 0 10px 24px -12px rgba(79,70,229,.7);
}}
[data-testid="stForm"] {{ border-radius: 18px; border: 1px solid {BORDER}; background: #fff; }}

.section-title {{ font-size: 1.15rem; font-weight: 700; color: {INK}; margin: 1.4rem 0 .6rem 0; }}
.footer {{ color: {INK_MUTED}; font-size: .8rem; text-align: center; margin-top: 2.5rem; }}

@media (max-width: 640px) {{
    .hero {{ padding: 1.6rem 1.3rem; }} .hero h1 {{ font-size: 1.8rem; }}
    .kpi-value {{ font-size: 1.5rem; }} .risk-score {{ font-size: 2.5rem; }}
}}
</style>
"""


# --------------------------------------------------------------------------- #
# Layout helpers
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    """Inject the global stylesheet (call once per run, from ``main.py``)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    """Gradient page title with a one-line description."""
    st.markdown(
        f'<div class="page-title">{escape(title)}</div>'
        f'<div class="page-subtitle">{escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    """Small section heading."""
    st.markdown(f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, help_text: str = "", icon: str = "") -> None:
    """Render one KPI card (use inside ``st.columns``)."""
    # Long text values (e.g. a model name) use a smaller size to stay on one line
    size = ' style="font-size:1.3rem;line-height:1.75"' if len(value) > 10 else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{escape(label)}</div>
            <div class="kpi-value"{size}>{escape(value)}</div>
            <div class="kpi-help">{escape(help_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str, emoji: str = "", tag: str = "") -> None:
    """Render a descriptive card (feature, insight, recommendation...)."""
    tag_html = f'<span class="tag">{escape(tag)}</span>' if tag else ""
    emoji_html = f'<span class="emoji">{emoji}</span>' if emoji else ""
    st.markdown(
        f'<div class="card">{emoji_html}<h4>{escape(title)}</h4><p>{escape(body)}</p>{tag_html}</div>',
        unsafe_allow_html=True,
    )


def footer() -> None:
    """Page footer."""
    st.markdown(
        '<div class="footer">Customer Churn Prediction · Built with Streamlit, '
        "scikit-learn, XGBoost, LightGBM &amp; SHAP</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Lottie
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_lottie(name: str) -> dict[str, Any] | None:
    """Load ``assets/lottie/<name>.json``; returns ``None`` if unavailable."""
    path = LOTTIE_DIR / f"{name}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Lottie animation %s unavailable: %s", path.name, exc)
        return None


def show_lottie(name: str, height: int = 260, key: str | None = None, loop: bool = True,
                fallback: str = "✨") -> None:
    """Display a Lottie animation, degrading gracefully to an emoji.

    The app must stay usable if ``streamlit-lottie`` is not installed or an
    animation file is missing, so every failure falls back to ``fallback``.
    """
    animation = load_lottie(name)
    try:
        from streamlit_lottie import st_lottie

        if animation is None:
            raise FileNotFoundError(name)
        st_lottie(animation, height=height, key=key or f"lottie-{name}", loop=loop, quality="high")
    except Exception:  # noqa: BLE001 - purely decorative, never fatal
        st.markdown(
            f'<div style="font-size:{height // 3}px;text-align:center;line-height:{height}px">'
            f"{fallback}</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Plotly theme
# --------------------------------------------------------------------------- #
PLOTLY_CONFIG: dict[str, Any] = {"displayModeBar": False}


def style_figure(fig: go.Figure, title: str | None = None, height: int = 380,
                 legend: bool = True) -> go.Figure:
    """Apply the house style to a Plotly figure (fonts, grid, legend)."""
    fig.update_layout(
        title={"text": title, "font": {"size": 16, "color": INK}, "x": 0, "xanchor": "left"}
        if title else None,
        font={"family": "Inter, Segoe UI, sans-serif", "color": INK_SECONDARY, "size": 13},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin={"l": 10, "r": 10, "t": 56 if title else 20, "b": 10},
        showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1,
                "title": {"text": ""}},
        hoverlabel={"bgcolor": "#fff", "font": {"family": "Inter, sans-serif", "color": INK}},
        bargap=0.25,
    )
    fig.update_layout(legend_traceorder="normal")
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickcolor=BORDER, title_font={"size": 12},
                     automargin=True, title_standoff=12)
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False, title_font={"size": 12},
                     automargin=True, title_standoff=12)
    return fig


def show_chart(fig: go.Figure, key: str | None = None) -> None:
    """Render a Plotly figure with the shared config inside a bordered card."""
    with st.container(border=True):
        st.plotly_chart(fig, config=PLOTLY_CONFIG, key=key, theme=None)


# --------------------------------------------------------------------------- #
# Risk helpers
# --------------------------------------------------------------------------- #
def risk_level(probability: float) -> dict[str, str]:
    """Map a churn probability to its risk tier (label, icon, colours)."""
    if probability >= HIGH_RISK_MIN:
        return RISK_LEVELS["high"]
    if probability >= LOW_RISK_MAX:
        return RISK_LEVELS["medium"]
    return RISK_LEVELS["low"]


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading customer data…")
def get_data() -> pd.DataFrame:
    """Cleaned dataset (target kept as Yes/No for display)."""
    return clean_data(load_raw_data())


@st.cache_resource(show_spinner="Loading model…")
def get_model_bundle() -> dict[str, Any]:
    """Load the model and the preprocessor saved by the training pipeline."""
    return {
        "model": joblib.load(MODEL_PATH),
        "preprocessor": joblib.load(PREPROCESSOR_PATH),
    }


@st.cache_data(show_spinner=False)
def get_metadata() -> dict[str, Any]:
    """Training metadata (best model, parameters, threshold...)."""
    return load_json(METADATA_PATH)


@st.cache_data(show_spinner=False)
def get_comparison() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Comparison table, raw per-model results and ROC curves."""
    results = load_json(METRICS_PATH)
    return build_comparison_table(results), results, load_json(ROC_PATH)


@st.cache_resource(show_spinner="Loading SHAP values…")
def get_shap_bundle() -> dict[str, Any] | None:
    """Pre-computed SHAP artefacts, or ``None`` if they could not be built."""
    if not SHAP_PATH.exists():
        return None
    try:
        return joblib.load(SHAP_PATH)
    except Exception:  # noqa: BLE001
        logger.exception("Could not load SHAP artefacts")
        return None


@st.cache_resource(show_spinner="Preparing the explainer…")
def get_explainer() -> Any | None:
    """SHAP explainer for live predictions, rebuilt from the saved background."""
    from src.explain import build_explainer

    bundle = get_shap_bundle()
    if bundle is None:
        return None
    try:
        return build_explainer(get_model_bundle()["model"], bundle["background"])
    except Exception:  # noqa: BLE001
        logger.exception("Could not build the SHAP explainer")
        return None


def ensure_artifacts() -> bool:
    """Make sure trained artefacts exist and load; train on first launch.

    This lets a fresh clone (or a Streamlit Cloud deployment whose library
    versions differ from the ones that pickled the model) start with a single
    command: the app trains the model itself when needed.

    Returns:
        ``True`` when the app is ready, ``False`` if training failed.
    """
    required = (MODEL_PATH, PREPROCESSOR_PATH, METRICS_PATH, ROC_PATH, METADATA_PATH)
    ready = all(path.exists() for path in required)
    if ready:
        try:
            import sklearn

            # A pickle is only guaranteed to work with the library version that wrote it
            trained_with = load_json(METADATA_PATH).get("sklearn_version")
            if trained_with != sklearn.__version__:
                raise RuntimeError(
                    f"model trained with scikit-learn {trained_with}, running {sklearn.__version__}"
                )
            get_model_bundle()
            return True
        except Exception:  # noqa: BLE001 - incompatible pickle -> retrain
            logger.exception("Saved model could not be loaded; retraining")
            get_model_bundle.clear()

    from src.train import run_training

    with st.spinner("First launch: training the models (about one minute)…"):
        try:
            run_training(n_iter=10)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Automatic training failed")
            st.error(
                "The model could not be trained automatically. Run `python -m src.train` "
                f"in a terminal to see the full error.\n\nDetails: {exc}"
            )
            return False
    st.cache_data.clear()
    st.cache_resource.clear()
    return True
