"""🩺 Monitoring - health of the model once it serves real predictions."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app import ui
from src.config import LOGS_DIR
from src.monitoring import (
    MIN_SAMPLES_FOR_DRIFT,
    SCORE_COLUMN,
    compute_drift_report,
    load_baseline,
    load_prediction_log,
)

ui.page_header("Model Monitoring", "Is the model still seeing the kind of customers it was trained on?")

metadata = ui.get_metadata()
baseline = load_baseline()
log = load_prediction_log()

k1, k2, k3, k4 = st.columns(4)
with k1:
    ui.kpi_card("Model", metadata["best_model"], f"Trained {metadata['trained_at'][:10]}", "🧠")
with k2:
    ui.kpi_card("Predictions logged", f"{len(log):,}", "Since the log was created", "📝")
with k3:
    mean_score = f"{log[SCORE_COLUMN].mean():.0%}" if len(log) else "–"
    expected = f"Training average: {baseline['score']['mean']:.0%}" if baseline else ""
    ui.kpi_card("Average live score", mean_score, expected, "📊")
with k4:
    high = f"{(log[SCORE_COLUMN] >= metadata['decision_threshold']).mean():.0%}" if len(log) else "–"
    ui.kpi_card("Flagged for retention", high, f"Score ≥ {metadata['decision_threshold']:.2f}", "🚩")

if log.empty:
    st.info("No prediction has been logged yet. Score a few customers on the **Prediction** page "
            "and come back.", icon="ℹ️")
    st.stop()

# ----------------------------------------------------------------------------
# Score distribution
# ----------------------------------------------------------------------------
ui.section("Live score distribution")
hist = go.Figure(
    go.Histogram(
        x=log[SCORE_COLUMN], xbins={"start": 0, "end": 1, "size": 0.1},
        marker={"color": ui.RETAINED, "line": {"color": "#fff", "width": 2}},
        hovertemplate="Score %{x}<br>%{y} predictions<extra></extra>",
    )
)
hist.update_xaxes(title_text="Predicted churn probability", range=[0, 1], tickformat=".0%")
hist.update_yaxes(title_text="Predictions")
ui.show_chart(ui.style_figure(hist, height=300, legend=False), key="scores")

# ----------------------------------------------------------------------------
# Drift
# ----------------------------------------------------------------------------
ui.section("Data drift (Population Stability Index)")
if baseline is None:
    st.warning("Training baseline missing. Run `python -m src.train` to create it.")
else:
    if len(log) < MIN_SAMPLES_FOR_DRIFT:
        st.warning(
            f"Only {len(log)} predictions logged: PSI needs at least {MIN_SAMPLES_FOR_DRIFT} "
            "to be meaningful. Treat the values below as indicative.", icon="⚠️",
        )
    report = compute_drift_report(log, baseline)
    icons = {"Stable": "🟢 Stable", "Moderate drift": "🟠 Moderate drift",
             "Significant drift": "🔴 Significant drift"}
    report["status"] = report["status"].map(icons)
    st.dataframe(
        report.rename(columns={"variable": "Variable", "kind": "Type", "psi": "PSI", "status": "Status"}),
        hide_index=True,
        column_config={"PSI": st.column_config.NumberColumn(format="%.3f")},
    )
    st.caption("PSI < 0.10: stable · 0.10–0.25: moderate drift · > 0.25: significant drift, "
               "investigate and consider retraining.")

# ----------------------------------------------------------------------------
# Logs
# ----------------------------------------------------------------------------
ui.section("Recent predictions")
recent = log.sort_values("timestamp", ascending=False).head(50)
st.dataframe(recent, hide_index=True, height=260)
st.download_button("Download the full prediction log (CSV)", log.to_csv(index=False).encode("utf-8"),
                   file_name="prediction_log.csv", mime="text/csv")

with st.expander("Application log (last 40 lines)"):
    log_file = LOGS_DIR / "app.log"
    try:
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        st.code("\n".join(lines) or "The log is empty.", language="log")
    except OSError:
        st.caption("No application log file found.")
