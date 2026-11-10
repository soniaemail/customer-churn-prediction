"""📊 Model Comparison - benchmark of the five candidate algorithms."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app import ui
from src.evaluate import METRIC_COLUMNS, METRIC_LABELS

ui.page_header("Model Comparison", "Five algorithms, one protocol: same split, same features, same metrics.")

table, results, roc_curves = ui.get_comparison()
metadata = ui.get_metadata()
best = metadata["best_model"]
# Fixed colour per model: the colour follows the model, not its rank
colors = {name: ui.SERIES_COLORS[i % len(ui.SERIES_COLORS)] for i, name in enumerate(results)}

# ----------------------------------------------------------------------------
# Winner
# ----------------------------------------------------------------------------
final = metadata["final_metrics"]
k1, k2, k3, k4 = st.columns(4)
with k1:
    ui.kpi_card("Best model", best, "Selected automatically", "🏆")
with k2:
    ui.kpi_card("Cross-validated AUC", f"{metadata['final_cv_roc_auc']:.3f}", "5-fold, training set", "🔁")
with k3:
    ui.kpi_card("Test ROC AUC", f"{final['roc_auc']:.3f}", f"{metadata['n_test']:,} unseen customers", "🎯")
with k4:
    ui.kpi_card("Test accuracy", f"{final['accuracy']:.1%}", "At the default 0.50 threshold", "✅")

st.caption(
    "The winner is picked on **cross-validated ROC AUC computed on the training set only**; "
    "the test set is used once, for the final report, so the figures below are unbiased."
)

# ----------------------------------------------------------------------------
# Table
# ----------------------------------------------------------------------------
ui.section("Comparison table (test set)")
display = table[METRIC_COLUMNS + ["cv_roc_auc", "fit_seconds"]].rename(
    columns={**METRIC_LABELS, "cv_roc_auc": "CV ROC AUC", "fit_seconds": "Fit time (s)"}
)
display.index = [f"🏆 {name}" if name == best else name for name in display.index]
display.index.name = "Model"
metric_names = list(METRIC_LABELS.values()) + ["CV ROC AUC"]
styled = (
    display.style.format("{:.3f}", subset=metric_names)
    .format("{:.2f}", subset=["Fit time (s)"])
    .highlight_max(subset=metric_names, props="font-weight:700; background-color:#EEF2FF; color:#3730A3")
)
st.dataframe(styled)
st.caption("Bold cells mark the best value of each column.")

# ----------------------------------------------------------------------------
# Radar + ROC
# ----------------------------------------------------------------------------
selected = st.multiselect("Models to display", list(results), default=list(results))
if not selected:
    st.info("Select at least one model to draw the charts.")
    st.stop()

left, right = st.columns(2)
with left:
    radar = go.Figure()
    axes = [METRIC_LABELS[m] for m in METRIC_COLUMNS]
    for name in selected:
        values = [results[name]["test"][m] for m in METRIC_COLUMNS]
        radar.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=axes + axes[:1],
                name=name,
                mode="lines+markers",
                line={"color": colors[name], "width": 3 if name == best else 2},
                marker={"size": 8},
                hovertemplate=f"<b>{name}</b><br>%{{theta}}: %{{r:.3f}}<extra></extra>",
            )
        )
    radar.update_layout(
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {"range": [0.4, 0.9], "tickformat": ".1f", "gridcolor": "#E2E8F0",
                           "tickfont": {"size": 10, "color": ui.INK_MUTED}, "angle": 90,
                           "tickangle": 0, "dtick": 0.1},
            "angularaxis": {"gridcolor": "#E2E8F0", "tickfont": {"size": 12, "color": ui.INK}},
        }
    )
    fig = ui.style_figure(radar, "Metric profile (radar)", height=460)
    fig.update_layout(legend={"y": -0.18, "yanchor": "top", "x": 0.5, "xanchor": "center"},
                      margin={"t": 70, "b": 60, "l": 30, "r": 30})
    ui.show_chart(fig, key="radar")

with right:
    roc = go.Figure()
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                             line={"dash": "dash", "color": ui.INK_MUTED, "width": 1.5},
                             hoverinfo="skip"))
    for name in selected:
        auc = results[name]["test"]["roc_auc"]
        roc.add_trace(
            go.Scatter(
                x=roc_curves[name]["fpr"], y=roc_curves[name]["tpr"], mode="lines",
                name=f"{name} (AUC {auc:.3f})",
                line={"color": colors[name], "width": 3 if name == best else 2},
                hovertemplate=f"<b>{name}</b><br>False positive rate: %{{x:.2f}}"
                              "<br>True positive rate: %{y:.2f}<extra></extra>",
            )
        )
    roc.update_xaxes(title_text="False positive rate", range=[0, 1], showgrid=True, gridcolor="#EEF2F7")
    roc.update_yaxes(title_text="True positive rate", range=[0, 1.02])
    fig = ui.style_figure(roc, "ROC curves", height=460)
    fig.update_layout(legend={"orientation": "v", "y": 0.04, "yanchor": "bottom", "x": 0.98,
                              "xanchor": "right", "bgcolor": "rgba(255,255,255,.85)"})
    ui.show_chart(fig, key="roc")

st.caption(
    "The five models land within one AUC point of each other: on this small tabular dataset "
    "the signal is mostly linear, so simple and complex models perform alike — which favours "
    "the fastest and most interpretable one."
)

# ----------------------------------------------------------------------------
# Decision threshold
# ----------------------------------------------------------------------------
ui.section(f"Business decision threshold — {best}")
threshold = metadata["decision_threshold"]
tuned = metadata["threshold_metrics"]
left, right = st.columns([1.2, 1])
with left:
    st.markdown(
        f"""
        Only about a quarter of customers churn, so the default **0.50** cut-off misses many of
        them. The threshold that maximises F1 on out-of-fold training predictions is
        **{threshold:.2f}**: every customer scored above it is flagged for retention.

        | Metric | Threshold 0.50 | Threshold {threshold:.2f} |
        |---|---|---|
        | Recall (churners caught) | {final['recall']:.1%} | **{tuned['recall']:.1%}** |
        | Precision | {final['precision']:.1%} | {tuned['precision']:.1%} |
        | F1 Score | {final['f1']:.3f} | **{tuned['f1']:.3f}** |
        | Accuracy | {final['accuracy']:.1%} | {tuned['accuracy']:.1%} |

        Catching more churners costs some precision: the right balance depends on the
        price of a retention offer versus the value of a saved customer.
        """
    )
    if metadata["best_params"]:
        st.markdown("**Tuned hyper-parameters** (RandomizedSearchCV)")
        st.json(metadata["best_params"], expanded=False)
    else:
        st.caption(
            "RandomizedSearchCV did not beat the default hyper-parameters in cross-validation, "
            "so the default configuration was kept."
        )
with right:
    (tn, fp), (fn, tp) = metadata["threshold_confusion_matrix"]
    matrix = go.Figure(
        go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Predicted: stays", "Predicted: churns"],
            y=["Actual: stays", "Actual: churns"],
            colorscale=[[0, "#EFF6FF"], [1, ui.RETAINED]],
            showscale=False,
            text=[[f"{tn:,}<br>true negatives", f"{fp:,}<br>false alarms"],
                  [f"{fn:,}<br>missed churners", f"{tp:,}<br>churners caught"]],
            texttemplate="%{text}",
            textfont={"size": 14},
            hovertemplate="%{y} · %{x}<br>%{z:,} customers<extra></extra>",
            xgap=2, ygap=2,
        )
    )
    matrix.update_yaxes(autorange="reversed", showgrid=False)
    ui.show_chart(ui.style_figure(matrix, f"Confusion matrix at {threshold:.2f}", height=340,
                                  legend=False), key="confusion")
