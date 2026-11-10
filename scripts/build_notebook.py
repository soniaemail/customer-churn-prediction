"""Build (and execute) ``notebooks/churn_analysis.ipynb``.

The notebook is generated from this script so that it is always reproducible
and never ships with stale outputs::

    python scripts/build_notebook.py            # write + execute
    python scripts/build_notebook.py --no-run   # write only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "churn_analysis.ipynb"

CELLS: list[tuple[str, str]] = [
    ("md", """# Customer Churn Prediction — Exploratory Data Analysis

**Goal:** understand who churns and why, before building a model that predicts it.

**Dataset:** IBM *Telco Customer Churn* — 7,043 customers of a telecom operator, 20 attributes and the `Churn` label.

**Contents**
1. Setup and data loading
2. Descriptive analysis (shape, types, missing values, statistics)
3. Visual analysis (target, contract, tenure, charges, correlations)
4. Preprocessing pipeline
5. Model comparison and explainability
6. Conclusions"""),
    ("md", "## 1. Setup and data loading"),
    ("code", """import sys
import warnings
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

from src.data_loader import load_raw_data
from src.preprocessing import build_preprocessor, clean_data, split_data

# Colours: blue = retained, orange = churned (colour-blind safe pair)
RETAINED, CHURNED = "#2A78D6", "#EB6834"
COLORS = {"No": RETAINED, "Yes": CHURNED}

pio.renderers.default = "notebook_connected"
pio.templates.default = "plotly_white"
sns.set_theme(style="whitegrid", palette=[RETAINED, CHURNED])
pd.set_option("display.max_columns", 30)"""),
    ("code", """raw = load_raw_data()   # downloads the CSV on first run
raw.head()"""),
    ("md", "## 2. Descriptive analysis\n### 2.1 Shape and types"),
    ("code", """print(f"Rows: {raw.shape[0]:,}  |  Columns: {raw.shape[1]}")
raw.dtypes.to_frame("dtype").T"""),
    ("md", """`TotalCharges` is stored as text although it is a number: a sign of hidden missing values.

### 2.2 Missing values"""),
    ("code", """explicit_nan = raw.isna().sum().sum()
blank_total = (raw["TotalCharges"].astype(str).str.strip() == "").sum()
print(f"Explicit NaN values: {explicit_nan}")
print(f"Blank TotalCharges:  {blank_total}")

raw.loc[raw["TotalCharges"].astype(str).str.strip() == "", ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]]"""),
    ("md", """The 11 blank `TotalCharges` all belong to customers with `tenure = 0`: they have not been billed yet.
They are not errors — the pipeline fills them with `tenure × MonthlyCharges` (= 0).

### 2.3 Statistics"""),
    ("code", """df = clean_data(raw)
df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2)"""),
    ("code", """df.describe(include="object").T"""),
    ("code", """print(f"Duplicated customer IDs: {raw['customerID'].duplicated().sum()}")
churn_rate = (df["Churn"] == "Yes").mean()
print(f"Churn rate: {churn_rate:.1%}")"""),
    ("md", """## 3. Visual analysis
### 3.1 Churn distribution"""),
    ("code", """counts = df["Churn"].value_counts().reindex(["No", "Yes"])
fig = go.Figure(go.Pie(
    labels=["Retained", "Churned"], values=counts.values, hole=0.6, sort=False,
    marker=dict(colors=[RETAINED, CHURNED], line=dict(color="white", width=2)),
    textinfo="label+percent", textposition="outside",
))
fig.update_layout(title="Churn distribution", showlegend=False, height=380)
fig.show()"""),
    ("md", """**26.5% of customers churned.** The classes are imbalanced: accuracy alone would be misleading
(a model predicting "nobody leaves" would reach 73.5%), so ROC AUC, recall and F1 are the metrics to watch.

### 3.2 Churn vs contract"""),
    ("code", """def churn_rate_by(column: str) -> pd.DataFrame:
    out = (df.assign(churned=df["Churn"].eq("Yes"))
             .groupby(column).agg(churn_rate=("churned", "mean"), customers=("churned", "size"))
             .reset_index())
    out["churn_rate"] = (100 * out["churn_rate"]).round(1)
    return out

by_contract = churn_rate_by("Contract")
fig = px.bar(by_contract, x="Contract", y="churn_rate", text="churn_rate",
             hover_data=["customers"], color_discrete_sequence=[RETAINED],
             title="Churn rate by contract type")
fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside", marker_cornerradius=4)
fig.update_layout(yaxis_title="Churn rate (%)", xaxis_title="", height=400)
fig.show()
by_contract"""),
    ("md", """Month-to-month customers churn at **43%**, versus 11% (one year) and 3% (two years).
Commitment is the strongest structural protection against churn.

### 3.3 Churn vs tenure"""),
    ("code", """fig = px.histogram(df, x="tenure", color="Churn", nbins=36, barmode="stack",
                   color_discrete_map=COLORS, category_orders={"Churn": ["No", "Yes"]},
                   title="Tenure distribution by churn status")
fig.update_traces(marker_line_color="white", marker_line_width=1.5)
fig.update_layout(xaxis_title="Tenure (months)", yaxis_title="Customers", height=400, legend_title="Churn")
fig.show()"""),
    ("code", """df["TenureGroup"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72], labels=["0-1 year", "1-2 years", "2-4 years", "4+ years"])
by_tenure = churn_rate_by("TenureGroup")
fig = px.bar(by_tenure, x="TenureGroup", y="churn_rate", text="churn_rate", color_discrete_sequence=[RETAINED],
             title="Churn rate by tenure stage")
fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside", marker_cornerradius=4)
fig.update_layout(yaxis_title="Churn rate (%)", xaxis_title="", height=380)
fig.show()"""),
    ("md", """Churn is an **early-life problem**: almost half of first-year customers leave, and the rate falls
steadily afterwards.

### 3.4 Churn vs monthly charges"""),
    ("code", """fig = px.violin(df, x="Churn", y="MonthlyCharges", color="Churn", box=True, points=False,
                color_discrete_map=COLORS, category_orders={"Churn": ["No", "Yes"]},
                title="Monthly charges by churn status")
fig.update_layout(yaxis_title="Monthly charges ($)", xaxis_title="Churn", showlegend=False, height=420)
fig.show()
df.groupby("Churn")["MonthlyCharges"].describe().round(2)"""),
    ("code", """# Static version (renders on GitHub, where interactive Plotly charts do not)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", hue_order=["No", "Yes"], fill=True, common_norm=False, ax=axes[0])
axes[0].set_title("Monthly charges")
sns.kdeplot(data=df, x="tenure", hue="Churn", hue_order=["No", "Yes"], fill=True, common_norm=False, ax=axes[1])
axes[1].set_title("Tenure (months)")
sns.barplot(data=df.assign(churned=df["Churn"].eq("Yes") * 100), x="Contract", y="churned", color=RETAINED, errorbar=None, ax=axes[2])
axes[2].set_title("Churn rate by contract (%)"); axes[2].set_ylabel("")
plt.tight_layout(); plt.show()"""),
    ("md", """Churned customers pay more on average (median ≈ $80 vs ≈ $64). The retained group has a large
mass of low bills (~$20): phone-only customers, who rarely leave.

### 3.5 Other segments"""),
    ("code", """segments = ["InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity", "SeniorCitizen", "PaperlessBilling"]
rows = []
for column in segments:
    part = churn_rate_by(column).rename(columns={column: "value"})
    part.insert(0, "feature", column)
    rows.append(part)
segment_table = pd.concat(rows, ignore_index=True)

fig = px.bar(segment_table, x="churn_rate", y="value", facet_col="feature", facet_col_wrap=3,
             orientation="h", text="churn_rate", color_discrete_sequence=[RETAINED],
             title="Churn rate (%) across customer segments", height=560)
fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside", marker_cornerradius=4)
fig.update_yaxes(matches=None, showticklabels=True, title="")
fig.update_xaxes(title="", range=[0, 60])
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.show()"""),
    ("md", "### 3.6 Correlation matrix"),
    ("code", """encoded = df.drop(columns=["TenureGroup"]).copy()
encoded["Churn"] = encoded["Churn"].eq("Yes").astype(int)
binary = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for column in binary:
    encoded[column] = encoded[column].isin(["Yes", "Male"]).astype(int)
encoded = pd.get_dummies(encoded, columns=["Contract", "InternetService", "PaymentMethod"], dtype=int)
encoded = encoded.select_dtypes("number")
corr = encoded.corr().round(2)

# Diverging scale: blue = negative, neutral grey = 0, orange = positive
scale = [[0, RETAINED], [0.5, "#F1F1EF"], [1, CHURNED]]
fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                           colorscale=scale, xgap=1, ygap=1, colorbar=dict(title="r"),
                           hovertemplate="%{y} × %{x}<br>r = %{z}<extra></extra>"))
fig.update_layout(title="Correlation matrix", height=720, yaxis_autorange="reversed")
fig.show()"""),
    ("code", """target_corr = corr["Churn"].drop("Churn").sort_values()
fig, ax = plt.subplots(figsize=(8, 6.5))
ax.barh(target_corr.index, target_corr.values, color=[CHURNED if v > 0 else RETAINED for v in target_corr.values])
ax.axvline(0, color="#475569", linewidth=1)
ax.set_title("Correlation with churn (orange = more churn, blue = less churn)")
plt.tight_layout(); plt.show()"""),
    ("md", """**Reading:** month-to-month contracts, fiber optic and electronic checks are the strongest positive
correlates of churn; tenure and two-year contracts are the strongest protective factors.
`TotalCharges` is highly correlated with `tenure` (r ≈ 0.83) — it is essentially tenure × price.

## 4. Preprocessing pipeline

The same `Pipeline` is used for training and for the web app:

| Step | Columns | Transformation |
|---|---|---|
| Feature engineering | all | `AvgMonthlySpend`, `ChargeDelta`, `NumServices`, `TenureGroup` |
| Numeric | 6 | median imputation → `StandardScaler` |
| Binary | 6 | label (ordinal) encoding |
| Nominal | 11 | one-hot encoding |"""),
    ("code", """X_train, X_test, y_train, y_test = split_data(raw)
preprocessor = build_preprocessor()
X_train_t = preprocessor.fit_transform(X_train)
print(f"Before: {X_train.shape}  →  after: {X_train_t.shape}")
preprocessor"""),
    ("code", "X_train_t.head(3).round(2)"),
    ("md", """## 5. Model comparison and explainability

The full training pipeline lives in `src/train.py` (`python -m src.train`). It cross-validates five
models, tunes the best one with `RandomizedSearchCV` and saves every artefact. Results of the last run:"""),
    ("code", """from src.config import METADATA_PATH, METRICS_PATH
from src.evaluate import load_comparison_table, load_json, load_roc_curves

if not METRICS_PATH.exists():          # first run: train the models
    from src.train import run_training
    run_training()

table = load_comparison_table()
metadata = load_json(METADATA_PATH)
print(f"Best model: {metadata['best_model']} (selected on cross-validated ROC AUC)")
table.drop(columns=["cv_roc_auc_std"]).style.format("{:.3f}").highlight_max(axis=0, props="font-weight:bold")"""),
    ("code", """palette = ["#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4"]
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(dash="dash", color="#94A3B8")))
for color, (name, curve) in zip(palette, load_roc_curves().items()):
    fig.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", line=dict(color=color, width=2),
                             name=f"{name} (AUC {table.loc[name, 'roc_auc']:.3f})"))
fig.update_layout(title="ROC curves (test set)", xaxis_title="False positive rate", yaxis_title="True positive rate",
                  height=480, legend=dict(x=0.98, y=0.04, xanchor="right", yanchor="bottom"))
fig.show()"""),
    ("code", """print(f"Decision threshold tuned on out-of-fold predictions: {metadata['decision_threshold']:.2f}")
pd.DataFrame({"threshold 0.50": metadata["final_metrics"],
              f"threshold {metadata['decision_threshold']:.2f}": metadata["threshold_metrics"]}).round(3)"""),
    ("md", "### SHAP — what drives the predictions?"),
    ("code", """import joblib
import shap
from src.config import SHAP_PATH
from src.explain import summary_plot

bundle = joblib.load(SHAP_PATH)
explanation = shap.Explanation(values=bundle["values"], base_values=bundle["base_values"], data=bundle["data"],
                               display_data=bundle["display_data"], feature_names=bundle["feature_names"])
summary_plot(explanation, max_display=14);"""),
    ("code", """index = int(np.argmax(bundle["probabilities"]))   # highest-risk customer of the sample
print(f"Churn probability: {bundle['probabilities'][index]:.0%}")
shap.plots.waterfall(explanation[index], max_display=10)"""),
    ("md", """## 6. Conclusions

**What the data says**
- **Contract** — month-to-month customers churn at 43% vs 3% on two-year contracts.
- **Tenure** — churn is an early-life problem: ~47% in the first year, under 10% after four years.
- **Price and product** — churners pay more; fiber-optic customers are over-represented among them.
- **Payment** — electronic-check payers churn far more than customers on automatic payment.
- **Support** — customers without tech support or online security leave more often.

**What the model adds**
- The five models perform within one AUC point (≈ 0.84): the signal is mostly linear, so the simplest,
  most interpretable model is a sound choice.
- Lowering the decision threshold from 0.50 to the tuned value catches about three churners in four,
  at the cost of more false alarms — a trade-off to set with the retention team.

**Recommended actions**
1. Strengthen onboarding during the first 12 months.
2. Incentivise month-to-month customers to commit to a one-year contract.
3. Review fiber pricing / quality of service and bundle tech support for at-risk customers.
4. Nudge electronic-check payers towards automatic payment."""),
]


def build() -> nbformat.NotebookNode:
    """Assemble the notebook from ``CELLS``."""
    notebook = new_notebook()
    notebook.cells = [
        new_markdown_cell(source) if kind == "md" else new_code_cell(source) for kind, source in CELLS
    ]
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python"}
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-run", action="store_true", help="write the notebook without executing it")
    args = parser.parse_args()

    notebook = build()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_run:
        from nbconvert.preprocessors import ExecutePreprocessor

        ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(
            notebook, {"metadata": {"path": str(NOTEBOOK_PATH.parent)}}
        )
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"wrote {NOTEBOOK_PATH.relative_to(ROOT)} ({NOTEBOOK_PATH.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
