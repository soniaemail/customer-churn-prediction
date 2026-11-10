"""📈 Dashboard - business view of the customer base and its churn."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import ui


def churn_rate_by(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn rate (%) and customer count for each value of ``column``."""
    grouped = (
        frame.assign(churned=(frame["Churn"] == "Yes"))
        .groupby(column, observed=True)
        .agg(churn_rate=("churned", "mean"), customers=("churned", "size"))
        .reset_index()
    )
    grouped["churn_rate"] *= 100
    return grouped


def rate_bar(frame: pd.DataFrame, column: str, title: str, horizontal: bool = False) -> go.Figure:
    """Single-series bar chart of the churn rate with direct labels."""
    labels = [f"{v:.0f}%" for v in frame["churn_rate"]]
    hover = "<b>%{customdata[0]}</b><br>Churn rate: %{customdata[1]:.1f}%<br>Customers: %{customdata[2]:,}<extra></extra>"
    custom = frame[[column, "churn_rate", "customers"]].to_numpy()
    if horizontal:
        bar = go.Bar(y=frame[column], x=frame["churn_rate"], orientation="h", text=labels)
    else:
        bar = go.Bar(x=frame[column], y=frame["churn_rate"], text=labels)
    bar.update(marker={"color": ui.RETAINED, "cornerradius": 4}, textposition="outside",
               textfont={"color": ui.INK}, customdata=custom, hovertemplate=hover, cliponaxis=False)
    fig = ui.style_figure(go.Figure(bar), title, legend=False)
    limit = max(frame["churn_rate"].max() * 1.2, 10)
    if horizontal:
        fig.update_xaxes(range=[0, limit], ticksuffix="%", showgrid=True, gridcolor="#EEF2F7")
        fig.update_yaxes(showgrid=False, autorange="reversed", ticksuffix="  ")
    else:
        fig.update_yaxes(range=[0, limit], ticksuffix="%")
    return fig


data = ui.get_data()

header, animation = st.columns([3, 1], vertical_alignment="center")
with header:
    ui.page_header("Customer Dashboard", "Who are our customers, and where does churn come from?")
with animation:
    ui.show_lottie("analytics", height=150, fallback="📊")

# ----------------------------------------------------------------------------
# Filters (one row, above every chart they control)
# ----------------------------------------------------------------------------
f1, f2, f3 = st.columns(3)
contracts = f1.multiselect("Contract", sorted(data["Contract"].unique()), placeholder="All contracts")
internet = f2.multiselect("Internet service", sorted(data["InternetService"].unique()),
                          placeholder="All services")
payment = f3.multiselect("Payment method", sorted(data["PaymentMethod"].unique()),
                         placeholder="All payment methods")

view = data
if contracts:
    view = view[view["Contract"].isin(contracts)]
if internet:
    view = view[view["InternetService"].isin(internet)]
if payment:
    view = view[view["PaymentMethod"].isin(payment)]

if view.empty:
    st.warning("No customer matches these filters. Remove a filter to see the dashboard.")
    st.stop()

# ----------------------------------------------------------------------------
# KPIs
# ----------------------------------------------------------------------------
churned = view[view["Churn"] == "Yes"]
k1, k2, k3, k4 = st.columns(4)
with k1:
    ui.kpi_card("Total customers", f"{len(view):,}", f"{len(view) / len(data):.0%} of the base", "👥")
with k2:
    ui.kpi_card("Churn rate", f"{len(churned) / len(view):.1%}", f"{len(churned):,} customers lost", "📉")
with k3:
    ui.kpi_card("Average revenue", f"${view['MonthlyCharges'].mean():.2f}",
                "Per customer, per month", "💳")
with k4:
    ui.kpi_card("Average tenure", f"{view['tenure'].mean():.1f} mo",
                f"Churned customers: {churned['tenure'].mean():.1f} mo" if len(churned) else "", "⏳")

st.write("")

# ----------------------------------------------------------------------------
# Row 1: churn split + tenure distribution
# ----------------------------------------------------------------------------
left, right = st.columns([1, 1.6])
with left:
    counts = view["Churn"].value_counts().reindex(["No", "Yes"]).fillna(0)
    pie = go.Figure(
        go.Pie(
            labels=[ui.CHURN_LABELS[k] for k in counts.index],
            values=counts.values,
            hole=0.62,
            sort=False,
            marker={"colors": [ui.CHURN_COLORS[k] for k in counts.index],
                    "line": {"color": "#fff", "width": 2}},
            textinfo="label+percent",
            textposition="outside",
            textfont={"color": ui.INK},
            hovertemplate="<b>%{label}</b><br>%{value:,} customers (%{percent})<extra></extra>",
        )
    )
    pie.add_annotation(text=f"<b>{len(view):,}</b><br>customers", showarrow=False,
                       font={"size": 16, "color": ui.INK})
    pie_fig = ui.style_figure(pie, "Churn distribution", legend=False)
    pie_fig.update_layout(margin={"l": 70, "r": 70, "t": 56, "b": 30})
    ui.show_chart(pie_fig, key="pie")

with right:
    hist = go.Figure()
    for status in ("No", "Yes"):
        hist.add_trace(
            go.Histogram(
                x=view.loc[view["Churn"] == status, "tenure"],
                name=ui.CHURN_LABELS[status],
                xbins={"start": 0, "end": 73, "size": 6},
                marker={"color": ui.CHURN_COLORS[status], "line": {"color": "#fff", "width": 2}},
                hovertemplate=f"<b>{ui.CHURN_LABELS[status]}</b><br>Tenure: %{{x}} months"
                              "<br>Customers: %{y:,}<extra></extra>",
            )
        )
    hist.update_layout(barmode="stack")
    hist.update_xaxes(title_text="Tenure (months)")
    hist.update_yaxes(title_text="Customers")
    ui.show_chart(ui.style_figure(hist, "Tenure — churn hits new customers"), key="hist")

# ----------------------------------------------------------------------------
# Row 2: churn rate by contract and by payment method
# ----------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    by_contract = churn_rate_by(view, "Contract").set_index("Contract")
    order = [c for c in ("Month-to-month", "One year", "Two year") if c in by_contract.index]
    ui.show_chart(rate_bar(by_contract.loc[order].reset_index(), "Contract", "Churn rate by contract"),
                  key="contract")
with right:
    by_payment = churn_rate_by(view, "PaymentMethod").sort_values("churn_rate", ascending=False)
    ui.show_chart(rate_bar(by_payment, "PaymentMethod", "Churn rate by payment method", horizontal=True),
                  key="payment")

# ----------------------------------------------------------------------------
# Row 3: monthly charges
# ----------------------------------------------------------------------------
box = go.Figure()
for status in ("No", "Yes"):
    box.add_trace(
        go.Box(
            x=view.loc[view["Churn"] == status, "MonthlyCharges"],
            name=ui.CHURN_LABELS[status],
            marker={"color": ui.CHURN_COLORS[status]},
            line={"width": 2},
            boxmean=True,
            orientation="h",
        )
    )
box.update_xaxes(title_text="Monthly charges ($)", tickprefix="$", showgrid=True, gridcolor="#EEF2F7")
box.update_yaxes(showgrid=False)
ui.show_chart(ui.style_figure(box, "Monthly charges — churned customers pay more", height=300,
                              legend=False), key="charges")

with st.expander("📋 View the underlying data"):
    st.dataframe(view, hide_index=True, height=320)
    st.download_button("Download as CSV", view.to_csv(index=False).encode("utf-8"),
                       file_name="customers_filtered.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# Takeaways computed from the current selection
# ----------------------------------------------------------------------------
ui.section("Key takeaways")
new_customers = view[view["tenure"] <= 12]
loyal = view[view["tenure"] > 24]
c1, c2, c3 = st.columns(3)
with c1:
    if len(new_customers) and len(loyal):
        ui.info_card(
            "The first year is critical",
            f"{(new_customers['Churn'] == 'Yes').mean():.0%} of customers in their first 12 months "
            f"leave, versus {(loyal['Churn'] == 'Yes').mean():.0%} after two years.",
            emoji="⏱️",
        )
with c2:
    worst = churn_rate_by(view, "Contract").sort_values("churn_rate").iloc[-1]
    ui.info_card(
        "Commitment protects revenue",
        f"'{worst['Contract']}' contracts show the highest churn rate "
        f"({worst['churn_rate']:.0f}%) in this selection.",
        emoji="📝",
    )
with c3:
    lost = churned["MonthlyCharges"].sum()
    ui.info_card(
        "Revenue at stake",
        f"Churned customers represented ${lost:,.0f} of monthly recurring revenue "
        f"({lost / view['MonthlyCharges'].sum():.0%} of the total).",
        emoji="💸",
    )
