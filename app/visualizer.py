"""
app/visualizer.py — Chart functions for the Streamlit dashboard.

All functions return matplotlib figures so Streamlit can render them
with st.pyplot(fig).
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")

ANOMALY_COLOR  = "#E05C5C"
NORMAL_COLOR   = "#4C72B0"
CATEGORY_PALETTE = sns.color_palette("muted", 10)


def _rupee(x, _):
    """Format axis tick as ₹ amount."""
    return f"₹{x:,.0f}"


def plot_spending_by_category(category_df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of total spend per category."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = CATEGORY_PALETTE[:len(category_df)]
    bars = ax.barh(category_df["category"], category_df["total"],
                   color=colors, edgecolor="white")
    ax.bar_label(bars, labels=[f"₹{v:,.0f}" for v in category_df["total"]],
                 padding=5, fontsize=9)
    ax.set_title("Total Spend by Category", fontsize=13, fontweight="bold")
    ax.set_xlabel("Total Spend (₹)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_rupee))
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_monthly_trend(monthly_df: pd.DataFrame) -> plt.Figure:
    """Line chart of total spend per month."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly_df["month"], monthly_df["total"],
            marker="o", color=NORMAL_COLOR, lw=2.5, markersize=7)
    ax.fill_between(monthly_df["month"], monthly_df["total"],
                    alpha=0.15, color=NORMAL_COLOR)
    ax.set_title("Monthly Spending Trend", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Spend (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_rupee))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


def plot_anomaly_scatter(df: pd.DataFrame) -> plt.Figure:
    """
    Scatter plot of all transactions.
    Anomalies shown in red, normal in blue.
    X-axis = date, Y-axis = amount.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    normal   = df[df["is_anomaly"] == 0]
    anomalies = df[df["is_anomaly"] == 1]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(normal["date"],    normal["amount"],
               color=NORMAL_COLOR, alpha=0.6, s=50, label="Normal")
    ax.scatter(anomalies["date"], anomalies["amount"],
               color=ANOMALY_COLOR, s=100, zorder=5, marker="D",
               edgecolors="white", lw=0.5, label="⚠ Anomaly")

    for _, row in anomalies.iterrows():
        ax.annotate(
            f"₹{row['amount']:,.0f}",
            xy=(row["date"], row["amount"]),
            xytext=(8, 5), textcoords="offset points",
            fontsize=8, color=ANOMALY_COLOR,
        )

    ax.set_title("Transaction Anomaly Map", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_rupee))
    ax.legend()
    plt.tight_layout()
    return fig


def plot_zscore_distribution(df: pd.DataFrame, threshold: float) -> plt.Figure:
    """
    Histogram of transaction amounts with Z-score threshold lines.
    Shows visually where anomalies begin.
    """
    amounts = df["amount"].values
    mu      = amounts.mean()
    sigma   = amounts.std()
    upper   = mu + threshold * sigma

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(amounts, bins=30, color=NORMAL_COLOR, alpha=0.75,
            edgecolor="white", label="All Transactions")
    ax.axvline(upper, color=ANOMALY_COLOR, lw=2, ls="--",
               label=f"Anomaly Threshold (Z={threshold}) = ₹{upper:,.0f}")
    ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 50],
                     upper, amounts.max(),
                     alpha=0.1, color=ANOMALY_COLOR, label="Anomaly Zone")
    ax.set_title("Spend Distribution & Z-Score Threshold", fontsize=13, fontweight="bold")
    ax.set_xlabel("Transaction Amount (₹)")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_rupee))
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_category_pie(category_df: pd.DataFrame) -> plt.Figure:
    """Donut chart of spend share by category."""
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        category_df["total"],
        labels=category_df["category"],
        autopct="%1.1f%%",
        colors=CATEGORY_PALETTE[:len(category_df)],
        pctdistance=0.82,
        wedgeprops=dict(width=0.5),
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("Spend Share by Category", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig
