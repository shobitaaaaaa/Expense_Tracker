"""
dashboard.py — Streamlit dashboard for the Personal Expense Tracker.

Run with:
    streamlit run dashboard.py

What is Streamlit?
------------------
Streamlit turns a Python script into an interactive web app.
Every time the user interacts (clicks a button, fills a form),
the script re-runs from top to bottom and the UI updates.
No HTML, CSS, or JavaScript needed.
"""

import streamlit as st
import pandas as pd
from datetime import date

import config
from app import (
    Database,
    AnomalyDetectionEngine,
    plot_spending_by_category,
    plot_monthly_trend,
    plot_anomaly_scatter,
    plot_zscore_distribution,
    plot_category_pie,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = config.PAGE_TITLE,
    page_icon  = config.PAGE_ICON,
    layout     = "wide",
)

# ── Init database & load sample data on first run ────────────────────────────
@st.cache_resource
def get_db() -> Database:
    db = Database(config.DB_PATH)
    db.load_from_csv("data/sample_data.csv")
    return db

db = get_db()

# ── Run anomaly detection & update DB ────────────────────────────────────────
@st.cache_data(ttl=30)
def run_detection():
    df      = db.fetch_all()
    engine  = AnomalyDetectionEngine()
    results = engine.run(df)
    db.update_anomaly_flags(results["combined"])
    return results, df

detection_results, all_transactions = run_detection()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{config.PAGE_ICON} Expense Tracker")
    st.divider()

    # Add new transaction form
    st.subheader("➕ Add Transaction")
    with st.form("add_transaction", clear_on_submit=True):
        txn_date  = st.date_input("Date", value=date.today())
        txn_desc  = st.text_input("Description", placeholder="e.g. Swiggy Order")
        txn_amt   = st.number_input("Amount (₹)", min_value=1.0, step=10.0)
        txn_cat   = st.selectbox("Category", config.CATEGORIES)
        submitted = st.form_submit_button("Add", use_container_width=True)

        if submitted:
            if txn_desc.strip() == "":
                st.error("Please enter a description.")
            else:
                db.add_transaction(
                    date        = str(txn_date),
                    description = txn_desc.strip(),
                    amount      = txn_amt,
                    category    = txn_cat,
                )
                st.success(f"Added ₹{txn_amt:,.0f} — {txn_desc}")
                st.cache_data.clear()
                st.rerun()

    st.divider()

    # Detection method info
    st.subheader("🔍 Detection Methods")
    zs = detection_results["zscore"]
    iso = detection_results["iso"]
    st.metric("Z-Score Flagged",        zs.total_flagged)
    st.metric("Isolation Forest Flagged", iso.total_flagged)
    st.metric("Combined (Union)",       len(detection_results["combined"]))
    st.caption(f"Z-Score threshold: {config.ZSCORE_THRESHOLD}σ")
    st.caption(f"IF contamination: {config.ISOLATION_CONTAMINATION:.0%}")

# ── Main dashboard ────────────────────────────────────────────────────────────
st.title(f"{config.PAGE_ICON} {config.PAGE_TITLE}")

df = db.fetch_all()

if df.empty:
    st.warning("No transactions yet. Add one from the sidebar!")
    st.stop()

# ── KPI metrics row ──────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Spend",       f"₹{db.total_spend():,.0f}")
col2.metric("📋 Transactions",      db.count())
col3.metric("⚠️ Anomalies Flagged", len(detection_results["combined"]))
col4.metric("📂 Categories",        df["category"].nunique())

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "⚠️ Anomalies",
    "📋 All Transactions",
    "📥 Import CSV",
])

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        category_df = db.spend_by_category()
        st.pyplot(plot_spending_by_category(category_df))

    with col_right:
        st.pyplot(plot_category_pie(category_df))

    monthly_df = db.monthly_summary()
    if len(monthly_df) > 1:
        st.pyplot(plot_monthly_trend(monthly_df))

    st.pyplot(plot_anomaly_scatter(df))

# ── Tab 2: Anomalies ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("⚠️ Flagged Transactions")

    anomaly_df = db.fetch_anomalies()

    if anomaly_df.empty:
        st.success("✅ No anomalies detected in current transactions.")
    else:
        st.error(f"{len(anomaly_df)} transaction(s) flagged as potentially anomalous.")

        # Z-score details
        st.markdown("#### Z-Score Analysis")
        st.pyplot(plot_zscore_distribution(df, config.ZSCORE_THRESHOLD))

        z_flagged = detection_results["zscore"].anomaly_df
        if not z_flagged.empty:
            st.markdown("**Z-Score Flagged:**")
            display_cols = ["date", "description", "amount", "category", "z_score"]
            available = [c for c in display_cols if c in z_flagged.columns]
            st.dataframe(
                z_flagged[available].style.format({"amount": "₹{:,.0f}"}),
                use_container_width=True,
            )

        # Isolation Forest details
        st.markdown("#### Isolation Forest Analysis")
        if_flagged = detection_results["iso"].anomaly_df
        if not if_flagged.empty:
            st.markdown("**Isolation Forest Flagged:**")
            display_cols = ["date", "description", "amount", "category", "anomaly_score"]
            available = [c for c in display_cols if c in if_flagged.columns]
            st.dataframe(
                if_flagged[available].style.format({
                    "amount": "₹{:,.0f}",
                    "anomaly_score": "{:.4f}",
                }),
                use_container_width=True,
            )

        # Combined anomalies table
        st.markdown("#### All Flagged Transactions (Combined)")
        st.dataframe(
            anomaly_df[["date", "description", "amount", "category"]]
            .style.format({"amount": "₹{:,.0f}"}),
            use_container_width=True,
        )

# ── Tab 3: All Transactions ───────────────────────────────────────────────────
with tab3:
    st.subheader("📋 All Transactions")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        cat_filter = st.multiselect("Filter by Category", options=config.CATEGORIES)
    with col_f2:
        show_anomalies_only = st.checkbox("Show anomalies only")
    with col_f3:
        sort_by = st.selectbox("Sort by", ["date", "amount"])

    filtered = df.copy()
    if cat_filter:
        filtered = filtered[filtered["category"].isin(cat_filter)]
    if show_anomalies_only:
        filtered = filtered[filtered["is_anomaly"] == 1]
    filtered = filtered.sort_values(sort_by, ascending=False)

    # Highlight anomaly rows in red
    def highlight_anomaly(row):
        if row["is_anomaly"] == 1:
            return ["background-color: #3d1515"] * len(row)
        return [""] * len(row)

    display_df = filtered[["date", "description", "amount", "category", "is_anomaly"]].copy()
    display_df["is_anomaly"] = display_df["is_anomaly"].map({0: "✅ Normal", 1: "⚠️ Flagged"})

    st.dataframe(
        display_df.style
        .apply(highlight_anomaly, axis=1)
        .format({"amount": "₹{:,.0f}"}),
        use_container_width=True,
        height=400,
    )

    # Delete transaction
    st.markdown("#### 🗑️ Delete a Transaction")
    del_id = st.number_input("Enter Transaction ID to delete", min_value=1, step=1)
    if st.button("Delete", type="secondary"):
        db.delete_transaction(int(del_id))
        st.success(f"Deleted transaction ID {del_id}")
        st.cache_data.clear()
        st.rerun()

# ── Tab 4: Import CSV ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("📥 Import Transactions from CSV")
    st.markdown("""
    Upload a CSV file with these columns:
    ```
    date, description, amount, category
    ```
    Dates should be in `YYYY-MM-DD` format.
    """)

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded:
        try:
            new_df = pd.read_csv(uploaded, parse_dates=["date"])
            new_df["date"] = new_df["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(new_df.head(10), use_container_width=True)

            if st.button("Confirm Import", type="primary"):
                for _, row in new_df.iterrows():
                    db.add_transaction(
                        date        = str(row["date"]),
                        description = str(row["description"]),
                        amount      = float(row["amount"]),
                        category    = str(row["category"]),
                    )
                st.success(f"✅ Imported {len(new_df)} transactions.")
                st.cache_data.clear()
                st.rerun()
        except Exception as e:
            st.error(f"Error reading file: {e}")
