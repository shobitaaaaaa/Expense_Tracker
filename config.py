"""
config.py — Central configuration for the Expense Tracker.
"""

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = "expenses.db"          # SQLite file created automatically on first run

# ── Anomaly Detection ─────────────────────────────────────────────────────────
ZSCORE_THRESHOLD        = 2.5    # flag transaction if z-score exceeds this
ISOLATION_CONTAMINATION = 0.05   # expected % of anomalies (5%)

# ── Categories ────────────────────────────────────────────────────────────────
CATEGORIES = [
    "Food & Dining",
    "Transport",
    "Shopping",
    "Entertainment",
    "Utilities",
    "Health",
    "Travel",
    "Other",
]

# ── Dashboard ─────────────────────────────────────────────────────────────────
PAGE_TITLE  = "Personal Expense Tracker"
PAGE_ICON   = "💳"
