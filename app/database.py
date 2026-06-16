"""
app/database.py — SQLite database layer.

Handles all read/write operations for transactions.

What is SQLite?
---------------
SQLite is a lightweight database stored as a single .db file on your disk.
No server needed — Python's built-in sqlite3 library handles everything.
Think of it as a smarter, queryable Excel file.

Table schema:
  id          — auto-incrementing unique ID for each transaction
  date        — transaction date (YYYY-MM-DD)
  description — what the transaction was for
  amount      — amount spent (₹)
  category    — one of the categories defined in config.py
  is_anomaly  — 0 or 1, flagged by the anomaly detector
"""

import sqlite3
import pandas as pd
from datetime import datetime
import config


class Database:
    """
    Manages all SQLite operations for the expense tracker.

    Parameters
    ----------
    db_path : str — path to the .db file (created automatically if missing)
    """

    def __init__(self, db_path: str = config.DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Create the transactions table if it doesn't exist yet."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    date        TEXT    NOT NULL,
                    description TEXT    NOT NULL,
                    amount      REAL    NOT NULL,
                    category    TEXT    NOT NULL,
                    is_anomaly  INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_transaction(
        self,
        date: str,
        description: str,
        amount: float,
        category: str,
        is_anomaly: int = 0,
    ) -> int:
        """Insert a single transaction. Returns the new row's ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO transactions (date, description, amount, category, is_anomaly)
                   VALUES (?, ?, ?, ?, ?)""",
                (date, description, round(amount, 2), category, is_anomaly),
            )
            conn.commit()
            return cursor.lastrowid

    def load_from_csv(self, csv_path: str) -> int:
        """
        Bulk-load transactions from a CSV file.
        Skips load if data already exists (avoids duplicates on re-run).
        Returns number of rows inserted.
        """
        if self.count() > 0:
            print(f"[Database] Already has {self.count()} rows — skipping CSV load.")
            return 0

        df = pd.read_csv(csv_path, parse_dates=["date"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df["is_anomaly"] = 0

        with self._connect() as conn:
            df.to_sql("transactions", conn, if_exists="append", index=False)
            conn.commit()

        print(f"[Database] Loaded {len(df)} transactions from {csv_path}")
        return len(df)

    def update_anomaly_flags(self, anomaly_ids: list):
        """
        Mark specific transaction IDs as anomalies (is_anomaly = 1).
        Resets all flags first so re-runs stay accurate.
        """
        with self._connect() as conn:
            conn.execute("UPDATE transactions SET is_anomaly = 0")
            if anomaly_ids:
                placeholders = ",".join("?" * len(anomaly_ids))
                conn.execute(
                    f"UPDATE transactions SET is_anomaly = 1 WHERE id IN ({placeholders})",
                    anomaly_ids,
                )
            conn.commit()

    def delete_transaction(self, transaction_id: int):
        with self._connect() as conn:
            conn.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
            conn.commit()

    # ── Read ──────────────────────────────────────────────────────────────────

    def fetch_all(self) -> pd.DataFrame:
        """Return all transactions as a DataFrame, newest first."""
        with self._connect() as conn:
            df = pd.read_sql(
                "SELECT * FROM transactions ORDER BY date DESC, id DESC", conn
            )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def fetch_anomalies(self) -> pd.DataFrame:
        """Return only flagged anomalous transactions."""
        with self._connect() as conn:
            df = pd.read_sql(
                "SELECT * FROM transactions WHERE is_anomaly = 1 ORDER BY amount DESC",
                conn,
            )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    def total_spend(self) -> float:
        with self._connect() as conn:
            result = conn.execute("SELECT SUM(amount) FROM transactions").fetchone()[0]
        return result or 0.0

    def spend_by_category(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql(
                """SELECT category, SUM(amount) as total, COUNT(*) as count
                   FROM transactions GROUP BY category ORDER BY total DESC""",
                conn,
            )

    def monthly_summary(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql(
                """SELECT strftime('%Y-%m', date) as month,
                          SUM(amount) as total,
                          COUNT(*) as transactions
                   FROM transactions
                   GROUP BY month ORDER BY month""",
                conn,
            )
