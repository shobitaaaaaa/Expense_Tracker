# Personal Expense Tracker with Fraud Detection

A modular **personal finance dashboard** with dual-method anomaly detection, built with Python and Streamlit.

## Features

- Add, view, filter, and delete transactions
- Import transactions from CSV
- Automatic anomaly detection using two methods:
  - **Z-Score** — flags statistically unusual spend amounts
  - **Isolation Forest** — ML-based detection across multiple features
- 5 interactive charts: category breakdown, monthly trend, anomaly scatter, spend distribution, donut chart
- SQLite persistence — all data stored locally in `expenses.db`

## Project Structure

```
ExpenseTracker/
├── app/
│   ├── database.py      # Database — SQLite read/write layer
│   ├── detector.py      # ZScoreDetector, IsolationForestDetector, AnomalyDetectionEngine
│   └── visualizer.py    # All matplotlib chart functions
├── data/
│   └── sample_data.csv  # 60 sample transactions loaded on first run
├── dashboard.py         # Streamlit UI — run this to launch
├── config.py            # All settings
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Opens at `http://localhost:8501` in your browser automatically.

## Configuration

Edit `config.py`:

```python
ZSCORE_THRESHOLD        = 2.5    # flag if Z-score exceeds this
ISOLATION_CONTAMINATION = 0.05   # expected % anomalies for Isolation Forest
```

## Anomaly Detection

| Method | How it works | Best for |
|---|---|---|
| Z-Score | Flags amounts > 2.5 standard deviations from mean | Large one-off transactions |
| Isolation Forest | ML model using amount + category + day of week | Complex spending patterns |
| Combined | Union of both methods | Maximum recall |
