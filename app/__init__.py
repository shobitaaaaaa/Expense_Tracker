"""
app — Expense Tracker package.

Public API
----------
from app.database import Database
from app.detector import AnomalyDetectionEngine
from app.visualizer import (
    plot_spending_by_category,
    plot_monthly_trend,
    plot_anomaly_scatter,
    plot_zscore_distribution,
    plot_category_pie,
)
"""

from .database   import Database
from .detector   import AnomalyDetectionEngine
from .visualizer import (
    plot_spending_by_category,
    plot_monthly_trend,
    plot_anomaly_scatter,
    plot_zscore_distribution,
    plot_category_pie,
)

__all__ = [
    "Database",
    "AnomalyDetectionEngine",
    "plot_spending_by_category",
    "plot_monthly_trend",
    "plot_anomaly_scatter",
    "plot_zscore_distribution",
    "plot_category_pie",
]
