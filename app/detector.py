"""
app/detector.py — Fraud & Anomaly Detection engine.

Two detection methods:
  1. Z-Score          — statistical, fast, interpretable
  2. Isolation Forest — ML-based, catches complex patterns

What is an anomaly in spending?
--------------------------------
A transaction that is unusually large compared to your normal spending
patterns — could be fraud, a one-off purchase, or a billing error.

Both methods look at transaction AMOUNTS and flag outliers.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
import config


@dataclass
class DetectionResult:
    """
    Structured result from anomaly detection.

    Fields
    ------
    anomaly_ids   : list of transaction IDs flagged as anomalies
    anomaly_df    : DataFrame of flagged transactions with scores
    method        : which method was used
    threshold     : the cutoff value used
    total_flagged : how many transactions were flagged
    """
    anomaly_ids   : list
    anomaly_df    : pd.DataFrame
    method        : str
    threshold     : float
    total_flagged : int


class ZScoreDetector:
    """
    Z-Score Anomaly Detector.

    What is a Z-Score?
    ------------------
    Z-score measures how many standard deviations a value is from the mean.

        Z = (X - μ) / σ

    where:
      X  = the transaction amount
      μ  = mean (average) of all transaction amounts
      σ  = standard deviation (how spread out amounts are)

    Example:
      Average spend = ₹500, std dev = ₹300
      A transaction of ₹5000:
        Z = (5000 - 500) / 300 = 15  ← very suspicious!

      A transaction of ₹800:
        Z = (800 - 500) / 300 = 1.0  ← normal, within 1 std dev

    Threshold: if |Z| > 2.5, we flag it as an anomaly.
    This means only the top ~1.2% of transactions get flagged.

    Parameters
    ----------
    threshold : float — Z-score cutoff (default 2.5 from config)
    """

    def __init__(self, threshold: float = config.ZSCORE_THRESHOLD):
        self.threshold = threshold

    def detect(self, df: pd.DataFrame) -> DetectionResult:
        """
        Run Z-score detection on transaction amounts.

        Parameters
        ----------
        df : pd.DataFrame — must have columns: id, amount, date, description, category

        Returns
        -------
        DetectionResult
        """
        if df.empty:
            return DetectionResult([], pd.DataFrame(), "Z-Score", self.threshold, 0)

        amounts  = df["amount"].values
        mu       = amounts.mean()
        sigma    = amounts.std()

        if sigma == 0:
            return DetectionResult([], pd.DataFrame(), "Z-Score", self.threshold, 0)

        z_scores = (amounts - mu) / sigma

        result_df = df.copy()
        result_df["z_score"] = np.round(z_scores, 3)
        result_df["anomaly_score"] = np.abs(z_scores)

        flagged   = result_df[result_df["anomaly_score"] > self.threshold].copy()
        flagged   = flagged.sort_values("anomaly_score", ascending=False)

        return DetectionResult(
            anomaly_ids   = flagged["id"].tolist(),
            anomaly_df    = flagged,
            method        = "Z-Score",
            threshold     = self.threshold,
            total_flagged = len(flagged),
        )


class IsolationForestDetector:
    """
    Isolation Forest Anomaly Detector.

    What is Isolation Forest?
    -------------------------
    A machine learning algorithm that detects anomalies by isolation:
    - Anomalies are rare and different → easier to isolate
    - Normal points cluster together → harder to isolate

    How it works:
      1. Randomly pick a feature (e.g. amount) and a random split value
      2. Keep splitting until each point is isolated in its own leaf
      3. Anomalies get isolated in FEWER splits (shorter path length)
      4. Points with short average path lengths = anomalies

    Advantage over Z-score:
      - Works on multiple features simultaneously (amount + category + day-of-week)
      - Catches complex patterns, not just high spend amounts
      - No assumption about normal distribution

    Parameters
    ----------
    contamination : float — expected proportion of anomalies (default 0.05 = 5%)
    """

    def __init__(self, contamination: float = config.ISOLATION_CONTAMINATION):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination = contamination,
            random_state  = 42,
            n_estimators  = 100,
        )

    def detect(self, df: pd.DataFrame) -> DetectionResult:
        """
        Run Isolation Forest on amount + encoded category + day of week.

        Using multiple features makes detection richer than amount alone.
        """
        if df.empty or len(df) < 10:
            return DetectionResult([], pd.DataFrame(), "Isolation Forest",
                                   self.contamination, 0)

        result_df = df.copy()
        result_df["day_of_week"] = pd.to_datetime(result_df["date"]).dt.dayofweek
        result_df["category_code"] = result_df["category"].astype("category").cat.codes

        features = result_df[["amount", "day_of_week", "category_code"]].values

        preds   = self.model.fit_predict(features)       # -1 = anomaly, 1 = normal
        scores  = self.model.score_samples(features)     # lower = more anomalous

        result_df["anomaly_score"] = -scores             # flip so higher = more anomalous
        result_df["if_label"]      = preds

        flagged = result_df[result_df["if_label"] == -1].copy()
        flagged = flagged.sort_values("anomaly_score", ascending=False)

        return DetectionResult(
            anomaly_ids   = flagged["id"].tolist(),
            anomaly_df    = flagged,
            method        = "Isolation Forest",
            threshold     = self.contamination,
            total_flagged = len(flagged),
        )


class AnomalyDetectionEngine:
    """
    Orchestrator — runs both detectors and combines results.

    Flags a transaction if EITHER method marks it as an anomaly.
    Union approach = higher recall (catches more potential fraud).
    """

    def __init__(self):
        self.zscore_detector = ZScoreDetector()
        self.if_detector     = IsolationForestDetector()

    def run(self, df: pd.DataFrame) -> dict:
        """
        Run both detectors and return combined results.

        Returns
        -------
        dict with keys:
          zscore   : DetectionResult
          iso      : DetectionResult
          combined : list of all unique anomaly IDs (union)
        """
        zscore_result = self.zscore_detector.detect(df)
        iso_result    = self.if_detector.detect(df)

        combined_ids = list(
            set(zscore_result.anomaly_ids) | set(iso_result.anomaly_ids)
        )

        return {
            "zscore"   : zscore_result,
            "iso"      : iso_result,
            "combined" : combined_ids,
        }
