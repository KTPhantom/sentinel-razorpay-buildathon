"""
Aggregates rule hits and SHAP feature attributions into a single
`triggered_signals` payload — the only thing passed to the explanation layer.
"""

from typing import Any

import pandas as pd

from src.detection.classifier import FraudClassifier
from src.detection.features import extract_synthetic_features_single
from src.detection.rules import RuleEngine


class SignalAggregator:

    def __init__(
        self,
        rule_engine: RuleEngine | None = None,
        classifier: FraudClassifier | None = None,
    ):
        self.rule_engine = rule_engine or RuleEngine()
        self.classifier = classifier

    def process_synthetic_transaction(
        self,
        txn: dict[str, Any],
        recent_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run feature extraction + rule engine on a single synthetic transaction.

        Risk score is a step function of signal count (0 → 0.05, 1 → 0.85,
        2 → 0.94, 3+ → 0.99). XGBoost is not used here — the synthetic schema
        has human-readable fields, not the Kaggle PCA feature set.
        """
        features = extract_synthetic_features_single(txn, recent_history)
        rule_signals = self.rule_engine.evaluate_transaction(features)
        n = len(rule_signals)

        risk_score = {0: 0.05, 1: 0.85, 2: 0.94}.get(n, 0.99)

        return {
            "transaction_id": txn.get("transaction_id", "synthetic_txn"),
            "flag": n > 0,
            "risk_score": round(risk_score, 2),
            "triggered_signals": rule_signals,
            "raw_features_summary": {
                "amount": features.get("amount"),
                "amount_ratio": features.get("amount_ratio"),
                "geo_delta_km": features.get("geo_delta_km"),
                "is_new_device": features.get("is_new_device"),
                "txn_velocity_2min": features.get("txn_velocity_2min"),
                "is_odd_hour": features.get("is_odd_hour"),
            },
        }

    def process_synthetic_batch(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Run feature extraction + rule engine on a batch of synthetic transactions."""
        from src.detection.features import extract_synthetic_features_batch

        features_df = extract_synthetic_features_batch(df)
        results = []

        for _, row in features_df.iterrows():
            features_dict = row.to_dict()
            rule_signals = self.rule_engine.evaluate_transaction(features_dict)
            n = len(rule_signals)
            risk_score = {0: 0.05, 1: 0.85, 2: 0.94}.get(n, 0.99)

            results.append({
                "transaction_id": features_dict.get("transaction_id", "synthetic_txn"),
                "flag": n > 0,
                "risk_score": round(risk_score, 2),
                "triggered_signals": rule_signals,
                "ground_truth_cause": features_dict.get("root_cause", "unknown"),
            })

        return results

    def process_kaggle_transaction(
        self,
        txn: dict[str, Any],
        transaction_id: str | None = None,
    ) -> dict[str, Any]:
        """Run XGBoost + SHAP on a Kaggle transaction (Time, V1..V28, Amount)."""
        if self.classifier is None:
            raise RuntimeError(
                "FraudClassifier must be loaded to process Kaggle transactions. "
                "Run: python scripts/train_classifier.py"
            )

        result = self.classifier.predict_single(txn)
        signals = []
        if result["flag"]:
            signals = [f for f in result["top_shap_features"] if f["contribution"] > 0]

        return {
            "transaction_id": transaction_id or f"kaggle_txn_{int(txn.get('Time', 0))}",
            "flag": result["flag"],
            "risk_score": result["risk_score"],
            "triggered_signals": signals,
        }
