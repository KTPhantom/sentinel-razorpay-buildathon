"""
Main pipeline — wires together the detection and explanation layers.

The two layers are kept intentionally separate: detection produces
a `triggered_signals` list, and that's the only thing the explanation
layer ever sees. No raw transaction data crosses that boundary.

Run `python scripts/train_classifier.py` before using analyze_kaggle_transaction
or the dashboard — the model artifact isn't committed to the repo.
"""

import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MODELS_DIR
from src.detection.aggregator import SignalAggregator
from src.detection.classifier import FraudClassifier
from src.detection.rules import RuleEngine
from src.explanation.generator import ExplanationGenerator
from src.explanation.validator import GroundingValidator

_MODEL_PATH = MODELS_DIR / "xgboost_fraud_model.joblib"


class TransactionAnomalyPipeline:

    def __init__(
        self,
        classifier: FraudClassifier | None = None,
        rule_engine: RuleEngine | None = None,
        generator: ExplanationGenerator | None = None,
        validator: GroundingValidator | None = None,
    ):
        self.classifier = classifier
        if self.classifier is None:
            if _MODEL_PATH.exists():
                try:
                    self.classifier = FraudClassifier.load()
                except Exception as exc:
                    warnings.warn(
                        f"Could not load model from {_MODEL_PATH}: {exc}\n"
                        "Kaggle-path analysis will be unavailable. "
                        "Run: python scripts/train_classifier.py",
                        stacklevel=2,
                    )
                    self.classifier = None
            else:
                warnings.warn(
                    f"No trained model found at {_MODEL_PATH}.\n"
                    "analyze_kaggle_transaction will raise RuntimeError until you run:\n"
                    "    python scripts/train_classifier.py\n"
                    "Synthetic transaction analysis still works without it.",
                    stacklevel=2,
                )

        self.rule_engine = rule_engine or RuleEngine()
        self.aggregator = SignalAggregator(
            rule_engine=self.rule_engine,
            classifier=self.classifier,
        )
        self.validator = validator or GroundingValidator()
        self.generator = generator or ExplanationGenerator(validator=self.validator)

    @property
    def classifier_ready(self) -> bool:
        return self.classifier is not None

    def analyze_synthetic_transaction(
        self,
        txn: dict[str, Any],
        recent_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run a single synthetic transaction through the full pipeline.

        Uses rule engine only for detection — the synthetic schema has human-readable
        fields (amount, location, device) but not the Kaggle PCA features, so XGBoost
        doesn't apply here.
        """
        detection_out = self.aggregator.process_synthetic_transaction(txn, recent_history)
        explanation_out = self.generator.generate(detection_out["triggered_signals"])
        signal_names = [s.get("name") for s in detection_out["triggered_signals"] if s.get("name")]

        return {
            "transaction_id": detection_out["transaction_id"],
            "flag": detection_out["flag"],
            "risk_score": detection_out["risk_score"],
            "confidence": explanation_out["confidence"],
            "confidence_justification": explanation_out["confidence_justification"],
            "explanation": explanation_out["explanation"],
            "triggered_signals": signal_names,
            "triggered_signals_detail": detection_out["triggered_signals"],
            "is_grounded": explanation_out["grounding_passed"],
            "grounding_violations": explanation_out.get("grounding_violations", []),
        }

    def analyze_synthetic_batch(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Run a batch DataFrame of synthetic transactions through the full pipeline."""
        detection_results = self.aggregator.process_synthetic_batch(df)
        results = []

        for d_out in detection_results:
            explanation_out = self.generator.generate(d_out["triggered_signals"])
            signal_names = [s.get("name") for s in d_out["triggered_signals"] if s.get("name")]

            results.append({
                "transaction_id": d_out["transaction_id"],
                "flag": d_out["flag"],
                "risk_score": d_out["risk_score"],
                "confidence": explanation_out["confidence"],
                "confidence_justification": explanation_out["confidence_justification"],
                "explanation": explanation_out["explanation"],
                "triggered_signals": signal_names,
                "triggered_signals_detail": d_out["triggered_signals"],
                "ground_truth_cause": d_out.get("ground_truth_cause", "unknown"),
                "is_grounded": explanation_out["grounding_passed"],
            })

        return results

    def analyze_kaggle_transaction(
        self,
        txn: dict[str, Any],
        transaction_id: str | None = None,
    ) -> dict[str, Any]:
        """Run a single Kaggle transaction through the full pipeline.

        Raises RuntimeError if the classifier hasn't been trained yet.
        Run `python scripts/train_classifier.py` first.
        """
        if not self.classifier_ready:
            raise RuntimeError(
                "XGBoost classifier is not loaded. Train it first:\n"
                "    python scripts/train_classifier.py"
            )

        detection_out = self.aggregator.process_kaggle_transaction(txn, transaction_id=transaction_id)
        explanation_out = self.generator.generate(detection_out["triggered_signals"])
        signal_names = [s.get("name") for s in detection_out["triggered_signals"] if s.get("name")]

        return {
            "transaction_id": detection_out["transaction_id"],
            "flag": detection_out["flag"],
            "risk_score": detection_out["risk_score"],
            "confidence": explanation_out["confidence"],
            "confidence_justification": explanation_out["confidence_justification"],
            "explanation": explanation_out["explanation"],
            "triggered_signals": signal_names,
            "triggered_signals_detail": detection_out["triggered_signals"],
            "is_grounded": explanation_out["grounding_passed"],
            "grounding_violations": explanation_out.get("grounding_violations", []),
        }
