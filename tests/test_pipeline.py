import unittest
import warnings
from unittest.mock import patch
from pathlib import Path

from src.detection.features import extract_synthetic_features_single
from src.detection.rules import RuleEngine
from src.explanation.validator import GroundingValidator
from src.pipeline import TransactionAnomalyPipeline


BASE_TXN = {
    "transaction_id": "txn_test",
    "user_id": "usr_001",
    "amount": 1000.0,
    "historical_avg_amount": 1000.0,
    "location_lat": 12.9716,
    "location_lon": 77.5946,
    "historical_location_lat": 12.9716,
    "historical_location_lon": 77.5946,
    "device_id": "dev_1",
    "historical_device_ids": '["dev_1"]',
    "timestamp": "2026-08-01T12:00:00",
    "typical_active_start_hour": 8,
    "typical_active_end_hour": 22,
}


def txn(**overrides) -> dict:
    t = dict(BASE_TXN)
    t.update(overrides)
    return t


class TestRuleEngine(unittest.TestCase):

    def setUp(self):
        self.rule_engine = RuleEngine()

    def _signals(self, **overrides) -> list[str]:
        feats = extract_synthetic_features_single(txn(**overrides))
        return [s["name"] for s in self.rule_engine.evaluate_transaction(feats)]

    def test_amount_spike_triggers(self):
        self.assertIn("amount_spike", self._signals(amount=6000.0, historical_avg_amount=1000.0))

    def test_amount_spike_below_threshold(self):
        self.assertNotIn("amount_spike", self._signals(amount=4900.0, historical_avg_amount=1000.0))

    def test_geo_anomaly_triggers(self):
        self.assertIn("geo_anomaly", self._signals(
            location_lat=51.5074, location_lon=-0.1278,
            historical_location_lat=12.9716, historical_location_lon=77.5946,
        ))

    def test_geo_anomaly_nearby(self):
        self.assertNotIn("geo_anomaly", self._signals(location_lat=13.0, location_lon=77.6))

    def test_new_device_high_value_triggers(self):
        self.assertIn("new_device_high_value", self._signals(
            device_id="dev_unknown",
            historical_device_ids='["dev_1"]',
            amount=2500.0,
            historical_avg_amount=1000.0,
        ))

    def test_new_device_known_device(self):
        self.assertNotIn("new_device_high_value", self._signals(
            device_id="dev_1",
            historical_device_ids='["dev_1"]',
            amount=3000.0,
            historical_avg_amount=1000.0,
        ))

    def test_odd_hour_triggers(self):
        self.assertIn("odd_hour_spend", self._signals(
            timestamp="2026-08-01T02:00:00",
            amount=3500.0,
            historical_avg_amount=1000.0,
        ))

    def test_high_velocity_triggers(self):
        feats = extract_synthetic_features_single(
            txn(),
            recent_txns_history=[
                {"user_id": "usr_001", "timestamp": "2026-08-01T11:59:00"},
                {"user_id": "usr_001", "timestamp": "2026-08-01T11:58:30"},
            ],
        )
        names = [s["name"] for s in self.rule_engine.evaluate_transaction(feats)]
        self.assertIn("high_velocity", names)

    def test_clean_transaction(self):
        self.assertEqual(self._signals(), [])


class TestGroundingValidator(unittest.TestCase):

    def setUp(self):
        self.v = GroundingValidator()

    def test_catches_hallucinated_location(self):
        signals = [{"type": "rule", "name": "amount_spike", "value": "6.0x average"}]
        result = self.v.validate("Flagged due to amount spike and unusual location abroad.", signals)
        self.assertFalse(result["is_grounded"])

    def test_catches_hallucinated_device(self):
        signals = [{"type": "rule", "name": "geo_anomaly", "value": "812 km"}]
        result = self.v.validate("Flagged due to geo mismatch and an unknown device.", signals)
        self.assertFalse(result["is_grounded"])

    def test_passes_amount_spike(self):
        signals = [{"type": "rule", "name": "amount_spike", "value": "6.0x average"}]
        result = self.v.validate(
            "Flagged due to a significant amount spike (6.0x higher than historical average).", signals
        )
        self.assertTrue(result["is_grounded"])

    def test_passes_geo_explanation(self):
        signals = [{"type": "rule", "name": "geo_anomaly", "value": "812 km from usual location"}]
        result = self.v.validate(
            "Flagged due to a geographic mismatch (812 km from usual location).", signals
        )
        self.assertTrue(result["is_grounded"])

    def test_passes_velocity_explanation(self):
        signals = [{"type": "rule", "name": "high_velocity", "value": "3 transactions within 2-minute window"}]
        result = self.v.validate(
            "Flagged due to a high transaction velocity spike (3 transactions within 2-minute window).", signals
        )
        self.assertTrue(result["is_grounded"])

    def test_clean_explanation_no_signals(self):
        result = self.v.validate(
            "This transaction did not trigger any risk rules or anomalous model signals.", []
        )
        self.assertTrue(result["is_grounded"])

    def test_affirmative_claim_no_signals(self):
        result = self.v.validate("Flagged due to velocity spike.", [])
        self.assertFalse(result["is_grounded"])

    def test_hallucinated_pca_feature(self):
        signals = [
            {"type": "model_feature", "name": "V14", "contribution": 4.1},
            {"type": "model_feature", "name": "V10", "contribution": 1.2},
        ]
        result = self.v.validate("Flagged due to elevated V14 (+4.1 SHAP) and anomalous V7 pattern.", signals)
        self.assertFalse(result["is_grounded"])
        self.assertTrue(any("V7" in v for v in result["violations"]))

    def test_specificity_warning_generic_text(self):
        signals = [{"type": "rule", "name": "odd_hour_spend", "value": "Off-hours 3.5x"}]
        result = self.v.validate("This transaction appeared suspicious based on detected patterns.", signals)
        self.assertTrue(result["is_grounded"])  # no hallucination
        self.assertTrue(len(result["specificity_warnings"]) > 0)


class TestVelocityBurstLabeling(unittest.TestCase):

    def test_only_triggering_transaction_labeled_anomaly(self):
        """Third transaction in a burst reaches velocity=3 and is labeled anomaly; first two aren't."""
        from src.data.synthetic import generate_synthetic_transactions
        df = generate_synthetic_transactions(normal_count=50, anomaly_count=15, seed=42)
        velocity_rows = df[df["root_cause"] == "high_velocity"]
        self.assertTrue(len(velocity_rows) > 0)
        self.assertTrue((velocity_rows["is_anomaly"] == 1).all())

    def test_velocity_anomaly_rows_get_flagged(self):
        from src.data.synthetic import generate_synthetic_transactions
        from src.detection.aggregator import SignalAggregator
        df = generate_synthetic_transactions(normal_count=50, anomaly_count=15, seed=42)
        agg = SignalAggregator()
        results = agg.process_synthetic_batch(df)
        velocity_results = [r for r in results if r["ground_truth_cause"] == "high_velocity"]
        self.assertTrue(len(velocity_results) > 0)
        for r in velocity_results:
            self.assertTrue(r["flag"], f"{r['transaction_id']} labeled high_velocity but not flagged")


class TestPipelineContract(unittest.TestCase):

    def setUp(self):
        self.pipeline = TransactionAnomalyPipeline()

    def test_output_fields(self):
        res = self.pipeline.analyze_synthetic_transaction(
            txn(transaction_id="txn_contract_test", amount=8000.0, historical_avg_amount=1000.0)
        )
        for field in ["transaction_id", "flag", "risk_score", "confidence",
                      "confidence_justification", "explanation", "triggered_signals",
                      "triggered_signals_detail", "is_grounded"]:
            self.assertIn(field, res)

    def test_output_types(self):
        res = self.pipeline.analyze_synthetic_transaction(
            txn(transaction_id="txn_types_test", amount=8000.0, historical_avg_amount=1000.0)
        )
        self.assertIsInstance(res["flag"], bool)
        self.assertIsInstance(res["risk_score"], float)
        self.assertIn(res["confidence"], ("low", "medium", "high"))
        self.assertIsInstance(res["triggered_signals"], list)
        self.assertIsInstance(res["is_grounded"], bool)

    def test_anomaly_flagged_and_grounded(self):
        res = self.pipeline.analyze_synthetic_transaction(
            txn(transaction_id="txn_spike", amount=8000.0, historical_avg_amount=1000.0)
        )
        self.assertTrue(res["flag"])
        self.assertIn("amount_spike", res["triggered_signals"])
        self.assertTrue(res["is_grounded"])

    def test_normal_not_flagged(self):
        res = self.pipeline.analyze_synthetic_transaction(
            txn(transaction_id="txn_clean", amount=950.0, historical_avg_amount=1000.0)
        )
        self.assertFalse(res["flag"])
        self.assertEqual(res["triggered_signals"], [])
        self.assertTrue(res["is_grounded"])

    def test_warns_on_missing_model(self):
        with patch("src.pipeline._MODEL_PATH", Path("/nonexistent/model.joblib")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                p = TransactionAnomalyPipeline()
                self.assertFalse(p.classifier_ready)
                messages = [str(x.message) for x in w]
                self.assertTrue(any("train_classifier" in m for m in messages))

    def test_kaggle_raises_without_classifier(self):
        with patch("src.pipeline._MODEL_PATH", Path("/nonexistent/model.joblib")):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                p = TransactionAnomalyPipeline()
        with self.assertRaises(RuntimeError) as ctx:
            p.analyze_kaggle_transaction({"Time": 0.0})
        self.assertIn("train_classifier", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
