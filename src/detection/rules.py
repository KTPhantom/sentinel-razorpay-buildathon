"""
Five deterministic rules for transaction anomaly detection.

Each rule produces a structured signal dict when it fires. The signal list
is the only thing passed downstream to the explanation layer.
"""

from typing import Any

from src.config import (
    RULE_AMOUNT_RATIO,
    RULE_GEO_DISTANCE_KM,
    RULE_NEW_DEVICE_AMOUNT_RATIO,
    RULE_ODD_HOUR_AMOUNT_RATIO,
    RULE_VELOCITY_THRESHOLD,
)


class RuleEngine:

    def __init__(
        self,
        velocity_threshold: int = RULE_VELOCITY_THRESHOLD,
        geo_distance_km: float = RULE_GEO_DISTANCE_KM,
        amount_ratio_threshold: float = RULE_AMOUNT_RATIO,
        new_device_ratio_threshold: float = RULE_NEW_DEVICE_AMOUNT_RATIO,
        odd_hour_ratio_threshold: float = RULE_ODD_HOUR_AMOUNT_RATIO,
    ):
        self.velocity_threshold = velocity_threshold
        self.geo_distance_km = geo_distance_km
        self.amount_ratio_threshold = amount_ratio_threshold
        self.new_device_ratio_threshold = new_device_ratio_threshold
        self.odd_hour_ratio_threshold = odd_hour_ratio_threshold

    def evaluate_transaction(self, txn_features: dict[str, Any]) -> list[dict[str, Any]]:
        """Evaluate all 5 rules and return a list of triggered signal dicts."""
        signals: list[dict[str, Any]] = []

        velocity = txn_features.get("txn_velocity_2min", 1)
        if velocity >= self.velocity_threshold:
            signals.append({
                "type": "rule",
                "name": "high_velocity",
                "value": f"{velocity} transactions within 2-minute window",
            })

        geo_delta = txn_features.get("geo_delta_km", 0.0)
        if geo_delta > self.geo_distance_km:
            signals.append({
                "type": "rule",
                "name": "geo_anomaly",
                "value": f"{geo_delta:.1f} km from usual location",
            })

        amount_ratio = txn_features.get("amount_ratio", 1.0)
        if amount_ratio > self.amount_ratio_threshold:
            signals.append({
                "type": "rule",
                "name": "amount_spike",
                "value": f"{amount_ratio:.1f}x higher than historical average",
            })

        is_new_dev = txn_features.get("is_new_device", False)
        if is_new_dev and amount_ratio > self.new_device_ratio_threshold:
            dev_id = txn_features.get("device_id", "unrecognized")
            signals.append({
                "type": "rule",
                "name": "new_device_high_value",
                "value": f"Unrecognized device '{dev_id}' with {amount_ratio:.1f}x average amount",
            })

        is_odd_hr = txn_features.get("is_odd_hour", False)
        if is_odd_hr and amount_ratio > self.odd_hour_ratio_threshold:
            signals.append({
                "type": "rule",
                "name": "odd_hour_spend",
                "value": f"Off-hours transaction with {amount_ratio:.1f}x average amount",
            })

        return signals
