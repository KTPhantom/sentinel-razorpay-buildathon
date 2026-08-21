"""
Quick demo of the detection layer — shows triggered_signals output for
one example of each anomaly category and two Kaggle transactions.

Usage:
    python scripts/run_detection.py
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_kaggle_dataset, split_kaggle_dataset
from src.data.synthetic import generate_synthetic_transactions
from src.detection.aggregator import SignalAggregator
from src.detection.classifier import FraudClassifier


def main() -> None:
    print("Loading classifier...")
    try:
        classifier = FraudClassifier.load()
    except FileNotFoundError:
        print("Model not found — run scripts/train_classifier.py first.")
        return

    aggregator = SignalAggregator(classifier=classifier)

    # Synthetic — one example per category
    print("\nSynthetic transactions (rule engine):")
    print("-" * 60)
    syn_df = generate_synthetic_transactions()
    batch = aggregator.process_synthetic_batch(syn_df)

    for cat in ["normal", "geo_anomaly", "amount_spike", "new_device_high_value", "odd_hour_spend", "high_velocity"]:
        if cat == "normal":
            match = next(r for r in batch if r["ground_truth_cause"] == cat)
        else:
            match = next((r for r in batch if r["ground_truth_cause"] == cat and r["flag"]), None)
            if match is None:
                match = next(r for r in batch if r["ground_truth_cause"] == cat)

        print(f"\n[{cat}]  {match['transaction_id']}  flagged={match['flag']}  score={match['risk_score']}")
        print(json.dumps(match["triggered_signals"], indent=2))

    # Kaggle — one normal, one fraud
    print("\nKaggle transactions (XGBoost + SHAP):")
    print("-" * 60)
    kaggle_df = load_kaggle_dataset()
    _, _, test_df = split_kaggle_dataset(kaggle_df)

    for label, txn_dict in [
        ("Normal (0)", test_df[test_df["Class"] == 0].iloc[0].to_dict()),
        ("Fraud (1)", test_df[test_df["Class"] == 1].iloc[0].to_dict()),
    ]:
        out = aggregator.process_kaggle_transaction(txn_dict, transaction_id=f"kaggle_{label.split()[0].lower()}_demo")
        print(f"\n[{label}]  {out['transaction_id']}  flagged={out['flag']}  score={out['risk_score']}")
        print(json.dumps(out["triggered_signals"], indent=2))


if __name__ == "__main__":
    main()
