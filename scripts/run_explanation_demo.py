"""
End-to-end pipeline demo — runs detection through to grounded explanation
for a sample of synthetic and Kaggle transactions.

Usage:
    python scripts/run_explanation_demo.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_kaggle_dataset, split_kaggle_dataset
from src.data.synthetic import generate_synthetic_transactions
from src.pipeline import TransactionAnomalyPipeline


def print_result(r: dict) -> None:
    print(f"  id         : {r['transaction_id']}")
    print(f"  flag       : {r['flag']}  score={r['risk_score']}  confidence={r['confidence']}")
    print(f"  explanation: {r['explanation']}")
    print(f"  signals    : {r['triggered_signals']}")
    print(f"  grounded   : {r['is_grounded']}")


def main() -> None:
    pipeline = TransactionAnomalyPipeline()

    print("Synthetic transactions:")
    print("=" * 60)
    syn_df = generate_synthetic_transactions()
    batch = pipeline.analyze_synthetic_batch(syn_df)

    demo_cases = [
        ("normal", "clean transaction"),
        ("geo_anomaly", "geo anomaly"),
        ("amount_spike", "amount spike"),
        ("new_device_high_value", "new device"),
        ("odd_hour_spend", "odd hour"),
        ("high_velocity", "velocity burst"),
    ]

    total = 0
    grounded = 0
    for cause, label in demo_cases:
        if cause == "normal":
            r = next(x for x in batch if x["ground_truth_cause"] == cause)
        else:
            r = next((x for x in batch if x["ground_truth_cause"] == cause and x["flag"]), None)
            if r is None:
                r = next(x for x in batch if x["ground_truth_cause"] == cause)
        total += 1
        if r["is_grounded"]:
            grounded += 1
        print(f"\n[{label}]")
        print_result(r)

    print("\nKaggle transactions:")
    print("=" * 60)
    kaggle_df = load_kaggle_dataset()
    _, _, test_df = split_kaggle_dataset(kaggle_df)

    fraud_rows = test_df[test_df["Class"] == 1].head(2)
    normal_row = test_df[test_df["Class"] == 0].iloc[0]

    for i, (_, row) in enumerate(fraud_rows.iterrows(), 1):
        r = pipeline.analyze_kaggle_transaction(row.to_dict(), transaction_id=f"kaggle_fraud_{i:03d}")
        total += 1
        if r["is_grounded"]:
            grounded += 1
        print(f"\n[Kaggle fraud #{i}]")
        print_result(r)

    r = pipeline.analyze_kaggle_transaction(normal_row.to_dict(), transaction_id="kaggle_normal_001")
    total += 1
    if r["is_grounded"]:
        grounded += 1
    print("\n[Kaggle normal]")
    print_result(r)

    print(f"\nGrounding: {grounded}/{total} ({grounded/total*100:.0f}%)")


if __name__ == "__main__":
    main()
