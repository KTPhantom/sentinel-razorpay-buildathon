"""
Generate and save the synthetic transaction dataset.

Usage:
    python scripts/generate_synthetic_data.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR
from src.data.synthetic import generate_synthetic_transactions


def main() -> None:
    df = generate_synthetic_transactions()
    out = DATA_DIR / "synthetic_transactions.csv"
    df.to_csv(out, index=False)

    print(f"Saved {len(df):,} rows to {out}")
    print(f"  normal    : {(df['is_anomaly'] == 0).sum():,}")
    print(f"  anomalies : {(df['is_anomaly'] == 1).sum():,}")
    print("\nBreakdown by root_cause:")
    for cause, count in df["root_cause"].value_counts().items():
        print(f"  {cause:<28} {count:>5}")

    print("\nSample rows:")
    print(df[["transaction_id", "user_id", "amount", "merchant_category", "is_anomaly", "root_cause"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
