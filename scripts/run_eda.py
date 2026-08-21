"""
Basic EDA for both datasets — prints stats and confirms splits look right.

Usage:
    python scripts/run_eda.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import get_kaggle_class_distribution, load_kaggle_dataset, split_kaggle_dataset
from src.data.synthetic import generate_synthetic_transactions


def eda_kaggle() -> None:
    print("Kaggle dataset:")
    print("-" * 50)
    df = load_kaggle_dataset()
    dist = get_kaggle_class_distribution(df)
    total = len(df)
    fraud_pct = dist["fraud"] / total * 100

    print(f"  rows    : {total:,}")
    print(f"  columns : {df.shape[1]}")
    print(f"  normal  : {dist['normal']:,}  ({100 - fraud_pct:.3f}%)")
    print(f"  fraud   : {dist['fraud']:,}  ({fraud_pct:.3f}%)")
    print(f"  nulls   : {df.isna().sum().sum()}")
    print(f"  amount  : min={df['Amount'].min():.2f}  mean={df['Amount'].mean():.2f}  max={df['Amount'].max():.2f}")
    print(f"  time span: {df['Time'].max() / 3600:.1f} hours")

    fraud_amt = df[df["Class"] == 1]["Amount"]
    norm_amt = df[df["Class"] == 0]["Amount"]
    print(f"  amount (normal): mean={norm_amt.mean():.2f}  median={norm_amt.median():.2f}")
    print(f"  amount (fraud) : mean={fraud_amt.mean():.2f}  median={fraud_amt.median():.2f}")

    train_df, val_df, test_df = split_kaggle_dataset(df)
    print(f"\n  splits (70/15/15 stratified):")
    print(f"    train : {len(train_df):,}  fraud={train_df['Class'].sum()}  ({train_df['Class'].mean()*100:.3f}%)")
    print(f"    val   : {len(val_df):,}   fraud={val_df['Class'].sum()}  ({val_df['Class'].mean()*100:.3f}%)")
    print(f"    test  : {len(test_df):,}   fraud={test_df['Class'].sum()}  ({test_df['Class'].mean()*100:.3f}%)")


def eda_synthetic() -> None:
    print("\nSynthetic dataset:")
    print("-" * 50)
    df = generate_synthetic_transactions()
    total = len(df)
    anom = (df["is_anomaly"] == 1).sum()
    norm = total - anom

    print(f"  rows     : {total:,}")
    print(f"  normal   : {norm:,}  ({norm/total*100:.1f}%)")
    print(f"  anomalies: {anom:,}  ({anom/total*100:.1f}%)")

    print("\n  ground truth breakdown:")
    for cause, count in df["root_cause"].value_counts().items():
        print(f"    {cause:<28} {count:>5}  ({count/total*100:.1f}%)")

    print("\n  amount by category:")
    for cause in df["root_cause"].unique():
        sub = df[df["root_cause"] == cause]["amount"]
        print(f"    {cause:<28} mean={sub.mean():>8.0f}  median={sub.median():>8.0f}  max={sub.max():>8.0f}")


def main() -> None:
    eda_kaggle()
    eda_synthetic()


if __name__ == "__main__":
    main()
