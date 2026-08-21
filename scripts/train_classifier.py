"""
Train the XGBoost classifier and save the model artifact.

Loads the Kaggle dataset, does a 70/15/15 stratified split, trains with
scale_pos_weight to handle class imbalance, tunes the decision threshold
on the validation set, then saves the model to models/.

Usage:
    python scripts/train_classifier.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_kaggle_dataset, split_kaggle_dataset
from src.detection.classifier import FraudClassifier


def main() -> None:
    print("Loading dataset...")
    df = load_kaggle_dataset()
    train_df, val_df, test_df = split_kaggle_dataset(df)
    print(f"  train {len(train_df):,}  val {len(val_df):,}  test {len(test_df):,}")
    print(f"  fraud in train: {train_df['Class'].sum()}  val: {val_df['Class'].sum()}  test: {test_df['Class'].sum()}")

    print("\nTraining...")
    classifier = FraudClassifier.train(train_df, val_df)

    audit = getattr(classifier, "val_threshold_audit", None)
    print(f"  threshold  : {classifier.optimal_threshold}")
    if audit:
        print(f"  val F1     : {audit['f1']:.4f}")
        print(f"  val P/R    : {audit['precision']:.4f} / {audit['recall']:.4f}")
        print("  (threshold chosen by max-F1 grid sweep 0.05→0.95 on val split)")

    print("\nVal set evaluation:")
    val_results = classifier.evaluate(val_df)
    print(f"  P {val_results['precision']:.4f}  R {val_results['recall']:.4f}  F1 {val_results['f1']:.4f}")

    saved_path = classifier.save()
    print(f"\nSaved to {saved_path}")

    print("\nSanity check — one fraud sample from test set:")
    sample = test_df[test_df["Class"] == 1].iloc[0]
    result = classifier.predict_single(sample)
    print(f"  true label : Fraud")
    print(f"  risk score : {result['risk_score']}  flagged: {result['flag']}")
    for feat in result["top_shap_features"]:
        print(f"  {feat['name']}  val={feat['value']}  shap={feat['contribution']:+.4f}")


if __name__ == "__main__":
    main()
