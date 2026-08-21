"""
Run the full evaluation and write results to evaluation_results.json.

Two parts:
1. Detection metrics (precision/recall/F1) on the held-out Kaggle test split.
2. Explanation quality on a 20-transaction hand-graded sample, using the rubric
   from the product spec: Correct=1.0, Partial=0.5, Incorrect=0.0.
   Kaggle fraud cases score Correct only if the explanation text explicitly
   names the top SHAP feature.

Usage:
    python scripts/run_evaluation.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_kaggle_dataset, split_kaggle_dataset
from src.data.synthetic import generate_synthetic_transactions
from src.pipeline import TransactionAnomalyPipeline


def evaluate_detection(pipeline: TransactionAnomalyPipeline, test_df: pd.DataFrame) -> dict:
    print("=" * 70)
    print("Detection evaluation — Kaggle held-out test set")
    print("=" * 70)
    print(f"rows: {len(test_df):,}   fraud: {test_df['Class'].sum()} ({test_df['Class'].mean()*100:.3f}%)")

    metrics = pipeline.classifier.evaluate(test_df)

    print(f"\nthreshold  {metrics['threshold']}")
    print(f"precision  {metrics['precision']:.4f}")
    print(f"recall     {metrics['recall']:.4f}")
    print(f"F1         {metrics['f1']:.4f}")

    report = metrics["classification_report"]
    print(f"\n{'':12} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    print("-" * 50)
    for cls_name in ["Normal", "Fraud"]:
        s = report[cls_name]
        print(f"{cls_name:<12} {s['precision']:>10.4f} {s['recall']:>10.4f} {s['f1-score']:>10.4f} {s['support']:>10,.0f}")
    print("-" * 50)
    ma = report["macro avg"]
    wa = report["weighted avg"]
    print(f"{'macro avg':<12} {ma['precision']:>10.4f} {ma['recall']:>10.4f} {ma['f1-score']:>10.4f} {ma['support']:>10,.0f}")
    print(f"{'weighted avg':<12} {wa['precision']:>10.4f} {wa['recall']:>10.4f} {wa['f1-score']:>10.4f} {wa['support']:>10,.0f}")
    print("=" * 70)
    return metrics


def grade_explanations(pipeline: TransactionAnomalyPipeline, test_df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("Explanation quality — 20-transaction graded sample")
    print("=" * 70)
    print("Rubric: Correct=1.0 (names root cause / top SHAP feature)")
    print("        Partial=0.5 (grounded but misses primary driver)")
    print("        Incorrect=0.0 (hallucination, not flagged, or too generic)")
    print("-" * 70)

    syn_df = generate_synthetic_transactions()
    syn_results = pipeline.analyze_synthetic_batch(syn_df)

    graded = []
    total_score = 0.0
    grounding_passes = 0

    # 15 synthetic samples — 3 normal + 2 or 3 of each anomaly type
    synthetic_spec = [
        ("normal", 3),
        ("geo_anomaly", 3),
        ("amount_spike", 3),
        ("new_device_high_value", 2),
        ("odd_hour_spend", 2),
        ("high_velocity", 2),
    ]

    for cause, count in synthetic_spec:
        if cause == "normal":
            matches = [r for r in syn_results if r["ground_truth_cause"] == cause][:count]
        else:
            matches = [r for r in syn_results if r["ground_truth_cause"] == cause and r["flag"]][:count]
            if len(matches) < count:
                matches += [r for r in syn_results if r["ground_truth_cause"] == cause and not r["flag"]][:(count - len(matches))]

        for item in matches:
            explanation = item["explanation"]
            signals = item["triggered_signals"]
            is_grounded = item["is_grounded"]
            if is_grounded:
                grounding_passes += 1

            gt = item["ground_truth_cause"]
            if gt == "normal":
                score = 1.0 if (not item["flag"] and "no" in explanation.lower()) else 0.0
                grade = "Correct (1.0)" if score == 1.0 else "Incorrect (0.0)"
            else:
                if gt in signals and is_grounded:
                    score, grade = 1.0, "Correct (1.0)"
                elif is_grounded and signals:
                    score, grade = 0.5, "Partial (0.5)"
                else:
                    score, grade = 0.0, "Incorrect (0.0)"

            total_score += score
            graded.append({
                "id": item["transaction_id"],
                "dataset": "Synthetic",
                "ground_truth": gt,
                "flagged": item["flag"],
                "top_shap_feature": "n/a (rule-based)",
                "signals": signals,
                "confidence": item["confidence"],
                "explanation": explanation,
                "is_grounded": is_grounded,
                "score": score,
                "grade": grade,
            })

    # 5 Kaggle samples — 3 fraud, 2 normal
    kaggle_fraud = test_df[test_df["Class"] == 1].head(3)
    kaggle_normal = test_df[test_df["Class"] == 0].head(2)

    for idx, (_, row) in enumerate(kaggle_fraud.iterrows(), 1):
        item = pipeline.analyze_kaggle_transaction(row.to_dict(), transaction_id=f"kaggle_eval_fraud_{idx:02d}")
        explanation = item["explanation"]
        signals = item["triggered_signals"]
        is_grounded = item["is_grounded"]
        if is_grounded:
            grounding_passes += 1

        top_feature = signals[0].lower() if signals else ""
        if item["flag"] and is_grounded and top_feature and top_feature in explanation.lower():
            score, grade = 1.0, "Correct (1.0)"
        elif item["flag"] and is_grounded and signals:
            score, grade = 0.5, "Partial (0.5) — grounded but top feature not named"
        else:
            score, grade = 0.0, "Incorrect (0.0)"

        total_score += score
        graded.append({
            "id": item["transaction_id"],
            "dataset": "Kaggle",
            "ground_truth": "Fraud (1)",
            "flagged": item["flag"],
            "top_shap_feature": signals[0] if signals else "none",
            "signals": signals,
            "confidence": item["confidence"],
            "explanation": explanation,
            "is_grounded": is_grounded,
            "score": score,
            "grade": grade,
        })

    for idx, (_, row) in enumerate(kaggle_normal.iterrows(), 1):
        item = pipeline.analyze_kaggle_transaction(row.to_dict(), transaction_id=f"kaggle_eval_normal_{idx:02d}")
        explanation = item["explanation"]
        signals = item["triggered_signals"]
        is_grounded = item["is_grounded"]
        if is_grounded:
            grounding_passes += 1

        if not item["flag"] and is_grounded:
            score, grade = 1.0, "Correct (1.0)"
        elif item["flag"]:
            score, grade = 0.0, "Incorrect (0.0) — false positive"
        else:
            score, grade = 0.0, "Incorrect (0.0)"

        total_score += score
        graded.append({
            "id": item["transaction_id"],
            "dataset": "Kaggle",
            "ground_truth": "Normal (0)",
            "flagged": item["flag"],
            "top_shap_feature": signals[0] if signals else "none",
            "signals": signals,
            "confidence": item["confidence"],
            "explanation": explanation,
            "is_grounded": is_grounded,
            "score": score,
            "grade": grade,
        })

    w = 126
    print(f"\n{'#':<3} {'ID':<26} {'Dataset':<10} {'Ground Truth':<22} {'Top Feat':<10} {'Score':<6} {'Grade':<38} {'Grounded'}")
    print("-" * w)
    for i, g in enumerate(graded, 1):
        top_f = g.get("top_shap_feature", "n/a")[:9]
        grounded_str = "Yes" if g["is_grounded"] else "No"
        print(f"{i:<3} {g['id']:<26} {g['dataset']:<10} {g['ground_truth']:<22} {top_f:<10} {g['score']:<6.1f} {g['grade']:<38} {grounded_str}")

    accuracy_pct = (total_score / len(graded)) * 100
    grounding_pct = (grounding_passes / len(graded)) * 100

    print("-" * w)
    print(f"sample size        : {len(graded)}")
    print(f"total score        : {total_score:.1f} / {len(graded)}")
    print(f"explanation accuracy : {accuracy_pct:.1f}%")
    print(f"grounding compliance : {grounding_pct:.1f}%")
    print("=" * 70)

    return {
        "sample_size": len(graded),
        "total_score": total_score,
        "explanation_accuracy": accuracy_pct,
        "grounding_compliance": grounding_pct,
        "graded_items": graded,
    }


def main() -> None:
    pipeline = TransactionAnomalyPipeline()
    kaggle_df = load_kaggle_dataset()
    _, _, test_df = split_kaggle_dataset(kaggle_df)

    detection = evaluate_detection(pipeline, test_df)
    explanation = grade_explanations(pipeline, test_df)

    out_path = PROJECT_ROOT / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "detection_metrics": {
                "dataset": "Kaggle Credit Card Fraud (held-out test set)",
                "test_rows": len(test_df),
                "threshold": detection["threshold"],
                "precision": detection["precision"],
                "recall": detection["recall"],
                "f1": detection["f1"],
            },
            "explanation_metrics": {
                "sample_size": explanation["sample_size"],
                "explanation_accuracy_pct": explanation["explanation_accuracy"],
                "grounding_compliance_pct": explanation["grounding_compliance"],
                "graded_sample": explanation["graded_items"],
            },
        }, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
