"""
Quick smoke test — checks imports, loads the dataset, verifies splits.

Usage:
    python scripts/verify_setup.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies() -> bool:
    print("Checking imports...")
    deps = ["pandas", "numpy", "sklearn", "xgboost", "shap", "anthropic", "dotenv"]
    ok = True
    for dep in deps:
        try:
            __import__(dep)
            print(f"  ok  {dep}")
        except ImportError:
            print(f"  MISSING  {dep}")
            ok = False
    if not ok:
        print("  -> run: pip install -r requirements.txt")
    return ok


def check_kaggle_data() -> bool:
    print("\nLoading Kaggle dataset...")
    from src.data.loader import get_kaggle_class_distribution, load_kaggle_dataset

    try:
        df = load_kaggle_dataset()
    except FileNotFoundError as e:
        print(f"  {e}")
        return False
    except ValueError as e:
        print(f"  schema error: {e}")
        return False

    dist = get_kaggle_class_distribution(df)
    fraud_pct = dist["fraud"] / len(df) * 100
    print(f"  {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  normal={dist['normal']:,}  fraud={dist['fraud']:,}  ({fraud_pct:.3f}%)")
    return True


def check_splits() -> bool:
    print("\nChecking splits...")
    from src.config import RANDOM_SEED, TEST_RATIO, TRAIN_RATIO, VALIDATION_RATIO
    from src.data.loader import load_kaggle_dataset, split_kaggle_dataset

    try:
        df = load_kaggle_dataset()
    except (FileNotFoundError, ValueError):
        print("  skipped — dataset not available")
        return False

    train_df, val_df, test_df = split_kaggle_dataset(df)
    total = len(df)
    print(f"  seed={RANDOM_SEED}  ratios={TRAIN_RATIO}/{VALIDATION_RATIO}/{TEST_RATIO}")
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name} fraud rate: {sdf['Class'].mean()*100:.3f}%")

    ti, vi, sti = set(train_df.index), set(val_df.index), set(test_df.index)
    assert ti.isdisjoint(vi), "train/val overlap!"
    assert ti.isdisjoint(sti), "train/test overlap!"
    assert vi.isdisjoint(sti), "val/test overlap!"
    assert len(ti | vi | sti) == total, "splits don't cover all rows!"
    print("  no overlap, full coverage — ok")
    return True


def main() -> None:
    dep_ok = check_dependencies()
    data_ok = check_kaggle_data()
    split_ok = check_splits()

    print("\nSummary:")
    print(f"  imports : {'ok' if dep_ok else 'FAILED'}")
    print(f"  data    : {'ok' if data_ok else 'FAILED — see data/README.md'}")
    print(f"  splits  : {'ok' if split_ok else 'FAILED'}")

    if not (dep_ok and data_ok and split_ok):
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
