"""
Load and split the Kaggle Credit Card Fraud dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    KAGGLE_DATA_PATH,
    KAGGLE_EXPECTED_COLUMNS,
    KAGGLE_EXPECTED_ROW_COUNT,
    KAGGLE_FRAUD_CLASS,
    RANDOM_SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)


def load_kaggle_dataset(path: str | None = None) -> pd.DataFrame:
    """Load and validate the Kaggle Credit Card Fraud CSV.

    Raises FileNotFoundError if the file is missing, ValueError if the schema
    doesn't match (wrong columns or row count).
    """
    file_path = path or str(KAGGLE_DATA_PATH)

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'.\n"
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place creditcard.csv in the data/ directory."
        )

    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    actual_cols = list(df.columns)
    if actual_cols != KAGGLE_EXPECTED_COLUMNS:
        missing = set(KAGGLE_EXPECTED_COLUMNS) - set(actual_cols)
        extra = set(actual_cols) - set(KAGGLE_EXPECTED_COLUMNS)
        raise ValueError(
            f"Schema mismatch — missing: {missing or 'none'}, extra: {extra or 'none'}"
        )
    if len(df) != KAGGLE_EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Expected {KAGGLE_EXPECTED_ROW_COUNT:,} rows, got {len(df):,}."
        )


def get_kaggle_class_distribution(df: pd.DataFrame) -> dict[str, int]:
    counts = df["Class"].value_counts()
    return {
        "normal": int(counts.get(0, 0)),
        "fraud": int(counts.get(KAGGLE_FRAUD_CLASS, 0)),
    }


def split_kaggle_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split. Seed is fixed in config.py."""
    holdout_ratio = VALIDATION_RATIO + TEST_RATIO
    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_ratio,
        stratify=df["Class"],
        random_state=RANDOM_SEED,
    )
    relative_test_ratio = TEST_RATIO / holdout_ratio
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test_ratio,
        stratify=holdout_df["Class"],
        random_state=RANDOM_SEED,
    )
    return train_df, val_df, test_df
