"""
Split the built dataset into train/validation/test sets.

70/15/15 split, stratified by label so class balance is preserved across
all three sets - important here since the raw dataset is naturally
imbalanced (far more legitimate URLs than phishing ones from the free
data sources used).

The validation set exists specifically so hyperparameter tuning (Phase 2
model training) never touches the test set until the very final, one-time
evaluation. Touching the test set during tuning would silently leak
information and make the final reported metrics overly optimistic - a
subtle mistake that's easy to make and worth explicitly avoiding for a
CP that's being evaluated on ML rigor.

Usage:
    python split_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
INPUT_PATH = PROCESSED_DIR / "dataset.csv"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_STATE = 42


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found - run build_dataset.py first."
        )

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows. Label distribution:\n{df['label'].value_counts()}")

    if df["label"].value_counts().min() < 2:
        raise ValueError(
            "At least one class has fewer than 2 examples - stratified split "
            "needs at least 2 per class. Collect more data before splitting."
        )

    # First split off the test set, then split remaining into train/val.
    # Stratify on both splits so all three sets keep the same class ratio.
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_FRAC, stratify=df["label"], random_state=RANDOM_STATE
    )
    val_relative_frac = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_relative_frac,
        stratify=train_val_df["label"], random_state=RANDOM_STATE,
    )

    for name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        out_path = PROCESSED_DIR / f"{name}.csv"
        split_df.to_csv(out_path, index=False)
        pct_phishing = (split_df["label"] == 1).mean()
        print(f"{name:>10}: {len(split_df):>5} rows  ({pct_phishing:.1%} phishing)  -> {out_path}")


if __name__ == "__main__":
    main()
