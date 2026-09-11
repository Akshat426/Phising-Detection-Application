"""
Data transformation: missing-value indicators, imputation, and scaling.

Design decisions worth documenting in the CP report:

1. Missing-value indicator columns. Rather than only imputing missing
   values silently, we add a binary "<feature>_was_missing" column for any
   feature with non-trivial missingness in the training set. This is
   deliberate: a failed WHOIS lookup or unreachable page is not random
   noise here - it can itself be a meaningful signal (e.g. domains that
   actively evade automated inspection, expired/hidden WHOIS records).
   Throwing that information away during imputation would discard a
   potentially real predictive signal.

2. Median imputation (not KNN). The reference project used KNNImputer.
   We use per-column median imputation here instead: it's simpler, faster,
   more transparent to explain in a viva, and - combined with the missing
   indicators above - the model can still learn from "value was imputed"
   directly rather than relying on KNN to guess a plausible value from
   correlated features. This is a legitimate, defensible alternative choice,
   not a downgrade; if useful, a KNN-imputation ablation can be added later
   as a comparison experiment.

3. Fit only on train. The imputer and scaler are fit exclusively on the
   training set and then applied (not re-fit) to validation and test data.
   Fitting on the full dataset before splitting is a common and subtle
   source of data leakage - the transformation parameters would otherwise
   have "seen" information from data the model is supposed to be evaluated
   on as if it were unseen.

Usage:
    python transformation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "transform_artifacts"

NON_FEATURE_COLUMNS = {"url", "label", "source", "verified_time", "fetched_time", "extraction_error"}
MISSING_INDICATOR_THRESHOLD = 0.01  # add an indicator if >1% of training rows are missing that feature


class DataTransformer:
    """Fit on training data, then apply the same fitted transformation to
    validation/test/live-prediction data. Save/load via joblib so the
    exact same transformation is reused at serving time (Phase 5), not
    re-derived and risking a train/serve mismatch.
    """

    def __init__(self):
        self.feature_columns: list[str] = []
        self.indicator_columns: list[str] = []
        self.imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | None = None

    def fit(self, train_df: pd.DataFrame) -> "DataTransformer":
        self.feature_columns = [c for c in train_df.columns if c not in NON_FEATURE_COLUMNS]
        missing_rate = train_df[self.feature_columns].isna().mean()
        self.indicator_columns = list(missing_rate[missing_rate > MISSING_INDICATOR_THRESHOLD].index)

        self.imputer = SimpleImputer(strategy="median")
        self.imputer.fit(train_df[self.feature_columns])

        imputed = self.imputer.transform(train_df[self.feature_columns])
        self.scaler = StandardScaler()
        self.scaler.fit(imputed)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.imputer is None or self.scaler is None:
            raise RuntimeError("Call fit() before transform(), or load() a fitted transformer.")

        result = pd.DataFrame(index=df.index)

        # Missing indicators computed BEFORE imputation fills the values in
        for col in self.indicator_columns:
            result[f"{col}_was_missing"] = df[col].isna().astype(int) if col in df.columns else 1

        # Some feature columns might not exist yet on brand-new live-prediction
        # input (e.g. a column added later) - reindex to the fitted column
        # order and let the imputer's own missing-value handling take it from
        # there, rather than crashing on a KeyError.
        aligned = df.reindex(columns=self.feature_columns)
        imputed = self.imputer.transform(aligned)
        scaled = self.scaler.transform(imputed)

        scaled_df = pd.DataFrame(scaled, columns=self.feature_columns, index=df.index)
        result = pd.concat([scaled_df, result], axis=1)

        if "label" in df.columns:
            result["label"] = df["label"].values

        return result

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.imputer, path / "imputer.pkl")
        joblib.dump(self.scaler, path / "scaler.pkl")
        with open(path / "metadata.json", "w") as f:
            json.dump({
                "feature_columns": self.feature_columns,
                "indicator_columns": self.indicator_columns,
            }, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DataTransformer":
        obj = cls()
        obj.imputer = joblib.load(path / "imputer.pkl")
        obj.scaler = joblib.load(path / "scaler.pkl")
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        obj.feature_columns = meta["feature_columns"]
        obj.indicator_columns = meta["indicator_columns"]
        return obj


def main():
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "validation.csv"
    test_path = PROCESSED_DIR / "test.csv"
    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found - run split_dataset.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    transformer = DataTransformer().fit(train_df)
    print(f"Fitted on {len(transformer.feature_columns)} feature columns.")
    print(f"Added missing-indicator columns for: {transformer.indicator_columns}")

    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        transformed = transformer.transform(df)
        out_path = PROCESSED_DIR / f"{name}_transformed.csv"
        transformed.to_csv(out_path, index=False)
        print(f"  {name}: {transformed.shape} -> {out_path}")

    transformer.save(ARTIFACTS_DIR)
    print(f"\nSaved fitted imputer/scaler -> {ARTIFACTS_DIR}")
    print("(Reused unchanged at serving time in Phase 5 - never re-fit on live prediction data.)")


if __name__ == "__main__":
    main()
