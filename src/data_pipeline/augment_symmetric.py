"""
Symmetric adversarial augmentation (experiment arm C).

The existing augment_adversarial_examples.py adds root-stripped copies of
PHISHING URLs only. That fixed path-stripping evasion (48% -> 1%) but taught the
model that bare roots lean phishing: 3.6% of legitimate URLs were flagged after
path-stripping (0% before). This variant adds root-stripped copies of BOTH
classes, so "bare root" carries no label information.

Optional --add-slash-variants additionally adds a trailing-slash copy of every
legitimate bare-root URL that has none. Use it only if the dataset shows
legitimate roots are written without a slash while phishing roots have one
(a formatting shortcut through path_length / num_slashes).

SAFE BY DESIGN: reads the clean base train_pre_augmentation.csv and writes a NEW
file train_augmented_symmetric.csv. It never touches train.csv, validation.csv
or test.csv. To use it:
    Copy-Item -Force data\\processed\\train_augmented_symmetric.csv data\\processed\\train.csv
    python src\\features\\transformation.py
    python src\\training\\model_trainer.py

Place at: src/data_pipeline/augment_symmetric.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.feature_extractor import URLFeatureExtractor  # noqa: E402
from features.url_perturbations import strip_path, LEXICAL_FEATURES  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
BASE_PATH = PROCESSED_DIR / "train_pre_augmentation.csv"
OUT_PATH = PROCESSED_DIR / "train_augmented_symmetric.csv"


def make_variant(row, new_url, extractor, suffix):
    lex = extractor.extract(new_url).to_dict()
    new_row = row.copy()
    new_row["url"] = new_url
    for feat in LEXICAL_FEATURES:
        new_row[feat] = lex[feat]
    new_row["source"] = str(row.get("source", "")) + suffix
    return new_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-slash-variants", action="store_true")
    args = ap.parse_args()

    if not BASE_PATH.exists():
        raise FileNotFoundError(f"{BASE_PATH} not found.")
    base = pd.read_csv(BASE_PATH)
    if "url" not in base.columns:
        raise ValueError("base training file has no 'url' column.")
    print(f"Base (unaugmented) training set: {len(base)} rows "
          f"({int((base.label == 1).sum())} phishing / {int((base.label == 0).sum())} legitimate)")

    extractor = URLFeatureExtractor(fetch_content=False, fetch_whois=False)
    new_rows, n_strip = [], {0: 0, 1: 0}
    n_skipped = {0: 0, 1: 0}
    n_slash = 0

    for _, row in base.iterrows():
        url, label = row["url"], int(row["label"])
        stripped = strip_path(url)
        if stripped == url:
            n_skipped[label] += 1
        else:
            new_rows.append(make_variant(row, stripped, extractor, "_adversarial_augmented_symmetric"))
            n_strip[label] += 1

        if args.add_slash_variants and label == 0 and urlparse(url).path == "":
            new_rows.append(make_variant(row, url + "/", extractor, "_slash_variant"))
            n_slash += 1

    aug = pd.DataFrame(new_rows)
    combined = pd.concat([base, aug], ignore_index=True)
    combined.to_csv(OUT_PATH, index=False)

    print(f"\nRoot-stripped copies added: {n_strip[1]} phishing, {n_strip[0]} legitimate "
          f"(skipped as already bare: {n_skipped[1]} phishing, {n_skipped[0]} legitimate)")
    if args.add_slash_variants:
        print(f"Trailing-slash copies of legitimate bare roots added: {n_slash}")
    print(f"Training set size: {len(base)} -> {len(combined)} rows")
    pct = combined.label.mean()
    print(f"New label distribution: {int((combined.label == 1).sum())} phishing / "
          f"{int((combined.label == 0).sum())} legitimate ({pct:.1%} phishing)")
    print(f"\nSaved -> {OUT_PATH}\ntrain.csv, validation.csv and test.csv were NOT modified.")


if __name__ == "__main__":
    main()
