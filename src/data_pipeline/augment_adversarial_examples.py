"""
Adversarial training augmentation: add root-stripped phishing URL variants
directly into the training set, so the model learns that "no path" alone is
not sufficient evidence of legitimacy.

This directly targets a real, measured vulnerability from adversarial_check.py:
stripping a phishing URL's path down to its bare domain root achieved a 100%
evasion rate against the trained model, while switching to HTTPS alone
achieved 0% and removing brand-keyword subdomains was rarely even
applicable. So this augmentation deliberately targets only the proven
vulnerability (path-stripping) rather than blanket-augmenting every
strategy - a narrower, evidence-based fix rather than a shotgun one.

Root cause (for the report): a meaningful fraction of "legitimate" training
URLs are bare-domain-root fallbacks from fetch_tranco.py (when a homepage
had no crawlable internal links), so the model partly learned "no path =
legitimate" as a shortcut. This augmentation counteracts that shortcut by
explicitly showing the model root-shaped PHISHING examples too.

IMPORTANT: only train.csv is modified. validation.csv and test.csv are left
completely untouched, so evaluation always reflects genuine generalization
to data the augmentation step never influenced.

Run this AFTER split_dataset.py and BEFORE transformation.py:
    python split_dataset.py
    python augment_adversarial_examples.py
    python transformation.py
    python model_trainer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.feature_extractor import URLFeatureExtractor  # noqa: E402
from features.url_perturbations import strip_path, LEXICAL_FEATURES  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
BACKUP_PATH = PROCESSED_DIR / "train_pre_augmentation.csv"


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"{TRAIN_PATH} not found - run split_dataset.py first.")

    train_df = pd.read_csv(TRAIN_PATH)
    if "url" not in train_df.columns:
        raise ValueError("train.csv is missing the 'url' column - can't generate perturbed variants.")

    # Keep an explicit, honest backup of the pre-augmentation training set -
    # nothing here is silently destroyed, and this file lets you reproduce
    # or inspect exactly what augmentation changed.
    train_df.to_csv(BACKUP_PATH, index=False)
    print(f"Backed up original training set -> {BACKUP_PATH}")

    extractor = URLFeatureExtractor(fetch_content=False, fetch_whois=False)
    phishing_rows = train_df[train_df["label"] == 1]

    new_rows = []
    skipped = 0
    for _, row in phishing_rows.iterrows():
        url = row["url"]
        stripped_url = strip_path(url)
        if stripped_url == url:
            skipped += 1
            continue  # already a bare root - nothing new to add

        perturbed_lexical = extractor.extract(stripped_url).to_dict()
        new_row = row.copy()
        new_row["url"] = stripped_url
        for feat in LEXICAL_FEATURES:
            new_row[feat] = perturbed_lexical[feat]
        new_row["source"] = str(row.get("source", "")) + "_adversarial_augmented"
        new_rows.append(new_row)

    if not new_rows:
        print("No new adversarial examples generated (all phishing URLs were already bare roots).")
        return

    augmented_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([train_df, augmented_df], ignore_index=True)
    combined_df.to_csv(TRAIN_PATH, index=False)

    print(f"\nGenerated {len(augmented_df)} root-stripped adversarial phishing examples "
          f"({skipped} phishing URLs were already bare roots and skipped).")
    print(f"Training set size: {len(train_df)} -> {len(combined_df)} rows")
    print(f"New label distribution:\n{combined_df['label'].value_counts()}")
    print(f"\nSaved augmented training set -> {TRAIN_PATH}")
    print("Remember: validation.csv and test.csv are UNCHANGED - re-run transformation.py "
          "and model_trainer.py next, then re-run adversarial_check.py to confirm the "
          "strip_path evasion rate actually improved.")


if __name__ == "__main__":
    main()
