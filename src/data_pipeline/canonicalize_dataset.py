"""
Apply URL canonicalization to the EXISTING dataset without re-fetching anything.

Only the 20 lexical columns are recomputed (pure string logic, no network), and
only for rows whose URL actually changes under canonicalize_url (bare roots with a
trailing slash). Domain/content columns, labels and row order are untouched, so
every split keeps exactly the same URLs - before/after comparisons are paired.

Files processed (if present) in data/processed/:
    dataset.csv, dataset_checkpoint.csv, train_pre_augmentation.csv,
    validation.csv, test.csv
Originals are copied once to data/processed/pre_canonicalization/ first.

Also prints a consistency check: for rows that canonicalization does NOT change,
the recomputed lexical features must equal the stored ones. Any mismatch means the
dataset was built with a different extractor version - worth knowing.

Place at: src/data_pipeline/canonicalize_dataset.py
Run from project root:  python src\\data_pipeline\\canonicalize_dataset.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.feature_extractor import URLFeatureExtractor, canonicalize_url  # noqa: E402
from features.url_perturbations import LEXICAL_FEATURES  # noqa: E402

FILES = ["dataset.csv", "dataset_checkpoint.csv", "train_pre_augmentation.csv",
         "validation.csv", "test.csv"]


def main(root: Path):
    proc = root / "data" / "processed"
    backup = proc / "pre_canonicalization"
    backup.mkdir(exist_ok=True)
    ex = URLFeatureExtractor(fetch_content=False, fetch_whois=False)
    cache = {}

    def lex(u):
        if u not in cache:
            d = ex.extract(u).to_dict()
            cache[u] = {f: d[f] for f in LEXICAL_FEATURES}
        return cache[u]

    for name in FILES:
        path = proc / name
        if not path.exists():
            print(f"[skip] {name}: not found")
            continue
        df = pd.read_csv(path)
        if "url" not in df.columns:
            print(f"[skip] {name}: no 'url' column")
            continue
        missing = [f for f in LEXICAL_FEATURES if f not in df.columns]
        if missing:
            print(f"[skip] {name}: missing lexical columns {missing[:3]}...")
            continue
        if not (backup / name).exists():
            shutil.copy2(path, backup / name)

        changes = df["url"].map(lambda u: canonicalize_url(u) != u).to_numpy()
        per_feat = {f: 0 for f in LEXICAL_FEATURES}
        drift = {f: 0 for f in LEXICAL_FEATURES}
        for i, (u, ch) in enumerate(zip(df["url"], changes)):
            new = lex(u)
            for f in LEXICAL_FEATURES:
                old = df.at[i, f]
                same = (pd.isna(old) and pd.isna(new[f])) or (
                    not pd.isna(old) and np.isclose(float(old), float(new[f])))
                if ch:
                    if not same:
                        per_feat[f] += 1
                        df.at[i, f] = new[f]
                elif not same:
                    drift[f] += 1  # reported only; NOT overwritten

        df.to_csv(path, index=False)
        lab = ""
        if "label" in df.columns:
            c = df.loc[changes, "label"].value_counts().to_dict()
            lab = f" (phishing {c.get(1, 0)}, legitimate {c.get(0, 0)})"
        print(f"\n[{name}] {len(df)} rows; URLs changed by canonicalization: {int(changes.sum())}{lab}")
        shown = {f: n for f, n in per_feat.items() if n}
        print("   cells updated per feature:", shown if shown else "none")
        bad = {f: n for f, n in drift.items() if n}
        print("   consistency check on unchanged rows:",
              "OK (0 mismatches)" if not bad else f"MISMATCHES {bad}")
    print(f"\nOriginals backed up in {backup}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    main(ap.parse_args().root)
