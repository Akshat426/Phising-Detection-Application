"""
Build the final labeled training dataset by running feature extraction over
every URL collected from PhishTank (label=1) and Tranco (label=0).

This is the slowest script in the pipeline - it makes real network calls
(page fetch + WHOIS) for every single URL, so for thousands of URLs this can
take a long time and WILL have partial failures (dead sites, WHOIS rate
limits, timeouts). Design choices to handle that honestly:

  - Runs extraction concurrently (ThreadPoolExecutor) since these are I/O-bound
    network calls, not CPU-bound work.
  - Checkpoints progress every N URLs so a crash or Ctrl+C doesn't lose hours
    of work - re-running the script skips URLs already in the checkpoint file.
  - Missing features (failed WHOIS/page fetch) are kept as real NaN values,
    not silently filled with fake defaults. The transformation stage's
    imputer is responsible for handling them properly and honestly.

Usage:
    python build_dataset.py --workers 16 --checkpoint-every 100
    python build_dataset.py --sample 200   # quick test run on a small subset
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.feature_extractor import URLFeatureExtractor, load_tranco_list  # noqa: E402

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CHECKPOINT_PATH = PROCESSED_DIR / "dataset_checkpoint.csv"
FINAL_PATH = PROCESSED_DIR / "dataset.csv"


def extract_one(extractor: URLFeatureExtractor, url: str, label: int) -> dict:
    try:
        feats = extractor.extract(url).to_dict()
    except Exception as e:
        # Extraction itself should rarely throw (internal calls are already
        # wrapped in try/except), but guard the outer call too so one bad
        # URL never kills the whole batch job.
        feats = {"extraction_error": str(e)}
    feats["url"] = url
    feats["label"] = label
    return feats


def main():
    parser = argparse.ArgumentParser(description="Build labeled dataset via feature extraction")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent threads for network calls")
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--sample", type=int, default=None,
                         help="Only process this many URLs total (for a quick test run)")
    parser.add_argument("--no-content", action="store_true", help="Skip live page fetch (faster, fewer features)")
    parser.add_argument("--no-whois", action="store_true", help="Skip WHOIS lookups (faster, fewer features)")
    args = parser.parse_args()

    phishing_path = RAW_DIR / "phishing_urls.csv"
    legit_path = RAW_DIR / "legitimate_urls.csv"
    if not phishing_path.exists() or not legit_path.exists():
        print("ERROR: Run fetch_phishtank.py and fetch_tranco.py first.", file=sys.stderr)
        sys.exit(1)

    phishing_df = pd.read_csv(phishing_path)
    legit_df = pd.read_csv(legit_path)
    all_urls = list(zip(phishing_df["url"], [1] * len(phishing_df))) + \
               list(zip(legit_df["url"], [0] * len(legit_df)))

    if args.sample:
        import random
        random.seed(42)
        random.shuffle(all_urls)
        all_urls = all_urls[: args.sample]

    print(f"Total URLs to process: {len(all_urls)}")

    # Resume from checkpoint if one exists
    already_done = set()
    results = []
    if CHECKPOINT_PATH.exists():
        checkpoint_df = pd.read_csv(CHECKPOINT_PATH)
        already_done = set(checkpoint_df["url"])
        results = checkpoint_df.to_dict("records")
        print(f"Resuming - {len(already_done)} URLs already processed in checkpoint.")

    remaining = [(u, l) for u, l in all_urls if u not in already_done]
    print(f"{len(remaining)} URLs remaining to process.")

    tranco_list = None
    if not args.no_whois:
        try:
            print("Loading Tranco list for rank lookups...")
            tranco_list = load_tranco_list()
        except Exception as e:
            print(f"Warning: could not load Tranco list for rank lookups ({e}). Continuing without it.")

    extractor = URLFeatureExtractor(
        tranco_list=tranco_list,
        fetch_content=not args.no_content,
        fetch_whois=not args.no_whois,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_since_checkpoint = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_one, extractor, url, label): url for url, label in remaining}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
            results.append(future.result())
            processed_since_checkpoint += 1
            if processed_since_checkpoint >= args.checkpoint_every:
                pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
                processed_since_checkpoint = 0

    final_df = pd.DataFrame(results)
    final_df.to_csv(FINAL_PATH, index=False)
    print(f"\nDone. Saved {len(final_df)} rows -> {FINAL_PATH}")
    print(f"Label distribution:\n{final_df['label'].value_counts()}")

    # Report missing-ness grouped by feature category, not just "top 10" -
    # a plain top-10-by-missing-% list silently hides domain/WHOIS features
    # whenever 10+ content features are tied at exactly the same missing
    # rate (e.g. all 100% missing when --no-content is passed), which is
    # exactly the confusing situation this project hit during testing.
    feature_groups = {
        "domain (WHOIS/DNS/Tranco)": [
            "domain_age_days", "domain_registration_length_days",
            "dns_resolves", "tranco_rank",
        ],
        "content (live page fetch)": [
            "num_forms", "form_action_external_ratio", "num_iframes",
            "num_external_links", "num_internal_links", "external_link_ratio",
            "has_password_field", "favicon_external", "num_scripts",
            "title_hostname_overlap", "response_status_code",
        ],
    }
    print("\nMissing-value rate by feature category:")
    for group_name, cols in feature_groups.items():
        present_cols = [c for c in cols if c in final_df.columns]
        if not present_cols:
            continue
        group_missing = final_df[present_cols].isna().mean().sort_values(ascending=False)
        print(f"\n  {group_name}:")
        for col, pct in group_missing.items():
            print(f"    {col:<35} {pct:.1%}")


if __name__ == "__main__":
    main()
