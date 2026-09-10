"""
Build a legitimate-URL list from the Tranco top-sites ranking.

Tranco (https://tranco-list.eu) is a research-oriented, manipulation-resistant
domain popularity ranking, updated daily, free to use. It's a modern, credible
replacement for the "Alexa Rank" that classic phishing-detection datasets
(including the original UCI dataset this project's reference implementation
was based on) used to rely on before Alexa was discontinued in 2022.

Design note: sampling ONLY the top 100-1000 domains would bias the legitimate
class toward mega-sites (google.com, facebook.com) which don't look like the
small/medium businesses phishing pages usually imitate. We instead sample
across several rank bands for more realistic diversity.

Usage:
    python fetch_tranco.py --total 5000

Output:
    data/raw/legitimate_urls.csv   (url, label=0, source, tranco_rank)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from tranco import Tranco

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# (start_rank, end_rank, how many to sample from this band)
# Weighted toward the more "ordinary website" ranges rather than the top 100.
DEFAULT_BANDS = [
    (1, 1_000, 500),
    (1_000, 10_000, 1_500),
    (10_000, 100_000, 2_000),
    (100_000, 1_000_000, 1_000),
]


def main():
    parser = argparse.ArgumentParser(description="Build legitimate URL list from Tranco ranking")
    parser.add_argument("--total", type=int, default=None,
                         help="Override total count (splits proportionally across default bands)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Downloading/loading Tranco list (cached after first run)...")
    t = Tranco(cache=True, cache_dir=str(RAW_DIR / ".tranco_cache"))
    tlist = t.list()
    top_1m = tlist.top(1_000_000)
    print(f"Loaded Tranco list with {len(top_1m)} domains (list date: {tlist.date if hasattr(tlist, 'date') else 'n/a'}).")

    bands = DEFAULT_BANDS
    if args.total:
        total_default = sum(n for _, _, n in DEFAULT_BANDS)
        scale = args.total / total_default
        bands = [(s, e, max(1, int(n * scale))) for s, e, n in DEFAULT_BANDS]

    rows = []
    for start, end, count in bands:
        band_domains = top_1m[start:end]
        sample_size = min(count, len(band_domains))
        # Sample indices, not values - avoids an O(n) .index() lookup per
        # domain, which would be painfully slow on bands with 100k+ entries.
        sampled_indices = random.sample(range(len(band_domains)), sample_size)
        for idx in sampled_indices:
            domain = band_domains[idx]
            rows.append({
                "url": f"https://{domain}",
                "label": 0,
                "source": "tranco",
                "tranco_rank": start + idx,
            })
        print(f"  Sampled {sample_size} domains from rank band {start}-{end}")

    df = pd.DataFrame(rows).drop_duplicates(subset="url")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "legitimate_urls.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} unique legitimate URLs -> {out_path}")


if __name__ == "__main__":
    main()
