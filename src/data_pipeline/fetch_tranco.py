"""
Build a legitimate-URL list from the Tranco top-sites ranking.

Tranco (https://tranco-list.eu) is a research-oriented, manipulation-resistant
domain popularity ranking, updated daily, free to use. It's a modern, credible
replacement for the "Alexa Rank" that classic phishing-detection datasets
(including the original UCI dataset this project's reference implementation
was based on) used to rely on before Alexa was discontinued in 2022.

IMPORTANT DESIGN NOTE - this script fetches real internal page URLs, not bare
domain roots. An earlier version of this script used "https://{domain}" as
the legitimate URL for every sampled domain. That was a serious, if subtle,
data leakage bug: phishing URLs from OpenPhish are real-world URLs with paths,
query strings, and subdomains, while a bare domain root has none of that by
construction. A model trained on that data doesn't learn "does this look like
phishing" - it trivially learns "does this URL have a path or not," which
perfectly (and meaninglessly) separates the two classes regardless of any
genuine phishing signal. This was caught via a feature-importance sanity
check after training produced suspicious 100% accuracy across every model.

The fix: for each sampled domain, fetch its homepage and extract a handful of
real internal links (actual sub-pages), so legitimate examples have realistic
URL structure comparable to real phishing URLs. Falls back to the bare domain
only if the homepage is unreachable or has no usable internal links.

Usage:
    python fetch_tranco.py --total 5000

Output:
    data/raw/legitimate_urls.csv   (url, label=0, source, tranco_rank, is_fallback_root)
"""

from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from tranco import Tranco

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
USER_AGENT = "Mozilla/5.0 (compatible; PhishingResearchBot/1.0; +https://example.edu)"
REQUEST_TIMEOUT = 6

# (start_rank, end_rank, how many to sample from this band)
# Weighted toward the more "ordinary website" ranges rather than the top 100.
DEFAULT_BANDS = [
    (1, 1_000, 500),
    (1_000, 10_000, 1_500),
    (10_000, 100_000, 2_000),
    (100_000, 1_000_000, 1_000),
]

SKIP_HREF_PREFIXES = ("mailto:", "tel:", "javascript:", "#")


def get_real_internal_url(domain: str) -> tuple[str, bool]:
    """Fetch a domain's homepage and return one real internal page URL.

    Returns (url, is_fallback_root). is_fallback_root=True means the
    homepage was unreachable or had no usable internal links, and we fell
    back to the bare domain root - worth tracking so we can report what
    fraction of "legitimate" examples still have this weaker, root-only shape.
    """
    root_url = f"https://{domain}"
    try:
        resp = requests.get(root_url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return root_url, True

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception:
        return root_url, True

    candidates = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(SKIP_HREF_PREFIXES):
            continue
        full_url = urljoin(resp.url, href)
        parsed = urlparse(full_url)
        if parsed.scheme not in ("http", "https"):
            continue
        link_host = parsed.hostname or ""
        # Keep only internal links (same domain), with an actual non-trivial
        # path - a link to "/" is just the root again, not a real sub-page.
        if domain in link_host and len(parsed.path) > 1:
            candidates.append(full_url)

    if not candidates:
        return root_url, True

    return random.choice(candidates), False


def main():
    parser = argparse.ArgumentParser(description="Build legitimate URL list from Tranco ranking")
    parser.add_argument("--total", type=int, default=None,
                         help="Override total count (splits proportionally across default bands)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Downloading/loading Tranco list (cached after first run)...")
    t = Tranco(cache=True, cache_dir=str(RAW_DIR / ".tranco_cache"))
    tlist = t.list()
    top_1m = tlist.top(1_000_000)
    print(f"Loaded Tranco list with {len(top_1m)} domains.")

    bands = DEFAULT_BANDS
    if args.total:
        total_default = sum(n for _, _, n in DEFAULT_BANDS)
        scale = args.total / total_default
        bands = [(s, e, max(1, int(n * scale))) for s, e, n in DEFAULT_BANDS]

    sampled_domains = []
    for start, end, count in bands:
        band_domains = top_1m[start:end]
        sample_size = min(count, len(band_domains))
        sampled_indices = random.sample(range(len(band_domains)), sample_size)
        for idx in sampled_indices:
            sampled_domains.append((band_domains[idx], start + idx))
        print(f"  Sampled {sample_size} domains from rank band {start}-{end}")

    print(f"\nFetching real internal page URLs for {len(sampled_domains)} domains "
          f"(this makes real network requests - takes a while)...")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(get_real_internal_url, domain): (domain, rank)
            for domain, rank in sampled_domains
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching pages"):
            domain, rank = futures[future]
            url, is_fallback_root = future.result()
            rows.append({
                "url": url,
                "label": 0,
                "source": "tranco",
                "tranco_rank": rank,
                "is_fallback_root": is_fallback_root,
            })

    df = pd.DataFrame(rows).drop_duplicates(subset="url")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "legitimate_urls.csv"
    df.to_csv(out_path, index=False)

    fallback_pct = df["is_fallback_root"].mean()
    print(f"\nSaved {len(df)} unique legitimate URLs -> {out_path}")
    print(f"{fallback_pct:.1%} fell back to bare domain root (homepage unreachable or no internal links found).")
    print("Worth reporting this rate in your CP write-up as a real data-collection limitation.")


if __name__ == "__main__":
    main()
