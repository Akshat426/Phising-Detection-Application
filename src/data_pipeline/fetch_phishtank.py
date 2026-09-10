"""
Fetch verified phishing URLs from PhishTank.

PhishTank requires a free registered API key (anti-abuse measure) - get one at
https://www.phishtank.com/api_register.php after creating a free account.

Usage:
    python fetch_phishtank.py --limit 5000

Output:
    data/raw/phishtank_raw.json   (full raw feed, kept for provenance/audit)
    data/raw/phishing_urls.csv    (url, label=1, source, verified_time)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
FEED_URL_TEMPLATE = "https://data.phishtank.com/data/{api_key}/online-valid.json"
USER_AGENT = "phishing-research-cp/1.0"


def fetch_feed(api_key: str) -> list[dict]:
    url = FEED_URL_TEMPLATE.format(api_key=api_key)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Fetch verified phishing URLs from PhishTank")
    parser.add_argument("--limit", type=int, default=None, help="Max number of URLs to keep (default: all)")
    parser.add_argument("--verified-only", action="store_true", default=True,
                         help="Keep only PhishTank-verified entries (default: True)")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("PHISHTANK_API_KEY")
    if not api_key:
        print("ERROR: Set PHISHTANK_API_KEY in your .env file.", file=sys.stderr)
        print("Register a free key at https://www.phishtank.com/api_register.php", file=sys.stderr)
        sys.exit(1)

    print("Fetching PhishTank feed (this can take a minute, the feed is large)...")
    entries = fetch_feed(api_key)
    print(f"Fetched {len(entries)} total entries.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / "phishtank_raw.json", "w") as f:
        json.dump(entries, f)

    if args.verified_only:
        entries = [e for e in entries if e.get("verified") == "yes"]
        print(f"{len(entries)} entries are PhishTank-verified.")

    if args.limit:
        entries = entries[: args.limit]

    rows = [
        {
            "url": e["url"],
            "label": 1,
            "source": "phishtank",
            "verified_time": e.get("verification_time"),
        }
        for e in entries
    ]
    df = pd.DataFrame(rows).drop_duplicates(subset="url")
    out_path = RAW_DIR / "phishing_urls.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} unique phishing URLs -> {out_path}")


if __name__ == "__main__":
    main()
