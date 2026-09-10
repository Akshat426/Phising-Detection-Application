"""
Fetch active phishing URLs from OpenPhish's free public feed.

Used as the primary phishing-URL source instead of PhishTank, since PhishTank
has temporarily disabled new API key registrations. OpenPhish's free feed
needs no signup: it's a plain list of currently active phishing URLs, updated
periodically (roughly every few hours), community-established and widely
cited in phishing-detection research.

Tradeoff vs PhishTank: less metadata (no submission time, no "verified by
volunteers" flag - OpenPhish already only lists URLs its own detection
system considers confirmed phishing, so there's no separate verification
step to expose). Worth a one-line mention in your report's data section.

Usage:
    python fetch_openphish.py

Output:
    data/raw/openphish_raw.txt     (raw feed, kept for provenance/audit)
    data/raw/phishing_urls.csv     (url, label=1, source, fetched_time)
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
FEED_URL = "https://openphish.com/feed.txt"
USER_AGENT = "phishing-research-cp/1.0"


def fetch_feed() -> list[str]:
    resp = requests.get(FEED_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    urls = [line.strip() for line in resp.text.splitlines() if line.strip()]
    return urls


def main():
    print("Fetching OpenPhish feed...")
    try:
        urls = fetch_feed()
    except requests.RequestException as e:
        print(f"ERROR: could not fetch OpenPhish feed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Fetched {len(urls)} active phishing URLs.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / "openphish_raw.txt", "w") as f:
        f.write("\n".join(urls))

    now = datetime.now(timezone.utc).isoformat()
    new_df = pd.DataFrame({
        "url": urls,
        "label": 1,
        "source": "openphish",
        "fetched_time": now,
    })

    out_path = RAW_DIR / "phishing_urls.csv"
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        before = len(existing_df)
    else:
        combined_df = new_df
        before = 0

    # Keep the FIRST time we ever saw a URL (earliest fetched_time) rather
    # than the most recent - that's the more honest "first observed" record
    # for a URL that stays active across multiple daily runs.
    combined_df = combined_df.sort_values("fetched_time").drop_duplicates(subset="url", keep="first")
    combined_df.to_csv(out_path, index=False)

    new_count = len(combined_df) - before
    print(f"Merged into existing dataset: {new_count} new URLs added, {len(combined_df)} total unique URLs -> {out_path}")

    if len(combined_df) < 500:
        print(
            "\nNote: still building up the dataset. Re-run this script daily "
            "(or every few hours) over the next 1-2 weeks - each run merges "
            "in new active URLs without duplicating ones you already have. "
            "This is expected, not a bug: OpenPhish's free feed only shows "
            "currently-active URLs, so accumulating over time is the "
            "intended way to reach a few thousand."
        )


if __name__ == "__main__":
    main()
