"""
Quick false-positive check: hit the running local API with a batch of
well-known, unambiguously legitimate root-level URLs (the exact URL shape
- bare domain, no path - that the recent adversarial training augmentation
specifically targeted). Every single one of these SHOULD be classified as
legitimate; any "phishing" result here is a real, reportable false positive,
not a guess.

Requires the FastAPI server to already be running:
    uvicorn src.serving.app:app --reload --port 8000

Usage:
    python test_known_legitimate_sites.py
"""

import requests

API_URL = "http://127.0.0.1:8000/predict"

# Deliberately diverse categories: tech, banking, media, e-commerce, social -
# a false-positive pattern isolated to one category (e.g. only banks) would
# tell a different story than one spread evenly across all of them.
KNOWN_LEGITIMATE_URLS = [
    "https://www.google.com/",
    "https://www.wikipedia.org/",
    "https://www.youtube.com/",
    "https://github.com/",
    "https://www.amazon.com/",
    "https://www.microsoft.com/",
    "https://www.apple.com/",
    "https://www.netflix.com/",
    "https://www.linkedin.com/",
    "https://www.reddit.com/",
    "https://stackoverflow.com/",
    "https://www.bbc.com/",
    "https://www.spotify.com/",
    "https://www.dropbox.com/",
    "https://www.adobe.com/",
    "https://www.paypal.com/",
    "https://www.chase.com/",
    "https://www.bankofamerica.com/",
    "https://www.zee5.com/",  # the original false positive, included for direct comparison
]


def main():
    results = []
    for url in KNOWN_LEGITIMATE_URLS:
        try:
            resp = requests.post(API_URL, json={"url": url}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results.append({
                "url": url,
                "prediction": data["prediction"],
                "phishing_probability": data["phishing_probability"],
            })
        except Exception as e:
            results.append({"url": url, "prediction": "ERROR", "phishing_probability": None, "error": str(e)})

    print(f"{'URL':<40} {'Prediction':<12} {'Phishing Prob':>15}")
    print("-" * 70)
    false_positives = 0
    for r in results:
        prob_str = f"{r['phishing_probability']:.1%}" if r["phishing_probability"] is not None else "ERROR"
        flag = "  <-- FALSE POSITIVE" if r["prediction"] == "phishing" else ""
        print(f"{r['url']:<40} {r['prediction']:<12} {prob_str:>15}{flag}")
        if r["prediction"] == "phishing":
            false_positives += 1

    print("-" * 70)
    print(f"\nFalse positive rate on known-legitimate root URLs: "
          f"{false_positives}/{len(KNOWN_LEGITIMATE_URLS)} "
          f"({false_positives / len(KNOWN_LEGITIMATE_URLS):.1%})")


if __name__ == "__main__":
    main()
