"""
Adversarial robustness check: how easily can a low-effort attacker evade
this model using only cheap, URL-string-level tricks?

Key design constraint, stated explicitly because it matters for interpreting
the results: an attacker can trivially change the URL string itself (scheme,
path, subdomain naming) but CANNOT cheaply fake a domain's WHOIS registration
age or a live page's real HTML content without actually registering a domain
and hosting real infrastructure - that's a fundamentally more expensive attack
this check does not attempt to simulate. So this check only perturbs LEXICAL
features (computed instantly from the URL string) and holds every domain- and
content-level feature at its real, original value. This is a deliberately
conservative, honestly-scoped test: it answers "how robust is the model to a
lazy attacker doing simple string tricks," not "how robust is it to a
well-resourced attacker" - that distinction is worth stating plainly in your
report rather than overclaiming what this test covers.

Strategies tested, roughly in order of real-world cheapness:
  1. Switch to HTTPS (trivial - many free certificate authorities exist)
  2. Strip the path/query down to the bare domain root
  3. Remove brand-keyword subdomains (e.g. "paypal-secure.evil.com" -> "evil.com")
  4. All three combined (the "attacker tries reasonably hard" case)

Usage:
    python adversarial_check.py
    python adversarial_check.py --model-path final_model/model_no_tranco.pkl --n-examples 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import joblib
import pandas as pd
import tldextract

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.feature_extractor import URLFeatureExtractor  # noqa: E402
from features.transformation import DataTransformer  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "final_model"
TRANSFORM_ARTIFACTS_DIR = PROCESSED_DIR / "transform_artifacts"

# Offline suffix list, matching feature_extractor.py's approach - reproducible,
# no network dependency for a simple hostname-parsing operation.
_TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

LEXICAL_FEATURES = [
    "url_length", "hostname_length", "path_length", "num_dots", "num_hyphens",
    "num_underscores", "num_slashes", "num_question_marks", "num_equals",
    "num_at_symbols", "num_percent", "num_digits", "num_subdomains",
    "has_ip_address", "uses_https", "has_port", "is_known_shortener",
    "double_slash_in_path", "suspicious_tld_in_path", "brand_keyword_in_subdomain",
]

BRAND_KEYWORDS = ["paypal", "amazon", "apple", "microsoft", "google", "bank", "secure", "login", "account"]


def to_https(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(scheme="https"))


def strip_path(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(path="", query="", fragment=""))


def remove_brand_subdomain(url: str) -> str:
    """Removes brand-keyword-containing labels from the SUBDOMAIN only,
    never touching the registrable domain (the part an attacker actually
    owns/controls). An earlier version of this function operated on the
    whole hostname and could accidentally strip parts of the domain itself
    (e.g. "verify-account-xyz123.com" losing "verify-account-xyz123"
    because "account" matched the keyword list), producing a nonsensical,
    unrealistic URL like "http://com/..." - caught during testing before
    this script was ever run against real data.
    """
    parsed = urlparse(url)
    ext = _TLD_EXTRACT(url)
    if not ext.subdomain:
        return url  # nothing to remove

    sub_labels = ext.subdomain.split(".")
    cleaned_labels = [l for l in sub_labels if not any(kw in l.lower() for kw in BRAND_KEYWORDS)]
    if cleaned_labels == sub_labels:
        return url  # no brand keywords found in the subdomain - unchanged

    registrable_domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    new_hostname = ".".join(cleaned_labels + [registrable_domain]) if cleaned_labels else registrable_domain
    new_netloc = new_hostname if not parsed.port else f"{new_hostname}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=new_netloc))


STRATEGIES = {
    "https_only": to_https,
    "strip_path_only": strip_path,
    "remove_brand_subdomain_only": remove_brand_subdomain,
    "combined_all_three": lambda u: remove_brand_subdomain(strip_path(to_https(u))),
}


def main():
    parser = argparse.ArgumentParser(description="Test model robustness to cheap URL-level evasion tricks")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--n-examples", type=int, default=5,
                         help="How many correctly-caught phishing URLs to test evasion against")
    args = parser.parse_args()

    model_path = Path(args.model_path) if args.model_path else MODELS_DIR / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found - run model_trainer.py first.")
    if not TRANSFORM_ARTIFACTS_DIR.exists():
        raise FileNotFoundError(f"{TRANSFORM_ARTIFACTS_DIR} not found - run transformation.py first.")

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    transformer = DataTransformer.load(TRANSFORM_ARTIFACTS_DIR)
    extractor = URLFeatureExtractor(fetch_content=False, fetch_whois=False)

    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    if "url" not in test_df.columns:
        raise ValueError("test.csv is missing the 'url' column - can't recover original URLs for perturbation.")

    phishing_rows = test_df[test_df["label"] == 1].reset_index(drop=True)

    # Only test against examples the model ALREADY correctly catches -
    # testing evasion against something it already misses would be meaningless.
    transformed_all = transformer.transform(phishing_rows)
    model_input = transformed_all[[c for c in transformed_all.columns if c != "label"]]
    if hasattr(model, "feature_names_in_"):
        model_input = model_input[list(model.feature_names_in_)]
    elif hasattr(model, "get_booster"):
        model_input = model_input[model.get_booster().feature_names]
    original_preds = model.predict(model_input)
    original_probs = model.predict_proba(model_input)[:, 1]

    caught_mask = original_preds == 1
    caught_indices = phishing_rows.index[caught_mask][: args.n_examples]

    if len(caught_indices) == 0:
        print("No correctly-caught phishing examples found in test set to test evasion against.")
        return

    print(f"\nTesting {len(caught_indices)} correctly-caught phishing URLs against "
          f"{len(STRATEGIES)} evasion strategies.\n")

    results = []
    for idx in caught_indices:
        row = phishing_rows.loc[idx]
        url = row["url"]
        original_prob = original_probs[phishing_rows.index.get_loc(idx)]
        print(f"URL: {url}")
        print(f"  Original phishing probability: {original_prob:.1%}")

        for strategy_name, strategy_fn in STRATEGIES.items():
            perturbed_url = strategy_fn(url)
            if perturbed_url == url:
                print(f"  [{strategy_name}] no change applicable to this URL - skipped")
                continue

            perturbed_lexical = extractor.extract(perturbed_url).to_dict()
            new_row = row.copy()
            for feat in LEXICAL_FEATURES:
                new_row[feat] = perturbed_lexical[feat]

            new_row_df = pd.DataFrame([new_row])
            transformed = transformer.transform(new_row_df)
            new_input = transformed[[c for c in transformed.columns if c != "label"]]
            if hasattr(model, "feature_names_in_"):
                new_input = new_input[list(model.feature_names_in_)]
            elif hasattr(model, "get_booster"):
                new_input = new_input[model.get_booster().feature_names]

            new_prob = model.predict_proba(new_input)[0, 1]
            new_pred = model.predict(new_input)[0]
            flipped = "  <-- EVADED DETECTION" if new_pred == 0 else ""
            print(f"  [{strategy_name}] {perturbed_url}")
            print(f"      phishing probability: {new_prob:.1%} (was {original_prob:.1%}){flipped}")

            results.append({
                "url": url, "strategy": strategy_name, "perturbed_url": perturbed_url,
                "original_prob": float(original_prob), "new_prob": float(new_prob),
                "evaded": bool(new_pred == 0),
            })
        print()

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        evasion_rate_by_strategy = results_df.groupby("strategy")["evaded"].mean()
        print("=== Summary: evasion success rate by strategy ===")
        for strategy, rate in evasion_rate_by_strategy.items():
            print(f"  {strategy:<30} {rate:.1%} of tested examples evaded detection")

        overall_rate = results_df["evaded"].mean()
        print(f"\nOverall evasion rate across all cheap-trick attempts: {overall_rate:.1%}")

        out_path = MODELS_DIR / "explainability" / "adversarial_check_results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path, index=False)
        print(f"Full results saved -> {out_path}")


if __name__ == "__main__":
    main()
