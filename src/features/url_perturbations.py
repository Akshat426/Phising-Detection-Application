"""
Shared URL perturbation functions for adversarial testing and training
augmentation. Kept as a single module so adversarial_check.py (which finds
vulnerabilities) and augment_adversarial_examples.py (which fixes them) use
IDENTICAL perturbation logic - if these ever drifted apart, a "fix" could
silently target a slightly different attack than the one actually measured.
"""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse

import tldextract

# Offline suffix list - reproducible, no network dependency for simple
# hostname parsing, matching the approach in feature_extractor.py.
_TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

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
    owns/controls).
    """
    parsed = urlparse(url)
    ext = _TLD_EXTRACT(url)
    if not ext.subdomain:
        return url

    sub_labels = ext.subdomain.split(".")
    cleaned_labels = [l for l in sub_labels if not any(kw in l.lower() for kw in BRAND_KEYWORDS)]
    if cleaned_labels == sub_labels:
        return url

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

LEXICAL_FEATURES = [
    "url_length", "hostname_length", "path_length", "num_dots", "num_hyphens",
    "num_underscores", "num_slashes", "num_question_marks", "num_equals",
    "num_at_symbols", "num_percent", "num_digits", "num_subdomains",
    "has_ip_address", "uses_https", "has_port", "is_known_shortener",
    "double_slash_in_path", "suspicious_tld_in_path", "brand_keyword_in_subdomain",
]
