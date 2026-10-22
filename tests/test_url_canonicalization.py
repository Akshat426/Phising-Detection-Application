"""Unit tests for URL canonicalization (the trailing-slash leak fix).
Place at: tests/test_url_canonicalization.py     Run:  pytest tests -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features.feature_extractor import URLFeatureExtractor, canonicalize_url  # noqa: E402


@pytest.mark.parametrize("raw,expected", [
    ("http://canardd.com/", "http://canardd.com"),
    ("https://github.com", "https://github.com"),
    ("https://a.com:8080/", "https://a.com:8080"),
    ("a.com/", "a.com"),
    ("https://a.com/path/", "https://a.com/path/"),   # real path: untouched
    ("https://a.com/x", "https://a.com/x"),
    ("https://a.com/?q=1", "https://a.com/?q=1"),     # query: untouched
    ("https://a.com/#top", "https://a.com/#top"),     # fragment: untouched
    ("https://a.com//", "https://a.com//"),           # not a plain root slash
])
def test_canonicalize_url(raw, expected):
    assert canonicalize_url(raw) == expected


def test_canonicalize_is_idempotent():
    for u in ["http://x.com/", "https://x.com/a/", "https://x.com"]:
        assert canonicalize_url(canonicalize_url(u)) == canonicalize_url(u)


@pytest.mark.parametrize("with_slash,without", [
    ("http://canardd.com/", "http://canardd.com"),
    ("https://www.reddit.com/", "https://www.reddit.com"),
    ("https://app.cloudx.io/", "https://app.cloudx.io"),
])
def test_features_identical_with_and_without_trailing_slash(with_slash, without):
    """Invariance by construction: the formatting leak cannot reach the model."""
    ex = URLFeatureExtractor(fetch_content=False, fetch_whois=False)
    assert ex.extract(with_slash).to_dict() == ex.extract(without).to_dict()


def test_real_paths_still_change_features():
    ex = URLFeatureExtractor(fetch_content=False, fetch_whois=False)
    a, b = ex.extract("https://a.com/login").to_dict(), ex.extract("https://a.com").to_dict()
    assert a["path_length"] > b["path_length"] and a["url_length"] > b["url_length"]
