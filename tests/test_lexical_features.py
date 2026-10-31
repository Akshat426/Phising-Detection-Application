"""Unit tests for the lexical features of the URL feature extractor.

Lexical features are pure string logic (no network), so they are tested with
WHOIS and page fetching switched off. Also covers the earlier "www." bias fix.
Place at: tests/test_lexical_features.py     Run:  pytest tests -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features.feature_extractor import (  # noqa: E402
    URLFeatureExtractor,
    _strip_leading_www,
    _strip_leading_www_label,
)

LEXICAL = [
    "url_length", "hostname_length", "path_length", "num_dots", "num_hyphens",
    "num_underscores", "num_slashes", "num_question_marks", "num_equals",
    "num_at_symbols", "num_percent", "num_digits", "num_subdomains",
    "has_ip_address", "uses_https", "has_port", "is_known_shortener",
    "double_slash_in_path", "suspicious_tld_in_path", "brand_keyword_in_subdomain",
]
DOMAIN = ["domain_age_days", "domain_registration_length_days", "dns_resolves", "tranco_rank"]
CONTENT = [
    "num_forms", "form_action_external_ratio", "num_iframes", "num_external_links",
    "num_internal_links", "external_link_ratio", "has_password_field", "favicon_external",
    "num_scripts", "title_hostname_overlap", "response_status_code",
]

_EX = URLFeatureExtractor(fetch_content=False, fetch_whois=False)


def feats(url):
    return _EX.extract(url).to_dict()


# ---------------------------------------------------------------- schema
def test_schema_is_35_features_in_three_families():
    d = feats("https://example.com/")
    assert list(d) == LEXICAL + DOMAIN + CONTENT
    assert len(d) == 35


def test_network_features_are_honestly_missing_when_not_fetched():
    d = feats("https://example.com/login")
    for name in DOMAIN + CONTENT:
        assert d[name] is None, name


def test_extraction_is_deterministic():
    assert feats("http://a-b.example.com/p?x=1") == feats("http://a-b.example.com/p?x=1")


# ---------------------------------------------------------------- counts
def test_character_counts():
    d = feats("http://a-b.example.com/p_q?x=1&y=2%20z@w")
    assert d["url_length"] == 40
    assert d["hostname_length"] == len("a-b.example.com")
    assert d["path_length"] == len("/p_q")
    assert d["num_dots"] == 2
    assert d["num_hyphens"] == 1
    assert d["num_underscores"] == 1
    assert d["num_slashes"] == 3
    assert d["num_question_marks"] == 1
    assert d["num_equals"] == 2
    assert d["num_at_symbols"] == 1
    assert d["num_percent"] == 1
    assert d["num_digits"] == 4


# ---------------------------------------------------------------- protocol / port / ip / shortener
@pytest.mark.parametrize("url,https", [
    ("https://example.com/x", 1),
    ("http://example.com/x", 0),
])
def test_uses_https(url, https):
    assert feats(url)["uses_https"] == https


@pytest.mark.parametrize("url,port", [
    ("https://example.com:8443/x", 1),
    ("http://example.com/x", 0),
])
def test_has_port(url, port):
    assert feats(url)["has_port"] == port


@pytest.mark.parametrize("url,flag", [
    ("http://192.168.1.1/login", 1),
    ("http://example.com/login", 0),
])
def test_has_ip_address(url, flag):
    assert feats(url)["has_ip_address"] == flag


@pytest.mark.parametrize("url,flag", [
    ("http://bit.ly/abc", 1),
    ("http://example.com/abc", 0),
])
def test_known_shortener(url, flag):
    assert feats(url)["is_known_shortener"] == flag


# ---------------------------------------------------------------- subdomains and the www. fix
@pytest.mark.parametrize("url,n", [
    ("https://example.com/", 0),
    ("https://shop.example.com/", 1),
    ("http://a.b.example.com/", 2),
    ("https://foo.example.co.uk/", 1),          # multi-part public suffix
])
def test_num_subdomains(url, n):
    assert feats(url)["num_subdomains"] == n


@pytest.mark.parametrize("url,n", [
    ("https://www.youtube.com/watch?v=1", 0),   # leading www is not a subdomain
    ("https://WWW.Example.com/x", 0),           # case-insensitive
    ("https://www.shop.example.com/", 1),       # only the leading www is stripped
    ("https://shop.www.example.com/", 2),       # a non-leading www still counts
    ("https://www2.example.com/", 1),           # www2 is a real label
])
def test_www_is_not_counted_as_subdomain(url, n):
    assert feats(url)["num_subdomains"] == n


def test_www_and_bare_host_look_identical_to_the_model():
    with_www = feats("https://www.example.com/a")
    bare = feats("https://example.com/a")
    assert with_www["hostname_length"] == bare["hostname_length"] == len("example.com")
    assert with_www["num_subdomains"] == bare["num_subdomains"] == 0


@pytest.mark.parametrize("host,expected", [
    ("www.youtube.com", "youtube.com"),
    ("WWW.Example.com", "Example.com"),
    ("www2.example.com", "www2.example.com"),
    ("wwwexample.com", "wwwexample.com"),
    ("youtube.com", "youtube.com"),
])
def test_strip_leading_www(host, expected):
    assert _strip_leading_www(host) == expected


@pytest.mark.parametrize("sub,expected", [
    ("www", ""),
    ("www.shop", "shop"),
    ("WWW.shop", "shop"),
    ("shop.www", "shop.www"),
    ("", ""),
])
def test_strip_leading_www_label(sub, expected):
    assert _strip_leading_www_label(sub) == expected


# ---------------------------------------------------------------- obfuscation patterns
@pytest.mark.parametrize("url,flag", [
    ("http://example.com//x", 1),
    ("http://real.com/http://evil.com", 1),
    ("http://example.com/x", 0),
])
def test_double_slash_in_path(url, flag):
    assert feats(url)["double_slash_in_path"] == flag


@pytest.mark.parametrize("url,flag", [
    ("http://evil.com/paypal.com/login", 1),
    ("http://evil.com/login", 0),
])
def test_suspicious_tld_in_path(url, flag):
    assert feats(url)["suspicious_tld_in_path"] == flag


@pytest.mark.parametrize("url,flag", [
    ("http://paypal-secure.evil.com/", 1),
    ("http://login.example.com/", 1),
    ("http://shop.example.com/", 0),
    ("http://evil.com/paypal", 0),              # brand in the path, not the subdomain
    ("https://www.example.com/", 0),            # plain www carries no brand keyword
])
def test_brand_keyword_in_subdomain(url, flag):
    assert feats(url)["brand_keyword_in_subdomain"] == flag
