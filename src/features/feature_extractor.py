"""
URL feature extractor for phishing detection.

Design principle: every feature here is genuinely computed from the URL,
its DNS/WHOIS records, or its live page content. Nothing is a hardcoded
placeholder. This directly addresses the weakness found in the reference
project, where 5 of 30 features (SFH, web_traffic, Page_Rank, Google_Index,
Links_pointing_to_page) were constant values regardless of the input URL.

Features fall into three groups:
  1. Lexical   - computed instantly from the URL string alone
  2. Domain    - requires a DNS/WHOIS lookup (network call, can be slow/fail)
  3. Content   - requires fetching the live page (network call, can be slow/fail)

Each group degrades gracefully: if a network call fails (site is down,
WHOIS is rate-limited, etc.) we return None for that feature rather than
crashing or silently substituting a fake value. Downstream, the imputer
in the training pipeline handles the missing values honestly.
"""

from __future__ import annotations

import re
import socket
import ssl
import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import requests
import tldextract
import whois  # python-whois package

try:
    from tranco import Tranco
    _TRANCO_AVAILABLE = True
except ImportError:
    _TRANCO_AVAILABLE = False

USER_AGENT = "Mozilla/5.0 (compatible; PhishingResearchBot/1.0; +https://example.edu)"
REQUEST_TIMEOUT = 6  # seconds - fail fast, don't hang the pipeline on one bad URL

# Use tldextract's bundled offline snapshot of the public suffix list instead
# of fetching it live on every run. This makes feature extraction reproducible
# (no silent behavior change if the live list updates mid-project) and removes
# a network dependency + failure mode from a function that should be fast and
# reliable. Run `python -m tldextract --update` manually if you ever want to
# refresh the snapshot to a newer suffix list.
#
# include_psl_private_domains=True: diagnosed as a bias fix (see
# docs/known-limitations or the project report, "rank bias" / "hosting
# platforms"). The Public Suffix List's "private" section is contributed by
# hosting providers specifically so that each customer's subdomain is treated
# as its own registered domain - e.g. foo.vercel.app and bar.vercel.app are
# unrelated sites, not two pages on one "vercel.app" site. Without this flag,
# tldextract instead parsed "vercel.app" itself as domain+suffix (treating
# ".app" as the effective TLD), which meant every phishing URL hosted on a
# free platform inherited that platform's own Tranco popularity rank and
# WHOIS registration age - a highly-ranked, old domain's data attributed to
# a brand-new phishing subdomain that happens to live on it. Enabling this
# flag fixes attribution for 17 of the hosting platforms this project has
# observed phishing on (see KNOWN_PLATFORM_GAPS below for the rest, which
# the bundled offline PSL snapshot does not yet cover).
_TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)

# Hosting platforms confirmed (by direct testing - see
# src/training/model_diagnostics.py) to NOT be in tldextract's bundled
# offline Public Suffix List snapshot even with include_psl_private_domains
# enabled above. For these specific platforms only, tldextract would still
# incorrectly treat e.g. "foo.weebly.com" as domain="weebly" within the
# "weebly.com" suffix, attributing weebly.com's own rank/WHOIS data to every
# phishing site hosted there. Patched explicitly as a documented, narrow
# fallback - NOT a general-purpose replacement for the PSL mechanism above.
KNOWN_PLATFORM_GAPS = {
    "weebly.com", "godaddysites.com", "edgeone.dev", "zapier.app", "readme.io",
}

IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "shorte.st", "rebrand.ly",
}


@dataclass
class URLFeatures:
    """One row of the final training/prediction dataset.

    Numeric features use Optional[float] so a failed lookup produces a
    real, honest missing value (None -> NaN in pandas) instead of a fake
    default that would silently corrupt the model's understanding of
    that feature.
    """

    # --- lexical features (always available, no network needed) ---
    url_length: int = 0
    hostname_length: int = 0
    path_length: int = 0
    num_dots: int = 0
    num_hyphens: int = 0
    num_underscores: int = 0
    num_slashes: int = 0
    num_question_marks: int = 0
    num_equals: int = 0
    num_at_symbols: int = 0
    num_percent: int = 0
    num_digits: int = 0
    num_subdomains: int = 0
    has_ip_address: int = 0
    uses_https: int = 0
    has_port: int = 0
    is_known_shortener: int = 0
    double_slash_in_path: int = 0
    suspicious_tld_in_path: int = 0
    brand_keyword_in_subdomain: int = 0  # e.g. "paypal-secure.evil.com"

    # --- domain features (WHOIS / DNS, network required) ---
    domain_age_days: Optional[float] = None
    domain_registration_length_days: Optional[float] = None
    dns_resolves: Optional[int] = None
    tranco_rank: Optional[float] = None  # None if not in top 1M (suspicious signal)

    # --- content features (live page fetch, network required) ---
    num_forms: Optional[float] = None
    form_action_external_ratio: Optional[float] = None  # replaces old fake "SFH"
    num_iframes: Optional[float] = None
    num_external_links: Optional[float] = None
    num_internal_links: Optional[float] = None
    external_link_ratio: Optional[float] = None
    has_password_field: Optional[int] = None
    favicon_external: Optional[int] = None
    num_scripts: Optional[float] = None
    title_hostname_overlap: Optional[int] = None
    response_status_code: Optional[float] = None

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class URLFeatureExtractor:
    """Extracts URLFeatures from a single URL.

    Usage:
        extractor = URLFeatureExtractor()
        features = extractor.extract("https://example.com/login")
    """

    def __init__(self, tranco_list=None, fetch_content: bool = True, fetch_whois: bool = True):
        """
        tranco_list: a loaded tranco.Tranco().list() object, or None.
                     Pass this in once and reuse across many extract() calls -
                     loading the Tranco list is slow, doing it per-URL is not.
        fetch_content: whether to fetch the live page (set False for fast,
                        lexical-only feature extraction, e.g. quick screening)
        fetch_whois: whether to run WHOIS lookups (these are the slowest part
                     and some registrars rate-limit aggressively)
        """
        self.tranco_list = tranco_list
        self.fetch_content = fetch_content
        self.fetch_whois = fetch_whois

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def extract(self, url: str) -> URLFeatures:
        url = canonicalize_url(url)  # see canonicalize_url: removes the bare-root trailing-slash leak
        feats = URLFeatures()
        parsed = urlparse(url if "://" in url else f"http://{url}")
        hostname = parsed.hostname or ""
        extracted = _TLD_EXTRACT(url)
        registered_domain = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain

        # Known-gap fallback (see KNOWN_PLATFORM_GAPS above): if tldextract's
        # bundled PSL snapshot doesn't yet cover this specific hosting
        # platform, and the URL does have a subdomain on it, treat that
        # subdomain's leftmost label plus the platform as the effective
        # registered domain - e.g. "foo.weebly.com" -> "foo.weebly.com", not
        # just "weebly.com". This only affects which domain tranco_rank and
        # WHOIS are looked up for (see _fill_domain); it does not change
        # num_subdomains/hostname_length, which are a separate fix (below).
        if registered_domain in KNOWN_PLATFORM_GAPS and extracted.subdomain:
            first_label = extracted.subdomain.split(".")[0]
            registered_domain = f"{first_label}.{registered_domain}"

        self._fill_lexical(feats, url, parsed, hostname, extracted)

        if self.fetch_whois:
            self._fill_domain(feats, hostname, registered_domain)

        if self.fetch_content:
            self._fill_content(feats, url, hostname)

        return feats

    # ------------------------------------------------------------------
    # Lexical features - pure string/parsing logic, no network
    # ------------------------------------------------------------------
    def _fill_lexical(self, feats: URLFeatures, url: str, parsed, hostname: str, extracted) -> None:
        feats.url_length = len(url)
        # Diagnosed bias fix: a leading "www." was being counted as a real
        # subdomain, and every legitimate bare-root URL in the training data
        # (sourced from Tranco as plain names, e.g. "youtube.com") happened
        # to never start with "www.", so "bare root + a subdomain" was a
        # pattern that - by how the dataset was built, not by anything
        # meaningful about phishing - appeared only in phishing examples.
        # A real URL like https://www.youtube.com/ matched that pattern and
        # was flagged at 95%+. We normalize by stripping exactly one leading
        # "www" label before measuring hostname length or counting
        # subdomains, so "www.youtube.com" and "youtube.com" are treated
        # identically by these two features (all other lexical features,
        # and has_ip_address/is_known_shortener/etc. below, intentionally
        # keep using the raw, unmodified hostname).
        normalized_hostname = _strip_leading_www(hostname)
        feats.hostname_length = len(normalized_hostname)
        feats.path_length = len(parsed.path or "")
        feats.num_dots = url.count(".")
        feats.num_hyphens = url.count("-")
        feats.num_underscores = url.count("_")
        feats.num_slashes = url.count("/")
        feats.num_question_marks = url.count("?")
        feats.num_equals = url.count("=")
        feats.num_at_symbols = url.count("@")
        feats.num_percent = url.count("%")
        feats.num_digits = sum(c.isdigit() for c in url)

        feats.has_ip_address = int(bool(IP_PATTERN.match(hostname)))
        feats.uses_https = int(parsed.scheme == "https")
        feats.has_port = int(parsed.port is not None)
        feats.is_known_shortener = int(hostname.lower() in SHORTENER_DOMAINS)

        subdomain = extracted.subdomain or ""
        normalized_subdomain = _strip_leading_www_label(subdomain)
        feats.num_subdomains = len([s for s in normalized_subdomain.split(".") if s]) if normalized_subdomain else 0

        # "//"" appearing after the initial protocol separator is a classic
        # redirect-obfuscation trick (e.g. http://real.com//http://evil.com)
        rest_of_url = url.split("://", 1)[-1]
        feats.double_slash_in_path = int("//" in rest_of_url)

        # a TLD-looking token appearing in the path (not the actual host) is
        # a common trick, e.g. http://evil.com/paypal.com/login
        path_lower = (parsed.path or "").lower()
        feats.suspicious_tld_in_path = int(
            bool(re.search(r"\.(com|net|org|info|biz)\b", path_lower))
        )

        brand_keywords = ["paypal", "amazon", "apple", "microsoft", "google", "bank", "secure", "login", "account"]
        feats.brand_keyword_in_subdomain = int(
            any(kw in subdomain.lower() for kw in brand_keywords) and subdomain != ""
        )

    # ------------------------------------------------------------------
    # Domain features - WHOIS + DNS + Tranco rank
    # ------------------------------------------------------------------
    def _fill_domain(self, feats: URLFeatures, hostname: str, registered_domain: str) -> None:
        # DNS resolution
        try:
            socket.gethostbyname(hostname)
            feats.dns_resolves = 1
        except Exception:
            feats.dns_resolves = 0

        # Tranco rank (proxy for the old "web_traffic"/"Page_Rank" features,
        # but backed by a real, downloadable, current ranking dataset).
        # self.tranco_list is now a plain {domain: rank} dict (see
        # load_tranco_list()) - a dict miss unambiguously means "not in the
        # top 1M", returned as None, with no sentinel-value ambiguity.
        if self.tranco_list is not None:
            rank = self.tranco_list.get(registered_domain)
            feats.tranco_rank = float(rank) if rank is not None else None

        # WHOIS lookup - this is the slowest and least reliable call. Many
        # registrars throttle, misconfigure, or drop automated WHOIS queries,
        # so we retry a couple of times with backoff before giving up. This
        # squeezes out the transient failures (server briefly overloaded)
        # without pretending WHOIS is a reliable API - some failures are
        # permanent (registrar blocks bots entirely) and will still end up
        # as an honest missing value, which is correct, not a bug.
        if not self.fetch_whois:
            return
        w = _whois_with_retry(registered_domain, max_retries=2, base_delay=0.75)
        if w is None:
            return
        try:
            created = _first_date(w.creation_date)
            expires = _first_date(w.expiration_date)
            now = datetime.now(timezone.utc)

            if created:
                feats.domain_age_days = (now - _as_utc(created)).days
            if created and expires:
                feats.domain_registration_length_days = (_as_utc(expires) - _as_utc(created)).days
        except Exception:
            pass  # leave as None - honest missing value, handled by imputer later

    # ------------------------------------------------------------------
    # Content features - fetch the live page and parse it
    # ------------------------------------------------------------------
    def _fill_content(self, feats: URLFeatures, url: str, hostname: str) -> None:
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            feats.response_status_code = float(resp.status_code)
        except Exception:
            return  # page unreachable - everything content-based stays None

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception:
            return

        forms = soup.find_all("form")
        feats.num_forms = float(len(forms))
        if forms:
            external_actions = 0
            for f in forms:
                action = f.get("action", "")
                if action and action.startswith("http"):
                    action_host = urlparse(action).hostname or ""
                    if action_host and action_host != hostname:
                        external_actions += 1
            feats.form_action_external_ratio = external_actions / len(forms)
        else:
            feats.form_action_external_ratio = 0.0

        feats.num_iframes = float(len(soup.find_all("iframe")))
        feats.num_scripts = float(len(soup.find_all("script")))
        feats.has_password_field = int(bool(soup.find("input", {"type": "password"})))

        links = soup.find_all("a", href=True)
        internal, external = 0, 0
        for a in links:
            href = a["href"]
            if href.startswith("http"):
                link_host = urlparse(href).hostname or ""
                if link_host and link_host != hostname:
                    external += 1
                else:
                    internal += 1
            elif href.startswith("/") or not href.startswith(("mailto:", "tel:", "javascript:", "#")):
                internal += 1
        feats.num_internal_links = float(internal)
        feats.num_external_links = float(external)
        total_links = internal + external
        feats.external_link_ratio = (external / total_links) if total_links else 0.0

        favicon = soup.find("link", rel=lambda v: v and "icon" in v.lower())
        if favicon and favicon.get("href", "").startswith("http"):
            fav_host = urlparse(favicon["href"]).hostname or ""
            feats.favicon_external = int(bool(fav_host and fav_host != hostname))
        else:
            feats.favicon_external = 0

        title_tag = soup.find("title")
        title_text = (title_tag.text if title_tag else "").lower()
        domain_token = hostname.split(".")[0].lower() if hostname else ""
        feats.title_hostname_overlap = int(bool(domain_token) and domain_token in title_text)


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def canonicalize_url(url: str) -> str:
    """Remove the trailing slash from a bare-root URL, and nothing else.

    'http://site.com/' -> 'http://site.com'. URLs with a real path, a query or
    a fragment are returned unchanged.

    Diagnosed formatting leak: in the training data every bare-root PHISHING
    URL (from OpenPhish) was written WITH a trailing slash (974 of 974) and
    every bare-root LEGITIMATE URL (built from Tranco names) WITHOUT one
    (974 of 974). A tree model could read class off path_length /
    num_slashes / url_length without learning anything about phishing: adding
    only a "/" made 17.7% of legitimate roots look phishing, and removing it
    let 1 in 3 phishing roots evade. 'site.com' and 'site.com/' are the same
    resource, so we normalise them before any feature is computed - the same
    treatment as the leading-"www." fix. This function is also applied at
    serving time (it lives inside extract()), so training and the live app
    can never disagree about URL form.
    """
    if not url.endswith("/"):
        return url
    parsed = urlparse(url if "://" in url else f"http://{url}")
    if parsed.path == "/" and not parsed.params and not parsed.query and not parsed.fragment:
        return url[:-1]
    return url


def _strip_leading_www(hostname: str) -> str:
    """Remove a single leading 'www.' prefix from a hostname, case-insensitive.

    'www.youtube.com' -> 'youtube.com'; 'www2.example.com' is left alone
    (only an exact 'www' label, matching the diagnosed pattern - not a
    general-purpose "remove all www-like prefixes" normalizer).
    """
    if hostname.lower().startswith("www."):
        return hostname[4:]
    return hostname


def _strip_leading_www_label(subdomain: str) -> str:
    """Remove a leading 'www' label from a tldextract subdomain string.

    tldextract's `.subdomain` for 'www.youtube.com' is 'www' -> stripped to
    ''. For 'www.shop.example.com' it is 'www.shop' -> stripped to 'shop'.
    For 'shop.www.example.com' it is 'shop.www' -> left unchanged, since
    'www' is not the leading label there and this is not the diagnosed
    pattern (which is specifically a leading "www.").
    """
    labels = subdomain.split(".") if subdomain else []
    if labels and labels[0].lower() == "www":
        labels = labels[1:]
    return ".".join(labels)


def _whois_with_retry(domain: str, max_retries: int = 2, base_delay: float = 0.75):
    """Attempt a WHOIS lookup with a couple of retries and exponential
    backoff before giving up.

    WHOIS is a genuinely flaky protocol in practice: some registrars briefly
    rate-limit rapid automated queries (common when a batch job like
    build_dataset.py hits many different domains from many threads at once),
    and a short retry recovers most of those transient failures. It will
    NOT recover permanent failures (registrar blocks bots outright, WHOIS
    server misconfigured for that TLD) - those still correctly end up as
    None, which is the honest result, not a bug.

    Returns the whois.whois() result object, or None if every attempt fails.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return whois.whois(domain)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))  # 0.75s, then 1.5s
    return None


def _first_date(value):
    """python-whois sometimes returns a list of dates for a field. Take the earliest."""
    if isinstance(value, list):
        value = [v for v in value if v is not None]
        return min(value) if value else None
    return value


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def load_tranco_list():
    """Load the Tranco top-sites list as an explicit {domain: rank} lookup dict.

    Deliberately built as our own dict rather than relying on the tranco
    package's TrancoList.rank() method for out-of-list domains: testing on
    real data showed that method returns a finite sentinel value for
    domains outside the loaded list rather than a clean "not found" signal.
    Since every legitimate URL in this project was SOURCED from Tranco,
    every legitimate row always got a real, present rank - while phishing
    URLs (which are essentially never globally popular domains) all got
    the same sentinel value. That made "does this row have a real Tranco
    rank" a near-perfect proxy for "which data source did this URL come
    from," not a genuine popularity signal - a serious, silent data leakage
    bug. This explicit dict makes "not found" unambiguous: a plain Python
    dict lookup miss, returned as None, every time.
    """
    if not _TRANCO_AVAILABLE:
        raise ImportError("pip install tranco")
    t = Tranco(cache=True, cache_dir=".tranco_cache")
    tlist = t.list()
    top_1m = tlist.top(1_000_000)
    return {domain: rank for rank, domain in enumerate(top_1m, start=1)}
