"""
Reproducible diagnostics behind the findings quoted in the report.

Every number in report Sections 6.4, 7.7 and 10.6 that is NOT in
model_comparison.json comes from this script, so anyone (including a
reviewer) can re-run it and get the same figures.

What it measures (all offline, using the processed CSVs and the served model):
  1. Hosting-platform confound: how much phishing lives on free hosts
     (vercel.app, pages.dev, ...) and what Tranco rank those rows carry.
  2. Rank bias: share of top-ranked rows that are phishing.
  3. "www." quirk: legitimate bare-root URLs that start with "www." in the data.
  4. Deployed-model dependence on tranco_rank: false-alarm rate on held-out
     legitimate URLs when the rank is present vs missing.
  5. Honest evaluation: random 5-fold vs domain-grouped 5-fold F1.

Usage (project root):   python src/training/model_diagnostics.py
Output: printed report + final_model/diagnostics_report.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "processed"

# Free / multi-tenant hosting platforms: every customer gets a subdomain, so
# domain-level features (WHOIS age, Tranco rank) describe the PLATFORM.
PLATFORMS = {
    "pages.dev", "vercel.app", "blogspot.com", "weebly.com", "github.io", "godaddysites.com",
    "workers.dev", "netlify.app", "edgeone.dev", "webflow.io", "zapier.app", "wixsite.com",
    "web.app", "firebaseapp.com", "herokuapp.com", "r2.dev", "notion.site", "framer.app",
    "glitch.me", "readme.io", "onrender.com", "framer.website",
}


def registered_domain(url: str) -> str:
    """Approximate registered domain (last two labels; last three for co.uk-style)."""
    parts = (urlparse(url).hostname or "").split(".")
    if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "net", "ac", "gov") and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def main():
    raw = pd.concat(
        [pd.read_csv(D / f"{s}.csv") for s in ("train_pre_augmentation", "validation", "test")], ignore_index=True
    )
    n_train = len(pd.read_csv(D / "train_pre_augmentation.csv"))
    trans_train = pd.read_csv(D / "train_transformed.csv").iloc[:n_train]  # first rows = un-augmented
    trans_val = pd.read_csv(D / "validation_transformed.csv")
    trans_test = pd.read_csv(D / "test_transformed.csv")
    X_all = pd.concat([trans_train, trans_val, trans_test], ignore_index=True)
    y = X_all.pop("label").values
    assert len(X_all) == len(raw) and (y == raw.label.values).all(), "row alignment check failed"

    raw["host"] = raw.url.map(lambda u: urlparse(u).hostname or "")
    raw["reg"] = raw.url.map(registered_domain)
    raw["platform"] = raw.reg.isin(PLATFORMS)
    raw["www"] = raw.host.str.startswith("www.")
    phish, legit = raw[raw.label == 1], raw[raw.label == 0]
    out = {"n_rows": int(len(raw)), "n_phishing": int(len(phish)), "n_legit": int(len(legit))}

    # 1 + 2: platform confound and rank bias
    out["phishing_on_platforms"] = int(phish.platform.sum())
    out["phishing_with_rank"] = int(phish.tranco_rank.notna().sum())
    r = raw.tranco_rank
    out["top1000_rows"] = int((r <= 1000).sum())
    out["top1000_phishing"] = int(((r <= 1000) & (raw.label == 1)).sum())
    out["top1000_phishing_share"] = round(float(raw[r <= 1000].label.mean()), 3)
    out["overall_phishing_share"] = round(float(raw.label.mean()), 3)

    # 3: www quirk
    bare = raw.path_length <= 1
    out["legit_bare_root_total"] = int(((raw.label == 0) & bare).sum())
    out["legit_bare_root_www"] = int(((raw.label == 0) & bare & raw.www).sum())

    # 4: deployed-model dependence on tranco_rank (held-out validation + test only)
    model = joblib.load(ROOT / "final_model" / "model.pkl")
    held = pd.concat([trans_val, trans_test], ignore_index=True)
    y_held = held.pop("label").values
    cols = list(X_all.columns)
    imputed_rank = trans_train.loc[trans_train.tranco_rank_was_missing == 1, "tranco_rank"].mode()[0]
    missing = held[cols].copy()
    missing["tranco_rank_was_missing"], missing["tranco_rank"] = 1, imputed_rank
    p_norm, p_miss = model.predict_proba(held[cols])[:, 1], model.predict_proba(missing)[:, 1]
    is_legit = y_held == 0
    out["legit_heldout"] = int(is_legit.sum())
    out["false_alarm_rate_rank_present"] = round(float((p_norm[is_legit] >= 0.5).mean()), 4)
    out["false_alarm_rate_rank_missing"] = round(float((p_miss[is_legit] >= 0.5).mean()), 4)

    # 4b: what if other network-dependent features fail instead? (each set to "missing", like a failed lookup)
    def fail_group(indicators):
        X = held[cols].copy()
        for c in indicators:
            base = c[: -len("_was_missing")]
            const = trans_train.loc[trans_train[c] == 1, base].mode()
            X[c] = 1
            if len(const):
                X[base] = const[0]
        return float((model.predict_proba(X)[is_legit, 1] >= 0.5).mean())

    ind = [c for c in cols if c.endswith("_was_missing")]
    whois_ind = [c for c in ind if c.startswith("domain_")]
    content_ind = [c for c in ind if c not in whois_ind and c != "tranco_rank_was_missing"]
    out["false_alarm_rate_whois_fails"] = round(fail_group(whois_ind), 4)
    out["false_alarm_rate_page_fetch_fails"] = round(fail_group(content_ind), 4)

    # 5: random vs domain-grouped cross-validation (same XGBoost settings for both)
    oof = np.zeros(len(y))

    def cv_f1(splits, keep_oof=False):
        scores = []
        for a, b in splits:
            m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=(y[a] == 0).sum() / (y[a] == 1).sum(), eval_metric="logloss",
                              n_jobs=4, random_state=0)
            m.fit(X_all.iloc[a], y[a])
            pred = m.predict(X_all.iloc[b])
            if keep_oof:
                oof[b] = pred
            scores.append(f1_score(y[b], pred))
        return float(np.mean(scores)), float(np.std(scores))

    rnd = StratifiedKFold(5, shuffle=True, random_state=1).split(X_all, y)
    grp = StratifiedGroupKFold(5, shuffle=True, random_state=1).split(X_all, y, raw.reg.values)
    out["cv_f1_random"], out["cv_f1_random_std"] = cv_f1(rnd)
    out["cv_f1_grouped"], out["cv_f1_grouped_std"] = cv_f1(grp, keep_oof=True)
    on_platform = raw.platform.values
    out["recall_platform_hosted_phishing"] = round(float(oof[(y == 1) & on_platform].mean()), 3)
    out["recall_own_domain_phishing"] = round(float(oof[(y == 1) & ~on_platform].mean()), 3)

    print("=== Model diagnostics ===")
    print(f"Rows: {out['n_rows']} ({out['n_legit']} legitimate, {out['n_phishing']} phishing)")
    print(f"1. Phishing on free-hosting platforms: {out['phishing_on_platforms']}/{out['n_phishing']} "
          f"({out['phishing_on_platforms']/out['n_phishing']:.0%}); with a Tranco rank: {out['phishing_with_rank']}/{out['n_phishing']} "
          f"({out['phishing_with_rank']/out['n_phishing']:.0%})")
    print(f"2. Rows ranked in Tranco top 1000: {out['top1000_rows']}, of which phishing: {out['top1000_phishing']} "
          f"({out['top1000_phishing_share']:.0%}; overall phishing share {out['overall_phishing_share']:.0%})")
    print(f"3. Legitimate bare-root URLs: {out['legit_bare_root_total']}; starting with 'www.': {out['legit_bare_root_www']}")
    print(f"4. False-alarm rate on {out['legit_heldout']} held-out legitimate URLs: rank present "
          f"{out['false_alarm_rate_rank_present']:.1%} -> rank missing {out['false_alarm_rate_rank_missing']:.1%}")
    print(f"   Other lookups failing instead: WHOIS -> {out['false_alarm_rate_whois_fails']:.1%}, page fetch -> {out['false_alarm_rate_page_fetch_fails']:.1%}")
    print(f"5. 5-fold CV F1: random {out['cv_f1_random']:.3f} (+/-{out['cv_f1_random_std']:.3f}) vs "
          f"domain-grouped {out['cv_f1_grouped']:.3f} (+/-{out['cv_f1_grouped_std']:.3f})")

    with open(ROOT / "final_model" / "diagnostics_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"   Phishing recall (domain-grouped CV): platform-hosted {out['recall_platform_hosted_phishing']:.1%} vs own-domain {out['recall_own_domain_phishing']:.1%}")
    print("Saved final_model/diagnostics_report.json")


if __name__ == "__main__":
    main()
