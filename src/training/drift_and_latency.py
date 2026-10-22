"""
Deployment-section evidence for the paper: temporal drift + inference latency.

A. TEMPORAL DRIFT (real, not simulated). Phishing URLs were collected in batches
   (data/raw/phishing_urls.csv has fetched_time). Rows are split at the MEDIAN fetch
   time into an earlier "reference" group and a later "recent" group. We report
     * two-sample Kolmogorov-Smirnov tests per feature (Bonferroni-corrected), and
     * the final model's recall on each period, using ONLY validation + test rows
       (never training rows).
B. LATENCY.
     * in-process: transform + predict for one URL at a time (as the API does),
       lexical feature extraction, and SHAP explanation time (if shap is installed)
     * optional end-to-end through the running API (--server), using LEGITIMATE
       URLs only so no phishing page is visited.

Place at: src/training/drift_and_latency.py
Run (project root):
    python src\\training\\drift_and_latency.py
    python src\\training\\drift_and_latency.py --server http://127.0.0.1:8000 --n-live 25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from math import sqrt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.feature_extractor import URLFeatureExtractor  # noqa: E402
from features.transformation import DataTransformer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FINAL = ROOT / "final_model"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def model_matrix(model, transformed):
    X = transformed[[c for c in transformed.columns if c != "label"]]
    if hasattr(model, "feature_names_in_"):
        X = X[list(model.feature_names_in_)]
    elif hasattr(model, "get_booster"):
        X = X[model.get_booster().feature_names]
    return X


def pct(a, q):
    return float(np.percentile(a, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=FINAL / "model.pkl")
    ap.add_argument("--artifacts-dir", type=Path, default=PROCESSED / "transform_artifacts")
    ap.add_argument("--server", type=str, default=None)
    ap.add_argument("--n-live", type=int, default=25)
    a = ap.parse_args()

    model = joblib.load(a.model_path)
    transformer = DataTransformer.load(a.artifacts_dir)
    report = {}

    # ------------------------------------------------------------ A. drift
    raw = pd.read_csv(ROOT / "data" / "raw" / "phishing_urls.csv")
    raw["t"] = pd.to_datetime(raw["fetched_time"], utc=True, errors="coerce")
    tmap = raw.dropna(subset=["t"]).drop_duplicates("url").set_index("url")["t"]

    ds = pd.read_csv(PROCESSED / "dataset.csv")
    ph = ds[ds["label"] == 1].copy()
    ph["t"] = ph["url"].map(tmap)
    ph = ph.dropna(subset=["t"])
    print("=== A. TEMPORAL DRIFT (phishing collection batches) ===")
    print("Rows per fetch date:")
    print(ph["t"].dt.date.value_counts().sort_index().to_string())
    cutoff = ph["t"].median()
    ref, rec = ph[ph["t"] <= cutoff], ph[ph["t"] > cutoff]
    print(f"\nCutoff (median fetch time): {cutoff}\nreference (earlier): {len(ref)} rows, "
          f"recent (later): {len(rec)} rows")
    if len(ref) < 30 or len(rec) < 30:
        print("Not enough rows in one period for a drift test - skipping part A.")
        report["drift"] = None
    else:
        feats = [c for c in ds.columns if c not in ("url", "label", "source", "fetched_time")
                 and pd.api.types.is_numeric_dtype(ds[c])]
        alpha = 0.05 / len(feats)
        rows = []
        for c in feats:
            x, y = ref[c].dropna(), rec[c].dropna()
            if len(x) < 20 or len(y) < 20:
                continue
            s = ks_2samp(x, y)
            rows.append((c, float(s.statistic), float(s.pvalue)))
        rows.sort(key=lambda r: -r[1])
        flagged = [r for r in rows if r[2] < alpha]
        print(f"\nKS tests on {len(rows)} features; Bonferroni threshold p < {alpha:.5f}")
        print(f"Features with significant shift: {len(flagged)} of {len(rows)}")
        print(f"{'feature':<36}{'KS stat':>9}{'p-value':>12}  significant")
        for c, st, p in rows[:12]:
            print(f"{c:<36}{st:>9.3f}{p:>12.2e}  {'YES' if p < alpha else 'no'}")
        report["drift"] = {"cutoff": str(cutoff), "n_reference": len(ref), "n_recent": len(rec),
                           "bonferroni_alpha": alpha, "n_significant": len(flagged),
                           "n_features": len(rows),
                           "top": [{"feature": c, "ks": st, "p": p} for c, st, p in rows[:12]]}

        # recall by period, held-out rows only
        print("\nModel recall on held-out phishing rows, by collection period (validation + test):")
        recs = {"reference": [0, 0, []], "recent": [0, 0, []]}
        for split in ("validation", "test"):
            df = pd.read_csv(PROCESSED / f"{split}.csv")
            tt = df["url"].map(tmap)
            probs = model.predict_proba(model_matrix(model, transformer.transform(df)))[:, 1]
            for i in np.where((df["label"] == 1).to_numpy() & tt.notna().to_numpy())[0]:
                key = "reference" if tt.iloc[i] <= cutoff else "recent"
                recs[key][0] += 1
                recs[key][1] += int(probs[i] >= 0.5)
                recs[key][2].append(probs[i])
        report["recall_by_period"] = {}
        for k, (n, hit, pr) in recs.items():
            lo, hi = wilson(hit, n)
            print(f"  {k:<10} n={n:<4} recall {hit / n:.1%} [{lo:.1%}, {hi:.1%}]  mean P(phishing) {np.mean(pr):.3f}"
                  if n else f"  {k:<10} n=0")
            if n:
                report["recall_by_period"][k] = {"n": n, "recall": hit / n, "ci95": [lo, hi]}

    # --------------------------------------------------------- B. latency
    print("\n=== B. LATENCY ===")
    test = pd.read_csv(PROCESSED / "test.csv")
    sample = test.sample(n=min(300, len(test)), random_state=42)
    ms = []
    for k, (_, row) in enumerate(sample.iterrows()):
        t0 = time.perf_counter()
        X = model_matrix(model, transformer.transform(pd.DataFrame([row])))
        model.predict_proba(X)
        if k >= 5:  # first calls are warm-up
            ms.append((time.perf_counter() - t0) * 1000)
    print(f"transform + predict, one URL at a time (n={len(ms)}): "
          f"median {np.median(ms):.1f} ms, p95 {pct(ms, 95):.1f} ms")
    report["transform_predict_ms"] = {"n": len(ms), "median": float(np.median(ms)), "p95": pct(ms, 95)}

    ex = URLFeatureExtractor(fetch_content=False, fetch_whois=False)
    urls = test["url"].sample(n=min(500, len(test)), random_state=1).tolist()
    lx = []
    for u in urls:
        t0 = time.perf_counter()
        ex.extract(u)
        lx.append((time.perf_counter() - t0) * 1000)
    print(f"lexical feature extraction, offline (n={len(lx)}): median {np.median(lx):.2f} ms, p95 {pct(lx, 95):.2f} ms")
    report["lexical_extract_ms"] = {"median": float(np.median(lx)), "p95": pct(lx, 95)}

    try:
        import shap
        expl = shap.TreeExplainer(model)
        sh = []
        for _, row in sample.head(40).iterrows():
            X = model_matrix(model, transformer.transform(pd.DataFrame([row])))
            t0 = time.perf_counter()
            expl.shap_values(X)
            sh.append((time.perf_counter() - t0) * 1000)
        sh = sh[3:]
        print(f"SHAP explanation, one URL (n={len(sh)}): median {np.median(sh):.1f} ms, p95 {pct(sh, 95):.1f} ms")
        report["shap_ms"] = {"median": float(np.median(sh)), "p95": pct(sh, 95)}
    except Exception as e:  # noqa: BLE001
        print(f"SHAP timing skipped ({type(e).__name__}: {e})")

    if a.server:
        import requests
        legit = test[test["label"] == 0]["url"].sample(n=a.n_live, random_state=7).tolist()
        print(f"\nEnd-to-end via {a.server}/predict, {len(legit)} legitimate URLs (live WHOIS + page fetch):")
        el = []
        for i, u in enumerate(legit):
            t0 = time.perf_counter()
            r = requests.post(f"{a.server}/predict", json={"url": u}, timeout=90)
            dt = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                print(f"  HTTP {r.status_code} on first call: {r.text[:300]}\n  -> request schema differs "
                      f"from {{'url': ...}}; stopping live test.")
                break
            el.append(dt)
        if el:
            print(f"  median {np.median(el) / 1000:.2f} s, p95 {pct(el, 95) / 1000:.2f} s, max {max(el) / 1000:.2f} s")
            report["end_to_end_s"] = {"n": len(el), "median": float(np.median(el)) / 1000,
                                      "p95": pct(el, 95) / 1000, "max": max(el) / 1000}

    (FINAL / "drift_latency_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nSaved -> {FINAL / 'drift_latency_report.json'}")


if __name__ == "__main__":
    main()
