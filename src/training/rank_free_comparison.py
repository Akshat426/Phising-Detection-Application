"""
Paired comparison: rank-free models vs the full model, on the SAME test URLs.

The rank-free models (tranco_rank and tranco_rank_was_missing excluded) were
already trained by model_trainer.py --exclude-features ... into
final_model/all_models_ablation_no_rank/. Nothing is retrained here. This
script only adds the statistics that were missing:

  * test metrics side by side (F1, precision, recall, AUC, FP, FN)
  * exact McNemar test of each rank-free model against the full XGBoost
    (Bonferroni-adjusted over the number of comparisons)
  * paired bootstrap 95% CI for the F1 difference (full minus rank-free)
  * false-alarm rate on held-out legitimate URLs (validation + test pooled,
    n legit shown) next to the full model's false-alarm rate when the rank is
    unavailable at inference, read from diagnostics_report.json

Safety check: every model's re-scored validation F1 must match the F1 saved in
the JSON files (4 decimals), otherwise the script stops.

Usage (project root):
    python src/training/rank_free_comparison.py
Output: final_model/rank_free_comparison.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
MD = ROOT / "final_model"
PROC = ROOT / "data" / "processed"
N_BOOT = 10_000
SEED = 42

FULL = ("full XGBoost", MD / "all_models" / "xgboost.pkl")
NO_RANK = {
    "xgboost": "no-rank XGBoost",
    "random_forest": "no-rank Random Forest",
    "decision_tree": "no-rank Decision Tree",
    "logistic_regression": "no-rank Logistic Regression",
}


def load_scorer(path: Path):
    obj = joblib.load(path)
    if isinstance(obj, dict):
        obj = obj.get("model", obj)
    cols = getattr(obj, "feature_names_in_", None)
    if cols is None and hasattr(obj, "get_booster"):
        cols = obj.get_booster().feature_names
    return lambda X: obj.predict_proba(X[list(cols)] if cols is not None else X)[:, 1]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def main():
    with open(MD / "six_model_comparison.json", encoding="utf-8") as f:
        six = json.load(f)
    with open(MD / "model_comparison_ablation_no_rank.json", encoding="utf-8") as f:
        abl = json.load(f)
    saved_val = {"full": six["validation_f1"]["xgboost"]}
    for k in NO_RANK:
        saved_val[k] = abl["validation_results"][k]["f1"]

    val = pd.read_csv(PROC / "validation_transformed.csv")
    yv = val.pop("label").values
    test = pd.read_csv(PROC / "test_transformed.csv")
    yt = test.pop("label").values

    entries = [("full", FULL[0], FULL[1])] + [
        (k, lab, MD / "all_models_ablation_no_rank" / f"{k}.pkl") for k, lab in NO_RANK.items()
    ]
    P_test, P_val, labels = {}, {}, {}
    for key, label, path in entries:
        if not path.exists():
            sys.exit(f"Missing model file: {path}")
        sc = load_scorer(path)
        pv, pt = sc(val), sc(test)
        f1v = f1_score(yv, (pv >= 0.5).astype(int))
        if abs(f1v - saved_val[key]) > 1e-4:
            sys.exit(f"{label}: re-scored validation F1 {f1v:.4f} != saved {saved_val[key]:.4f}. "
                     "Transformed CSVs and model files are out of sync.")
        P_val[key], P_test[key], labels[key] = pv, pt, label
        print(f"  verified {label:<28} validation F1 {f1v:.4f}")

    pred = {k: (p >= 0.5).astype(int) for k, p in P_test.items()}
    pred_val = {k: (p >= 0.5).astype(int) for k, p in P_val.items()}
    n_legit_t = int((yt == 0).sum())
    legit_pool = np.concatenate([yv == 0, yt == 0])

    diag = {}
    dpath = MD / "diagnostics_report.json"
    if dpath.exists():
        with open(dpath, encoding="utf-8") as f:
            diag = json.load(f)

    rows = {}
    print(f"\nTEST SET (n = {len(yt)}, phishing {int(yt.sum())}, legitimate {n_legit_t})")
    print(f"{'model':<30}{'F1':>8}{'prec':>8}{'rec':>8}{'AUC':>8}{'FP':>5}{'FN':>5}{'FPR val+test legit [95% CI]':>34}")
    for key in labels:
        p = pred[key]
        fp = int(((p == 1) & (yt == 0)).sum())
        fn = int(((p == 0) & (yt == 1)).sum())
        pooled_pred = np.concatenate([pred_val[key], p])
        fa = int(pooled_pred[legit_pool].sum())
        n_leg = int(legit_pool.sum())
        lo, hi = wilson(fa, n_leg)
        rows[key] = {
            "label": labels[key],
            "f1": float(f1_score(yt, p)),
            "precision": float(((p == 1) & (yt == 1)).sum() / max(p.sum(), 1)),
            "recall": float(((p == 1) & (yt == 1)).sum() / yt.sum()),
            "auc": float(roc_auc_score(yt, P_test[key])),
            "fp": fp, "fn": fn,
            "pooled_legit_false_alarms": fa, "pooled_legit_n": n_leg,
            "pooled_legit_fpr": fa / n_leg, "pooled_legit_fpr_ci95": [lo, hi],
        }
        r = rows[key]
        print(f"{labels[key]:<30}{r['f1']:>8.4f}{r['precision']:>8.4f}{r['recall']:>8.4f}{r['auc']:>8.4f}"
              f"{fp:>5}{fn:>5}{r['pooled_legit_fpr']:>12.1%} [{lo:.1%}, {hi:.1%}]")

    if "false_alarm_rate_rank_missing" in diag:
        print(f"\nContext: the FULL model's false-alarm rate on held-out legitimate URLs is "
              f"{diag['false_alarm_rate_rank_present']:.1%} with the rank present and "
              f"{diag['false_alarm_rate_rank_missing']:.1%} when the rank is unavailable "
              f"(diagnostics_report.json, n = {diag.get('legit_heldout')}).")

    # ---- paired statistics: full vs each rank-free model, same test URLs
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(yt), size=(N_BOOT, len(yt)))
    yb = yt[idx]

    def boot_f1(p):
        pb = p[idx]
        tp = ((pb == 1) & (yb == 1)).sum(1)
        fp = ((pb == 1) & (yb == 0)).sum(1)
        fn = ((pb == 0) & (yb == 1)).sum(1)
        return 2 * tp / np.maximum(2 * tp + fp + fn, 1)

    f_full = boot_f1(pred["full"])
    others = [k for k in labels if k != "full"]
    alpha = 0.05 / len(others)
    print(f"\nPAIRED COMPARISON vs full XGBoost (same {len(yt)} test URLs; Bonferroni alpha = {alpha:.4f})")
    print(f"{'rank-free model':<30}{'dF1 (full-nr) [95% CI]':>30}{'nr wrong, full right':>22}{'full wrong, nr right':>22}{'McNemar p':>12}  verdict")
    paired = {}
    c_full = pred["full"] == yt
    for k in others:
        c_o = pred[k] == yt
        b = int((c_full & ~c_o).sum())  # other wrong, full right
        c = int((~c_full & c_o).sum())  # full wrong, other right
        pval = float(binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) > 0 else 1.0
        d = f_full - boot_f1(pred[k])
        lo, hi = np.percentile(d, [2.5, 97.5])
        dpt = rows["full"]["f1"] - rows[k]["f1"]
        verdict = "full better" if (pval < alpha and b > c) else ("rank-free better" if (pval < alpha and c > b) else "no detectable difference")
        paired[k] = {"label": labels[k], "delta_f1": dpt, "delta_f1_ci95": [float(lo), float(hi)],
                     "other_wrong_full_right": b, "full_wrong_other_right": c,
                     "mcnemar_p": pval, "bonferroni_alpha": alpha, "verdict": verdict}
        print(f"{labels[k]:<30}{dpt:>+12.4f} [{lo:+.4f}, {hi:+.4f}]{b:>22}{c:>22}{pval:>12.4f}  {verdict}")

    out = {"n_test": int(len(yt)), "n_bootstrap": N_BOOT, "seed": SEED,
           "models": rows, "paired_vs_full": paired,
           "full_model_false_alarm_rank_present": diag.get("false_alarm_rate_rank_present"),
           "full_model_false_alarm_rank_missing": diag.get("false_alarm_rate_rank_missing")}
    dest = MD / "rank_free_comparison.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
