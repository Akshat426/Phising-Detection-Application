"""Six-model comparison on the held-out TEST set (scored once, no tuning here).

Loads the exact fitted models from disk (no refitting):
    final_model/all_models/{logistic_regression,decision_tree,random_forest,xgboost}.pkl
        -> written by model_trainer.py
    final_model/extra_models/{svm_rbf,hist_gb}.pkl  -> written by extra_models.py
Each model's validation F1 is recomputed and checked against the JSON the training
scripts saved, so a stale or mismatched file is caught instead of silently used.
The "best" model is chosen on VALIDATION F1 (same rule as model_trainer.py).

Produces: metrics table with paired-bootstrap 95% CIs, McNemar exact tests vs. the
validation-best model, error-overlap analysis and a soft-voting preview.

Place at: src/training/compare_models.py
Run from project root:  python src\\training\\compare_models.py
"""
import argparse
import json
import pickle
import warnings
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

warnings.filterwarnings("ignore")
SEED = 42
N_BOOT = 2000
FOUR = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
TWO = ["svm_rbf", "hist_gb"]


def load_any(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def unwrap(obj):
    if hasattr(obj, "predict_proba"):
        return obj
    if isinstance(obj, dict):
        for k in ("model", "classifier", "clf", "estimator"):
            if k in obj and hasattr(obj[k], "predict_proba"):
                return obj[k]
    raise TypeError(f"No model with predict_proba found in {type(obj)}")


def metr(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred),
                recall=recall_score(y, pred), f1=f1_score(y, pred),
                roc_auc=roc_auc_score(y, p))


def main(root: Path):
    proc, fm = root / "data" / "processed", root / "final_model"
    va = pd.read_csv(proc / "validation_transformed.csv")
    te = pd.read_csv(proc / "test_transformed.csv")
    cols = [c for c in va.columns if c != "label"]
    yva, yte = va["label"].to_numpy(), te["label"].to_numpy()
    frames = {"val": va[cols], "test": te[cols]}
    arrays = {k: np.ascontiguousarray(v.to_numpy(dtype=np.float64)) for k, v in frames.items()}

    expected = {n: v["f1"] for n, v in
                json.loads((fm / "model_comparison.json").read_text())["validation_results"].items()}
    expected.update({n: v["validation"]["f1"] for n, v in
                     json.loads((fm / "extra_models_results.json").read_text()).items()})

    paths = {n: fm / "all_models" / f"{n}.pkl" for n in FOUR}
    paths.update({n: fm / "extra_models" / f"{n}.pkl" for n in TWO})
    use_frames = set(FOUR)  # these were trained on DataFrames; the extras on arrays

    probs_v, probs_t = {}, {}
    print("Model                  val F1 (recomputed)  saved val F1   check")
    for n, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - re-run model_trainer.py / extra_models.py")
        mdl = unwrap(load_any(path))
        src = frames if n in use_frames else arrays
        try:
            pv, pt = mdl.predict_proba(src["val"])[:, 1], mdl.predict_proba(src["test"])[:, 1]
        except Exception:
            other = arrays if src is frames else frames
            pv, pt = mdl.predict_proba(other["val"])[:, 1], mdl.predict_proba(other["test"])[:, 1]
        probs_v[n], probs_t[n] = pv, pt
        vf1 = f1_score(yva, (pv >= 0.5).astype(int))
        ok = abs(vf1 - expected[n]) < 1e-4
        print(f"{n:22s} {vf1:.4f}               {expected[n]:.4f}         "
              f"{'OK' if ok else 'MISMATCH - files are from different runs!'}")

    val_f1 = {n: f1_score(yva, (probs_v[n] >= 0.5).astype(int)) for n in paths}
    order = sorted(paths, key=lambda n: -val_f1[n])
    best = order[0]
    print(f"\nValidation-best model among the six: {best} (val F1 {val_f1[best]:.4f})")

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(yte), size=(N_BOOT, len(yte)))
    rows, ci = [], {}
    for n in order:
        met = metr(yte, probs_t[n])
        f1s = np.array([f1_score(yte[i], (probs_t[n][i] >= 0.5).astype(int)) for i in idx])
        aucs = np.array([roc_auc_score(yte[i], probs_t[n][i]) for i in idx])
        ci[n] = {"f1": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
                 "auc": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]}
        rows.append({"model": n, "val_f1": round(val_f1[n], 4),
                     **{k: round(v, 4) for k, v in met.items()},
                     "f1_ci95": f"[{ci[n]['f1'][0]:.4f}, {ci[n]['f1'][1]:.4f}]",
                     "auc_ci95": f"[{ci[n]['auc'][0]:.4f}, {ci[n]['auc'][1]:.4f}]"})
    table = pd.DataFrame(rows)
    print(f"\n=== TEST SET RESULTS (n={len(yte)}), 95% paired-bootstrap CIs, {N_BOOT} resamples ===")
    print(table.to_string(index=False))

    wrong = {n: ((probs_t[n] >= 0.5).astype(int) != yte) for n in order}
    print(f"\n=== McNemar exact test: each model vs {best} (discordant test URLs) ===")
    mc = {}
    for n in order[1:]:
        b = int(np.sum(wrong[best] & ~wrong[n]))
        c = int(np.sum(~wrong[best] & wrong[n]))
        pval = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
        mc[n] = {"best_only_wrong": b, "other_only_wrong": c, "p_value": pval}
        print(f"  {n:22s} {best}-only-wrong={b:3d}  other-only-wrong={c:3d}  p={pval:.4f}")
    print(f"  (Bonferroni threshold for {len(order)-1} comparisons: {0.05/(len(order)-1):.4f})")

    print("\n=== ERROR OVERLAP (test set) ===")
    for n in order:
        print(f"  {n:22s} errors: {int(wrong[n].sum())}")
    top4 = order[:4]
    print("  Pairwise Jaccard of error sets among the top four (1.0 = identical mistakes):")
    for a, b in combinations(top4, 2):
        u = np.sum(wrong[a] | wrong[b])
        print(f"    {a} vs {b}: {(np.sum(wrong[a] & wrong[b]) / u) if u else 1.0:.2f}")
    nxt = order[1:4]
    fixable = int(np.sum(wrong[best] & ~np.all([wrong[n] for n in nxt], axis=0)))
    print(f"  {best} errors: {int(wrong[best].sum())}; at least one of {nxt} gets {fixable} of them right")

    top3 = order[:3]
    vote = np.mean([probs_t[n] for n in top3], axis=0)
    vm = metr(yte, vote)
    print(f"\n=== SOFT-VOTE PREVIEW (mean probability of {', '.join(top3)}) ===")
    print("  " + ", ".join(f"{k} {v:.4f}" for k, v in vm.items()))
    print(f"  errors: {int(((vote >= 0.5).astype(int) != yte).sum())} (best single model: {int(wrong[best].sum())})")

    out = {"selected_on_validation": best, "validation_f1": val_f1, "metrics": rows,
           "ci95": ci, "mcnemar_vs_best": mc, "soft_vote_top3": {"models": top3, **vm},
           "n_test": int(len(yte))}
    (fm / "six_model_comparison.json").write_text(json.dumps(out, indent=2, default=float))
    table.to_csv(fm / "six_model_comparison.csv", index=False)
    pd.DataFrame({"y_true": yte, **{n: probs_t[n] for n in order}}).to_csv(
        fm / "test_probabilities_all_models.csv", index=False)
    print(f"\nSaved -> {fm / 'six_model_comparison.json'} (+ .csv, test_probabilities_all_models.csv)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    main(ap.parse_args().root)
