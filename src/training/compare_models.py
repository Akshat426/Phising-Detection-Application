"""Six-model comparison on the held-out TEST set (scored once, no tuning here).

Produces: metrics table, paired-bootstrap 95% CIs, McNemar tests vs. the best
model, error-overlap analysis and a soft-voting preview (is stacking worthwhile?).

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
N_BOOT = 2000

# Best parameters printed by model_trainer.py on the augmented training set.
REFIT = {
    "logistic_regression": (LogisticRegression(C=3.338086111390218, penalty="l2",
                            solver="liblinear", max_iter=2000, random_state=SEED), 0.9636),
    "decision_tree": (DecisionTreeClassifier(criterion="gini", max_depth=11,
                      min_samples_leaf=1, min_samples_split=9, random_state=SEED), 0.9566),
    "random_forest": (RandomForestClassifier(max_depth=17, max_features="sqrt",
                      min_samples_leaf=1, min_samples_split=8, n_estimators=108,
                      random_state=SEED, n_jobs=-1), 0.9766),
}


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
    raise TypeError(f"Cannot find a model with predict_proba in {type(obj)}")


def m(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred),
                recall=recall_score(y, pred), f1=f1_score(y, pred),
                roc_auc=roc_auc_score(y, p))


def main(root: Path):
    proc = root / "data" / "processed"
    fm = root / "final_model"
    tr = pd.read_csv(proc / "train_transformed.csv")
    va = pd.read_csv(proc / "validation_transformed.csv")
    te = pd.read_csv(proc / "test_transformed.csv")
    cols = [c for c in tr.columns if c != "label"]
    arr = lambda d: np.ascontiguousarray(d[cols].to_numpy(dtype=np.float64))
    Xtr, ytr = arr(tr), tr["label"].to_numpy()
    Xva, yva = arr(va), va["label"].to_numpy()
    Xte, yte = arr(te), te["label"].to_numpy()

    probs = {}

    # 1) refit the three original models that were not saved as files
    for name, (est, expected) in REFIT.items():
        est.fit(Xtr, ytr)
        vf1 = f1_score(yva, est.predict(Xva))
        flag = "OK" if abs(vf1 - expected) < 0.002 else "WARNING: differs from trainer output"
        print(f"{name:22s} refit val F1 {vf1:.4f} (trainer reported {expected:.4f})  {flag}")
        probs[name] = est.predict_proba(Xte)[:, 1]

    # 2) saved models
    for name, path in [("svm_rbf", fm / "extra_models" / "svm_rbf.pkl"),
                       ("hist_gb", fm / "extra_models" / "hist_gb.pkl")]:
        est = unwrap(load_any(path))
        probs[name] = est.predict_proba(Xte)[:, 1]
        print(f"{name:22s} loaded from {path.name}; val F1 "
              f"{f1_score(yva, est.predict(Xva)):.4f}")

    xgb = unwrap(load_any(fm / "model.pkl"))
    try:
        p = xgb.predict_proba(te[cols])[:, 1]
        pv = xgb.predict(va[cols])
    except Exception:
        p = xgb.predict_proba(Xte)[:, 1]
        pv = xgb.predict(Xva)
    probs["xgboost"] = p
    print(f"{'xgboost':22s} loaded from model.pkl; val F1 {f1_score(yva, pv):.4f}")

    order = ["xgboost", "hist_gb", "random_forest", "logistic_regression",
             "svm_rbf", "decision_tree"]

    # ---- metrics + paired bootstrap CIs (same resamples for every model)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(yte), size=(N_BOOT, len(yte)))
    rows, ci = [], {}
    for n in order:
        met = m(yte, probs[n])
        f1s = np.array([f1_score(yte[i], (probs[n][i] >= 0.5).astype(int)) for i in idx])
        aucs = np.array([roc_auc_score(yte[i], probs[n][i]) for i in idx])
        ci[n] = {"f1": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
                 "auc": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]}
        rows.append({"model": n, **{k: round(v, 4) for k, v in met.items()},
                     "f1_ci95": f"[{ci[n]['f1'][0]:.4f}, {ci[n]['f1'][1]:.4f}]",
                     "auc_ci95": f"[{ci[n]['auc'][0]:.4f}, {ci[n]['auc'][1]:.4f}]"})
    table = pd.DataFrame(rows)
    print("\n=== TEST SET RESULTS (n=%d), 95%% paired-bootstrap CIs, %d resamples ===" % (len(yte), N_BOOT))
    print(table.to_string(index=False))

    # ---- McNemar (exact) vs xgboost
    wrong = {n: ((probs[n] >= 0.5).astype(int) != yte) for n in order}
    print("\n=== McNemar exact test: each model vs xgboost (errors only) ===")
    mc = {}
    for n in order[1:]:
        b = int(np.sum(wrong["xgboost"] & ~wrong[n]))   # xgb wrong, other right
        c = int(np.sum(~wrong["xgboost"] & wrong[n]))   # xgb right, other wrong
        pval = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
        mc[n] = {"xgb_only_wrong": b, "other_only_wrong": c, "p_value": pval}
        print(f"  {n:22s} xgb-only-wrong={b:3d}  other-only-wrong={c:3d}  p={pval:.4f}")

    # ---- error overlap
    print("\n=== ERROR OVERLAP (test set) ===")
    for n in order:
        print(f"  {n:22s} errors: {int(wrong[n].sum())}")
    print("  Pairwise Jaccard of error sets (1.0 = identical mistakes):")
    for a, b in combinations(order[:4], 2):
        u = np.sum(wrong[a] | wrong[b])
        j = np.sum(wrong[a] & wrong[b]) / u if u else 1.0
        print(f"    {a} vs {b}: {j:.2f}")
    others = np.any([wrong[n] for n in order[1:4]], axis=0)
    fixable = int(np.sum(wrong["xgboost"] & ~np.all([wrong[n] for n in order[1:4]], axis=0)))
    print(f"  xgboost errors: {int(wrong['xgboost'].sum())}; at least one of the next "
          f"three models gets {fixable} of them right")

    # ---- soft-vote preview
    vote = np.mean([probs[n] for n in order[:3]], axis=0)
    vm = m(yte, vote)
    print("\n=== SOFT-VOTE PREVIEW (mean of xgboost + hist_gb + random_forest) ===")
    print("  " + ", ".join(f"{k} {v:.4f}" for k, v in vm.items()))
    print(f"  errors: {int(((vote >= 0.5).astype(int) != yte).sum())} "
          f"(best single model: {int(wrong['xgboost'].sum())})")

    out = {"metrics": rows, "ci95": ci, "mcnemar_vs_xgboost": mc,
           "soft_vote": vm, "n_test": int(len(yte))}
    (fm / "six_model_comparison.json").write_text(json.dumps(out, indent=2, default=float))
    table.to_csv(fm / "six_model_comparison.csv", index=False)
    pd.DataFrame({"y_true": yte, **{n: probs[n] for n in order}}).to_csv(
        fm / "test_probabilities_all_models.csv", index=False)
    print(f"\nSaved -> {fm / 'six_model_comparison.json'} (+ .csv, test_probabilities_all_models.csv)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    main(ap.parse_args().root)
