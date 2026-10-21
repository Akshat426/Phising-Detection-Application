"""Train extra models (LightGBM, SVM-RBF, HistGradientBoosting) with the same
protocol as the original four: 25-config random search, 5-fold stratified CV on
train, validation for comparison, test set scored once and never used to choose.

Place at: src/training/extra_models.py
Run from project root, e.g.:
    python src\\training\\extra_models.py --models svm_rbf
    python src\\training\\extra_models.py --models lightgbm
    python src\\training\\extra_models.py --models hist_gb
Each model writes its own files, so separate runs never overwrite each other.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC

SEED = 42
N_ITER = 25


def metrics(y, pred, proba):
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def clean(params):
    out = {}
    for k, v in params.items():
        if isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.floating):
            v = float(v)
        out[k] = v
    return out


def build_specs():
    specs = {
        "svm_rbf": (
            SVC(kernel="rbf", random_state=SEED),
            {"C": loguniform(0.1, 100), "gamma": loguniform(1e-3, 1.0)},
            -1,
        ),
        "hist_gb": (
            HistGradientBoostingClassifier(random_state=SEED),
            {
                "learning_rate": loguniform(0.03, 0.3),
                "max_iter": randint(100, 500),
                "max_leaf_nodes": randint(8, 64),
                "max_depth": randint(3, 12),
                "min_samples_leaf": randint(10, 50),
                "l2_regularization": loguniform(1e-3, 10),
            },
            -1,
        ),
    }
    try:
        from lightgbm import LGBMClassifier
        specs["lightgbm"] = (
            LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=4),
            {
                "n_estimators": randint(100, 600),
                "learning_rate": loguniform(0.02, 0.3),
                "num_leaves": randint(8, 64),
                "max_depth": randint(2, 12),
                "subsample": uniform(0.6, 0.4),
                "subsample_freq": [1],
                "colsample_bytree": uniform(0.5, 0.5),
                "min_child_samples": randint(5, 40),
                "reg_lambda": loguniform(1e-3, 10),
            },
            1,  # single-process search for LightGBM on Windows
        )
    except Exception as e:  # import failure
        print(f"LightGBM could not be imported: {e}")
    return specs


def main(root: Path, wanted):
    proc = root / "data" / "processed"
    out_dir = root / "final_model" / "extra_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(proc / "train_transformed.csv")
    va = pd.read_csv(proc / "validation_transformed.csv")
    te = pd.read_csv(proc / "test_transformed.csv")
    feat_cols = [c for c in tr.columns if c != "label"]
    print("Feature dtypes:", dict(tr[feat_cols].dtypes.astype(str).value_counts()))

    def to_arr(df):
        return np.ascontiguousarray(df[feat_cols].to_numpy(dtype=np.float64))

    Xtr, ytr = to_arr(tr), tr["label"].to_numpy(dtype=np.int64)
    Xva, yva = to_arr(va), va["label"].to_numpy(dtype=np.int64)
    Xte, yte = to_arr(te), te["label"].to_numpy(dtype=np.int64)
    print(f"Train {Xtr.shape}, Validation {Xva.shape}, Test {Xte.shape}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    specs = build_specs()
    res_path = root / "final_model" / "extra_models_results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}

    for name in wanted:
        if name not in specs:
            print(f"\nSkipping {name}: not available.")
            continue
        est, space, search_jobs = specs[name]
        print(f"\n=== {name}: random search ({N_ITER} iterations, 5-fold CV on train) ===")
        t0 = time.time()
        try:
            search = RandomizedSearchCV(est, space, n_iter=N_ITER, scoring="f1",
                                        cv=cv, random_state=SEED,
                                        n_jobs=search_jobs, refit=True)
            search.fit(Xtr, ytr)
            best = search.best_estimator_
            if name == "svm_rbf":  # Platt scaling only for the winning config
                best = SVC(kernel="rbf", probability=True, random_state=SEED,
                           **search.best_params_).fit(Xtr, ytr)
            p_va = best.predict_proba(Xva)[:, 1]
            p_te = best.predict_proba(Xte)[:, 1]
        except Exception:
            print(f"!! {name} FAILED - continuing with the next model.")
            traceback.print_exc(limit=2)
            continue

        print(f"  Best CV F1 (train): {search.best_score_:.4f}")
        print(f"  Best params: {clean(search.best_params_)}")
        va_m = metrics(yva, (p_va >= 0.5).astype(int), p_va)
        te_m = metrics(yte, (p_te >= 0.5).astype(int), p_te)
        print(f"  Validation metrics: { {k: round(v, 4) for k, v in va_m.items()} }")
        print(f"  TEST metrics (reported once, not used for any decision): "
              f"{ {k: round(v, 4) for k, v in te_m.items()} }")
        print(f"  Elapsed: {time.time() - t0:.0f}s")

        joblib.dump(best, out_dir / f"{name}.pkl")
        pd.DataFrame({
            "split": ["val"] * len(yva) + ["test"] * len(yte),
            "y_true": list(yva) + list(yte),
            "p": list(p_va) + list(p_te),
        }).to_csv(out_dir / f"{name}_probabilities.csv", index=False)
        results[name] = {
            "best_cv_f1_train": float(search.best_score_),
            "best_params": clean(search.best_params_),
            "validation": va_m,
            "test": te_m,
        }
        res_path.write_text(json.dumps(results, indent=2))
        print(f"  Saved {name} -> {out_dir}")

    print(f"\nResults file: {res_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parents[2])
    ap.add_argument("--models", nargs="+",
                    default=["svm_rbf", "lightgbm", "hist_gb"])
    a = ap.parse_args()
    main(a.root, a.models)
