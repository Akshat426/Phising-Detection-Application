"""
Export per-model evaluation data (ROC and precision-recall curves, confusion
matrices) for the web UI's Model Analysis page.

Why this script exists: model_trainer.py saves only summary metrics for the
four candidate models, but the page also shows curves and confusion matrices
for ALL four. Each candidate's fitted model is stored in MLflow, so this
script locates it there and re-scores the validation set.

Safety check: a candidate model is only accepted if re-scoring reproduces the
F1 and ROC-AUC saved in final_model/model_comparison.json (to 4 decimals).
That guarantees the curves belong to the models that were actually compared,
not to an older experiment run that happens to share a model name.

The test set is scored for the SELECTED model only, matching the project's
"test set used once, for the winner" protocol.

Usage (from the project root):
    python src/training/export_model_analysis.py
    python src/training/export_model_analysis.py --mlruns path/to/mlruns

Output: final_model/model_analysis_data.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
MAX_POINTS = 150  # curve points sent to the browser (plenty for a smooth line)


def load_scorer(path: str):
    """Return a function X(DataFrame) -> P(phishing), for a saved model file."""
    if path.endswith(".xgb"):
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(path)
        # These boosters were saved without feature names; column ORDER of the
        # transformed CSVs is the training order, so we score by position.
        return lambda X: booster.predict(xgb.DMatrix(X.values))
    model = joblib.load(path)
    cols = getattr(model, "feature_names_in_", None)
    return lambda X: model.predict_proba(X[cols] if cols is not None else X)[:, 1]


def thin(*arrays, n=MAX_POINTS):
    """Evenly thin parallel arrays to at most n points, keeping both endpoints."""
    length = len(arrays[0])
    if length <= n:
        return [np.round(a, 5).tolist() for a in arrays]
    idx = np.unique(np.linspace(0, length - 1, n).astype(int))
    return [np.round(a[idx], 5).tolist() for a in arrays]


def find_matching_model(name: str, mlruns: str, X_val, y_val, saved: dict):
    """Search MLflow artifacts for the model whose validation scores match the saved JSON."""
    candidates = glob.glob(f"{mlruns}/*/*/artifacts/{name}/model.pkl") + glob.glob(
        f"{mlruns}/*/*/artifacts/{name}/model.xgb"
    )
    candidates.sort(key=os.path.getmtime, reverse=True)
    for path in candidates:
        try:
            p = load_scorer(path)(X_val)
        except Exception:
            continue
        f1 = f1_score(y_val, (p >= 0.5).astype(int))
        auc = roc_auc_score(y_val, p)
        if abs(f1 - saved["f1"]) < 1e-4 and abs(auc - saved["roc_auc"]) < 1e-4:
            return path, p, f1, auc
    return None, None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlruns", default=str(ROOT / "mlruns"))
    ap.add_argument("--model-dir", default=str(ROOT / "final_model"))
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()

    with open(Path(args.model_dir) / "model_comparison.json", encoding="utf-8") as f:
        comparison = json.load(f)
    saved = comparison["validation_results"]
    selected = comparison["selected_model"]

    val = pd.read_csv(Path(args.data_dir) / "validation_transformed.csv")
    y_val = val.pop("label").values
    test = pd.read_csv(Path(args.data_dir) / "test_transformed.csv")
    y_test = test.pop("label").values

    out = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "validation_size": int(len(y_val)),
        "validation_positives": int(y_val.sum()),
        "threshold": 0.5,
        "models": {},
    }
    selected_path = None

    for name in MODEL_NAMES:
        if name not in saved:
            continue
        path, p, f1, auc = find_matching_model(name, args.mlruns, val, y_val, saved[name])
        if path is None:
            sys.exit(
                f"Could not find a saved '{name}' model in {args.mlruns} whose validation "
                f"F1/AUC match model_comparison.json. Re-run model_trainer.py, or pass --mlruns."
            )
        if name == selected:
            selected_path = path
        tn, fp, fn, tp = confusion_matrix(y_val, (p >= 0.5).astype(int)).ravel()
        fpr, tpr, _ = roc_curve(y_val, p)
        prec, rec, _ = precision_recall_curve(y_val, p)
        fpr_t, tpr_t = thin(fpr, tpr)
        rec_t, prec_t = thin(rec, prec)
        out["models"][name] = {
            "verified": {"f1": round(float(f1), 6), "roc_auc": round(float(auc), 6)},
            "average_precision": round(float(average_precision_score(y_val, p)), 6),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "roc": {"fpr": fpr_t, "tpr": tpr_t},
            "pr": {"recall": rec_t, "precision": prec_t},
        }
        print(f"  {name:<20} verified (F1 {f1:.4f}, AUC {auc:.4f})  <- {Path(path).parts[-4][:8]}")

    p_test = load_scorer(selected_path)(test)
    tn, fp, fn, tp = confusion_matrix(y_test, (p_test >= 0.5).astype(int)).ravel()
    out["test_selected"] = {
        "model": selected,
        "size": int(len(y_test)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    dest = Path(args.model_dir) / "model_analysis_data.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Wrote {dest} ({dest.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
