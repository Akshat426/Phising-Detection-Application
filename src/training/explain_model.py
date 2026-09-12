"""
SHAP explainability for the trained phishing detection model.

Produces:
  1. A global summary plot (which features matter most, and in which
     direction) - the headline visual for your CP presentation.
  2. A global mean-|SHAP| bar chart and a matching JSON table, for a
     text-based feature ranking to cite in your report.
  3. Per-example waterfall plots for a correctly-caught phishing URL and a
     correctly-cleared legitimate URL - shows exactly which features drove
     one specific prediction, which is what the live app's UI would show
     a user in Phase 5.

Design note: SHAP's output shape differs by model type (binary tree models
vs. multi-class-shaped output vs. linear models), and this project's winning
model type varies run to run (Logistic Regression, Decision Tree, Random
Forest, and XGBoost have all won across different runs so far). This script
normalizes all of those cases explicitly rather than assuming one shape,
since a script that only works for whichever model happened to win last time
would be fragile.

Usage:
    python explain_model.py
    python explain_model.py --model-path final_model/model_no_tranco.pkl  (explain a specific saved model)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless - just save files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

MODELS_DIR = Path(__file__).resolve().parents[2] / "final_model"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_DIR = MODELS_DIR / "explainability"

BACKGROUND_SAMPLE_SIZE = 100  # keeps SHAP computation fast; doesn't need the full training set
TEST_SAMPLE_SIZE = 200  # cap test-set SHAP computation size for speed on larger datasets


def normalize_shap_values(shap_values) -> np.ndarray:
    """Return a plain (n_samples, n_features) array of SHAP values for the
    POSITIVE (phishing) class, regardless of which model type produced them.

    Different model/explainer combinations return different shapes:
      - Binary tree models (XGBoost, sometimes Decision Tree) via the modern
        shap.Explainer: (n_samples, n_features) directly.
      - Random Forest classifiers: sometimes (n_samples, n_features, n_classes)
        - one SHAP value per class per feature.
      - Linear models (Logistic Regression): (n_samples, n_features), same as
        the tree case, computed relative to the linear decision function.

    Rather than branching on model type by name (which breaks the moment a
    different model wins a future run), we branch on the actual returned
    array shape, which is more robust.
    """
    values = shap_values.values
    if values.ndim == 3:
        # (n_samples, n_features, n_classes) - take the positive class (index 1)
        return values[:, :, 1]
    return values


def get_base_value(shap_values) -> float:
    base = shap_values.base_values
    base = np.asarray(base)
    if base.ndim == 2:
        return float(base[0, 1])  # multi-class shape: positive class base value
    return float(np.atleast_1d(base)[0])


def main():
    parser = argparse.ArgumentParser(description="Generate SHAP explainability report for the trained model")
    parser.add_argument("--model-path", type=str, default=None,
                         help="Path to a specific model .pkl (default: final_model/model.pkl)")
    args = parser.parse_args()

    model_path = Path(args.model_path) if args.model_path else MODELS_DIR / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found - run model_trainer.py first.")

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    train_df = pd.read_csv(PROCESSED_DIR / "train_transformed.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_transformed.csv")
    X_train = train_df.drop(columns=["label"])
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    # Some saved models (from --exclude-features runs) were trained on fewer
    # columns than the current transformed CSVs have. Align to whatever
    # columns the model actually expects, so this script works for ablation
    # models too, not just the main one.
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):  # XGBoost
        expected_cols = model.get_booster().feature_names
    else:
        expected_cols = list(X_train.columns)
    X_train = X_train[expected_cols]
    X_test = X_test[expected_cols]

    background = X_train.sample(min(BACKGROUND_SAMPLE_SIZE, len(X_train)), random_state=42)
    test_sample = X_test.sample(min(TEST_SAMPLE_SIZE, len(X_test)), random_state=42)
    y_test_sample = y_test.loc[test_sample.index]

    # Tree-based models (XGBoost, Decision Tree, Random Forest) use
    # TreeExplainer with tree_path_dependent perturbation, which reads
    # feature-split statistics directly from the trained trees rather than
    # needing background data. This also sidesteps a real, specific SHAP/
    # XGBoost incompatibility: the default "interventional" perturbation
    # mode raises NotImplementedError on XGBoost models with categorical
    # splits, discovered by testing this script against all 4 model types
    # this project can produce - it would otherwise fail silently only for
    # whichever model type happens to win a given training run.
    is_tree_model = hasattr(model, "get_booster") or hasattr(model, "tree_") or hasattr(model, "estimators_")
    print(f"Building SHAP explainer ({'tree-based' if is_tree_model else 'linear/other'} model detected)...")
    if is_tree_model:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    else:
        explainer = shap.Explainer(model.predict_proba, background)
    print(f"Computing SHAP values for {len(test_sample)} test rows (this can take a bit)...")
    shap_values = explainer(test_sample)

    sv_matrix = normalize_shap_values(shap_values)
    base_value = get_base_value(shap_values)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Global summary plot (the headline visual) ---
    plt.figure()
    shap.summary_plot(sv_matrix, test_sample, show=False)
    plt.tight_layout()
    summary_path = OUTPUT_DIR / "global_summary.png"
    plt.savefig(summary_path, dpi=150)
    plt.close()
    print(f"Saved global summary plot -> {summary_path}")

    # --- 2. Global mean-|SHAP| bar chart + JSON table ---
    plt.figure()
    shap.summary_plot(sv_matrix, test_sample, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = OUTPUT_DIR / "global_importance_bar.png"
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"Saved global importance bar chart -> {bar_path}")

    mean_abs_shap = np.abs(sv_matrix).mean(axis=0)
    importance_table = sorted(
        zip(test_sample.columns, mean_abs_shap), key=lambda x: -x[1]
    )
    importance_json_path = OUTPUT_DIR / "global_importance.json"
    with open(importance_json_path, "w") as f:
        json.dump(
            [{"feature": f, "mean_abs_shap": float(v)} for f, v in importance_table],
            f, indent=2,
        )
    print(f"Saved global importance table -> {importance_json_path}")
    print("\nTop 10 features by mean |SHAP value|:")
    for name, val in importance_table[:10]:
        print(f"  {name:<40} {val:.4f}")

    # --- 3. Per-example waterfall plots ---
    preds = model.predict(test_sample)
    correct_phish_mask = (preds == 1) & (y_test_sample.values == 1)
    correct_legit_mask = (preds == 0) & (y_test_sample.values == 0)

    for label, mask, filename in [
        ("correctly-flagged phishing example", correct_phish_mask, "example_phishing.png"),
        ("correctly-cleared legitimate example", correct_legit_mask, "example_legitimate.png"),
    ]:
        idx_options = np.where(mask)[0]
        if len(idx_options) == 0:
            print(f"  No {label} found in this test sample - skipping its waterfall plot.")
            continue
        idx = idx_options[0]
        explanation = shap.Explanation(
            values=sv_matrix[idx],
            base_values=base_value,
            data=test_sample.iloc[idx].values,
            feature_names=list(test_sample.columns),
        )
        plt.figure()
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        out_path = OUTPUT_DIR / filename
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved waterfall plot for a {label} -> {out_path}")

    print(f"\nAll explainability artifacts saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
