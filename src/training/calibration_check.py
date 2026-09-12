"""
Calibration check: does the model's predicted probability actually mean what
it claims?

If the model says "92% confident this is phishing," a well-calibrated model
should be right about 92% of the time among all predictions it made at that
confidence level. This matters specifically for this project because the
live app (Phase 5) will display a confidence percentage to users via the
Gemini-generated explanation - an overconfident, poorly-calibrated security
tool is a real, reportable risk, not just an academic nicety.

Produces:
  - A reliability diagram (predicted probability vs. observed frequency)
  - Brier score (lower is better; 0 = perfect, 0.25 = the "always guess 50%" baseline)
  - A plain-language verdict: is the model over- or under-confident, and where

Usage:
    python calibration_check.py
    python calibration_check.py --model-path final_model/model_no_tranco.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

MODELS_DIR = Path(__file__).resolve().parents[2] / "final_model"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_DIR = MODELS_DIR / "explainability"

N_BINS = 10  # standard choice for reliability diagrams; fewer bins if the test set is small


def main():
    parser = argparse.ArgumentParser(description="Check probability calibration of the trained model")
    parser.add_argument("--model-path", type=str, default=None,
                         help="Path to a specific model .pkl (default: final_model/model.pkl)")
    args = parser.parse_args()

    model_path = Path(args.model_path) if args.model_path else MODELS_DIR / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found - run model_trainer.py first.")

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    test_df = pd.read_csv(PROCESSED_DIR / "test_transformed.csv")
    y_test = test_df["label"]
    X_test = test_df.drop(columns=["label"])

    # Align columns for models trained on a feature subset (ablation runs)
    if hasattr(model, "feature_names_in_"):
        X_test = X_test[list(model.feature_names_in_)]
    elif hasattr(model, "get_booster"):
        X_test = X_test[model.get_booster().feature_names]

    probs = model.predict_proba(X_test)[:, 1]

    n_bins = min(N_BINS, len(np.unique(probs)))
    if n_bins < N_BINS:
        print(f"Note: reducing to {n_bins} bins - test set/model doesn't have enough distinct "
              f"probability values for the default {N_BINS}.")

    observed_freq, predicted_prob = calibration_curve(y_test, probs, n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y_test, probs)

    print(f"\nBrier score: {brier:.4f}  (0 = perfect, 0.25 = 'always guess 50%' baseline, "
          f"lower is better)")

    print("\nReliability table (predicted confidence vs. actual observed rate):")
    print(f"  {'Predicted':>12} {'Observed':>12} {'Gap':>8}")
    max_gap = 0.0
    max_gap_bin = None
    for pred, obs in zip(predicted_prob, observed_freq):
        gap = pred - obs
        print(f"  {pred:>11.1%} {obs:>11.1%} {gap:>+7.1%}")
        if abs(gap) > abs(max_gap):
            max_gap = gap
            max_gap_bin = pred

    # Plain-language verdict
    print("\nVerdict:", end=" ")
    if abs(max_gap) < 0.05:
        print("The model is reasonably well-calibrated - predicted confidence closely "
              "tracks actual accuracy across the probability range.")
    elif max_gap > 0:
        print(f"The model is somewhat OVERCONFIDENT around the {max_gap_bin:.0%} confidence "
              f"level - it claims higher certainty than it actually achieves there. Worth "
              f"noting as a limitation if the app surfaces raw confidence scores to users.")
    else:
        print(f"The model is somewhat UNDERCONFIDENT around the {max_gap_bin:.0%} confidence "
              f"level - it's actually more often correct than its stated confidence there.")

    # Reliability diagram
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(predicted_prob, observed_freq, marker="o", label="Model")
    plt.xlabel("Predicted probability (confidence)")
    plt.ylabel("Observed frequency (actual accuracy)")
    plt.title(f"Reliability diagram (Brier score: {brier:.4f})")
    plt.legend()
    plt.tight_layout()
    out_path = OUTPUT_DIR / "calibration_curve.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved reliability diagram -> {out_path}")


if __name__ == "__main__":
    main()
