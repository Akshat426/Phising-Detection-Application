"""
Train and compare 4 candidate models: Logistic Regression, Decision Tree,
Random Forest, and XGBoost.

Workflow (this is the part that's meaningfully more rigorous than a naive
train/test-only setup, and worth calling out explicitly in the CP report):

  1. For each model type, run RandomizedSearchCV with 5-fold cross-validation
     ON THE TRAINING SET ONLY to find good hyperparameters. Cross-validation
     within train (not a single validation-set check) makes the hyperparameter
     choice itself more robust to which particular rows happened to land in
     any one split.
  2. Refit each model type's best hyperparameters on the full training set,
     then evaluate all 4 refit models on the VALIDATION set to pick the
     single best model type. The validation set is used exactly once here,
     for model selection - never touched during the hyperparameter search
     itself.
  3. The TEST set is touched exactly once, at the very end, purely to report
     the final, honest performance of the one chosen model. It is never used
     for any decision-making (not hyperparameters, not model choice) - that
     is what keeps the final test metric an unbiased estimate of real-world
     performance rather than a number that's been implicitly tuned to.

Everything is logged to MLflow (local file store by default) so every run's
hyperparameters, metrics, and artifacts are tracked and comparable over time.

Usage:
    python model_trainer.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "final_model"
MLFLOW_DB_PATH = Path(__file__).resolve().parents[2] / "mlflow.db"

RANDOM_STATE = 42
CV_FOLDS = 5
SEARCH_ITER = 25  # RandomizedSearchCV draws this many hyperparameter combinations


def load_split(name: str, exclude_features: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    path = PROCESSED_DIR / f"{name}_transformed.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run transformation.py first.")
    df = pd.read_csv(path)
    y = df["label"]
    X = df.drop(columns=["label"])
    if exclude_features:
        present = [c for c in exclude_features if c in X.columns]
        missing = [c for c in exclude_features if c not in X.columns]
        if missing:
            print(f"  Note: --exclude-features listed columns not found (already absent): {missing}")
        X = X.drop(columns=present)
    return X, y


def get_model_search_specs(class_weight_dict) -> dict:
    """Returns {model_name: (estimator, param_distributions)} for RandomizedSearchCV.

    Hyperparameter ranges here are meaningfully wider than the reference
    project's grids (e.g. its XGBoost grid only tried 8 total combinations).
    class_weight='balanced' is used where supported, since the dataset has
    real (if moderate) class imbalance from how the data was collected.
    """
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            {
                "C": uniform(0.001, 10),
                "penalty": ["l2"],  # l1 needs a different solver; keep the search space clean
                "solver": ["lbfgs", "liblinear"],
            },
        ),
        "decision_tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {
                "max_depth": randint(3, 30),
                "min_samples_split": randint(2, 40),
                "min_samples_leaf": randint(1, 20),
                "criterion": ["gini", "entropy"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": randint(100, 600),
                "max_depth": randint(3, 30),
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 10),
                "max_features": ["sqrt", "log2", None],
            },
        ),
        "xgboost": (
            XGBClassifier(
                eval_metric="logloss", random_state=RANDOM_STATE,
                scale_pos_weight=class_weight_dict["xgb_scale_pos_weight"],
            ),
            {
                "n_estimators": randint(100, 600),
                "max_depth": randint(2, 12),
                "learning_rate": uniform(0.01, 0.3),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.6, 0.4),
            },
        ),
    }


def evaluate(model, X, y) -> dict:
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else preds
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, probs),
    }


def main():
    parser = argparse.ArgumentParser(description="Train and compare phishing detection models")
    parser.add_argument(
        "--exclude-features", type=str, default=None,
        help="Comma-separated feature names to drop before training (for ablation experiments, "
             "e.g. --exclude-features tranco_rank,tranco_rank_was_missing)",
    )
    parser.add_argument(
        "--output-suffix", type=str, default=None,
        help="Suffix for output files, e.g. --output-suffix ablation_no_tranco keeps this run's "
             "model.pkl and comparison report separate from your main final_model/model.pkl, "
             "so an experiment never silently overwrites your primary submitted model.",
    )
    args = parser.parse_args()

    exclude_features = [f.strip() for f in args.exclude_features.split(",")] if args.exclude_features else None
    if exclude_features:
        print(f"ABLATION MODE: excluding features {exclude_features}\n")

    X_train, y_train = load_split("train", exclude_features)
    X_val, y_val = load_split("validation", exclude_features)
    X_test, y_test = load_split("test", exclude_features)

    print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    print(f"Train class balance:\n{y_train.value_counts(normalize=True)}")

    # scale_pos_weight for XGBoost = (# negative / # positive) in training data
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    class_weight_dict = {"xgb_scale_pos_weight": n_neg / n_pos if n_pos > 0 else 1.0}

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment("phishing-detection-model-comparison")

    model_specs = get_model_search_specs(class_weight_dict)
    val_results = {}
    fitted_models = {}

    for name, (estimator, param_dist) in model_specs.items():
        print(f"\n=== {name}: hyperparameter search ({SEARCH_ITER} iterations, {CV_FOLDS}-fold CV on train) ===")
        with mlflow.start_run(run_name=name):
            search = RandomizedSearchCV(
                estimator, param_dist, n_iter=SEARCH_ITER, cv=CV_FOLDS,
                scoring="f1", random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            print(f"  Best CV F1 (train): {search.best_score_:.4f}")
            print(f"  Best params: {search.best_params_}")

            val_metrics = evaluate(best_model, X_val, y_val)
            print(f"  Validation metrics: {val_metrics}")

            mlflow.log_params(search.best_params_)
            mlflow.log_metric("cv_f1_train", search.best_score_)
            for metric_name, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", value)

            # XGBoost's internal Booster type isn't automatically trusted by
            # MLflow's generic sklearn-flavor security check (skops), so it
            # needs its own dedicated logging function rather than
            # mlflow.sklearn.log_model, which every other model here uses.
            if name == "xgboost":
                mlflow.xgboost.log_model(best_model, name)
            else:
                mlflow.sklearn.log_model(best_model, name)

            val_results[name] = val_metrics
            fitted_models[name] = best_model

    # Select the best model type by validation F1 - the ONE decision that
    # uses the validation set, made exactly once.
    best_name = max(val_results, key=lambda n: val_results[n]["f1"])
    best_model = fitted_models[best_name]
    print(f"\n=== Best model on validation: {best_name} (F1={val_results[best_name]['f1']:.4f}) ===")

    # Test set touched exactly once, here, for final reporting only.
    test_metrics = evaluate(best_model, X_test, y_test)
    print(f"Final TEST metrics for {best_name} (reported once, not used for any decision): {test_metrics}")

    with mlflow.start_run(run_name=f"FINAL_{best_name}"):
        mlflow.log_param("selected_model", best_name)
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)
        if best_name == "xgboost":
            mlflow.xgboost.log_model(best_model, "final_model")
        else:
            mlflow.sklearn.log_model(best_model, "final_model")

    # Diagnostic: print the top features driving the selected model's
    # predictions. This is a leakage sanity-check as much as an
    # explainability step - if a feature that shouldn't plausibly matter
    # this much (e.g. a missing-value indicator, or a rank field that
    # perfectly tracks which raw data source a row came from) dominates
    # the list, that's a strong signal something is leaking the label
    # through a shortcut rather than genuine phishing characteristics,
    # not a sign of a great model.
    print("\n=== Feature importance / coefficients for the selected model (leakage sanity check) ===")
    feature_names = X_train.columns.tolist()
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        ranked = sorted(zip(feature_names, importances), key=lambda x: -abs(x[1]))
        for name, val in ranked[:15]:
            print(f"  {name:<40} {val:.4f}")
    elif hasattr(best_model, "coef_"):
        coefs = best_model.coef_[0]
        ranked = sorted(zip(feature_names, coefs), key=lambda x: -abs(x[1]))
        for name, val in ranked[:15]:
            print(f"  {name:<40} {val:+.4f}")
    else:
        print("  (Selected model type doesn't expose feature importances/coefficients directly.)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    model_path = MODELS_DIR / f"model{suffix}.pkl"
    report_path = MODELS_DIR / f"model_comparison{suffix}.json"
    joblib.dump(best_model, model_path)

    # Save ALL four fitted models (not just the winner) so downstream scripts -
    # compare_models.py, significance tests - evaluate the exact models that were
    # selected on validation instead of refitting them from printed parameters.
    all_models_dir = MODELS_DIR / f"all_models{suffix}"
    all_models_dir.mkdir(parents=True, exist_ok=True)
    for model_name, fitted in fitted_models.items():
        joblib.dump(fitted, all_models_dir / f"{model_name}.pkl")
    print(f"Saved all {len(fitted_models)} fitted models -> {all_models_dir}")

    comparison_report = {
        "excluded_features": exclude_features,
        "validation_results": val_results,
        "selected_model": best_name,
        "test_metrics": test_metrics,
    }
    with open(report_path, "w") as f:
        json.dump(comparison_report, f, indent=2)

    print(f"\nSaved best model -> {model_path}")
    print(f"Saved comparison report -> {report_path}")


if __name__ == "__main__":
    main()
