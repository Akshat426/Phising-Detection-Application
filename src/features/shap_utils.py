"""
Shared SHAP utilities, used by both explain_model.py (offline analysis) and
the live serving app (per-prediction explanations). Kept in one place so a
live prediction's explanation is always computed the exact same way as the
offline analysis - if these drifted apart, the app could show an explanation
that doesn't actually match the methodology documented in the report.
"""

from __future__ import annotations

import numpy as np
import shap


def build_explainer(model, background=None):
    """Returns the correctly configured SHAP explainer for this model type.

    Tree-based models (XGBoost, Decision Tree, Random Forest) use
    TreeExplainer with tree_path_dependent perturbation - this reads
    feature-split statistics directly from the trained trees and, critically,
    sidesteps a real SHAP/XGBoost incompatibility where the default
    "interventional" mode raises NotImplementedError on models with
    categorical splits (found via testing against all 4 model types this
    project can produce, before ever running this against real data).
    """
    is_tree_model = hasattr(model, "get_booster") or hasattr(model, "tree_") or hasattr(model, "estimators_")
    if is_tree_model:
        return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    return shap.Explainer(model.predict_proba, background)


def normalize_shap_values(shap_values) -> np.ndarray:
    """Return a plain (n_samples, n_features) array of SHAP values for the
    POSITIVE (phishing) class, regardless of which model type produced them.
    Different model/explainer combinations return different shapes - see
    the longer explanation in explain_model.py's original implementation.
    """
    values = shap_values.values
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def get_base_value(shap_values) -> float:
    base = shap_values.base_values
    base = np.asarray(base)
    if base.ndim == 2:
        return float(base[0, 1])
    return float(np.atleast_1d(base)[0])
