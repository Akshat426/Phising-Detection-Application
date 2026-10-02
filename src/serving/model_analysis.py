"""
Model-analysis routes for the web UI: shows how the 4 candidate models were
compared, which one was selected and why, plus robustness/explainability
results. Everything is read from the JSON/PNG files the training scripts
already write into final_model/ - nothing here is hardcoded except the
Brier score (calibration_check.py only prints it), so re-running
model_trainer.py automatically refreshes the page.

Endpoints (mounted by app.py via include_router):
    GET /analysis                       -> the HTML page
    GET /api/model-analysis             -> JSON consumed by the page
    GET /analysis-assets/{name}.png     -> whitelisted SHAP/calibration plots
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "final_model"
EXPLAIN_DIR = MODEL_DIR / "explainability"
PAGE_PATH = Path(__file__).resolve().parent / "static" / "analysis.html"

BRIER_SCORE = 0.0136  # from calibration_check.py (see PROJECT_STATUS / report)
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
ASSETS = {  # whitelist: the URL never maps to an arbitrary file path
    "calibration_curve": "calibration_curve.png",
    "global_summary": "global_summary.png",
    "example_phishing": "example_phishing.png",
    "example_legitimate": "example_legitimate.png",
}

router = APIRouter()


def _load(name: str):
    path = MODEL_DIR / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _summarise(report: dict | None) -> dict | None:
    if not report:
        return None
    return {
        "selected": report.get("selected_model"),
        "excluded_features": report.get("excluded_features"),
        "validation": report.get("validation_results", {}),
        "test": report.get("test_metrics"),
    }


@router.get("/api/model-analysis")
def model_analysis():
    main = _load("model_comparison.json")
    if main is None:
        raise HTTPException(404, "final_model/model_comparison.json not found - run model_trainer.py first.")

    importance = []
    imp_path = EXPLAIN_DIR / "global_importance.json"
    if imp_path.exists():
        with open(imp_path, encoding="utf-8") as f:
            importance = json.load(f)[:10]

    return {
        "labels": MODEL_LABELS,
        "final": _summarise(main),
        "no_tranco": _summarise(_load("model_comparison_no_tranco.json")),
        "pre_augmentation": _summarise(_load("model_comparison_pre_augmentation.json")),
        "importance": importance,
        "brier_score": BRIER_SCORE,
        "curves": _load("model_analysis_data.json"),  # from export_model_analysis.py (None if not generated yet)
        "available_assets": [k for k, v in ASSETS.items() if (EXPLAIN_DIR / v).exists()],
    }


@router.get("/analysis-assets/{name}.png")
def analysis_asset(name: str):
    filename = ASSETS.get(name)
    if filename is None or not (EXPLAIN_DIR / filename).exists():
        raise HTTPException(404, "Unknown asset.")
    return FileResponse(EXPLAIN_DIR / filename, media_type="image/png")


@router.get("/analysis", response_class=HTMLResponse)
def analysis_page():
    return PAGE_PATH.read_text(encoding="utf-8")
