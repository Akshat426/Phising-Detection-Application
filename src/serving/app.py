"""
FastAPI serving app: live phishing URL prediction with a per-prediction
explanation.

Scope, deliberately kept minimal for now: URL prediction + probability +
top contributing features. No authentication, no database, no prediction
history yet - those are real, planned future work (see PROJECT_STATUS.md),
deferred to keep the demoable core reliable ahead of a time-constrained
review rather than risking a bigger, more fragile build.

The model, transformer, and SHAP explainer are all loaded ONCE at startup,
not per-request - re-loading a model from disk or rebuilding a SHAP
explainer on every API call would make each prediction far slower than
necessary for something meant to feel responsive in a live demo.

Run with:
    uvicorn src.serving.app:app --reload --port 8000
Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.feature_extractor import URLFeatureExtractor, load_tranco_list  # noqa: E402
from features.transformation import DataTransformer  # noqa: E402
from features.shap_utils import build_explainer, normalize_shap_values  # noqa: E402
from serving.model_analysis import router as analysis_router  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[2] / "final_model"
TRANSFORM_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "transform_artifacts"

TOP_N_FEATURES = 5


class PredictRequest(BaseModel):
    url: str = Field(..., description="The URL to classify", examples=["http://example.com/login"])


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "pushes toward phishing" or "pushes toward legitimate"
    value: Optional[float] = None  # the raw feature value the model saw (None = unavailable)
    missing: bool = False  # True when the live extractor could not compute this feature


class PredictResponse(BaseModel):
    url: str
    prediction: str  # "phishing" or "legitimate"
    phishing_probability: float
    top_contributing_features: list[FeatureContribution]
    extraction_warnings: Optional[list[str]] = None
    all_contributions: list[FeatureContribution] = []  # every model input, sorted by |SHAP|
    checks: dict[str, Any] = {}  # which live lookups worked for this URL


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # silence pydantic's model_* naming warning - harmless collision

    status: str
    model_type: str
    tranco_loaded: bool = False
    model_metrics: Optional[dict] = None


app = FastAPI(title="Phishing URL Detector", version="1.0")

# CORS is left open for local development/demo purposes - no auth or cookies
# exist yet for a cross-origin request to abuse. Tighten this to a specific
# frontend origin once one exists and before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router)  # /analysis, /api/model-analysis, /analysis-assets/*

# --- Loaded once at startup, reused across every request ---
_model = None
_transformer = None
_explainer = None
_extractor = None
_tranco_loaded = False
_expected_columns: list[str] = []
_model_comparison: dict = {}


@app.on_event("startup")
def load_artifacts():
    global _model, _transformer, _explainer, _extractor, _expected_columns, _model_comparison, _tranco_loaded

    model_path = MODELS_DIR / "model.pkl"
    if not model_path.exists():
        raise RuntimeError(f"{model_path} not found - run model_trainer.py first.")
    _model = joblib.load(model_path)

    if not TRANSFORM_ARTIFACTS_DIR.exists():
        raise RuntimeError(f"{TRANSFORM_ARTIFACTS_DIR} not found - run transformation.py first.")
    _transformer = DataTransformer.load(TRANSFORM_ARTIFACTS_DIR)

    if hasattr(_model, "feature_names_in_"):
        _expected_columns = list(_model.feature_names_in_)
    elif hasattr(_model, "get_booster"):
        _expected_columns = _model.get_booster().feature_names
    else:
        _expected_columns = _transformer.feature_columns + [
            f"{c}_was_missing" for c in _transformer.indicator_columns
        ]

    _explainer = build_explainer(_model)

    try:
        tranco_list = load_tranco_list()
    except Exception as e:
        print(f"Warning: could not load Tranco list at startup ({e}). "
              f"tranco_rank feature will always be missing for live predictions.")
        tranco_list = None
    _tranco_loaded = tranco_list is not None
    _extractor = URLFeatureExtractor(tranco_list=tranco_list, fetch_content=True, fetch_whois=True)

    comparison_path = MODELS_DIR / "model_comparison.json"
    if comparison_path.exists():
        import json
        with open(comparison_path) as f:
            _model_comparison = json.load(f)

    print("Model, transformer, and SHAP explainer loaded successfully.")


@app.get("/", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if _model is not None else "model not loaded",
        model_type=type(_model).__name__ if _model is not None else "none",
        tranco_loaded=_tranco_loaded,
        model_metrics=_model_comparison.get("test_metrics") if _model_comparison else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _model is None or _transformer is None or _extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet - try again shortly.")

    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url must not be empty.")

    warnings = []
    try:
        features = _extractor.extract(url).to_dict()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not extract features from this URL: {e}")

    # Surface which network-dependent features actually failed for this
    # specific URL - useful transparency for a live demo, not just a
    # silent NaN a user would never see.
    if features.get("domain_age_days") is None:
        warnings.append("WHOIS lookup failed - domain age/registration are treated as unknown (missing-value indicator set).")
    if not _tranco_loaded:
        warnings.append("Tranco popularity list is not loaded - tranco_rank is missing for every URL, "
                        "which biases results toward 'phishing'. Restart the server when online.")
    if features.get("response_status_code") is None:
        warnings.append("Could not fetch the live page - content features are treated as unknown (missing-value indicator set).")

    row_df = pd.DataFrame([features])
    transformed = _transformer.transform(row_df)
    model_input = transformed[[c for c in transformed.columns if c != "label"]]
    model_input = model_input.reindex(columns=_expected_columns, fill_value=0.0)

    probability = float(_model.predict_proba(model_input)[0, 1])
    prediction = "phishing" if probability >= 0.5 else "legitimate"

    shap_values = _explainer(model_input)
    sv = normalize_shap_values(shap_values)[0]
    ranked_all = sorted(zip(_expected_columns, sv), key=lambda x: -abs(x[1]))

    def _raw_value(name: str):
        """The human-meaningful raw value behind a model input column."""
        if name.endswith("_was_missing"):
            return 1.0 if features.get(name[: -len("_was_missing")]) is None else 0.0, False
        v = features.get(name)
        return (None, True) if v is None else (float(v), False)

    contributions = []
    for name, value in ranked_all:
        raw_value, missing = _raw_value(name)
        contributions.append(
            FeatureContribution(
                feature=name,
                shap_value=float(value),
                direction="pushes toward phishing" if value > 0 else "pushes toward legitimate",
                value=raw_value,
                missing=missing,
            )
        )
    top_features = contributions[:TOP_N_FEATURES]

    checks = {
        "tranco_list_loaded": _tranco_loaded,
        "tranco_rank": features.get("tranco_rank"),
        "dns_resolves": bool(features.get("dns_resolves")) if features.get("dns_resolves") is not None else None,
        "whois_ok": features.get("domain_age_days") is not None,
        "page_fetch_ok": features.get("response_status_code") is not None,
        "http_status": int(features["response_status_code"]) if features.get("response_status_code") is not None else None,
    }

    return PredictResponse(
        url=url,
        prediction=prediction,
        phishing_probability=probability,
        top_contributing_features=top_features,
        extraction_warnings=warnings or None,
        all_contributions=contributions,
        checks=checks,
    )


STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    return (STATIC_DIR / "checker.html").read_text(encoding="utf-8")
