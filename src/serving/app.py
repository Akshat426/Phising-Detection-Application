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
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.feature_extractor import URLFeatureExtractor, load_tranco_list  # noqa: E402
from features.transformation import DataTransformer  # noqa: E402
from features.shap_utils import build_explainer, normalize_shap_values  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[2] / "final_model"
TRANSFORM_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "transform_artifacts"

TOP_N_FEATURES = 5


class PredictRequest(BaseModel):
    url: str = Field(..., description="The URL to classify", examples=["http://example.com/login"])


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "pushes toward phishing" or "pushes toward legitimate"


class PredictResponse(BaseModel):
    url: str
    prediction: str  # "phishing" or "legitimate"
    phishing_probability: float
    top_contributing_features: list[FeatureContribution]
    extraction_warnings: Optional[list[str]] = None


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # silence pydantic's model_* naming warning - harmless collision

    status: str
    model_type: str
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

# --- Loaded once at startup, reused across every request ---
_model = None
_transformer = None
_explainer = None
_extractor = None
_expected_columns: list[str] = []
_model_comparison: dict = {}


@app.on_event("startup")
def load_artifacts():
    global _model, _transformer, _explainer, _extractor, _expected_columns, _model_comparison

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
        warnings.append("WHOIS lookup failed - domain age/registration features are estimated from other data.")
    if features.get("response_status_code") is None:
        warnings.append("Could not fetch the live page - content-based features are estimated from other data.")

    row_df = pd.DataFrame([features])
    transformed = _transformer.transform(row_df)
    model_input = transformed[[c for c in transformed.columns if c != "label"]]
    model_input = model_input.reindex(columns=_expected_columns, fill_value=0.0)

    probability = float(_model.predict_proba(model_input)[0, 1])
    prediction = "phishing" if probability >= 0.5 else "legitimate"

    shap_values = _explainer(model_input)
    sv = normalize_shap_values(shap_values)[0]
    ranked = sorted(zip(_expected_columns, sv), key=lambda x: -abs(x[1]))[:TOP_N_FEATURES]
    top_features = [
        FeatureContribution(
            feature=name,
            shap_value=float(value),
            direction="pushes toward phishing" if value > 0 else "pushes toward legitimate",
        )
        for name, value in ranked
    ]

    return PredictResponse(
        url=url,
        prediction=prediction,
        phishing_probability=probability,
        top_contributing_features=top_features,
        extraction_warnings=warnings or None,
    )


DEMO_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Phishing URL Detector - Demo</title>
<style>
  :root {
    --bg: #0f172a; --card: #ffffff; --text: #1e293b; --muted: #64748b;
    --safe: #16a34a; --danger: #dc2626; --accent: #4f46e5;
  }
  body {
    font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    background: linear-gradient(135deg, #4f46e5 0%, #0f172a 100%);
    min-height: 100vh; margin: 0; display: flex; align-items: center;
    justify-content: center; padding: 24px; box-sizing: border-box;
  }
  .card {
    background: var(--card); border-radius: 16px; padding: 40px;
    max-width: 640px; width: 100%; box-shadow: 0 20px 60px rgba(0,0,0,0.3);
  }
  h1 { color: var(--text); font-size: 26px; margin: 0 0 6px 0; }
  .subtitle { color: var(--muted); font-size: 15px; margin-bottom: 28px; }
  .input-row { display: flex; gap: 10px; }
  input[type=text] {
    flex: 1; padding: 14px 16px; border-radius: 10px; border: 2px solid #e2e8f0;
    font-size: 15px; outline: none; box-sizing: border-box;
  }
  input[type=text]:focus { border-color: var(--accent); }
  button {
    padding: 14px 24px; border-radius: 10px; border: none; background: var(--accent);
    color: white; font-size: 15px; font-weight: 600; cursor: pointer;
  }
  button:hover { opacity: 0.9; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  #result { margin-top: 28px; display: none; }
  .verdict-banner {
    padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;
  }
  .verdict-banner.phishing { background: #fef2f2; border: 2px solid var(--danger); }
  .verdict-banner.legitimate { background: #f0fdf4; border: 2px solid var(--safe); }
  .verdict-text { font-size: 22px; font-weight: 700; margin: 0; }
  .verdict-banner.phishing .verdict-text { color: var(--danger); }
  .verdict-banner.legitimate .verdict-text { color: var(--safe); }
  .prob-text { color: var(--muted); font-size: 14px; margin-top: 4px; }
  .features-title { font-size: 14px; font-weight: 600; color: var(--text); margin-bottom: 10px; }
  .feature-row {
    display: flex; justify-content: space-between; padding: 8px 0;
    border-bottom: 1px solid #f1f5f9; font-size: 14px;
  }
  .feature-name { color: var(--text); }
  .feature-dir { font-size: 12px; padding: 2px 8px; border-radius: 6px; }
  .feature-dir.phishing { background: #fef2f2; color: var(--danger); }
  .feature-dir.legitimate { background: #f0fdf4; color: var(--safe); }
  .loading { text-align: center; color: var(--muted); padding: 20px; display: none; }
  .warning { background: #fffbeb; border: 1px solid #fbbf24; border-radius: 8px;
             padding: 10px 14px; font-size: 13px; color: #92400e; margin-top: 14px; }
  .error { background: #fef2f2; border: 1px solid var(--danger); border-radius: 8px;
           padding: 10px 14px; font-size: 13px; color: var(--danger); margin-top: 14px; }
</style>
</head>
<body>
  <div class="card">
    <h1>&#128737; Phishing URL Detector</h1>
    <p class="subtitle">Paste any URL below to check whether it looks like phishing.</p>
    <div class="input-row">
      <input type="text" id="urlInput" placeholder="https://example.com" />
      <button id="checkBtn" onclick="checkUrl()">Check</button>
    </div>
    <div class="loading" id="loading">Analyzing URL&hellip;</div>
    <div id="result"></div>
  </div>

<script>
async function checkUrl() {
  const url = document.getElementById('urlInput').value.trim();
  if (!url) return;
  const btn = document.getElementById('checkBtn');
  const loading = document.getElementById('loading');
  const result = document.getElementById('result');
  btn.disabled = true;
  loading.style.display = 'block';
  result.style.display = 'none';

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    const data = await resp.json();
    if (!resp.ok) {
      result.innerHTML = `<div class="error">${data.detail || 'Something went wrong.'}</div>`;
    } else {
      const isPhishing = data.prediction === 'phishing';
      const pct = (data.phishing_probability * 100).toFixed(1);
      let featuresHtml = data.top_contributing_features.map(f => `
        <div class="feature-row">
          <span class="feature-name">${f.feature}</span>
          <span class="feature-dir ${f.direction.includes('phishing') ? 'phishing' : 'legitimate'}">
            ${f.direction.includes('phishing') ? 'toward phishing' : 'toward legitimate'}
          </span>
        </div>`).join('');
      let warningsHtml = data.extraction_warnings
        ? `<div class="warning">&#9888;&#65039; ${data.extraction_warnings.join(' ')}</div>` : '';
      result.innerHTML = `
        <div class="verdict-banner ${isPhishing ? 'phishing' : 'legitimate'}">
          <p class="verdict-text">${isPhishing ? '&#9888;&#65039; Likely Phishing' : '&#9989; Looks Legitimate'}</p>
          <p class="prob-text">Phishing probability: ${pct}%</p>
        </div>
        <div class="features-title">Top factors in this decision:</div>
        ${featuresHtml}
        ${warningsHtml}
      `;
    }
    result.style.display = 'block';
  } catch (e) {
    result.innerHTML = `<div class="error">Could not reach the server. Is it still running?</div>`;
    result.style.display = 'block';
  } finally {
    btn.disabled = false;
    loading.style.display = 'none';
  }
}
document.getElementById('urlInput').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') checkUrl();
});
</script>
</body>
</html>
"""


@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    return DEMO_PAGE_HTML
