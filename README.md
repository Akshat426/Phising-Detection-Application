# Phishing URL Detection — ML Capstone

Rebuilt from scratch, using a reference project as an architectural guide only.
Every dataset, feature, and model here is built fresh for this project.

## Why rebuild?

The reference project this is based on had a real, fixable weakness: 5 of its
30 features were hardcoded constants rather than actually computed per URL,
and its dataset was the ~2015 UCI phishing dataset with several features
(Alexa Rank, Google Index) that rely on services that no longer exist or are
no longer freely queryable. This rebuild:
- Computes every feature genuinely (see `src/features/feature_extractor.py`)
- Uses current, real data: OpenPhish's live feed (PhishTank has had
  new-registration disabled since ~2022) + Tranco (a maintained,
  research-grade domain popularity ranking) instead of a decade-old static
  dataset

## Status: mid-semester review complete, ~55% of the full plan

- **Phase 1 (data + features): done.** 35 live-computed features, 3,914
  processed URLs (2,770 legitimate / 1,144 phishing), ~600 more phishing
  URLs already collected and awaiting processing.
- **Phase 2 (model training): ~60% done.** Four models compared
  (Logistic Regression, Decision Tree, Random Forest, XGBoost); XGBoost
  selected (test F1 0.964). Stacking ensemble and a character-level deep
  model are still pending.
- **Phase 3 (explainability + robustness): done.** SHAP explanations,
  calibration check (Brier 0.0136), adversarial evasion testing and fix.
- **Phase 4 (MLOps): just started.** MLflow experiment tracking is in
  place; scheduled retraining and drift alerts are not yet built.
- **Phase 5 (serving): ~40% done.** A FastAPI backend with two web pages —
  a URL checker with SHAP-based explanations, and a Model Analysis page
  comparing all four models. No accounts or prediction history yet.

**A known, diagnosed limitation:** live testing surfaced false positives on
several well-known sites (e.g. `www.youtube.com`). Traced to two data
biases — `www.` being counted as a subdomain, and popularity-rank
inflation from phishing hosted on free platforms (vercel.app, pages.dev,
etc.). Confirmed by intervention: removing only `www.` drops the YouTube
false-positive from 95.1% to 1.6%. Full diagnosis and reproducible numbers
in `src/training/model_diagnostics.py` and the project report. The fix
(ignore `www.` in subdomain/hostname features; treat hosting platforms as
their own registered domains) is planned but not yet applied — retraining
now would change every reported number, so it's deferred to after
reporting milestones.

## Setup

This project needs full internet access (OpenPhish, Tranco, live site
fetches, WHOIS) — it will NOT work in a network-sandboxed environment.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env if you need to override any defaults (see .env.example for keys)
```

## Running the data pipeline (in order)

```bash
# 1. Fetch phishing URLs from the OpenPhish live feed (merges new URLs
#    into data/raw/ without duplicating on repeat runs)
python src/data_pipeline/fetch_openphish.py
# or on Windows: run_fetch_openphish.bat

# 2. Fetch legitimate URLs sampled across Tranco rank bands, crawling each
#    domain's homepage for a real internal link (not just the bare root —
#    see "the leakage story" in PROJECT_STATUS.md for why that matters)
python src/data_pipeline/fetch_tranco.py --total 5000

# 3. Extract features for every URL and build the labeled dataset
#    Start small to make sure everything works before committing hours to it:
python src/data_pipeline/build_dataset.py --sample 200

# Once that looks good, run the full build (this takes a while — it's
# making thousands of real network requests). Safe to Ctrl+C and re-run;
# it resumes from a checkpoint automatically.
python src/data_pipeline/build_dataset.py --workers 16

# 4. Validate, split (70/15/15) and transform
python src/data_pipeline/data_validation.py
python src/data_pipeline/split_dataset.py
```

## Training and evaluation

```bash
# Compare all 4 models, select the best by validation F1, score test once
python src/training/model_trainer.py

# Explainability, calibration, and robustness
python src/training/explain_model.py
python src/training/calibration_check.py
python src/training/adversarial_check.py

# Curves and confusion matrices for the Model Analysis web page
# (re-scores the saved models and verifies they match model_comparison.json
# before writing anything)
python src/training/export_model_analysis.py

# Reproduces every diagnostic figure behind the known-limitation writeup
# (platform-hosting bias, rank bias, the "www." finding, honest-eval check)
python src/training/model_diagnostics.py
```

## Running the web app

```bash
uvicorn src.serving.app:app --port 8000
```

- `http://127.0.0.1:8000/demo` — the checker: paste a URL, get a verdict,
  a probability gauge, a plain-English summary, and per-feature SHAP bars
  with the actual value the model saw for each one.
- `http://127.0.0.1:8000/analysis` — Model Analysis: the four-model
  comparison, F1 ranking, ROC/PR curves, confusion matrices, calibration,
  SHAP importance, and the honest-limitations writeup. Reads the JSON
  files written by training, so it refreshes automatically on retrain
  (curves/confusion matrices need `export_model_analysis.py` re-run by hand).
- `http://127.0.0.1:8000/` — health check; also reports whether the
  Tranco popularity list loaded (the model depends heavily on it — see
  the known-limitation note above).

## Project structure

```
data/
  raw/            OpenPhish + Tranco URL lists (not committed — regenerate via scripts)
  processed/      Final extracted-feature dataset, splits (not committed)
final_model/      Saved model comparison reports, SHAP/calibration plots,
                  diagnostics (model.pkl itself is gitignored — regenerate
                  by running the training pipeline above)
src/
  features/       feature_extractor.py — the core, reusable feature computation logic
  data_pipeline/  scripts that fetch URLs, validate, split and build the dataset
  training/       model comparison, explainability, calibration, robustness,
                  and the diagnostics behind the known-limitation writeup
  serving/        FastAPI app, the Model Analysis router, and the two web pages
tests/            unit tests for the feature extractor (still to add)
```

## A note on feature extraction reliability

Every feature that requires a network call (WHOIS, live page fetch) is
allowed to fail and produce a real missing value (`None`/`NaN`) rather than
a fake default — with a matching `_was_missing` indicator column added
before imputation, so "this lookup failed" is preserved as a signal rather
than silently smoothed over. This is a deliberate design choice: some
sites will be down, some WHOIS servers will rate-limit, and pretending
that didn't happen would quietly corrupt the dataset.
