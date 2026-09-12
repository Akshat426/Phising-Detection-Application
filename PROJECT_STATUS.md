# Project Status — Phishing URL Detection CP

## What we're building

A phishing-URL detection system, rebuilt from scratch as an ML capstone project.
A friend's existing project (FastAPI + React + a phishing classifier) was used
purely as an **architectural reference** — the overall shape (data pipeline →
training → serving) was kept, but every dataset, feature, and line of model
code here is original and independently understood, developed, and defended.

**Why rebuild rather than extend the reference project:** the reference
project used the classic ~2015 UCI "Phishing Websites" dataset, which came
pre-extracted with 30 static features — several of which (`web_traffic`,
`Page_Rank`, `Google_Index`) relied on services (Alexa Rank, the old Google
Index API) that no longer exist. Its live prediction code silently hardcoded
those 5 features as constants for every URL, regardless of input. This
project's entire premise is fixing that: real, current data, and every
feature genuinely computed.

## Current status: Phase 1–2 complete, Phase 3 (explainability) starting next

---

## What's been built so far

### 1. Feature extraction (`src/features/feature_extractor.py`)
35 features across three groups, each with an honest degradation story:
- **Lexical** (20 features, instant, no network): URL length, dot/hyphen/slash
  counts, IP-address detection, known-shortener detection, brand-keyword-in-
  subdomain detection, etc.
- **Domain** (4 features, requires WHOIS/DNS/Tranco): domain age, registration
  length, DNS resolution, Tranco popularity rank — with retry logic (2 retries,
  exponential backoff) for the notoriously flaky WHOIS protocol.
- **Content** (11 features, requires a live page fetch): form count, external
  link ratio, iframe count, password-field detection, favicon domain match, etc.

Failed network calls produce real `NaN` values, never a fake default — this
is deliberate and is itself a data-quality finding worth reporting.

### 2. Data collection
- **Phishing URLs**: OpenPhish's free feed (PhishTank has had new-registration
  disabled since ~2022, confirmed via research — a legitimate, citable
  reason for choosing OpenPhish instead). `fetch_openphish.py` merges new
  URLs into the dataset on every run without duplicating.
- **Legitimate URLs**: sampled from Tranco's popularity ranking across
  multiple rank bands (not just top-100 mega-sites). Critically,
  `fetch_tranco.py` fetches each domain's **actual homepage and extracts a
  real internal page URL** — not just the bare domain root (see the leakage
  story below for why this matters).
- Current dataset size: ~3,900 URLs (2,770 legitimate / 1,144 phishing).

### 3. Data pipeline (`src/data_pipeline/`)
- `build_dataset.py`: runs feature extraction over every URL concurrently,
  with checkpointing every 200 rows (safe to interrupt/resume).
- `split_dataset.py`: stratified 70/15/15 train/validation/test split.
- `data_validation.py`: schema checks + KS-test drift detection between
  train and test (has already caught one real, reportable drift instance).

### 4. Transformation (`src/features/transformation.py`)
- Adds a missing-value **indicator column** for every feature with
  non-trivial missingness *before* imputing — so "this WHOIS lookup failed"
  is preserved as a signal rather than silently smoothed over.
- Median imputation + standardization, fit strictly on train, applied
  unchanged to validation/test/live data (no leakage).

### 5. Model training (`src/training/model_trainer.py`)
- Compares 4 models (Logistic Regression, Decision Tree, Random Forest,
  XGBoost) via `RandomizedSearchCV` (25 iterations, 5-fold CV) — meaningfully
  wider search than the reference project's 8-combination grids.
- Strict train → validate → test discipline: cross-validation happens only
  within train; validation is touched exactly once to pick the best model;
  test is touched exactly once, at the end, purely for reporting.
- Full MLflow experiment tracking (SQLite-backed).
- Built-in feature-importance printout as a **leakage sanity check** — this
  is what caught the bug described below.
- `--exclude-features` flag for running ablation experiments on demand.

---

## The data leakage story (worth its own section in your report)

The first full training run produced **perfect 1.0000 accuracy/F1/AUC across
all 4 models** — an immediate red flag. Diagnosis via feature importance
found two real bugs, in order:

1. **`tranco_rank` sentinel bug**: out-of-list domains were getting a
   numeric "not found" value below the code's detection threshold instead
   of `None`, meaning *every* legitimate URL (sourced from Tranco) had a
   real rank, and *every* phishing URL didn't — trivially leaking the data
   source. Fixed with an explicit `{domain: rank}` dict lookup.
2. **URL-shape leakage (the bigger one)**: legitimate URLs were being built
   as bare domain roots (`https://google.com`), while phishing URLs are
   real-world URLs with paths and query strings. The model was trivially
   learning "does this URL have a path" rather than anything about phishing.
   Fixed by rewriting `fetch_tranco.py` to fetch real internal page URLs
   from each domain's homepage.

After both fixes: **credible, non-perfect results** (F1 in the 0.94–0.99
range across runs, varying with the dataset), with feature importance now
dominated by genuine signals (internal link count, hostname length, brand
keywords in subdomains, etc.) rather than a single dominant artifact.

An ablation (`--exclude-features tranco_rank,tranco_rank_was_missing`)
quantified the remaining, structurally-unavoidable residual signal: F1 drops
from 0.968 to 0.944 without it — a real but moderate contribution, honestly
reported rather than hidden.

**This whole debugging arc is genuinely strong material for your CP report's
methodology section** — it demonstrates real ML engineering judgment, not
just a model that happened to work.

---

## Current best result

- **Model**: XGBoost (winner across most runs)
- **Test F1**: ~0.94–0.97 (varies by dataset draw; ablated/conservative
  estimate ~0.944)
- **Test ROC-AUC**: ~0.996–0.999

## What's next: Phase 3 — Explainability

- Integrate SHAP for per-prediction feature attribution
- Global feature importance analysis vs. domain-expert intuition
- (Later) calibration checking, deep learning baseline comparison,
  stacking ensemble, MLOps automation, frontend polish

## Repository

Private GitHub repo, actively committed with descriptive messages at each
meaningful milestone (environment setup, each bug fix, each ablation).
`.env`, raw/processed data, and model artifacts (`.pkl`) are git-ignored;
code and experiment-report JSON files are tracked.
