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
quantified the residual signal under two setups: F1 drops from 0.968 to
0.944 before adversarial augmentation (Section below), and from 0.964 to
0.953 in the final pipeline — a real but moderate contribution, honestly
reported rather than hidden.

**Correction to an earlier assumption in this document:** we originally
described the residual Tranco signal as "structurally unavoidable, since
only legitimate URLs can ever have a rank." That's wrong, and the mistake
is itself a finding. Phishing hosted on free platforms (vercel.app,
pages.dev, blogspot.com, weebly.com, github.io) inherits the *platform's*
rank: 765 of 1,144 phishing URLs (67%) carry a Tranco rank, and among URLs
ranked in the top 1,000, 66% are phishing (vs. 29% overall). See
`src/training/model_diagnostics.py` for the reproducible numbers.

**This whole debugging arc is genuinely strong material for your CP report's
methodology section** — it demonstrates real ML engineering judgment, not
just a model that happened to work.

---

## Current best result (final pipeline, with adversarial augmentation)

- **Model**: XGBoost
- **Validation F1**: 0.988 (Random Forest 0.982, Decision Tree 0.956,
  Logistic Regression 0.942) — margin over Random Forest is thin: 3 missed
  phishing URLs vs. 5, out of 172
- **Test F1**: 0.964 (precision 98.8%, recall 94.2%, accuracy 98.0%,
  ROC-AUC 0.997) — scored once, after model selection
- **Honest-evaluation check**: domain-grouped 5-fold CV gives F1 0.966–0.967
  (vs. 0.987–0.988 for a random split), confirming the test score isn't
  inflated by same-domain URLs leaking across the train/test boundary

## Phase 3 — Explainability and robustness: done

- **SHAP** (`src/training/explain_model.py`): per-prediction feature
  attribution via `TreeExplainer`. Top global features: `path_length`
  (1.83), `hostname_length` (1.37), `num_internal_links` (1.20).
- **Calibration** (`src/training/calibration_check.py`): Brier score
  0.0136 — the model's stated confidence is trustworthy on data like the
  test set.
- **Adversarial robustness** (`src/training/adversarial_check.py` +
  `augment_adversarial_examples.py`): path-stripping evasion found at
  100% on correctly-caught phishing URLs, fixed via adversarial training
  (root-stripped copies added to the training set only) — evasion rate
  dropped to 0% with test F1 essentially unchanged (0.968 → 0.964).
  Caveat: this check used ~10 URLs, so the percentage is coarse; a larger
  evasion test set is on the roadmap.

## A diagnosed live limitation: false positives on famous sites

Live-testing the served model against well-known legitimate sites (via
`test_known_legitimate_sites.py`) surfaced false positives — e.g.
`https://www.youtube.com/` flagged at 95.1%. SHAP traced it to two
compounding data biases, confirmed by intervention
(`src/training/model_diagnostics.py`):

1. **`www.` is counted as a subdomain.** Every legitimate bare-root URL in
   the training data (1,412 of them) comes from Tranco with no leading
   `www.`; bare-root-plus-a-subdomain appears only in phishing.
2. **Platform-inflated popularity rank** (see the ablation correction
   above) reinforces the same false signal for platform-hosted phishing.

Removing only `www.` from the URL (changing nothing else) drops YouTube's
phishing probability from 95.1% to 1.6%; five other sites tested the same
way show the same pattern. **Status: diagnosed, not yet fixed** — the fix
(ignore `www.` in subdomain/hostname features; treat hosting platforms as
their own registered domains) is deferred until after current reporting
milestones, since retraining now would change every reported number.

## Phase 5 — Serving: ~40% done

FastAPI app (`src/serving/app.py`) with two pages:
- **`/demo`** — the checker. Live feature extraction, prediction, and a
  SHAP explanation shown as diverging bars with the actual feature value
  next to each one, a plain-English summary sentence, and a live-checks
  strip (DNS/WHOIS/page-fetch/Tranco status) so a failed lookup is visible
  rather than silently guessed around.
- **`/analysis`** — Model Analysis (`src/serving/model_analysis.py` +
  `src/serving/static/analysis.html`). Reads `model_comparison.json` and
  `model_analysis_data.json` (produced by `export_model_analysis.py`,
  which re-scores the four saved models and verifies the result matches
  the saved comparison before writing anything): the F1 ranking table,
  ROC/PR curves, confusion matrices for all four models, the robustness
  table, SHAP importance by feature family, the calibration diagram, and
  an "honest limitations" card summarizing the `www.`/rank findings above.

No accounts or prediction history yet (that's the remaining Phase 5 work).

## What's next

1. Fix the two diagnosed biases (`www.` handling, hosting-platform
   domains) and retrain all four models on the enlarged dataset (~600
   more phishing URLs already collected, not yet processed)
2. Re-run `export_model_analysis.py` and the calibration/SHAP scripts so
   the web app's numbers refresh automatically
3. Broaden the model comparison (LightGBM, SVM) under the same protocol
4. Stacking ensemble; character-level deep-learning baseline
5. Schema-based drift detection (KS-test), a `tests/` folder
6. MLflow registry, scheduled retraining, drift alerts
7. Accounts and prediction history

## Repository

GitHub repo (`Akshat426/Phising-Detection-Application`), actively
committed with descriptive messages at each meaningful milestone
(environment setup, each bug fix, each ablation, each robustness check).
`.env`, raw/processed data, and model artifacts (`.pkl`) are git-ignored;
code and experiment-report JSON files are tracked. **Note:** since
`data/processed/` and `*.pkl` are gitignored, a fresh clone cannot run the
app or reproduce reported numbers without first running the full pipeline
(see README's "Running the data pipeline" and "Training and evaluation").
