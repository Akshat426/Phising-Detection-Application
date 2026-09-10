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
- Uses current, real data: PhishTank (verified phishing URLs) + Tranco
  (a maintained, research-grade domain popularity ranking) instead of a
  decade-old static dataset

## Status: Phase 1 (data + features) in progress

See `CP_Roadmap.md` (in the original project chat) for the full semester plan.

## Setup

This project needs full internet access (PhishTank, Tranco, live site fetches,
WHOIS) - it will NOT work in a network-sandboxed environment.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your PhishTank API key
```

## Running the data pipeline (in order)

```bash
# 1. Fetch verified phishing URLs from PhishTank
python src/data_pipeline/fetch_phishtank.py --limit 5000

# 2. Fetch legitimate URLs sampled across Tranco rank bands
python src/data_pipeline/fetch_tranco.py --total 5000

# 3. Extract features for every URL and build the labeled dataset
#    Start small to make sure everything works before committing hours to it:
python src/data_pipeline/build_dataset.py --sample 200

# Once that looks good, run the full build (this takes a while - it's making
# thousands of real network requests). Safe to Ctrl+C and re-run; it resumes
# from data/processed/dataset_checkpoint.csv automatically.
python src/data_pipeline/build_dataset.py --workers 16
```

## Project structure

```
data/
  raw/            PhishTank + Tranco URL lists (not committed - regenerate via scripts)
  processed/      Final extracted-feature dataset (not committed)
src/
  features/       feature_extractor.py - the core, reusable feature computation logic
  data_pipeline/  scripts that fetch URLs and build the dataset
  training/       (Phase 2+) model training, comparison, explainability
  serving/        (Phase 5+) FastAPI app for live predictions
tests/            unit tests for the feature extractor (add these before Phase 2)
```

## A note on feature extraction reliability

Every feature that requires a network call (WHOIS, live page fetch) is
allowed to fail and produce a real missing value (`None`/`NaN`) rather than
a fake default. This is a deliberate design choice, not an oversight - some
sites will be down, some WHOIS servers will rate-limit, and pretending that
didn't happen would quietly corrupt the dataset. The transformation stage
(Phase 1, still to build) handles these missing values explicitly via
imputation, and this rate of missingness itself is worth reporting in the
final CP writeup as a real-world data quality finding.
