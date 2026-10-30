# Project status — last updated 24 Oct 2026

## Done
- Dataset: 5,296 URLs, split 3,706 / 795 / 795 (47.7 % phishing each); schema and drift checks clean.
- Extractor fixes: `www.` handling, platform attribution, **URL canonicalization** (14 unit tests).
- Six models trained and compared on identical data with bootstrap CIs and exact McNemar tests (XGBoost selected on validation; test F1 0.9831).
- Augmentation strategies compared on validation by a rule fixed in advance; symmetric root-stripping selected.
- Explainability (SHAP), calibration (Brier 0.0126), domain-grouped CV, rank ablations and stress test, large-sample adversarial evaluation, temporal drift (KS), latency.
- Live check on 19 well-known sites: 0 false positives.
- IEEE paper (PDF, Word and LaTeX source).

## Known issues / not done
- The `/analysis` page still shows numbers from earlier pipeline versions (stale `model_analysis_data.json`, hard-coded Brier value, old limitations text); regenerate with `export_model_analysis.py` and update `model_analysis.py`.
- No forward-in-time evaluation and no cross-dataset test (PhreshPhish, PhiUSIIL).
- Residual coverage gap: 34 phishing root-with-query URLs have no legitimate counterpart in training.
- Model depends strongly on the Tranco rank (34.6 % false alarms if the rank is unavailable at inference); consider a rank-free fallback model.
- LightGBM not usable in the development environment; no deep-learning baseline.

## Next
1. Sync the `/analysis` page with the paper's numbers (if the web app is part of the submission).
2. Forward-in-time split and cross-dataset validation.
3. Rank-dropout training or a rank-free fallback model, compared with paired statistics.
