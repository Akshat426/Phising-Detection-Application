# Project status — last updated 1 Nov 2026

## Done
- Dataset: 5,296 URLs, split 3,706 / 795 / 795 (47.7 % phishing each); schema and drift checks clean.
- Extractor fixes: `www.` handling, platform attribution, **URL canonicalization** (14 unit tests).
- Six models trained and compared on identical data with bootstrap CIs and exact McNemar tests (XGBoost selected on validation; test F1 0.9831; statistically tied with Random Forest and HistGradientBoosting).
- Augmentation strategies compared on validation by a rule fixed in advance; symmetric root-stripping selected.
- Explainability (SHAP), calibration (Brier 0.0126), domain-grouped CV, rank ablations and stress test, large-sample adversarial evaluation, temporal drift (KS), latency.
- Live check on 19 well-known sites: 0 false positives.
- **Phishing-side forward-in-time test** with a size-matched random control (`forward_in_time.py`): no measurable loss on the October batch (99.7 % vs 99.5 % recall); small gap (97.9 % vs 99.0 %) with a mid-batch cutoff. Legitimate URLs have no timestamps, so this is not a full forward-in-time split.
- **Rank-free models compared with the full model** using paired statistics (`rank_free_comparison.py`): rank-free Random Forest has test F1 0.9792 (McNemar p = 0.66, no detectable loss) and about 4 % false alarms versus 34.6 % for the full model when the rank is unavailable.
- IEEE paper (PDF, Word and LaTeX source) — draft; will need the two experiments above added.

## Known issues / not done
- The `/analysis` page still shows numbers from earlier pipeline versions; it is being rebuilt together with the new UI.
- No cross-dataset test (PhreshPhish, PhiUSIIL).
- Residual coverage gap: 34 phishing root-with-query URLs have no legitimate counterpart in training.
- The full model depends strongly on the Tranco rank (34.6 % false alarms if the rank is unavailable at inference). A rank-free fallback model exists but is not wired into the app.
- LightGBM not usable in the development environment; no deep-learning baseline.
- Unit tests cover URL canonicalization only.

## Next
1. New `/analysis` page and UI (backend ready, waiting on the design reference).
2. Optional: route to the rank-free model when the rank is unavailable; wider unit tests; cross-dataset validation.
