# Phishing URL Detector — an audited ML pipeline

URL-based phishing detection with **35 features** (20 lexical, 4 domain, 11 page-content) on **5,296 URLs**
(2,526 phishing from OpenPhish, 2,770 legitimate from the Tranco list). Six classifiers are compared, and — the
main point of the project — the pipeline is **audited for dataset artifacts** that inflate accuracy.
A FastAPI app serves live predictions with SHAP explanations.

> Paper: *Auditing a Phishing URL Detector: Finding and Removing Dataset Artifacts That Inflate Machine-Learning Accuracy* (IEEE conference format).

## Headline results (held-out test set, n = 795)

| Model | Test F1 [95% CI] | McNemar p vs XGBoost |
|---|---|---|
| **XGBoost** (selected on validation) | **0.9831** [0.9736, 0.9914] | – |
| Random Forest | 0.9805 [0.9706, 0.9896] | 0.73 (tied) |
| HistGradientBoosting | 0.9805 [0.9702, 0.9895] | 0.73 (tied) |
| Decision Tree | 0.9690 [0.9566, 0.9804] | 0.0034 |
| SVM (RBF) | 0.9657 [0.9525, 0.9780] | 0.0044 |
| Logistic Regression | 0.9595 [0.9451, 0.9725] | 0.0003 |

XGBoost: precision 96.9 %, recall 99.7 %, AUC 0.9988, Brier 0.0126. Its 13 test errors are 12 false positives and 1 missed phishing URL.
The first, leaky pipeline scored 0.9934; the lower number above is the honest one.

## What the audit found (and fixed)

1. **Popularity-rank confound** – free-hosting platforms lent their Tranco rank and WHOIS age to phishing pages, and a leading `www.` was counted as a subdomain.
2. **Trailing-slash leak** – every bare-root phishing URL ended in `/` and every bare-root legitimate URL did not (974 of 974 in each class).
   On the unaugmented model, adding one `/` flipped 17.7 % of legitimate roots to "phishing" and stripping the path let 48.4 % of phishing URLs evade.
   Fixed by **URL canonicalization** inside the feature extractor (`canonicalize_url`), shared by training and the live app.
3. **Augmentation trade-off** – phishing-only root-stripping removes evasion but causes false alarms (7.5 % on validation); **symmetric** augmentation
   (both classes) keeps evasion at 0.8 % [0.3, 2.3] on test with 0.5 % false flags.

## Known limitations (read before trusting the numbers)

- Legitimate URLs come from a ranked list, so the Tranco rank partly encodes the data source: with the rank blanked, 34.6 % of legitimate URLs are flagged.
- Test prevalence is 47.7 % phishing. At 1 % prevalence the same error rates would give a precision of roughly 26 % (illustration, assumes the rates transfer).
- Splits are random, not forward in time; 7 of 35 features drift across collection batches.
- 795 test URLs ⇒ ±0.9 F1 points; XGBoost, Random Forest and HistGradientBoosting cannot be separated.
- Evasion probes are cheap lexical string tricks, not adaptive attacks.

## Repository layout

```
src/data_pipeline/   fetch_openphish.py, fetch_tranco.py, build_dataset.py, split_dataset.py, data_validation.py,
                     canonicalize_dataset.py, augment_symmetric.py (+ older augment_adversarial_examples.py)
src/features/        feature_extractor.py (incl. canonicalize_url), transformation.py, url_perturbations.py
src/training/        model_trainer.py, extra_models.py, compare_models.py, adversarial_eval.py,
                     explain_model.py, calibration_check.py, model_diagnostics.py, drift_and_latency.py
src/serving/         app.py (FastAPI: /demo checker, /analysis page, POST /predict)
tests/               test_url_canonicalization.py (14 tests)
final_model/         saved metrics and reports (model files are not committed)
```

## Reproducing the results (PowerShell, project root, seed 42)

```
python src\data_pipeline\fetch_openphish.py
python src\data_pipeline\fetch_tranco.py
python src\data_pipeline\build_dataset.py --workers 16
python src\data_pipeline\split_dataset.py
python src\data_pipeline\data_validation.py
python src\data_pipeline\augment_symmetric.py
Copy-Item -Force data\processed\train_augmented_symmetric.csv data\processed\train.csv
python src\features\transformation.py
python src\training\model_trainer.py
python src\training\extra_models.py --models svm_rbf hist_gb
python src\training\compare_models.py
python src\training\adversarial_eval.py --tag canon_symm --split test
python src\training\explain_model.py
python src\training\calibration_check.py
python src\training\model_diagnostics.py
python -m pytest tests -q
```

Notes: features are extracted live (WHOIS, DNS, HTTP), so a rebuilt dataset will differ from the one in the paper. The trainer is deterministic on identical input.
Re-running `transformation.py` overwrites `data/processed/transform_artifacts`; keep a copy per model variant.
LightGBM crashed in the development environment (Windows access violation), so scikit-learn's HistGradientBoosting is used instead.

## Run the app

```
uvicorn src.serving.app:app --port 8000
```
Open http://127.0.0.1:8000/demo (URL checker) and /analysis (model analysis page).
The raw OpenPhish feed is not part of the repository; collect it with `fetch_openphish.py` under OpenPhish's terms.
