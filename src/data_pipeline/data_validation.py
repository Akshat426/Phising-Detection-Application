"""
Data validation: schema checks + train/test distribution drift detection.

Two separate concerns, both worth checking before training:

  1. Schema validation - catches structural problems (wrong columns,
     unexpected label values, an entirely-empty feature column) that would
     indicate a bug upstream in the data pipeline, not real-world noise.

  2. Drift detection - checks whether the train and test sets come from
     genuinely the same distribution using the Kolmogorov-Smirnov test per
     numeric feature. This matters more than it might seem: since Tranco
     rank bands and OpenPhish's feed both change over time, running
     build_dataset.py weeks apart could accidentally introduce systematic
     differences between old and new rows that a naive train/test split
     wouldn't catch. A KS-test flag here is a legitimate early-warning
     signal, not just a pipeline formality (this is also a place where the
     reference project already had a reasonable idea worth keeping -
     rewritten here rather than reused).

Usage:
    python data_validation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "validation_reports"

EXPECTED_COLUMNS = {
    "url_length", "hostname_length", "path_length", "num_dots", "num_hyphens",
    "num_underscores", "num_slashes", "num_question_marks", "num_equals",
    "num_at_symbols", "num_percent", "num_digits", "num_subdomains",
    "has_ip_address", "uses_https", "has_port", "is_known_shortener",
    "double_slash_in_path", "suspicious_tld_in_path", "brand_keyword_in_subdomain",
    "domain_age_days", "domain_registration_length_days", "dns_resolves",
    "tranco_rank", "num_forms", "form_action_external_ratio", "num_iframes",
    "num_external_links", "num_internal_links", "external_link_ratio",
    "has_password_field", "favicon_external", "num_scripts",
    "title_hostname_overlap", "response_status_code", "url", "label",
}

DRIFT_P_VALUE_THRESHOLD = 0.01  # conventional threshold; flag features below this


def validate_schema(df: pd.DataFrame, name: str) -> list[str]:
    """Returns a list of human-readable problems found. Empty list = passed."""
    problems = []

    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        problems.append(f"[{name}] Missing expected columns: {sorted(missing_cols)}")

    if "label" in df.columns:
        bad_labels = set(df["label"].dropna().unique()) - {0, 1}
        if bad_labels:
            problems.append(f"[{name}] Unexpected label values found: {bad_labels}")
    else:
        problems.append(f"[{name}] No 'label' column present")

    fully_empty_cols = [c for c in df.columns if df[c].isna().all()]
    if fully_empty_cols:
        problems.append(f"[{name}] Columns that are 100% missing (likely a bug, not noise): {fully_empty_cols}")

    if len(df) == 0:
        problems.append(f"[{name}] Dataset is empty")

    return problems


def detect_drift(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Runs a KS-test per numeric feature comparing train vs test distributions.

    Returns a dict of {feature: {"p_value": float, "drifted": bool}}.
    Rows with NaN are dropped per-feature before testing (KS-test needs
    complete samples) - features with heavy missingness will naturally have
    smaller effective sample sizes here, which is worth noting rather than
    hiding.
    """
    numeric_cols = train_df.select_dtypes(include="number").columns
    numeric_cols = [c for c in numeric_cols if c != "label"]

    results = {}
    for col in numeric_cols:
        train_vals = train_df[col].dropna()
        test_vals = test_df[col].dropna()
        if len(train_vals) < 5 or len(test_vals) < 5:
            results[col] = {"p_value": None, "drifted": None, "note": "too few non-missing values to test"}
            continue
        stat, p_value = ks_2samp(train_vals, test_vals)
        results[col] = {
            "p_value": float(p_value),
            "drifted": bool(p_value < DRIFT_P_VALUE_THRESHOLD),
        }
    return results


def main():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run split_dataset.py first to create train.csv and test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    all_problems = validate_schema(train_df, "train") + validate_schema(test_df, "test")

    print("=== Schema validation ===")
    if all_problems:
        for p in all_problems:
            print(f"  FAIL: {p}")
    else:
        print("  All checks passed.")

    print("\n=== Drift detection (train vs test, KS-test) ===")
    drift_results = detect_drift(train_df, test_df)
    drifted_features = [f for f, r in drift_results.items() if r.get("drifted")]

    for feature, result in sorted(drift_results.items(), key=lambda x: (x[1].get("p_value") or 1)):
        p = result.get("p_value")
        flag = " <-- DRIFT" if result.get("drifted") else ""
        p_str = f"{p:.4f}" if p is not None else "n/a"
        print(f"  {feature:<35} p={p_str}{flag}")

    if drifted_features:
        print(f"\n  {len(drifted_features)} feature(s) show statistically significant drift: {drifted_features}")
        print("  This doesn't necessarily block training, but is worth investigating and")
        print("  reporting - e.g. it may reflect genuinely different URL populations if")
        print("  train/test rows were collected on different days from a live feed.")
    else:
        print("\n  No significant drift detected between train and test.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_problems": all_problems,
        "schema_valid": len(all_problems) == 0,
        "drift_results": drift_results,
        "drifted_features": drifted_features,
    }
    report_path = REPORTS_DIR / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved -> {report_path}")

    if all_problems:
        raise SystemExit(
            "\nSchema validation FAILED - fix the data pipeline before proceeding to "
            "training. Training on data that failed schema checks would silently "
            "produce misleading results."
        )


if __name__ == "__main__":
    main()
