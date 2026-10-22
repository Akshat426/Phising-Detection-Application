"""Large-sample evasion evaluation (replaces the 5-URL spot check for the paper).

For a given model + its matching transform artifacts, this applies the same cheap
URL-string tricks as adversarial_check.py (via features.url_perturbations) to
  * phishing test URLs the model correctly catches  -> "evasion rate"
  * legitimate test URLs the model correctly clears -> "false-flag rate" (control)
and reports counts with Wilson 95% confidence intervals.

All URLs come from test.csv by default (or validation.csv with --split validation);
neither is ever used in training.

Place at: src/training/adversarial_eval.py
Examples (project root):
  python src\\training\\adversarial_eval.py --tag post_aug
  python src\\training\\adversarial_eval.py --tag pre_aug ^
      --model-path final_model\\model_PRE_AUG.pkl ^
      --artifacts-dir data\\processed\\transform_artifacts_PRE_AUG
"""
from __future__ import annotations

import argparse
import json
import sys
from math import sqrt
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.feature_extractor import URLFeatureExtractor  # noqa: E402
from features.transformation import DataTransformer  # noqa: E402
from features.url_perturbations import STRATEGIES, LEXICAL_FEATURES  # noqa: E402

def add_trailing_slash(url):
    """site.com -> site.com/ : a pure formatting change (no new information).
    Applies only to bare roots with no slash, i.e. the format all legitimate
    roots in the training data share."""
    p = urlparse(url)
    return url + "/" if p.path == "" and not p.query and not p.fragment else url


ALL_STRATEGIES = dict(STRATEGIES)
ALL_STRATEGIES["add_trailing_slash"] = add_trailing_slash

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FINAL = ROOT / "final_model"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def model_matrix(model, transformed):
    X = transformed[[c for c in transformed.columns if c != "label"]]
    if hasattr(model, "feature_names_in_"):
        X = X[list(model.feature_names_in_)]
    elif hasattr(model, "get_booster"):
        X = X[model.get_booster().feature_names]
    return X


def evaluate(rows, true_label, model, transformer, extractor, n_max, seed, name):
    rows = rows.reset_index(drop=True)
    pred = model.predict(model_matrix(model, transformer.transform(rows)))
    ok = rows[pred == true_label].reset_index(drop=True)
    print(f"\n[{name}] {len(rows)} test URLs; model gets {len(ok)} right "
          f"({len(ok)/len(rows):.1%}) -> these are the URLs we perturb")
    if n_max and len(ok) > n_max:
        ok = ok.sample(n=n_max, random_state=seed).reset_index(drop=True)
        print(f"[{name}] random sample of {n_max} used")

    summary, per_url = {}, []
    for sname, fn in ALL_STRATEGIES.items():
        pert = ok["url"].map(fn)
        app = (pert != ok["url"]).to_numpy()
        n_app = int(app.sum())
        if n_app == 0:
            summary[sname] = {"applicable": 0, "flipped": 0, "rate": None, "ci95": None}
            continue
        sub = ok[app].copy().reset_index(drop=True)
        sub_p = pert[app].reset_index(drop=True)
        for i, u in enumerate(sub_p):
            lex = extractor.extract(u).to_dict()
            for f in LEXICAL_FEATURES:
                sub.at[i, f] = lex[f]
        new_pred = model.predict(model_matrix(model, transformer.transform(sub)))
        flipped = (new_pred != true_label)
        k = int(flipped.sum())
        lo, hi = wilson(k, n_app)
        summary[sname] = {"applicable": n_app, "flipped": k, "rate": k / n_app,
                          "ci95": [lo, hi]}
        for u, pu, fl in zip(ok["url"][app], sub_p, flipped):
            per_url.append({"class": name, "strategy": sname, "url": u,
                            "perturbed_url": pu, "flipped": bool(fl)})
    return summary, per_url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=FINAL / "model.pkl")
    ap.add_argument("--artifacts-dir", type=Path, default=PROCESSED / "transform_artifacts")
    ap.add_argument("--n-examples", type=int, default=0,
                    help="max URLs per class (0 = use all correctly-classified)")
    ap.add_argument("--split", choices=["test", "validation"], default="test",
                    help="which held-out split to attack (use validation to choose "
                         "between variants, test only for final reporting)")
    ap.add_argument("--tag", type=str, default="post_aug")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    print(f"Model: {a.model_path}\nArtifacts: {a.artifacts_dir}")
    model = joblib.load(a.model_path)
    transformer = DataTransformer.load(a.artifacts_dir)
    extractor = URLFeatureExtractor(fetch_content=False, fetch_whois=False)

    test = pd.read_csv(PROCESSED / ("test.csv" if a.split == "test" else "validation.csv"))
    print(f"Split attacked: {a.split}")
    phish = test[test["label"] == 1]
    legit = test[test["label"] == 0]

    res_p, urls_p = evaluate(phish, 1, model, transformer, extractor,
                             a.n_examples, a.seed, "phishing")
    res_l, urls_l = evaluate(legit, 0, model, transformer, extractor,
                             a.n_examples, a.seed, "legitimate")

    def show(title, res, what):
        print(f"\n=== {title} ===")
        print(f"  {'strategy':<30}{'applicable':>11}{what:>9}{'rate':>9}   95% CI")
        for s, r in res.items():
            if r["applicable"] == 0:
                print(f"  {s:<30}{0:>11}{'-':>9}{'-':>9}")
            else:
                print(f"  {s:<30}{r['applicable']:>11}{r['flipped']:>9}"
                      f"{r['rate']:>9.1%}   [{r['ci95'][0]:.1%}, {r['ci95'][1]:.1%}]")

    show("PHISHING URLs: evasion rate (caught -> missed)", res_p, "evaded")
    show("LEGITIMATE URLs (control): false-flag rate (cleared -> flagged)", res_l, "flagged")

    out = FINAL / "explainability"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(urls_p + urls_l).to_csv(out / f"adversarial_eval_{a.tag}_{a.split}_urls.csv", index=False)
    (out / f"adversarial_eval_{a.tag}_{a.split}.json").write_text(json.dumps(
        {"model": str(a.model_path), "phishing": res_p, "legitimate": res_l}, indent=2))
    print(f"\nSaved -> {out / f'adversarial_eval_{a.tag}_{a.split}.json'} (+ _urls.csv)")


if __name__ == "__main__":
    main()
