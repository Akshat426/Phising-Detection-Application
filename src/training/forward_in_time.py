"""
Forward-in-time evaluation (phishing side) with a size-matched random control.

Question: is a model trained on EARLIER phishing batches able to catch LATER
phishing, as well as a model that saw the same number of phishing URLs drawn
from ALL periods?

Only the phishing class has fetch times (data/raw/phishing_urls.csv). The
legitimate class is a single Tranco snapshot, so legitimate URLs are NOT split
by time: all training legitimate URLs are used in every model, and false alarms
are measured on the held-out legitimate URLs. This is therefore a
phishing-side temporal-shift test, not a full forward-in-time split.

Arms (per cutoff), each repeated over several seeds:
  forward : train = training legit + training phishing fetched <= cutoff
  control : train = training legit + a RANDOM subset of training phishing
            (same size as the forward arm, drawn from all periods)
Evaluation (held-out = validation + test splits; neither was used to choose
anything here): recall on phishing fetched AFTER the cutoff (primary), recall
on held-out phishing fetched <= cutoff, false-alarm rate on held-out
legitimate URLs, and F1 on (later phishing + legitimate).

Model: the final model's hyperparameters (cloned from final_model/model.pkl),
trained on the UNAUGMENTED canonicalized base. This is not the final model
(which uses symmetric augmentation); the point is to isolate temporal shift.

Usage (project root):
    python src/training/forward_in_time.py
    python src/training/forward_in_time.py --seeds 20 --cutoffs 2026-10-01 median

Output: final_model/forward_in_time_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.transformation import DataTransformer  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MIN_N = 30


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def design(tr: DataTransformer, df: pd.DataFrame) -> pd.DataFrame:
    out = tr.transform(df)
    return out[[c for c in out.columns if c not in ("label", "url")]]


def fit_predict(train_df, eval_df, params, seed):
    tr = DataTransformer().fit(train_df)
    Xtr = design(tr, train_df)
    Xev = design(tr, eval_df)[Xtr.columns]
    ytr = train_df["label"].to_numpy()
    n_pos, n_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    p = dict(params)
    p["scale_pos_weight"] = n_neg / n_pos
    p["random_state"] = seed
    model = XGBClassifier(**p)
    model.fit(Xtr, ytr)
    return model.predict_proba(Xev)[:, 1], Xtr.shape[1]


def score(prob, y, later, early):
    pred = prob >= 0.5
    legit = y == 0
    sel = later | legit
    return {
        "recall_later": float(pred[later].mean()),
        "recall_early": float(pred[early].mean()),
        "false_alarm": float(pred[legit].mean()),
        "f1_later": float(f1_score(y[sel], pred[sel])),
    }


def summarise(rows):
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows])
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "min": float(v.min()), "max": float(v.max())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=str(ROOT / "final_model" / "model.pkl"))
    ap.add_argument("--train-file", default="train_pre_augmentation.csv")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cutoffs", nargs="+", default=["2026-10-01", "median"])
    args = ap.parse_args()

    final = joblib.load(args.model_path)
    if isinstance(final, dict):
        final = final.get("model", final)
    params = final.get_params()
    print(f"Hyperparameters cloned from {Path(args.model_path).name}: "
          f"depth={params.get('max_depth')}, n_estimators={params.get('n_estimators')}, "
          f"lr={params.get('learning_rate')}")

    raw = pd.read_csv(ROOT / "data" / "raw" / "phishing_urls.csv")
    raw["t"] = pd.to_datetime(raw["fetched_time"], utc=True, errors="coerce")
    tmap = raw.dropna(subset=["t"]).drop_duplicates("url").set_index("url")["t"]

    ds = pd.read_csv(PROCESSED / "dataset.csv")
    all_ph_t = ds.loc[ds["label"] == 1, "url"].map(tmap).dropna()
    print("\nPhishing rows per fetch date (whole dataset):")
    print(all_ph_t.dt.date.value_counts().sort_index().to_string())

    train = pd.read_csv(PROCESSED / args.train_file)
    held = pd.concat([pd.read_csv(PROCESSED / "validation.csv"),
                      pd.read_csv(PROCESSED / "test.csv")], ignore_index=True)
    for df in (train, held):
        df["t"] = df["url"].map(tmap)

    tr_ph = train[train["label"] == 1]
    print(f"\nTraining base: {len(train)} rows ({len(tr_ph)} phishing, "
          f"{int(tr_ph['t'].notna().sum())} with a fetch time)")
    if tr_ph["t"].notna().mean() < 0.95:
        sys.exit("Less than 95% of training phishing URLs matched a fetch time - URL join is broken "
                 "(is the training file's url column canonicalized/rewritten?). Stopping.")
    train_legit = train[train["label"] == 0]
    train_ph = tr_ph.dropna(subset=["t"])

    y_held = held["label"].to_numpy()
    report = {"model_hyperparameters_from": str(Path(args.model_path).name),
              "train_file": args.train_file, "n_seeds": args.seeds, "configs": {}}

    for cname in args.cutoffs:
        cutoff = all_ph_t.median() if cname == "median" else pd.Timestamp(cname, tz="UTC")
        later = ((held["label"] == 1) & (held["t"] > cutoff)).to_numpy()
        early = ((held["label"] == 1) & (held["t"] <= cutoff)).to_numpy()
        early_train = train_ph[train_ph["t"] <= cutoff]
        n_fwd = len(early_train)
        print(f"\n{'=' * 78}\nCUTOFF {cname} ({cutoff})")
        print(f"  training phishing: {n_fwd} earlier / {len(train_ph) - n_fwd} later; "
              f"held-out phishing: {int(early.sum())} earlier / {int(later.sum())} later; "
              f"held-out legit: {int((y_held == 0).sum())}")
        if later.sum() < MIN_N or early.sum() < MIN_N or n_fwd < MIN_N:
            print("  not enough rows on one side - skipping this cutoff.")
            report["configs"][cname] = {"skipped": True}
            continue

        eval_df = held.drop(columns=["t"])
        fwd_rows, ctl_rows, hits0 = [], [], None
        for s in range(args.seeds):
            tdf = pd.concat([train_legit, early_train]).drop(columns=["t"])
            prob, n_feat = fit_predict(tdf, eval_df, params, seed=s)
            sc = score(prob, y_held, later, early)
            fwd_rows.append(sc)
            if s == 0:
                hits0 = int(round(sc["recall_later"] * later.sum()))

            sub = train_ph.sample(n=n_fwd, random_state=1000 + s)
            cdf = pd.concat([train_legit, sub]).drop(columns=["t"])
            prob, _ = fit_predict(cdf, eval_df, params, seed=s)
            ctl_rows.append(score(prob, y_held, later, early))
            print(f"  seed {s:>2}: forward later-recall {fwd_rows[-1]['recall_later']:.3f} | "
                  f"control {ctl_rows[-1]['recall_later']:.3f}", flush=True)

        fs, cs = summarise(fwd_rows), summarise(ctl_rows)
        lo, hi = wilson(hits0, int(later.sum()))
        frac = float(np.mean([r["recall_later"] <= fs["recall_later"]["mean"] for r in ctl_rows]))
        print(f"\n  model inputs: {n_feat}")
        print(f"  {'metric':<14}{'forward (mean +/- sd)':>26}{'control (mean +/- sd)':>26}")
        for k in ("recall_later", "recall_early", "false_alarm", "f1_later"):
            print(f"  {k:<14}{fs[k]['mean']:>14.4f} +/- {fs[k]['sd']:.4f}"
                  f"{cs[k]['mean']:>14.4f} +/- {cs[k]['sd']:.4f}")
        print(f"  forward later-recall, seed 0 Wilson 95% CI: [{lo:.3f}, {hi:.3f}] (n={int(later.sum())})")
        print(f"  share of control seeds with later-recall <= forward mean: {frac:.2f}")
        report["configs"][cname] = {
            "cutoff": str(cutoff), "n_train_phishing_forward": n_fwd,
            "n_heldout_phishing_earlier": int(early.sum()), "n_heldout_phishing_later": int(later.sum()),
            "n_heldout_legit": int((y_held == 0).sum()),
            "forward": fs, "control": cs,
            "forward_seed0_later_recall_wilson95": [lo, hi],
            "share_control_seeds_leq_forward_mean": frac,
        }

    dest = ROOT / "final_model" / "forward_in_time_report.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
