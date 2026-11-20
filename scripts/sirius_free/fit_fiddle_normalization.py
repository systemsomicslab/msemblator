#!/usr/bin/env python3
"""Fit the FIDDLE score / score-difference normalisation pipelines and build its features.

Same recipe as the other formula tools (python/msemblator_preprosessing.ipynb):
rank, score and score_diff (score minus the next rank's score; 0 for the last rank)
are taken from every candidate, a Pipeline(clip -> transform -> StandardScaler ->
MinMaxScaler) is fitted on the training spectra only, and saved with joblib.

The final MinMaxScaler(clip=True) maps the features to [0, 1] like the NIST26 ML
tables (tools_result/formula/nist_*_formula*.csv), where a tool without the
candidate is filled with 0 = worst. With z-scores 0 would mean "average" instead.

FIDDLE-specific choices (see the notes in the README / the fit report):
- score is FIDDLE's rescore, a probability in (0, 1) spanning ~10 orders of magnitude.
  Transform: logit. log2 compresses scores above 0.5 into a narrow band, where the
  top candidates are.
- No upper clipping at the 97.5% quantile: over all candidates that quantile is 0.71
  for score and 0.24 for score_diff, which would flatten ~66% / ~53% of the rank-1
  values. Upper bounds are 1 - 1e-6 (score) and 1 (score_diff) instead.
- normalized_rank: 1 - log(rank) / log(top_n + 1), the formula of
  normalize_rank_score with the NIST26 ML tables' top_n = 100 (rank 1 -> 1,
  rank 100 -> 0.0022); it is a fixed formula, so it has no pipeline.

Usage (msemblator environment):
    python scripts/sirius_free/fit_fiddle_normalization.py \
        --fiddle-dir E:/MS-Emblator/data/nist26/tools_result/formula/fiddle \
        --train-msp E:/MS-Emblator/data/nist26/NIST26_id_train.msp \
        --pipeline-dir E:/MS-Emblator/Preprocessing/formula
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from msemblator.scoring.struc_score_normalization import (  # noqa: E402
    ClippingTransformer, safe_log2, safe_logit)

logger = logging.getLogger("fiddle_normalization")
EPS = 1e-6


def msp_names(path: Path) -> set[str]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return {line.split(":", 1)[1].strip() for line in handle if line.upper().startswith("NAME:")}


def add_score_diff(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["query_id", "rank"]).reset_index(drop=True)
    df["score_diff"] = (df["score"] - df.groupby("query_id")["score"].shift(-1)).fillna(0.0)
    return df


def hill_formula(formula: str) -> str:
    """FIDDLE writes elements alphabetically (C17ClH18N4O); the other tools use Hill order (C17H18ClN4O)."""
    counts: dict[str, int] = {}
    for element, number in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula)):
        counts[element] = counts.get(element, 0) + (int(number) if number else 1)
    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else []) + sorted(e for e in counts if e not in ("C", "H"))
    else:
        order = sorted(counts)
    return "".join(f"{e}{counts[e] if counts[e] != 1 else ''}" for e in order)


def describe(values: pd.Series) -> dict:
    q = values.quantile([0, .025, .25, .5, .75, .975, 1])
    return {f"q{k:g}": float(v) for k, v in q.items()} | {"mean": float(values.mean()), "std": float(values.std())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fiddle-dir", required=True, type=Path,
                        help="Folder with fiddle_candidates.csv and spectra_index.csv")
    parser.add_argument("--train-msp", required=True, type=Path,
                        help="MSP whose NAME fields are the training query_ids (fit uses only these)")
    parser.add_argument("--pipeline-dir", required=True, type=Path, help="Where the pipeline .pkl files go")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Ranks kept in the feature table and the base of normalized_rank (NIST26 tables: 100)")
    args = parser.parse_args(argv)

    args.pipeline_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.fiddle_dir / "fiddle_normalization.log", encoding="utf-8")])

    cand = pd.read_csv(args.fiddle_dir / "fiddle_candidates.csv", dtype={"query_id": str})
    index = pd.read_csv(args.fiddle_dir / "spectra_index.csv", dtype={"query_id": str})
    cand = add_score_diff(cand)
    train_ids = msp_names(args.train_msp)
    cand["split"] = np.where(cand["query_id"].isin(train_ids), "train", "test")
    train = cand[cand["split"] == "train"]
    logger.info("%d candidates (%d spectra); fitting on %d train candidates (%d spectra)",
                len(cand), cand.query_id.nunique(), len(train), train.query_id.nunique())

    score_low = float(train["score"].quantile(0.025))
    score_pipeline = Pipeline([
        ("clip", ClippingTransformer(q_low=score_low, q_high=1 - EPS)),
        ("logit", FunctionTransformer(safe_logit, validate=True)),
        ("z_score", StandardScaler()),
        ("minmax", MinMaxScaler(clip=True)),
    ]).fit(train[["score"]])

    diff_low = float(train["score_diff"].quantile(0.025))
    diff_pipeline = Pipeline([
        ("clip_diff", ClippingTransformer(q_low=diff_low, q_high=1.0)),
        ("log_diff", FunctionTransformer(safe_log2, validate=True)),
        ("z_score_diff", StandardScaler()),
        ("minmax_diff", MinMaxScaler(clip=True)),
    ]).fit(train[["score_diff"]])

    score_path = args.pipeline_dir / "pipeline_fiddle_score.pkl"
    diff_path = args.pipeline_dir / "pipeline_fiddle_score_diff.pkl"
    joblib.dump(score_pipeline, score_path)
    joblib.dump(diff_pipeline, diff_path)
    logger.info("saved %s and %s", score_path, diff_path)

    # Reload and apply, so the saved files are what is checked.
    score_pipeline, diff_pipeline = joblib.load(score_path), joblib.load(diff_path)
    cand["Score_NZ"] = score_pipeline.transform(cand[["score"]]).ravel()
    cand["Score_NZ_diff"] = diff_pipeline.transform(cand[["score_diff"]]).ravel()

    features = cand[cand["rank"] <= args.top_n].copy()
    features = features.rename(columns={"query_id": "filename", "formula": "formula_fiddle"})
    features["formula"] = features["formula_fiddle"].map(hill_formula)
    features["adduct"] = features["filename"].map(index.set_index("query_id")["adduct"])
    features["tool_name"] = "fiddle"
    features["Used_tools"] = features["rank"].map(lambda r: f"FIDDLE_Rank:{r}")
    features["normalized_rank"] = 1 - np.log(features["rank"]) / np.log(args.top_n + 1)
    features = features[["filename", "adduct", "rank", "formula", "score", "score_diff",
                         "Score_NZ", "Score_NZ_diff", "normalized_rank", "tool_name", "Used_tools", "split",
                         "formula_fiddle"]]
    features_path = args.fiddle_dir / f"fiddle_features_top{args.top_n}.pkl"
    features.to_pickle(features_path)
    logger.info("saved %s (%d rows, %d spectra)", features_path, len(features), features.filename.nunique())

    report = {
        "fit_rows": int(len(train)), "fit_spectra": int(train.query_id.nunique()),
        "score_pipeline": {"clip": [score_low, 1 - EPS], "transform": "logit",
                           "mean": float(score_pipeline["z_score"].mean_[0]),
                           "scale": float(score_pipeline["z_score"].scale_[0]),
                           "minmax_z_range": [float(score_pipeline["minmax"].data_min_[0]),
                                              float(score_pipeline["minmax"].data_max_[0])]},
        "score_diff_pipeline": {"clip": [diff_low, 1.0], "transform": "log2(x + 1e-6)",
                                "mean": float(diff_pipeline["z_score_diff"].mean_[0]),
                                "scale": float(diff_pipeline["z_score_diff"].scale_[0]),
                                "minmax_z_range": [float(diff_pipeline["minmax_diff"].data_min_[0]),
                                                   float(diff_pipeline["minmax_diff"].data_max_[0])]},
        "Score_NZ_train_all": describe(cand.loc[cand.split == "train", "Score_NZ"]),
        "Score_NZ_diff_train_all": describe(cand.loc[cand.split == "train", "Score_NZ_diff"]),
        f"Score_NZ_top{args.top_n}": describe(features["Score_NZ"]),
        f"Score_NZ_diff_top{args.top_n}": describe(features["Score_NZ_diff"]),
        "normalized_rank_range": [float(features["normalized_rank"].min()), float(features["normalized_rank"].max())],
    }
    (args.fiddle_dir / "fiddle_normalization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("report: %s", json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
