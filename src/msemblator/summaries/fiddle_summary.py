"""FIDDLE features for the formula ranking model.

Same recipe as scripts/sirius_free/fit_fiddle_normalization.py, which fitted the
pipelines and built the training features:
- score_diff = score minus the next rank's score (0 for the last rank), computed
  over all FIDDLE candidates before keeping the top_n ranks;
- Score_NZ / Score_NZ_diff from pipeline_fiddle_score(_diff).pkl;
- normalized_rank = 1 - log(rank) / log(top_n + 1) (fixed base, not the max rank);
- formulas rewritten in Hill order, as FIDDLE writes elements alphabetically.
"""
import os
import re

import joblib
import numpy as np
import pandas as pd

TOOL_NAME = "fiddle"


def hill_formula(formula):
    """FIDDLE writes C17ClH18N4O; the other tools use Hill order (C17H18ClN4O)."""
    counts = {}
    for element, number in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula)):
        counts[element] = counts.get(element, 0) + (int(number) if number else 1)
    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else []) + sorted(e for e in counts if e not in ("C", "H"))
    else:
        order = sorted(counts)
    return "".join(f"{e}{counts[e] if counts[e] != 1 else ''}" for e in order)


def process_fiddle_summary(fiddle_folder, machine_dir, name_adduct_df, summary_df, score_df, top_n=100):
    candidates_path = os.path.join(fiddle_folder, "fiddle_candidates.csv")
    if not os.path.isfile(candidates_path):
        print(f"No FIDDLE candidates found in {fiddle_folder}")
        return summary_df, score_df

    fiddle_df = pd.read_csv(candidates_path, dtype={"query_id": str})
    if fiddle_df.empty:
        print(f"FIDDLE output is empty: {candidates_path}")
        return summary_df, score_df

    fiddle_df = fiddle_df.sort_values(["query_id", "rank"]).reset_index(drop=True)
    fiddle_df["score_diff"] = (
        fiddle_df["score"] - fiddle_df.groupby("query_id")["score"].shift(-1)
    ).fillna(0.0)
    fiddle_df = fiddle_df[fiddle_df["rank"] <= top_n].copy()

    score_pipeline = joblib.load(os.path.join(machine_dir, "pipeline_fiddle_score.pkl"))
    SD_pipeline = joblib.load(os.path.join(machine_dir, "pipeline_fiddle_score_diff.pkl"))
    fiddle_df["Score_NZ"] = score_pipeline.transform(fiddle_df[["score"]]).ravel()
    fiddle_df["Score_NZ_diff"] = SD_pipeline.transform(fiddle_df[["score_diff"]]).ravel()

    fiddle_df["filename"] = fiddle_df["query_id"]
    fiddle_df["formula"] = fiddle_df["formula"].map(hill_formula)
    fiddle_df["adduct"] = fiddle_df["filename"].map(name_adduct_df.set_index("filename")["adduct"])

    fiddle_score_calc_df = fiddle_df[
        ["filename", "adduct", "rank", "formula", "Score_NZ", "Score_NZ_diff"]
    ].copy()
    fiddle_score_calc_df["tool_name"] = TOOL_NAME
    fiddle_score_calc_df["Used_tools"] = fiddle_score_calc_df["rank"].apply(lambda r: f"FIDDLE_Rank:{r}")
    fiddle_score_calc_df["normalized_rank"] = 1 - np.log(fiddle_score_calc_df["rank"]) / np.log(top_n + 1)

    top5_df = fiddle_df[fiddle_df["rank"] <= 5]
    formula_pivot = top5_df.pivot(index="filename", columns="rank", values="formula")
    formula_pivot.columns = [f"fiddle_formula_{rank}" for rank in formula_pivot.columns]
    fiddle_formula_df = summary_df.merge(formula_pivot.reset_index(), on=["filename"], how="outer")

    score_df = pd.concat([score_df, fiddle_score_calc_df], ignore_index=True)
    return fiddle_formula_df, score_df
