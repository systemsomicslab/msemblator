"""Joining GLACIER features with the MetFrag / MS-FINDER structure candidates.

The existing structure tables (scoring/struc_score_calc.machine_input_generation,
analysis/make_training_data.py) have one row per (filename, adduct, Short_InChIKey).
GLACIER features use the same keys, so the two can be joined with an outer join:
candidates of either side are kept, GLACIER-only candidates are added, and
queries without GLACIER output keep their MetFrag / MS-FINDER candidates.

GLACIER_Candidate_Count is a per-query value and is filled for every candidate
of a query that GLACIER processed. The other GLACIER columns stay NaN for
candidates that GLACIER did not score; the other
tools' columns are NaN for GLACIER-only candidates. How they are filled for a
model (the current models use 0) is left to the training step.
"""
from __future__ import annotations

import pandas as pd

from msemblator.glacier.features import CANDIDATE_COUNT

KEYS = ("filename", "adduct", "Short_InChIKey")
SOURCE_COLUMN = "candidate_source"


def merge_with_candidates(candidates: pd.DataFrame, glacier_features: pd.DataFrame,
                          glacier_columns: list[str] | None = None, keys=KEYS) -> pd.DataFrame:
    """Outer join on (filename, adduct, Short_InChIKey).

    candidate_source is "existing", "glacier" or "both". SMILES is taken from the
    existing table and, for GLACIER-only candidates, from GLACIER.
    """
    keys = list(keys)
    if glacier_columns is None:
        glacier_columns = [c for c in glacier_features.columns if c.startswith("GLACIER_")]
    glacier = glacier_features[keys + ["SMILES"] + glacier_columns].rename(columns={"SMILES": "SMILES_glacier"})
    left, right = candidates.copy(), glacier.copy()
    for frame in (left, right):
        frame["filename"] = frame["filename"].astype(str)
    duplicated = right.duplicated(keys)
    if duplicated.any():
        raise ValueError(f"GLACIER features have {int(duplicated.sum())} duplicated {keys} rows")
    merged = left.merge(right, on=keys, how="outer", indicator=SOURCE_COLUMN)
    merged[SOURCE_COLUMN] = merged[SOURCE_COLUMN].map(
        {"left_only": "existing", "right_only": "glacier", "both": "both"}).astype(str)
    # The candidate count describes the query, so it also applies to candidates that
    # GLACIER did not score; it stays NaN for queries without GLACIER output.
    if CANDIDATE_COUNT in glacier_columns:
        query_keys = [k for k in keys if k != "Short_InChIKey"]
        counts = right[query_keys + [CANDIDATE_COUNT]].drop_duplicates(query_keys)
        merged = merged.drop(columns=CANDIDATE_COUNT).merge(counts, on=query_keys, how="left")
    if "SMILES" in merged:
        merged["SMILES"] = merged["SMILES"].where(merged["SMILES"].notna() & merged["SMILES"].ne(""),
                                                  merged["SMILES_glacier"])
    else:
        merged["SMILES"] = merged["SMILES_glacier"]
    return merged.drop(columns="SMILES_glacier")
