"""Collision-energy and Short InChIKey aggregation and the GLACIER ranker features.

Per query and candidate structure (Short InChIKey), for each similarity:
    GLACIER_<S>_CE<e>    best score of the candidate's predicted spectra at energy e
                         (NaN when the library has no spectrum at that energy)
    GLACIER_<S>_Max      max over the configured energies (each similarity separately)
    GLACIER_<S>_Rank     1 = highest Max within the query; ties are ordered by
                         Short InChIKey so the result is reproducible
    GLACIER_<S>_ScoreDiff  Max minus the Max of the next rank (0 for the last rank)
Rank and ScoreDiff follow the definitions used for MetFrag and MS-FINDER in
summaries/metfrag_summary.py and summaries/msfinder_struc_summary.py (rank by
order within the query, score difference to the next candidate). Values are raw:
no normalization is applied. Missing values stay NaN (they are not set to 0).

Per query (the same value on every candidate row of the query):
    GLACIER_Candidate_Count  number of candidate structures (Short InChIKeys) that
                             GLACIER scored for the query. With one candidate the
                             top rank is trivially correct; with hundreds it rarely
                             is, so scores and ranks mean different things.

Spectra without a collision energy, or with an energy that is not configured,
are kept in the raw output but do not contribute to the features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from msemblator.glacier.config import GlacierConfig

KEY_COLUMNS = ["query_index", "filename", "adduct", "formula", "Short_InChIKey"]
CANDIDATE_COUNT = "GLACIER_Candidate_Count"


def ce_label(energy: float) -> str:
    return f"CE{energy:g}"


def feature_columns(config: GlacierConfig) -> list[str]:
    columns = []
    for method in config.similarities:
        label = config.method_label(method)
        columns += [f"GLACIER_{label}_{ce_label(e)}" for e in config.collision_energies]
        columns += [f"GLACIER_{label}_Max", f"GLACIER_{label}_Rank", f"GLACIER_{label}_ScoreDiff"]
    return columns + [CANDIDATE_COUNT]


def adduct_feature_name(adduct: str) -> str:
    """Same naming as scoring/struc_score_calc.machine_input_generation."""
    name = adduct.replace("+", "plus").replace("-", "minus").replace("[", "").replace("]", "")
    return f"adduct_{name}"


def rank_and_difference(df: pd.DataFrame, score: str, group: str = "query_index",
                        tie_breaker: str = "Short_InChIKey") -> tuple[pd.Series, pd.Series]:
    ordered = df.sort_values([group, score, tie_breaker], ascending=[True, False, True],
                             na_position="last", kind="mergesort")
    values = ordered[score]
    valid = values.notna()
    rank = valid.astype(int).groupby(ordered[group]).cumsum().where(valid)
    following = values.groupby(ordered[group]).shift(-1)
    difference = (values - following).where(valid)
    difference[valid & following.isna()] = 0.0
    return rank.reindex(df.index).astype("Int64"), difference.reindex(df.index)


def build_features(raw: pd.DataFrame, config: GlacierConfig) -> pd.DataFrame:
    """One row per (query, Short InChIKey) with the raw similarity features."""
    methods = list(config.similarities)
    labels = {m: config.method_label(m) for m in methods}
    energies = list(config.collision_energies)
    base_columns = KEY_COLUMNS + ["InChIKey", "SMILES", "n_spectra", "n_inchikeys", "n_collision_energies"]
    columns = base_columns + feature_columns(config) + [adduct_feature_name(a) for a in config.adduct_features]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    ce = raw["collision_energy"].to_numpy(dtype=float)
    configured = np.array(energies)
    match = np.isclose(ce[:, None], configured[None, :], atol=1e-6)
    used = raw[match.any(axis=1)].copy()
    if used.empty:
        return pd.DataFrame(columns=columns)
    used["_ce"] = configured[match[match.any(axis=1)].argmax(axis=1)]

    # Best score per candidate and energy (several spectra of one Short InChIKey and
    # energy, e.g. stereoisomers, are all kept in the raw output).
    per_ce = used.groupby(KEY_COLUMNS + ["_ce"], sort=False)[methods].max()
    wide = per_ce.unstack("_ce")
    wide = wide.reindex(columns=pd.MultiIndex.from_product([methods, energies]))
    wide.columns = [f"GLACIER_{labels[m]}_{ce_label(e)}" for m, e in wide.columns]
    features = wide.reset_index()

    # Representative structure: the first InChIKey in sorted order.
    first = (used.sort_values(["query_index", "Short_InChIKey", "InChIKey"], kind="mergesort")
             .drop_duplicates(KEY_COLUMNS)[KEY_COLUMNS + ["InChIKey", "SMILES"]])
    counts = used.groupby(KEY_COLUMNS, sort=False).agg(
        n_spectra=("InChIKey", "size"), n_inchikeys=("InChIKey", "nunique"),
        n_collision_energies=("_ce", "nunique")).reset_index()
    features = features.merge(first, on=KEY_COLUMNS, how="left").merge(counts, on=KEY_COLUMNS, how="left")

    for method in methods:
        label = labels[method]
        ce_columns = [f"GLACIER_{label}_{ce_label(e)}" for e in energies]
        features[f"GLACIER_{label}_Max"] = features[ce_columns].max(axis=1, skipna=True)
        rank, difference = rank_and_difference(features, f"GLACIER_{label}_Max")
        features[f"GLACIER_{label}_Rank"] = rank
        features[f"GLACIER_{label}_ScoreDiff"] = difference

    features[CANDIDATE_COUNT] = features.groupby("query_index")["Short_InChIKey"].transform("size").astype("int64")

    for adduct in config.adduct_features:
        features[adduct_feature_name(adduct)] = features["adduct"].eq(adduct).astype("int8")
    features = features.sort_values(["query_index", f"GLACIER_{labels[methods[0]]}_Rank", "Short_InChIKey"],
                                    na_position="last", kind="mergesort")
    return features[columns].reset_index(drop=True)
