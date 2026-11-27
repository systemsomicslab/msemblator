"""Run summary and annotation accuracy of the GLACIER features.

Accuracy is reported per adduct (and for all adducts together) with two denominators:
    all_queries          every query of an adduct whose library was available
    queries_with_answer  queries whose correct structure is among the candidates
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from msemblator.glacier import queries as q
from msemblator.glacier.config import GlacierConfig
from msemblator.glacier.features import ce_label

TOP_K = (1, 5, 10)


def accuracy_table(features: pd.DataFrame, queries: pd.DataFrame, config: GlacierConfig,
                   available_adducts) -> pd.DataFrame:
    """Top-k accuracy of each similarity's Max rank (features must have TF).

    all_queries counts every selected query of an adduct with an available library,
    including those without a valid formula or peaks.
    """
    scope = queries[queries["adduct"].isin(set(available_adducts)) & queries["status"].ne("not_selected")]
    answers = features[features["TF"].eq(1)]
    rows = []
    for adduct, group in [("all", scope)] + list(scope.groupby("adduct")):
        ids = set(group["query_index"])
        hits = answers[answers["query_index"].isin(ids)]
        n_all, n_answer = len(ids), hits["query_index"].nunique()
        for method in config.similarities:
            label = config.method_label(method)
            rank = hits.groupby("query_index")[f"GLACIER_{label}_Rank"].min().astype(float)
            row = {"adduct": adduct, "similarity": method, "n_queries": n_all, "n_queries_with_answer": n_answer}
            for k in TOP_K:
                n_hit = int((rank <= k).sum())
                row[f"top{k}"] = n_hit
                row[f"top{k}_acc_all"] = n_hit / n_all if n_all else np.nan
                row[f"top{k}_acc_with_answer"] = n_hit / n_answer if n_answer else np.nan
            row["median_answer_rank"] = float(rank.median()) if len(rank) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def spectra_count_check(features: pd.DataFrame, config: GlacierConfig) -> pd.DataFrame:
    """Max scores by the number of predicted spectra per candidate (bias check)."""
    if features.empty:
        return pd.DataFrame()
    rows = []
    for n_spectra, group in features.groupby("n_spectra"):
        row = {"n_spectra": int(n_spectra), "n_candidates": len(group)}
        if "TF" in group:
            row["n_correct"] = int(group["TF"].sum())
        for method in config.similarities:
            column = f"GLACIER_{config.method_label(method)}_Max"
            row[f"{method}_max_mean"] = group[column].mean()
            if "TF" in group:
                row[f"{method}_max_mean_correct"] = group.loc[group["TF"].eq(1), column].mean()
                row[f"{method}_max_mean_incorrect"] = group.loc[group["TF"].eq(0), column].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(result, queries: pd.DataFrame | None = None, features: pd.DataFrame | None = None) -> dict:
    """Counts for glacier_summary.json; pass the labelled tables when answers are known."""
    config = result.config
    queries = result.queries if queries is None else queries
    features = result.features if features is None else features
    raw = result.raw
    status_counts = queries["status"].value_counts().to_dict()
    # Queries of an adduct with a library that were selected for this run.
    in_scope = queries["adduct"].ne("") & ~queries["status"].isin(["not_selected", q.STATUS_UNSUPPORTED_ADDUCT])
    summary = {
        "n_queries_input": int(len(queries)),
        "queries_by_precursor_type": queries["adduct_raw"].value_counts().to_dict(),
        "queries_by_adduct": queries.loc[queries["adduct"].ne(""), "adduct"].value_counts().to_dict(),
        "query_status": {k: int(v) for k, v in status_counts.items()},
        "n_queries_with_candidates": int((queries["status"] == q.STATUS_OK).sum()),
        "n_queries_without_candidates": int((queries["status"] == q.STATUS_NO_CANDIDATE).sum()),
        "n_queries_unsupported_adduct": int((queries["status"] == q.STATUS_UNSUPPORTED_ADDUCT).sum()),
        "unsupported_adducts": sorted(queries.loc[queries["status"] == q.STATUS_UNSUPPORTED_ADDUCT,
                                                  "adduct_raw"].unique().tolist()),
        "n_queries_library_not_available": int((queries["status"] == q.STATUS_LIBRARY_NOT_AVAILABLE).sum()),
        "adducts_library_not_available": sorted({lib["adduct"] for lib in result.libraries
                                                 if lib.get("status") != "available"}),
        "n_unsupported_adduct_types": int(queries.loc[queries["status"] == q.STATUS_UNSUPPORTED_ADDUCT,
                                                      "adduct_raw"].nunique()),
        "n_queries_invalid_peaks_removed": int((in_scope & (queries["n_invalid_peaks"] > 0)).sum()),
        "precursor_peak_removal": {
            "settings": dict(config.precursor_peak_removal),
            "measured_spectra": queries.loc[in_scope, "precursor_peak_removal"].value_counts().to_dict(),
            "measured_peaks_removed": int(queries.loc[in_scope, "n_precursor_peaks_removed"].sum()),
            "predicted_spectra_with_peak_removed": sum(lib.get("n_spectra_precursor_removed", 0)
                                                       for lib in result.libraries),
            "predicted_peaks_removed": sum(lib.get("n_precursor_peaks_removed", 0) for lib in result.libraries),
        },
        "n_raw_rows": int(len(raw)),
        "n_similarity_failed_rows": int((raw["status"] == "similarity_failed").sum()) if len(raw) else 0,
        "n_similarity_failed_queries": int((queries["status"] == q.STATUS_SIMILARITY_FAILED).sum()),
        "n_raw_rows_missing_collision_energy": int((raw["status"] == "missing_collision_energy").sum())
        if len(raw) else 0,
        "n_feature_rows": int(len(features)),
        "timings_seconds": result.timings,
    }
    if not features.empty:
        summary["n_candidates_missing_ce"] = {}
        for energy in config.collision_energies:
            column = f"GLACIER_{config.method_label(config.similarities[0])}_{ce_label(energy)}"
            summary["n_candidates_missing_ce"][ce_label(energy)] = int(features[column].isna().sum())
        summary["candidates_per_query"] = features.groupby("query_index").size().describe().round(2).to_dict()
    if "TF" in features:
        with_answer = features.loc[features["TF"].eq(1), "query_index"].nunique()
        summary["n_queries_with_correct_candidate"] = int(with_answer)
    if "formula_is_reference" in queries:
        scope = queries[queries["status"].isin([q.STATUS_OK, q.STATUS_NO_CANDIDATE])]
        summary["formula_condition"] = {
            "formula_source": sorted(queries["formula_source"].unique().tolist()),
            "n_queries_in_scope": int(len(scope)),
            "n_formula_is_reference": int(scope["formula_is_reference"].sum()),
        }
    return summary
