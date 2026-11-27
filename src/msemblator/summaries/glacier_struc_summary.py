"""GLACIER in structure elucidation: run it and join its features with the other tools.

GLACIER compares each spectrum of the formula-fixed MSP with the GLACIER
predicted spectra of every library compound that has the spectrum's formula and
adduct (msemblator.glacier). Its features (GLACIER_*) are joined to the
MetFrag / MS-FINDER / SIRIUS candidates on (filename, Short_InChIKey):

- candidates of either side are kept; GLACIER-only candidates are added;
- GLACIER columns are 0 for candidates GLACIER did not score (as for the other
  tools), except GLACIER_Candidate_Count, which describes the query;
- the features are raw. Scaling belongs to the trained ranking model.
"""
import os

import pandas as pd

from msemblator.glacier.features import CANDIDATE_COUNT
from msemblator.scoring.struc_score_calc import add_adduct_columns

TOOL_NAME = "glacier"
SUMMARY_COLUMNS = 5


def glacier_rank_column(config) -> str:
    """Rank used for the summary columns and the Used_tools tie-break: entropy if computed."""
    method = "entropy" if "entropy" in config.similarities else config.similarities[0]
    return f"GLACIER_{config.method_label(method)}_Rank"


def run_glacier_structure(input_msp, output_dir):
    """Run GLACIER on the formula-fixed MSP and keep its outputs in output_dir.

    Returns (features, rank_column). Queries whose adduct has no GLACIER library
    are only reported in glacier_query_status.csv.
    """
    from msemblator.glacier import load_config, run_glacier

    config = load_config()
    result = run_glacier(input_msp, config)
    os.makedirs(output_dir, exist_ok=True)
    result.features.to_csv(os.path.join(output_dir, "glacier_features.csv"), index=False, encoding="utf-8-sig")
    result.queries.to_csv(os.path.join(output_dir, "glacier_query_status.csv"), index=False, encoding="utf-8-sig")
    status = result.queries["status"].value_counts().to_dict()
    print(f"GLACIER query status: {status}")
    return result.features, glacier_rank_column(config)


def glacier_summary_smiles(features, rank_column, summary_smiles_df):
    """glacier_structure_1..5 columns (top SMILES by rank_column), like the other tools."""
    if features is None or features.empty:
        return summary_smiles_df.copy()
    # The rank is empty (NA) when the similarity could not be calculated.
    top = features[(features[rank_column] <= SUMMARY_COLUMNS).fillna(False).astype(bool)].copy()
    top["filename"] = top["filename"].astype(str)
    top[rank_column] = top[rank_column].astype(int)
    pivot = top.pivot_table(index="filename", columns=rank_column, values="SMILES", aggfunc="first")
    pivot.columns = [f"glacier_structure_{rank}" for rank in pivot.columns]
    return summary_smiles_df.merge(pivot.reset_index(), on="filename", how="outer")


def add_glacier_features(wide_df, features, rank_column, name_adduct_df):
    """Outer join of the GLACIER features with the structure ranking input (one row per candidate)."""
    if features is None or features.empty:
        return wide_df
    glacier_columns = [c for c in features.columns if c.startswith("GLACIER_")]
    glacier = features[["filename", "Short_InChIKey", "SMILES"] + glacier_columns].copy()
    glacier["filename"] = glacier["filename"].astype(str)
    glacier = glacier.rename(columns={"SMILES": "SMILES_glacier"})
    rank = glacier[rank_column]
    glacier["Used_tools_glacier"] = ["" if pd.isna(r) else f"GLACIER_Rank:{int(r)}" for r in rank]

    left = wide_df.copy()
    left["filename"] = left["filename"].astype(str)
    left = left.drop(columns=[c for c in glacier_columns if c in left.columns])
    merged = left.merge(glacier, on=["filename", "Short_InChIKey"], how="outer")

    # Adduct of GLACIER-only candidates comes from the input MSP, as for the other tools.
    adducts = (name_adduct_df.assign(filename=name_adduct_df["filename"].astype(str))
               .drop_duplicates("filename").set_index("filename")["adduct"]
               .fillna("").astype(str).replace({"[M+CO2]-": "[M+FA-H]-"}))
    merged["adduct"] = merged["adduct"].where(merged["adduct"].notna(), merged["filename"].map(adducts))

    if "SMILES" in merged:
        has_smiles = merged["SMILES"].notna() & merged["SMILES"].astype(str).str.strip().ne("")
        merged["SMILES"] = merged["SMILES"].where(has_smiles, merged["SMILES_glacier"])
    else:
        merged["SMILES"] = merged["SMILES_glacier"]

    existing = merged["Used_tools"] if "Used_tools" in merged else pd.Series("", index=merged.index)
    merged["Used_tools"] = [
        ",".join(t for t in (str(a) if pd.notna(a) else "", str(b) if pd.notna(b) else "") if t)
        for a, b in zip(existing, merged["Used_tools_glacier"])
    ]

    # The candidate count describes the query, so every candidate of the query gets it.
    if CANDIDATE_COUNT in glacier_columns:
        counts = glacier.drop_duplicates("filename").set_index("filename")[CANDIDATE_COUNT]
        merged[CANDIDATE_COUNT] = merged["filename"].map(counts)

    merged = merged.drop(columns=["SMILES_glacier", "Used_tools_glacier"])
    return add_adduct_columns(merged)
