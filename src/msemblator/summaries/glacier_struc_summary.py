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

    Returns (features, rank_column, mirror_spectra). Queries whose adduct has no
    GLACIER library are only reported in glacier_query_status.csv.
    """
    from msemblator.glacier import load_config, run_glacier

    config = load_config()
    result = run_glacier(input_msp, config, collect_spectra=True)
    os.makedirs(output_dir, exist_ok=True)
    result.features.to_csv(os.path.join(output_dir, "glacier_features.csv"), index=False, encoding="utf-8-sig")
    result.queries.to_csv(os.path.join(output_dir, "glacier_query_status.csv"), index=False, encoding="utf-8-sig")
    status = result.queries["status"].value_counts().to_dict()
    print(f"GLACIER query status: {status}")
    rank_column = glacier_rank_column(config)
    return result.features, rank_column, mirror_spectra(result, rank_column)


MIRROR_COLUMNS = ["filename", "adduct", "Short_InChIKey", "SMILES", "collision_energy"]
MIRROR_TOP = 100   # GLACIER candidates per spectrum whose predicted spectra are kept


def mirror_spectra(result, rank_column, top=MIRROR_TOP):
    """Predicted spectra of the GLACIER candidates (up to `top` per spectrum by rank_column),
    with the similarity of each, for the mirror plots of the GUI."""
    methods = [m for m in result.config.similarities if m in result.raw.columns]
    if result.spectra is None or result.raw.empty or result.features.empty:
        return pd.DataFrame(columns=MIRROR_COLUMNS + methods + ["mz", "intensity"])
    ranks = result.features[["query_index", "Short_InChIKey", rank_column]]
    raw = result.raw[["query_index", "candidate_spec_id"] + MIRROR_COLUMNS + methods]
    keep = raw.merge(ranks, on=["query_index", "Short_InChIKey"], how="inner")
    keep = keep[(keep[rank_column] <= top).fillna(False).astype(bool)]
    keep = keep.merge(result.spectra, on=["adduct", "candidate_spec_id"], how="inner")
    return keep[MIRROR_COLUMNS + methods + ["mz", "intensity"]].reset_index(drop=True)


def write_glacier_spectra(spectra, name_df, output_dir):
    """glacier_spectra.csv.gz (or _1, _2, ...) in output_dir, with the input MSP names."""
    names = dict(zip(name_df["Updated_NAME"].astype(str), name_df["Original_NAME"].astype(str)))
    table = spectra.copy()
    table["filename"] = table["filename"].astype(str).map(names).fillna(table["filename"])
    counter, name = 0, "glacier_spectra.csv.gz"
    while os.path.exists(os.path.join(output_dir, name)):
        counter += 1
        name = f"glacier_spectra_{counter}.csv.gz"
    table.to_csv(os.path.join(output_dir, name), index=False, compression="gzip")
    return name


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
