import os
import pandas as pd
import joblib

from msemblator.chemistry.convert_struc_data_type import smiles_list_to_inchikeys, normalize_rank_score, safe_log2
from msemblator.scoring.struc_score_normalization import ClippingTransformer


def process_sirius_output(
    sirius_folder, machine_dir, name_adduct_df, summary_inchikey_df, summary_smiles_df,
    class_summary_df, smiles_score_df, top_n=3,
):
    """
    Process a SIRIUS structure_identifications_top-100.tsv file.

    Returns
    -------
    tuple
        (
            sirius_inchikey_df,
            sirius_smiles_df,
            class_summary_df,
            smiles_score_df,
        )
    """
    sirius_path = os.path.join(sirius_folder, "structure_identifications_top-100.tsv")

    # Check whether the SIRIUS output file exists
    if not os.path.isfile(sirius_path):
        print(f"No SIRIUS file found: {sirius_path}")
        return summary_inchikey_df, summary_smiles_df, class_summary_df, smiles_score_df

    # Read the SIRIUS output file
    try:
        sirius6_df = pd.read_csv(sirius_path, sep="\t", low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        print(f"Error reading {sirius_path}: {exc}")
        return summary_inchikey_df, summary_smiles_df, class_summary_df, smiles_score_df

    # Check whether the SIRIUS output is empty
    if sirius6_df.empty:
        print(f"SIRIUS output is empty: {sirius_path}")
        return summary_inchikey_df, summary_smiles_df, class_summary_df, smiles_score_df

    # Normalize the SMILES column name
    if "SMILES" not in sirius6_df.columns:
        if "smiles" in sirius6_df.columns:
            sirius6_df = sirius6_df.rename(columns={"smiles": "SMILES"})
        else:
            raise KeyError(
                "Neither 'SMILES' nor 'smiles' exists in the SIRIUS output.\n"
                f"Available columns: {sirius6_df.columns.tolist()}"
            )

    # Validate the required columns
    required_columns = {"structurePerIdRank", "CSI:FingerIDScore", "adduct", "SMILES"}
    missing_columns = required_columns - set(sirius6_df.columns)

    if missing_columns:
        raise KeyError(
            f"Required columns are missing: {sorted(missing_columns)}\n"
            f"Available columns: {sirius6_df.columns.tolist()}"
        )

    # Extract a fallback identifier from the folder name
    folder_name = os.path.basename(os.path.normpath(sirius_folder))
    folder_identifier = folder_name.rsplit("_", maxsplit=1)[-1]

    # Determine the compound or feature identifier
    if "filename" not in sirius6_df.columns:
        identifier_candidates = [
            "mappingFeatureId", "featureId", "feature_id",
            "compoundId", "compound_id", "name"
        ]
        identifier_column = next(
            (column for column in identifier_candidates if column in sirius6_df.columns),
            None,
        )
        sirius6_df["filename"] = (
            sirius6_df[identifier_column]
            if identifier_column is not None
            else folder_identifier
        )

    # Normalize the filename values
    sirius6_df["filename"] = (
        sirius6_df["filename"].astype("string").str.strip()
        .replace("", pd.NA).fillna(folder_identifier)
        .str.rsplit("_", n=1).str[-1]
    )

    # Convert numeric columns
    sirius6_df["structurePerIdRank"] = pd.to_numeric(
        sirius6_df["structurePerIdRank"], errors="coerce"
    )
    sirius6_df["CSI:FingerIDScore"] = pd.to_numeric(
        sirius6_df["CSI:FingerIDScore"], errors="coerce"
    )
    sirius6_df = sirius6_df.dropna(
        subset=["filename", "structurePerIdRank", "CSI:FingerIDScore"]
    ).copy()
    sirius6_df["rank"] = sirius6_df["structurePerIdRank"].astype(int)

    # Sort before calculating differences
    sirius6_df = sirius6_df.sort_values(
        ["filename", "rank", "CSI:FingerIDScore"], ascending=[True, True, False]
    ).reset_index(drop=True)

    # Remove duplicate filename/rank entries; retain the highest score.
    duplicate_mask = sirius6_df.duplicated(subset=["filename", "rank"], keep=False)

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        print(
            f"Warning: {duplicate_count} rows have duplicate "
            "'filename + rank' combinations. "
            "The row with the highest CSI:FingerIDScore is retained."
        )
        sirius6_df = sirius6_df.drop_duplicates(
            subset=["filename", "rank"], keep="first"
        ).copy()

    # Calculate score difference within each compound
    next_rank = sirius6_df.groupby("filename")["rank"].shift(-1)
    next_score = sirius6_df.groupby("filename")["CSI:FingerIDScore"].shift(-1)
    consecutive_rank = next_rank.eq(sirius6_df["rank"] + 1)
    sirius6_df["Score_Difference"] = 0.0
    sirius6_df.loc[consecutive_rank, "Score_Difference"] = (
        sirius6_df.loc[consecutive_rank, "CSI:FingerIDScore"]
        - next_score.loc[consecutive_rank]
    )

    # Normalize adduct notation
    replace_dict = {
        r"\[M \+ H3N \+ H\]\+": "[M+NH4]+",
        r"\[M \+ CH2O2 - H\]-": "[M+FA-H]-",
    }
    sirius6_df["adduct"] = sirius6_df["adduct"].fillna("").astype(str)

    for pattern, replacement in replace_dict.items():
        sirius6_df["adduct"] = sirius6_df["adduct"].str.replace(
            pattern, replacement, regex=True
        )

    # Load normalization pipelines
    score_pipeline_path = os.path.join(machine_dir, "pipeline_csi_fingerid.pkl")
    score_diff_pipeline_path = os.path.join(machine_dir, "pipeline_csi_fingerid_diff.pkl")

    if not os.path.isfile(score_pipeline_path):
        raise FileNotFoundError(f"Pipeline not found: {score_pipeline_path}")
    if not os.path.isfile(score_diff_pipeline_path):
        raise FileNotFoundError(f"Pipeline not found: {score_diff_pipeline_path}")

    sirius_score_pipeline = joblib.load(score_pipeline_path)
    sirius_score_diff_pipeline = joblib.load(score_diff_pipeline_path)

    # Normalize SIRIUS scores
    normalization_score = sirius_score_pipeline.transform(sirius6_df[["CSI:FingerIDScore"]])
    normalization_score_diff = sirius_score_diff_pipeline.transform(
        sirius6_df[["Score_Difference"]]
    )
    sirius6_df["normalization_score"] = normalization_score.ravel()
    sirius6_df["normalization_score_diff"] = normalization_score_diff.ravel()

    # Add common output columns
    sirius6_df["tool_name"] = "sirius"
    sirius6_df["Used_tools"] = sirius6_df["rank"].map(lambda rank: f"SIRIUS_Rank:{rank}")

    # Select top N candidates for each compound
    filtered_df = (
        sirius6_df.sort_values(["filename", "rank"])
        .groupby("filename", group_keys=False)
        .head(top_n)
        .copy()
    )

    # Prepare score output
    sirius_score_calc_df = filtered_df[
        [
            "filename", "rank", "SMILES", "normalization_score", "normalization_score_diff",
            "tool_name", "Used_tools",
        ]
    ].copy()

    # Map adducts from name_adduct_df
    if {"filename", "adduct"}.issubset(name_adduct_df.columns):
        adduct_map = (
            name_adduct_df.drop_duplicates(subset=["filename"], keep="first")
            .set_index("filename")["adduct"]
        )
        sirius_score_calc_df["adduct"] = sirius_score_calc_df["filename"].map(adduct_map)
    else:
        # Fall back to the adduct contained in the SIRIUS output
        sirius_adduct_map = (
            filtered_df.drop_duplicates(subset=["filename"], keep="first")
            .set_index("filename")["adduct"]
        )
        sirius_score_calc_df["adduct"] = sirius_score_calc_df["filename"].map(
            sirius_adduct_map
        )

    # normalize_rank_score may modify in place or return a DataFrame
    normalized_result = normalize_rank_score(sirius_score_calc_df)
    if normalized_result is not None:
        sirius_score_calc_df = normalized_result

    # Convert SMILES to InChIKey
    filtered_df["SMILES"] = filtered_df["SMILES"].fillna("").astype(str)
    filtered_df["InChIKey"] = smiles_list_to_inchikeys(filtered_df["SMILES"])
    filtered_df["InChIKey"] = (
        pd.Series(filtered_df["InChIKey"], index=filtered_df.index).fillna("").astype(str)
    )

    # Create pivot tables
    inchikey_pivot = filtered_df.pivot_table(
        index="filename", columns="rank", values="InChIKey", aggfunc="first"
    )
    inchikey_pivot.columns = [f"sirius_structure_{rank}" for rank in inchikey_pivot.columns]
    # Limit the SMILES summary to rank 1.
    top5_smiles_df = filtered_df[filtered_df["rank"] <= 5].copy()
    smiles_pivot = top5_smiles_df.pivot_table(
        index="filename", columns="rank", values="SMILES", aggfunc="first"
    )
    smiles_pivot.columns = [f"sirius_structure_{rank}" for rank in smiles_pivot.columns]

    # Merge summary tables
    sirius_inchikey_df = summary_inchikey_df.merge(
        inchikey_pivot.reset_index(), on="filename", how="outer"
    )
    sirius_smiles_df = summary_smiles_df.merge(
        smiles_pivot.reset_index(), on="filename", how="outer"
    )

    # Extract rank-1 candidates for classification
    sirius_class_data = filtered_df.loc[
        filtered_df["rank"].eq(1), ["filename", "InChIKey", "SMILES"]
    ].copy()
    sirius_class_data = sirius_class_data[
        sirius_class_data["InChIKey"].str.strip().ne("")
        & sirius_class_data["SMILES"].str.strip().ne("")
    ].copy()
    sirius_class_data["tool_name"] = "SIRIUS"

    # Append results
    class_summary_df = pd.concat([class_summary_df, sirius_class_data], ignore_index=True)
    smiles_score_df = pd.concat([smiles_score_df, sirius_score_calc_df], ignore_index=True)

    return sirius_inchikey_df, sirius_smiles_df, class_summary_df, smiles_score_df
