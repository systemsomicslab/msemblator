import numpy as np
import os
import joblib
import pandas as pd
from convert_struc_data_type import convert_to_shortinchikey

def predict_and_append(df, machine_dir, adduct_column="adduct"):
    """
    This function adds the `Predicted_Probability_TF1` column to the given DataFrame
    by using the appropriate model (either per-adduct or the `all` model).

    - If the adduct column exists, the corresponding model is used.
    - If no specific adduct model is available, the `all` model is used.
    """

    feature_columns = [
        "normalization_Zscore_metfrag",
        "normalization_Zscore_msfinder",
        "normalization_Zscore_sirius",
        "normalization_z_score_diff_metfrag",
        "normalization_z_score_diff_msfinder",
        "normalization_z_score_diff_sirius",
        "normalized_rank_metfrag",
        "normalized_rank_msfinder",
        "normalized_rank_sirius",
        'adduct_MplusHplus',
        'adduct_MplusNaplus',
        'adduct_MplusNH4plus',
        'adduct_MminusHminus',
        'adduct_MplusClminus',
        'adduct_MplusFAminusHminus'
        ]

    # Ensure that all required feature columns exist, filling missing ones with 0
    df_onehot = df.reindex(columns=feature_columns, fill_value=0)
    df_onehot.fillna(0, inplace=True)  

    df_original = df.copy()

    # Load the general model that applies to all adducts
    model_all_path = os.path.join(machine_dir, "random_forest_final_all.pkl")
    model_all = joblib.load(model_all_path)

    # Extract available adduct models from the directory
    trained_adducts = [
        file.split("_")[-2].replace(".pkl", "") for file in os.listdir(machine_dir) if "random_forest_" in file
    ]

    predicted_probs = []

    for _, row in df.iterrows():
        # Convert the adduct name to a format that matches the model filenames
        adduct = str(row.get(adduct_column, "all")).replace("+", "plus").replace("-", "minus") 

        # Select the specific adduct model if available; otherwise, use the general model
        model_path = os.path.join(machine_dir, f"random_forest_{adduct}_final.pkl") if adduct in trained_adducts else model_all_path

        # Load the selected model
        logistic_model = joblib.load(model_path)

        # Extract the feature values for prediction
        row_features = row[feature_columns].values.reshape(1, -1)

        # Handle potential NaN values in the feature data
        if pd.isna(row_features).any():
            print(f"Warning: NaN detected in row {row.name}, filling with 0")
            row_features = pd.DataFrame(row_features).fillna(0).values.reshape(1, -1)

        # Predict the probability using the selected model
        predicted_proba = logistic_model.predict_proba(row_features)[:, 1][0]
        predicted_probs.append(predicted_proba)

    # Append the predicted probabilities as a new column
    df_original["confidence_score"] = predicted_probs

    return df_original



def aggregate_probability_with_rank(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Aggregate confidence scores per structure per file, rank them, and list the tools that reported each structure.

    Parameters:
        df (pd.DataFrame): Input DataFrame with the following columns:
            - filename
            - Canonical_SMILES
            - confidence_score
            - rank (tool-specific rank)
            - adduct
            - tool_name_metfrag, tool_name_msfinder, tool_name_sirius (as flags: 0 or 1)
        top_n (int): Number of top SMILES to return per filename.

    Returns:
        pd.DataFrame: Summary with filename, structure, rank, score, and tools used.
    """
    # aggregate confidence scores
    grouped = df.groupby(["filename", "Canonical_SMILES"], as_index=False).agg(
        confidence_score_sum=("confidence_score", "sum")
    )
    grouped["rank"] = grouped.groupby("filename")["confidence_score_sum"] \
                             .rank(method="first", ascending=False).astype(int)
    grouped = grouped.sort_values(["filename", "rank"], ascending=[True, True])
    top = grouped[grouped["rank"] <= top_n]

    merged = df.merge(
        top.rename(columns={"rank": "agg_rank"})[["filename", "Canonical_SMILES", "agg_rank", "confidence_score_sum"]],
        on=["filename", "Canonical_SMILES"],
        how="inner"
    )

    summary = merged.groupby(["filename", "Canonical_SMILES", "agg_rank"]).agg(
        adduct=("adduct", "first"),
        confidence_score_sum=("confidence_score_sum", "first"),
        Used_Tool=("Used_tools", lambda x: ','.join(sorted(set(','.join(x).split(',')))))
    ).reset_index().rename(columns={"agg_rank": "rank"})

    return summary


def machine_input_generation(df):
    adduct_list = ['[M+H]+', '[M+Na]+', '[M+NH4]+', '[M-H]-', '[M+Cl]-', '[M+FA-H]-']
    # Convert SMILES to Short InChIKey
    convert_to_shortinchikey(df, "SMILES", new_column_name="Short_InChIKey")

    for tool in ["metfrag", "sirius", "msfinder"]:
        tool_mask = df["tool_name"] == tool
        df.loc[tool_mask, f"normalization_{tool}_score"] = df.loc[tool_mask, "normalization_Zscore"]
        df.loc[tool_mask, f"normalization_{tool}_diff"] = df.loc[tool_mask, "normalization_z_score_diff"]
        df.loc[tool_mask, f"normalization_{tool}_rank"] = df.loc[tool_mask, "normalized_rank"]

    base_columns = ["filename", "adduct", "Short_InChIKey"]

    # Keep only score-related columns and convert to long format
    score_cols = ["normalization_Zscore", "normalization_z_score_diff", "normalized_rank"]
    long_df = df[base_columns + ["tool_name"] + score_cols].copy()

    # Pivot to wide format using tool_name and score columns
    wide_df = long_df.pivot_table(
        index=base_columns,
        columns="tool_name",
        values=score_cols,
        aggfunc="max"  # If duplicates exist, take the maximum value
    )

    # Flatten MultiIndex columns
    wide_df.columns = [f"{score}_{tool}" for score, tool in wide_df.columns]
    wide_df = wide_df.reset_index()
    wide_df = wide_df.fillna(0)

    # Retrieve representative SMILES for each Short_InChIKey (e.g., take the first one)
    smiles_df = df.drop_duplicates(subset=["Short_InChIKey"])[["Short_InChIKey", "SMILES"]]

    used_tools_df = (
        df.groupby(base_columns)['Used_tools']
        .apply(lambda x: ','.join(sorted(', '.join(x).split(','))))
        .reset_index()
    )

    # Merge SMILES into wide_df
    wide_df = pd.merge(wide_df, smiles_df, on="Short_InChIKey", how="left")
    wide_df = wide_df.merge(used_tools_df, on=base_columns, how="left")

    for ad in adduct_list:
        ad_norm = ad.replace("+", "plus").replace("-", "minus").replace("[", "").replace("]", "")
        col_name = f"adduct_{ad_norm}"
        
        wide_df[col_name] = (
            wide_df['adduct']
            .str.replace("+", "plus", regex=False)
            .str.replace("-", "minus", regex=False)
            == ad_norm
        ).astype(int)   
    return wide_df