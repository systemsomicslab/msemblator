import numpy as np
import os
import joblib
import pandas as pd

def predict_and_append(df, machine_dir, adduct_column="adduct"):
    """
    This function adds the `Predicted_Probability_TF1` column to the given DataFrame
    by using the appropriate model (either per-adduct or the `all` model).

    - If the adduct column exists, the corresponding model is used.
    - If no specific adduct model is available, the `all` model is used.
    """

    feature_columns = [
        "normalization_Zscore",
        "normalization_z_score_diff",
        "normalized_rank",
        "tool_name_metfrag",
        "tool_name_msfinder",
        "tool_name_sirius"
    ]

    # Ensure that all required feature columns exist, filling missing ones with 0
    df_onehot = df.reindex(columns=feature_columns, fill_value=0)
    df_onehot.fillna(0, inplace=True)  

    df_original = df.copy()

    # Load the general model that applies to all adducts
    model_all_path = os.path.join(machine_dir, "random_forest_all_optimized.pkl")
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
        model_path = os.path.join(machine_dir, f"random_forest_{adduct}_optimized.pkl") if adduct in trained_adducts else model_all_path

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


def aggregate_probability_with_rank(df, top_n=3):
    """
    Aggregates `confidence_score` for each `Canonical_SMILES` within each `filename`,
    selects the top `top_n` `Canonical_SMILES` with the highest aggregated score per `filename`,
    and retrieves the corresponding original tool ranks.

    Parameters:
        df (pd.DataFrame): Input DataFrame, must include original tool ranks like `rank_metfrag`, etc.
        top_n (int): Number of top-ranked SMILES to include per filename (based on aggregated score).

    Returns:
        pd.DataFrame: Summary DataFrame with filename, adduct, score, SMILES, and original tool ranks.
    """
    # Step 1: Aggregate confidence_score by filename + Canonical_SMILES
    df_filtered = df.groupby(["filename", "Canonical_SMILES"], as_index=False).agg(
        confidence_score_sum=("confidence_score", "sum")
    )
    df_filtered["confidence_score_sum"] = df_filtered["confidence_score_sum"].fillna(0)

    # Step 2: Assign rank based on aggregated score within each filename
    df_filtered["rank"] = df_filtered.groupby("filename")["confidence_score_sum"] \
                                     .rank(method="first", ascending=False).astype(int)

    # Step 3: Keep only top-N entries
    df_top_smiles = df_filtered[df_filtered["rank"] <= top_n]

    # Step 4: Merge original DataFrame with top-ranked SMILES (do not change this line)
    df_max_info = df.merge(df_top_smiles, on=["filename", "Canonical_SMILES"], how="inner")

    # Step 5: Ensure correct data types
    df_max_info["filename"] = df_max_info["filename"].astype(str)
    df_max_info["adduct"] = df_max_info["adduct"].astype(str)

    # Step 6: Restore rank if missing after merge
    if "rank" not in df_max_info.columns:
        df_max_info = df_max_info.merge(
            df_top_smiles[["filename", "Canonical_SMILES", "rank"]],
            on=["filename", "Canonical_SMILES"],
            how="left"
        )
    df_max_info["rank"] = df_max_info["rank"].astype(int)

    # Step 7: Create Tool_Rank using original tool-specific rank values (do not change logic here)
    def extract_tool_rank(row):
        ranks = []
        if row.get("tool_name_metfrag", 0) == 1 and pd.notnull(row.get("rank")):
            ranks.append(f"MetFrag(rank={int(row['rank'])})")
        if row.get("tool_name_msfinder", 0) == 1 and pd.notnull(row.get("rank")):
            ranks.append(f"MS-FINDER(rank={int(row['rank'])})")
        if row.get("tool_name_sirius", 0) == 1 and pd.notnull(row.get("rank")):
            ranks.append(f"SIRIUS(rank={int(row['rank'])})")
        return ", ".join(ranks)


    df_max_info["Tool_Rank"] = df_max_info.apply(extract_tool_rank, axis=1)
    df_max_info["Used_Tool"] = df_max_info["Tool_Rank"]

    # Step 8: Group and summarize final output
    df_summary = df_max_info.groupby(["filename", "Canonical_SMILES", "rank"]).agg(
        adduct=("adduct", "first"),
        confidence_score_sum=("confidence_score_sum", "first"),
        Used_Tool=("Used_Tool", lambda x: ", ".join(set(x)))
    ).reset_index()

    return df_summary




