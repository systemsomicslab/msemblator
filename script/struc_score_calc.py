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

import pandas as pd

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
    def extract_tool_rank(row):
        tools = []
        if row.get("tool_name_metfrag", 0) == 1:
            tools.append(f"MetFrag(rank={int(row['rank'])})")
        if row.get("tool_name_msfinder", 0) == 1:
            tools.append(f"MS-FINDER(rank={int(row['rank'])})")
        if row.get("tool_name_sirius", 0) == 1:
            tools.append(f"SIRIUS(rank={int(row['rank'])})")
        return ", ".join(tools)

    df["Used_Tool"] = df.apply(extract_tool_rank, axis=1)

    grouped = df.groupby(["filename", "Canonical_SMILES"], as_index=False).agg(
        confidence_score_sum=("confidence_score", "sum")
    )
    grouped["rank"] = grouped.groupby("filename")["confidence_score_sum"] \
                             .rank(method="first", ascending=False).astype(int)
    top = grouped[grouped["rank"] <= top_n]

    merged = df.merge(
        top.rename(columns={"rank": "agg_rank"})[["filename", "Canonical_SMILES", "agg_rank", "confidence_score_sum"]],
        on=["filename", "Canonical_SMILES"],
        how="inner"
    )

    summary = merged.groupby(["filename", "Canonical_SMILES", "agg_rank"]).agg(
        adduct=("adduct", "first"),
        confidence_score_sum=("confidence_score_sum", "first"),
        Used_Tool=("Used_Tool", lambda x: ", ".join(sorted(set(x))))
    ).reset_index().rename(columns={"agg_rank": "rank"})

    return summary






