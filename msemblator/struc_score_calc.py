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

def aggregate_probability_with_rank(df, top_n=3):
    """
    Aggregates `confidence_score` for each `Canonical_SMILES` within each `filename`,
    selects the top `top_n` `Canonical_SMILES` with the highest aggregated score per `filename`,
    and retrieves the corresponding score, tool, and rank in a new DataFrame.

    Parameters:
        df (pd.DataFrame): The original DataFrame.
        top_n (int): Number of top-ranked `Canonical_SMILES` to include per `filename`.

    Returns:
        pd.DataFrame: A new DataFrame containing the aggregated results.
    """
    # Aggregate `confidence_score` for each `Canonical_SMILES` within each `filename`
    df_filtered = df.groupby(["filename", "Canonical_SMILES"], as_index=False).agg(
        confidence_score_sum=("confidence_score", "sum")
    )

    # **NaN check**: Ensure there are no NaN values in confidence_score_sum
    df_filtered["confidence_score_sum"] = df_filtered["confidence_score_sum"].fillna(0)

    # **Create rank column**
    df_filtered["rank"] = df_filtered.groupby("filename")["confidence_score_sum"].rank(
        method="first", ascending=False
    ).astype(int)  # **Convert to integer type**

    # **Debugging: Check if rank column exists**
    print("df_filtered.head():")
    print(df_filtered.head())

    # Select the top `top_n` ranked Canonical_SMILES per filename
    df_top_smiles = df_filtered[df_filtered["rank"] <= top_n]

    # Merge to retrieve additional details from the original DataFrame
    df_max_info = df.merge(df_top_smiles, on=["filename", "Canonical_SMILES"], how="inner")

    # **Check if rank column is retained after merging**
    print("df_max_info after merge:")
    print(df_max_info.head())
    print("Columns in df_max_info:", df_max_info.columns)

    # **If rank is missing after merging, re-merge**
    if "rank" not in df_max_info.columns:
        df_max_info = df_max_info.merge(df_top_smiles[["filename", "Canonical_SMILES", "rank"]], 
                                        on=["filename", "Canonical_SMILES"], how="left")

    # **Ensure rank column remains integer type**
    df_max_info["rank"] = df_max_info["rank"].astype(int)

    # Convert 'filename' and 'adduct' columns to string
    df_max_info["filename"] = df_max_info["filename"].astype(str)
    df_max_info["adduct"] = df_max_info["adduct"].astype(str)

    # Generate a list of tools that reported the selected `Canonical_SMILES` along with their ranks
    df_max_info["Tool_Rank"] = df_max_info.apply(lambda row: 
        [f"MetFrag(rank={row['rank']})" if row.get("tool_name_metfrag", 0) == 1 else None,
         f"MS-FINDER(rank={row['rank']})" if row.get("tool_name_msfinder", 0) == 1 else None,
         f"SIRIUS(rank={row['rank']})" if row.get("tool_name_sirius", 0) == 1 else None], axis=1)

    # Flatten the list and remove None values
    df_max_info["Used_Tool"] = df_max_info["Tool_Rank"].apply(lambda x: ", ".join(filter(None, x)))

    # Keep only relevant columns
    df_summary = df_max_info.groupby(["filename", "rank"]).agg(
        adduct=("adduct", "first"),
        confidence_score_sum=("confidence_score_sum", "first"),
        Canonical_SMILES=("Canonical_SMILES", "first"),
        Used_Tool=("Used_Tool", lambda x: ", ".join(set(x)))  # Combine tool-rank pairs without duplication
    ).reset_index()

    return df_summary

