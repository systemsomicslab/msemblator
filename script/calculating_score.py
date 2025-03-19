import os
import joblib
import pandas as pd

def predict_and_append(df, machine_dir, adduct_column="adduct"):
    """
    For the given DataFrame, this function uses the appropriate model (adduct-specific or the default 'all' model)
    to compute and append the 'confidence_score' column representing the predicted probability of TF=1.

    - If an adduct-specific model exists, it is used.
    - Otherwise, the default 'all' model is used.
    """
    # Define the feature columns used during training.
    feature_columns = [
        "Score_NZ",
        "Score_NZ_diff",
        "normalized_rank",
        "tool_name_buddy",
        "tool_name_msfinder",
        "tool_name_sirius"
    ]

    # Create a copy of the original DataFrame.
    df_original = df.copy()

    # Ensure that the DataFrame contains all required feature columns; if missing, add them with 0.
    df_predict = df_original.copy()
    for col in feature_columns:
        if col not in df_predict.columns:
            df_predict[col] = 0
    df_predict[feature_columns] = df_predict[feature_columns].fillna(0)

    # Load the default 'all' model.
    model_all_path = os.path.join(machine_dir, "random_forest_all.pkl")
    model_all = joblib.load(model_all_path)

    # Preload all available adduct-specific models (excluding the 'all' model) into a dictionary.
    model_dict = {}
    for filename in os.listdir(machine_dir):
        if "random_forest_" in filename and filename != "random_forest_all.pkl":
            tokens = filename.split("_")
            adduct_name = tokens[-2].replace(".pkl", "")
            model_path = os.path.join(machine_dir, filename)
            model_dict[adduct_name] = joblib.load(model_path)

    # Function to predict the probability for a given row.
    def predict_row(row):
        adduct = str(row.get(adduct_column, "all")).replace("+", "plus").replace("-", "minus")
        model = model_dict.get(adduct, model_all)
        row_features = row[feature_columns].values.reshape(1, -1)

        if pd.isna(row_features).any():
            print(f"Warning: NaN detected in row {row.name}, filling with 0")
            row_features = pd.DataFrame(row_features).fillna(0).values.reshape(1, -1)
        return model.predict_proba(row_features)[:, 1][0]

    df_original["confidence_score"] = df_predict.apply(predict_row, axis=1)

    return df_original

def aggregate_probability_with_rank_top3(df, top_n=3):
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
    df_filtered = df.groupby(["filename", "formula"], as_index=False).agg(
        confidence_score_sum=("confidence_score", "sum")
    )

    # Check for NaN: fill NaN values in confidence_score_sum with 0
    df_filtered["confidence_score_sum"] = df_filtered["confidence_score_sum"].fillna(0)

    # Create a rank column based on aggregated confidence_score per filename
    df_filtered["rank"] = df_filtered.groupby("filename")["confidence_score_sum"].rank(
        method="first", ascending=False
    ).astype(int)  

    # Debug: print the head of df_filtered to verify rank column
    print("df_filtered.head():")
    print(df_filtered.head())

    # Select the top 'top_n' entries based on rank
    df_top_smiles = df_filtered[df_filtered["rank"] <= top_n]

    # Merge to retrieve additional details from the original DataFrame
    df_max_info = df.merge(df_top_smiles, on=["filename", "formula"], how="inner")

    # After merge, check if the rank column still exists
    print("df_max_info after merge:")
    print(df_max_info.head())
    print("Columns in df_max_info:", df_max_info.columns)

    # If the rank column is missing, perform a re-merge to add it back
    if "rank" not in df_max_info.columns:
        df_max_info = df_max_info.merge(df_top_smiles[["filename", "formula", "rank"]], on=["filename", "formula"], how="left")
    df_max_info["rank"] = df_max_info["rank"].astype(int)
    df_max_info["filename"] = df_max_info["filename"].astype(str)
    df_max_info["adduct"] = df_max_info["adduct"].astype(str)

    df_max_info["Tool_Rank"] = df_max_info.apply(lambda row: 
        [f"msbuddy(rank={row['rank']})" if row.get("tool_name_buddy", 0) == 1 else None,
         f"MS-FINDER(rank={row['rank']})" if row.get("tool_name_msfinder", 0) == 1 else None,
         f"SIRIUS(rank={row['rank']})" if row.get("tool_name_sirius", 0) == 1 else None], axis=1)

    df_max_info["Used_Tool"] = df_max_info["Tool_Rank"].apply(lambda x: ", ".join(filter(None, x)))

    df_summary = df_max_info.groupby(["filename", "rank"]).agg(
        adduct=("adduct", "first"),
        confidence_score_sum=("confidence_score_sum", "first"),
        formula=("formula", "first"),
        Used_Tool=("Used_Tool", lambda x: ", ".join(set(x)))  
    ).reset_index()

    return df_summary
