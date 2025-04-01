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

def aggregate_probability_with_rank_top3(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Aggregate confidence scores per (filename, formula), rank them, and report tools that found each formula.

    Parameters:
        df (pd.DataFrame): Input data. Must include:
            - 'filename', 'formula', 'confidence_score', 'adduct'
            - 'rank' (tool-specific rank)
            - 'tool_name_buddy', 'tool_name_msfinder', 'tool_name_sirius'
        top_n (int): Number of top formulas to retain per filename.

    Returns:
        pd.DataFrame: Summary with confidence_score_sum, rank, and Used_Tool per formula.
    """
    def extract_tool_rank(row):
        tools = []
        if row.get("tool_name_buddy", 0) == 1:
            tools.append(f"msbuddy(rank={int(row['rank'])})")
        if row.get("tool_name_msfinder", 0) == 1:
            tools.append(f"MS-FINDER(rank={int(row['rank'])})")
        if row.get("tool_name_sirius", 0) == 1:
            tools.append(f"SIRIUS(rank={int(row['rank'])})")
        return ", ".join(tools)

    # Add Used_Tool info per row
    df["Used_Tool"] = df.apply(extract_tool_rank, axis=1)

    # Aggregate confidence scores
    grouped = df.groupby(["filename", "formula"], as_index=False).agg(
        confidence_score_sum=("confidence_score", "sum")
    )
    grouped["rank"] = grouped.groupby("filename")["confidence_score_sum"] \
                             .rank(method="first", ascending=False).astype(int)

    # Keep top-N formulas per file
    top = grouped[grouped["rank"] <= top_n]

    # Merge back into original to collect all tool-specific rows
    merged = df.merge(
        top.rename(columns={"rank": "agg_rank"})[["filename", "formula", "agg_rank", "confidence_score_sum"]],
        on=["filename", "formula"],
        how="inner"
    )

    # Final summary table
    summary = merged.groupby(["filename", "formula", "agg_rank"]).agg(
        adduct=("adduct", "first"),
        confidence_score_sum=("confidence_score_sum", "first"),
        Used_Tool=("Used_Tool", lambda x: ", ".join(sorted(set(x))))
    ).reset_index().rename(columns={"agg_rank": "rank"})

    return summary

