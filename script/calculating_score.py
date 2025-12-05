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
        "Score_NZ_buddy",
        "Score_NZ_msfinder",
        "Score_NZ_sirius",
        "Score_NZ_diff_buddy",
        "Score_NZ_diff_msfinder",
        "Score_NZ_diff_sirius",
        'normalized_rank_buddy',
        'normalized_rank_msfinder',
        'normalized_rank_sirius',
        'adduct_MplusHplus',
        'adduct_MplusNaplus',
        'adduct_MplusNH4plus',
        'adduct_MminusHminus',
        'adduct_MplusClminus',
        'adduct_MplusFAminusHminus'
        
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
    model_all_path = os.path.join(machine_dir, "xgboost_final_all.pkl")
    model_all = joblib.load(model_all_path)

    # Preload all available adduct-specific models (excluding the 'all' model) into a dictionary.
    model_dict = {}
    for filename in os.listdir(machine_dir):
        if "xgboost_" in filename and filename != "xgboost_final_all.pkl":
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

def aggregate_probability_with_rank(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Aggregate confidence scores per (filename, formula), rank them, and report tools that found each formula.

    Parameters:
        df (pd.DataFrame): Input data. Must include:
            - 'filename', 'formula', 'confidence_score', 'adduct'
            - 'rank' (tool-specific rank)
            - 'tool_name_buddy', 'tool_name_msfinder', 'tool_name_sirius'
        top_n (int): Number of top formulas to retain per filename.

    Returns:
        pd.DataFrame: Summary with confidence_score, rank, and Used_Tool per formula.
    """

    df["rank"] = df.groupby("filename")["confidence_score"] \
                             .rank(method="first", ascending=False).astype(int)
    df = df.sort_values(["filename", "rank"], ascending=[True, True])

    # Keep top-N formulas per file
    top = df[df["rank"] <= top_n].copy()
    top = top.rename(columns={"rank": "agg_rank",
                              "confidence_score": "agg_confidence_score"})

    # Merge back into original to collect all tool-specific rows
    merged = df.merge(
        top[["filename", "formula", "agg_rank", "agg_confidence_score"]],
        on=["filename", "formula"],
        how="inner",
    )

    # Final summary table
    summary = merged.groupby(["filename", "formula", "agg_rank"]).agg(
        adduct=("adduct", "first"),
        confidence_score_sum=("agg_confidence_score", "first"),
        Used_Tool=("Used_tools", lambda x: ','.join(sorted(set(','.join(x).split(',')))))
    ).reset_index().rename(columns={"agg_rank": "rank"})

    return summary

def formula_machine_input(df):
    for tool in ['msfinder', 'sirius', 'msbuddy']:
        tool_mask = df['tool_name'] == tool
        df.loc[tool_mask, f'normalization_{tool}_score'] = df.loc[tool_mask, "Score_NZ"]
        df.loc[tool_mask, f"normalization_{tool}_diff"] = df.loc[tool_mask, "Score_NZ_diff"]
        df.loc[tool_mask, f"normalization_{tool}_rank"] = df.loc[tool_mask, "normalized_rank"]

    base_columns = ['filename', 'adduct', 'formula']
    score_cols = ["Score_NZ", "Score_NZ_diff", "normalized_rank"]
    long_df = df[base_columns + ['tool_name'] + score_cols].copy()
    wide_df = long_df.pivot_table(
        index = base_columns,
        columns = 'tool_name',
        values = score_cols,
        aggfunc='max'
        )
    
    # flatten MultiIndex columns
    wide_df.columns = [f'{score}_{tool}' for score, tool in wide_df.columns]
    wide_df = wide_df.reset_index()
    wide_df = wide_df.fillna(0)

    used_tools_df = (
        df.groupby(base_columns)['Used_tools']
        .apply(lambda x: ','.join(sorted(', '.join(x).split(','))))
        .reset_index()
    )
    wide_df = wide_df.merge(used_tools_df, on=base_columns, how='left')

    return wide_df