import numpy as np
import os
import joblib
import pandas as pd
from catboost import CatBoostRanker
from msemblator.scoring.struc_score_normalization import ClippingTransformer

def predict_and_append(df, machine_dir, model_type="catboost_ranker"):
    feature_columns = [
        "Score_NZ_msbuddy","Score_NZ_msfinder","Score_NZ_sirius",
        "Score_NZ_diff_msbuddy","Score_NZ_diff_msfinder","Score_NZ_diff_sirius",
        "normalized_rank_msbuddy","normalized_rank_msfinder","normalized_rank_sirius",
        "adduct_MplusHplus", "adduct_MplusNaplus", "adduct_MplusNH4plus",
        "adduct_MminusHminus", "adduct_MplusClminus", "adduct_MplusFAminusHminus"
    ]

    df_original = df.copy()
    X = df.reindex(columns=feature_columns, fill_value=0).fillna(0)

    if model_type == "catboost_ranker":
        model_path = os.path.join(machine_dir, "catboostranker_final_all.pkl")
        model = CatBoostRanker()
        model = joblib.load(model_path)
    elif model_type == "lgbm_ranker":
        model_path = os.path.join(machine_dir, "lgbmranker_final_all.pkl")
        model = joblib.load(model_path)
    else:
        model_path = os.path.join(machine_dir, "xgbranker_final_all.pkl")
        model = joblib.load(model_path)

    df_original["rank_score"] = model.predict(X)
    return df_original

def aggregate_probability_with_rank(
    df: pd.DataFrame,
    top_n: int = 3,
    softmax_top_k: int = 10,
    group_col: str = "filename",
    subgroup_col: str = "adduct",
    smiles_col: str = "formula",
    score_col: str = "rank_score",
) -> pd.DataFrame:

    required_cols = [group_col, smiles_col, score_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Existing columns: {df.columns.tolist()}")

    group_cols = [group_col]
    if subgroup_col in df.columns:
        group_cols.append(subgroup_col)

    agg_kwargs = {"rank_score": (score_col, "sum")}
    if "Used_tools" in df.columns:
        agg_kwargs["Used_Tool"] = (
            "Used_tools",
            lambda values: ",".join(
                sorted(
                    {
                        token.strip()
                        for value in values.dropna().astype(str)
                        for token in value.split(",")
                        if token.strip()
                    }
                )
            ),
        )

    grouped = (
        df.groupby(group_cols + [smiles_col], as_index=False, dropna=False)
          .agg(**agg_kwargs)
    )

    grouped = grouped.sort_values(
        group_cols + ["rank_score"],
        ascending=[True] * len(group_cols) + [False]
    ).reset_index(drop=True)

    grouped["rank"] = grouped.groupby(group_cols, dropna=False).cumcount() + 1

    def softmax_topk(s: pd.Series) -> pd.Series:
        scores = s.to_numpy(dtype=float)
        out = np.zeros_like(scores, dtype=float)

        k = min(softmax_top_k, len(scores))
        if k > 0:
            top_scores = scores[:k]
            shifted = top_scores - np.max(top_scores)
            exp_s = np.exp(shifted)
            out[:k] = exp_s / (exp_s.sum() + 1e-12)

        return pd.Series(out, index=s.index)

    grouped["softmax_score"] = (
        grouped.groupby(group_cols, dropna=False)["rank_score"]
               .transform(softmax_topk)
    )

    grouped["next_rank_score"] = (
        grouped.groupby(group_cols, dropna=False)["rank_score"]
               .shift(-1)
    )

    grouped["margin_score"] = grouped["rank_score"] - grouped["next_rank_score"]
    grouped = grouped.drop(columns=["next_rank_score"])

    return (
        grouped[grouped["rank"] <= top_n]
        .sort_values(group_cols + ["rank"])
        .reset_index(drop=True)
    )

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
