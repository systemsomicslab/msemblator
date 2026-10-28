import numpy as np
import os
import joblib
import pandas as pd
from msemblator.paths import find_ranker_model
from msemblator.scoring.struc_score_normalization import ClippingTransformer

def predict_and_append(df, machine_dir):
    feature_columns = [
        "normalization_score_msbuddy","normalization_score_msfinder","normalization_score_sirius",
        "normalization_score_diff_msbuddy","normalization_score_diff_msfinder","normalization_score_diff_sirius",
        "normalization_rank_msbuddy","normalization_rank_msfinder","normalization_rank_sirius",
        "adduct_MplusHplus", "adduct_MplusNaplus", "adduct_MplusNH4plus",
        "adduct_MminusHminus", "adduct_MplusClminus", "adduct_MplusFAminusHminus"
    ]

    df_original = df.copy()
    X = df.reindex(columns=feature_columns, fill_value=0).fillna(0)

    model_path = find_ranker_model(machine_dir)
    print(f"Formula ranking model: {model_path.name}")
    model = joblib.load(model_path)

    df_original["rank_score"] = model.predict(X)
    return df_original

def _tool_support(used_tool):
    """Return (number of tools, best tool rank) from a "TOOL_Rank:n,..." string."""
    tools = set()
    best_rank = float("inf")
    for token in str(used_tool).split(","):
        name, separator, rank = token.strip().rpartition("_Rank:")
        if not separator:
            continue
        tools.add(name)
        try:
            best_rank = min(best_rank, float(rank))
        except ValueError:
            pass
    return len(tools), best_rank


def sort_by_score_and_tool_support(grouped, group_cols):
    """Sort candidates by rank_score; break ties by tool agreement, then best tool rank.

    Without a tie-breaker, equal scores (e.g. a constant model output) keep the
    alphabetical order left by groupby, which puts "B..." formulas first.
    """
    sort_cols = group_cols + ["rank_score"]
    ascending = [True] * len(group_cols) + [False]
    if "Used_Tool" in grouped.columns:
        support = grouped["Used_Tool"].apply(_tool_support)
        grouped = grouped.assign(
            _n_tools=support.str[0],
            _best_tool_rank=support.str[1],
        )
        sort_cols += ["_n_tools", "_best_tool_rank"]
        ascending += [False, True]
    return (
        grouped.sort_values(sort_cols, ascending=ascending, kind="stable")
        .drop(columns=["_n_tools", "_best_tool_rank"], errors="ignore")
        .reset_index(drop=True)
    )


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

    grouped = sort_by_score_and_tool_support(grouped, group_cols)

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
    df = df.copy()

    df["adduct"] = (
        df["adduct"]
        .fillna("")
        .astype(str)
        .replace({"[M+CO2]-": "[M+FA-H]-"})
    )

    adduct_list = [
        '[M+H]+', '[M+Na]+', '[M+NH4]+',
        '[M-H]-', '[M+Cl]-', '[M+FA-H]-'
    ]

    base_columns = ['filename', 'adduct', 'formula']
    score_cols = ["Score_NZ", "Score_NZ_diff", "normalized_rank"]

    long_df = df[
        base_columns + ['tool_name'] + score_cols
    ].copy()

    wide_df = long_df.pivot_table(
        index=base_columns,
        columns='tool_name',
        values=score_cols,
        aggfunc='max'
    )

    # カラム名の対応
    score_name_map = {
        "Score_NZ": "normalization_score",
        "Score_NZ_diff": "normalization_score_diff",
        "normalized_rank": "normalization_rank",
    }

    # flatten MultiIndex columns
    wide_df.columns = [
        f'{score_name_map[score]}_{tool}'
        for score, tool in wide_df.columns
    ]

    wide_df = wide_df.reset_index()
    wide_df = wide_df.fillna(0)

    used_tools_df = (
        df.groupby(base_columns)['Used_tools']
        .apply(lambda x: ','.join(sorted(', '.join(x).split(','))))
        .reset_index()
    )

    wide_df = wide_df.merge(
        used_tools_df,
        on=base_columns,
        how='left'
    )

    for ad in adduct_list:
        ad_norm = (
            ad.replace("+", "plus")
              .replace("-", "minus")
              .replace("[", "")
              .replace("]", "")
        )
        col_name = f"adduct_{ad_norm}"

        wide_df[col_name] = (
            wide_df['adduct']
            .str.replace("+", "plus", regex=False)
            .str.replace("-", "minus", regex=False)
            .str.replace("[", "", regex=False)
            .str.replace("]", "", regex=False)
            == ad_norm
        ).astype(int)

    return wide_df
