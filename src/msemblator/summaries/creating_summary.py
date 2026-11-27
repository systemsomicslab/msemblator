import pandas as pd
import glob
import os
import joblib
from functools import reduce
from msemblator.chemistry.converting_data_type import normalize_rank, ClippingTransformer, read_msp_file, extract_compound_and_ionization
from msemblator.summaries.msfinder_summary import process_msfinder_summary
from msemblator.summaries.sirius_summary import process_sirius_summary
from msemblator.summaries.msbuddy_summary import process_buddy_summary
from msemblator.summaries.fiddle_summary import process_fiddle_summary
from msemblator.scoring.calculating_score import predict_and_append, aggregate_probability_with_rank, formula_machine_input

# Tools of the original workflow; used when no tool selection is given.
LEGACY_TOOLS = ("msfinder", "sirius", "msbuddy")
TOOL_LABELS = {"msfinder": "MS-FINDER", "sirius": "SIRIUS", "msbuddy": "msbuddy", "fiddle": "FIDDLE"}


def creating_output_summary(input_msp, sirius_folder, msfinder_file_path, buddy_folder, machine_dir, top_n, summary_n,
                            fiddle_folder=None, selection=None):
    """Formula candidates of the selected tools -> ranking -> score and summary tables.

    selection is a toolsets.ToolSelection. Without it, SIRIUS, MS-FINDER and
    msbuddy are summarized and every model file is read from machine_dir.
    """
    if selection is None:
        tools = LEGACY_TOOLS
        pipeline_dir = lambda tool: machine_dir  # noqa: E731
        ranker_dir = machine_dir
    else:
        tools = selection.tools
        pipeline_dir = selection.pipeline_dir
        ranker_dir = str(selection.model_dir)

    msp_data = read_msp_file(input_msp)
    compound_ionization_data = extract_compound_and_ionization(msp_data)
    summary_df = pd.DataFrame(columns=['filename', 'adduct'])
    name_adduct_df = pd.DataFrame(columns=['filename', 'adduct'])
    score_df=pd.DataFrame(columns=['filename',"tool_name",'adduct',"rank","formula","Score_NZ","Score_NZ_diff","normalized_rank"])
    # Assign the compound names to the 'filename' column and ionization information to the 'adduct' column
    for idx, (compound, ionization) in enumerate(compound_ionization_data):
        summary_df.at[idx, 'filename'] = compound
        summary_df.at[idx, 'adduct'] = ionization
        name_adduct_df.at[idx, 'filename'] = compound
        name_adduct_df.at[idx, 'adduct'] = ionization

    # Per-tool top formulas, in the order MS-FINDER, SIRIUS, msbuddy, FIDDLE.
    tool_formula_dfs = []
    if "msfinder" in tools:
        msfinder_formula_df, score_df = process_msfinder_summary(msfinder_file_path, pipeline_dir("msfinder"), name_adduct_df, summary_df, score_df, top_n)
        tool_formula_dfs.append(msfinder_formula_df)
    if "sirius" in tools:
        sirius_formula_df, score_df = process_sirius_summary(sirius_folder, pipeline_dir("sirius"), name_adduct_df, summary_df, score_df, top_n)
        tool_formula_dfs.append(sirius_formula_df)
    if "msbuddy" in tools:
        buddy_formula_df, score_df = process_buddy_summary(buddy_folder, pipeline_dir("msbuddy"), name_adduct_df, summary_df, score_df, top_n)
        tool_formula_dfs.append(buddy_formula_df)
    if "fiddle" in tools:
        fiddle_formula_df, score_df = process_fiddle_summary(fiddle_folder, pipeline_dir("fiddle"), name_adduct_df, summary_df, score_df, top_n)
        tool_formula_dfs.append(fiddle_formula_df)

    # A tool without output contributes no rows; its ranking features are filled with 0.
    if score_df.empty:
        labels = ", ".join(TOOL_LABELS[t] for t in tools)
        raise RuntimeError(f"No formula candidates from any tool ({labels}).")
    wide_df = formula_machine_input(score_df)
    if selection is None:
        calc_score_df = predict_and_append(wide_df, ranker_dir)
    else:
        calc_score_df = predict_and_append(wide_df, ranker_dir, selection)
    # Fix rank_score data type
    def fix_conf(x):
        if hasattr(x, "__array__"):
            return float(x[0])

        if isinstance(x, (list, tuple)):
            return float(x[0])

        try:
            return float(x)
        except:
            return 0.0

    calc_score_df['rank_score'] = calc_score_df['rank_score'].apply(fix_conf)

    summary_score_df = aggregate_probability_with_rank(calc_score_df, top_n)
    summary_score_df = summary_score_df[summary_score_df["rank"] <= summary_n]
    score_sort_cols = ['filename', 'rank']
    if 'adduct' in summary_score_df.columns:
        score_sort_cols = ['filename', 'adduct', 'rank']
    summary_score_df = summary_score_df.sort_values(score_sort_cols, ascending=[True] * len(score_sort_cols))

    summary_score_top_df = summary_score_df[summary_score_df["rank"]==1].copy()
    summary_output = reduce(lambda left, right: pd.merge(left, right, on=['filename', 'adduct'], how='inner'),
                            tool_formula_dfs + [summary_score_top_df])

    return summary_score_df, summary_output
