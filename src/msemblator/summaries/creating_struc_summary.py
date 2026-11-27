import os
import glob
import re
import joblib
import pandas as pd
from msemblator.chemistry.convert_struc_data_type import read_msp_file,extract_compound_and_ionization,convert_to_canonical_smiles
from tqdm import tqdm
from functools import reduce
from msemblator.scoring.struc_score_normalization import ClippingTransformer
from msemblator.summaries.msfinder_struc_summary import process_msfinder_output
from msemblator.summaries.sirius_struc_summary import process_sirius_output
from msemblator.scoring.struc_score_calc import predict_and_append_rank, aggregate_probability_with_rank, machine_input_generation
from msemblator.summaries.metfrag_summary import process_metfrag_output
from msemblator.summaries.glacier_struc_summary import add_glacier_features, glacier_summary_smiles

# Tools of the original workflow; used when no tool selection is given.
LEGACY_TOOLS = ("msfinder", "sirius", "metfrag")
TOOL_LABELS = {"msfinder": "MS-FINDER", "sirius": "SIRIUS", "metfrag": "MetFrag", "glacier": "GLACIER"}


def struc_summary(input_msp, msfinder_folder, machine_dir, sirius_folder, metfrag_folder, top_n = 100, summary_n = 5,
                  glacier_features=None, glacier_rank_column=None, selection=None):
    """Structure candidates of the selected tools -> ranking -> score and summary tables.

    selection is a toolsets.ToolSelection. Without it, MS-FINDER, SIRIUS and
    MetFrag are summarized and every model file is read from machine_dir.
    glacier_features / glacier_rank_column come from run_glacier_structure.
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
    summary_inchikey_df = pd.DataFrame(columns=['filename', 'adduct'])
    summary_smiles_df = pd.DataFrame(columns=['filename', 'adduct'])
    class_summary_df = pd.DataFrame(columns=['filename','tool_name','InChIKey','SMILES'])
    smiles_score_df=pd.DataFrame(columns=['filename',"tool_name",'adduct',"rank","SMILES","normalization_Zscore","normalization_z_score_diff","normalized_rank"])
    name_adduct_df = pd.DataFrame(columns=['filename', 'adduct'])
    # Assign the compound names to the 'filename' column and ionization information to the 'adduct' column
    for idx, (compound, ionization) in enumerate(compound_ionization_data):
        summary_inchikey_df.at[idx, 'filename'] = compound
        summary_inchikey_df.at[idx, 'adduct'] = ionization
        summary_smiles_df.at[idx, 'filename'] = compound
        summary_smiles_df.at[idx, 'adduct'] = ionization
        name_adduct_df.at[idx, 'filename'] = compound
        name_adduct_df.at[idx, 'adduct'] = ionization

    # Per-tool top structures, in the order MS-FINDER, SIRIUS, MetFrag, GLACIER.
    tool_smiles_dfs = []
    if "msfinder" in tools:
        msfinder_inchikey_df, msfinder_smiles_df, class_summary_df, smiles_score_df = process_msfinder_output(msfinder_folder, pipeline_dir("msfinder"), name_adduct_df, summary_inchikey_df, summary_smiles_df, class_summary_df, smiles_score_df,top_n)
        tool_smiles_dfs.append(msfinder_smiles_df)
    if "sirius" in tools:
        sirius_inchikey_df, sirius_smiles_df, class_summary_df, smiles_score_df = process_sirius_output(sirius_folder, pipeline_dir("sirius"), name_adduct_df, summary_inchikey_df, summary_smiles_df, class_summary_df, smiles_score_df,top_n)
        tool_smiles_dfs.append(sirius_smiles_df)
    if "metfrag" in tools:
        metfrag_inchikey_df, metfrag_smiles_df, class_summary_df, smiles_score_df = process_metfrag_output(metfrag_folder, pipeline_dir("metfrag"), name_adduct_df, summary_inchikey_df, summary_smiles_df, class_summary_df,smiles_score_df,top_n)
        tool_smiles_dfs.append(metfrag_smiles_df)
    use_glacier = "glacier" in tools and glacier_features is not None and not glacier_features.empty
    if use_glacier:
        tool_smiles_dfs.append(glacier_summary_smiles(glacier_features, glacier_rank_column, summary_smiles_df))

    print("Generating structural scoring input...")
    # A tool without output contributes no rows; its ranking features are filled with 0.
    has_tool_candidates = not smiles_score_df.dropna(subset=["SMILES"]).empty
    if not has_tool_candidates and not use_glacier:
        labels = ", ".join(TOOL_LABELS[t] for t in tools)
        raise RuntimeError(f"No structure candidates from any tool ({labels}).")

    if has_tool_candidates:
        score_df = machine_input_generation(smiles_score_df)
    else:
        score_df = pd.DataFrame(columns=["filename", "adduct", "Short_InChIKey", "SMILES", "Used_tools"])
    if use_glacier:
        score_df = add_glacier_features(score_df, glacier_features, glacier_rank_column, name_adduct_df)
    if selection is None:
        calced_score_df = predict_and_append_rank(score_df, ranker_dir)
    else:
        calced_score_df = predict_and_append_rank(score_df, ranker_dir, selection)
    print("Structural scoring complete. Generating summary... finishing")

    convert_to_canonical_smiles(calced_score_df, 'SMILES')
    result_score_df = aggregate_probability_with_rank(calced_score_df, summary_n)
    score_sort_cols = ['filename', 'rank']
    if 'adduct' in result_score_df.columns:
        score_sort_cols = ['filename', 'adduct', 'rank']
    result_score_df = result_score_df.sort_values(score_sort_cols, ascending=[True] * len(score_sort_cols))

    # smiles output summary
    summary_smiles_df = reduce(lambda left, right: pd.merge(left, right, on=["filename", "adduct"], how='outer'), tool_smiles_dfs)
    result_score_top_df = result_score_df[result_score_df["rank"]==1].copy()
    summary_output_score = pd.merge(summary_smiles_df, result_score_top_df, on=['filename', 'adduct'], how='inner')

    return result_score_df, summary_output_score
