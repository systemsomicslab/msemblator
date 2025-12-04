import os
import glob
import pandas as pd
import joblib
from convert_struc_data_type import normalize_rank_score
from struc_score_normalization import ClippingTransformer

def process_msfinder_output(msfinder_folder, machine_dir, name_adduct_df, 
                            summary_inchikey_df, summary_smiles_df, 
                            class_summary_df, smiles_score_df, top_n=3):
    """
    Processes MS-FINDER output and generates updated InChIKey, SMILES, score, and classification data.

    Parameters:
        msfinder_folder (str): Directory containing MS-FINDER output files.
        machine_dir (str): Directory containing score normalization pipelines.
        name_adduct_df (pd.DataFrame): DataFrame mapping filenames to adducts.
        summary_inchikey_df (pd.DataFrame): Existing InChIKey summary DataFrame.
        summary_smiles_df (pd.DataFrame): Existing SMILES summary DataFrame.
        class_summary_df (pd.DataFrame): Existing classification summary DataFrame.
        smiles_score_df (pd.DataFrame): Existing score summary DataFrame.

    Returns:
        tuple: (msfinder_inchikey_df, msfinder_smiles_df, class_summary_df, smiles_score_df)
    """

    # Retrieve MS-FINDER output files
    file_paths = glob.glob(os.path.join(msfinder_folder, "Structure result*.txt"))
    if not file_paths:
        # Return the original DataFrames unchanged if no files are found
        return summary_inchikey_df, summary_smiles_df, class_summary_df, smiles_score_df
    msfinder_output_combined = pd.concat([pd.read_table(file_path) for file_path in file_paths], ignore_index=True)

    # Extract filename without extension
    msfinder_output_combined["filename"] = msfinder_output_combined["File name"].astype(str).apply(lambda x: x.split('.')[0])
    
    # Assign rank based on filename groups
    msfinder_output_combined["rank"] = (msfinder_output_combined.groupby("filename").cumcount() + 1).astype(int)

    # Determine the appropriate score column
    score_column = "Score" if "Score" in msfinder_output_combined.columns else "Total score"
    
    # Compute score difference
    msfinder_output_combined["score_diff"] = 0 
    next_rank = msfinder_output_combined["rank"].shift(-1)
    mask = (msfinder_output_combined["rank"] + 1 == next_rank)
    msfinder_output_combined.loc[mask, "score_diff"] = (
        msfinder_output_combined[score_column] - msfinder_output_combined[score_column].shift(-1)
    )
    msfinder_output_combined["score_diff"] = msfinder_output_combined["score_diff"].fillna(0)

    # Select top 3 ranked candidates per filename
    filtered_df = msfinder_output_combined.groupby('filename').head(top_n).copy()
    filtered_df = filtered_df.fillna('')
    filtered_df.rename(columns={"Precursor type": "adduct"}, inplace=True)

    # Load score normalization pipelines
    msfinder_score_pipeline_path = os.path.join(machine_dir, "pipeline_msfinder_score.pkl")
    msfinder_SD_pipeline_path = os.path.join(machine_dir, "pipeline_msfinder_score_diff.pkl")

    score_pipeline = joblib.load(msfinder_score_pipeline_path)
    SD_pipeline = joblib.load(msfinder_SD_pipeline_path)

    # Normalize scores
    filtered_df["normalization_Zscore"] = score_pipeline.transform(filtered_df[[score_column]])
    filtered_df["normalization_z_score_diff"] = SD_pipeline.transform(filtered_df[["score_diff"]])

    # Prepare score calculation DataFrame
    msfinder_score_calc_df = filtered_df[["filename", "adduct", "rank", "SMILES", "normalization_Zscore", "normalization_z_score_diff"]].copy()
    msfinder_score_calc_df["tool_name"] = "msfinder"
    msfinder_score_calc_df["Used_tools"] = msfinder_score_calc_df["rank"].apply(lambda r: f"MS-FINDER_Rank:{r}")

    # Apply rank normalization function (assumes `normalize_rank` is defined)
    normalize_rank_score(msfinder_score_calc_df)

    # Map adducts from `name_adduct_df`
    msfinder_score_calc_df['adduct'] = msfinder_score_calc_df['filename'].map(name_adduct_df.set_index('filename')['adduct'])

    # Pivot InChIKey and SMILES data
    filtered_df['rank'] = filtered_df['rank'].astype(int)
    top5_smiles_df = filtered_df[filtered_df['rank'] <= 5]
    filtered_df = filtered_df.astype(str)

    inchikey_pivot = filtered_df.pivot(index=["filename", "adduct"], columns=["rank"], values=["InChIKey"])
    smiles_pivot = top5_smiles_df.pivot(index=["filename", "adduct"], columns=["rank"], values=["SMILES"])

    inchikey_pivot.columns = [f'msfinder_structure_{col[1]}' for col in inchikey_pivot.columns.values]
    smiles_pivot.columns = [f'msfinder_structure_{col[1]}' for col in smiles_pivot.columns.values]

    # Merge InChIKey data
    msfinder_inchikey_df = summary_inchikey_df.merge(inchikey_pivot.reset_index(), on=["filename", "adduct"], how="outer")

    # Merge SMILES data
    msfinder_smiles_df = summary_smiles_df.merge(smiles_pivot.reset_index(), on=["filename", "adduct"], how="outer")

    # Extract classification data (only rank 1)
    msfinder_class_data = filtered_df[filtered_df['rank'] == '1'][['filename', 'InChIKey', 'SMILES']]
    msfinder_class_data = msfinder_class_data[(msfinder_class_data['InChIKey'].str.strip() != '') & 
                                              (msfinder_class_data['SMILES'].str.strip() != '')]
    msfinder_class_data['tool_name'] = "MS-FINDER"
    msfinder_class_data.columns = ['filename', 'InChIKey', 'SMILES', 'tool_name']
    class_summary_df = pd.concat([class_summary_df, msfinder_class_data], ignore_index=True)

    # Append new MS-FINDER scores to the existing DataFrame
    smiles_score_df = pd.concat([smiles_score_df, msfinder_score_calc_df], ignore_index=True)

    return msfinder_inchikey_df, msfinder_smiles_df, class_summary_df, smiles_score_df
