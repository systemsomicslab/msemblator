import os
import glob
import pandas as pd
import joblib
from functools import reduce
from convert_struc_data_type import normalize_rank_n
from struc_score_normalization import ClippingTransformer

def process_metfrag_output(metfrag_folder, machine_dir, name_adduct_df, 
                          summary_inchikey_df, summary_smiles_df, 
                          class_summary_df, smiles_score_df, top_n=5):
    """
    Processes MetFrag output files, normalizes scores, extracts top candidates, 
    and merges data with existing datasets.

    Parameters:
        metfrag_folder (str): Directory containing MetFrag output files.
        machine_dir (str): Directory containing score normalization pipelines.
        name_adduct_df (pd.DataFrame): DataFrame mapping filenames to adducts.
        summary_inchikey_df (pd.DataFrame): Existing InChIKey summary DataFrame.
        summary_smiles_df (pd.DataFrame): Existing SMILES summary DataFrame.
        class_summary_df (pd.DataFrame): Existing classification summary DataFrame.

    Returns:
        tuple: (metfrag_inchikey_df, metfrag_smiles_df, class_summary_df, metfrag_score_calc_df)
    """

    # Retrieve MetFrag output files
    file_paths = glob.glob(f'{metfrag_folder}/*.xls')

    if not file_paths:
        print(f"No MetFrag files found in {metfrag_folder}")
        return summary_inchikey_df, summary_smiles_df, class_summary_df, smiles_score_df

    data_frames = []
    for file in file_paths:
        try:
            df = pd.read_excel(file, engine='xlrd')
            df["filename"] = os.path.basename(file).split(".")[0]  # Extract filename without extension
            data_frames.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not data_frames:
        return summary_inchikey_df, summary_smiles_df, class_summary_df, smiles_score_df

    # Combine all files into a single DataFrame
    combined_data = pd.concat(data_frames, ignore_index=True)

    # Assign rank
    combined_data["rank"] = combined_data.groupby("filename").cumcount() + 1

    # Determine score column
    score_column = "Score" if "Score" in combined_data.columns else "Total score"

    # Compute score difference
    combined_data["Score_Difference"] = 0
    mask = (combined_data["rank"] + 1 == combined_data["rank"].shift(-1).fillna(0).astype(int))
    combined_data.loc[mask, "Score_Difference"] = (
        combined_data[score_column] - combined_data[score_column].shift(-1)
    )
    combined_data["Score_Difference"] = combined_data["Score_Difference"].fillna(0)

    # Select top 3 ranked candidates
    filtered_df = combined_data.groupby('filename').head(top_n).copy()
    filtered_df = filtered_df.fillna('')

    # Load score normalization pipelines
    metfrag_score_pipeline_path = os.path.join(machine_dir, "pipeline_metfrag_score.pkl")
    metfrag_SD_pipeline_path = os.path.join(machine_dir, "pipeline_metfrag_score_diff.pkl")

    score_pipeline = joblib.load(metfrag_score_pipeline_path)
    SD_pipeline = joblib.load(metfrag_SD_pipeline_path)

    # Normalize scores
    filtered_df["normalization_Zscore"] = score_pipeline.transform(filtered_df[[score_column]])
    filtered_df["normalization_z_score_diff"] = SD_pipeline.transform(filtered_df[["Score_Difference"]])

    # Prepare score calculation DataFrame
    metfrag_score_calc_df = filtered_df[["filename", "rank", "SMILES", "normalization_Zscore", "normalization_z_score_diff"]].copy()
    metfrag_score_calc_df["tool_name"] = "metfrag"
    metfrag_score_calc_df["Used_tools"] = metfrag_score_calc_df["rank"].apply(lambda r: f"MetFrag_Rank:{r}")

    # Apply rank normalization function
    normalize_rank_n(metfrag_score_calc_df)

    # Merge with adduct information
    metfrag_score_calc_df = metfrag_score_calc_df.merge(name_adduct_df, on="filename", how="outer")

    # Convert filenames
    filtered_df["filename"] = filtered_df["filename"].apply(lambda x: x.split('.')[0] if isinstance(x, str) else None)
    filtered_df = filtered_df.astype(str).fillna('')

    # Pivot InChIKey and SMILES data
    inchikey_pivot = filtered_df.pivot(index=["filename"], columns=["rank"], values=["InChIKey"])
    smiles_pivot = filtered_df.pivot(index=["filename"], columns=["rank"], values=["SMILES"])

    inchikey_pivot.columns = [f'metfrag_structure_{col[1]}' for col in inchikey_pivot.columns.values]
    smiles_pivot.columns = [f'metfrag_structure_{col[1]}' for col in smiles_pivot.columns.values]

    # Merge InChIKey data
    metfrag_inchikey_df = summary_inchikey_df.merge(inchikey_pivot.reset_index(), on=["filename"], how="outer")

    # Merge SMILES data
    metfrag_smiles_df = summary_smiles_df.merge(smiles_pivot.reset_index(), on=["filename"], how="outer")

    # Extract classification data (only rank 1)
    metfrag_class_data = filtered_df[filtered_df['rank'] == '1'][['filename', 'InChIKey', 'SMILES']]
    metfrag_class_data = metfrag_class_data[(metfrag_class_data['InChIKey'].str.strip() != '') & 
                                            (metfrag_class_data['SMILES'].str.strip() != '')]
    metfrag_class_data['tool_name'] = "MetFrag"
    metfrag_class_data.columns = ['filename', 'InChIKey', 'SMILES', 'tool_name']

    # Append new classification data
    class_summary_df = pd.concat([class_summary_df, metfrag_class_data], ignore_index=True)

    smiles_score_df = pd.concat([smiles_score_df, metfrag_score_calc_df], ignore_index=True)

    return metfrag_inchikey_df, metfrag_smiles_df, class_summary_df, smiles_score_df