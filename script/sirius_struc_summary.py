import os
import glob
import pandas as pd
import joblib
from convert_struc_data_type import normalize_rank,smiles_list_to_inchikeys
from struc_score_normalization import ClippingTransformer

def process_sirius_output(sirius_folder, machine_dir, name_adduct_df, 
                          summary_inchikey_df, summary_smiles_df, 
                          class_summary_df, smiles_score_df):
    """
    Processes SIRIUS output and generates updated InChIKey, SMILES, score, and classification data.

    Parameters:
        sirius_folder (str): Directory containing SIRIUS output files.
        machine_dir (str): Directory containing score normalization pipelines.
        name_adduct_df (pd.DataFrame): DataFrame mapping filenames to adducts.
        summary_inchikey_df (pd.DataFrame): Existing InChIKey summary DataFrame.
        summary_smiles_df (pd.DataFrame): Existing SMILES summary DataFrame.
        class_summary_df (pd.DataFrame): Existing classification summary DataFrame.
        smiles_score_df (pd.DataFrame): Existing score summary DataFrame.

    Returns:
        tuple: (sirius_inchikey_df, sirius_smiles_df, class_summary_df, smiles_score_df)
    """

    # Retrieve SIRIUS output files
    sirius_paths = glob.glob(f"{sirius_folder}/*/structure_candidates.tsv")

    if not sirius_paths:
        print(f"No SIRIUS files found in {sirius_folder}")
        return summary_inchikey_df, summary_smiles_df, class_summary_df, smiles_score_df

    data_frames = []
    for file in sirius_paths:
        try:
            df = pd.read_csv(file, sep='\t')
            df['filename'] = os.path.basename(os.path.dirname(file)).split('_')[-1]
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
    score_column = "CSI:FingerIDScore" if "CSI:FingerIDScore" in combined_data.columns else "score"

    # Compute score difference
    combined_data["score_diff"] = 0
    mask = (combined_data["rank"] + 1 == combined_data["rank"].shift(-1).fillna(0).astype(int))
    combined_data.loc[mask, "score_diff"] = (
        combined_data[score_column] - combined_data[score_column].shift(-1)
    )
    combined_data["score_diff"] = combined_data["score_diff"].fillna(0)

    # Adduct replacement
    replace_dict = {
        r"\[M \+ H3N \+ H\]\+": "[M+NH4]+",
        r"\[M \+ CH2O2 - H\]-": "[M+FA+H]+"
    }
    for pattern, replacement in replace_dict.items():
        combined_data["adduct"] = combined_data["adduct"].fillna("").str.replace(pattern, replacement, regex=True)

    # Load score normalization pipelines
    sirius_score_pipeline_path = os.path.join(machine_dir, "pipeline_CSI_FingerIDScore.pkl")
    sirius_SD_pipeline_path = os.path.join(machine_dir, "pipeline_sirius_score_diff.pkl")

    score_pipeline = joblib.load(sirius_score_pipeline_path)
    SD_pipeline = joblib.load(sirius_SD_pipeline_path)

    # Select top 3 ranked candidates
    filtered_df = combined_data.groupby('filename').head(3).copy()

    # Normalize scores
    filtered_df["normalization_Zscore"] = score_pipeline.transform(filtered_df[[score_column]])
    filtered_df["normalization_z_score_diff"] = SD_pipeline.transform(filtered_df[["score_diff"]])

    # Prepare score calculation DataFrame
    sirius_score_calc_df = filtered_df[["filename", "adduct", "rank", "smiles", "normalization_Zscore", "normalization_z_score_diff"]].copy()
    sirius_score_calc_df = sirius_score_calc_df.rename(columns={"smiles": "SMILES"})
    sirius_score_calc_df["tool_name"] = "sirius"

    # Map adducts from `name_adduct_df`
    sirius_score_calc_df['adduct'] = sirius_score_calc_df['filename'].map(name_adduct_df.set_index('filename')['adduct'])

    # Apply rank normalization function
    normalize_rank(sirius_score_calc_df)

    # Convert SMILES to InChIKey
    filtered_df["InChIKey"] = smiles_list_to_inchikeys(filtered_df["smiles"])
    filtered_df = filtered_df.astype(str).fillna('')

    # Pivot InChIKey and SMILES data
    inchikey_pivot = filtered_df.pivot(index=["filename"], columns=["rank"], values=["InChIKey"])
    smiles_pivot = filtered_df.pivot(index=["filename"], columns=["rank"], values=["smiles"])

    inchikey_pivot.columns = [f'sirius_structure_{col[1]}' for col in inchikey_pivot.columns.values]
    smiles_pivot.columns = [f'sirius_structure_{col[1]}' for col in smiles_pivot.columns.values]

    # Merge InChIKey data
    sirius_inchikey_df = summary_inchikey_df.merge(inchikey_pivot.reset_index(), on=["filename"], how="outer")

    # Merge SMILES data
    sirius_smiles_df = summary_smiles_df.merge(smiles_pivot.reset_index(), on=["filename"], how="outer")

    # Extract classification data (only rank 1)
    sirius_class_data = filtered_df[filtered_df['rank'] == '1'][['filename', 'InChIKey', 'smiles']]
    sirius_class_data = sirius_class_data[(sirius_class_data['InChIKey'].str.strip() != '') & 
                                          (sirius_class_data['smiles'].str.strip() != '')]
    sirius_class_data['tool_name'] = "SIRIUS"
    sirius_class_data.columns = ['filename', 'InChIKey', 'SMILES', 'tool_name']

    # Append new classification data
    class_summary_df = pd.concat([class_summary_df, sirius_class_data], ignore_index=True)

    # Append new score data
    smiles_score_df = pd.concat([smiles_score_df, sirius_score_calc_df], ignore_index=True)

    return sirius_inchikey_df, sirius_smiles_df, class_summary_df, smiles_score_df
