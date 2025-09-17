import os
import glob
import re
import joblib
import pandas as pd
from convert_struc_data_type import read_msp_file,extract_compound_and_ionization,convert_to_canonical_smiles,normalize_rank
from tqdm import tqdm
from functools import reduce
from struc_score_normalization import ClippingTransformer 
from msfinder_struc_summary import process_msfinder_output
from sirius_struc_summary import process_sirius_output
from struc_score_calc import predict_and_append, aggregate_probability_with_rank
from metfrag_summary import process_metfrag_output

def struc_summary(input_msp, msfinder_folder, machine_dir, sirius_folder, metfrag_folder):
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
    # msfinder summary
    msfinder_inchikey_df, msfinder_smiles_df, class_summary_df, smiles_score_df = process_msfinder_output(msfinder_folder, machine_dir, name_adduct_df, summary_inchikey_df, summary_smiles_df, class_summary_df, smiles_score_df)
    # sirius summary
    sirius_inchikey_df, sirius_smiles_df, class_summary_df, smiles_score_df = process_sirius_output(sirius_folder, machine_dir, name_adduct_df, summary_inchikey_df, summary_smiles_df, class_summary_df, smiles_score_df)
    # metfrag summary
    metfrag_inchikey_df, metfrag_smiles_df, class_summary_df, metfrag_score_calc_df = process_metfrag_output(metfrag_folder, machine_dir, name_adduct_df, summary_inchikey_df, summary_smiles_df, class_summary_df)
    # Merge summary data across all tools
    dataframes = [msfinder_smiles_df, sirius_smiles_df, metfrag_smiles_df]
    summary_smiles_df = reduce(lambda left, right: pd.merge(left, right, on=["filename", "adduct"], how='outer'), dataframes)

    inchikey_dataframes = [msfinder_inchikey_df, sirius_inchikey_df, metfrag_inchikey_df]
    summary_inchikey_df = reduce(lambda left, right: pd.merge(left, right, on=["filename", "adduct"], how='outer'), inchikey_dataframes)
    # Append MetFrag score data
    smiles_score_df = pd.concat([smiles_score_df, metfrag_score_calc_df], ignore_index=True)

    smiles_score_df["tool_name"] = (
        smiles_score_df["tool_name"]
        .astype("string")
        .fillna("")        
        .str.strip().str.lower()
    )
    df_onehot = pd.get_dummies(smiles_score_df, columns=['tool_name'])
    required_col = ['tool_name_metfrag', 'tool_name_msfinder', 'tool_name_sirius']
    for col in required_col:
        if col not in df_onehot.columns:
            df_onehot[col] = 0
    oh_cols = [c for c in df_onehot.columns if c.startswith("tool_name_")]
    df_onehot[oh_cols] = df_onehot[oh_cols].astype(int)
    df_onehot = df_onehot[['filename', 'adduct', 'rank', 'SMILES', 'normalization_Zscore', 'normalization_z_score_diff', 'normalized_rank'] + required_col]
    df_onehot = df_onehot[df_onehot['SMILES'].notna() & (df_onehot['SMILES'].str.strip() != '')].copy()
    
    score_calc_df = predict_and_append(df_onehot, machine_dir, adduct_column="adduct")
    convert_to_canonical_smiles(score_calc_df, 'SMILES')
    result_score_df = aggregate_probability_with_rank(score_calc_df)
    result_score_top_df = result_score_df[result_score_df["rank"]==1]
    summary_output_score = pd.merge(summary_smiles_df, result_score_top_df, on=['filename', 'adduct'], how='inner')
    return result_score_df, summary_output_score