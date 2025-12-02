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
from struc_score_calc import predict_and_append, aggregate_probability_with_rank, machine_input_generation
from metfrag_summary import process_metfrag_output
from functools import reduce

def struc_summary(input_msp, msfinder_folder, machine_dir, sirius_folder, metfrag_folder,top_n = 100, summary_n = 5):
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
    msfinder_inchikey_df, msfinder_smiles_df, class_summary_df, smiles_score_df = process_msfinder_output(msfinder_folder, machine_dir, name_adduct_df, summary_inchikey_df, summary_smiles_df, class_summary_df, smiles_score_df,top_n)
    
    msfinder_score = smiles_score_df
    msfinder_score = msfinder_score.dropna(subset=["adduct"])
    # sirius summary
    sirius_inchikey_df, sirius_smiles_df, class_summary_df, smiles_score_df = process_sirius_output(sirius_folder, machine_dir, name_adduct_df, summary_inchikey_df, summary_smiles_df, class_summary_df, smiles_score_df,top_n)

    sirius_score = smiles_score_df
    sirius_score = sirius_score.dropna(subset=["adduct"])
    # metfrag summary
    metfrag_inchikey_df, metfrag_smiles_df, class_summary_df, smiles_score_df = process_metfrag_output(metfrag_folder, machine_dir, name_adduct_df, summary_inchikey_df, summary_smiles_df, class_summary_df,smiles_score_df,top_n)
    metfrag_score = smiles_score_df
    metfrag_score = metfrag_score.dropna(subset=["adduct"])

    print("Generating structural scoring input...")

    score_df = machine_input_generation(smiles_score_df)
    calced_score_df = predict_and_append(score_df, machine_dir, adduct_column="adduct")
    convert_to_canonical_smiles(calced_score_df, 'SMILES')
    result_score_df = aggregate_probability_with_rank(calced_score_df, summary_n)

    # smiles output summary
    dfs = [msfinder_smiles_df, sirius_smiles_df, metfrag_smiles_df]
    summary_smiles_df = reduce(lambda left, right: pd.merge(left, right, on=["filename", "adduct"], how='outer'), dfs)
    result_score_top_df = result_score_df[result_score_df["rank"]==1]
    summary_output_score = pd.merge(summary_smiles_df, result_score_top_df, on=['filename', 'adduct'], how='inner')
    
    return result_score_df, summary_output_score