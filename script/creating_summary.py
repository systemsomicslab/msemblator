import pandas as pd
import glob
import os
import joblib
from converting_data_type import normalize_rank, ClippingTransformer, read_msp_file, extract_compound_and_ionization
from msfinder_summary import process_msfinder_summary
from sirius_summary import process_sirius_summary
from msbuddy_summary import process_buddy_summary
from calculating_score import predict_and_append,aggregate_probability_with_rank_top3

def creating_output_summary(input_msp, sirius_folder, msfinder_file_path, buddy_folder, machine_dir):
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

    # MS-FINDER summary
    msfinder_formula_df, score_df =process_msfinder_summary(msfinder_file_path, machine_dir, name_adduct_df, summary_df, score_df)

    # sirius summary
    sirius_formula_df, score_df=process_sirius_summary(sirius_folder, machine_dir, name_adduct_df, summary_df, score_df)

    # msbuddy summuary
    buddy_formula_df, score_df=process_buddy_summary(buddy_folder, machine_dir, name_adduct_df, summary_df, score_df)
    score_df["tool_name"] = (
        score_df["tool_name"]
        .astype("string")
        .fillna("")        
        .str.strip().str.lower()
    )
    df_onehot = pd.get_dummies(score_df, columns=['tool_name'])
    required_col = ['tool_name_buddy', 'tool_name_msfinder', 'tool_name_sirius']
    for col in required_col:
        if col not in df_onehot.columns:
            df_onehot[col] = 0
    oh_cols = [c for c in df_onehot.columns if c.startswith("tool_name_")]
    df_onehot[oh_cols] = df_onehot[oh_cols].astype(int)
    df_onehot = df_onehot[['filename', 'adduct', 'rank', 'formula', 'Score_NZ', 'Score_NZ_diff', 'normalized_rank'] + required_col]
    df_onehot = df_onehot[df_onehot['formula'].notna() & (df_onehot['formula'].str.strip() != '')].copy()

    calc_score_df = predict_and_append(df_onehot, machine_dir, adduct_column="adduct")
    summary_score_df = aggregate_probability_with_rank_top3(calc_score_df)
    summary_score_top_df =summary_score_df[summary_score_df["rank"]==1]
    summary_output = pd.merge(msfinder_formula_df, sirius_formula_df, on=['filename', 'adduct'], how='inner')
    summary_output = pd.merge(summary_output, buddy_formula_df, on=['filename', 'adduct'], how='inner')
    summary_output = pd.merge(summary_output, summary_score_top_df, on=['filename', 'adduct'], how='inner')

    return summary_score_df, summary_output
    



    


    

        