import os
import glob
import pandas as pd
import joblib
from converting_data_type import ClippingTransformer
from convert_struc_data_type import normalize_rank_score

def process_msfinder_summary(msfinder_file_path, machine_dir, name_adduct_df, summary_df, score_df, top_n = 5):
    # MS-FINDER summary
    file_paths = glob.glob(msfinder_file_path)
    if not file_paths:
        raise FileNotFoundError(f"No files found matching pattern: {msfinder_file_path}")
    
    msfinder_output_combined = pd.concat([pd.read_table(file) for file in file_paths], ignore_index=True)
    msfinder_output_combined['File name'] = msfinder_output_combined['File name'].astype(str)
    msfinder_output_combined["name"] = msfinder_output_combined['File name'].str.split('.').str[0].str.split('_').str[-1]
    msfinder_output_combined["Rank"] = msfinder_output_combined.groupby("File name").cumcount() + 1
    
    score_column = "Score" if "Score" in msfinder_output_combined.columns else "Formula score"
    msfinder_output_combined["score_diff"] = 0 

    mask = (msfinder_output_combined["Rank"] + 1 == msfinder_output_combined["Rank"].shift(-1).fillna(0).astype(int))
    msfinder_output_combined.loc[mask, "score_diff"] = (
        msfinder_output_combined[score_column] - msfinder_output_combined[score_column].shift(-1)
    )
    
    msfinder_output_combined.rename(columns={"name": "filename"}, inplace=True)
    msfinder_output_combined["score_diff"] = msfinder_output_combined["score_diff"].fillna(0)
    
    filtered_df = msfinder_output_combined.groupby('filename').head(top_n).copy()
    filtered_df = filtered_df.fillna('')
    filtered_df.rename(columns={"Precursor type": "adduct", "Rank": "rank", "Formula": "formula"}, inplace=True)
    
    # Load machine learning pipelines
    msfinder_score_pipeline_path = os.path.join(machine_dir, "pipline_msfinder_score.pkl")
    msfinder_SD_pipeline_path = os.path.join(machine_dir, "pipline_msfinder_score_diff.pkl")
    
    score_pipeline = joblib.load(msfinder_score_pipeline_path)
    SD_pipeline = joblib.load(msfinder_SD_pipeline_path)
    
    filtered_df["Score_NZ"] = score_pipeline.transform(filtered_df[[score_column]])
    filtered_df["Score_NZ_diff"] = SD_pipeline.transform(filtered_df[["score_diff"]])
    
    msfinder_score_calc_df = filtered_df[["filename", "adduct", "rank", "formula", "Score_NZ", "Score_NZ_diff"]].copy()
    msfinder_score_calc_df["tool_name"] = "msfinder"
    msfinder_score_calc_df["Used_tools"] = msfinder_score_calc_df["rank"].apply(lambda r: f"MS-FINDER_Rank:{r}")

    
    normalize_rank_score(msfinder_score_calc_df)  # Assuming normalize_rank is defined elsewhere
    msfinder_score_calc_df['adduct'] = msfinder_score_calc_df['filename'].map(name_adduct_df.set_index('filename')['adduct'])

    filtered_df['rank'] = filtered_df['rank'].astype(int)
    top5_df = filtered_df[filtered_df['rank'] <= 5]
    filtered_df = filtered_df.astype(str)
    formula_pivot = top5_df[["filename", "adduct", "rank", "formula"]].pivot(
        index=["filename", "adduct"], 
        columns=["rank"], 
        values=["formula"]
    )
    
    formula_pivot.columns = [f'msfinder_formula_{col[1]}' for col in formula_pivot.columns.values]
    pivot = formula_pivot.reset_index()
    msfinder_formula_df = summary_df.merge(pivot, on=["filename", "adduct"], how="outer")
    
    score_df = pd.concat([score_df, msfinder_score_calc_df], ignore_index=True)
    
    return msfinder_formula_df, score_df