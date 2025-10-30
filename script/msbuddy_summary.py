import os
import glob
import pandas as pd
import joblib
from converting_data_type import normalize_rank,ClippingTransformer

def process_buddy_summary(buddy_folder, machine_dir, name_adduct_df, summary_df, score_df, top_n = 5):
    folder = os.path.join(buddy_folder,"detailed_summary_*.csv")
    buddy_files = glob.glob(folder)
    if not buddy_files:
        print(f"No Buddy files found in {buddy_folder}")
        return summary_df, score_df
    
    buddy_result = pd.concat([pd.read_csv(file) for file in buddy_files], ignore_index=True)
    buddy_result['Scan_ID'] = buddy_result['Scan_ID'].astype(str)
    buddy_result.rename(columns={"Formula": "formula", "Scan_ID": "filename"}, inplace=True)
    buddy_result["Estimated_FDR_2"] = 1 - buddy_result["Estimated_FDR"]
    buddy_result["score_diff"] = 0  
    
    mask = (buddy_result["Rank"] + 1 == buddy_result["Rank"].shift(-1).fillna(0).astype(int))
    buddy_result.loc[mask, "score_diff"] = (
        buddy_result["Estimated_FDR_2"] - buddy_result["Estimated_FDR_2"].shift(-1)
    )
    
    filtered_df = buddy_result[["filename", "Rank", "formula", "Estimated_FDR_2", "score_diff"]]
    filtered_df = filtered_df[filtered_df["Rank"] <= top_n]
    
    buddy_score_pipeline_path = os.path.join(machine_dir, "pipline_buddy_score.pkl")
    buddy_SD_pipeline_path = os.path.join(machine_dir, "pipline_buddy_score_diff.pkl")
    
    score_pipeline = joblib.load(buddy_score_pipeline_path)
    SD_pipeline = joblib.load(buddy_SD_pipeline_path)
    
    filtered_df["Score_NZ"] = score_pipeline.transform(filtered_df[["Estimated_FDR_2"]])
    filtered_df["Score_NZ_diff"] = SD_pipeline.transform(filtered_df[["score_diff"]])
    filtered_df.rename(columns={"Rank": "rank"}, inplace=True)
    
    filtered_df["adduct"] = ""
    filtered_df['adduct'] = filtered_df['filename'].map(name_adduct_df.set_index('filename')['adduct'])
    
    buddy_score_calc_df = filtered_df[["filename", "adduct", "rank", "formula", "Score_NZ", "Score_NZ_diff"]]
    buddy_score_calc_df["tool_name"] = "buddy"
    buddy_score_calc_df['Used_tools'] = buddy_score_calc_df["rank"].apply(lambda r: f"msbuddy_Rank:{r}")
    
    buddy_score_calc_df['adduct'] = buddy_score_calc_df['filename'].map(name_adduct_df.set_index('filename')['adduct'])
    normalize_rank(buddy_score_calc_df)
    
    filtered_df = filtered_df.astype(str).fillna('')
    filtered_df = filtered_df.drop_duplicates(subset=["filename", "rank"])
    
    formula_pivot = filtered_df[["filename", "adduct", "rank", "formula"]].pivot(
        index=["filename"], 
        columns=["rank"], 
        values=["formula"]
    )
    
    formula_pivot.columns = [f'buddy_formula_{col[1]}' for col in formula_pivot.columns.values]
    pivot = formula_pivot.reset_index()
    buddy_formula_df = summary_df.merge(pivot, on=["filename"], how="outer")
    
    score_df = pd.concat([score_df, buddy_score_calc_df], ignore_index=True)
    
    return buddy_formula_df, score_df