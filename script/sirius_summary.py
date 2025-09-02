import os
import glob
import pandas as pd
import joblib
from converting_data_type import normalize_rank,ClippingTransformer

def process_sirius_summary(sirius_folder, machine_dir, name_adduct_df, summary_df, score_df, top_n = 5):
    # Read Sirius output files
    sirius_paths = glob.glob(f"{sirius_folder}/*/formula_candidates.tsv")
    if not sirius_paths:
        print(f"No Sirius files found in {sirius_folder}")
        return summary_df, score_df
    
    data_frames = []
    for file in sirius_paths:
        try:
            df = pd.read_csv(file, sep='\t')
            df['filename'] = os.path.basename(os.path.dirname(file)).split('_')[-1]
            data_frames.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not data_frames:
        return summary_df, score_df
    
    conbine_data = pd.concat(data_frames, ignore_index=True)
    conbine_data["rank"] = conbine_data.groupby("filename").cumcount() + 1
    score_column = "SiriusScore" if "SiriusScore" in conbine_data.columns else "score"
    conbine_data["score_diff"] = 0 
    
    mask = (conbine_data["rank"] + 1 == conbine_data["rank"].shift(-1).fillna(0).astype(int))
    conbine_data.loc[mask, "score_diff"] = (
        conbine_data[score_column] - conbine_data[score_column].shift(-1)
    )
    conbine_data["score_diff"] = conbine_data["score_diff"].fillna(0)
    
    replace_dict = {
        r"\[M \+ H3N \+ H\]\+": "[M+NH4]+",
        r"\[M \+ CH2O2 - H\]-": "[M+FA+H]+"
    }
    for pattern, replacement in replace_dict.items():
        conbine_data["adduct"] = conbine_data["adduct"].fillna("").str.replace(pattern, replacement, regex=True)
    
    sirius_score_pipeline_path = os.path.join(machine_dir, "pipline_sirius_score.pkl")
    sirius_SD_pipeline_path = os.path.join(machine_dir, "pipline_sirius_score_diff.pkl")
    
    score_pipeline = joblib.load(sirius_score_pipeline_path)
    SD_pipeline = joblib.load(sirius_SD_pipeline_path)
    
    filtered_df = conbine_data.groupby('filename').head(top_n).copy()
    filtered_df["Score_NZ"] = score_pipeline.transform(filtered_df[[score_column]])
    filtered_df["Score_NZ_diff"] = SD_pipeline.transform(filtered_df[["score_diff"]])
    filtered_df.rename(columns={"molecularFormula": "formula"}, inplace=True)
    
    sirius_score_calc_df = filtered_df[["filename", "adduct", "rank", "formula", "Score_NZ", "Score_NZ_diff"]]
    sirius_score_calc_df["tool_name"] = "sirius"
    
    sirius_score_calc_df['adduct'] = sirius_score_calc_df['filename'].map(name_adduct_df.set_index('filename')['adduct'])
    
    normalize_rank(sirius_score_calc_df)  # Assuming normalize_rank is defined elsewhere
    
    filtered_df = filtered_df.astype(str).fillna('')
    formula_pivot = filtered_df[["filename", "adduct", "rank", "formula"]].pivot(
        index=["filename"], 
        columns=["rank"], 
        values=["formula"]
    )
    
    formula_pivot.columns = [f'sirius_formula_{col[1]}' for col in formula_pivot.columns.values]
    pivot = formula_pivot.reset_index()
    sirius_formula_df = summary_df.merge(pivot, on=["filename"], how="outer")
    
    score_df = pd.concat([score_df, sirius_score_calc_df], ignore_index=True)
    
    return sirius_formula_df, score_df
