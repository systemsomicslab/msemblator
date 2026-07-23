import os
import glob
import pandas as pd
import joblib
from msemblator.chemistry.converting_data_type import ClippingTransformer
from msemblator.chemistry.convert_struc_data_type import normalize_rank_score

def process_sirius_summary(sirius_folder, machine_dir, name_adduct_df, summary_df, score_df, top_n = 5):
    # Read Sirius output files
    sirius_paths = os.path.join(sirius_folder, "formula_identifications_top-100.tsv")
    if not os.path.isfile(sirius_paths):
        print(f"No Sirius files found in {sirius_folder}")
        return summary_df, score_df

    try:
        sirius6_df = pd.read_table(sirius_paths, sep='\t')
        sirius6_df['filename'] = sirius6_df['mappingFeatureId'].str.split('_').str[-1]
    except Exception as e:
        print(f"Error reading {sirius_paths}: {e}")
    
    if sirius6_df.empty:
        return summary_df, score_df

    score_column = "SiriusScore" if "SiriusScore" in sirius6_df.columns else "score"

    sirius6_df.rename(columns={'molecularFormula': 'formula'}, inplace=True)
    sirius6_df["Score_Difference"] = 0.0
    sirius6_df["formulaRank"] = sirius6_df["formulaRank"].astype(int)
    sirius6_df["rank"] = sirius6_df["formulaRank"].astype(int)
    sirius6_df["SiriusScore"] = pd.to_numeric(sirius6_df["SiriusScore"], errors="coerce")
    sirius6_df = sirius6_df.reset_index(drop=True)
    mask = (sirius6_df["formulaRank"] + 1 == sirius6_df["formulaRank"].shift(-1))
    sirius6_df.loc[mask, "Score_Difference"] = (
        sirius6_df["SiriusScore"] - sirius6_df["SiriusScore"].shift(-1)
    )
    
    replace_dict = {
        r"\[M \+ H3N \+ H\]\+": "[M+NH4]+",
        r"\[M \+ CH2O2 - H\]-": "[M+FA+H]+"
    }
    for pattern, replacement in replace_dict.items():
        sirius6_df["adduct"] = sirius6_df["adduct"].fillna("").str.replace(pattern, replacement, regex=True)
    
    sirius_score_pipeline_path = os.path.join(machine_dir, "pipeline_sirius_score.pkl")
    sirius_SD_pipeline_path = os.path.join(machine_dir, "pipeline_sirius_score_diff.pkl")
    
    score_pipeline = joblib.load(sirius_score_pipeline_path)
    SD_pipeline = joblib.load(sirius_SD_pipeline_path)
    
    filtered_df = sirius6_df.groupby('filename').head(top_n).copy()
    filtered_df["Score_NZ"] = score_pipeline.transform(filtered_df[[score_column]])
    filtered_df["Score_NZ_diff"] = SD_pipeline.transform(filtered_df[["Score_Difference"]])
    filtered_df.rename(columns={"molecularFormula": "formula"}, inplace=True)
    
    sirius_score_calc_df = filtered_df[["filename", "adduct", "rank", "formula", "Score_NZ", "Score_NZ_diff"]]
    sirius_score_calc_df["tool_name"] = "sirius"
    sirius_score_calc_df['Used_tools'] = sirius_score_calc_df["rank"].apply(lambda r: f"SIRIUS_Rank:{r}")
    
    sirius_score_calc_df['adduct'] = sirius_score_calc_df['filename'].map(name_adduct_df.set_index('filename')['adduct'])
    
    normalize_rank_score(sirius_score_calc_df)  # Assuming normalize_rank is defined elsewhere
    
    filtered_df['rank'] = filtered_df['rank'].astype(int)
    top5_df = filtered_df[filtered_df['rank'] <= 5]
    filtered_df = filtered_df.astype(str).fillna('')
    formula_pivot = top5_df[["filename", "adduct", "rank", "formula"]].pivot(
        index=["filename"], 
        columns=["rank"], 
        values=["formula"]
    )
    
    formula_pivot.columns = [f'sirius_formula_{col[1]}' for col in formula_pivot.columns.values]
    pivot = formula_pivot.reset_index()
    sirius_formula_df = summary_df.merge(pivot, on=["filename"], how="outer")
    
    score_df = pd.concat([score_df, sirius_score_calc_df], ignore_index=True)
    
    return sirius_formula_df, score_df
