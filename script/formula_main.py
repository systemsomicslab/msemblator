import os
import sys
import shutil
import pandas as pd
import time
import glob
import yaml
from splitting_msp import read_msp
from msp_to_mgf import convert_msp_to_mgf, split_mgf_by_adduct_in_memory
from msp_to_ms import convert_msp_file_to_ms
from msfinder_cmd import run_msfinder
from sirius_cmd import sirius_login, run_sirius
from buddy_cmd import run_msbuddy
from creating_summary import creating_output_summary
from converting_data_type import generate_unique_filename, ClippingTransformer, modify_msfinder_config_in_place

def formula_elucidation(input_msp_path, summary_output_dir, name_df):
    print("Running formula elucidation")

    # Set current directory and add it to the Python path.
    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)

    # Start timer to measure processing time.
    start = time.time()

    # Define necessary directories.
    msp_folder = os.path.join(current_dir, "save_folder", "msfinder_msp")
    ms_folder = os.path.join(current_dir, "save_folder", "sirius_ms")
    mgf_folder = os.path.join(current_dir, "save_folder", "buddy_mgf")
    msfinder_folder = os.path.join(current_dir, "msfinder", "output")
    buddy_folder = os.path.join(current_dir, "buddy")
    sirius_folder = os.path.join(current_dir, "sirius", "output")
    msfinder_directorys = os.path.join(current_dir, "msfinder", "MSFINDER*")
    msfinder_dirs = glob.glob(msfinder_directorys)
    msfinder_directory = msfinder_dirs[0]
    msfinder_method_path = os.path.join(current_dir, "msfinder", "MsfinderConsoleApp_Param_formula.txt")
    model_dir = os.path.join(current_dir, "formula_scoring_model")
    sirius_path = os.path.join(current_dir, "sirius", "sirius.exe")
    msfinder_file_path = os.path.join(current_dir, "msfinder", "output", "Formula*.txt")
    parameter_path = os.path.join(current_dir, 'msemblator_parameter_file.yaml')

    def load_parameters(param_path):
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    config = load_parameters(parameter_path)


    # Function to clear folder contents.
    def clear_folder(folder):
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    # Clear contents of required folders.
    for folder in [msp_folder, ms_folder, mgf_folder, msfinder_folder, buddy_folder, sirius_folder]:
        clear_folder(folder)

    # Ensure necessary folders exist.
    for folder in [msp_folder, ms_folder, mgf_folder, msfinder_folder, buddy_folder, sirius_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Function to save file content.
    def save_file(file_path, content):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

    # 1. Split MSP file into individual files and save them.
    split_data = read_msp(input_msp_path)
    for filename, content in split_data.items():
        msp_output = os.path.join(msp_folder, f"{filename}.msp")
        save_file(msp_output, content)
    print("Saved split msp")

    # 2. Convert MSP file to MS format and save it.
    ms_data = convert_msp_file_to_ms(input_msp_path)
    ms_output = os.path.join(ms_folder, 'converted_ms.ms')
    save_file(ms_output, ms_data)
    print(f"Saved MS data to {ms_output}")

    # 3. Convert MSP to MGF format, split by adduct, and save each file.
    if not os.path.exists(mgf_folder):
        os.makedirs(mgf_folder)
    mgf_content = convert_msp_to_mgf(input_msp_path)
    adduct_data = split_mgf_by_adduct_in_memory(mgf_content)
    for adduct, spectra_list in adduct_data.items():
        output_file_path = os.path.join(mgf_folder, f'{adduct.replace("/", "_")}.mgf')
        content = "\n\n".join(spectra_list)
        save_file(output_file_path, content)
    print(f"Saved MGF files for adducts: {list(adduct_data.keys())}")

    # 4. Run SIRIUS processing.
    print("SIRIUS processing start")
    run_sirius(sirius_folder, ms_output, sirius_path, config)
    print("SIRIUS processing complete")

    # 5. Run MS-FINDER processing.
    print("MS-FINDER processing start")
    modify_msfinder_config_in_place(msfinder_method_path, config)
    run_msfinder(msfinder_directory, msp_folder, msfinder_folder, msfinder_method_path)
    print("MS-FINDER processing complete")

    # 6. Run Msbuddy processing.
    print("msbuddy processing start")
    run_msbuddy(mgf_folder, buddy_folder, config)
    print("msbuddy processing complete")

    # 7. Generate summary output.
    summary_score_df, summary_output = creating_output_summary(
        input_msp_path, sirius_folder, msfinder_file_path, buddy_folder, model_dir, top_n = 100, summary_n=config['formula_prediction']['msemblator_output_records']
    )

    summary_score_df = pd.merge(name_df, summary_score_df, left_on = "Updated_NAME", right_on = "filename")
    summary_score_df.drop(columns=["Updated_NAME", "filename"], inplace=True)
    summary_score_df.rename(columns={"Original_NAME":"filename"},inplace=True)
    
    formula_fix = summary_output
    summary_output = pd.merge(name_df, summary_output, left_on = "Updated_NAME", right_on = "filename")
    summary_output.drop(columns=["Updated_NAME", "filename"], inplace=True)
    summary_output.rename(columns={"Original_NAME": "filename", "formula": "Top_score_formula"}, inplace=True)

    # Save summary output files.
    summary_file = "formula_summary.csv"
    score_file = "formula_score.csv"
    unique_summary_file = generate_unique_filename(summary_output_dir, summary_file)
    unique_score_file = generate_unique_filename(summary_output_dir, score_file)
    summary_output.to_csv(os.path.join(summary_output_dir, unique_summary_file), index=False)
    summary_score_df.to_csv(os.path.join(summary_output_dir, unique_score_file), index=False)

    # Display processing time.
    end = time.time()
    time_diff = end - start
    print(f"Processing completed in {time_diff} seconds.")

    # Return the summary_score_df DataFrame.
    return formula_fix