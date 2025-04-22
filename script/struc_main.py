import os
import shutil
import getpass
import glob
import pandas as pd
import time
import logging
from metfrag_file_processing import creat_metfrag_file
from metfrag_struc_cmd import run_metfrag_command
from splitting_msp import read_msp
from msfinder_struc_cmd import run_msfinder
from msp_to_ms import convert_msp_file_to_ms
from sirius_struc_cmd import sirius_login, run_sirius_struc
from creating_struc_summary import struc_summary
from struc_utility import clear_folder, clear_folder_except, save_file, generate_unique_filename
from struc_score_normalization import ClippingTransformer

# Clear required folders
def structure_elucidation(input_msp, summary_output_dir, username, password, msfinder_method_file, name_df):
    print("Running structure elucidation")
    # Set up logging configuration.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Define directories and paths.
    current_dir = os.path.abspath(os.path.dirname(__file__))
    msfinder_directorys = os.path.join(current_dir, "msfinder", "MSFINDER*")
    msfinder_dirs = glob.glob(msfinder_directorys)
    msfinder_directory = msfinder_dirs[0]
    msfinder_folder = os.path.join(current_dir, "msfinder", "output")
    library_path = os.path.join(current_dir, "msfinder", "coconutandBLEXP.txt")
    method_path = os.path.join(current_dir, "msfinder", msfinder_method_file)
    msp_folder = os.path.join(current_dir, "msfinder", "msp")
    metfrag_paramater_dir = os.path.join(current_dir, "metfrag")
    ms_dir = os.path.join(current_dir, "sirius", "ms")
    sirius_directory = os.path.join(current_dir, "sirius")
    sirius_outputdir = os.path.join(sirius_directory, "output")
    sirius_inputdir = os.path.join(ms_dir, "converted_ms.ms")
    sirius_path = os.path.join(sirius_directory, "sirius.exe")
    structure_search_db = os.path.join(sirius_directory, "database")
    machine_dir = os.path.join(current_dir, "structure_scoring_model")

    # Clear required folders.
    metfrag_exclude_items = ["example_paramater.txt", "library_psv_v2.txt", "MetFragCommandLine-2.5.0.jar"]
    clear_folder_except(metfrag_paramater_dir, metfrag_exclude_items)
    for folder in [msp_folder, ms_dir, msfinder_folder, sirius_outputdir]:
        clear_folder(folder)

    # Ensure necessary folders exist.
    for folder in [msp_folder, metfrag_paramater_dir, ms_dir, msfinder_folder, sirius_outputdir]:
        if not os.path.exists(folder):
            os.makedirs(folder)


    # SIRIUS Processing
    sirius_start_time = time.time()
    print("SIRIUS processing start")
    try:
        ms_file = convert_msp_file_to_ms(input_msp)
        save_file(sirius_inputdir, ms_file)
        sirius_login(sirius_directory, username, password)
        run_sirius_struc(sirius_outputdir, sirius_inputdir, sirius_path, structure_search_db)
    except Exception as e:
        logging.error(f"SIRIUS processing failed: {e}")
    sirius_end_time = time.time()
    print("SIRIUS processing complete")
    logging.info(f"SIRIUS processing time: {sirius_end_time - sirius_start_time:.2f} seconds")
    
    # MetFrag Processing
    metfrag_start_time = time.time()
    print("MetFrag processing start")
    try:
        creat_metfrag_file(
            input_msp, 
            os.path.join(metfrag_paramater_dir, "example_paramater.txt"),
            metfrag_paramater_dir, 
            os.path.join(metfrag_paramater_dir, "library_psv_v2.txt")
        )
        run_metfrag_command(metfrag_paramater_dir)
    except Exception as e:
        logging.error(f"MetFrag processing failed: {e}")
    print("MetFrag processing complete")
    metfrag_end_time = time.time()
    logging.info(f"MetFrag processing time: {metfrag_end_time - metfrag_start_time:.2f} seconds")
    
    # MS-FINDER Processing 
    msfinder_start_time = time.time()
    print("MS-FINDER processing start")
    try:
        split_data = read_msp(input_msp)
        for filename, content in split_data.items():
            save_file(os.path.join(msp_folder, f"{filename}.msp"), content)
        run_msfinder(msfinder_directory, msp_folder, msfinder_folder, method_path, library_path)
    except Exception as e:
        logging.error(f"MSFinder processing failed: {e}")
    print("MS-FINDER processing complete")
    msfinder_end_time = time.time()
    logging.info(f"MSFinder processing time: {msfinder_end_time - msfinder_start_time:.2f} seconds")
    
    # Summary Generation
    summary_start_time = time.time()
    print("Generating output files...")
    try:
        result_score_df, summary_smiles_df = struc_summary(
            input_msp, msfinder_folder, machine_dir, sirius_outputdir, metfrag_paramater_dir
        )
        result_score_df = pd.merge(name_df, result_score_df, left_on = "Updated_NAME", right_on = "filename")
        result_score_df.drop(columns=["Updated_NAME", "filename"], inplace=True)
        result_score_df.rename(columns={"Original_NAME":"filename"},inplace=True)
        
        summary_smiles_df = pd.merge(name_df, summary_smiles_df, left_on = "Updated_NAME", right_on = "filename")
        summary_smiles_df.drop(columns=["Updated_NAME", "filename"], inplace=True)
        summary_smiles_df.rename(columns={"Original_NAME":"filename","Canonical_SMILES":"Top_score_Canonical_SMILES"},inplace=True)

        result_score_file = generate_unique_filename(summary_output_dir, "structure_score.csv")
        summary_smiles_file = generate_unique_filename(summary_output_dir, "structure_summary.csv")
        result_score_df.to_csv(os.path.join(summary_output_dir, result_score_file), index=False)
        summary_smiles_df.to_csv(os.path.join(summary_output_dir, summary_smiles_file), index=False)
        logging.info(f"Summary saved as: {result_score_file} and {summary_smiles_file}")
    except Exception as e:
        logging.error(f"Summary generation failed: {e}")
    summary_end_time = time.time()
    logging.info(f"Summary generation time: {summary_end_time - summary_start_time:.2f} seconds")
    print("structure elucidation complete")
    