import os
import shutil
import getpass
import glob
import pandas as pd
import time
import logging
import yaml
from msemblator.runners.metfrag_file_processing import creat_metfrag_file
from msemblator.runners.metfrag_struc_cmd import run_metfrag_command
from msemblator.formats.splitting_msp import read_msp
from msemblator.runners.msfinder_struc_cmd import run_msfinder, process_folder
from msemblator.formats.msp_to_ms import convert_msp_file_to_ms
from msemblator.runners.sirius_struc_cmd import sirius_login, run_sirius_struc
from msemblator.summaries.creating_struc_summary import struc_summary
from msemblator.chemistry.converting_data_type import modify_msfinder_config_in_place
from msemblator.utils.struc_utility import clear_folder, clear_folder_except, save_file, generate_unique_filename
from msemblator.scoring.struc_score_normalization import ClippingTransformer
from msemblator.paths import METFRAG_DIR, METFRAG_CONFIG_DIR, LIBRARY_DIR, MSFINDER_DIR, MSFINDER_CONFIG_DIR, PARAMETER_FILE, SIRIUS_DIR, STRUCTURE_MODEL_DIR, WORK_DIR, ensure_runtime_directories

# Clear required folders
def structure_elucidation(input_msp, summary_output_dir, username, password, name_df):
    print("Running structure elucidation")
    # Set up logging configuration.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    ensure_runtime_directories()
    current_dir = str(WORK_DIR / "structure")
    msfinder_directorys = os.path.join(str(MSFINDER_DIR), "MSFINDER*")
    msfinder_dirs = glob.glob(msfinder_directorys)
    msfinder_directory = msfinder_dirs[0]
    msfinder_folder = os.path.join(current_dir, "msfinder_output")
    library_path = os.path.join(str(LIBRARY_DIR), "MsfinderStructureDB_all.txt")
    msfinder_formula_method_path = os.path.join(str(MSFINDER_CONFIG_DIR), "MsfinderConsoleApp_Param_formula.txt")
    msfinder_structure_method_path = os.path.join(str(MSFINDER_CONFIG_DIR), "MsfinderConsoleApp-Param2_structure.txt")
    msp_folder = os.path.join(current_dir, "msfinder_msp")
    metfrag_parameter_dir = str(METFRAG_CONFIG_DIR)
    metfrag_run_dir = str(METFRAG_DIR)
    ms_dir = os.path.join(current_dir, "sirius", "ms")
    sirius_directory = str(SIRIUS_DIR)
    sirius_outputdir = os.path.join(current_dir, "sirius_output")
    sirius_inputdir = os.path.join(ms_dir, "converted_ms.ms")
    sirius_path = os.path.join(sirius_directory, "sirius.exe")
    structure_search_db = os.path.join(str(LIBRARY_DIR), "sirius_structure_db.siriusdb")
    machine_dir = str(STRUCTURE_MODEL_DIR)
    parameter_path = str(PARAMETER_FILE)
    def load_parameters(param_path):
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    config = load_parameters(parameter_path)

    # Clear required folders.
    metfrag_exclude_items = ["example_paramater.txt", "library_psv_v2.txt", "MetFragCommandLine-2.5.0.jar", "metfrag_StructureDB.txt"]
    clear_folder_except(metfrag_run_dir, metfrag_exclude_items)
    for folder in [msp_folder, ms_dir, msfinder_folder, sirius_outputdir]:
        clear_folder(folder)

    # Ensure necessary folders exist.
    for folder in [msp_folder, metfrag_run_dir, ms_dir, msfinder_folder, sirius_outputdir]:
        if not os.path.exists(folder):
            os.makedirs(folder)


    # SIRIUS Processing
    sirius_start_time = time.time()
    print("SIRIUS processing start")
    try:
        ms_file = convert_msp_file_to_ms(input_msp)
        save_file(sirius_inputdir, ms_file)
        sirius_login(sirius_directory, username, password)
        run_sirius_struc(sirius_outputdir, sirius_inputdir, sirius_path, structure_search_db, config)
    except Exception as e:
        logging.error(f"SIRIUS processing failed: {e}")
    sirius_end_time = time.time()
    print("SIRIUS processing complete")
    logging.info(f"SIRIUS processing time: {sirius_end_time - sirius_start_time:.2f} seconds")
    
    # MetFrag Processing
    metfrag_start_time = time.time()
    print("MetFrag processing start")
    with open(os.path.join(metfrag_parameter_dir, "example_paramater.txt"), 'r') as file:
        lines = file.readlines()
    with open(os.path.join(metfrag_run_dir, "example_paramater.txt"), 'w') as file:
        for line in lines:
            if line.startswith('FragmentPeakMatchAbsoluteMassDeviation'):
                line = f'FragmentPeakMatchAbsoluteMassDeviation = {config["structure_prediction"]["metfrag"]["MS2_Da"]}\n'
            elif line.startswith('FragmentPeakMatchRelativeMassDeviation'):
                line = f'FragmentPeakMatchRelativeMassDeviation = {config["structure_prediction"]["metfrag"]["MS2_ppm"]}\n'
            file.write(line)
    

    try:
        creat_metfrag_file(
            input_msp, 
            os.path.join(metfrag_parameter_dir, "example_paramater.txt"),
            metfrag_run_dir, 
            os.path.join(LIBRARY_DIR, "metfrag_StructureDB.txt")
        )
        run_metfrag_command(metfrag_run_dir)
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
        run_msfinder(msfinder_directory, msp_folder, msfinder_folder, msfinder_formula_method_path, '', config) # Run formula prediction
        process_folder(msp_folder) # Process the MSP files to extract formulas and prepare MS-FINDER input
        clear_folder(msfinder_folder) # Clear formula prediction results to prepare for structure prediction
        run_msfinder(msfinder_directory, msp_folder, msfinder_folder, msfinder_structure_method_path, library_path, config) # Run structure prediction 
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
            input_msp, msfinder_folder, machine_dir, sirius_outputdir, metfrag_run_dir, top_n=100, summary_n=config['structure_prediction']['msemblator_output_records']
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
