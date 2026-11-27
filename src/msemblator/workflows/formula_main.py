import os
import re
import sys
import shutil
import pandas as pd
import time
import glob
import yaml
import logging
from msemblator.formats.splitting_msp import read_msp
from msemblator.formats.msp_to_mgf import convert_msp_to_mgf, split_mgf_by_adduct_in_memory
from msemblator.formats.msp_to_ms import convert_msp_file_to_ms
from msemblator.runners.msfinder_cmd import run_formula_msfinder
from msemblator.runners.sirius_cmd import run_sirius
from msemblator.runners.buddy_cmd import run_msbuddy
from msemblator.runners.fiddle_cmd import run_fiddle
from msemblator.summaries.creating_summary import creating_output_summary
from msemblator.chemistry.converting_data_type import generate_unique_filename, ClippingTransformer, modify_msfinder_config_in_place
from msemblator.paths import FORMULA_MODEL_DIR, MSFINDER_DIR, PARAMETER_FILE, SIRIUS_DIR, WORK_DIR, MSFINDER_CONFIG_DIR, ensure_runtime_directories
from msemblator import toolsets

# Adduct names become MGF file names, so only bracket notation such as
# "[M+H]+", "[M+FA-H]-" or "[M+2H]2+" is accepted (no path separators or "..").
ADDUCT_FILENAME_PATTERN = re.compile(r"\[[A-Za-z0-9+\-]+\][0-9]*[+-]")


def adduct_mgf_path(mgf_folder, adduct):
    """Return the MGF path for an adduct, or None if the name is not a safe file name."""
    if not ADDUCT_FILENAME_PATTERN.fullmatch(adduct):
        return None
    folder = os.path.realpath(mgf_folder)
    path = os.path.realpath(os.path.join(folder, f"{adduct}.mgf"))
    if os.path.dirname(path) != folder:
        return None
    return path


def formula_elucidation(input_msp_path, summary_output_dir, name_df, failed_steps=None, selection=None):
    """Run formula elucidation with the selected tools (a toolsets.ToolSelection).

    Without a selection the default combination of config/toolsets.yaml is used.
    Names of failed tools are appended to failed_steps.
    """
    selection = selection or toolsets.resolve_selection(toolsets.FORMULA)
    print(f"Running formula elucidation: {selection.describe()}")

    ensure_runtime_directories()
    current_dir = str(WORK_DIR / "formula")

    # Start timer to measure processing time.
    start = time.time()

    # Define necessary directories.
    msp_folder = os.path.join(current_dir, "save_folder", "msfinder_msp")
    ms_folder = os.path.join(current_dir, "save_folder", "sirius_ms")
    mgf_folder = os.path.join(current_dir, "save_folder", "buddy_mgf")
    msfinder_folder = os.path.join(current_dir, "msfinder_output")
    buddy_folder = os.path.join(current_dir, "buddy")
    sirius_folder = os.path.join(current_dir, "sirius", "output")
    fiddle_run_dir = os.path.join(current_dir, "fiddle_output")
    fiddle_folder = os.path.join(fiddle_run_dir, "fiddle")
    msfinder_directorys = os.path.join(str(MSFINDER_DIR), "MSFINDER*")
    msfinder_dirs = glob.glob(msfinder_directorys)
    msfinder_directory = msfinder_dirs[0] if msfinder_dirs else None
    msfinder_method_path = os.path.join(str(MSFINDER_CONFIG_DIR), "MsfinderConsoleApp_Param_formula.txt")
    model_dir = str(FORMULA_MODEL_DIR)
    sirius_path = os.path.join(str(SIRIUS_DIR), "sirius.exe")
    msfinder_file_path = os.path.join(msfinder_folder, "Formula*.txt")
    parameter_path = str(PARAMETER_FILE)

    def load_parameters(param_path):
        with open(param_path, 'r', encoding='utf-8') as file:
            params = yaml.safe_load(file)
        return params
    config = load_parameters(parameter_path)


    # Function to clear folder contents.
    def clear_folder(folder):
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    # Clear contents of required folders. Folders of tools that are not selected
    # are cleared too, so an old result cannot be mistaken for this run's output.
    for folder in [msp_folder, ms_folder, mgf_folder, msfinder_folder, buddy_folder, sirius_folder, fiddle_run_dir]:
        clear_folder(folder)

    # Function to save file content.
    def save_file(file_path, content):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

    # 1. Split MSP file into individual files for MS-FINDER.
    if selection.uses("msfinder"):
        split_data = read_msp(input_msp_path)
        for filename, content in split_data.items():
            msp_output = os.path.join(msp_folder, f"{filename}.msp")
            save_file(msp_output, content)
        print("Saved split msp")

    # 2. Convert MSP file to MS format for SIRIUS.
    ms_output = os.path.join(ms_folder, 'converted_ms.ms')
    if selection.uses("sirius"):
        ms_data = convert_msp_file_to_ms(input_msp_path)
        save_file(ms_output, ms_data)
        print(f"Saved MS data to {ms_output}")

    # 3. Convert MSP to MGF format for msbuddy, split by adduct, and save each file.
    if selection.uses("msbuddy"):
        mgf_content = convert_msp_to_mgf(input_msp_path)
        adduct_data = split_mgf_by_adduct_in_memory(mgf_content)
        saved_adducts = []
        for adduct, spectra_list in adduct_data.items():
            output_file_path = adduct_mgf_path(mgf_folder, adduct)
            if output_file_path is None:
                logging.warning(f"Skipped {len(spectra_list)} spectra for msbuddy: unsupported adduct name {adduct!r}")
                continue
            content = "\n\n".join(spectra_list)
            save_file(output_file_path, content)
            saved_adducts.append(adduct)
        print(f"Saved MGF files for adducts: {saved_adducts}")

    # A failing tool does not stop the others; its ranking features become 0.
    if failed_steps is None:
        failed_steps = []

    # 4. Run SIRIUS processing.
    if selection.uses("sirius"):
        print("SIRIUS processing start")
        try:
            run_sirius(sirius_folder, ms_output, sirius_path, config)
        except Exception as e:
            logging.error(f"SIRIUS processing failed: {e}")
            failed_steps.append("SIRIUS")
        print("SIRIUS processing complete")

    # 5. Run MS-FINDER processing.
    if selection.uses("msfinder"):
        print("MS-FINDER processing start")
        try:
            modify_msfinder_config_in_place(msfinder_method_path, config)
            run_formula_msfinder(msfinder_directory, msp_folder, msfinder_folder, msfinder_method_path)
        except Exception as e:
            logging.error(f"MS-FINDER processing failed: {e}")
            failed_steps.append("MS-FINDER")
        print("MS-FINDER processing complete")

    # 6. Run msbuddy processing.
    if selection.uses("msbuddy"):
        print("msbuddy processing start")
        try:
            run_msbuddy(mgf_folder, buddy_folder, config)
        except Exception as e:
            logging.error(f"msbuddy processing failed: {e}")
            failed_steps.append("msbuddy")
        print("msbuddy processing complete")

    # 7. Run FIDDLE processing (in its own environment when msfiddle is not installed here).
    if selection.uses("fiddle"):
        print("FIDDLE processing start")
        try:
            run_fiddle(input_msp_path, fiddle_run_dir, config['formula_prediction'].get('fiddle'))
        except Exception as e:
            logging.error(f"FIDDLE processing failed: {e}")
            failed_steps.append("FIDDLE")
        print("FIDDLE processing complete")

    # 8. Generate summary output.
    summary_score_df, summary_output = creating_output_summary(
        input_msp_path, sirius_folder, msfinder_file_path, buddy_folder, model_dir, 100,
        config['formula_prediction']['msemblator_output_records'], fiddle_folder, selection
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
