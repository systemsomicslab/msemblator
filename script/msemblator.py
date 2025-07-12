import os
import argparse
import pandas as pd
from formula_main import formula_elucidation
from struc_main import structure_elucidation
from msp_format_change import msp_formula_changer, convert_name_to_peakid, save_updated_msp, modify_msp_data_type
from struc_score_normalization import ClippingTransformer
import sys


def main():
    parser = argparse.ArgumentParser(description="Metabolomics annotation tool with ensemble learning.")
    
    parser.add_argument("--input", required=True, help="Input MSP file path")
    parser.add_argument("--output", required=True, help="Summary output folder directory")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], required=True, 
                        help="1: Formula elucidation, 2: Both formula and structure, 3: Structure elucidation only")
    parser.add_argument("--sirius_user", help="SIRIUS Username (Required for mode 2 and 3)")
    parser.add_argument("--sirius_pass", help="SIRIUS Password (Required for mode 2 and 3)")

    args = parser.parse_args()

    
    input_msp_path = args.input.strip('"').strip("'")
    output_dir = args.output
    current_dir = os.path.abspath(os.path.dirname(__file__))
    formula_fixed_msp_path = os.path.join(current_dir, "save_folder", "formula_fixed_msp", "formula_fixed.msp").replace("\\", "\\\\")
    converted_msp_path = os.path.join(current_dir, "save_folder", "formula_fixed_msp", "id_change.msp").replace("\\", "\\\\")
    save_folder = os.path.join(current_dir, "save_folder", "formula_fixed_msp")
    for folder in [save_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    original_data = modify_msp_data_type(input_msp_path)
    updated_data, name_df = convert_name_to_peakid(original_data)
    save_updated_msp(converted_msp_path, updated_data)

    if args.mode == 1:
        print("\nRunning formula elucidation only...")
        formula_summary = formula_elucidation(converted_msp_path, output_dir, name_df)
    
    elif args.mode in (2, 3):
        if not args.sirius_user or not args.sirius_pass:
            print("\nError: SIRIUS credentials are required for mode 2 and 3.")
            return
        
        if args.mode == 2:
            print("\nRunning both formula and structure elucidation...")
            formula_summary = formula_elucidation(converted_msp_path, output_dir, name_df)
            msp_formula_changer(converted_msp_path, formula_summary, formula_fixed_msp_path)
            structure_elucidation(formula_fixed_msp_path, output_dir, args.sirius_user, args.sirius_pass, 
                                  "MsfinderConsoleApp-Param_all_processing.txt", name_df)
        else:
            print("\nRunning structure elucidation only...")
            structure_elucidation(converted_msp_path, output_dir, args.sirius_user, args.sirius_pass, 
                                  "MsfinderConsoleApp-Param_all_processing.txt", name_df)
    
    else:
        print("\nInvalid mode selected. Exiting.")

if __name__ == "__main__":
    main()
