import os
import pandas as pd
from formula_main import formula_elucidation
from struc_main import structure_elucidation
from msp_format_change import msp_formula_changer, read_msp_file, convert_name_to_peakid, save_updated_msp
from struc_score_normalization import ClippingTransformer

def main():
    # Prompt for basic inputs.
    current_dir = os.path.abspath(os.path.dirname(__file__))
    input_msp_path = input("Input MSP file path: ").strip("'").strip('"')
    output_dir = input("Summary output folder directory: ").strip("'").strip('"')
    formula_fixed_msp_path = os.path.join(current_dir, "save_folder", "formula_fixed_msp", "formula_fixed.msp")
    converted_msp_path = os.path.join(current_dir, "save_folder", "formula_fixed_msp", "id_change.msp")
    
    original_data = read_msp_file(input_msp_path)
    updated_data, name_df = convert_name_to_peakid(original_data)
    save_updated_msp(converted_msp_path, updated_data)
    
    # Display analysis mode options.
    print("\nSelect analysis mode:")
    print("1. Formula elucidation only")
    print("2. Both formula and structure elucidation")
    print("3. Structure elucidation only")
    
    mode = input("Enter option (1/2/3): ").strip()
    
    if mode == "1":
        # Run formula elucidation only.
        print("\nRunning formula elucidation only...")
        # Assuming formula_elucidation in formula_main does not require SIRIUS credentials.
        formula_summary = formula_elucidation(converted_msp_path, output_dir, name_df)
    elif mode in ("2", "3"):
        # For modes 2 and 3, prompt for SIRIUS credentials.
        sirius_username = input("SIRIUS Username: ")
        sirius_password = input("SIRIUS Password: ")
        if mode == "2":
            print("\nRunning both formula and structure elucidation...")
            # Assuming both functions require SIRIUS credentials.
            formula_summary = formula_elucidation(converted_msp_path, output_dir, name_df)
            msp_formula_changer(converted_msp_path, formula_summary, formula_fixed_msp_path)
            structure_elucidation(formula_fixed_msp_path, output_dir, sirius_username, sirius_password, "MsfinderConsoleApp-Param2_structure.txt", name_df)
        else:
            print("\nRunning structure elucidation only...")
            structure_elucidation(converted_msp_path, output_dir, sirius_username, sirius_password, "MsfinderConsoleApp-Param_all_processing.txt",name_df)
    else:
        print("\nInvalid option selected. Exiting.")

if __name__ == "__main__":
    main()