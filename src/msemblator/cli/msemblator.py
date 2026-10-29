import os
import argparse
import pandas as pd
from msemblator.workflows.formula_main import formula_elucidation
from msemblator.workflows.struc_main import structure_elucidation
from msemblator.formats.msp_format_change import msp_formula_changer, convert_name_to_peakid, save_updated_msp, modify_msp_data_type
from msemblator.scoring.struc_score_normalization import ClippingTransformer, safe_log2, safe_sqrt
from msemblator.paths import SIRIUS_DIR, WORK_DIR, ensure_runtime_directories, validate_runtime
from msemblator.runners.sirius_cmd import sirius_login
import sys


def _register_pickle_helpers():
    """Expose helpers that the model pickles reference as __main__.* attributes.

    Under ``python -m msemblator.cli.msemblator`` this module is __main__, but the
    ``msemblator`` console script has its own __main__ and loading would fail.
    """
    main_module = sys.modules["__main__"]
    for helper in (ClippingTransformer, safe_log2, safe_sqrt):
        if not hasattr(main_module, helper.__name__):
            setattr(main_module, helper.__name__, helper)


def main():
    _register_pickle_helpers()
    parser = argparse.ArgumentParser(
        prog="msemblator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Metabolomics annotation tool with ensemble learning.\n"
            "\n"
            "modes:\n"
            "  1  formula elucidation               (SIRIUS, MS-FINDER, msbuddy)\n"
            "  2  formula and structure elucidation (SIRIUS, MS-FINDER, msbuddy, MetFrag)\n"
            "  3  structure elucidation only        (SIRIUS, MS-FINDER, MetFrag)\n"
            "     mode 3 requires molecular formulas in the input MSP records."
        ),
        epilog=(
            "SIRIUS login:\n"
            "  SIRIUS 6 requires a login in every mode, including formula prediction.\n"
            "  Modes 2 and 3 require --sirius_user and --sirius_pass. In mode 1 they are\n"
            "  optional; without them, the login stored by 'sirius.exe login' is used.\n"
            "\n"
            "examples (PowerShell):\n"
            "  msemblator --input data\\example.msp --output results --mode 1 `\n"
            "             --sirius_user your_email@example.com --sirius_pass your_password\n"
            "  msemblator --input data\\example.msp --output results --mode 2 `\n"
            "             --sirius_user your_email@example.com --sirius_pass your_password\n"
            "  msemblator --input data\\formula_predicted.msp --output results --mode 3 `\n"
            "             --sirius_user your_email@example.com --sirius_pass your_password"
        ),
    )

    parser.add_argument("--input", required=True, metavar="MSP", help="Input MSP file path")
    parser.add_argument("--output", required=True, metavar="DIR",
                        help="Folder for the summary CSV files (created if missing)")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], required=True,
                        help="1: Formula elucidation, 2: Both formula and structure, 3: Structure elucidation only")
    parser.add_argument("--sirius_user", metavar="USER",
                        help="SIRIUS account email or username (required for mode 2 and 3; optional for mode 1)")
    parser.add_argument("--sirius_pass", metavar="PASSWORD",
                        help="SIRIUS account password (required for mode 2 and 3; optional for mode 1)")

    args = parser.parse_args()
    validate_runtime(args.mode)

    
    input_msp_path = args.input.strip('"').strip("'")
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ensure_runtime_directories()
    current_dir = str(WORK_DIR)
    formula_fixed_msp_path = os.path.join(current_dir, "save_folder", "formula_fixed_msp", "formula_fixed.msp").replace("\\", "\\\\")
    converted_msp_path = os.path.join(current_dir, "save_folder", "formula_fixed_msp", "id_change.msp").replace("\\", "\\\\")
    save_folder = os.path.join(current_dir, "save_folder", "formula_fixed_msp")
    for folder in [save_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    original_data = modify_msp_data_type(input_msp_path)
    updated_data, name_df = convert_name_to_peakid(original_data)
    save_updated_msp(converted_msp_path, updated_data)

    has_credentials = bool(args.sirius_user and args.sirius_pass)
    if args.mode in (2, 3) and not has_credentials:
        print("\nError: SIRIUS credentials are required for mode 2 and 3.")
        return 1

    # SIRIUS 6 requires a login even for formula prediction, so log in before the
    # formula step. Without credentials, mode 1 relies on a stored `sirius login`.
    if args.mode in (1, 2):
        if has_credentials:
            try:
                sirius_login(str(SIRIUS_DIR), args.sirius_user, args.sirius_pass)
            except RuntimeError as e:
                print(f"\nError: {e}")
                return 1
        else:
            print("\nNo SIRIUS credentials given; using the login stored by `sirius login`.")

    formula_failed_steps = []
    structure_failed_steps = []
    if args.mode == 1:
        print("\nRunning formula elucidation only...")
        formula_summary = formula_elucidation(converted_msp_path, output_dir, name_df, formula_failed_steps)

    elif args.mode in (2, 3):
        if args.mode == 2:
            print("\nRunning both formula and structure elucidation...")
            formula_summary = formula_elucidation(converted_msp_path, output_dir, name_df, formula_failed_steps)
            msp_formula_changer(converted_msp_path, formula_summary, formula_fixed_msp_path)
            structure_failed_steps = structure_elucidation(formula_fixed_msp_path, output_dir, args.sirius_user, args.sirius_pass,
                                  name_df)
        else:
            print("\nRunning structure elucidation only...")
            structure_failed_steps = structure_elucidation(converted_msp_path, output_dir, args.sirius_user, args.sirius_pass,
                                  name_df)

    else:
        print("\nInvalid mode selected. Exiting.")
        return 1

    # Results are written even when a tool fails; the exit code still reports it.
    if formula_failed_steps:
        print(f"\nError: formula elucidation completed with failed steps: {', '.join(formula_failed_steps)}")
    if structure_failed_steps:
        print(f"\nError: structure elucidation completed with failed steps: {', '.join(structure_failed_steps)}")
    if formula_failed_steps or structure_failed_steps:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
