import os
import argparse
import getpass
import pandas as pd
from msemblator.workflows.formula_main import formula_elucidation
from msemblator.workflows.struc_main import structure_elucidation
from msemblator.formats.msp_format_change import msp_formula_changer, convert_name_to_peakid, save_updated_msp, modify_msp_data_type
from msemblator.scoring.struc_score_normalization import ClippingTransformer, safe_log2, safe_logit, safe_sqrt
from msemblator.paths import SIRIUS_DIR, WORK_DIR, ensure_runtime_directories, validate_runtime
from msemblator.runners.sirius_cmd import sirius_login
from msemblator import toolsets
import sys


def _register_pickle_helpers():
    """Expose helpers that the model pickles reference as __main__.* attributes.

    Under ``python -m msemblator.cli.msemblator`` this module is __main__, but the
    ``msemblator`` console script has its own __main__ and loading would fail.
    """
    main_module = sys.modules["__main__"]
    for helper in (ClippingTransformer, safe_log2, safe_logit, safe_sqrt):
        if not hasattr(main_module, helper.__name__):
            setattr(main_module, helper.__name__, helper)


def print_tool_combinations(toolsets_file=None):
    """Print every registered tool combination and whether its files are in place."""
    status = toolsets.available_selections(toolsets.load_toolsets(toolsets_file))
    for step, entries in status.items():
        print(f"\n{step} tools (--{step}_tools):")
        for entry in entries:
            default = " [default]" if entry.is_default else ""
            state = "ready" if entry.ready else "not ready"
            print(f"  {entry.selection.name:<8} {', '.join(entry.selection.labels):<34} {state}{default}")
            for item in entry.missing:
                print(f"           missing: {item}")
    return 0


def main():
    _register_pickle_helpers()
    parser = argparse.ArgumentParser(
        prog="msemblator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Metabolomics annotation tool with ensemble learning.\n"
            "\n"
            "modes:\n"
            "  1  formula elucidation\n"
            "  2  formula and structure elucidation\n"
            "  3  structure elucidation only (requires molecular formulas in the input MSP)\n"
            "\n"
            "tools (--formula_tools / --structure_tools; a name or a list of tools):\n"
            "  formula    sirius   SIRIUS, MS-FINDER, msbuddy          (default)\n"
            "             fiddle   MS-FINDER, msbuddy, FIDDLE\n"
            "             all      SIRIUS, MS-FINDER, msbuddy, FIDDLE\n"
            "  structure  sirius   MS-FINDER, SIRIUS, MetFrag          (default)\n"
            "             glacier  MS-FINDER, GLACIER, MetFrag\n"
            "             all      MS-FINDER, SIRIUS, MetFrag, GLACIER\n"
            "  Each combination needs its own trained ranking model (config/toolsets.yaml).\n"
            "  --list_tools shows which combinations are ready on this computer."
        ),
        epilog=(
            "SIRIUS login (only when SIRIUS is among the selected tools):\n"
            "  SIRIUS 6 requires a login, also for formula prediction.\n"
            "  Recommended: log in once with 'sirius.exe login'; that login is reused.\n"
            "  Otherwise give --sirius_user (or SIRIUS_USER). The password is read from\n"
            "  SIRIUS_PASS or asked for at the prompt; --sirius_pass also works but is\n"
            "  kept in shell history.\n"
            "\n"
            "examples (PowerShell):\n"
            "  msemblator --input data\\example.msp --output results --mode 1\n"
            "  msemblator --input data\\example.msp --output results --mode 2 `\n"
            "             --sirius_user your_email@example.com\n"
            "  msemblator --input data\\formula_predicted.msp --output results --mode 3\n"
            "  msemblator --input data\\example.msp --output results --mode 2 `\n"
            "             --formula_tools fiddle --structure_tools glacier\n"
            "  msemblator --input data\\example.msp --output results --mode 1 `\n"
            "             --formula_tools msfinder,msbuddy,fiddle\n"
            "  msemblator --list_tools"
        ),
    )

    parser.add_argument("--input", metavar="MSP", help="Input MSP file path (required)")
    parser.add_argument("--output", metavar="DIR",
                        help="Folder for the summary CSV files (created if missing; required)")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3],
                        help="1: Formula elucidation, 2: Both formula and structure, 3: Structure elucidation only (required)")
    parser.add_argument("--formula_tools", metavar="TOOLS",
                        help="Formula tools: sirius (default), fiddle, all, or a list such as msfinder,msbuddy,fiddle")
    parser.add_argument("--structure_tools", metavar="TOOLS",
                        help="Structure tools: sirius (default), glacier, all, or a list such as msfinder,glacier,metfrag")
    parser.add_argument("--toolsets", metavar="YAML",
                        help="Tool combinations and their model folders (default: config/toolsets.yaml)")
    parser.add_argument("--list_tools", action="store_true",
                        help="Show the tool combinations and the files each one still needs, then exit")
    parser.add_argument("--sirius_user", metavar="USER",
                        help="SIRIUS account email or username (optional; defaults to SIRIUS_USER, then the stored login)")
    parser.add_argument("--sirius_pass", metavar="PASSWORD",
                        help="SIRIUS account password (optional; defaults to SIRIUS_PASS, then a prompt)")

    args = parser.parse_args()
    if args.list_tools:
        return print_tool_combinations(args.toolsets)
    missing_args = [f"--{name}" for name in ("input", "output", "mode") if getattr(args, name) is None]
    if missing_args:
        parser.error(f"the following arguments are required: {', '.join(missing_args)}")

    try:
        toolset_config = toolsets.load_toolsets(args.toolsets)
        formula_tools = toolsets.resolve_selection(toolsets.FORMULA, args.formula_tools, toolset_config)
        structure_tools = toolsets.resolve_selection(toolsets.STRUCTURE, args.structure_tools, toolset_config)
    except ValueError as e:
        print(f"\nError: {e}")
        return 2
    if args.mode in (1, 2):
        print(f"Formula tools:   {formula_tools.describe()}")
    if args.mode in (2, 3):
        print(f"Structure tools: {structure_tools.describe()}")
    validate_runtime(args.mode, formula_tools, structure_tools)

    
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

    # SIRIUS 6 requires a login, so log in once before any SIRIUS step when SIRIUS
    # is among the selected tools. Without a user name, the login stored by
    # `sirius.exe login` is used.
    uses_sirius = ((args.mode in (1, 2) and formula_tools.uses("sirius"))
                   or (args.mode in (2, 3) and structure_tools.uses("sirius")))
    sirius_user = args.sirius_user or os.environ.get("SIRIUS_USER")
    if not uses_sirius:
        pass
    elif sirius_user:
        sirius_pass = args.sirius_pass or os.environ.get("SIRIUS_PASS")
        if not sirius_pass:
            sirius_pass = getpass.getpass(f"SIRIUS password for {sirius_user}: ")
        if not sirius_pass:
            print("\nError: no SIRIUS password given.")
            return 1
        try:
            sirius_login(str(SIRIUS_DIR), sirius_user, sirius_pass)
        except RuntimeError as e:
            print(f"\nError: {e}")
            return 1
    else:
        print("\nNo SIRIUS credentials given; using the login stored by `sirius.exe login`.")

    formula_failed_steps = []
    structure_failed_steps = []
    if args.mode == 1:
        print("\nRunning formula elucidation only...")
        formula_summary = formula_elucidation(converted_msp_path, output_dir, name_df, formula_failed_steps, formula_tools)

    elif args.mode in (2, 3):
        if args.mode == 2:
            print("\nRunning both formula and structure elucidation...")
            formula_summary = formula_elucidation(converted_msp_path, output_dir, name_df, formula_failed_steps, formula_tools)
            msp_formula_changer(converted_msp_path, formula_summary, formula_fixed_msp_path)
            structure_failed_steps = structure_elucidation(formula_fixed_msp_path, output_dir, name_df, structure_tools)
        else:
            print("\nRunning structure elucidation only...")
            structure_failed_steps = structure_elucidation(converted_msp_path, output_dir, name_df, structure_tools)

    else:
        print("\nInvalid mode selected. Exiting.")
        return 1

    # Results are written even when a tool fails; the exit code still reports it.
    if formula_failed_steps:
        print(f"\nError: formula elucidation completed with failed steps: {', '.join(formula_failed_steps)}")
    if structure_failed_steps:
        print(f"\nError: structure elucidation completed with failed steps: {', '.join(structure_failed_steps)}")
    if "SIRIUS" in formula_failed_steps + structure_failed_steps:
        print("If SIRIUS stopped with 'Login ERROR', log in again with "
              "'sirius.exe login' or pass --sirius_user.")
    if formula_failed_steps or structure_failed_steps:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
