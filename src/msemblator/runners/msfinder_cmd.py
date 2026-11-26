import subprocess
import os
import glob


def run_formula_msfinder(msfinder_directory, input_path, output_path, method_path):
    """Run MS-FINDER formula prediction; raise if it cannot run or writes no results."""
    msfinder_exe = os.path.join(msfinder_directory, "MsfinderConsoleApp.exe")
    if not os.path.exists(msfinder_exe):
        raise FileNotFoundError(f"MS-FINDER executable not found: {msfinder_exe}")

    # command preparation
    command = [
        msfinder_exe,
        "predict",
        "-i", input_path,
        "-o", output_path,
        '-m', method_path
    ]

    # MS-FINDER exits with code 1 even after a successful run, so success is judged
    # by the formula result files it writes (read later as "Formula*.txt").
    result_pattern = os.path.join(output_path, "Formula*.txt")
    results_before = set(glob.glob(result_pattern))
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end="")
            return_code = proc.wait()
        if not set(glob.glob(result_pattern)) - results_before:
            raise RuntimeError(f"MS-FINDER wrote no formula result files (exit code {return_code}).")

    except Exception as e:
        print(f"An error occurred during MS-FINDER execution: {e}")
        raise
