import subprocess
import os
import sys
from convert_struc_data_type import modify_msfinder_config_in_place


def run_msfinder(msfinder_directory, input_path, output_path, method_path, library_path, config):

    # Modify config before running
    modify_msfinder_config_in_place(method_path, library_path, config)

    # Full path to MSFinder executable
    msfinder_exe = os.path.join(msfinder_directory, "MsfinderConsoleApp.exe")

    if not os.path.exists(msfinder_exe):
        print(f"Error: Executable not found at {msfinder_exe}")
        return

    # Prepare command
    command = [
        msfinder_exe,
        "predict",
        "-i", input_path,
        "-o", output_path,
        "-m", method_path
    ]

    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end="")
            proc.wait()

    except Exception as e:
        print(f"An error occurred during MS-FINDER execution: {e}")

