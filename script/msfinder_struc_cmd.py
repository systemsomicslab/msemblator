import subprocess
from convert_struc_data_type import modify_msfinder_config_in_place


def run_msfinder(msfinder_directory, input_path, output_path, method_path, library_path):
    """
    Executes the MSFinder tool using the specified input, output, and method paths.

    Args:
        msfinder_directory (str): Directory containing the MSFinder executable.
        input_path (str): Path to the input data (e.g., MSP files).
        output_path (str): Path to the output directory where results will be saved.
        method_path (str): Path to the MSFinder method parameter file.

    Returns:
        None
    """
    # modify msfinder config
    modify_msfinder_config_in_place(method_path,library_path)
    
    # Define the MSFinder executable
    msfinder_exe = "MsfinderConsoleApp.exe"

    # Construct the PowerShell command to run MSFinder
    command = f'$env:PATH += ";{msfinder_directory}"; {msfinder_exe} predict -i "{input_path}" -o "{output_path}" -m "{method_path}"'

    try:
        # Execute the command in PowerShell
        with subprocess.Popen(
            ["powershell", "-Command", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        ) as proc:
            # Capture and print standard output in real-time
            for line in proc.stdout:
                print(line, end="")

            # Capture and handle errors
            stderr_output = proc.stderr.read()
            if stderr_output:
                print("Error occurred during MSFinder execution:")
                print(stderr_output)

    except FileNotFoundError as e:
        print(f"Error: MSFinder executable not found in {msfinder_directory}. Ensure the path is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


