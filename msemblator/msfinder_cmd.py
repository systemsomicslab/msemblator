import subprocess
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_msfinder(msfinder_directory, input_path, output_path, method_path):
    """
    Run MS-FINDER using the command-line interface.

    Args:
        msfinder_directory (str): Path to the directory containing MS-FINDER executable.
        input_path (str): Path to the input file or folder.
        output_path (str): Path to the output folder.
        method_path (str): Path to the MS-FINDER method file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the MS-FINDER executable is not found.
        Exception: If an unexpected error occurs during execution.
    """
    try:
        # Ensure the MS-FINDER executable exists
        msfinder_exe = 'MsfinderConsoleApp.exe'
        msfinder_exe_path = fr"{msfinder_directory}\{msfinder_exe}"

        if not os.path.exists(msfinder_exe_path):
            raise FileNotFoundError(f"MS-FINDER executable not found at: {msfinder_exe_path}")

        # Construct the PowerShell command
        command = (
            f'$env:PATH += ";{msfinder_directory}"; '
            f'{msfinder_exe} predict -i "{input_path}" -o "{output_path}" -m "{method_path}"'
        )

        # Run the command and capture output
        with subprocess.Popen(
            ["powershell", "-Command", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        ) as proc:
            # Process stdout in real-time
            for line in proc.stdout:
                print(line, end="")  # Avoid double newlines
                logging.info(line.strip())

            # Handle stderr
            stderr_output = proc.stderr.read()
            if stderr_output:
                logging.error(f"MS-FINDER Error: {stderr_output}")
                raise Exception(f"MS-FINDER execution failed with error: {stderr_output.strip()}")

    except FileNotFoundError as e:
        logging.error(e)
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while running MS-FINDER: {e}")
        raise
