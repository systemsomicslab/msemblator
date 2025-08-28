import subprocess
import os
import glob
import csv
from tqdm import tqdm

def clean_psv_file(psv_file):
    """
    Cleans a PSV (Pipe Separated Values) file by removing empty rows.

    Args:
        psv_file (str): Path to the PSV file to clean.

    Returns:
        None
    """
    cleaned_rows = []

    # Read the PSV file and filter out empty rows
    with open(psv_file, 'r') as file:
        reader = csv.reader(file, delimiter='|')
        headers = next(reader)  # Keep the header row
        cleaned_rows.append(headers)

        for row in reader:
            if not row or all(cell.strip() == '' for cell in row):
                continue  # Skip empty rows
            cleaned_rows.append(row)

    # Write the cleaned data back to the same file
    with open(psv_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='|')
        writer.writerows(cleaned_rows)

    print(f"Cleaned PSV file: {psv_file}")

def run_metfrag_command(metfrag_dir):
    """
    Runs MetFrag for each parameter file in the specified directory.

    Args:
        metfrag_dir (str): Directory containing the MetFrag JAR file and parameter files.

    Returns:
        None
    """
    # Define the path to the MetFrag JAR file
    metfrag_jar = os.path.join(metfrag_dir, 'MetFragCommandLine-2.5.0.jar')

    # Collect all parameter files in the directory
    parameter_files = glob.glob(os.path.join(metfrag_dir, 'parameter_*.txt'))

    # Clean all library files before running MetFrag
    psv_files = glob.glob(os.path.join(metfrag_dir, '*_library.txt'))
    for psv_file in psv_files:
        clean_psv_file(psv_file)

    # Run MetFrag for each parameter file
    with tqdm(total=len(parameter_files), desc="MetFrag Processing", unit="file") as pbar:
        for parameter_file in parameter_files:
            cmd = ["java", "-jar", str(metfrag_jar), str(parameter_file)]

            try:
                # Execute the MetFrag command via PowerShell
                with subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=metfrag_dir
                ) as proc:
                    stdout, stderr = proc.communicate()
                    # Handle errors and warnings
                    if proc.returncode != 0:
                        print(f"Error processing {parameter_file}:\n{stderr}")

            except Exception as e:
                print(f"Exception occurred while running MetFrag for {parameter_file}: {e}")

            pbar.update(1)