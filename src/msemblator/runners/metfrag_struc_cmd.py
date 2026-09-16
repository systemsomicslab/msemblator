import subprocess
import os
import glob
from contextlib import closing
from msemblator.runners.metfrag_psv import read_psv, read_header, validate_row, write_psv
from tqdm import tqdm

def clean_psv_file(psv_file):
    """
    Removes blank rows and normalizes name newlines for MetFrag's PSV reader.
    Rejects malformed rows before overwriting the file.

    Args:
        psv_file (str): Path to the PSV file to clean.

    Returns:
        None
    """
    with closing(read_psv(psv_file)) as reader:
        headers = read_header(reader, psv_file)
        cleaned_rows = [validate_row(row, headers, psv_file, line) for line, row in reader]
    write_psv(psv_file, headers, cleaned_rows)

    print(f"Cleaned PSV file: {psv_file}")

def run_metfrag_command(metfrag_dir):
    """
    Runs MetFrag for each parameter file in the specified directory.

    Args:
        metfrag_dir (str): Directory containing the MetFrag JAR file and parameter files.

    Returns:
        None
    """
    metfrag_dir = os.path.abspath(metfrag_dir)
    # Define the path to the MetFrag JAR file
    metfrag_jar = os.path.join(metfrag_dir, 'MetFragCommandLine-2.5.0.jar')

    # Collect all parameter files in the directory
    parameter_files = glob.glob(os.path.join(metfrag_dir, 'parameter_*.txt'))

    # Validate the databases actually referenced by the parameters, including
    # custom filenames. Relative database paths are resolved against Java's cwd.
    psv_files = set()
    for parameter_file in parameter_files:
        with open(parameter_file, encoding='utf-8-sig') as file:
            for line in file:
                key, separator, value = line.partition('=')
                if separator and key.strip().lower() == 'localdatabasepath':
                    psv_files.add(os.path.join(metfrag_dir, value.strip()))
    for psv_file in sorted(psv_files):
        clean_psv_file(psv_file)

    # Run MetFrag for each parameter file
    with tqdm(total=len(parameter_files), desc="MetFrag Processing", unit="file") as pbar:
        for parameter_file in parameter_files:
            cmd = ["java", "-Dfile.encoding=UTF-8", "-jar", metfrag_jar, parameter_file]

            try:
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
            finally:
                pbar.update(1)
