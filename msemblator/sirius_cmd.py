import subprocess
import os
import wexpect
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sirius_login(sirius_directory, username, password):
    """
    Log in to Sirius using the command-line interface.

    Args:
        sirius_directory (str): Path to the Sirius executable directory.
        username (str): Sirius username for login.
        password (str): Sirius password for login.

    Returns:
        None
    """
    try:
        os.chdir(sirius_directory)
        login_command = f".\\sirius.exe login -u {username} -p"

        # Use wexpect to handle the login process
        child = wexpect.spawn(f"powershell {login_command}")
        child.expect("Enter value for --password")
        child.sendline(password)
        child.expect("Login successful!", timeout=60)
        logging.info("Login successful!")
    except wexpect.TIMEOUT:
        logging.error("Login process timed out.")
    except wexpect.EOF:
        logging.error("Login failed or process ended unexpectedly.")
    finally:
        child.close()


def run_sirius(sirius_outputdir, sirius_inputdir, sirius_path):
    """
    Runs the Sirius structure prediction tool with updated parameters.

    Args:
        sirius_outputdir (str): Directory to save the output results.
        sirius_inputdir (str): Path to the input MS data file.
        sirius_path (str): Path to the Sirius executable.
        structure_search_db (str): Path to the structure search database.
    """
    command = [
        sirius_path,  # Path to the executable
        "-i", sirius_inputdir,  # Input file path
        "-o", sirius_outputdir,  # Output directory
        "--ignore-formula",
        "config",
        "--IsotopeSettings.filter=true",
        "--FormulaSearchDB=",
        "--Timeout.secondsPerTree=100",
        "--FormulaSettings.enforced=HCNOP",
        "--Timeout.secondsPerInstance=100",
        "--AdductSettings.detectable=[[M+Na]+,[M-H4O2+H]+,[M+H3N+H]+,[M+Cl]-,[M-H]-,[M+H]+,[M-H2O+H]+,[M-H2O-H]-]",
        "--UseHeuristic.mzToUseHeuristicOnly=650",
        "--AlgorithmProfile=qtof",
        "--IsotopeMs2Settings=IGNORE",
        "--MS2MassDeviation.allowedMassDeviation=10.0ppm",
        "--NumberOfCandidatesPerIon=1",
        "--UseHeuristic.mzToUseHeuristic=300",
        "--FormulaSettings.detectable=B,Cl,Br,Se,S",
        "--NumberOfCandidates=5",
        "--AdductSettings.fallback=[[M+Na]+,[M-H4O2+H]+,[M+H3N+H]+,[M+Cl]-,[M-H]-,[M+H]+,[M-H2O+H]+,[M-H2O-H]-]",
        "--ZodiacNumberOfConsideredCandidatesAt300Mz=10",
        "--ZodiacRunInTwoSteps=true",
        "--ZodiacEdgeFilterThresholds.minLocalConnections=10",
        "--ZodiacEdgeFilterThresholds.thresholdFilter=0.95",
        "--ZodiacEpochs.burnInPeriod=2000",
        "--ZodiacEpochs.numberOfMarkovChains=10",
        "--ZodiacNumberOfConsideredCandidatesAt800Mz=50",
        "--ZodiacEpochs.iterations=20000",
        "--RecomputeResults=false",
        "formula",
        "zodiac",
        "write-summaries",
        "--output", sirius_outputdir
    ]

    try:
        # Run the command and capture the output
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Command executed successfully:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("An error occurred while executing the command:")
        print(e.stderr)

