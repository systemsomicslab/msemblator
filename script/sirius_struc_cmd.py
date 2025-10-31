import os
import subprocess
import wexpect
import sys

def sirius_login(sirius_directory, username, password):
    """
    Logs into the Sirius tool using the provided username and password.

    Args:
        sirius_directory (str): Directory where Sirius is installed.
        username (str): Sirius account username.
        password (str): Sirius account password.

    Returns:
        None
    """
    # Change to the Sirius installation directory
    os.chdir(sirius_directory)
    
    # Construct the login command
    login_command = f'.\\sirius.exe login -u {username} -p'

    # Spawn a PowerShell process with wexpect to handle the login interaction
    child = wexpect.spawn(f'powershell {login_command}')
    child.logfile = sys.stdout  # Log output to the console

    try:
        # Expect the password prompt
        child.expect('Enter value for --password')
        child.sendline(password)  # Send the password

        # Expect a success message
        child.expect('Login successful!', timeout=30)
        print("Login successful!")
    except wexpect.EOF:
        print("Login failed or process ended unexpectedly.")
    finally:
        child.close()

def run_sirius_struc(sirius_outputdir, sirius_inputdir, sirius_path, structure_search_db):
    """
    Runs the Sirius structure prediction tool with the specified parameters.

    Args:
        sirius_outputdir (str): Directory to save the output results.
        sirius_inputdir (str): Path to the input MS data file.
        sirius_path (str): Path to the Sirius executable.
        structure_search_db (str): Path to the structure search database.

    Returns:
        None
    """
    # Construct the command for running Sirius with the necessary parameters
    command = [
        sirius_path,
        "-i", sirius_inputdir,
        "-o", sirius_outputdir,
        "config",
        "--IsotopeSettings.filter=true",
        "--FormulaSearchDB=",
        "--Timeout.secondsPerTree=0",
        "--FormulaSettings.enforced=HCNOPSFClBrI",
        "--Timeout.secondsPerInstance=600",
        "--AdductSettings.detectable=[M-H]-,[M+H]+,[M-H2O-H]-,[M-H2O+H]+,[M-H4O2+H]+,[M+H3N+H]+",
        "--UseHeuristic.mzToUseHeuristicOnly=650",
        "--AlgorithmProfile=qtof",
        "--IsotopeMs2Settings=IGNORE",
        "--MS2MassDeviation.allowedMassDeviation=10.0ppm",
        "--NumberOfCandidatesPerIon=1",
        "--UseHeuristic.mzToUseHeuristic=300",
        "--FormulaSettings.detectable=B,Cl,Br,Se,S",
        "--NumberOfCandidates=10",
        "--ZodiacNumberOfConsideredCandidatesAt300Mz=10",
        "--ZodiacRunInTwoSteps=true",
        "--ZodiacEdgeFilterThresholds.minLocalConnections=10",
        "--ZodiacEdgeFilterThresholds.thresholdFilter=0.95",
        "--ZodiacEpochs.burnInPeriod=2000",
        "--ZodiacEpochs.numberOfMarkovChains=10",
        "--ZodiacNumberOfConsideredCandidatesAt800Mz=50",
        "--ZodiacEpochs.iterations=20000",
        "--AdductSettings.enforced=",
        "--AdductSettings.fallback=[M+H]+,[M-H]-,[M-H2O+H]+,[M-H2O-H]-,[M+CH3OH+H]+,[M+HCOO]-",
        "--FormulaResultThreshold=true",
        "--InjectElGordoCompounds=true",
        f"--StructureSearchDB={structure_search_db}",
        "--RecomputeResults=false",
        "formula",
        "fingerprint",
        "structure",
        "canopus",
        "write-summaries",
        "--output", sirius_outputdir
    ]

    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end="")
            proc.wait()

    except Exception as e:
        print(f"An error occurred during SIRIUS execution: {e}")
