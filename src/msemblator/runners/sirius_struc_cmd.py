import os
import subprocess
import threading
import time
import wexpect

DEFAULT_ADDUCTS = (
    "[M+H]+,[M-H]-,[M-H2O+H]+,[M-H2O-H]-,[M+Na]+,[M+Cl]-,"
    "[M+H3N+H]+,[M-H4O2+H]+,[M+CH3OH+H]+,[M+HCOO]-"
)
DEFAULT_STRUCTURE_DB_NAME = "sirius_structure_db"
SIRIUS_COMMAND_TIMEOUT_SECONDS = 1800
SIRIUS_INTER_STEP_DELAY_SECONDS = 10


def _script_dir():
    return os.path.abspath(os.path.dirname(__file__))


def _resolve_sirius6_path(sirius_path=None):
    bundled_path = os.path.join(_script_dir(), "sirius6", "sirius.exe")
    if os.path.exists(bundled_path):
        return bundled_path
    return sirius_path


def _child_output(child):
    output = child.before or ""
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    return output.strip()


def _resolve_structure_db(structure_search_db=None):
    bundled_db = os.path.abspath(
        os.path.join(_script_dir(), "..", "library", "sirius_structure_db.siriusdb")
    )
    if os.path.exists(bundled_db):
        return bundled_db
    return structure_search_db


def _run_sirius_capture(command, cwd):
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=cwd
    ) as proc:
        output_text = proc.stdout.read()
        return_code = proc.wait()
    return return_code, output_text


def _custom_db_info(sirius_path, db_name=DEFAULT_STRUCTURE_DB_NAME):
    command = [
        sirius_path,
        "custom-db",
        "show",
        f"--db={db_name}"
    ]
    return_code, output_text = _run_sirius_capture(command, os.path.dirname(sirius_path))
    if return_code != 0 or db_name not in output_text or "Unexpected Error!" in output_text:
        return None

    info = {}
    for line in output_text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    return info


def _remove_custom_db_registration(sirius_path, db_name=DEFAULT_STRUCTURE_DB_NAME):
    command = [
        sirius_path,
        "custom-db",
        "remove",
        f"--db={db_name}"
    ]
    return_code, output_text = _run_sirius_capture(command, os.path.dirname(sirius_path))
    if return_code != 0 or "Unexpected Error!" in output_text:
        raise RuntimeError(f"Failed to remove SIRIUS custom database registration: {db_name}")


def _ensure_custom_db_registered(sirius_path, structure_search_db):
    if not structure_search_db or not os.path.isfile(structure_search_db):
        raise FileNotFoundError(f"SIRIUS structure database was not found: {structure_search_db}")

    db_info = _custom_db_info(sirius_path)
    if db_info and os.path.abspath(db_info.get("Location", "")) == os.path.abspath(structure_search_db):
        print(f"SIRIUS custom database '{DEFAULT_STRUCTURE_DB_NAME}' is already registered.")
        return
    if db_info:
        print(f"Updating SIRIUS custom database '{DEFAULT_STRUCTURE_DB_NAME}' registration.")
        _remove_custom_db_registration(sirius_path)

    command = [
        sirius_path,
        "custom-db",
        "add",
        f"--location={structure_search_db}"
    ]
    output_lines = []
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=os.path.dirname(sirius_path)
    ) as proc:
        for line in proc.stdout:
            output_lines.append(line)
            print(line, end="")
        return_code = proc.wait()

    output_text = "".join(output_lines)
    if "already exists" in output_text:
        return
    if return_code != 0 or "Unexpected Error!" in output_text:
        raise RuntimeError(f"Failed to register SIRIUS custom database: {structure_search_db}")


def _run_sirius_command(command, cwd, step_name, timeout_seconds=SIRIUS_COMMAND_TIMEOUT_SECONDS):
    print(f"SIRIUS {step_name} start", flush=True)
    output_lines = []
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=cwd
    )

    def stream_output():
        for line in proc.stdout:
            output_lines.append(line)
            print(line, end="", flush=True)

    output_thread = threading.Thread(target=stream_output, daemon=True)
    output_thread.start()

    try:
        return_code = proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        output_thread.join(timeout=5)
        raise RuntimeError(
            f"SIRIUS {step_name} timed out after {timeout_seconds} seconds. "
            "If this happens during fingerprints, CSI:FingerID web-service processing is not returning."
        )
    output_thread.join(timeout=5)

    output_text = "".join(output_lines)
    if return_code != 0:
        tail = "\n".join(output_text.splitlines()[-40:])
        raise RuntimeError(f"SIRIUS {step_name} failed with exit code {return_code}.\nLast SIRIUS output:\n{tail}")
    if "Error When Executing ToolChain" in output_text or "Unexpected Error!" in output_text:
        tail = "\n".join(output_text.splitlines()[-40:])
        raise RuntimeError(f"SIRIUS {step_name} failed. Last SIRIUS output:\n{tail}")
    print(f"SIRIUS {step_name} complete", flush=True)


def _pause_after_sirius_step(step_name):
    print(f"Waiting {SIRIUS_INTER_STEP_DELAY_SECONDS} seconds after SIRIUS {step_name}.", flush=True)
    time.sleep(SIRIUS_INTER_STEP_DELAY_SECONDS)


def _ensure_summary_has_rows(summary_path, summary_name):
    if not os.path.exists(summary_path):
        raise RuntimeError(f"SIRIUS did not write {summary_name}: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as file:
        line_count = sum(1 for line in file if line.strip())
    if line_count <= 1:
        raise RuntimeError(
            f"SIRIUS wrote {summary_name}, but it contains no candidate rows. "
            "The CSI:FingerID fingerprint/structure step likely did not finish or returned no hits."
        )


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
    sirius_path = _resolve_sirius6_path(os.path.join(sirius_directory, "sirius.exe"))
    sirius_directory = os.path.dirname(sirius_path)
    
    # Construct the login command
    login_command = f'.\\sirius.exe login -u {username} -p'

    # Spawn a PowerShell process with wexpect to handle the login interaction
    child = wexpect.spawn(f'powershell {login_command}', cwd=sirius_directory, echo=False)

    try:
        # Expect the password prompt
        child.expect('Enter value for --password')
        child.sendline(password)  # Send the password

        # Expect a success message
        child.expect('Login successful!', timeout=30)
        print("Login successful!")
    except wexpect.TIMEOUT:
        message = _child_output(child)
        print("Login process timed out.")
        raise RuntimeError(f"SIRIUS login timed out. Output: {message}")
    except wexpect.EOF:
        message = _child_output(child)
        print("Login failed or process ended unexpectedly.")
        raise RuntimeError(f"SIRIUS login failed or process ended unexpectedly. Output: {message}")
    finally:
        child.close()


def run_sirius_struc(sirius_outputdir, sirius_inputdir, sirius_path, structure_search_db, config):
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
    ms2 = config['structure_prediction']['sirius']['MS2_ppm']
    sirius_path = _resolve_sirius6_path(sirius_path)
    structure_search_db = _resolve_structure_db(structure_search_db)
    os.makedirs(sirius_outputdir, exist_ok=True)
    sirius_project = os.path.join(sirius_outputdir, "sirius_project.sirius")
    _ensure_custom_db_registered(sirius_path, structure_search_db)
    
    workflow_command = [
        sirius_path,
        "-i", sirius_inputdir,
        "-o", sirius_project,
        "config",
        "--IsotopeSettings.filter=true",
        "--FormulaSearchDB=",
        "--Timeout.secondsPerTree=0",
        "--Timeout.secondsPerInstance=300",
        "--FormulaSettings.enforced=H,C,N,O,P",
        "--FormulaSettings.detectable=B,Mg,S,Cl,Fe,Zn,Se,Br",
        "--FormulaSettings.fallback=S",
        f"--AdductSettings.detectable={DEFAULT_ADDUCTS}",
        "--AdductSettings.fallback=[M+H]+,[M+Na]+,[M+K]+,[M-H]-",
        "--AdductSettings.enforced=",
        "--PossibleAdductSwitches=[M+Na]+:[M+H]+,[M+K]+:[M+H]+,[M+Cl]-:[M-H]-",
        "--UseHeuristic.useHeuristicAboveMz=300",
        "--UseHeuristic.useOnlyHeuristicAboveMz=650",
        f"--AlgorithmProfile=qtof",
        "--IsotopeMs2Settings=IGNORE",
        f"--MS2MassDeviation.allowedMassDeviation={ms2}ppm",
        f"--MS2MassDeviation.standardMassDeviation={ms2}ppm",
        "--NumberOfCandidatesPerIonization=1",
        "--NumberOfCandidates=100",
        "--NumberOfStructureCandidates=100",
        f"--StructureSearchDB={DEFAULT_STRUCTURE_DB_NAME}",
        "formulas",
        "fingerprints",
        "classes",
        "structures",
        "summaries",
        "--top-k-summary=100",
        "--output", sirius_outputdir
    ]

    try:
        cwd = os.path.dirname(sirius_path)
        _run_sirius_command(workflow_command, cwd, "workflow")
        _ensure_summary_has_rows(
            os.path.join(sirius_outputdir, "structure_identifications_top-100.tsv"),
            "structure_identifications_top-100.tsv"
        )

    except Exception as e:
        print(f"An error occurred during SIRIUS execution: {e}")
        raise
