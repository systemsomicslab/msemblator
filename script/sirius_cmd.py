import subprocess
import os
import wexpect
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_ADDUCTS = (
    "[M+Na]+,[M-H4O2+H]+,[M+H3N+H]+,[M+Cl]-,[M-H]-,[M+H]+,"
    "[M-H2O+H]+,[M-H2O-H]-"
)


def _script_dir():
    return os.path.abspath(os.path.dirname(__file__))


def _resolve_sirius6_path(sirius_path=None):
    """Return the bundled SIRIUS 6 executable, falling back to the provided path."""
    bundled_path = os.path.join(_script_dir(), "sirius6", "sirius.exe")
    if os.path.exists(bundled_path):
        return bundled_path
    return sirius_path


def _child_output(child):
    output = child.before or ""
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    return output.strip()


def _ensure_summary_has_rows(summary_path, summary_name):
    if not os.path.exists(summary_path):
        raise RuntimeError(f"SIRIUS did not write {summary_name}: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as file:
        line_count = sum(1 for line in file if line.strip())
    if line_count <= 1:
        raise RuntimeError(f"SIRIUS wrote {summary_name}, but it contains no candidate rows.")


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
    sirius_path = _resolve_sirius6_path(os.path.join(sirius_directory, "sirius.exe"))
    try:
        sirius_directory = os.path.dirname(sirius_path)
        login_command = f".\\sirius.exe login -u {username} -p"

        # Use wexpect to handle the login process
        child = wexpect.spawn(f"powershell {login_command}", cwd=sirius_directory, echo=False)
        child.expect("Enter value for --password")
        child.sendline(password)
        child.expect("Login successful!", timeout=60)
        logging.info("Login successful!")
    except wexpect.TIMEOUT:
        message = _child_output(child)
        logging.error("Login process timed out.")
        raise RuntimeError(f"SIRIUS login timed out. Output: {message}")
    except wexpect.EOF:
        message = _child_output(child)
        logging.error("Login failed or process ended unexpectedly.")
        raise RuntimeError(f"SIRIUS login failed or process ended unexpectedly. Output: {message}")
    finally:
        if "child" in locals():
            child.close()


def run_sirius(sirius_outputdir, sirius_inputdir, sirius_path, config):
    """
    Runs the Sirius structure prediction tool with updated parameters.

    Args:
        sirius_outputdir (str): Directory to save the output results.
        sirius_inputdir (str): Path to the input MS data file.
        sirius_path (str): Path to the Sirius executable.
        structure_search_db (str): Path to the structure search database.
    """
    ms1 = config['formula_prediction']['sirius']['MS1']
    ms2 = config['formula_prediction']['sirius']['MS2_ppm']
    atoms = config['formula_prediction']['sirius']['halogen']
    if atoms == True:
        atoms_detectable = "B,Cl,Br,Se,S"
        atoms_enforced = "HCNOF[5]PI[5]"
    else:
        atoms_detectable = "B,Se,S"
        atoms_enforced = "HCNOP"


    sirius_path = _resolve_sirius6_path(sirius_path)
    os.makedirs(sirius_outputdir, exist_ok=True)
    sirius_project = os.path.join(sirius_outputdir, "sirius_project.sirius")

    command = [
        sirius_path,
        "--ignore-formula",
        "-i", sirius_inputdir,
        "-o", sirius_project,
        "formulas",
        "--tree-timeout=100",
        "--compound-timeout=100",
        f"--elements-enforced={atoms_enforced}",
        f"--elements-considered={atoms_detectable}",
        f"--adducts-considered={DEFAULT_ADDUCTS}",
        "--heuristic-only=650",
        "--heuristic=300",
        f"--profile={ms1}",
        f"--ppm-max-ms2={ms2}",
        "--candidates-per-ionization=1",
        "--candidates=100",
        "summaries",
        "--no-top-hit-summary=false",
        "--top-k-summary=100",
        "--output", sirius_outputdir
    ]

    try:
        output_lines = []
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                output_lines.append(line)
                print(line, end="")
            return_code = proc.wait()
        output_text = "".join(output_lines)
        if return_code != 0:
            raise RuntimeError(f"SIRIUS execution failed with exit code {return_code}.")
        if "Error When Executing ToolChain" in output_text or "Unexpected Error!" in output_text:
            raise RuntimeError("SIRIUS execution failed. See output above for details.")
        _ensure_summary_has_rows(
            os.path.join(sirius_outputdir, "formula_identifications_top-100.tsv"),
            "formula_identifications_top-100.tsv"
        )

    except Exception as e:
        print(f"An error occurred during SIRIUS execution: {e}")
        raise
