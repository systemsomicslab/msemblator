"""Create, update and use the MS-Emblator environment (standard library only).

Started by msemblator_gui.cmd and msemblator.cmd (scripts/find_runtime.cmd picks
the kind of environment), so it works before the environment exists:

    python scripts/msemblator_env.py --conda <conda.exe> COMMAND    conda environment (environment.yml)
    python scripts/msemblator_env.py --venv COMMAND                 Python venv in .venv

    COMMAND: setup | gui | cli ARGS | status

Everything (MS-Emblator, msbuddy, FIDDLE with CPU PyTorch, GLACIER) runs in this
one environment. It is created on first use, and updated when environment.yml,
requirements.txt, requirements-extras.txt, envs/glacier/requirements-glacier.txt
or pyproject.toml change (their SHA-256 is kept in the environment folder,
msemblator_env.json). The FIDDLE model weights are downloaded once.

conda brings Java 21 for MetFrag along; with a venv, Java 21 or newer must be
installed separately (only MetFrag needs it).

MSEMBLATOR_ENV_NAME (conda) and MSEMBLATOR_VENV_DIR (venv) choose another
environment, e.g. for tests.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / "environment.yml"
EXTRAS_FILE = ROOT / "requirements-extras.txt"
TRACKED = ("environment.yml", "requirements.txt", "requirements-extras.txt",
           "envs/glacier/requirements-glacier.txt", "pyproject.toml")
STATE_FILE = "msemblator_env.json"
VENV_PYTHONS = ((3, 10), (3, 11), (3, 12))   # numpy 1.26 and RDKit 2023.9 have no wheels for 3.13
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DETACHED = NO_WINDOW | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


def say(text=""):
    print(text, flush=True)


def fingerprint() -> str:
    digest = hashlib.sha256()
    for name in TRACKED:
        path = ROOT / name
        digest.update(name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") if path.is_file() else b"-")
    return digest.hexdigest()


def clean_environment() -> dict:
    """Environment variables for the MS-Emblator processes: no PYTHONPATH / PYTHONHOME
    from other programs, no user site-packages."""
    env = {k: v for k, v in os.environ.items() if k.upper() not in ("PYTHONPATH", "PYTHONHOME")}
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONUTF8": "1"})
    return env


def run(command) -> int:
    """Run a command with its output shown in this console."""
    return subprocess.run(command, cwd=str(ROOT), env=clean_environment()).returncode


def read_state(folder: Path) -> dict:
    try:
        return json.loads((folder / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_state(folder: Path, kind: str):
    (folder / STATE_FILE).write_text(json.dumps({
        "fingerprint": fingerprint(), "kind": kind, "repository": str(ROOT),
        "updated_at": datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")


def msfiddle_weights_missing() -> bool:
    match = re.search(r"^\s*msfiddle(?:\[[^\]]*\])?\s*==\s*([\w.]+)", EXTRAS_FILE.read_text(encoding="utf-8"), re.M)
    if not match:
        return False
    folder = Path.home() / ".msfiddle" / "check_point" / f"v{match.group(1)}"
    names = [f"fiddle_{k}_{i}.pt" for k in ("tcn", "rescore") for i in ("qtof", "orbitrap", "timstof")]
    return not all((folder / n).is_file() for n in names)


class CondaRuntime:
    """The conda environment of environment.yml (Python, Java 21 and pip packages)."""

    kind = "conda"

    def __init__(self, conda: str):
        self.conda = conda
        self.name = os.environ.get("MSEMBLATOR_ENV_NAME") or self._name_in_env_file()

    @staticmethod
    def _name_in_env_file() -> str:
        match = re.search(r"^name:\s*(\S+)", ENV_FILE.read_text(encoding="utf-8"), re.M)
        return match.group(1) if match else "msemblator"

    def find(self) -> Path | None:
        result = subprocess.run([self.conda, "env", "list", "--json"], capture_output=True, text=True,
                                env=clean_environment(), creationflags=NO_WINDOW)
        try:
            envs = json.loads(result.stdout).get("envs", [])
        except ValueError:
            return None
        return next((Path(p) for p in envs if Path(p).name.lower() == self.name.lower()), None)

    def describe(self) -> str:
        return f"conda environment '{self.name}'"

    def python(self, *args, windowed=False) -> list:
        return [self.conda, "run", "--no-capture-output", "-n", self.name, "pythonw" if windowed else "python", *args]

    def ensure(self) -> Path:
        env = self.find()
        if env is None:
            say(f"Creating the MS-Emblator environment '{self.name}' (conda, environment.yml).")
            say("This happens once and takes about 10 minutes (about 3 GB of downloads and disk).")
            say()
            if run([self.conda, "env", "create", "-f", str(ENV_FILE), "-n", self.name]) != 0:
                raise SystemExit("Creating the environment failed (see above). Starting again continues where it stopped.")
            env = self.find()
            if env is None:
                raise SystemExit(f"The environment '{self.name}' was not found after it was created.")
        elif read_state(env).get("fingerprint") != fingerprint():
            say(f"Updating the MS-Emblator environment '{self.name}' (environment files changed).")
            say("Close other MS-Emblator windows first. This can take several minutes.")
            say()
            if run([self.conda, "env", "update", "-f", str(ENV_FILE), "-n", self.name]) != 0:
                raise SystemExit("Updating the environment failed (see above). "
                                 "Close every MS-Emblator window and start again.")
        write_state(env, self.kind)
        return env


class VenvRuntime:
    """A Python venv (default .venv in the repository) with the same pip packages; no conda needed."""

    kind = "venv"

    def __init__(self):
        self.folder = Path(os.environ.get("MSEMBLATOR_VENV_DIR") or ROOT / ".venv")

    def _bin(self, name) -> Path:
        scripts = self.folder / ("Scripts" if os.name == "nt" else "bin")
        return scripts / (f"{name}.exe" if os.name == "nt" else name)

    def find(self) -> Path | None:
        return self.folder if self._bin("python").is_file() else None

    def describe(self) -> str:
        return f"Python venv {self.folder}"

    def python(self, *args, windowed=False) -> list:
        executable = self._bin("pythonw") if windowed and self._bin("pythonw").is_file() else self._bin("python")
        return [str(executable), *args]

    def ensure(self) -> Path:
        if sys.version_info[:2] not in VENV_PYTHONS:
            raise SystemExit(f"Python {sys.version.split()[0]} cannot be used for the MS-Emblator venv: "
                             "Python 3.12 is needed (3.10 and 3.11 also work). Install it from python.org; "
                             "the Python launcher 'py -3.12' is then used automatically.")
        fresh = self.find() is None
        if fresh:
            say(f"Creating the MS-Emblator environment (Python {sys.version.split()[0]} venv) in {self.folder}.")
            say("This happens once and takes several minutes (about 2 GB of downloads and disk).")
            say()
            if run([sys.executable, "-m", "venv", str(self.folder)]) != 0:
                raise SystemExit("Creating the venv failed (see above).")
        if fresh or read_state(self.folder).get("fingerprint") != fingerprint():
            if not fresh:
                say(f"Updating the MS-Emblator environment in {self.folder} (environment files changed).")
                say("Close other MS-Emblator windows first.")
            python = str(self._bin("python"))
            steps = [[python, "-m", "pip", "install", "--upgrade", "pip"],
                     [python, "-m", "pip", "install", "-e", str(ROOT), "-r", str(EXTRAS_FILE)]]
            for step in steps:
                if run(step) != 0:
                    raise SystemExit(
                        "Installing the packages failed (see above). Starting again continues.\n"
                        "If the error says a file name is too long (WinError 206), PyTorch's deep folders do not fit "
                        "under this path: set MSEMBLATOR_VENV_DIR to a short folder such as C:\\msemblator-venv, "
                        "or enable long paths in Windows, delete the venv folder and start again.")
        write_state(self.folder, self.kind)
        self._check_java()
        return self.folder

    @staticmethod
    def _check_java():
        """MetFrag needs Java 21+; conda brings it, a venv does not."""
        java = shutil.which("java")
        if os.environ.get("JAVA_HOME"):
            candidate = Path(os.environ["JAVA_HOME"]) / "bin" / ("java.exe" if os.name == "nt" else "java")
            java = str(candidate) if candidate.is_file() else java
        if not java:
            say("Note: Java was not found. MetFrag (structure elucidation) needs Java 21 or newer, "
                "e.g. from https://adoptium.net/.")
            return
        try:
            output = subprocess.run([java, "-version"], capture_output=True, text=True, creationflags=NO_WINDOW).stderr
            major = int(re.search(r'version "(\d+)', output).group(1))
        except (OSError, AttributeError, ValueError):
            return
        if major < 21:
            say(f"Note: Java {major} was found; MetFrag needs Java 21 or newer (https://adoptium.net/).")


def ensure(runtime) -> Path:
    """Create or update the environment when needed; download the FIDDLE weights once."""
    folder = runtime.ensure()
    if msfiddle_weights_missing():
        say("Downloading the FIDDLE model weights (about 0.25 GB, once).")
        if run(runtime.python("-m", "msfiddle", "download")) != 0:
            say("The FIDDLE weights could not be downloaded; FIDDLE will not run until they are "
                "(Settings > Set up FIDDLE in the GUI).")
    return folder


def start_gui(runtime):
    """Start the GUI inside the environment and return at once (the console can close)."""
    subprocess.Popen(runtime.python("-m", "msemblator.gui", windowed=True), cwd=str(ROOT), env=clean_environment(),
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=DETACHED)


def status(runtime) -> int:
    folder = runtime.find()
    say(f"environment : {runtime.describe()} -> {folder or 'not created yet'}")
    if folder is not None:
        up_to_date = read_state(folder).get("fingerprint") == fingerprint()
        say(f"up to date  : {'yes' if up_to_date else 'no (updated at the next start)'}")
    say(f"FIDDLE weights: {'missing (downloaded at the next start)' if msfiddle_weights_missing() else 'found'}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--conda"] and len(argv) >= 3:
        runtime, argv = CondaRuntime(argv[1]), argv[2:]
    elif argv[:1] == ["--venv"] and len(argv) >= 2:
        runtime, argv = VenvRuntime(), argv[1:]
    else:
        say(__doc__)
        return 2
    command, rest = argv[0], argv[1:]
    if command not in ("setup", "gui", "cli", "status"):
        say(__doc__)
        return 2
    if command == "status":
        return status(runtime)
    ensure(runtime)
    if command == "setup":
        say(f"The {runtime.describe()} is ready.")
        return 0
    if command == "gui":
        say("Starting the MS-Emblator GUI ...")
        start_gui(runtime)
        return 0
    return run(runtime.python("-m", "msemblator.cli.msemblator", *rest))


if __name__ == "__main__":
    sys.exit(main())
