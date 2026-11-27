"""Make FIDDLE usable in the MS-Emblator environment: its packages and model weights.

    python -m msemblator.fiddle_setup            # show what is missing
    python -m msemblator.fiddle_setup --run      # install it

FIDDLE runs in the MS-Emblator environment: environment.yml installs its packages
(requirements-extras.txt: CPU PyTorch, msfiddle). This module installs them into
an environment created before that, and downloads the FIDDLE weights to
%USERPROFILE%\\.msfiddle\\check_point\\v<msfiddle version>, where msfiddle looks for them.

formula_prediction.fiddle.conda_env or .python in config/msemblator_parameter_file.yaml
runs FIDDLE in another environment instead; the weights are then downloaded there.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from msemblator import paths

EXTRAS_FILE = paths.PROJECT_ROOT / "requirements-extras.txt"
INSTRUMENTS = ("qtof", "orbitrap", "timstof")
WEIGHT_FILES = tuple(f"fiddle_{kind}_{instrument}.pt" for kind in ("tcn", "rescore") for instrument in INSTRUMENTS)
WEIGHTS_URL = "https://github.com/JosieHong/FIDDLE/releases"
HERE, OTHER_PYTHON, OTHER_CONDA = "here", "python", "conda"


def console_python() -> str:
    """python.exe of this environment (also when the GUI runs under pythonw.exe)."""
    executable = Path(sys.executable)
    if executable.name.lower() == "pythonw.exe" and (executable.parent / "python.exe").is_file():
        return str(executable.parent / "python.exe")
    return str(executable)


def fiddle_settings() -> dict:
    try:
        with open(paths.PARAMETER_FILE, encoding="utf-8") as handle:
            return (yaml.safe_load(handle) or {}).get("formula_prediction", {}).get("fiddle") or {}
    except OSError:
        return {}


def msfiddle_version() -> str:
    """msfiddle version pinned in requirements-extras.txt (e.g. '2.3.0')."""
    try:
        text = EXTRAS_FILE.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = re.search(r"^\s*msfiddle(?:\[[^\]]*\])?\s*==\s*([\w.]+)", text, re.M)
    return match.group(1) if match else ""


def weights_dir() -> Path:
    return Path.home() / ".msfiddle" / "check_point" / f"v{msfiddle_version()}"


def missing_weights() -> list[str]:
    folder = weights_dir()
    return [str(folder / name) for name in WEIGHT_FILES if not (folder / name).is_file()]


def installed_here() -> bool:
    importlib.invalidate_caches()   # packages may have been installed while the GUI runs
    return all(importlib.util.find_spec(name) is not None for name in ("msfiddle", "torch"))


# --- another environment (only when configured) -------------------------------------

def find_conda() -> str | None:
    """conda executable: CONDA_EXE, PATH, or the installations listed in ~/.conda/environments.txt."""
    candidates = [os.environ.get("CONDA_EXE"), shutil.which("conda")]
    roots = []
    listed = Path.home() / ".conda" / "environments.txt"
    try:
        roots += [Path(line.strip()) for line in listed.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        pass
    roots += [Path(sys.prefix).parent.parent, Path.home() / "anaconda3", Path.home() / "miniconda3",
              Path("C:/anaconda"), Path("C:/ProgramData/anaconda3"), Path("C:/ProgramData/miniconda3")]
    for root in roots:
        candidates += [str(root / "Scripts" / "conda.exe"), str(root / "condabin" / "conda.bat"), str(root / "bin" / "conda")]
    return next((c for c in candidates if c and Path(c).is_file()), None)


def find_env(name: str, conda: str | None = None, ask_conda: bool = True) -> Path | None:
    """Folder of the conda environment `name`, or None (`conda env list` only with ask_conda)."""
    roots = [Path.home() / ".conda" / "envs"]
    for variable in ("CONDA_EXE", "CONDA_PYTHON_EXE"):
        if os.environ.get(variable):
            roots.append(Path(os.environ[variable]).resolve().parents[1] / "envs")
    if os.environ.get("CONDA_PREFIX"):
        prefix = Path(os.environ["CONDA_PREFIX"])
        roots += [prefix.parent, prefix / "envs"]
    if conda:
        roots.append(Path(conda).resolve().parents[1] / "envs")
    for root in roots:
        if (root / name / "python.exe").is_file() or (root / name / "bin" / "python").is_file():
            return root / name
    if conda and ask_conda:
        try:
            result = subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True, timeout=120,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return next((Path(p) for p in json.loads(result.stdout).get("envs", []) if Path(p).name == name), None)
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    return None


# --- status and steps ---------------------------------------------------------------

@dataclass
class FiddleStatus:
    where: str = HERE                 # here | python | conda
    installed: bool = False           # msfiddle and torch importable in this environment
    python: str = ""
    env_name: str = ""
    env_path: Path | None = None
    conda: str | None = None
    weights_missing: list = field(default_factory=list)

    @property
    def environment_ready(self) -> bool:
        if self.where == OTHER_PYTHON:
            return Path(self.python).is_file()
        if self.where == OTHER_CONDA:
            return self.env_path is not None
        return self.installed

    @property
    def ready(self) -> bool:
        return self.environment_ready and not self.weights_missing

    def describe(self) -> list[tuple[str, bool]]:
        """(text, ok) for each part of the set-up."""
        if self.where == OTHER_PYTHON:
            lines = [(f"FIDDLE Python (formula_prediction.fiddle.python): {self.python}", self.environment_ready)]
        elif self.where == OTHER_CONDA:
            lines = [(f"FIDDLE conda environment '{self.env_name}' (formula_prediction.fiddle.conda_env): "
                      + (str(self.env_path) if self.env_path else "not found"), self.environment_ready)]
        else:
            lines = [(f"msfiddle and PyTorch in this environment ({sys.prefix}): "
                      + ("installed" if self.installed else "not installed"), self.installed)]
        lines.append((f"Weights in {weights_dir()}: "
                      + ("all found" if not self.weights_missing else f"{len(self.weights_missing)} file(s) missing"),
                      not self.weights_missing))
        return lines


def status(ask_conda: bool = True) -> FiddleStatus:
    settings = fiddle_settings()
    weights = missing_weights()
    if settings.get("python"):
        return FiddleStatus(OTHER_PYTHON, False, str(settings["python"]), weights_missing=weights)
    if settings.get("conda_env"):
        name = str(settings["conda_env"])
        conda = find_conda()
        return FiddleStatus(OTHER_CONDA, False, "", name, find_env(name, conda, ask_conda), conda, weights)
    return FiddleStatus(HERE, installed_here(), weights_missing=weights)


@dataclass
class Step:
    title: str
    command: list


def setup_steps(current: FiddleStatus) -> list[Step]:
    """Commands that make FIDDLE usable; the last one checks the result."""
    steps = []
    if current.where == OTHER_PYTHON:
        if not current.environment_ready:
            raise RuntimeError(f"formula_prediction.fiddle.python does not exist: {current.python}")
        runner = [current.python]
    elif current.where == OTHER_CONDA:
        if not current.conda or current.env_path is None:
            raise RuntimeError(f"The conda environment '{current.env_name}' (formula_prediction.fiddle.conda_env) "
                               "was not found. Clear that setting to run FIDDLE in the MS-Emblator environment.")
        runner = [current.conda, "run", "--no-capture-output", "-n", current.env_name, "python"]
    else:
        runner = [console_python()]
        if not current.installed:
            steps.append(Step(f"Install FIDDLE into this environment (CPU PyTorch, msfiddle {msfiddle_version()})",
                              runner + ["-m", "pip", "install", "-r", str(EXTRAS_FILE)]))
    if current.weights_missing:   # weights are per user, shared by every environment
        steps.append(Step("Download the FIDDLE model weights (about 0.25 GB)", runner + ["-m", "msfiddle", "download"]))
    steps.append(Step("Check that msfiddle and torch can be imported",
                      runner + ["-c", "import msfiddle, torch; print('msfiddle and torch', torch.__version__, 'OK')"]))
    return steps


def run_steps(steps, on_line=None, cancel=None, on_process=None) -> tuple[bool, str]:
    """Run the steps one after another; returns (success, title of the step that failed or was cancelled)."""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1"}
    for number, step in enumerate(steps, 1):
        if cancel is not None and cancel.is_set():
            return False, step.title
        if on_line:
            on_line(f"[{number}/{len(steps)}] {step.title}", True)
        process = subprocess.Popen(step.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=env, cwd=str(paths.PROJECT_ROOT),
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if on_process:
            on_process(process)
        for raw in iter(process.stdout.readline, b""):
            if on_line:
                from msemblator.gui.backend import decode_line
                on_line(decode_line(raw), False)
        if process.wait() != 0:
            return False, step.title
    return True, ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="install what is missing")
    args = parser.parse_args(argv)
    current = status()
    print("\n".join(("ok   " if ok else "MISS ") + text for text, ok in current.describe()))
    if current.ready and not args.run:
        print("FIDDLE is ready.")
        return 0
    steps = setup_steps(current)
    if not args.run:
        print("\nSteps (run with --run):")
        for step in steps:
            print(f"  {step.title}\n    {' '.join(step.command)}")
        return 0
    ok, failed = run_steps(steps, lambda text, is_step: print(text, flush=True))
    print("FIDDLE is ready." if ok else f"Failed: {failed}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
