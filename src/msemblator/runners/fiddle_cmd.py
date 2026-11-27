"""Run FIDDLE (msfiddle) formula prediction for the formula workflow.

FIDDLE needs torch and lives in its own environment (envs/fiddle). When msfiddle
is importable here it is called directly; otherwise
msemblator.sirius_free.run_fiddle is started with the FIDDLE environment's
Python (formula_prediction.fiddle.python, or conda_env via `conda run`).
"""
import importlib.util
import json
import os
import shutil
import subprocess

from msemblator.paths import PROJECT_ROOT

# Settings that select the environment; everything else is passed to FIDDLE.
ENVIRONMENT_KEYS = ("python", "conda_env")


def fiddle_python(settings):
    """Command prefix that starts Python in the FIDDLE environment."""
    if settings.get("python"):
        return [str(settings["python"])]
    env = settings.get("conda_env")
    if env:
        conda = os.environ.get("CONDA_EXE") or shutil.which("conda") or "conda"
        return [conda, "run", "--no-capture-output", "-n", str(env), "python"]
    raise RuntimeError(
        "msfiddle is not installed in this environment and neither "
        "formula_prediction.fiddle.python nor conda_env is set"
    )


def run_fiddle(input_msp, output_dir, settings=None):
    """Run FIDDLE on input_msp; candidates are written to <output_dir>/fiddle/fiddle_candidates.csv."""
    settings = dict(settings or {})
    params = {k: v for k, v in settings.items() if k not in ENVIRONMENT_KEYS}
    candidates = os.path.join(output_dir, "fiddle", "fiddle_candidates.csv")

    if importlib.util.find_spec("msfiddle") is not None:
        from msemblator.sirius_free.annotators import fiddle_runner
        fiddle_runner.run(input_msp, output_dir, params)
    else:
        command = fiddle_python(settings) + [
            "-m", "msemblator.sirius_free.run_fiddle",
            "--input", str(input_msp),
            "--output", str(output_dir),
            "--params", json.dumps(params),
        ]
        src_dir = str(PROJECT_ROOT / "src")
        python_path = os.pathsep.join(p for p in (src_dir, os.environ.get("PYTHONPATH")) if p)
        env = {**os.environ, "PYTHONPATH": python_path, "PYTHONNOUSERSITE": "1", "PYTHONUTF8": "1"}
        return_code = subprocess.run(command, env=env, check=False).returncode
        if return_code != 0:
            raise RuntimeError(f"FIDDLE failed with exit code {return_code}.")

    if not os.path.isfile(candidates):
        raise RuntimeError(f"FIDDLE did not write {candidates}")
    return candidates
