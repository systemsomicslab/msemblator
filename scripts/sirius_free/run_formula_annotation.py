#!/usr/bin/env python3
"""Run the formula annotation tools of the SIRIUS-free model on an MSP file.

Usage (MS-Emblator environment):
    python scripts/sirius_free/run_formula_annotation.py --input data.msp --output out_dir
    python scripts/sirius_free/run_formula_annotation.py --input data.msp --output out_dir --tools msbuddy msfinder

FIDDLE runs in the MS-Emblator environment (requirements-extras.txt). Only when
msfiddle is not importable here, the script re-runs itself with another
environment's Python (annotation.fiddle.python or annotation.fiddle.conda_env in
the config).

Outputs (out_dir):
    spectra_index.csv                   one row per spectrum (query_id = position in the MSP)
    <tool>/<tool>_candidates.csv        candidates in a common long format
    <tool>/<tool>_status.csv            per-spectrum status (ok, no_candidate, unsupported_adduct, ...)
    <tool>/manifest.json                versions, parameters, input checksum
    annotation.log
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from msemblator.sirius_free import msp_io  # noqa: E402

DEFAULT_CONFIG = ROOT / "config" / "sirius_free_formula.yaml"
TOOLS = ("msbuddy", "msfinder", "fiddle")
logger = logging.getLogger("sirius_free.annotation")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(output_dir / "annotation.log", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def fiddle_python(settings: dict) -> list[str]:
    """Command prefix that starts Python in the FIDDLE environment."""
    if settings.get("python"):
        return [settings["python"]]
    env = settings.get("conda_env")
    if env:
        conda = os.environ.get("CONDA_EXE") or shutil.which("conda") or "conda"
        return [conda, "run", "--no-capture-output", "-n", env, "python"]
    raise RuntimeError("msfiddle is not installed here and annotation.fiddle.python/conda_env is not set")


def run_fiddle_elsewhere(args, settings: dict) -> int:
    command = fiddle_python(settings) + [
        str(Path(__file__).resolve()), "--input", str(args.input), "--output", str(args.output),
        "--config", str(args.config), "--tools", "fiddle", "--no-index"]
    logger.info("Running FIDDLE in its own environment: %s", " ".join(command))
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONUTF8": "1"}
    return subprocess.run(command, env=env, check=False).returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Input MSP file")
    parser.add_argument("--output", required=True, type=Path, help="Output folder")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tools", nargs="+", choices=TOOLS, help="Override annotation.tools of the config")
    parser.add_argument("--no-index", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    setup_logging(args.output)
    config = (yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}).get("annotation", {})
    tools = args.tools or config.get("tools") or list(TOOLS)
    if not args.input.exists():
        logger.error("Input MSP not found: %s", args.input)
        return 2
    if not args.no_index:
        msp_io.write_spectra_index(msp_io.read_msp_records(args.input), args.output / "spectra_index.csv")

    failed = []
    for tool in tools:
        settings = dict(config.get(tool) or {})
        logger.info("=== %s ===", tool)
        try:
            if tool == "fiddle" and importlib.util.find_spec("msfiddle") is None:
                if run_fiddle_elsewhere(args, settings) != 0:
                    raise RuntimeError("FIDDLE subprocess failed; see the log above")
                manifest = json.loads((args.output / "fiddle" / "manifest.json").read_text(encoding="utf-8"))
            else:
                if tool == "msbuddy":
                    from msemblator.sirius_free.annotators import msbuddy_runner as runner
                elif tool == "msfinder":
                    from msemblator.sirius_free.annotators import msfinder_runner as runner
                else:
                    from msemblator.sirius_free.annotators import fiddle_runner as runner
                    settings = {k: v for k, v in settings.items() if k not in ("python", "conda_env")}
                manifest = runner.run(args.input, args.output, settings)
            logger.info("%s finished: %s", tool, manifest["status_counts"])
        except Exception:  # noqa: BLE001 - other tools still run
            logger.exception("%s failed", tool)
            failed.append(tool)
    if failed:
        logger.error("Failed tools: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
