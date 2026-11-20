"""Run MS-FINDER formula prediction on an MSP file (Windows console app).

Reuses runners/msfinder_cmd.run_formula_msfinder and the formula parameter
template of the default workflow. The template is copied into the output folder
before tolerances are applied, so config/msfinder is never modified.

The per-spectrum MSP files written for MS-FINDER contain only NAME, precursor
m/z, precursor type, ion mode and peaks: answer fields (FORMULA, SMILES,
INCHIKEY) are left out so that they cannot influence the prediction.
"""
from __future__ import annotations

import datetime as _dt
import glob
import logging
import shutil
import time
from pathlib import Path

import pandas as pd

from msemblator.sirius_free import msp_io
from msemblator.sirius_free.annotators import common

logger = logging.getLogger(__name__)

TOOL = "msfinder"

# Precursor type written to the MSP for MS-FINDER. Formate is written as
# [M+FA-H]-, the name used by MS-DIAL / MS-FINDER for this adduct.
MSFINDER_ADDUCTS = {
    "[M+H]+": "[M+H]+",
    "[M+NH4]+": "[M+NH4]+",
    "[M+Na]+": "[M+Na]+",
    "[M-H]-": "[M-H]-",
    "[M+Cl]-": "[M+Cl]-",
    "[M+FA-H]-": "[M+FA-H]-",
}

DEFAULT_PARAMETERS = {
    "MS1_ppm": 10,
    "MS2_ppm": 20,
    "halogen": True,
    "top_n": 100,
    "msfinder_dir": None,  # folder containing MsfinderConsoleApp.exe; default external/msfinder/MSFINDER*
    "param_template": None,  # default config/msfinder/MsfinderConsoleApp_Param_formula.txt
}


def _write_spectrum_msp(record: msp_io.SpectrumRecord, adduct: str, path: Path) -> None:
    lines = [
        f"NAME: {record.query_id}",
        f"PRECURSORMZ: {record.precursor_mz}",
        f"PRECURSORTYPE: {adduct}",
        f"IONMODE: {'Positive' if record.ion_mode == 'positive' else 'Negative'}",
        f"Num Peaks: {record.n_peaks}",
    ]
    lines += [f"{mz}\t{intensity}" for mz, intensity in zip(record.mz, record.intensity)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_formula_results(pattern: str, top_n: int) -> pd.DataFrame:
    """Read MS-FINDER 'Formula result-*.txt' into the standard long format."""
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame(columns=common.CANDIDATE_BASE_COLUMNS)
    df = pd.concat([pd.read_table(f, dtype={"File name": str}) for f in files], ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=common.CANDIDATE_BASE_COLUMNS)
    score_column = "Formula score" if "Formula score" in df.columns else "Score"
    df["query_id"] = df["File name"].astype(str)
    # Rows are written in descending score order per spectrum (PredictProcess.cs).
    df["rank"] = df.groupby("query_id").cumcount() + 1
    df = df[df["rank"] <= top_n]
    return pd.DataFrame({
        "query_id": df["query_id"],
        "tool": TOOL,
        "rank": df["rank"],
        "formula": df["Formula"],
        "score": pd.to_numeric(df[score_column], errors="coerce"),
        "mass_error": pd.to_numeric(df.get("Mass error"), errors="coerce"),
        "precursor_type_reported": df.get("Precursor type"),
        "databases": df.get("Databases"),
    })


def run(input_msp: str | Path, output_dir: str | Path, parameters: dict | None = None) -> dict:
    from msemblator.chemistry.converting_data_type import modify_msfinder_config_in_place
    from msemblator.paths import MSFINDER_CONFIG_DIR, MSFINDER_DIR
    from msemblator.runners.msfinder_cmd import run_formula_msfinder

    params = {**DEFAULT_PARAMETERS, **(parameters or {})}
    input_msp, out = Path(input_msp), Path(output_dir) / TOOL
    started = _dt.datetime.now()
    msfinder_dir = Path(params["msfinder_dir"]) if params["msfinder_dir"] else next(
        iter(sorted(MSFINDER_DIR.glob("MSFINDER*"))), MSFINDER_DIR / "MSFINDER")
    exe = msfinder_dir / "MsfinderConsoleApp.exe"
    if not exe.exists():
        raise FileNotFoundError(f"MS-FINDER console app not found: {exe}")
    template = Path(params["param_template"] or MSFINDER_CONFIG_DIR / "MsfinderConsoleApp_Param_formula.txt")

    spectra_dir, raw_dir = out / "input_msp", out / "raw_output"
    for folder in (spectra_dir, raw_dir):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
    method_path = out / "MsfinderConsoleApp_Param_formula.txt"
    shutil.copyfile(template, method_path)
    modify_msfinder_config_in_place(str(method_path), {"formula_prediction": {"msfinder": {
        "MS1_ppm": params["MS1_ppm"], "MS2_ppm": params["MS2_ppm"], "halogen": params["halogen"]}}})

    records = msp_io.read_msp_records(input_msp)
    status: dict[str, dict] = {}
    submitted = []
    for r in records:
        if r.adduct not in MSFINDER_ADDUCTS:
            status[r.query_id] = common.status_row(
                r.query_id, TOOL, msp_io.STATUS_UNSUPPORTED_ADDUCT, message=r.precursor_type_raw)
            continue
        _write_spectrum_msp(r, MSFINDER_ADDUCTS[r.adduct], spectra_dir / f"{r.query_id}.msp")
        submitted.append(r.query_id)

    t0 = time.time()
    if submitted:
        run_formula_msfinder(str(msfinder_dir), str(spectra_dir), str(raw_dir), str(method_path))
    candidates = parse_formula_results(str(raw_dir / "Formula result*.txt"), int(params["top_n"]))
    if submitted and not glob.glob(str(raw_dir / "Formula result*.txt")):
        for qid in submitted:
            status[qid] = common.status_row(qid, TOOL, msp_io.STATUS_FAILED, message="no Formula result file")
    else:
        counts = candidates.groupby("query_id").size().to_dict()
        for qid in submitted:
            n = int(counts.get(qid, 0))
            status[qid] = common.status_row(qid, TOOL, msp_io.STATUS_OK if n else msp_io.STATUS_NO_CANDIDATE, n)

    status_rows = [status[r.query_id] for r in records]
    common.write_candidates(candidates.to_dict("records"), out / f"{TOOL}_candidates.csv",
                            ["mass_error", "precursor_type_reported", "databases"])
    common.write_status(status_rows, out / f"{TOOL}_status.csv")
    versions = {"msfinder_exe": str(exe), "msfinder_folder": msfinder_dir.name,
                "param_template_sha256": msp_io.file_sha256(template)}
    return common.write_manifest(out / "manifest.json", TOOL, input_msp,
                                 {**{k: v for k, v in params.items()}, "seconds": round(time.time() - t0, 1)},
                                 versions, status_rows, started)
