"""Run FIDDLE (msfiddle) formula prediction on an MSP file.

Runs in the MS-Emblator environment, or in another environment that has msfiddle
and torch (it does not need the rest of MS-Emblator).

Rules:
- Collision energy: the MSP value is used. A spectrum without one gets
  default_collision_energy (eV) when it is set; otherwise its status is
  missing_collision_energy and FIDDLE is not run (policy A, the default here, used
  for training). override_collision_energy uses default_collision_energy for every
  spectrum, also when the MSP has a value. Energies from the settings are noted in
  the status message.
- Adduct not supported by FIDDLE (e.g. [M+Cl]-) -> unsupported_adduct; never converted.
- Instrument not recognised -> unknown_instrument, unless a default instrument
  is set explicitly in the configuration.
- Anything else FIDDLE itself rejects (e.g. fewer than 5 peaks, stepped or
  unparseable collision energy) -> rejected_by_tool, with FIDDLE's message.
"""
from __future__ import annotations

import datetime as _dt
import logging
import re
import time
from pathlib import Path

from msemblator.sirius_free import msp_io
from msemblator.sirius_free.annotators import common

logger = logging.getLogger(__name__)

TOOL = "fiddle"

# Canonical adducts FIDDLE accepts, and the string passed to it.
# [M+NH4]+ is converted by FIDDLE to [M+H+NH3]+; [M+FA-H]- is folded into [M-H]- by FIDDLE.
# [M+Cl]- is absent from every released configuration.
FIDDLE_ADDUCTS = {
    "[M+H]+": "[M+H]+",
    "[M+NH4]+": "[M+NH4]+",
    "[M+Na]+": "[M+Na]+",
    "[M-H]-": "[M-H]-",
    "[M+FA-H]-": "[M+FA-H]-",
}

DEFAULT_PARAMETERS = {
    # Released models: qtof, orbitrap, timstof. Rules are checked in order on
    # "<Instrument_type> <Instrument>" (lower case); the first match is used.
    "instrument_rules": [
        ["timstof", "timstof"],
        ["tims", "timstof"],
        ["q-tof", "qtof"],
        ["qtof", "qtof"],
        ["tof", "qtof"],
        ["orbitrap", "orbitrap"],
        ["hcd", "orbitrap"],
        ["ft", "orbitrap"],
        ["q exactive", "orbitrap"],
    ],
    "default_instrument": None,  # None: skip spectra whose instrument is unknown
    "default_collision_energy": None,  # eV for spectra without one; None: skip them (policy A)
    "override_collision_energy": False,  # True: default_collision_energy for every spectrum
    "top_k": 20,  # = post_processing.top_k of the released configs
    "batch_size": 32,
    "chunk_size": 512,  # spectra per predict_batch call
    "device": "cpu",
}


# NIST writes the absolute energy as "NCE=27% 30V" as well as "NCE=27% 30eV".
# msfiddle 2.3.0 parses a bare "30V" as eV and uses the eV part of
# "NCE=27% 30eV", but has no pattern for "NCE=27% 30V". Only the unit is
# rewritten (V -> eV); the values are left unchanged.
_NCE_VOLT = re.compile(r"^(NCE=\d+(?:\.\d+)?%\s+\d+(?:\.\d+)?)\s*V$")


def normalize_collision_energy(text: str) -> str:
    text = text.strip()
    return _NCE_VOLT.sub(r"\1eV", text)


def collision_energy_for(msp_value: str, default=None, override: bool = False) -> tuple[str | None, str]:
    """(energy passed to FIDDLE, source "msp" / "default" / "override"); (None, "") when there is none."""
    text = (msp_value or "").strip()
    if default is not None and str(default).strip():
        value = f"{float(default):g}" if isinstance(default, (int, float)) else normalize_collision_energy(str(default))
        if override:
            return value, "override"
        if not text:
            return value, "default"
    if text:
        return normalize_collision_energy(text), "msp"
    return None, ""


def instrument_model(record: msp_io.SpectrumRecord, rules, default=None) -> str | None:
    text = f"{record.instrument_type} {record.instrument}".lower()
    for pattern, model in rules:
        if pattern in text:
            return model
    return default


def _checkpoint_info(predictor) -> dict:
    info = {}
    for attr in ("resume_path", "rescore_resume_path"):
        path = Path(getattr(predictor, attr))
        info[path.name] = msp_io.file_sha256(path) if path.exists() else "missing"
    return info


def _validate(predictor, spectrum: dict) -> str | None:
    """Return FIDDLE's rejection message for one spectrum, or None if it is accepted."""
    try:
        predictor._encode_spectrum(  # same check FIDDLE applies to every spectrum
            mz_array=spectrum["mz_array"],
            intensity_array=spectrum["intensity_array"],
            precursor_mz=spectrum["precursor_mz"],
            adduct=spectrum["adduct"],
            collision_energy=spectrum["collision_energy"],
            spec_id=spectrum["id"],
        )
    except ValueError as error:
        return str(error)
    return None


def _predict(predictor, spectra: list[dict], top_k: int) -> dict[str, list[dict]]:
    """predict_batch on a chunk; on failure retry one by one so one bad spectrum cannot sink the chunk."""
    try:
        return {r["id"]: r["candidates"] for r in predictor.predict_batch(spectra, top_k=top_k)}
    except Exception as error:  # noqa: BLE001 - isolate the failing spectrum below
        logger.warning("FIDDLE batch failed (%s); retrying %d spectra one by one", error, len(spectra))
    results = {}
    for spectrum in spectra:
        try:
            results[spectrum["id"]] = predictor.predict_batch([spectrum], top_k=top_k)[0]["candidates"]
        except Exception as error:  # noqa: BLE001
            results[spectrum["id"]] = error
    return results


def run(input_msp: str | Path, output_dir: str | Path, parameters: dict | None = None) -> dict:
    from msfiddle import MsFiddlePredictor  # imported here: only available in the FIDDLE env

    params = {**DEFAULT_PARAMETERS, **(parameters or {})}
    input_msp, out = Path(input_msp), Path(output_dir) / TOOL
    started = _dt.datetime.now()
    records = msp_io.read_msp_records(input_msp)

    status: dict[str, dict] = {}
    by_model: dict[str, list[dict]] = {}
    n_ce_rewritten = 0
    ce_notes: dict[str, str] = {}
    ce_from_settings = {"default": 0, "override": 0}
    for r in records:
        if r.adduct not in FIDDLE_ADDUCTS:
            status[r.query_id] = common.status_row(
                r.query_id, TOOL, msp_io.STATUS_UNSUPPORTED_ADDUCT, message=r.precursor_type_raw)
            continue
        energy, source = collision_energy_for(r.collision_energy, params["default_collision_energy"],
                                              bool(params["override_collision_energy"]))
        if energy is None:
            status[r.query_id] = common.status_row(r.query_id, TOOL, msp_io.STATUS_MISSING_CE)
            continue
        model = instrument_model(r, params["instrument_rules"], params["default_instrument"])
        if model is None:
            status[r.query_id] = common.status_row(
                r.query_id, TOOL, msp_io.STATUS_UNKNOWN_INSTRUMENT,
                message=f"{r.instrument_type} {r.instrument}".strip())
            continue
        if source == "msp":
            n_ce_rewritten += energy != r.collision_energy.strip()
        else:
            ce_from_settings[source] += 1
            ce_notes[r.query_id] = (f"collision energy {energy} eV from the settings "
                                    + ("(replaces the MSP value)" if source == "override" else "(none in the MSP)"))
        by_model.setdefault(model, []).append({
            "id": r.query_id,
            "mz_array": r.mz,
            "intensity_array": r.intensity,
            "precursor_mz": r.precursor_mz,
            "adduct": FIDDLE_ADDUCTS[r.adduct],
            "collision_energy": energy,
        })
    logger.info("FIDDLE: %d spectra to predict (%s); collision energy unit V -> eV rewritten for %d; "
                "from the settings: %d without one, %d replaced",
                sum(len(v) for v in by_model.values()),
                ", ".join(f"{m}: {len(v)}" for m, v in by_model.items()), n_ce_rewritten,
                ce_from_settings["default"], ce_from_settings["override"])

    rows: list[dict] = []
    checkpoints, timings = {}, {}
    for model, spectra in by_model.items():
        t0 = time.time()
        predictor = MsFiddlePredictor(instrument_type=model, device=params["device"],
                                      batch_size=params["batch_size"])
        checkpoints[model] = _checkpoint_info(predictor)
        accepted = []
        for spectrum in spectra:
            message = _validate(predictor, spectrum)
            if message is None:
                accepted.append(spectrum)
            else:
                status[spectrum["id"]] = common.status_row(
                    spectrum["id"], TOOL, msp_io.STATUS_REJECTED, message=message)
        logger.info("FIDDLE %s: %d spectra accepted, %d rejected",
                    model, len(accepted), len(spectra) - len(accepted))
        size = int(params["chunk_size"])
        for start in range(0, len(accepted), size):
            chunk = accepted[start:start + size]
            results = _predict(predictor, chunk, int(params["top_k"]))
            for spectrum in chunk:
                qid = spectrum["id"]
                result = results.get(qid)
                if isinstance(result, Exception) or result is None:
                    status[qid] = common.status_row(qid, TOOL, msp_io.STATUS_FAILED, message=result)
                    continue
                for rank, cand in enumerate(result, start=1):
                    rows.append({
                        "query_id": qid, "tool": TOOL, "rank": rank,
                        "formula": cand["formula"], "score": cand["score"],
                        "raw_rescore": cand["metadata"].get("raw_rescore"),
                        "candidate_mass": cand.get("mass"),
                        "pred_formula": cand["metadata"].get("pred_formula", ""),
                        "instrument_model": model,
                    })
                state = msp_io.STATUS_OK if result else msp_io.STATUS_NO_CANDIDATE
                status[qid] = common.status_row(qid, TOOL, state, n_candidates=len(result),
                                                message=ce_notes.get(qid, ""))
            logger.info("FIDDLE %s: %d/%d spectra predicted", model, min(start + size, len(accepted)), len(accepted))
        timings[model] = round(time.time() - t0, 1)

    status_rows = [status[r.query_id] for r in records]
    common.write_candidates(rows, out / f"{TOOL}_candidates.csv",
                            ["raw_rescore", "candidate_mass", "pred_formula", "instrument_model"])
    common.write_status(status_rows, out / f"{TOOL}_status.csv")
    versions = {p: common.package_version(p) for p in ("msfiddle", "torch", "numpy", "rdkit")}
    versions["checkpoints_sha256"] = checkpoints
    return common.write_manifest(out / "manifest.json", TOOL, input_msp,
                                 {**params, "seconds_per_model": timings,
                                  "collision_energy_unit_rewritten": n_ce_rewritten,
                                  "collision_energy_from_settings": ce_from_settings},
                                 versions, status_rows, started)
