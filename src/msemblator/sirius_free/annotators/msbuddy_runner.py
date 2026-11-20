"""Run MSBuddy formula annotation on an MSP file.

Uses the same MsbuddyConfig as the default workflow (runners/buddy_cmd.py:
ms_instr=None, ppm=True, ms1/ms2 tolerance and halogen from the parameter file,
timeout_secs=600). Differences that avoid errors in the default runner:
- spectra are passed with MetaFeature objects keyed by query_id, so candidates
  cannot be attached to the wrong spectrum when two spectra share a precursor m/z;
- adducts are parsed with report_invalid=True, so an unparseable adduct is
  reported instead of being replaced by [M+H]+ / [M-H]-;
- the mass error, ML probability and isotope similarity are kept.
"""
from __future__ import annotations

import datetime as _dt
import logging
import time
from pathlib import Path

from msemblator.sirius_free import msp_io
from msemblator.sirius_free.annotators import common

logger = logging.getLogger(__name__)

TOOL = "msbuddy"

# String passed to msbuddy for each canonical adduct (all six are parsed by msbuddy).
MSBUDDY_ADDUCTS = {
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
    "timeout_secs": 600,
    "batch_size": 1000,
    "top_n": 100,  # candidates kept per spectrum (default workflow keeps 100)
}

EXTRA_COLUMNS = ["estimated_fdr", "estimated_prob", "normed_estimated_prob", "mz_error",
                 "ms1_isotope_similarity", "db_existed"]


def _float_or_none(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def run(input_msp: str | Path, output_dir: str | Path, parameters: dict | None = None) -> dict:
    from msbuddy import Msbuddy, MsbuddyConfig
    from msbuddy.base import Adduct, MetaFeature, Spectrum

    params = {**DEFAULT_PARAMETERS, **(parameters or {})}
    input_msp, out = Path(input_msp), Path(output_dir) / TOOL
    started = _dt.datetime.now()
    records = msp_io.read_msp_records(input_msp)

    status: dict[str, dict] = {}
    features = []
    for r in records:
        if r.adduct not in MSBUDDY_ADDUCTS:
            status[r.query_id] = common.status_row(
                r.query_id, TOOL, msp_io.STATUS_UNSUPPORTED_ADDUCT, message=r.precursor_type_raw)
            continue
        adduct = MSBUDDY_ADDUCTS[r.adduct]
        positive = r.ion_mode == "positive"
        try:
            Adduct(adduct, positive, report_invalid=True)  # fail loudly, never fall back
            features.append(MetaFeature(
                identifier=r.query_id, mz=float(r.precursor_mz), charge=1 if positive else -1,
                adduct=adduct, ms2=Spectrum(r.mz, r.intensity) if r.mz else None))
        except Exception as error:  # noqa: BLE001 - one bad record must not stop the run
            status[r.query_id] = common.status_row(r.query_id, TOOL, msp_io.STATUS_REJECTED, message=error)

    config = MsbuddyConfig(ms_instr=None, ppm=True, ms1_tol=params["MS1_ppm"], ms2_tol=params["MS2_ppm"],
                           timeout_secs=params["timeout_secs"], halogen=params["halogen"])
    engine = Msbuddy(config)
    rows: list[dict] = []
    t0 = time.time()
    size = int(params["batch_size"])
    for start in range(0, len(features), size):
        chunk = features[start:start + size]
        engine.add_data(chunk)
        try:
            engine.annotate_formula()
        except Exception as error:  # noqa: BLE001
            logger.error("msbuddy failed on spectra %d-%d: %s", start + 1, start + len(chunk), error)
            for feature in chunk:
                status[str(feature.identifier)] = common.status_row(
                    feature.identifier, TOOL, msp_io.STATUS_FAILED, message=error)
            continue
        for feature in engine.data:
            qid = str(feature.identifier)
            candidates = getattr(feature, "candidate_formula_list", None) or []
            candidates = candidates[: int(params["top_n"])]
            for rank, cand in enumerate(candidates, start=1):
                fdr = _float_or_none(getattr(cand, "estimated_fdr", None))
                rows.append({
                    "query_id": qid, "tool": TOOL, "rank": rank,
                    "formula": cand.formula.__str__(),
                    "score": None if fdr is None else 1.0 - fdr,  # same score as the default model
                    "estimated_fdr": fdr,
                    "estimated_prob": _float_or_none(getattr(cand, "estimated_prob", None)),
                    "normed_estimated_prob": _float_or_none(getattr(cand, "normed_estimated_prob", None)),
                    "mz_error": _float_or_none(getattr(cand, "mz_error", None)),
                    "ms1_isotope_similarity": _float_or_none(getattr(cand, "ms1_isotope_similarity", None)),
                    "db_existed": getattr(cand, "db_existed", None),
                })
            state = msp_io.STATUS_OK if candidates else msp_io.STATUS_NO_CANDIDATE
            status[qid] = common.status_row(qid, TOOL, state, n_candidates=len(candidates))
        logger.info("msbuddy: %d/%d spectra processed", min(start + size, len(features)), len(features))

    status_rows = [status.get(r.query_id) or common.status_row(r.query_id, TOOL, msp_io.STATUS_FAILED,
                                                                message="no result returned")
                   for r in records]
    common.write_candidates(rows, out / f"{TOOL}_candidates.csv", EXTRA_COLUMNS)
    common.write_status(status_rows, out / f"{TOOL}_status.csv")
    versions = {p: common.package_version(p) for p in ("msbuddy", "numpy", "lightgbm", "scikit-learn")}
    return common.write_manifest(out / "manifest.json", TOOL, input_msp,
                                 {**params, "seconds": round(time.time() - t0, 1)}, versions, status_rows, started)
