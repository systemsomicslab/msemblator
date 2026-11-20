"""Read MSP files without dropping metadata the annotation tools need.

The default workflow rewrites the MSP (modify_msp_data_type) and keeps only a
fixed set of fields, which removes the collision energy and instrument type that
FIDDLE requires. This reader keeps every field.

Query IDs are the 1-based position of each NAME block, i.e. the same numbering
as msemblator.formats.msp_format_change.convert_name_to_peakid, so outputs of all
tools can be joined on query_id.

Only the standard library and numpy-free code is used here so that the module
can also be imported from the separate FIDDLE environment.
"""
from __future__ import annotations

import csv
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Canonical names of the six adducts used by MS-Emblator.
CANONICAL_ADDUCTS = ("[M+H]+", "[M+NH4]+", "[M+Na]+", "[M-H]-", "[M+Cl]-", "[M+FA-H]-")

# Spellings that denote exactly the same ion as a canonical adduct.
# "[M+CO2]-" is deliberately NOT included: it is a different ion (+CO2, not +HCOO).
ADDUCT_ALIASES = {
    "[M+H]+": "[M+H]+",
    "[M+NH4]+": "[M+NH4]+",
    "[M+H+NH3]+": "[M+NH4]+",
    "[M+Na]+": "[M+Na]+",
    "[M-H]-": "[M-H]-",
    "[M+Cl]-": "[M+Cl]-",
    "[M+FA-H]-": "[M+FA-H]-",
    "[M+HCOO]-": "[M+FA-H]-",
    "[M+CHO2]-": "[M+FA-H]-",
    "[M+HCO2]-": "[M+FA-H]-",
    "[M+HCOOH-H]-": "[M+FA-H]-",
    "[M+CH2O2-H]-": "[M+FA-H]-",
}

# MSP keys (case-insensitive) that hold each field. The first match wins.
_FIELD_KEYS = {
    "precursor_mz": ("PRECURSORMZ", "PRECURSOR_MZ", "PEPMASS"),
    "precursor_type": ("PRECURSORTYPE", "PRECURSOR_TYPE", "ADDUCT", "ADDUCTTYPE"),
    "ion_mode": ("ION_MODE", "IONMODE", "IONIZATION_MODE"),
    "collision_energy": ("COLLISION_ENERGY", "COLLISIONENERGY"),
    "instrument_type": ("INSTRUMENT_TYPE", "INSTRUMENTTYPE"),
    "instrument": ("INSTRUMENT",),
    "formula": ("FORMULA",),
    "inchikey": ("INCHIKEY",),
    "smiles": ("SMILES",),
}


@dataclass
class SpectrumRecord:
    query_id: str
    name: str
    precursor_mz: float | None
    precursor_type_raw: str
    adduct: str | None  # canonical name, None if not one of CANONICAL_ADDUCTS
    ion_mode: str | None  # "positive" / "negative" / None
    collision_energy: str
    instrument_type: str
    instrument: str
    reference_formula: str
    inchikey: str
    smiles: str
    mz: list[float] = field(default_factory=list)
    intensity: list[float] = field(default_factory=list)

    @property
    def n_peaks(self) -> int:
        return len(self.mz)


def canonical_adduct(precursor_type: str | None) -> str | None:
    """Return the canonical adduct name, or None when it is not supported."""
    if not precursor_type:
        return None
    key = re.sub(r"\s+", "", str(precursor_type))
    return ADDUCT_ALIASES.get(key)


def ion_mode_of(precursor_type: str | None, ion_mode_field: str = "") -> str | None:
    """Ion mode from the charge sign at the end of the adduct, else from the ION_MODE field."""
    text = re.sub(r"\s+", "", str(precursor_type or ""))
    if text.endswith("+"):
        return "positive"
    if text.endswith("-"):
        return "negative"
    mode = ion_mode_field.strip().lower()
    if mode in ("p", "pos", "positive", "+"):
        return "positive"
    if mode in ("n", "neg", "negative", "-"):
        return "negative"
    return None


def _parse_peak(line: str):
    parts = re.split(r"[\s,;]+", line.strip())
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _make_record(index: int, fields: dict, peaks: list) -> SpectrumRecord:
    def get(name):
        for key in _FIELD_KEYS[name]:
            if key in fields:
                return fields[key]
        return ""

    mz_text = get("precursor_mz")
    try:
        precursor_mz = float(mz_text.split()[0]) if mz_text else None
    except ValueError:
        precursor_mz = None
    precursor_type = get("precursor_type")
    return SpectrumRecord(
        query_id=str(index),
        name=fields.get("NAME", ""),
        precursor_mz=precursor_mz,
        precursor_type_raw=precursor_type,
        adduct=canonical_adduct(precursor_type),
        ion_mode=ion_mode_of(precursor_type, get("ion_mode")),
        collision_energy=get("collision_energy"),
        instrument_type=get("instrument_type"),
        instrument=get("instrument"),
        reference_formula=get("formula"),
        inchikey=get("inchikey"),
        smiles=get("smiles"),
        mz=[p[0] for p in peaks],
        intensity=[p[1] for p in peaks],
    )


def read_msp_records(path: str | Path) -> list[SpectrumRecord]:
    """Parse an MSP file into SpectrumRecord objects (one per NAME block)."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    records: list[SpectrumRecord] = []
    fields: dict | None = None
    peaks: list = []
    in_peaks = False

    def flush():
        if fields is not None:
            records.append(_make_record(len(records) + 1, fields, peaks))

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            in_peaks = False
            continue
        key, sep, value = line.partition(":")
        key_upper = key.strip().upper()
        if sep and key_upper == "NAME":
            flush()
            fields, peaks, in_peaks = {"NAME": value.strip()}, [], False
            continue
        if fields is None:
            continue
        if sep and key_upper in ("NUM PEAKS", "NUMPEAKS", "NUM_PEAKS"):
            in_peaks = True
            continue
        if in_peaks:
            peak = _parse_peak(line)
            if peak is not None:
                peaks.append(peak)
                continue
            in_peaks = False
        if sep:
            fields.setdefault(key_upper, value.strip())
    flush()
    logger.info("Read %d spectra from %s", len(records), path)
    return records


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


SPECTRA_INDEX_COLUMNS = [
    "query_id", "name", "precursor_mz", "precursor_type_raw", "adduct", "ion_mode",
    "instrument_type", "instrument", "collision_energy", "n_peaks",
    "reference_formula", "inchikey",
]


def write_spectra_index(records: list[SpectrumRecord], path: str | Path) -> None:
    """Write one row per spectrum; used to join tool outputs and reference answers."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SPECTRA_INDEX_COLUMNS)
        for r in records:
            writer.writerow([
                r.query_id, r.name, r.precursor_mz, r.precursor_type_raw, r.adduct or "",
                r.ion_mode or "", r.instrument_type, r.instrument, r.collision_energy,
                r.n_peaks, r.reference_formula, r.inchikey,
            ])


# Status values shared by every tool's *_status.csv.
STATUS_OK = "ok"
STATUS_NO_CANDIDATE = "no_candidate"
STATUS_UNSUPPORTED_ADDUCT = "unsupported_adduct"
STATUS_MISSING_CE = "missing_collision_energy"
STATUS_UNKNOWN_INSTRUMENT = "unknown_instrument"
STATUS_REJECTED = "rejected_by_tool"
STATUS_FAILED = "failed"

STATUS_COLUMNS = ["query_id", "tool", "status", "n_candidates", "message"]
