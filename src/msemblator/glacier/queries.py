"""Measured query spectra: read the MSP with msentity and prepare them for matching.

Only what candidate extraction and scoring need is kept: name, adduct, molecular
formula and peaks. Answer fields of the MSP (InChIKey, SMILES) are not read here;
labels.py reads them separately so that they cannot influence the scores.

Invalid peaks (not finite, m/z <= 0, intensity <= 0) are dropped. The precursor
peak is dropped only when glacier.precursor_peak_removal is enabled.

The molecular formula is the neutral molecule decided before structure
annotation: by default the MSP's formula field (the formula the default workflow
writes into the MSP), or a table given with ``formula_table``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from msemblator.glacier.adducts import AdductResolver
from msemblator.glacier.config import GlacierConfig
from msemblator.glacier.formula import normalize_formula
from msemblator.glacier.preprocessing import (REMOVAL_NO_PRECURSOR_MZ, REMOVAL_NOT_FOUND, REMOVAL_OFF,
                                              REMOVAL_REMOVED, precursor_peak_mask)

logger = logging.getLogger(__name__)

# Query status values (glacier_query_status.csv).
STATUS_OK = "ok"
STATUS_NO_CANDIDATE = "no_candidate"
STATUS_UNSUPPORTED_ADDUCT = "unsupported_adduct"
STATUS_LIBRARY_NOT_AVAILABLE = "library_not_available"
STATUS_MISSING_FORMULA = "missing_formula"
STATUS_INVALID_FORMULA = "invalid_formula"
STATUS_NO_VALID_PEAKS = "no_valid_peaks"
STATUS_SIMILARITY_FAILED = "similarity_failed"

QUERY_COLUMNS = [
    "query_index", "filename", "adduct_raw", "adduct", "formula_raw", "formula", "formula_source",
    "precursor_mz", "n_peaks_raw", "n_peaks", "n_invalid_peaks", "precursor_peak_removal",
    "n_precursor_peaks_removed", "status", "message",
]


def _first_column(dataset, names, what, required=True):
    for name in names:
        if name in dataset:
            return name
    if required:
        raise KeyError(f"No {what} column in the query MSP (looked for {names}; found {dataset.columns})")
    return None


def _to_float(value):
    try:
        number = float(str(value).split()[0])
    except (TypeError, ValueError, IndexError):
        return np.nan
    return number if np.isfinite(number) else np.nan


@dataclass
class QuerySet:
    table: pd.DataFrame          # one row per spectrum of the MSP, in file order
    peaks: list                  # cleaned (n, 2) float64 arrays, None without valid peaks
    source: str

    def dataset(self, rows):
        """msentity dataset of the given table rows (cleaned peaks and PrecursorMZ)."""
        import msentity as ms

        rows = list(rows)
        dataset = ms.from_peaks([self.peaks[r] for r in rows])
        dataset.set_column("PrecursorMZ", [float(self.table.at[r, "precursor_mz"]) for r in rows])
        return dataset


def read_formula_table(path: str | Path) -> tuple[dict, str]:
    """Formulas decided before structure annotation: CSV with a formula column and
    filename (MSP NAME) or query_index. Returns ({key: formula}, key column)."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    columns = {c.lower(): c for c in table.columns}
    formula_column = columns.get("formula")
    key_column = columns.get("filename") or columns.get("query_index")
    if not formula_column or not key_column:
        raise KeyError(f"{path}: needs a 'formula' column and a 'filename' or 'query_index' column")
    if table[key_column].duplicated().any():
        raise ValueError(f"{path}: {key_column} must be unique (give one formula per spectrum)")
    return dict(zip(table[key_column].str.strip(), table[formula_column].str.strip())), key_column.lower()


def read_queries(path: str | Path, config: GlacierConfig, resolver: AdductResolver,
                 formula_table: str | Path | None = None) -> QuerySet:
    import msentity as ms

    path = Path(path)
    dataset = ms.load(path, canonicalize_column=True)
    columns = config.query_columns
    name_col = _first_column(dataset, columns["name"], "name")
    adduct_col = _first_column(dataset, columns["adduct"], "adduct")
    mz_col = _first_column(dataset, columns["precursor_mz"], "precursor m/z", required=False)
    names = [str(v if v is not None else "").strip() for v in dataset[name_col]]
    adducts_raw = [str(v if v is not None else "").strip() for v in dataset[adduct_col]]
    precursor = [_to_float(v) for v in dataset[mz_col]] if mz_col else [np.nan] * len(dataset)

    if formula_table:
        formulas_by_key, key = read_formula_table(formula_table)
        keys = names if key == "filename" else [str(i + 1) for i in range(len(dataset))]
        formulas_raw = [formulas_by_key.get(k, "") for k in keys]
        formula_source = f"table:{Path(formula_table).name}"
    else:
        formula_col = _first_column(dataset, columns["formula"], "formula", required=False)
        formulas_raw = [str(v if v is not None else "").strip() for v in dataset[formula_col]] \
            if formula_col else [""] * len(dataset)
        formula_source = f"msp:{formula_col}" if formula_col else "msp:none"

    offsets = dataset.offsets
    all_mz, all_intensity = dataset.mz, dataset.intensity
    remove_precursor = config.removes_precursor("measured")
    rows, peaks = [], []
    for i in range(len(dataset)):
        mz = all_mz[offsets[i]:offsets[i + 1]]
        intensity = all_intensity[offsets[i]:offsets[i + 1]]
        valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
        keep, removal_state, n_removed = valid, REMOVAL_OFF, 0
        if remove_precursor:
            if np.isnan(precursor[i]):
                removal_state = REMOVAL_NO_PRECURSOR_MZ
            else:
                hit = valid & precursor_peak_mask(mz, precursor[i], config.precursor_peak_removal)
                n_removed = int(hit.sum())
                removal_state = REMOVAL_REMOVED if n_removed else REMOVAL_NOT_FOUND
                keep = valid & ~hit
        clean = np.column_stack([mz[keep], intensity[keep]]).astype(np.float64) if keep.any() else None
        peaks.append(clean)
        adduct = resolver.resolve(adducts_raw[i])
        formula = normalize_formula(formulas_raw[i])
        status, message = "", ""
        if adduct is None:
            status, message = STATUS_UNSUPPORTED_ADDUCT, adducts_raw[i]
        elif not formulas_raw[i]:
            status = STATUS_MISSING_FORMULA
        elif formula is None:
            status, message = STATUS_INVALID_FORMULA, formulas_raw[i]
        elif clean is None:
            status = STATUS_NO_VALID_PEAKS
            if valid.any():
                message = "no peak left after precursor peak removal"
        rows.append({
            "query_index": i + 1,
            "filename": names[i],
            "adduct_raw": adducts_raw[i],
            "adduct": adduct or "",
            "formula_raw": formulas_raw[i],
            "formula": formula or "",
            "formula_source": formula_source,
            "precursor_mz": precursor[i],
            "n_peaks_raw": int(len(mz)),
            "n_peaks": int(keep.sum()),
            "n_invalid_peaks": int(len(mz) - valid.sum()),
            "precursor_peak_removal": removal_state,
            "n_precursor_peaks_removed": n_removed,
            "status": status,
            "message": message,
        })
    table = pd.DataFrame(rows, columns=QUERY_COLUMNS)
    duplicated = table["filename"].duplicated(keep=False) & table["filename"].ne("")
    if duplicated.any():
        logger.warning("%d spectra share a NAME; use query_index to tell them apart", int(duplicated.sum()))
    logger.info("Read %d query spectra from %s (formula from %s)", len(table), path, formula_source)
    return QuerySet(table, peaks, str(path))
