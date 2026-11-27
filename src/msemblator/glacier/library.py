"""GLACIER predicted spectral libraries: file lookup, loading, checks and the formula index.

One MSDS file per adduct (glacier.libraries). A library is loaded only while its
adduct is processed and is released afterwards, so several large libraries are
never held in memory at the same time.

Candidate extraction uses only the molecular formula and the adduct: library
spectra are indexed by (formula key) within the file of one adduct. No precursor
m/z or MS1 tolerance is used for extraction.

FORMULA of a library may be the neutral molecule M (as in M_minus_H_predictions.msds)
or the formula after the neutral part of the adduct (e.g. M - H2O for [M-H2O+H]+).
The definition is detected from the library precursor m/z (formula_basis: auto) and
the query formula is converted once, only for the "modified" definition.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from msemblator.glacier.adducts import AdductResolver, apply_modification, parse_adduct
from msemblator.glacier.config import GlacierConfig
from msemblator.glacier.formula import formula_key, has_known_masses, parse_formula
from msemblator.glacier.preprocessing import precursor_peak_mask

logger = logging.getLogger(__name__)

LIBRARY_AVAILABLE = "available"
LIBRARY_NOT_CONFIGURED = "not_configured"
LIBRARY_FILE_MISSING = "file_missing"
LIBRARY_UNUSABLE = "unusable"

_SHORT_KEY = re.compile(r"^[A-Z]{14}$")
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
MZ_CHECK_DA, MZ_CHECK_PPM = 0.005, 10.0
FORMULA_BASIS_MIN_SHARE = 0.95


class LibraryError(RuntimeError):
    """The library file exists but cannot be used (missing columns, unknown formula definition)."""


def library_status(config: GlacierConfig, adduct: str) -> tuple[str, Path | None]:
    path = config.library_path(adduct)
    if path is None:
        return LIBRARY_NOT_CONFIGURED, None
    return (LIBRARY_AVAILABLE if path.is_file() else LIBRARY_FILE_MISSING), path


def parse_collision_energy(value) -> float:
    """Energy in eV from "10", "10.0 eV"; NaN when missing or not in eV (e.g. "NCE=30%")."""
    text = str(value if value is not None else "").strip()
    if not text or "%" in text or "nce" in text.lower():
        return np.nan
    match = _NUMBER.search(text)
    return float(match.group()) if match else np.nan


def short_inchikey(inchikey, smiles, cache) -> str:
    key = str(inchikey or "").strip().upper().split("-", 1)[0]
    if _SHORT_KEY.fullmatch(key):
        return key
    smiles = str(smiles or "").strip()
    if not smiles:
        return ""
    if smiles not in cache:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        molecule = Chem.MolFromSmiles(smiles)
        try:
            cache[smiles] = Chem.MolToInchiKey(molecule)[:14] if molecule is not None else ""
        except Exception:  # noqa: BLE001 - RDKit raises several error types
            cache[smiles] = ""
    return cache[smiles]


def _mz_close(expected, observed) -> bool:
    return abs(expected - observed) <= max(MZ_CHECK_DA, observed * MZ_CHECK_PPM * 1e-6)


def check_formula_basis(formulas, precursor_mz, adduct_name) -> dict:
    """Share of library spectra whose precursor m/z fits each formula definition."""
    adduct = parse_adduct(adduct_name)
    if adduct is None:
        return {"detected": None, "reason": f"adduct notation not understood: {adduct_name}"}
    tested = neutral_ok = modified_ok = 0
    for formula, mz in zip(formulas, precursor_mz):
        counts = parse_formula(formula)
        if counts is None or not has_known_masses(counts) or not np.isfinite(mz):
            continue
        tested += 1
        neutral_ok += _mz_close(adduct.ion_mz(counts), mz)
        if adduct.neutral_modification:
            neutral = apply_modification(counts, adduct, direction=-1)
            modified_ok += neutral is not None and _mz_close(adduct.ion_mz(neutral), mz)
    result = {"tested": tested, "neutral_share": neutral_ok / tested if tested else None,
              "modified_share": (modified_ok / tested if tested else None) if adduct.neutral_modification else None,
              "neutral_modification": adduct.neutral_modification}
    if not tested:
        result.update(detected=None, reason="no precursor m/z to check")
    elif result["neutral_share"] >= FORMULA_BASIS_MIN_SHARE:
        result.update(detected="neutral", reason="precursor m/z = adduct applied to FORMULA")
    elif adduct.neutral_modification and result["modified_share"] >= FORMULA_BASIS_MIN_SHARE:
        result.update(detected="modified", reason="FORMULA already includes the neutral part of the adduct")
    else:
        result.update(detected=None, reason="precursor m/z fits neither formula definition")
    return result


@dataclass
class LibraryStats:
    adduct: str
    path: str
    n_spectra_file: int = 0
    n_spectra_used: int = 0
    n_adduct_mismatch: int = 0
    n_invalid_formula: int = 0
    n_no_peaks: int = 0
    n_no_short_inchikey: int = 0
    n_ce_not_configured: int = 0
    n_ce_missing: int = 0
    spectra_by_ce: dict = field(default_factory=dict)
    n_formulas: int = 0
    precursor_peak_removal: bool = False
    n_spectra_precursor_removed: int = 0
    n_precursor_peaks_removed: int = 0
    n_spectra_no_precursor_mz: int = 0
    formula_basis: str = ""
    formula_basis_check: dict = field(default_factory=dict)
    columns_used: dict = field(default_factory=dict)
    load_seconds: float = 0.0


class PredictedLibrary:
    """The spectra of one adduct's MSDS file, indexed by molecular formula."""

    def __init__(self, path: str | Path, adduct: str, config: GlacierConfig, resolver: AdductResolver):
        import time

        import msentity as ms

        started = time.time()
        self.path, self.adduct, self.config = Path(path), adduct, config
        self.dataset = ms.load(self.path)
        n = len(self.dataset)
        self.stats = LibraryStats(adduct=adduct, path=str(self.path), n_spectra_file=n)

        columns = {}
        for name, candidates in config.library_columns.items():
            columns[name] = next((c for c in candidates if c in self.dataset), None)
        missing = [k for k in ("smiles", "inchikey", "formula", "collision_energy") if columns[k] is None]
        if missing:
            raise LibraryError(f"{self.path.name}: no column for {missing} (columns: {self.dataset.columns})")
        self.stats.columns_used = columns

        def text_column(key):
            values = self.dataset[columns[key]] if columns[key] else [None] * n
            return np.array(["" if v is None else str(v).strip() for v in values], dtype=object)

        self.spec_id = text_column("spec_id")
        self.smiles = text_column("smiles")
        self.inchikey = text_column("inchikey")
        formula_raw = text_column("formula")
        adduct_raw = text_column("adduct")
        ce_raw = self.dataset[columns["collision_energy"]]
        self.n_peaks = np.diff(self.dataset.offsets)
        self.precursor_mz = None
        if config.removes_precursor("predicted"):
            if columns["precursor_mz"] is None:
                raise LibraryError(f"{self.path.name}: precursor peak removal needs a precursor m/z column "
                                   f"(looked for {config.library_columns['precursor_mz']})")
            import pandas as pd
            self.precursor_mz = pd.to_numeric(pd.Series(self.dataset[columns["precursor_mz"]], dtype=object),
                                              errors="coerce").to_numpy(dtype=float)
            self.stats.precursor_peak_removal = True

        # Rows of another adduct are never used (no cross-adduct matching).
        resolved = {v: resolver.resolve(v) for v in set(adduct_raw)}
        adduct_ok = np.array([(resolved[v] == adduct) if v else True for v in adduct_raw])
        self.stats.n_adduct_mismatch = int((~adduct_ok).sum())

        keys = {v: (formula_key(c) if (c := parse_formula(v)) else "") for v in set(formula_raw)}
        self.formula_key = np.array([keys[v] for v in formula_raw], dtype=object)
        formula_ok = self.formula_key != ""
        self.stats.n_invalid_formula = int((adduct_ok & ~formula_ok).sum())

        cache: dict = {}
        self.short_key = np.array([short_inchikey(k, s, cache) for k, s in zip(self.inchikey, self.smiles)],
                                  dtype=object)
        key_ok = self.short_key != ""

        ce_text = {v: parse_collision_energy(v) for v in set(map(str, ce_raw))}
        self.collision_energy = np.array([ce_text[str(v)] for v in ce_raw], dtype=float)
        ce_missing = np.isnan(self.collision_energy)
        configured = np.array(config.collision_energies)
        ce_configured = np.isclose(self.collision_energy[:, None], configured[None, :], atol=1e-6).any(axis=1)
        peaks_ok = self.n_peaks > 0

        base = adduct_ok & formula_ok
        self.stats.n_no_peaks = int((base & ~peaks_ok).sum())
        self.stats.n_no_short_inchikey = int((base & peaks_ok & ~key_ok).sum())
        usable = base & peaks_ok & key_ok
        self.stats.n_ce_missing = int((usable & ce_missing).sum())
        self.stats.n_ce_not_configured = int((usable & ~ce_missing & ~ce_configured).sum())
        self.valid = usable & (ce_configured | ce_missing)
        self.stats.n_spectra_used = int(self.valid.sum())
        if self.stats.n_spectra_used == 0:
            # Fail loudly instead of reporting every query as no_candidate, e.g. when the
            # library writes the adduct in a notation that is not in adduct_aliases.
            other = sorted(v for v, r in resolved.items() if v and r != adduct)[:5]
            raise LibraryError(
                f"{self.path.name}: no usable spectrum for {adduct} ({n} spectra; adduct mismatch "
                f"{self.stats.n_adduct_mismatch}, invalid formula {self.stats.n_invalid_formula}, "
                f"no peaks {self.stats.n_no_peaks}, no Short InChIKey {self.stats.n_no_short_inchikey}, "
                f"energy not configured {self.stats.n_ce_not_configured})"
                + (f"; adduct values in the file: {other} (add them to glacier.adduct_aliases)" if other else ""))
        energies, counts = np.unique(self.collision_energy[self.valid & ~ce_missing], return_counts=True)
        self.stats.spectra_by_ce = {f"{e:g}": int(c) for e, c in zip(energies, counts)}

        self._decide_formula_basis(formula_raw, columns["precursor_mz"])

        positions = np.flatnonzero(self.valid)
        order = np.lexsort((self.collision_energy[positions], self.short_key[positions].astype(str),
                            self.formula_key[positions].astype(str)))
        positions = positions[order]
        sorted_keys = self.formula_key[positions]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        ends = np.r_[starts[1:], len(positions)]
        self._index = {sorted_keys[s]: positions[s:e] for s, e in zip(starts, ends)}
        self.stats.n_formulas = len(self._index)
        # The metadata needed later has been copied above. Hiding the other columns
        # (keeping the precursor m/z for modified cosine / BONANZA) makes each
        # candidate subset about 3x faster to hand over to msentity-similarity.
        self.dataset.columns = [columns["precursor_mz"]] if columns["precursor_mz"] else []
        self.stats.load_seconds = round(time.time() - started, 1)
        logger.info("%s %s: %d of %d spectra usable, %d formulas, formula basis %s (%.1f s)",
                    adduct, self.path.name, self.stats.n_spectra_used, n, self.stats.n_formulas,
                    self.formula_basis, self.stats.load_seconds)

    def _decide_formula_basis(self, formula_raw, mz_column):
        setting = self.config.formula_basis_for(self.adduct)
        rows = np.flatnonzero(self.valid)
        sample = rows[np.linspace(0, len(rows) - 1, min(2000, len(rows))).astype(int)] if len(rows) else rows
        if mz_column:
            mz = np.array([float(self.dataset.value(int(i), mz_column) or "nan") for i in sample])
        else:
            mz = np.full(len(sample), np.nan)
        check = check_formula_basis(formula_raw[sample], mz, self.adduct)
        self.stats.formula_basis_check = check
        adduct = parse_adduct(self.adduct)
        if setting == "auto":
            if check["detected"] is None:
                if adduct is not None and not adduct.neutral_modification and not check.get("tested"):
                    basis = "neutral"  # both definitions are the same; nothing to check against
                else:
                    raise LibraryError(f"{self.path.name}: cannot tell how FORMULA is defined ({check['reason']}); "
                                       f"set glacier.formula_basis for {self.adduct}")
            else:
                basis = check["detected"]
        else:
            basis = setting
            if check["detected"] not in (None, basis):
                logger.warning("%s: formula_basis is set to %s but the precursor m/z fits %s",
                               self.path.name, basis, check["detected"])
        if basis == "modified" and (adduct is None or not adduct.neutral_modification):
            basis = "neutral"  # nothing to convert for this adduct
        self.formula_basis = basis
        self._adduct_definition = adduct
        self.stats.formula_basis = basis

    def lookup_key(self, neutral_formula: str) -> str | None:
        """Library formula key for a neutral query formula (converted once for 'modified')."""
        if self.formula_basis == "neutral":
            return neutral_formula
        counts = parse_formula(neutral_formula)
        converted = apply_modification(counts, self._adduct_definition, direction=1) if counts else None
        return formula_key(converted) if converted else None

    def candidates(self, neutral_formula: str) -> np.ndarray:
        key = self.lookup_key(neutral_formula)
        return self._index.get(key, np.empty(0, dtype=np.int64)) if key else np.empty(0, dtype=np.int64)

    def spectra(self, positions: np.ndarray):
        """(msentity dataset of these spectra, peaks per spectrum) as used for scoring.

        With precursor_peak_removal.spectra = both, the precursor peak of each
        predicted spectrum is removed (a spectrum without precursor m/z is kept as it is).
        """
        subset = self.dataset.select(positions)
        if self.precursor_mz is None:
            return subset, self.n_peaks[positions]
        import msentity as ms

        removal = self.config.precursor_peak_removal
        offsets, all_mz, all_intensity = subset.offsets, subset.mz, subset.intensity
        peaks = []
        for k, position in enumerate(positions):
            mz = all_mz[offsets[k]:offsets[k + 1]]
            intensity = all_intensity[offsets[k]:offsets[k + 1]]
            precursor = self.precursor_mz[position]
            if not np.isfinite(precursor):
                self.stats.n_spectra_no_precursor_mz += 1
            hit = precursor_peak_mask(mz, precursor, removal)
            if hit.any():
                self.stats.n_spectra_precursor_removed += 1
                self.stats.n_precursor_peaks_removed += int(hit.sum())
            peaks.append(np.column_stack([mz[~hit], intensity[~hit]]))
        dataset = ms.from_peaks(peaks)
        dataset.set_column("PrecursorMZ", [float(v) if np.isfinite(v) else None
                                           for v in self.precursor_mz[positions]])
        return dataset, np.array([len(p) for p in peaks], dtype=np.int64)

    def close(self):
        self._index = {}
        self.dataset.close()
