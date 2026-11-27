"""Precursor peak removal (glacier.precursor_peak_removal; by default on, measured spectra only).

A peak is the precursor peak when |m/z - precursor m/z| <= tolerance, with the
tolerance in Da or in ppm of the precursor m/z. The precursor m/z is the
spectrum's own metadata value (PRECURSORMZ of the query MSP, PrecursorMZ of the
library). Spectra without a precursor m/z are left unchanged.
"""
from __future__ import annotations

import numpy as np

# Values of the precursor_peak_removal column of the query table.
REMOVAL_OFF = "off"
REMOVAL_REMOVED = "removed"
REMOVAL_NOT_FOUND = "not_found"
REMOVAL_NO_PRECURSOR_MZ = "no_precursor_mz"


def precursor_peak_mask(mz: np.ndarray, precursor_mz: float, removal: dict) -> np.ndarray:
    """True for the peaks within the precursor window."""
    mz = np.asarray(mz, dtype=float)
    if precursor_mz is None or not np.isfinite(precursor_mz):
        return np.zeros(len(mz), dtype=bool)
    if removal["tolerance_unit"] == "da":
        width = removal["tolerance"]
    else:
        width = precursor_mz * removal["tolerance"] * 1e-6
    return np.abs(mz - precursor_mz) <= width
