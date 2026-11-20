#!/usr/bin/env python3
"""Build the small demo MSP used to test the SIRIUS-free annotation scripts.

Source: the demo spectra shipped with msfiddle 2.3.0 (msfiddle/demo/input_msms.mgf,
EMBL-MCF 2.0 HRMS library, distributed in the Apache-2.0 FIDDLE repository).
They are written unchanged (Instrument_type: Orbitrap is added, as for the
FIDDLE CLI default). A few SYNTHETIC records derived from them follow; they
exist only to exercise every status path and are not real measurements:
precursor m/z values are recomputed from the reference formula for the new
adduct, fragment peaks are copied unchanged.

Usage:
    python scripts/sirius_free/make_demo_msp.py --mgf <msfiddle>/demo/input_msms.mgf --output tests/fixtures/sirius_free_demo.msp
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

MONO = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.99491461956,
        "P": 30.97376163, "S": 31.97207100, "Cl": 34.96885268, "Na": 22.9897692809,
        "F": 18.99840322, "Br": 78.9183371, "I": 126.904473}
ELECTRON = 0.00054857990946
PROTON = MONO["H"] - ELECTRON
# Mass added to the neutral molecule (before the electron correction) and charge sign.
ADDUCT_DELTA = {
    "[M+H]+": (MONO["H"], 1),
    "[M+NH4]+": (MONO["N"] + 4 * MONO["H"], 1),
    "[M+Na]+": (MONO["Na"], 1),
    "[M+K]+": (38.9637064864, 1),
    "[M-H]-": (-MONO["H"], -1),
    "[M+Cl]-": (MONO["Cl"], -1),
    "[M+FA-H]-": (MONO["C"] + MONO["H"] + 2 * MONO["O"], -1),
}


def neutral_mass(formula: str) -> float:
    return sum(MONO[el] * (int(n) if n else 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula))


def precursor_mz(formula: str, adduct: str) -> float:
    delta, sign = ADDUCT_DELTA[adduct]
    return neutral_mass(formula) + delta - sign * ELECTRON


def read_mgf(path: Path) -> list[dict]:
    spectra, current = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line == "BEGIN IONS":
            current = {"params": {}, "peaks": []}
        elif line == "END IONS":
            spectra.append(current)
            current = None
        elif current is not None and "=" in line and not line[0].isdigit():
            key, value = line.split("=", 1)
            current["params"][key.upper()] = value
        elif current is not None and line:
            mz, intensity = line.split()[:2]
            current["peaks"].append((mz, intensity))
    return spectra


def block(name, precursor, adduct, peaks, formula, ce="", instrument_type="Orbitrap", comment=""):
    lines = [f"NAME: {name}", f"PRECURSORMZ: {precursor}", f"PRECURSORTYPE: {adduct}"]
    if instrument_type:
        lines.append(f"Instrument_type: {instrument_type}")
    if ce:
        lines.append(f"Collision_energy: {ce}")
    lines += [f"FORMULA: {formula}", f"COMMENT: {comment}", f"Num Peaks: {len(peaks)}"]
    lines += [f"{mz}\t{intensity}" for mz, intensity in peaks]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mgf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    spectra = read_mgf(args.mgf)
    blocks = []
    for s in spectra:  # real spectra, unchanged
        p = s["params"]
        blocks.append(block(p["TITLE"], p["PRECURSOR_MZ"], p["PRECURSOR_TYPE"], s["peaks"], p.get("FORMULA", ""),
                            ce=p.get("COLLISION_ENERGY", ""), comment="msfiddle demo (EMBL-MCF 2.0)"))

    base = next(s for s in spectra if s["params"].get("FORMULA") == "C7H10O5")
    formula, peaks = base["params"]["FORMULA"], base["peaks"]
    synthetic = [
        ("SYN_no_ce", "[M-H]-", "", "Orbitrap", peaks),
        ("SYN_cl", "[M+Cl]-", "50.0", "Orbitrap", peaks),
        ("SYN_formate", "[M+CHO2]-", "50.0", "Orbitrap", peaks),
        ("SYN_pos_h", "[M+H]+", "35.0", "Q-TOF", peaks),
        ("SYN_pos_nh4", "[M+NH4]+", "35.0", "Q-TOF", peaks),
        ("SYN_pos_na", "[M+Na]+", "NCE=35%", "HCD", peaks),
        ("SYN_few_peaks", "[M-H]-", "50.0", "Orbitrap", peaks[:3]),
        ("SYN_stepped_ce", "[M-H]-", "20,40", "Orbitrap", peaks),
        ("SYN_unknown_adduct", "[M+K]+", "35.0", "Q-TOF", peaks),
        ("SYN_no_instrument", "[M-H]-", "50.0", "", peaks),
    ]
    for name, adduct, ce, instrument, spectrum_peaks in synthetic:
        mz_adduct = "[M+FA-H]-" if adduct == "[M+CHO2]-" else adduct
        mz = f"{precursor_mz(formula, mz_adduct):.5f}"
        blocks.append(block(name, mz, adduct, spectrum_peaks, formula, ce=ce, instrument_type=instrument,
                            comment="SYNTHETIC test record derived from EMBL_MCF_2_0_HRMS_Library000465"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Wrote {len(blocks)} spectra ({len(spectra)} real, {len(synthetic)} synthetic) to {args.output}")


if __name__ == "__main__":
    main()
