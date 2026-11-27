"""Molecular formulas: parsing, Hill-order keys and monoisotopic masses.

Two formulas match when they have the same elements with the same counts:
"C8H10N4O2" and "H10C8N4O2" give the same key. Charges, isotopes, brackets and
other notations are rejected instead of being guessed.
"""
from __future__ import annotations

import re
from collections import Counter

from msemblator.chemistry.chem_data import atomic_to_exact_mass

ELECTRON_MASS = 0.000548579909

_FORMULA_PATTERN = re.compile(r"(?:[A-Z][a-z]?\d*)+")
_ELEMENT_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(text) -> dict[str, int] | None:
    """Element counts of a plain formula ("C6H12O6"), or None if it is not one."""
    text = re.sub(r"\s+", "", str(text or ""))
    if not text or not _FORMULA_PATTERN.fullmatch(text):
        return None
    counts: Counter = Counter()
    for element, number in _ELEMENT_PATTERN.findall(text):
        counts[element] += int(number) if number else 1
    counts = {element: n for element, n in counts.items() if n > 0}
    return counts or None


def has_known_masses(counts: dict[str, int]) -> bool:
    """True when every element is in atomic_data.yaml (needed for masses only, not for matching)."""
    return all(element in atomic_to_exact_mass for element in counts)


def formula_key(counts: dict[str, int]) -> str:
    """Hill order: C, H, then the other elements alphabetically (all alphabetical without C)."""
    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else []) + sorted(e for e in counts if e not in ("C", "H"))
    else:
        order = sorted(counts)
    return "".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in order if counts[e] > 0)


def normalize_formula(text) -> str | None:
    counts = parse_formula(text)
    return formula_key(counts) if counts else None


def combine(counts: dict[str, int], delta: dict[str, int]) -> dict[str, int] | None:
    """counts + delta (delta may be negative); None if an element count would become negative."""
    result = Counter(counts)
    for element, n in delta.items():
        result[element] += n
    if any(n < 0 for n in result.values()):
        return None
    result = {e: n for e, n in result.items() if n > 0}
    return result or None


def monoisotopic_mass(counts: dict[str, int]) -> float:
    return sum(atomic_to_exact_mass[e] * n for e, n in counts.items())
