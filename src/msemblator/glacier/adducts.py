"""Adduct names: normalization, parsing, ion m/z and the neutral part of an adduct.

The adducts themselves come from the configuration (glacier.libraries and
glacier.adduct_aliases); this module only knows the notation
``[nM<terms>]<charge><sign>``, e.g. ``[M-H2O+H]+`` or ``[2M+Na]+``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from msemblator.glacier.formula import ELECTRON_MASS, combine, monoisotopic_mass, parse_formula

_ADDUCT_PATTERN = re.compile(r"^\[(\d*)M((?:[+-]\d*[A-Za-z][A-Za-z0-9]*)*)\](\d*)([+-])$")
_TERM_PATTERN = re.compile(r"([+-])(\d*)([A-Za-z][A-Za-z0-9]*)")

# Abbreviations that are not formulas themselves.
GROUP_ABBREVIATIONS = {
    "FA": "CH2O2", "ACN": "C2H3N", "Hac": "C2H4O2", "HAc": "C2H4O2", "AcOH": "C2H4O2",
    "MeOH": "CH4O", "TFA": "C2HF3O2", "DMSO": "C2H6OS", "IsoProp": "C3H8O",
}


def clean_adduct_text(text) -> str:
    """Remove spaces and write a charge of 1 as "+" / "-" ("[M+H]1+" -> "[M+H]+")."""
    text = re.sub(r"\s+", "", str(text or ""))
    return re.sub(r"\]1([+-])$", r"]\1", text)


@dataclass(frozen=True)
class AdductDefinition:
    name: str
    n_molecules: int
    terms: tuple[tuple[int, tuple[tuple[str, int], ...]], ...]  # (signed count, element counts)
    charge: int  # signed

    def ion_mz(self, neutral: dict[str, int]) -> float:
        mass = self.n_molecules * monoisotopic_mass(neutral)
        for count, elements in self.terms:
            mass += count * monoisotopic_mass(dict(elements))
        return (mass - self.charge * ELECTRON_MASS) / abs(self.charge)

    @property
    def neutral_modification(self) -> dict[str, int]:
        """Element change of every term except the last one, which carries the charge.

        [M-H2O+H]+ -> {"H": -2, "O": -1}; [M+H]+, [M-H]-, [M+NH4]+ -> {}.
        """
        delta: dict[str, int] = {}
        for count, elements in self.terms[:-1]:
            for element, n in elements:
                delta[element] = delta.get(element, 0) + count * n
        return {e: n for e, n in delta.items() if n != 0}


def parse_adduct(name: str) -> AdductDefinition | None:
    """Parse ``[nM...]z±``; None if the notation is not understood."""
    text = clean_adduct_text(name)
    match = _ADDUCT_PATTERN.fullmatch(text)
    if not match:
        return None
    n_text, terms_text, charge_text, sign = match.groups()
    terms = []
    for term_sign, count_text, group in _TERM_PATTERN.findall(terms_text):
        elements = parse_formula(GROUP_ABBREVIATIONS.get(group, group))
        if elements is None:
            return None
        count = int(count_text) if count_text else 1
        terms.append(((1 if term_sign == "+" else -1) * count, tuple(sorted(elements.items()))))
    if not terms:
        return None
    charge = (int(charge_text) if charge_text else 1) * (1 if sign == "+" else -1)
    return AdductDefinition(text, int(n_text) if n_text else 1, tuple(terms), charge)


def apply_modification(neutral: dict[str, int], adduct: AdductDefinition, direction: int = 1):
    """Neutral formula with the adduct's neutral modification added (direction=1) or removed (-1)."""
    delta = {e: direction * n for e, n in adduct.neutral_modification.items()}
    return combine(neutral, delta) if delta else dict(neutral)


class AdductResolver:
    """Map query and library adduct spellings to the adduct names of the configuration."""

    def __init__(self, supported: Iterable[str], aliases: dict[str, str] | None = None):
        self.supported = tuple(supported)
        self._map: dict[str, str] = {}
        for name in self.supported:
            self._map[clean_adduct_text(name)] = name
        for alias, target in (aliases or {}).items():
            target_clean = clean_adduct_text(target)
            if target_clean in self._map:
                self._map[clean_adduct_text(alias)] = self._map[target_clean]

    def resolve(self, raw) -> str | None:
        return self._map.get(clean_adduct_text(raw))
