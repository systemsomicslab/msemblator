"""Settings of the GLACIER module (config/glacier.yaml, top-level key ``glacier``).

Relative paths are resolved from the repository root. Nothing in the core
modules refers to a specific adduct, file name or collision energy: all of them
come from this configuration.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from msemblator.paths import CONFIG_DIR, PROJECT_ROOT

DEFAULT_CONFIG_PATH = CONFIG_DIR / "glacier.yaml"

# Feature label of each similarity method (GLACIER_<label>_CE10, ...).
METHOD_LABELS = {
    "cosine": "Cosine",
    "reverse_cosine": "Reverse",
    "modified_cosine": "ModifiedCosine",
    "bonanza": "Bonanza",
    "entropy": "Entropy",
}
FORMULA_BASES = ("auto", "neutral", "modified")
PRECURSOR_REMOVAL_SPECTRA = ("measured", "both")
_DEFAULT_PRECURSOR_REMOVAL = {"enabled": True,"tolerance": 20.0, "tolerance_unit": "ppm", "spectra": "measured"}

_DEFAULT_QUERY_COLUMNS = {
    "name": ["Name"],
    "adduct": ["AdductType"],
    "formula": ["Formula"],
    "precursor_mz": ["PrecursorMZ"],
}
_DEFAULT_LIBRARY_COLUMNS = {
    "spec_id": ["SPECID", "SpecID", "SCANS", "TITLE"],
    "smiles": ["SMILES"],
    "inchikey": ["INCHIKEY", "InChIKey"],
    "formula": ["FORMULA", "Formula"],
    "adduct": ["ION", "AdductType", "PredictionAdduct"],
    "collision_energy": ["PredictionCollisionEnergyEV", "CollisionEnergy", "COLLISIONENERGY"],
    "precursor_mz": ["PrecursorMZ", "PEPMASS", "PredictionPrecursorMZ"],
}


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


@dataclass
class GlacierConfig:
    library_dir: Path
    libraries: dict[str, str]
    adduct_aliases: dict[str, str] = field(default_factory=dict)
    formula_basis: dict[str, str] = field(default_factory=lambda: {"default": "auto"})
    collision_energies: tuple[float, ...] = (10.0, 40.0)
    ms2_tolerance: float = 20.0
    ms2_tolerance_unit: str = "ppm"
    intensity_exponent: float = 1.0
    similarities: tuple[str, ...] = ("cosine", "reverse_cosine")
    entropy: dict = field(default_factory=lambda: {"clean_spectra": True, "noise_threshold": 0.01})
    precursor_peak_removal: dict = field(default_factory=lambda: dict(_DEFAULT_PRECURSOR_REMOVAL))
    threads: int = 0
    adduct_features: tuple[str, ...] = ()
    query_columns: dict[str, list[str]] = field(default_factory=lambda: copy.deepcopy(_DEFAULT_QUERY_COLUMNS))
    library_columns: dict[str, list[str]] = field(default_factory=lambda: copy.deepcopy(_DEFAULT_LIBRARY_COLUMNS))
    source: str = ""

    def __post_init__(self):
        self.validate()

    # -- derived values -----------------------------------------------------------------
    @property
    def mz_tolerance_da(self) -> float:
        return self.ms2_tolerance if self.ms2_tolerance_unit == "da" else 0.0

    @property
    def mz_tolerance_ppm(self) -> float:
        return self.ms2_tolerance if self.ms2_tolerance_unit == "ppm" else 0.0

    def library_path(self, adduct: str) -> Path | None:
        name = self.libraries.get(adduct)
        return resolve_path(name, self.library_dir) if name else None

    def formula_basis_for(self, adduct: str) -> str:
        return self.formula_basis.get(adduct, self.formula_basis.get("default", "auto"))

    def method_label(self, method: str) -> str:
        return METHOD_LABELS[method]

    def removes_precursor(self, side: str) -> bool:
        """True when the precursor peak is removed from "measured" or "predicted" spectra."""
        removal = self.precursor_peak_removal
        if not removal["enabled"]:
            return False
        return side == "measured" or removal["spectra"] == "both"

    def to_dict(self) -> dict:
        return {
            "library_dir": str(self.library_dir),
            "libraries": dict(self.libraries),
            "adduct_aliases": dict(self.adduct_aliases),
            "formula_basis": dict(self.formula_basis),
            "collision_energies": list(self.collision_energies),
            "ms2_tolerance": self.ms2_tolerance,
            "ms2_tolerance_unit": self.ms2_tolerance_unit,
            "intensity_exponent": self.intensity_exponent,
            "similarities": list(self.similarities),
            "entropy": dict(self.entropy),
            "precursor_peak_removal": dict(self.precursor_peak_removal),
            "threads": self.threads,
            "adduct_features": list(self.adduct_features),
            "query_columns": copy.deepcopy(self.query_columns),
            "library_columns": copy.deepcopy(self.library_columns),
            "source": self.source,
        }

    # -- checks -------------------------------------------------------------------------
    def validate(self) -> None:
        self.library_dir = Path(self.library_dir)
        self.ms2_tolerance_unit = str(self.ms2_tolerance_unit).strip().lower()
        if self.ms2_tolerance_unit not in ("ppm", "da"):
            raise ValueError(f"ms2_tolerance_unit must be ppm or Da, got {self.ms2_tolerance_unit!r}")
        self.ms2_tolerance = float(self.ms2_tolerance)
        if not math.isfinite(self.ms2_tolerance) or self.ms2_tolerance <= 0:
            raise ValueError("ms2_tolerance must be a positive number")
        self.intensity_exponent = float(self.intensity_exponent)
        if not math.isfinite(self.intensity_exponent) or self.intensity_exponent <= 0:
            raise ValueError("intensity_exponent must be a positive number")
        if not self.libraries:
            raise ValueError("glacier.libraries is empty: register at least one adduct and its MSDS file")
        energies = []
        for value in self.collision_energies:
            energy = float(value)
            if not math.isfinite(energy) or energy <= 0:
                raise ValueError(f"collision_energies must be positive numbers, got {value!r}")
            energies.append(energy)
        if len(set(energies)) != len(energies):
            raise ValueError("collision_energies contains duplicates")
        self.collision_energies = tuple(energies)
        methods = [str(m).strip().lower() for m in self.similarities]
        unknown = [m for m in methods if m not in METHOD_LABELS]
        if unknown:
            raise ValueError(f"Unknown similarities {unknown}; choose from {list(METHOD_LABELS)}")
        if not methods or len(set(methods)) != len(methods):
            raise ValueError("similarities must list at least one method, each once")
        self.similarities = tuple(methods)
        for adduct, basis in self.formula_basis.items():
            if basis not in FORMULA_BASES:
                raise ValueError(f"formula_basis[{adduct!r}] must be one of {FORMULA_BASES}, got {basis!r}")
        removal = {**_DEFAULT_PRECURSOR_REMOVAL, **(self.precursor_peak_removal or {})}
        unknown = set(removal) - set(_DEFAULT_PRECURSOR_REMOVAL)
        if unknown:
            raise ValueError(f"Unknown precursor_peak_removal settings: {sorted(unknown)}")
        if not isinstance(removal["enabled"], bool):
            raise ValueError("precursor_peak_removal.enabled must be true or false")
        removal["tolerance"] = float(removal["tolerance"])
        if not math.isfinite(removal["tolerance"]) or removal["tolerance"] <= 0:
            raise ValueError("precursor_peak_removal.tolerance must be a positive number")
        removal["tolerance_unit"] = str(removal["tolerance_unit"]).strip().lower()
        if removal["tolerance_unit"] not in ("ppm", "da"):
            raise ValueError("precursor_peak_removal.tolerance_unit must be ppm or Da")
        if removal["spectra"] not in PRECURSOR_REMOVAL_SPECTRA:
            raise ValueError(f"precursor_peak_removal.spectra must be one of {PRECURSOR_REMOVAL_SPECTRA}")
        self.precursor_peak_removal = removal
        self.threads = int(self.threads)
        if self.threads < 0:
            raise ValueError("threads must be 0 (library default) or a positive number")
        self.adduct_features = tuple(self.adduct_features)
        for columns, defaults in ((self.query_columns, _DEFAULT_QUERY_COLUMNS),
                                  (self.library_columns, _DEFAULT_LIBRARY_COLUMNS)):
            for key, value in defaults.items():
                columns.setdefault(key, list(value))
                if isinstance(columns[key], str):
                    columns[key] = [columns[key]]


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> GlacierConfig:
    """Read the ``glacier`` block of a YAML file; ``overrides`` replace single settings."""
    path = resolve_path(path) if path else DEFAULT_CONFIG_PATH
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    settings = dict(data.get("glacier", data))
    settings.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return config_from_dict(settings, source=str(path))


def config_from_dict(settings: dict, source: str = "") -> GlacierConfig:
    settings = copy.deepcopy(settings)
    if "library_dir" not in settings:
        raise ValueError("glacier.library_dir is not set")
    settings["library_dir"] = resolve_path(settings["library_dir"])
    settings["libraries"] = {str(k): str(v) for k, v in (settings.get("libraries") or {}).items()}
    settings["adduct_aliases"] = {str(k): str(v) for k, v in (settings.get("adduct_aliases") or {}).items()}
    basis = settings.get("formula_basis") or {"default": "auto"}
    settings["formula_basis"] = {"default": basis} if isinstance(basis, str) else dict(basis)
    known = set(GlacierConfig.__dataclass_fields__)
    unknown = set(settings) - known
    if unknown:
        raise ValueError(f"Unknown glacier settings: {sorted(unknown)}")
    settings["source"] = source
    return GlacierConfig(**settings)
