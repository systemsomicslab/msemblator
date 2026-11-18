"""Write MS-FINDER method (parameter) files and check that tolerances are in ppm.

MS-FINDER reads Ms1Tolerance / Ms2Tolerance in the unit given by MassToleranceType
(Da or ppm). MS-Emblator's yaml gives tolerances in ppm, so every MS-FINDER run must
use MassToleranceType=ppm; otherwise e.g. "20" is read as +-20 Da.
"""
import logging
import math

MASS_TOLERANCE_TYPE = "ppm"


def _check_ppm_value(name, value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"MS-FINDER {name} must be a number in ppm, got {value!r}.")
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"MS-FINDER {name} must be a positive number in ppm, got {value!r}.")
    return number


def read_msfinder_method(method_path):
    """Return {key: [values...]} for the key=value lines of a method file."""
    params = {}
    with open(method_path, "r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            params.setdefault(key.strip(), []).append(value.strip())
    return params


def write_msfinder_method(method_path, settings):
    """
    Set each key in `settings` in the method file. Every existing line for the key is
    replaced; a key missing from the file is appended so MS-FINDER never falls back
    to its own default.
    """
    with open(method_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    written = set()
    new_lines = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if key in settings:
            new_lines.append(f"{key}={settings[key]}\n")
            written.add(key)
        else:
            new_lines.append(line)

    missing = [key for key in settings if key not in written]
    if missing:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        new_lines.extend(f"{key}={settings[key]}\n" for key in missing)

    with open(method_path, "w", encoding="utf-8") as file:
        file.writelines(new_lines)


def verify_msfinder_ppm(method_path, ms1_ppm, ms2_ppm):
    """Raise ValueError unless the method file makes MS-FINDER use the given ppm tolerances."""
    params = read_msfinder_method(method_path)
    expected = {
        "MassToleranceType": MASS_TOLERANCE_TYPE,
        "Ms1Tolerance": float(ms1_ppm),
        "Ms2Tolerance": float(ms2_ppm),
    }
    problems = []
    for key, want in expected.items():
        values = params.get(key)
        if not values:
            problems.append(f"{key} is missing")
            continue
        for value in values:
            if isinstance(want, float):
                try:
                    ok = math.isclose(float(value), want)
                except ValueError:
                    ok = False
            else:
                ok = value.lower() == want
            if not ok:
                problems.append(f"{key}={value} (expected {want})")
    if problems:
        raise ValueError(
            f"MS-FINDER method file is not set to ppm tolerances ({method_path}): "
            + "; ".join(problems)
        )


def apply_msfinder_settings(method_path, msfinder_config, extra_settings=None):
    """
    Write ppm tolerances and halogen settings from one yaml msfinder block into the
    method file, then read the file back and check it before MS-FINDER is started.
    A method file that is not in ppm (Da, or no MassToleranceType line) is rewritten to ppm.
    """
    ms1_ppm = _check_ppm_value("MS1_ppm", msfinder_config["MS1_ppm"])
    ms2_ppm = _check_ppm_value("MS2_ppm", msfinder_config["MS2_ppm"])
    halogen = str(msfinder_config["halogen"])

    settings = {
        "MassToleranceType": MASS_TOLERANCE_TYPE,
        "Ms1Tolerance": msfinder_config["MS1_ppm"],
        "Ms2Tolerance": msfinder_config["MS2_ppm"],
        "Fcheck": halogen,
        "ClCheck": halogen,
        "BrCheck": halogen,
        "Icheck": halogen,
    }
    if extra_settings:
        settings.update(extra_settings)

    current = read_msfinder_method(method_path).get("MassToleranceType")
    if not current or any(value.lower() != MASS_TOLERANCE_TYPE for value in current):
        logging.warning(
            "MS-FINDER method file %s has MassToleranceType=%s; rewriting it to %s.",
            method_path, ",".join(current) if current else "(missing)", MASS_TOLERANCE_TYPE,
        )

    write_msfinder_method(method_path, settings)
    verify_msfinder_ppm(method_path, ms1_ppm, ms2_ppm)
