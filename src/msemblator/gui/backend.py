"""Non-UI helpers of the MS-Emblator GUI (kept separate so they can be tested).

The GUI does not reimplement the workflow: it starts the msemblator command
line (python -m msemblator.cli.msemblator) as a subprocess, streams its output
and reads the CSV files it writes.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from msemblator import paths

MODES = {
    1: ("Formula", "molecular formulas only"),
    2: ("Formula + structure", "formulas, then structures"),
    3: ("Structure only", "needs formulas in the MSP"),
}
RESULT_NAMES = ("formula_summary", "formula_score", "structure_summary", "structure_score", "glacier_spectra")
RESULT_SUFFIX = {"glacier_spectra": ".csv.gz"}
STEP_PATTERN = re.compile(r"^(?:Running (formula|structure) elucidation|(\S+) processing (start|complete))")


# --- input ----------------------------------------------------------------------

@dataclass
class SpectrumInfo:
    index: int
    name: str
    precursor_mz: float | None
    adduct: str
    formula: str
    mz: list
    intensity: list


def read_spectra(path) -> list[SpectrumInfo]:
    """Spectra of an MSP file for the spectrum list and viewer."""
    from msemblator.sirius_free.msp_io import read_msp_records

    return [SpectrumInfo(i + 1, r.name, r.precursor_mz, r.precursor_type_raw, r.reference_formula,
                         list(r.mz), list(r.intensity))
            for i, r in enumerate(read_msp_records(path))]


# --- command --------------------------------------------------------------------

@dataclass
class RunRequest:
    input_msp: str
    output_dir: str
    mode: int
    formula_tools: str | None = None
    structure_tools: str | None = None
    toolsets: str | None = None
    sirius_user: str | None = None


def cli_arguments(request: RunRequest) -> list[str]:
    args = ["--input", request.input_msp, "--output", request.output_dir, "--mode", str(request.mode)]
    if request.mode in (1, 2) and request.formula_tools:
        args += ["--formula_tools", request.formula_tools]
    if request.mode in (2, 3) and request.structure_tools:
        args += ["--structure_tools", request.structure_tools]
    if request.toolsets:
        args += ["--toolsets", request.toolsets]
    return args


def build_command(request: RunRequest, python: str | None = None) -> list[str]:
    """The command the GUI runs. The SIRIUS account goes through the environment, never argv."""
    from msemblator.fiddle_setup import console_python   # python.exe even when the GUI runs under pythonw.exe

    return [python or console_python(), "-u", "-m", "msemblator.cli.msemblator"] + cli_arguments(request)


def display_command(request: RunRequest) -> str:
    """The same run as a command for the PowerShell prompt."""
    def quote(text):
        return f'"{text}"' if (" " in text or not text) else text
    return ".\\msemblator.cmd " + " ".join(quote(a) for a in cli_arguments(request))


def run_environment(request: RunRequest, sirius_password: str | None = None, base: dict | None = None) -> dict:
    env = dict(os.environ if base is None else base)
    src = str(paths.PROJECT_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH")) if p)
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    if request.sirius_user:
        env["SIRIUS_USER"] = request.sirius_user
        if sirius_password:
            env["SIRIUS_PASS"] = sirius_password
    return env


def decode_line(raw: bytes) -> str:
    """Python prints UTF-8; SIRIUS and MS-FINDER may print in the Windows code page."""
    for encoding in ("utf-8", "cp932"):
        try:
            return raw.decode(encoding).rstrip("\r\n")
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").rstrip("\r\n")


def step_of(line: str) -> str | None:
    """'MetFrag processing start' -> 'MetFrag'; 'Running formula elucidation...' -> 'Formula elucidation'."""
    match = STEP_PATTERN.match(line.strip())
    if not match:
        return None
    if match.group(1):
        return f"{match.group(1).capitalize()} elucidation"
    return match.group(2) if match.group(3) == "start" else None


# --- shortcuts ------------------------------------------------------------------

SHORTCUT_SCRIPT = paths.PROJECT_ROOT / "scripts" / "create_shortcut.ps1"


def create_shortcuts(folders=None) -> list[str]:
    """Desktop and Start menu shortcuts (or into `folders`) that start msemblator_gui.cmd.

    Uses scripts/create_shortcut.ps1, the same script as scripts/create_shortcut.cmd.
    """
    import subprocess

    base = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SHORTCUT_SCRIPT)]
    commands = [base + ["-Folder", str(f)] for f in folders] if folders else [base]
    created = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode != 0:
            message = " ".join(decode_line(line) for line in result.stderr.splitlines() if line.strip())
            raise RuntimeError(message or f"PowerShell exited with code {result.returncode}")
        created += [decode_line(line) for line in result.stdout.splitlines() if line.strip()]
    return created


# --- parameter file -------------------------------------------------------------

def read_parameters(path=None) -> dict:
    import yaml

    with open(path or paths.PARAMETER_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _format_value(value) -> str:
    if value is None or value == "":
        return "null"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_.\-]+", text) and text.lower() not in ("true", "false", "null", "yes", "no"):
        return text
    return "'" + text.replace("'", "''") + "'"


def _split_comment(rest: str) -> tuple[str, str]:
    """' 10   # note' -> (' 10', '   # note'), ignoring # inside quotes."""
    quote = None
    for i, ch in enumerate(rest):
        if ch in "'\"" and quote in (None, ch):
            quote = None if quote else ch
        elif ch == "#" and quote is None and (i == 0 or rest[i - 1] in " \t"):
            j = i
            while j > 0 and rest[j - 1] in " \t":
                j -= 1
            return rest[:j], rest[j:]
    return rest, ""


def update_parameters(values: dict, path=None) -> int:
    """Write {(section, tool, key): value} into the YAML file, keeping comments and layout.

    Only existing scalar entries are changed. Returns the number of changed lines.
    """
    path = Path(path or paths.PARAMETER_FILE)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    stack: list[tuple[int, str]] = []
    changed = 0
    for i, line in enumerate(lines):
        body = line.rstrip("\r\n")
        newline = line[len(body):]
        stripped = body.lstrip(" ")
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(body) - len(stripped)
        match = re.match(r"([A-Za-z0-9_]+):(.*)$", stripped)
        if not match:
            continue
        key, rest = match.groups()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        key_path = tuple(k for _, k in stack) + (key,)
        value_text, comment = _split_comment(rest)
        if not value_text.strip():
            stack.append((indent, key))           # a mapping such as "msfinder:"
            continue
        if key_path in values:
            new = f"{' ' * indent}{key}: {_format_value(values[key_path])}{comment}{newline}"
            if new != line:
                lines[i] = new
                changed += 1
    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def parse_entry(text: str, original):
    """Value typed in the settings dialog -> YAML value of the same kind as the original."""
    text = text.strip()
    if text == "" or text.lower() == "null":
        return None
    if isinstance(original, bool):
        return text.lower() in ("1", "true", "yes", "on")
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    return text


# --- results --------------------------------------------------------------------

def latest_results(output_dir, since: float | None = None) -> dict[str, Path]:
    """Newest formula/structure summary and score CSV in output_dir (name.csv, name_1.csv, ...)."""
    found = {}
    folder = Path(output_dir)
    if not folder.is_dir():
        return found
    for name in RESULT_NAMES:
        suffix = re.escape(RESULT_SUFFIX.get(name, ".csv"))
        pattern = re.compile(rf"{name}(_\d+)?{suffix}$")
        files = [p for p in folder.iterdir() if pattern.fullmatch(p.name)]
        if since is not None:
            files = [p for p in files if p.stat().st_mtime >= since - 1]
        if files:
            found[name] = max(files, key=lambda p: p.stat().st_mtime)
    return found


def load_results(files: dict) -> dict:
    import pandas as pd

    return {name: pd.read_csv(path, dtype={"filename": str}) for name, path in files.items()}


def structure_png(smiles: str, width: int = 260, height: int = 200) -> bytes | None:
    """2D depiction of a SMILES as PNG bytes (RDKit), or None if it cannot be parsed."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Draw import rdMolDraw2D

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        drawer.drawOptions().clearBackground = True
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None


def short_inchikey(smiles: str) -> str:
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToInchiKey(mol)[:14] if mol is not None else ""
    except Exception:
        return ""


def glacier_spectra_for(table, filename: str, smiles: str) -> list[dict]:
    """GLACIER predicted spectra (one per collision energy) of a structure candidate.

    The candidate is matched on its Short InChIKey, so stereoisomers share spectra;
    for each energy the best-scoring spectrum is returned.
    """
    import pandas as pd

    if table is None or table.empty or not smiles:
        return []
    key = short_inchikey(smiles)
    rows = table[(table["filename"].astype(str) == str(filename)) & (table["Short_InChIKey"] == key)]
    if rows.empty:
        return []
    score_columns = [c for c in ("entropy", "cosine", "reverse_cosine") if c in rows]
    if score_columns:
        rows = rows.sort_values(score_columns[0], ascending=False)
    out = []
    for energy, group in rows.groupby("collision_energy", sort=True, dropna=False):
        row = group.iloc[0]
        out.append({
            "collision_energy": None if pd.isna(energy) else float(energy),
            "scores": {c: float(row[c]) for c in score_columns if pd.notna(row[c])},
            "mz": [float(v) for v in str(row["mz"]).split(";") if v],
            "intensity": [float(v) for v in str(row["intensity"]).split(";") if v],
        })
    return out


def matched_peaks(mz_a, mz_b, ppm: float = 20.0):
    """Which peaks of a and of b have a partner within ppm (on the mean m/z), for highlighting."""
    import numpy as np

    a, b = np.asarray(mz_a, dtype=float), np.asarray(mz_b, dtype=float)
    if not len(a) or not len(b):
        return np.zeros(len(a), dtype=bool), np.zeros(len(b), dtype=bool)
    diff = np.abs(a[:, None] - b[None, :])
    close = diff <= (a[:, None] + b[None, :]) / 2 * ppm * 1e-6
    return close.any(axis=1), close.any(axis=0)


def candidates_for(scores, filename: str, value_column: str):
    """Ranked candidates of one spectrum from formula_score / structure_score."""
    import pandas as pd

    if scores is None or scores.empty or value_column not in scores:
        return []
    rows = scores[scores["filename"].astype(str) == str(filename)].sort_values("rank")
    out = []
    for _, row in rows.iterrows():
        tools = row.get("Used_Tool", "")
        out.append({
            "rank": int(row["rank"]),
            "candidate": str(row[value_column]),
            "score": float(row.get("rank_score", float("nan"))),
            "probability": float(row.get("softmax_score", float("nan"))),
            # "MS-FINDER_Rank:1,SIRIUS_Rank:3" -> "MS-FINDER #1, SIRIUS #3"
            "tools": "" if pd.isna(tools) else str(tools).replace("_Rank:", " #").replace(",", ", "),
        })
    return out
