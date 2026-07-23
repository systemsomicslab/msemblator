"""Central project paths for external tools, models, configuration, and runtime data."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
EXTERNAL_DIR = PROJECT_ROOT / "external"
MODELS_DIR = PROJECT_ROOT / "models"
WORK_DIR = PROJECT_ROOT / "work"
MSFINDER_DIR = EXTERNAL_DIR / "msfinder"
SIRIUS_DIR = EXTERNAL_DIR / "sirius"
METFRAG_DIR = EXTERNAL_DIR / "metfrag"
FORMULA_MODEL_DIR = MODELS_DIR / "formula"
STRUCTURE_MODEL_DIR = MODELS_DIR / "structure"
PARAMETER_FILE = CONFIG_DIR / "msemblator_parameter_file.yaml"

def ensure_runtime_directories():
    """Create writable runtime directories without modifying bundled tools."""
    for path in (WORK_DIR, WORK_DIR / "formula", WORK_DIR / "structure"):
        path.mkdir(parents=True, exist_ok=True)

def validate_runtime(mode):
    """Raise a clear error when a required local runtime asset is missing."""
    required = [PARAMETER_FILE, SIRIUS_DIR / "sirius.exe"]
    model_dir = FORMULA_MODEL_DIR if mode in (1, 2) else STRUCTURE_MODEL_DIR
    if not any(model_dir.glob("*.pkl")):
        required.append(model_dir / "*.pkl")
    if not any(MSFINDER_DIR.glob("MSFINDER*/MsfinderConsoleApp.exe")):
        required.append(MSFINDER_DIR / "MSFINDER*/MsfinderConsoleApp.exe")
    if mode in (2, 3):
        required.extend([SIRIUS_DIR / "sirius_structure_db.siriusdb", METFRAG_DIR / "MetFragCommandLine-2.5.0.jar", METFRAG_DIR / "metfrag_StructureDB.txt", MSFINDER_DIR / "MsfinderStructureDB_all.txt"])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing runtime assets:\n" + "\n".join(missing))
