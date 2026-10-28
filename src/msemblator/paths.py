"""Central project paths for external tools, models, configuration, and runtime data."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
EXTERNAL_DIR = PROJECT_ROOT / "external"
LIBRARY_DIR = PROJECT_ROOT / "library"
MODELS_DIR = PROJECT_ROOT / "models"
WORK_DIR = PROJECT_ROOT / "work"
MSFINDER_DIR = EXTERNAL_DIR / "msfinder"
MSFINDER_CONFIG_DIR = CONFIG_DIR / "msfinder"
SIRIUS_DIR = EXTERNAL_DIR / "sirius"
METFRAG_DIR = EXTERNAL_DIR / "metfrag"
METFRAG_CONFIG_DIR = CONFIG_DIR / "metfrag"
FORMULA_MODEL_DIR = MODELS_DIR / "formula"
STRUCTURE_MODEL_DIR = MODELS_DIR / "structure"
PARAMETER_FILE = CONFIG_DIR / "msemblator_parameter_file.yaml"

# Files loaded by the formula and structure workflows (summaries/ and scoring/).
FORMULA_MODEL_FILES = (
    "pipeline_msfinder_formula_score.pkl",
    "pipeline_msfinder_score_diff.pkl",
    "pipeline_sirius_score.pkl",
    "pipeline_sirius_score_diff.pkl",
    "pipeline_msbuddy_score.pkl",
    "pipeline_msbuddy_score_diff.pkl",
)
STRUCTURE_MODEL_FILES = (
    "pipeline_msfinder_score_diff.pkl",
    "pipeline_msfinder_structure_score.pkl",
    "pipeline_csi_fingerid.pkl",
    "pipeline_csi_fingerid_diff.pkl",
    "pipeline_metfrag_score.pkl",
    "pipeline_metfrag_score_diff.pkl",
)
# Final ranking models in order of preference; the first one found is used, so
# model sets distributed with older versions (CatBoost or LightGBM only) still work.
RANKER_MODEL_FILES = (
    "xgbranker_final_all.pkl",
    "catboostranker_final_all.pkl",
    "lgbmranker_final_all.pkl",
)

def find_ranker_model(model_dir):
    """Return the path of the preferred ranking model in model_dir."""
    for name in RANKER_MODEL_FILES:
        path = Path(model_dir) / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No ranking model found in {model_dir}: expected one of {', '.join(RANKER_MODEL_FILES)}")

def _missing_ranker_model(model_dir):
    if any((model_dir / name).exists() for name in RANKER_MODEL_FILES):
        return []
    return [f"{model_dir / RANKER_MODEL_FILES[0]} (or {', '.join(RANKER_MODEL_FILES[1:])})"]

def ensure_runtime_directories():
    """Create writable runtime directories without modifying bundled tools."""
    for path in (WORK_DIR, WORK_DIR / "formula", WORK_DIR / "structure"):
        path.mkdir(parents=True, exist_ok=True)

def validate_runtime(mode):
    """Raise a clear error when a required local runtime asset is missing."""
    required = [PARAMETER_FILE, SIRIUS_DIR / "sirius.exe", MSFINDER_CONFIG_DIR / "MsfinderConsoleApp_Param_formula.txt"]
    if not any(MSFINDER_DIR.glob("MSFINDER*/MsfinderConsoleApp.exe")):
        required.append(MSFINDER_DIR / "MSFINDER*/MsfinderConsoleApp.exe")
    if mode in (1, 2):
        required.extend(FORMULA_MODEL_DIR / name for name in FORMULA_MODEL_FILES)
    if mode in (2, 3):
        required.extend(STRUCTURE_MODEL_DIR / name for name in STRUCTURE_MODEL_FILES)
        required.extend([LIBRARY_DIR / "sirius_structure_db.siriusdb", METFRAG_DIR / "MetFragCommandLine-2.5.0.jar", LIBRARY_DIR / "metfrag_StructureDB.txt", LIBRARY_DIR / "MsfinderStructureDB_all.txt"])
        required.extend([MSFINDER_CONFIG_DIR / "MsfinderConsoleApp-Param2_structure.txt", METFRAG_CONFIG_DIR / "example_paramater.txt"])
    missing = [str(path) for path in required if not path.exists()]
    if mode in (1, 2):
        missing.extend(_missing_ranker_model(FORMULA_MODEL_DIR))
    if mode in (2, 3):
        missing.extend(_missing_ranker_model(STRUCTURE_MODEL_DIR))
    if missing:
        raise FileNotFoundError("Missing runtime assets:\n" + "\n".join(missing))
