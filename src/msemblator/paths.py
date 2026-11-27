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

# Normalization pipelines of the default tool combinations (SIRIUS, MS-FINDER,
# msbuddy / MS-FINDER, SIRIUS, MetFrag). The files of every tool, including FIDDLE,
# are listed in toolsets.TOOLS, which validate_runtime uses.
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

def ensure_runtime_directories():
    """Create writable runtime directories without modifying bundled tools."""
    for path in (WORK_DIR, WORK_DIR / "formula", WORK_DIR / "structure"):
        path.mkdir(parents=True, exist_ok=True)

def validate_runtime(mode, formula_tools=None, structure_tools=None):
    """Raise a clear error when a runtime asset of the selected tools is missing.

    formula_tools / structure_tools are toolsets.ToolSelection objects; None uses
    the default combination of config/toolsets.yaml (SIRIUS, MS-FINDER, msbuddy and
    MS-FINDER, SIRIUS, MetFrag).
    """
    from msemblator import toolsets

    missing = [] if PARAMETER_FILE.exists() else [str(PARAMETER_FILE)]
    if mode in (1, 2):
        missing += toolsets.missing_assets(formula_tools or toolsets.resolve_selection(toolsets.FORMULA))
    if mode in (2, 3):
        missing += toolsets.missing_assets(structure_tools or toolsets.resolve_selection(toolsets.STRUCTURE))
    missing = list(dict.fromkeys(missing))
    if missing:
        raise FileNotFoundError("Missing runtime assets:\n" + "\n".join(missing))
