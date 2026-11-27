"""Annotation tool selection: which tools run in formula and structure elucidation.

A tool combination is usable only when a ranking model has been trained for it,
so the combinations are listed in config/toolsets.yaml (built-in defaults below
when the file is missing). ``resolve_selection`` turns a preset name ("sirius",
"fiddle", "all") or a tool list ("msfinder,msbuddy,fiddle") into a ToolSelection,
and ``missing_assets`` lists the files a selection needs that are not in place.

The same functions are meant for a later GUI: ``available_selections`` reports
every combination with its missing files.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from msemblator import paths

FORMULA, STRUCTURE = "formula", "structure"
STEPS = (FORMULA, STRUCTURE)


@dataclass(frozen=True)
class Tool:
    key: str
    label: str
    pipelines: tuple[str, ...] = ()   # score normalization pipelines (pipeline_*.pkl)


# Registry order is the order of the per-tool columns in the summary files.
TOOLS = {
    FORMULA: {
        "msfinder": Tool("msfinder", "MS-FINDER", ("pipeline_msfinder_formula_score.pkl", "pipeline_msfinder_score_diff.pkl")),
        "sirius": Tool("sirius", "SIRIUS", ("pipeline_sirius_score.pkl", "pipeline_sirius_score_diff.pkl")),
        "msbuddy": Tool("msbuddy", "msbuddy", ("pipeline_msbuddy_score.pkl", "pipeline_msbuddy_score_diff.pkl")),
        "fiddle": Tool("fiddle", "FIDDLE", ("pipeline_fiddle_score.pkl", "pipeline_fiddle_score_diff.pkl")),
    },
    STRUCTURE: {
        "msfinder": Tool("msfinder", "MS-FINDER", ("pipeline_msfinder_structure_score.pkl", "pipeline_msfinder_score_diff.pkl")),
        "sirius": Tool("sirius", "SIRIUS", ("pipeline_csi_fingerid.pkl", "pipeline_csi_fingerid_diff.pkl")),
        "metfrag": Tool("metfrag", "MetFrag", ("pipeline_metfrag_score.pkl", "pipeline_metfrag_score_diff.pkl")),
        # GLACIER features are passed to the ranker as raw values; any scaling must
        # be part of the trained ranking model.
        "glacier": Tool("glacier", "GLACIER"),
    },
}

TOOLSETS_FILE = paths.CONFIG_DIR / "toolsets.yaml"

DEFAULT_TOOLSETS = {
    FORMULA: {
        "default": "sirius",
        "combinations": {
            "sirius": {"tools": ["sirius", "msfinder", "msbuddy"], "model_dir": "."},
            "fiddle": {"tools": ["msfinder", "msbuddy", "fiddle"], "model_dir": "msfinder_msbuddy_fiddle"},
            "all": {"tools": ["sirius", "msfinder", "msbuddy", "fiddle"], "model_dir": "sirius_msfinder_msbuddy_fiddle"},
        },
    },
    STRUCTURE: {
        "default": "sirius",
        "combinations": {
            "sirius": {"tools": ["msfinder", "sirius", "metfrag"], "model_dir": "."},
            "glacier": {"tools": ["msfinder", "glacier", "metfrag"], "model_dir": "msfinder_glacier_metfrag"},
            "all": {"tools": ["msfinder", "sirius", "metfrag", "glacier"], "model_dir": "msfinder_sirius_metfrag_glacier"},
        },
    },
}


def normalize_tool_name(name: str) -> str:
    """"MS-FINDER", "ms_finder", "MSFinder" -> "msfinder"."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def base_model_dir(step: str) -> Path:
    # Looked up at call time so that tests can point paths at a temporary folder.
    return Path(paths.FORMULA_MODEL_DIR if step == FORMULA else paths.STRUCTURE_MODEL_DIR)


@dataclass(frozen=True)
class ToolSelection:
    step: str
    name: str
    tools: tuple[str, ...]            # registry order
    model_subdir: str = "."

    @property
    def model_dir(self) -> Path:
        sub = Path(self.model_subdir)
        return sub if sub.is_absolute() else base_model_dir(self.step) / sub

    @property
    def labels(self) -> list[str]:
        return [TOOLS[self.step][t].label for t in self.tools]

    def uses(self, tool: str) -> bool:
        return tool in self.tools

    def pipeline_dir(self, tool: str) -> str:
        """Folder holding the tool's normalization pipelines: model_dir, else the step's model folder."""
        pipelines = TOOLS[self.step][tool].pipelines
        if pipelines and all((self.model_dir / p).exists() for p in pipelines):
            return str(self.model_dir)
        return str(base_model_dir(self.step))

    def describe(self) -> str:
        return f"{self.name} ({', '.join(self.labels)})"


def load_toolsets(path: str | Path | None = None) -> dict:
    """Tool combinations from config/toolsets.yaml (or DEFAULT_TOOLSETS when it is missing)."""
    path = Path(path) if path else TOOLSETS_FILE
    if not path.is_file():
        return DEFAULT_TOOLSETS
    with open(path, encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}
    result = {}
    for step in STEPS:
        block = settings.get(step) or DEFAULT_TOOLSETS[step]
        combinations = block.get("combinations") or {}
        if not combinations:
            raise ValueError(f"{path}: {step}.combinations is empty")
        for name, entry in combinations.items():
            tools = [normalize_tool_name(t) for t in entry.get("tools") or []]
            unknown = [t for t in tools if t not in TOOLS[step]]
            if unknown or not tools:
                raise ValueError(f"{path}: {step}.{name}: unknown tools {unknown or '(none)'}; "
                                 f"known: {', '.join(TOOLS[step])}")
        default = block.get("default")
        if default not in combinations:
            raise ValueError(f"{path}: {step}.default must be one of {', '.join(combinations)}")
        result[step] = {"default": default, "combinations": combinations}
    return result


def _selection(step: str, name: str, entry: dict) -> ToolSelection:
    wanted = {normalize_tool_name(t) for t in entry["tools"]}
    tools = tuple(t for t in TOOLS[step] if t in wanted)
    return ToolSelection(step, name, tools, str(entry.get("model_dir") or "."))


def all_selections(step: str, toolsets: dict | None = None) -> list[ToolSelection]:
    toolsets = toolsets or load_toolsets()
    return [_selection(step, name, entry) for name, entry in toolsets[step]["combinations"].items()]


def resolve_selection(step: str, spec: str | None = None, toolsets: dict | None = None) -> ToolSelection:
    """Preset name or tool list (comma or space separated, any order) -> ToolSelection."""
    toolsets = toolsets or load_toolsets()
    combinations = toolsets[step]["combinations"]
    if spec is None or not str(spec).strip():
        name = toolsets[step]["default"]
        return _selection(step, name, combinations[name])
    text = str(spec).strip()
    if text.lower() in combinations:
        return _selection(step, text.lower(), combinations[text.lower()])
    requested = [normalize_tool_name(t) for t in text.replace(",", " ").split()]
    unknown = [t for t in requested if t not in TOOLS[step]]
    if unknown:
        raise ValueError(f"Unknown {step} tool(s): {', '.join(unknown)}. Known: {', '.join(TOOLS[step])}")
    for selection in all_selections(step, toolsets):
        if set(selection.tools) == set(requested):
            return selection
    choices = "; ".join(s.describe() for s in all_selections(step, toolsets))
    raise ValueError(f"No {step} model is registered for the tools {', '.join(sorted(set(requested)))}. "
                     f"Choose one of: {choices} (see config/toolsets.yaml).")


# --- runtime assets -------------------------------------------------------------

def _msfinder_executable_missing() -> list[str]:
    if any(Path(paths.MSFINDER_DIR).glob("MSFINDER*/MsfinderConsoleApp.exe")):
        return []
    return [str(Path(paths.MSFINDER_DIR) / "MSFINDER*/MsfinderConsoleApp.exe")]


def _fiddle_environment_missing() -> list[str]:
    """msfiddle and PyTorch in this environment (or the configured other one) and the FIDDLE weights.

    Settings > Set up FIDDLE (or python -m msemblator.fiddle_setup --run) installs what is missing.
    """
    from msemblator import fiddle_setup

    current = fiddle_setup.status(ask_conda=False)
    missing = []
    if not current.environment_ready:
        if current.where == fiddle_setup.OTHER_PYTHON:
            missing.append(f"{current.python} (formula_prediction.fiddle.python)")
        elif current.where == fiddle_setup.OTHER_CONDA:
            missing.append(f"conda environment '{current.env_name}' (formula_prediction.fiddle.conda_env)")
        else:
            missing.append("FIDDLE packages msfiddle and torch in this environment "
                           "(Set up FIDDLE, or pip install -r requirements-extras.txt)")
    missing += [f"{w} (FIDDLE weights: Set up FIDDLE, or python -m msfiddle download)"
                for w in current.weights_missing]
    return missing


def _glacier_missing() -> list[str]:
    missing = [f"Python package {name} (requirements-extras.txt)"
               for name in ("msentity", "msentity_similarity") if importlib.util.find_spec(name) is None]
    glacier_config = Path(paths.CONFIG_DIR) / "glacier.yaml"
    if not glacier_config.is_file():
        return missing + [str(glacier_config)]
    try:
        from msemblator.glacier.config import load_config
        config = load_config(glacier_config)
    except Exception as exc:  # invalid settings are reported, not raised
        return missing + [f"{glacier_config} ({exc})"]
    if not any(config.library_path(a) and config.library_path(a).is_file() for a in config.libraries):
        missing.append(f"GLACIER libraries in {config.library_dir} (none of {', '.join(config.libraries.values())})")
    return missing


def tool_assets_missing(selection: ToolSelection) -> list[str]:
    """Programs, libraries and settings the selected tools need (models are checked separately)."""
    required: list[Path] = []
    missing: list[str] = []
    if selection.step == FORMULA:
        if selection.uses("sirius"):
            required.append(Path(paths.SIRIUS_DIR) / "sirius.exe")
        if selection.uses("msfinder"):
            required.append(Path(paths.MSFINDER_CONFIG_DIR) / "MsfinderConsoleApp_Param_formula.txt")
            missing += _msfinder_executable_missing()
        if selection.uses("fiddle"):
            missing += _fiddle_environment_missing()
    else:
        if selection.uses("sirius"):
            required += [Path(paths.SIRIUS_DIR) / "sirius.exe", Path(paths.LIBRARY_DIR) / "sirius_structure_db.siriusdb"]
        if selection.uses("metfrag"):
            required += [Path(paths.METFRAG_DIR) / "MetFragCommandLine-2.5.0.jar",
                         Path(paths.LIBRARY_DIR) / "metfrag_StructureDB.txt",
                         Path(paths.METFRAG_CONFIG_DIR) / "example_paramater.txt"]
        if selection.uses("msfinder"):
            required += [Path(paths.LIBRARY_DIR) / "MsfinderStructureDB_all.txt",
                         Path(paths.MSFINDER_CONFIG_DIR) / "MsfinderConsoleApp_Param_formula.txt",
                         Path(paths.MSFINDER_CONFIG_DIR) / "MsfinderConsoleApp-Param2_structure.txt"]
            missing += _msfinder_executable_missing()
        if selection.uses("glacier"):
            missing += _glacier_missing()
    missing = [str(p) for p in required if not p.exists()] + missing
    return list(dict.fromkeys(missing))


def model_assets_missing(selection: ToolSelection) -> list[str]:
    """Ranking model of the combination and normalization pipelines of its tools."""
    missing = []
    for tool in selection.tools:
        folder = Path(selection.pipeline_dir(tool))
        missing += [str(folder / p) for p in TOOLS[selection.step][tool].pipelines if not (folder / p).exists()]
    if not any((selection.model_dir / name).exists() for name in paths.RANKER_MODEL_FILES):
        missing.append(f"{selection.model_dir / paths.RANKER_MODEL_FILES[0]} "
                       f"(or {', '.join(paths.RANKER_MODEL_FILES[1:])}): ranking model for "
                       f"{selection.step} tools {selection.describe()}")
    return missing


def missing_assets(selection: ToolSelection) -> list[str]:
    return tool_assets_missing(selection) + model_assets_missing(selection)


@dataclass
class SelectionStatus:
    selection: ToolSelection
    is_default: bool
    missing: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.missing


def available_selections(toolsets: dict | None = None) -> dict[str, list[SelectionStatus]]:
    """Every registered combination and the files it still needs (for --list_tools or a GUI)."""
    toolsets = toolsets or load_toolsets()
    return {step: [SelectionStatus(s, s.name == toolsets[step]["default"], missing_assets(s))
                   for s in all_selections(step, toolsets)]
            for step in STEPS}


def ranking_features(selection: ToolSelection) -> list[str]:
    """Feature columns this selection produces for the ranking model (in the model's naming)."""
    adducts = ["adduct_MplusHplus", "adduct_MplusNaplus", "adduct_MplusNH4plus",
               "adduct_MminusHminus", "adduct_MplusClminus", "adduct_MplusFAminusHminus"]
    scored = sorted(t for t in selection.tools if TOOLS[selection.step][t].pipelines)
    rank = "normalization_rank" if selection.step == FORMULA else "normalized_rank"
    columns = ([f"normalization_score_{t}" for t in scored]
               + [f"normalization_score_diff_{t}" for t in scored]
               + [f"{rank}_{t}" for t in scored])
    return columns + adducts


def check_model_features(model_features: list[str], selection: ToolSelection, model_name: str) -> None:
    """Refuse a model that needs features of tools that are not selected."""
    other_tools = [t for t in TOOLS[selection.step] if t not in selection.tools]
    foreign = [c for c in model_features
               if any(c.lower().endswith(f"_{t}") for t in other_tools)
               or (c.startswith("GLACIER_") and not selection.uses("glacier"))]
    if foreign:
        raise ValueError(
            f"{model_name} in {selection.model_dir} expects features of tools that are not selected "
            f"({', '.join(foreign[:6])}{', ...' if len(foreign) > 6 else ''}). "
            f"Selected {selection.step} tools: {selection.describe()}. Check model_dir in config/toolsets.yaml.")
