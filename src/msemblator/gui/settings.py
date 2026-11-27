"""Remembered GUI settings: last files, analysis choices, window size, recent MSP files.

Stored per user in %APPDATA%\\MS-Emblator\\gui_settings.json (MSEMBLATOR_GUI_SETTINGS
overrides the location). Passwords are never stored.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

RECENT_LIMIT = 8
DEFAULTS = {
    "input_path": "", "output_dir": "", "mode": 2, "formula_tools": "", "structure_tools": "",
    "sirius_user": "", "geometry": "", "recent_inputs": [],
}


def settings_path() -> Path:
    if os.environ.get("MSEMBLATOR_GUI_SETTINGS"):
        return Path(os.environ["MSEMBLATOR_GUI_SETTINGS"])
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / "MS-Emblator" / "gui_settings.json"


def load_settings() -> dict:
    settings = dict(DEFAULTS)
    try:
        stored = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings
    if isinstance(stored, dict):
        settings.update({k: v for k, v in stored.items() if k in DEFAULTS and k != "sirius_password"})
    if not isinstance(settings.get("recent_inputs"), list):
        settings["recent_inputs"] = []
    return settings


def save_settings(settings: dict) -> None:
    path = settings_path()
    data = {k: settings.get(k, DEFAULTS[k]) for k in DEFAULTS}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass   # remembering settings is a convenience; never fail because of it


def add_recent(recent: list, path: str) -> list:
    """Most recent first, without duplicates (case-insensitive on Windows)."""
    key = os.path.normcase(os.path.abspath(path))
    rest = [p for p in recent if os.path.normcase(os.path.abspath(p)) != key]
    return [path] + rest[:RECENT_LIMIT - 1]
