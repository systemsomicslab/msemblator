"""Output helpers shared by the annotation runners (standard library only)."""
from __future__ import annotations

import csv
import datetime as _dt
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

from msemblator.sirius_free.msp_io import STATUS_COLUMNS, file_sha256

CANDIDATE_BASE_COLUMNS = ["query_id", "tool", "rank", "formula", "score"]


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def write_candidates(rows: list[dict], path: Path, extra_columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = CANDIDATE_BASE_COLUMNS + extra_columns
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_status(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def status_row(query_id, tool, status, n_candidates=0, message=""):
    return {
        "query_id": query_id,
        "tool": tool,
        "status": status,
        "n_candidates": n_candidates,
        "message": str(message).replace("\n", " ")[:500],
    }


def write_manifest(path: Path, tool: str, input_msp: Path, parameters: dict,
                   versions: dict, status_rows: list[dict], started: _dt.datetime) -> dict:
    counts: dict[str, int] = {}
    for row in status_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    manifest = {
        "tool": tool,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "input_msp": str(input_msp),
        "input_sha256": file_sha256(input_msp),
        "n_spectra": len(status_rows),
        "status_counts": counts,
        "parameters": parameters,
        "versions": {"python": sys.version.split()[0], "platform": platform.platform(), **versions},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
