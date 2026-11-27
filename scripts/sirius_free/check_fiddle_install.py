#!/usr/bin/env python3
"""Check that a FIDDLE environment reproduces the reference output.

Run inside the FIDDLE environment after `msfiddle download`:
    python scripts/sirius_free/check_fiddle_install.py      (MS-Emblator environment)

Checks:
  1. msfiddle version and SHA-256 of the downloaded checkpoints;
  2. runs the FIDDLE runner on tests/fixtures/sirius_free_demo.msp;
  3. compares statuses, candidate formulas/order (exact) and scores (tolerance)
     with envs/fiddle/reference/.
Exit code 0 = identical to the reference.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from msemblator.sirius_free import msp_io  # noqa: E402
from msemblator.sirius_free.annotators import common, fiddle_runner  # noqa: E402

REFERENCE = ROOT / "envs" / "fiddle" / "reference"
DEMO_MSP = ROOT / "tests" / "fixtures" / "sirius_free_demo.msp"


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    expected = json.loads((REFERENCE / "expected.json").read_text(encoding="utf-8"))
    problems = []

    version = common.package_version("msfiddle")
    print(f"msfiddle {version}, torch {common.package_version('torch')}")
    if version != "2.3.0":
        problems.append(f"msfiddle version {version} != 2.3.0")
    from msfiddle.download import get_checkpoint_dir
    checkpoint_dir = Path(get_checkpoint_dir())
    for name, digest in expected["checkpoints_sha256"].items():
        path = checkpoint_dir / name
        actual = msp_io.file_sha256(path) if path.exists() else "missing (run: msfiddle download)"
        if actual != digest:
            problems.append(f"checkpoint {name}: {actual}")
    if msp_io.file_sha256(DEMO_MSP) != expected["input_sha256"]:
        problems.append("demo MSP differs from the one used for the reference")

    with tempfile.TemporaryDirectory() as tmp:
        manifest = fiddle_runner.run(DEMO_MSP, tmp)
        status = read_csv(Path(tmp) / "fiddle" / "fiddle_status.csv")
        candidates = read_csv(Path(tmp) / "fiddle" / "fiddle_candidates.csv")
    if manifest["status_counts"] != expected["status_counts"]:
        problems.append(f"status counts {manifest['status_counts']} != {expected['status_counts']}")
    ref_status = read_csv(REFERENCE / "fiddle_status.csv")
    if [(r["query_id"], r["status"]) for r in status] != [(r["query_id"], r["status"]) for r in ref_status]:
        problems.append("per-spectrum statuses differ")
    ref_candidates = read_csv(REFERENCE / "fiddle_candidates.csv")
    key = lambda rows: [(r["query_id"], r["rank"], r["formula"]) for r in rows]  # noqa: E731
    if key(candidates) != key(ref_candidates):
        problems.append("candidate formulas or their order differ")
    else:
        tol = float(expected["score_tolerance"])
        worst = max((abs(float(a["score"]) - float(b["score"])) for a, b in zip(candidates, ref_candidates)),
                    default=0.0)
        print(f"{len(candidates)} candidates, max score difference {worst:.2e}")
        if worst > tol:
            problems.append(f"score difference {worst} > {tol}")

    if problems:
        print("NOT REPRODUCED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("OK: FIDDLE output is identical to the reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
