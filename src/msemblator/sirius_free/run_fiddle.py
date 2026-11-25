"""Run FIDDLE on an MSP file; started by msemblator.runners.fiddle_cmd with the FIDDLE environment's Python.

    python -m msemblator.sirius_free.run_fiddle --input data.msp --output out_dir --params '{"top_k": 100}'

Writes <out_dir>/fiddle/{fiddle_candidates.csv, fiddle_status.csv, manifest.json}.
Only the standard library and msfiddle are needed, not the rest of MS-Emblator.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from msemblator.sirius_free.annotators import fiddle_runner


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input MSP file")
    parser.add_argument("--output", required=True, help="Output folder; results go to <output>/fiddle")
    parser.add_argument("--params", default="{}", help="FIDDLE parameters as a JSON object")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    manifest = fiddle_runner.run(args.input, args.output, json.loads(args.params))
    logging.info("FIDDLE finished: %s", manifest["status_counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
