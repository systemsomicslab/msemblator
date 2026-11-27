#!/usr/bin/env python3
"""Build GLACIER similarity features for the measured spectra of an MSP file.

Usage (msemblator environment, from the repository root):
    python scripts/glacier/run_glacier_features.py --input data.msp --output out_dir
    python scripts/glacier/run_glacier_features.py --input data.msp --output out_dir --reference data.msp

Each spectrum is compared with the GLACIER predicted spectra of every library
compound that has the spectrum's molecular formula (the MSP formula field, or
--formula-table) and adduct. Libraries, collision energies, tolerance and
similarities come from config/glacier.yaml (see --config).

--reference adds the training label TF (Short InChIKey match) from an MSP with
answer fields or an answer CSV; answers are used for nothing else.

Outputs (out_dir; existing files are not overwritten without --overwrite):
    glacier_similarity_raw.csv        one row per (query, predicted spectrum): similarity per CE
    glacier_features.csv              one row per (query, Short InChIKey): GLACIER features
    glacier_query_status.csv          one row per input spectrum: status, candidate counts
    glacier_accuracy.csv              top-1/5/10 accuracy per similarity (with --reference)
    glacier_spectra_count_check.csv   Max scores by number of predicted spectra per candidate
    glacier_summary.json              counts, timings, settings and library versions
    glacier.log
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from msemblator.glacier import evaluation, labels  # noqa: E402
from msemblator.glacier.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from msemblator.glacier.pipeline import run_glacier  # noqa: E402
from msemblator.glacier.similarity import library_versions  # noqa: E402

OUTPUT_FILES = {
    "raw": "glacier_similarity_raw.csv",
    "features": "glacier_features.csv",
    "queries": "glacier_query_status.csv",
    "accuracy": "glacier_accuracy.csv",
    "spectra_count": "glacier_spectra_count_check.csv",
    "summary": "glacier_summary.json",
    "log": "glacier.log",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="MSP with the measured spectra")
    parser.add_argument("--output", required=True, help="output folder")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML settings (default: %(default)s)")
    parser.add_argument("--reference", help="answers for the TF label: MSP with InChIKey/SMILES or answer CSV")
    parser.add_argument("--formula-table", help="CSV (filename or query_index, formula) used instead of the MSP formula")
    parser.add_argument("--adducts", nargs="+", help="only these adducts (default: all)")
    parser.add_argument("--limit", type=int, help="score only the first N eligible spectra (quick tests)")
    parser.add_argument("--split", action="append", default=[], metavar="NAME=MSP",
                        help="add a split column: spectra whose NAME is in MSP get NAME (repeatable)")
    parser.add_argument("--ms2-tolerance", type=float, help="overrides glacier.ms2_tolerance")
    parser.add_argument("--ms2-tolerance-unit", choices=["ppm", "Da", "da"], help="overrides glacier.ms2_tolerance_unit")
    parser.add_argument("--similarities", nargs="+", help="overrides glacier.similarities")
    parser.add_argument("--remove-precursor", choices=["off", "measured", "both"],
                        help="overrides glacier.precursor_peak_removal (enabled / spectra)")
    parser.add_argument("--precursor-tolerance", type=float,
                        help="overrides glacier.precursor_peak_removal.tolerance")
    parser.add_argument("--precursor-tolerance-unit", choices=["ppm", "Da", "da"],
                        help="overrides glacier.precursor_peak_removal.tolerance_unit")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    return parser.parse_args(argv)


def split_names(specs) -> dict[str, str]:
    import msentity as ms

    names = {}
    for spec in specs:
        split, sep, path = spec.partition("=")
        if not sep:
            raise ValueError(f"--split expects NAME=MSP, got {spec!r}")
        dataset = ms.load(path, canonicalize_column=True)
        for name in dataset["Name"]:
            names.setdefault(str(name).strip(), split)
    return names


def write_csv(frame: pd.DataFrame, path: Path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {key: output / name for key, name in OUTPUT_FILES.items()}
    inputs = {Path(p).resolve() for p in (args.input, args.reference, args.formula_table) if p}
    if any(p.resolve() in inputs for p in paths.values()):
        raise SystemExit("An output file would replace an input file; choose another --output.")
    existing = [p.name for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"Output files already exist in {output}: {existing}. Use --overwrite to replace them.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(paths["log"], "w", encoding="utf-8")],
                        force=True)
    overrides = {"ms2_tolerance": args.ms2_tolerance, "ms2_tolerance_unit": args.ms2_tolerance_unit,
                 "similarities": args.similarities}
    config = load_config(args.config, overrides)
    removal = dict(config.precursor_peak_removal)
    if args.remove_precursor:
        removal["enabled"] = args.remove_precursor != "off"
        if removal["enabled"]:
            removal["spectra"] = args.remove_precursor
    if args.precursor_tolerance is not None:
        removal["tolerance"] = args.precursor_tolerance
    if args.precursor_tolerance_unit:
        removal["tolerance_unit"] = args.precursor_tolerance_unit
    config.precursor_peak_removal = removal
    config.validate()
    logging.info("Settings: %s", json.dumps(config.to_dict(), ensure_ascii=False))

    started = time.time()
    result = run_glacier(args.input, config, formula_table=args.formula_table,
                         adducts=args.adducts, limit=args.limit)
    queries, features = result.queries, result.features

    accuracy = None
    if args.reference:
        reference = labels.read_reference(args.reference)
        features = labels.attach_labels(features, reference)
        queries = labels.attach_reference_to_queries(queries, reference)
        available = [lib["adduct"] for lib in result.libraries if lib.get("status") == "available"]
        accuracy = evaluation.accuracy_table(features, queries, config, available)
    if args.split:
        names = split_names(args.split)
        for frame in (features, queries):
            frame["split"] = frame["filename"].map(names).fillna("")

    write_csv(result.raw, paths["raw"])
    write_csv(features, paths["features"])
    write_csv(queries, paths["queries"])
    write_csv(evaluation.spectra_count_check(features, config), paths["spectra_count"])
    if accuracy is not None:
        write_csv(accuracy, paths["accuracy"])

    summary = evaluation.summarize(result, queries, features)
    summary["timings_seconds"]["total_with_outputs"] = round(time.time() - started, 1)
    summary.update({
        "input_msp": str(Path(args.input).resolve()),
        "reference": str(Path(args.reference).resolve()) if args.reference else None,
        "formula_table": str(Path(args.formula_table).resolve()) if args.formula_table else None,
        "settings": config.to_dict(),
        "library_versions": library_versions(config),
        "libraries": result.libraries,
        "command": " ".join(sys.argv),
    })
    if accuracy is not None:
        summary["accuracy"] = accuracy.to_dict(orient="records")
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    logging.info("Queries: %s", summary["query_status"])
    if accuracy is not None:
        shown = accuracy[["adduct", "similarity", "n_queries", "n_queries_with_answer",
                          "top1_acc_all", "top5_acc_all", "top10_acc_all", "top1_acc_with_answer"]]
        logging.info("Accuracy:\n%s", shown.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    logging.info("Written to %s (%.1f s)", output, time.time() - started)


if __name__ == "__main__":
    main()
