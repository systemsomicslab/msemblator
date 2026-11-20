#!/usr/bin/env python3
"""Write FIDDLE features as machine-learning CSVs in the NIST26 table format.

Input: fiddle_features_top100.pkl from fit_fiddle_normalization.py.

Outputs (in --out-dir):
  fiddle_{train,test,all}_formula_score.csv
      long format, one row per spectrum x FIDDLE candidate; same columns as
      tools_result/formula/nist_*_formula_score.csv
  nist_fiddle_{train,test}_formula.csv      (only with --existing-dir)
      wide model input: the existing long tables (msbuddy, msfinder, sirius) plus
      FIDDLE, pivoted with formula_machine_input(); one row per spectrum x formula,
      features of a tool that did not propose the formula are 0.
      With --tools msbuddy msfinder fiddle the file is
      nist_msbuddy_msfinder_fiddle_{train,test}_formula.csv (SIRIUS-free).

Notes:
- TF = 1 when the candidate formula equals the reference formula (exact string;
  FIDDLE formulas are already in Hill order).
- The existing tables call MSBuddy "buddy"; it is renamed to "msbuddy" so the wide
  columns match nist_*_formula.csv.
- "[M+CHO2]-" is written as "[M+FA-H]-" before pivoting. formula_machine_input()
  only maps "[M+CO2]-", so formate spectra would otherwise get no adduct column.

Usage (msemblator environment):
    python scripts/sirius_free/build_fiddle_ml_csv.py \
        --features E:/MS-Emblator/data/nist26/tools_result/formula/fiddle/fiddle_features_top100.pkl \
        --answer E:/MS-Emblator/data/nist26/nist26_answer.csv \
        --out-dir E:/MS-Emblator/data/nist26/tools_result/formula/fiddle/ml_input \
        --existing-dir E:/MS-Emblator/data/nist26/tools_result/formula
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from msemblator.scoring.calculating_score import formula_machine_input  # noqa: E402

logger = logging.getLogger("fiddle_ml_csv")

LONG_COLUMNS = ["filename", "tool_name", "adduct", "rank", "formula", "Score_NZ", "Score_NZ_diff",
                "normalized_rank", "Used_tools", "name", "answer_formula", "inchikey", "TF"]
ALL_TOOLS = ["msbuddy", "msfinder", "sirius", "fiddle"]
ADDUCT_COLUMNS = ["adduct_MplusHplus", "adduct_MplusNaplus", "adduct_MplusNH4plus",
                  "adduct_MminusHminus", "adduct_MplusClminus", "adduct_MplusFAminusHminus"]


def feature_columns(tools: list[str]) -> list[str]:
    return ([f"normalization_score_{t}" for t in tools]
            + [f"normalization_score_diff_{t}" for t in tools]
            + [f"normalization_rank_{t}" for t in tools]
            + ADDUCT_COLUMNS)
ADDUCT_ALIASES = {"[M+CHO2]-": "[M+FA-H]-", "[M+HCOO]-": "[M+FA-H]-", "[M+HCO2]-": "[M+FA-H]-"}


def add_answers(frame: pd.DataFrame, answer: pd.DataFrame) -> pd.DataFrame:
    frame = frame.merge(answer, left_on="filename", right_on="name", how="left")
    frame["TF"] = (frame["formula"] == frame["answer_formula"]).astype("int8")
    return frame


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)
    logger.info("wrote %s: %d rows, %d spectra, TF=1 rows %d, spectra with TF=1 %d",
                path.name, len(frame), frame["filename"].nunique(), int(frame["TF"].sum()),
                frame.loc[frame["TF"] == 1, "filename"].nunique())


def build_wide(long_frame: pd.DataFrame, answer: pd.DataFrame, tools: list[str]) -> pd.DataFrame:
    columns = feature_columns(tools)
    long_frame = long_frame[long_frame["tool_name"].isin(tools)].copy()
    long_frame["adduct"] = long_frame["adduct"].replace(ADDUCT_ALIASES)
    t0 = time.time()
    wide = formula_machine_input(long_frame)
    logger.info("formula_machine_input: %d long rows -> %d wide rows (%.0f s)",
                len(long_frame), len(wide), time.time() - t0)
    for column in columns:
        if column not in wide.columns:
            wide[column] = 0.0
    wide = add_answers(wide, answer)
    return wide[["filename", "adduct", "formula"] + columns
                + ["Used_tools", "name", "answer_formula", "inchikey", "TF"]]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True, type=Path, help="fiddle_features_top100.pkl")
    parser.add_argument("--answer", required=True, type=Path, help="CSV with name, formula, inchikey")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--existing-dir", type=Path,
                        help="Folder with nist_{train,test}_formula_score.csv; enables the combined wide tables")
    parser.add_argument("--tools", nargs="+", choices=ALL_TOOLS, default=ALL_TOOLS,
                        help="Tools in the wide tables, e.g. --tools msbuddy msfinder fiddle (SIRIUS-free). "
                             "Formulas proposed only by excluded tools are dropped.")
    args = parser.parse_args(argv)
    tools = [t for t in ALL_TOOLS if t in args.tools]
    wide_name = "nist_fiddle" if tools == ALL_TOOLS else "nist_" + "_".join(tools)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.out_dir / "build_ml_csv.log", encoding="utf-8")])

    answer = pd.read_csv(args.answer, dtype=str)[["name", "formula", "inchikey"]]
    answer = answer.rename(columns={"formula": "answer_formula"})

    features = pd.read_pickle(args.features)
    features["filename"] = features["filename"].astype(str)
    fiddle = add_answers(features, answer)[LONG_COLUMNS + ["split"]]
    fiddle["TF"] = fiddle["TF"].astype(int)
    for split in ("train", "test"):
        write_csv(fiddle.loc[fiddle["split"] == split, LONG_COLUMNS], args.out_dir / f"fiddle_{split}_formula_score.csv")
    write_csv(fiddle[LONG_COLUMNS], args.out_dir / "fiddle_all_formula_score.csv")

    if args.existing_dir is None:
        return 0
    for split in ("train", "test"):
        existing = pd.read_csv(args.existing_dir / f"nist_{split}_formula_score.csv", dtype={"filename": str})
        existing["tool_name"] = existing["tool_name"].replace({"buddy": "msbuddy"})
        split_ids = set(existing["filename"])
        fiddle_split = fiddle[fiddle["split"] == split]
        unknown = set(fiddle_split["filename"]) - split_ids
        if unknown:
            logger.warning("%s: %d FIDDLE spectra are not in the existing %s table (kept)", split, len(unknown), split)
        long_frame = pd.concat([existing[LONG_COLUMNS], fiddle_split[LONG_COLUMNS]], ignore_index=True)
        logger.info("%s: tools %s, spectra per tool %s", split, tools,
                    long_frame[long_frame["tool_name"].isin(tools)]
                    .groupby("tool_name")["filename"].nunique().to_dict())
        write_csv(build_wide(long_frame, answer, tools), args.out_dir / f"{wide_name}_{split}_formula.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
