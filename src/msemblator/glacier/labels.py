"""Training labels for the GLACIER features (kept apart from the core modules).

TF = 1 when the candidate's Short InChIKey (first 14 characters) equals the
answer's, otherwise 0; TF is the label column of the existing training tables
(analysis/make_training_data.py). Answers are attached after the features have
been built; they are never used for candidate extraction or scoring, and
inference does not need them.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from msemblator.glacier.formula import normalize_formula
from msemblator.glacier.library import short_inchikey

REFERENCE_COLUMNS = ["query_index", "filename", "reference_formula", "reference_Short_InChIKey"]


def read_reference_msp(path: str | Path) -> pd.DataFrame:
    """Answers stored in the query MSP itself (Formula, InChIKey or SMILES)."""
    import msentity as ms

    dataset = ms.load(path, canonicalize_column=True)

    def column(*names):
        for name in names:
            if name in dataset:
                return ["" if v is None else str(v).strip() for v in dataset[name]]
        return [""] * len(dataset)

    names, formulas = column("Name"), column("Formula")
    inchikeys, smiles = column("InChIKey"), column("SMILES")
    cache: dict = {}
    return pd.DataFrame({
        "query_index": range(1, len(dataset) + 1),
        "filename": names,
        "reference_formula": [normalize_formula(f) or "" for f in formulas],
        "reference_Short_InChIKey": [short_inchikey(k, s, cache) for k, s in zip(inchikeys, smiles)],
    }, columns=REFERENCE_COLUMNS)


def read_reference_csv(path: str | Path) -> pd.DataFrame:
    """Answer table, e.g. nist26_answer_id.csv (answer_name, answer_formula, answer_inchikey, ...)."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    lower = {c.lower(): c for c in table.columns}

    def pick(*names):
        return next((lower[n] for n in names if n in lower), None)

    name = pick("filename", "name", "answer_name")
    if name is None:
        raise KeyError(f"{path}: needs a filename / name / answer_name column")
    key = pick("shortinchikey", "short_inchikey", "answer_shortinchikey", "inchikey", "answer_inchikey")
    smiles = pick("smiles", "answer_smiles")
    formula = pick("formula", "answer_formula")
    cache: dict = {}
    keys = table[key] if key else [""] * len(table)
    smiles_values = table[smiles] if smiles else [""] * len(table)
    return pd.DataFrame({
        "query_index": pd.NA,
        "filename": table[name].str.strip(),
        "reference_formula": [normalize_formula(f) or "" for f in (table[formula] if formula else [""] * len(table))],
        "reference_Short_InChIKey": [short_inchikey(k, s, cache) for k, s in zip(keys, smiles_values)],
    }, columns=REFERENCE_COLUMNS)


def read_reference(path: str | Path) -> pd.DataFrame:
    return read_reference_csv(path) if Path(path).suffix.lower() in (".csv", ".tsv") else read_reference_msp(path)


def _join_key(reference: pd.DataFrame) -> str:
    return "query_index" if reference["query_index"].notna().all() else "filename"


def attach_labels(features: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Add reference_Short_InChIKey and TF (Int8) to the feature table."""
    key = _join_key(reference)
    answers = reference[[key, "reference_Short_InChIKey"]].drop_duplicates(key)
    labelled = features.drop(columns=["reference_Short_InChIKey", "TF"], errors="ignore").merge(
        answers, on=key, how="left")
    answer = labelled["reference_Short_InChIKey"].fillna("")
    labelled["reference_Short_InChIKey"] = answer
    labelled["TF"] = (answer.ne("") & labelled["Short_InChIKey"].eq(answer)).astype("int8")
    return labelled


def attach_reference_to_queries(queries: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Per query: the answer and whether the formula given to GLACIER is the reference formula."""
    key = _join_key(reference)
    merged = queries.merge(reference[[key, "reference_formula", "reference_Short_InChIKey"]]
                           .drop_duplicates(key), on=key, how="left")
    merged[["reference_formula", "reference_Short_InChIKey"]] = \
        merged[["reference_formula", "reference_Short_InChIKey"]].fillna("")
    merged["formula_is_reference"] = merged["formula"].ne("") & merged["formula"].eq(merged["reference_formula"])
    return merged
