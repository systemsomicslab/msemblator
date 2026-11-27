"""GLACIER structure annotation: candidate extraction, scoring and features.

    result = run_glacier("queries.msp", load_config())
    result.features   # one row per (query, Short InChIKey), raw GLACIER features
    result.raw        # one row per (query, predicted spectrum), similarities per CE
    result.queries    # one row per input spectrum, with its status

The same call is used for training data and for inference; answers are added
afterwards with labels.attach_labels.

Processing order: queries are grouped by adduct, the adduct's library is loaded,
queries are grouped by molecular formula, every library spectrum with that
formula is scored against the group's queries, and the library is released
before the next adduct. A query is never compared with another adduct's library.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from msemblator.glacier import queries as q
from msemblator.glacier.adducts import AdductResolver
from msemblator.glacier.config import GlacierConfig
from msemblator.glacier.features import build_features
from msemblator.glacier.library import (LIBRARY_AVAILABLE, LibraryError, PredictedLibrary, library_status)
from msemblator.glacier.similarity import COVERAGE_COLUMNS, SimilarityCalculator

logger = logging.getLogger(__name__)

# Status of one (query, predicted spectrum) row of the raw output.
RAW_OK = "ok"
RAW_MISSING_CE = "missing_collision_energy"
RAW_FAILED = "similarity_failed"


def raw_columns(config: GlacierConfig) -> list[str]:
    return (["query_index", "filename", "adduct", "formula", "formula_source", "library_formula",
             "candidate_spec_id", "SMILES", "InChIKey", "Short_InChIKey", "collision_energy",
             "n_peaks_measured", "n_peaks_predicted"]
            + list(config.similarities) + list(COVERAGE_COLUMNS) + ["status", "message"])


@dataclass
class GlacierResult:
    queries: pd.DataFrame
    raw: pd.DataFrame
    features: pd.DataFrame
    libraries: list[dict]
    config: GlacierConfig
    timings: dict = field(default_factory=dict)
    # Predicted peaks of every scored spectrum (only with run_glacier(collect_spectra=True)):
    # adduct, candidate_spec_id, mz, intensity (";"-separated, as used for scoring).
    spectra: pd.DataFrame | None = None


def _peak_rows(library, candidates, dataset, seen):
    """Peaks of the candidate spectra as compact text, once per library spectrum."""
    offsets, all_mz, all_intensity = dataset.offsets, dataset.mz, dataset.intensity
    rows = []
    for k, position in enumerate(candidates):
        spec_id = library.spec_id[position]
        if (library.adduct, spec_id) in seen:
            continue
        seen.add((library.adduct, spec_id))
        mz = all_mz[offsets[k]:offsets[k + 1]]
        intensity = all_intensity[offsets[k]:offsets[k + 1]]
        rows.append((library.adduct, spec_id, ";".join(f"{v:.5f}" for v in mz),
                     ";".join(f"{v:.4g}" for v in intensity)))
    return rows


def _score_group(calculator, query_set, query_rows, query_dataset, query_positions, library, formula,
                 spectra_rows=None, seen=None):
    """Raw rows for one formula group (all of its queries x all candidate spectra)."""
    candidates = library.candidates(formula)
    if not len(candidates):
        return None, query_rows
    subset = query_dataset.select(query_positions)
    peaks = [query_set.peaks[r] for r in query_rows]
    candidate_spectra, n_peaks_predicted = library.spectra(candidates)
    if spectra_rows is not None:
        spectra_rows.extend(_peak_rows(library, candidates, candidate_spectra, seen))
    scores = calculator.score(subset, peaks, candidate_spectra)
    n_q, n_c = len(query_rows), len(candidates)
    qi = np.repeat(np.arange(n_q), n_c)
    ci = np.tile(candidates, n_q)
    table = query_set.table
    rows = np.asarray(query_rows)[qi]
    block = {
        "query_index": table["query_index"].to_numpy()[rows],
        "filename": table["filename"].to_numpy()[rows],
        "adduct": library.adduct,
        "formula": formula,
        "formula_source": table["formula_source"].to_numpy()[rows],
        "library_formula": library.lookup_key(formula),
        "candidate_spec_id": library.spec_id[ci],
        "SMILES": library.smiles[ci],
        "InChIKey": library.inchikey[ci],
        "Short_InChIKey": library.short_key[ci],
        "collision_energy": library.collision_energy[ci],
        "n_peaks_measured": table["n_peaks"].to_numpy()[rows],
        "n_peaks_predicted": np.tile(n_peaks_predicted, n_q),
    }
    methods = calculator.methods
    for name in list(methods) + list(COVERAGE_COLUMNS):
        block[name] = scores[name]
    failed = np.zeros(n_q * n_c, dtype=bool)
    for method in methods:
        failed |= np.isnan(scores[method])
    status = np.where(np.isnan(block["collision_energy"]), RAW_MISSING_CE, RAW_OK).astype(object)
    status[failed] = RAW_FAILED
    block["status"] = status
    block["message"] = np.where(failed, scores["error"] or "score missing", "")
    return pd.DataFrame(block), []


def run_glacier(input_msp: str | Path, config: GlacierConfig, *, formula_table: str | Path | None = None,
                adducts: list[str] | None = None, limit: int | None = None,
                collect_spectra: bool = False) -> GlacierResult:
    """Score the measured spectra of an MSP against the GLACIER libraries of their adducts.

    collect_spectra keeps the predicted peaks of every scored spectrum in
    GlacierResult.spectra (used for mirror plots).
    """
    started = time.time()
    spectra_rows, seen = ([], set()) if collect_spectra else (None, None)
    resolver = AdductResolver(config.libraries, config.adduct_aliases)
    query_set = q.read_queries(input_msp, config, resolver, formula_table)
    table = query_set.table
    calculator = SimilarityCalculator(config)
    if adducts:
        wanted = {resolver.resolve(a) or a for a in adducts}
        skipped = table["status"].eq("") & ~table["adduct"].isin(wanted)
        table.loc[skipped, ["status", "message"]] = ["not_selected", "adduct not selected for this run"]
    if limit:
        pending = table.index[table["status"].eq("")]
        table.loc[pending[limit:], ["status", "message"]] = ["not_selected", "beyond --limit"]

    blocks, libraries, timings = [], [], {"read_queries": round(time.time() - started, 1)}
    for adduct, group in table[table["status"].eq("")].groupby("adduct", sort=False):
        rows = group.index.to_numpy()
        status, path = library_status(config, adduct)
        info = {"adduct": adduct, "path": str(path) if path else "", "status": status, "n_queries": len(rows)}
        libraries.append(info)
        if status != LIBRARY_AVAILABLE:
            table.loc[rows, ["status", "message"]] = [q.STATUS_LIBRARY_NOT_AVAILABLE, f"{status}: {path or adduct}"]
            logger.warning("%s: library %s (%d queries skipped)", adduct, status, len(rows))
            continue
        adduct_started = time.time()
        try:
            library = PredictedLibrary(path, adduct, config, resolver)
        except LibraryError as exc:
            info["status"], info["message"] = "unusable", str(exc)
            table.loc[rows, ["status", "message"]] = [q.STATUS_LIBRARY_NOT_AVAILABLE, str(exc)]
            logger.error("%s", exc)
            continue
        try:
            info.update(library.stats.__dict__)
            info["status"] = status
            query_dataset = query_set.dataset(rows)
            position = {row: i for i, row in enumerate(rows)}
            formula_groups = [(formula, list(g.index)) for formula, g in group.groupby("formula", sort=False)]

            # Formula groups are scored one after another: msentity datasets must not be
            # used from several Python threads at once (concurrent use crashed in tests).
            # msentity-similarity itself uses several threads for large groups (glacier.threads).
            from tqdm import tqdm
            for formula, query_rows in tqdm(formula_groups, desc=f"GLACIER {adduct}", unit="formula",
                                            disable=None):
                block, no_candidate_rows = _score_group(
                    calculator, query_set, query_rows, query_dataset, [position[r] for r in query_rows],
                    library, formula, spectra_rows, seen)
                if block is not None:
                    blocks.append(block)
                if no_candidate_rows:
                    table.loc[no_candidate_rows, ["status", "message"]] = [q.STATUS_NO_CANDIDATE, ""]
            info.update(library.stats.__dict__)  # counts collected while scoring
            info["status"] = status
        finally:
            library.close()
        timings[f"adduct {adduct}"] = round(time.time() - adduct_started, 1)

    raw = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=raw_columns(config))
    raw = raw[raw_columns(config)]
    feature_started = time.time()
    features = build_features(raw, config)
    timings["features"] = round(time.time() - feature_started, 1)

    _finish_query_status(table, raw, features)
    timings["total"] = round(time.time() - started, 1)
    spectra = (pd.DataFrame(spectra_rows, columns=["adduct", "candidate_spec_id", "mz", "intensity"])
               if collect_spectra else None)
    return GlacierResult(table, raw, features, libraries, config, timings, spectra)


def _finish_query_status(table, raw, features):
    pending = table["status"].eq("")
    if raw.empty:
        table.loc[pending, "status"] = q.STATUS_NO_CANDIDATE
    else:
        per_query = raw.groupby("query_index").agg(n_spectra=("status", "size"),
                                                   n_failed=("status", lambda s: (s == RAW_FAILED).sum()))
        failed = per_query.index[per_query["n_failed"] == per_query["n_spectra"]]
        table.loc[pending & table["query_index"].isin(failed), ["status", "message"]] = \
            [q.STATUS_SIMILARITY_FAILED, "every candidate spectrum failed"]
        table.loc[table["status"].eq(""), "status"] = q.STATUS_OK
        table["n_candidate_spectra"] = table["query_index"].map(per_query["n_spectra"]).fillna(0).astype(int)
    if "n_candidate_spectra" not in table:
        table["n_candidate_spectra"] = 0
    n_structures = features.groupby("query_index").size() if not features.empty else pd.Series(dtype=int)
    table["n_candidates"] = table["query_index"].map(n_structures).fillna(0).astype(int)
