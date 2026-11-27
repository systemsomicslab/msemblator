"""GLACIER structure annotation: matching measured MS/MS spectra against GLACIER
predicted spectra and building ranker features (planned replacement of SIRIUS
in structure annotation: MetFrag + GLACIER + MS-FINDER).

    config.py      settings (config/glacier.yaml): libraries per adduct, CE, tolerance, similarities
    formula.py     formula parsing, Hill-order keys, masses
    adducts.py     adduct normalization and parsing, ion m/z, neutral part of an adduct
    queries.py     measured spectra from the MSP (no answer fields)
    preprocessing.py  precursor peak removal (on by default for measured spectra)
    library.py    predicted libraries: file lookup, loading, checks, formula index
    similarity.py  cosine, reverse cosine, spectral entropy (msentity-similarity)
    features.py    CE and Short InChIKey aggregation, Max / Rank / ScoreDiff features
    pipeline.py    run_glacier(): the whole flow, shared by training and inference
    labels.py      answers and the TF label (training only)
    evaluation.py  run summary and top-k accuracy
    integration.py join with the MetFrag / MS-FINDER candidate tables

See README_glacier.md for usage.

The default MS-Emblator workflow does not use this package yet.
"""
from msemblator.glacier.config import GlacierConfig, load_config
from msemblator.glacier.features import feature_columns

__all__ = ["GlacierConfig", "load_config", "feature_columns", "run_glacier"]


def run_glacier(*args, **kwargs):
    from msemblator.glacier.pipeline import run_glacier as _run
    return _run(*args, **kwargs)
