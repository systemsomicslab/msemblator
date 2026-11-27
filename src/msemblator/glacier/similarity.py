"""Similarity between measured spectra and GLACIER predicted spectra.

Every score is calculated by an existing library; nothing is reimplemented here.

msentity-similarity (cosine, reverse_cosine, modified_cosine, bonanza)
    Peaks pair when |mz1 - mz2| / ((mz1 + mz2) / 2) <= tolerance (ppm) or
    |mz1 - mz2| <= tolerance (Da); each peak pairs at most once, closest pairs first;
    intensities <= 0 never pair. ``library_search(query, reference)`` is called
    with the MEASURED spectrum as query (first) and the PREDICTED spectrum as
    reference (second):
        cosine         = sum(a*r over pairs) / (||a|| * ||r||)
        reverse_cosine = sum(a*r over pairs) / (||a matched|| * ||r||)
    so measured peaks that GLACIER did not predict do not lower reverse_cosine,
    while predicted peaks without a measured partner still do.
    peak_intensity_coverage_measured / _predicted = sum(x^2 over matched peaks) /
    sum(x^2 over all peaks) of each side: 1 - coverage is the unmatched share.

ms_entropy (entropy, optional)
    Weighted spectral entropy similarity (Li et al., 2021) with MSEntropy's own
    spectrum cleaning (glacier.entropy). It is symmetric.

Scores are in [0, 1]; a pair without matched peaks scores 0. A calculation that
fails leaves NaN, so it is never mistaken for a real score of 0.
"""
from __future__ import annotations

import logging

import numpy as np

from msemblator.glacier.config import GlacierConfig

logger = logging.getLogger(__name__)

MSSIM_METHODS = ("cosine", "reverse_cosine", "modified_cosine", "bonanza")
COVERAGE_COLUMNS = ("peak_intensity_coverage_measured", "peak_intensity_coverage_predicted")


def library_versions(config: GlacierConfig) -> dict:
    versions = {}
    import msentity
    versions["msentity"] = msentity.__version__
    if any(m in MSSIM_METHODS for m in config.similarities):
        import msentity_similarity
        versions["msentity_similarity"] = msentity_similarity.__version__
    if "entropy" in config.similarities:
        import ms_entropy
        versions["ms_entropy"] = getattr(ms_entropy, "__version__", "")
    return versions


class SimilarityCalculator:
    def __init__(self, config: GlacierConfig):
        self.config = config
        self.methods = tuple(config.similarities)
        self.mssim_methods = [m for m in MSSIM_METHODS if m in self.methods]
        self.use_entropy = "entropy" in self.methods
        if self.mssim_methods:
            try:
                import msentity_similarity  # noqa: F401
            except ImportError as exc:
                raise ImportError("msentity-similarity is required; see envs/glacier/requirements-glacier.txt") from exc
        if self.use_entropy:
            try:
                import ms_entropy  # noqa: F401
            except ImportError as exc:
                raise ImportError("The entropy similarity needs the ms_entropy package "
                                  "(pip install ms_entropy) or remove it from glacier.similarities") from exc

    def score(self, query_dataset, query_peaks, candidates) -> dict:
        """Scores of every (query, candidate) pair, query-major: row k = (k // n_c, k % n_c).

        query_dataset / query_peaks: the measured spectra; candidates: msentity
        dataset of the predicted spectra (as prepared by PredictedLibrary.spectra).
        """
        n_q, n_c = len(query_peaks), len(candidates)
        result = {m: np.full(n_q * n_c, np.nan) for m in self.methods}
        for column in COVERAGE_COLUMNS:
            result[column] = np.full(n_q * n_c, np.nan)
        errors = []
        if self.mssim_methods:
            try:
                self._mssim(query_dataset, candidates, n_c, result)
            except Exception as exc:  # noqa: BLE001 - recorded as a failed calculation
                errors.append(f"msentity-similarity: {exc}")
        if self.use_entropy:
            try:
                self._entropy(query_peaks, candidates, n_c, result)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"ms_entropy: {exc}")
        result["error"] = "; ".join(errors)
        return result

    def _mssim(self, query_dataset, candidates, n_c, result):
        import msentity_similarity as mss

        config = self.config
        hits = mss.library_search(
            query_dataset, candidates, threshold=0.0, methods=self.mssim_methods,
            mz_tolerance_da=config.mz_tolerance_da, mz_tolerance_ppm=config.mz_tolerance_ppm,
            intensity_exponent=config.intensity_exponent, threads=config.threads,
            include_matched_data=False)
        if not len(hits):
            return
        flat = np.asarray(hits["index1"], dtype=np.int64) * n_c + np.asarray(hits["index2"], dtype=np.int64)
        for method in self.mssim_methods:
            result[method][flat] = np.array(hits[method], dtype=float)  # None (missing) -> nan
        for ours, theirs in zip(COVERAGE_COLUMNS, ("peak_intensity_coverage1", "peak_intensity_coverage2")):
            result[ours][flat] = np.array(hits[theirs], dtype=float)

    def _entropy(self, query_peaks, candidates, n_c, result):
        from ms_entropy import calculate_entropy_similarity

        options = self.config.entropy
        if self.config.ms2_tolerance_unit == "ppm":
            tolerance = {"ms2_tolerance_in_da": -1, "ms2_tolerance_in_ppm": self.config.ms2_tolerance}
        else:
            tolerance = {"ms2_tolerance_in_da": self.config.ms2_tolerance, "ms2_tolerance_in_ppm": -1}
        reference = []
        for i in range(n_c):
            mz, intensity = candidates.spectrum(i)
            reference.append(np.column_stack([mz, intensity]).astype(np.float32))
        scores = result["entropy"]
        for q, peaks in enumerate(query_peaks):
            measured = np.asarray(peaks, dtype=np.float32).reshape(-1, 2)
            for c in range(n_c):
                if not len(measured) or not len(reference[c]):
                    scores[q * n_c + c] = 0.0  # an empty spectrum matches nothing (as in msentity-similarity)
                    continue
                try:
                    scores[q * n_c + c] = calculate_entropy_similarity(
                        measured, reference[c], clean_spectra=bool(options.get("clean_spectra", True)),
                        noise_threshold=float(options.get("noise_threshold", 0.01)), **tolerance)
                except Exception as exc:  # noqa: BLE001 - one failed pair stays NaN
                    logger.debug("entropy failed for pair (%d, %d): %s", q, c, exc)
