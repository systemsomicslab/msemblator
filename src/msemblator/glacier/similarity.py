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

entropy / unweighted_entropy (also msentity-similarity, from 2.1.0)
    Spectral entropy similarity (Li et al., 2021) on the same one-to-one peak
    pairs. Intensities are normalized to sum 1; for ``entropy`` a spectrum with
    entropy H < 3 is reweighted (p ** (0.25 + 0.25 H), normalized again). Each
    matched pair adds (f(p + q) - f(p) - f(q)) / 2 with f(x) = x log2 x. There is
    no peak cleaning (no noise filter, no merging of close peaks); it is symmetric.

Scores are in [0, 1]; a pair without matched peaks scores 0. A calculation that
fails leaves NaN, so it is never mistaken for a real score of 0.
"""
from __future__ import annotations

import logging

import numpy as np

from msemblator.glacier.config import GlacierConfig

logger = logging.getLogger(__name__)

MSSIM_METHODS = ("cosine", "reverse_cosine", "modified_cosine", "bonanza", "entropy", "unweighted_entropy")
COVERAGE_COLUMNS = ("peak_intensity_coverage_measured", "peak_intensity_coverage_predicted")


def library_versions(config: GlacierConfig) -> dict:
    import msentity
    import msentity_similarity

    return {"msentity": msentity.__version__, "msentity_similarity": msentity_similarity.__version__}


class SimilarityCalculator:
    def __init__(self, config: GlacierConfig):
        self.config = config
        self.methods = tuple(config.similarities)
        self.mssim_methods = [m for m in MSSIM_METHODS if m in self.methods]
        try:
            import msentity_similarity
        except ImportError as exc:
            raise ImportError("msentity-similarity is required; see envs/glacier/requirements-glacier.txt") from exc
        entropy = [m for m in self.mssim_methods if m.endswith("entropy")]
        if entropy and tuple(int(x) for x in msentity_similarity.__version__.split(".")[:2]) < (2, 1):
            raise ImportError(f"{', '.join(entropy)} needs msentity-similarity 2.1 or newer "
                              f"(installed: {msentity_similarity.__version__}); see envs/glacier/requirements-glacier.txt")

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
        try:
            self._mssim(query_dataset, candidates, n_c, result)
        except Exception as exc:  # noqa: BLE001 - recorded as a failed calculation
            errors.append(f"msentity-similarity: {exc}")
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
