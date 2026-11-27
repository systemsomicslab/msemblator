"""Tests for the GLACIER module with small synthetic libraries (no real data needed).

Run from the repository root (msemblator environment with envs/glacier/requirements-glacier.txt):
    python -m unittest discover -s tests/glacier -t .
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    import msentity as ms
    import msentity_similarity  # noqa: F401
except ImportError:  # pragma: no cover - environment without the GLACIER packages
    ms = None


from msemblator.glacier import labels  # noqa: E402
from msemblator.glacier.adducts import AdductResolver, parse_adduct  # noqa: E402
from msemblator.glacier.config import config_from_dict, load_config  # noqa: E402
from msemblator.glacier.features import feature_columns, rank_and_difference  # noqa: E402
from msemblator.glacier.formula import normalize_formula, parse_formula  # noqa: E402
from msemblator.glacier.library import check_formula_basis  # noqa: E402

KEY_A, KEY_B, KEY_D, KEY_E, KEY_F, KEY_G, KEY_H = (c * 14 for c in "ABDEFGH")
GLUCOSE = "C6H12O6"


def ion_mz(formula, adduct):
    return parse_adduct(adduct).ion_mz(parse_formula(formula))


def library_record(spec_id, formula, ion, ce, key, peaks, suffix="UHFFFAOYSA-N", mz_formula=None):
    """One MSP block of a predicted library; mz_formula is the neutral formula for PrecursorMZ."""
    lines = [f"NAME: {spec_id}", f"SPECID: {spec_id}", f"SMILES: C{spec_id}", f"FORMULA: {formula}",
             f"INCHIKEY: {key}-{suffix}", f"ION: {ion}", f"PredictionCollisionEnergyEV: {ce}",
             f"PrecursorMZ: {ion_mz(mz_formula or formula, ion):.6f}", f"Num Peaks: {len(peaks)}"]
    lines += [f"{mz}\t{intensity}" for mz, intensity in peaks]
    return "\n".join(lines) + "\n"


def write_library(path, records):
    ms.read_text("\n".join(records), "msp").save(str(path))


def query_record(name, adduct, formula, peaks, inchikey=""):
    lines = [f"NAME: {name}", f"PRECURSORTYPE: {adduct}", "PRECURSORMZ: 179.0561"]
    if formula is not None:
        lines.append(f"FORMULA: {formula}")
    if inchikey:
        lines.append(f"INCHIKEY: {inchikey}")
    lines.append(f"Num Peaks: {len(peaks)}")
    lines += [f"{mz}\t{intensity}" for mz, intensity in peaks]
    return "\n".join(lines) + "\n"


QUERY_PEAKS = [(59.0133, 100), (89.0244, 50), (101.0244, 30), (179.0561, 999)]


def build_fixture(folder: Path):
    """Libraries for [M-H]-, [M+H]+ and [M-H2O+H]+ (FORMULA = M - H2O), and a query MSP."""
    minus = [
        # A: the answer; CE10 and CE40
        library_record("A10", GLUCOSE, "[M-H]-", "10.0", KEY_A, [(59.0133, 1.0), (89.0244, 0.5), (179.0561, 0.9)]),
        library_record("A40", GLUCOSE, "[M-H]-", "40.0", KEY_A, [(59.0133, 1.0), (71.0133, 0.5)]),
        # C: same Short InChIKey as A (stereoisomer), CE10 only, better match at CE10
        library_record("C10", GLUCOSE, "[M-H]-", "10.0", KEY_A,
                       [(59.0133, 0.1), (89.0244, 0.05), (101.0244, 0.03), (179.0561, 1.0)],
                       suffix="ZZZZZZZZZZ-N"),
        # B: nothing matches at CE10 (score 0), one peak at CE40 (1.7 ppm away)
        library_record("B10", GLUCOSE, "[M-H]-", "10.0", KEY_B, [(113.0244, 1.0), (131.0350, 0.5)]),
        library_record("B40", GLUCOSE, "[M-H]-", "40.0", KEY_B, [(59.0134, 0.2), (200.0, 1.0)]),
        # D: other formula
        library_record("D10", "C6H6O", "[M-H]-", "10.0", KEY_D, [(93.0346, 1.0)]),
        # F: CE10 only -> CE40 missing
        library_record("F10", GLUCOSE, "[M-H]-", "10.0", KEY_F, [(89.0244, 1.0)]),
        # G: energy given as NCE -> missing; H: 20 eV -> not configured
        library_record("G10", GLUCOSE, "[M-H]-", "NCE=30%", KEY_G, [(59.0133, 1.0)]),
        library_record("H20", GLUCOSE, "[M-H]-", "20.0", KEY_H, [(59.0133, 1.0)]),
    ]
    write_library(folder / "minus_h.msds", minus)
    # E: same formula but [M+H]+: must never be a candidate of a [M-H]- query
    write_library(folder / "plus_h.msds", [
        library_record("E10", GLUCOSE, "[M+H]+", "10.0", KEY_E, [(61.0284, 1.0)]),
        library_record("E40", GLUCOSE, "[M+H]+", "40.0", KEY_E, [(61.0284, 1.0)]),
    ])
    # [M-H2O+H]+ library whose FORMULA is the dehydrated formula (C6H10O5)
    write_library(folder / "dehydrated_modified.msds", [
        library_record("W10", "C6H10O5", "[M+H-H2O]+", "10.0", KEY_A, [(145.0495, 1.0)], mz_formula=GLUCOSE),
    ])
    # the same with FORMULA = neutral molecule
    write_library(folder / "dehydrated_neutral.msds", [
        library_record("N10", GLUCOSE, "[M-H2O+H]+", "10.0", KEY_A, [(145.0495, 1.0)]),
    ])
    queries = [
        query_record("q1", "[M-H]-", GLUCOSE, QUERY_PEAKS, inchikey=f"{KEY_A}-UHFFFAOYSA-N"),
        query_record("q2", "[M-H]1-", "H12C6O6", [(59.0133, 100), (60.0, "nan"), (61.0, 0), (89.0244, 50)]),
        query_record("q3", "[M+Cl]-", GLUCOSE, QUERY_PEAKS),
        query_record("q4", "[M+Na]+", GLUCOSE, QUERY_PEAKS),
        query_record("q5", "[M+H]+", GLUCOSE, [(61.0284, 10)]),
        query_record("q6", "[M-H]-", "C7H8", QUERY_PEAKS),
        query_record("q7", "[M-H]-", None, QUERY_PEAKS),
        query_record("q8", "[M-H]-", "C6H12O6+", QUERY_PEAKS),
        query_record("q9", "[M+H-H2O]+", GLUCOSE, [(145.0495, 10)]),
        query_record("q10", "[M-H]-", GLUCOSE, [(179.0561, 999)]),  # precursor peak only
    ]
    (folder / "queries.msp").write_text("\n".join(queries), encoding="utf-8")


def make_config(folder: Path, **changes):
    settings = {
        "library_dir": str(folder),
        "libraries": {"[M-H]-": "minus_h.msds", "[M+H]+": "plus_h.msds",
                      "[M-H2O+H]+": "dehydrated_modified.msds", "[M+Na]+": "missing.msds"},
        "adduct_aliases": {"[M+H-H2O]+": "[M-H2O+H]+"},
        "collision_energies": [10, 40],
        "similarities": ["cosine", "reverse_cosine"],
        "adduct_features": ["[M+H]+", "[M-H]-"],
        # The fixture scores are worked out on the full peak lists.
        "precursor_peak_removal": {"enabled": False},
    }
    settings.update(changes)
    return config_from_dict(settings)


class FormulaAndAdductTest(unittest.TestCase):
    def test_formula_keys_compare_element_counts(self):
        self.assertEqual(normalize_formula("H12C6O6"), "C6H12O6")
        self.assertEqual(normalize_formula("CH3COOH"), "C2H4O2")
        self.assertEqual(normalize_formula("C2H6Se"), "C2H6Se")
        self.assertEqual(normalize_formula("NaCl"), "ClNa")
        for text in ("C6H12O6+", "[13C]H4", "", None, "c6h12o6"):
            self.assertIsNone(normalize_formula(text), text)

    def test_adduct_parsing_and_ion_mz(self):
        self.assertAlmostEqual(ion_mz(GLUCOSE, "[M-H]-"), 179.05611, places=4)
        self.assertAlmostEqual(ion_mz(GLUCOSE, "[M+Na]+"), 203.05261, places=4)
        self.assertAlmostEqual(ion_mz(GLUCOSE, "[M-H2O+H]+"), 163.06009, places=4)
        self.assertEqual(parse_adduct("[M-H2O+H]+").neutral_modification, {"H": -2, "O": -1})
        self.assertEqual(parse_adduct("[M+NH4]+").neutral_modification, {})
        self.assertEqual(parse_adduct("[M-H]-").neutral_modification, {})

    def test_resolver_uses_configured_names_and_aliases(self):
        resolver = AdductResolver(["[M-H]-", "[M-H2O+H]+"], {"[M+H-H2O]+": "[M-H2O+H]+"})
        self.assertEqual(resolver.resolve("[M-H]1-"), "[M-H]-")
        self.assertEqual(resolver.resolve(" [M - H]- "), "[M-H]-")
        self.assertEqual(resolver.resolve("[M+H-H2O]+"), "[M-H2O+H]+")
        self.assertIsNone(resolver.resolve("[M+Cl]-"))

    def test_formula_basis_detection(self):
        neutral = check_formula_basis([GLUCOSE], [ion_mz(GLUCOSE, "[M-H2O+H]+")], "[M-H2O+H]+")
        modified = check_formula_basis(["C6H10O5"], [ion_mz(GLUCOSE, "[M-H2O+H]+")], "[M-H2O+H]+")
        self.assertEqual(neutral["detected"], "neutral")
        self.assertEqual(modified["detected"], "modified")
        unknown = check_formula_basis([GLUCOSE], [300.0], "[M-H2O+H]+")
        self.assertIsNone(unknown["detected"])

    def test_default_config_is_valid(self):
        config = load_config()
        self.assertIn("[M-H]-", config.libraries)
        self.assertEqual(config.mz_tolerance_ppm, 20.0)
        self.assertEqual(config.mz_tolerance_da, 0.0)
        # Default: the precursor peak is removed from the measured spectra only.
        self.assertTrue(config.precursor_peak_removal["enabled"])
        self.assertTrue(config.removes_precursor("measured"))
        self.assertFalse(config.removes_precursor("predicted"))
        bare = config_from_dict({"library_dir": ".", "libraries": {"[M-H]-": "x"}})
        self.assertEqual(bare.precursor_peak_removal, config.precursor_peak_removal)
        with self.assertRaises(ValueError):
            config_from_dict({"library_dir": ".", "libraries": {"[M-H]-": "x"}, "similarities": ["dot"]})
        for bad in ({"tolerance_unit": "mDa"}, {"spectra": "predicted"}, {"tolerance": 0}, {"enabled": "yes"}):
            with self.assertRaises(ValueError, msg=bad):
                config_from_dict({"library_dir": ".", "libraries": {"[M-H]-": "x"},
                                  "precursor_peak_removal": bad})

    def test_precursor_window(self):
        from msemblator.glacier.preprocessing import precursor_peak_mask

        mz = np.array([179.0561, 179.0595, 179.0598, 179.065, 59.0])
        ppm = {"tolerance": 20, "tolerance_unit": "ppm"}
        da = {"tolerance": 0.01, "tolerance_unit": "da"}
        self.assertEqual(precursor_peak_mask(mz, 179.0561, ppm).tolist(), [True, True, False, False, False])
        self.assertEqual(precursor_peak_mask(mz, 179.0561, da).tolist(), [True, True, True, True, False])
        self.assertFalse(precursor_peak_mask(mz, float("nan"), ppm).any())


class RankTest(unittest.TestCase):
    def test_rank_and_score_difference_follow_existing_definition(self):
        frame = pd.DataFrame({
            "query_index": [1, 1, 1, 1, 2, 2],
            "Short_InChIKey": ["D", "A", "C", "B", "X", "Y"],
            "score": [0.2, 0.9, 0.5, 0.5, np.nan, 0.0],
        })
        rank, difference = rank_and_difference(frame, "score")
        self.assertEqual(rank.tolist(), [4, 1, 3, 2, pd.NA, 1])  # tie 0.5: B before C
        expected = [0.0, 0.4, 0.3, 0.0, np.nan, 0.0]
        for got, want in zip(difference.tolist(), expected):
            if math.isnan(want):
                self.assertTrue(math.isnan(got))
            else:
                self.assertAlmostEqual(got, want)


@unittest.skipIf(ms is None, "msentity / msentity-similarity are not installed")
class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from msemblator.glacier.pipeline import run_glacier

        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        build_fixture(cls.folder)
        cls.config = make_config(cls.folder)
        cls.result = run_glacier(cls.folder / "queries.msp", cls.config)
        cls.queries = cls.result.queries.set_index("filename")
        cls.raw, cls.features = cls.result.raw, cls.result.features

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self, frame, name):
        return frame[frame["filename"] == name]

    def test_query_status(self):
        status = self.queries["status"].to_dict()
        self.assertEqual(status, {
            "q1": "ok", "q2": "ok", "q3": "unsupported_adduct", "q4": "library_not_available",
            "q5": "ok", "q6": "no_candidate", "q7": "missing_formula", "q8": "invalid_formula", "q9": "ok",
            "q10": "ok"})
        self.assertTrue((self.result.queries["precursor_peak_removal"] == "off").all())

    def test_invalid_peaks_are_removed(self):
        self.assertEqual(self.queries.loc["q2", "n_peaks_raw"], 4)
        self.assertEqual(self.queries.loc["q2", "n_peaks"], 2)
        self.assertEqual(self.queries.loc["q2", "n_invalid_peaks"], 2)

    def test_queries_do_not_carry_answers(self):
        self.assertFalse({"inchikey", "InChIKey", "smiles", "SMILES"} & set(self.result.queries.columns))

    def test_candidates_come_from_the_query_adduct_only(self):
        minus = set(self.rows(self.features, "q1")["Short_InChIKey"])
        self.assertEqual(minus, {KEY_A, KEY_B, KEY_F})
        self.assertNotIn(KEY_E, set(self.rows(self.raw, "q1")["Short_InChIKey"]))
        self.assertEqual(set(self.rows(self.features, "q5")["Short_InChIKey"]), {KEY_E})
        self.assertTrue((self.rows(self.raw, "q1")["adduct"] == "[M-H]-").all())

    def test_collision_energy_aggregation(self):
        q1 = self.rows(self.features, "q1").set_index("Short_InChIKey")
        raw = self.rows(self.raw, "q1")
        a = raw[raw["Short_InChIKey"] == KEY_A]
        best_ce10 = a.loc[a["collision_energy"] == 10, "cosine"].max()
        self.assertEqual(len(a), 3)  # A10, A40 and C10 kept in the raw output
        self.assertAlmostEqual(q1.loc[KEY_A, "GLACIER_Cosine_CE10"], best_ce10)
        self.assertEqual(q1.loc[KEY_A, "n_spectra"], 3)
        self.assertEqual(q1.loc[KEY_A, "n_inchikeys"], 2)
        self.assertAlmostEqual(q1.loc[KEY_A, "GLACIER_Cosine_Max"],
                               max(q1.loc[KEY_A, "GLACIER_Cosine_CE10"], q1.loc[KEY_A, "GLACIER_Cosine_CE40"]))
        # missing energy stays missing; Max uses the energies that exist
        self.assertTrue(math.isnan(q1.loc[KEY_F, "GLACIER_Cosine_CE40"]))
        self.assertAlmostEqual(q1.loc[KEY_F, "GLACIER_Cosine_Max"], q1.loc[KEY_F, "GLACIER_Cosine_CE10"])
        # no matched peak is a real score of 0, not a missing value
        self.assertEqual(q1.loc[KEY_B, "GLACIER_Cosine_CE10"], 0.0)

    def test_energy_without_value_or_not_configured(self):
        raw = self.rows(self.raw, "q1")
        g = raw[raw["Short_InChIKey"] == KEY_G]
        self.assertEqual(g["status"].tolist(), ["missing_collision_energy"])
        self.assertTrue(g["collision_energy"].isna().all())
        self.assertNotIn(KEY_G, set(self.rows(self.features, "q1")["Short_InChIKey"]))
        self.assertNotIn(KEY_H, set(raw["Short_InChIKey"]))

    def test_rank_order_within_query(self):
        q1 = self.rows(self.features, "q1").sort_values("GLACIER_Cosine_Rank")
        self.assertEqual(q1["GLACIER_Cosine_Rank"].tolist(), [1, 2, 3])
        self.assertEqual(q1.iloc[0]["Short_InChIKey"], KEY_A)
        scores = q1["GLACIER_Cosine_Max"].to_numpy()
        np.testing.assert_allclose(q1["GLACIER_Cosine_ScoreDiff"].to_numpy(), np.r_[scores[:-1] - scores[1:], 0])

    def test_feature_columns(self):
        self.assertEqual(len(feature_columns(self.config)), 11)  # 2 similarities x 5 + candidate count
        counts = self.features.groupby("filename")["GLACIER_Candidate_Count"].agg(["min", "max", "size"])
        self.assertEqual(counts.loc["q1"].tolist(), [3, 3, 3])
        self.assertEqual(counts.loc["q5"].tolist(), [1, 1, 1])
        for column in feature_columns(self.config) + ["adduct_MminusHminus", "adduct_MplusHplus"]:
            self.assertIn(column, self.features.columns)
        q1 = self.rows(self.features, "q1")
        self.assertTrue((q1["adduct_MminusHminus"] == 1).all() and (q1["adduct_MplusHplus"] == 0).all())

    def test_modified_formula_library_is_converted_once(self):
        q9 = self.rows(self.raw, "q9")
        self.assertEqual(q9["candidate_spec_id"].tolist(), ["W10"])
        self.assertEqual(q9["library_formula"].iloc[0], "C6H10O5")

    def test_neutral_formula_library_for_the_same_adduct(self):
        from msemblator.glacier.pipeline import run_glacier

        config = make_config(self.folder, libraries={"[M-H2O+H]+": "dehydrated_neutral.msds"})
        result = run_glacier(self.folder / "queries.msp", config, adducts=["[M-H2O+H]+"])
        raw = result.raw
        self.assertEqual(raw["candidate_spec_id"].tolist(), ["N10"])
        self.assertEqual(raw["library_formula"].iloc[0], GLUCOSE)
        self.assertEqual(result.libraries[0]["formula_basis"], "neutral")

    def test_new_library_is_used_once_registered(self):
        from msemblator.glacier.pipeline import run_glacier

        write_library(self.folder / "plus_na.msds", [
            library_record("S10", GLUCOSE, "[M+Na]+", "10.0", KEY_A, [(203.0526, 1.0)])])
        config = make_config(self.folder, libraries={"[M+Na]+": "plus_na.msds"})
        result = run_glacier(self.folder / "queries.msp", config, adducts=["[M+Na]+"])
        status = result.queries.set_index("filename")["status"]
        self.assertEqual(status["q4"], "ok")
        self.assertEqual(result.raw["candidate_spec_id"].tolist(), ["S10"])

    def test_unknown_adduct_notation_in_library_is_reported(self):
        from msemblator.glacier.pipeline import run_glacier

        write_library(self.folder / "plus_na_odd.msds", [
            library_record("S10", GLUCOSE, "[M+Na]+", "10.0", KEY_A, [(203.0526, 1.0)]).replace(
                "ION: [M+Na]+", "ION: M+Na")])
        config = make_config(self.folder, libraries={"[M+Na]+": "plus_na_odd.msds"})
        result = run_glacier(self.folder / "queries.msp", config, adducts=["[M+Na]+"])
        query = result.queries.set_index("filename").loc["q4"]
        self.assertEqual(query["status"], "library_not_available")
        self.assertIn("M+Na", query["message"])
        self.assertEqual(result.libraries[0]["status"], "unusable")

    def test_failed_calculation_is_missing_not_zero(self):
        from msemblator.glacier.pipeline import run_glacier

        with mock.patch("msentity_similarity.library_search", side_effect=RuntimeError("boom")):
            result = run_glacier(self.folder / "queries.msp", self.config, adducts=["[M-H]-"])
        self.assertTrue(result.raw["cosine"].isna().all())
        self.assertTrue((result.raw["status"] == "similarity_failed").all())
        self.assertIn("boom", result.raw["message"].iloc[0])
        status = result.queries.set_index("filename")["status"]
        self.assertEqual(status["q1"], "similarity_failed")
        self.assertTrue(result.features["GLACIER_Cosine_Max"].isna().all())

    def removal_run(self, spectra):
        from msemblator.glacier.pipeline import run_glacier

        removal = {"enabled": True, "tolerance": 20, "tolerance_unit": "ppm", "spectra": spectra}
        config = make_config(self.folder, precursor_peak_removal=removal)
        return run_glacier(self.folder / "queries.msp", config, adducts=["[M-H]-"])

    def test_precursor_peak_removal_measured(self):
        result = self.removal_run("measured")
        queries = result.queries.set_index("filename")
        self.assertEqual(queries.loc["q1", "precursor_peak_removal"], "removed")
        self.assertEqual(queries.loc["q1", "n_precursor_peaks_removed"], 1)
        self.assertEqual(queries.loc["q1", "n_peaks"], 3)
        self.assertEqual(queries.loc["q2", "precursor_peak_removal"], "not_found")
        self.assertEqual(queries.loc["q10", "status"], "no_valid_peaks")
        self.assertIn("precursor", queries.loc["q10", "message"])
        raw = result.raw[result.raw["filename"] == "q1"]
        self.assertTrue((raw["n_peaks_measured"] == 3).all())
        a10 = raw[raw["candidate_spec_id"] == "A10"].iloc[0]
        self.assertEqual(a10["n_peaks_predicted"], 3)  # predicted spectra unchanged
        self.assertEqual(result.libraries[0]["n_precursor_peaks_removed"], 0)
        # the summary counts only the [M-H]- queries of this run (q1, q6, q7, q8, q10), not q3-q5, q9
        from msemblator.glacier.evaluation import summarize
        removal = summarize(result)["precursor_peak_removal"]
        self.assertEqual(removal["measured_peaks_removed"], 5)
        self.assertEqual(removal["measured_spectra"], {"removed": 5, "not_found": 1})

    def test_precursor_peak_removal_both(self):
        result = self.removal_run("both")
        raw = result.raw[result.raw["filename"] == "q1"].set_index("candidate_spec_id")
        self.assertEqual(raw.loc["A10", "n_peaks_predicted"], 2)
        self.assertEqual(raw.loc["C10", "n_peaks_predicted"], 3)
        self.assertEqual(raw.loc["B10", "n_peaks_predicted"], 2)
        self.assertEqual(result.libraries[0]["n_spectra_precursor_removed"], 2)  # A10 and C10
        # A10 without its precursor peak: 59.0133 / 89.0244 vs the measured 59 / 89 / 101 peaks
        a, r = np.array([100.0, 50.0, 30.0]), np.array([1.0, 0.5])
        expected = (a[:2] @ r) / (np.linalg.norm(a) * np.linalg.norm(r))
        self.assertAlmostEqual(raw.loc["A10", "cosine"], expected, places=5)

    def test_labels(self):
        reference = labels.read_reference(self.folder / "queries.msp")
        labelled = labels.attach_labels(self.features, reference)
        q1 = self.rows(labelled, "q1").set_index("Short_InChIKey")["TF"]
        self.assertEqual(q1.to_dict(), {KEY_A: 1, KEY_B: 0, KEY_F: 0})
        self.assertEqual(int(self.rows(labelled, "q2")["TF"].sum()), 0)  # q2 has no answer


class IntegrationTest(unittest.TestCase):
    def test_outer_join_keeps_candidates_of_both_sides(self):
        from msemblator.glacier.integration import merge_with_candidates

        existing = pd.DataFrame({"filename": [1, 1, 2], "adduct": ["[M-H]-"] * 3,
                                 "Short_InChIKey": ["A", "B", "C"], "SMILES": ["a", "b", "c"],
                                 "normalization_score_metfrag": [1.0, 0.5, 0.2]})
        glacier = pd.DataFrame({"filename": ["1", "1"], "adduct": ["[M-H]-"] * 2,
                                "Short_InChIKey": ["A", "G"], "SMILES": ["a2", "g"],
                                "GLACIER_Cosine_Max": [0.8, 0.0], "GLACIER_Candidate_Count": [2, 2]})
        merged = merge_with_candidates(existing, glacier).set_index(["filename", "Short_InChIKey"])
        self.assertEqual(merged.loc[("1", "B"), "GLACIER_Candidate_Count"], 2)  # same query, not scored
        self.assertTrue(math.isnan(merged.loc[("2", "C"), "GLACIER_Candidate_Count"]))  # no GLACIER output
        self.assertEqual(merged["candidate_source"].to_dict(), {
            ("1", "A"): "both", ("1", "B"): "existing", ("2", "C"): "existing", ("1", "G"): "glacier"})
        self.assertEqual(merged.loc[("1", "G"), "SMILES"], "g")
        self.assertEqual(merged.loc[("1", "A"), "SMILES"], "a")
        self.assertTrue(math.isnan(merged.loc[("2", "C"), "GLACIER_Cosine_Max"]))
        self.assertEqual(merged.loc[("1", "G"), "GLACIER_Cosine_Max"], 0.0)


@unittest.skipIf(ms is None, "msentity / msentity-similarity are not installed")
class SimilarityDirectionTest(unittest.TestCase):
    def scores(self, measured, predicted, methods=("cosine", "reverse_cosine")):
        from msemblator.glacier.similarity import SimilarityCalculator

        config = config_from_dict({"library_dir": ".", "libraries": {"[M-H]-": "x"}, "similarities": list(methods)})
        query = ms.from_peaks([measured])
        library = ms.from_peaks([predicted])
        return SimilarityCalculator(config).score(query, [np.asarray(measured, float)], library)

    def test_reverse_cosine_ignores_extra_measured_peaks_only(self):
        base = [[100.0, 1.0], [150.0, 2.0]]
        extra_measured = self.scores(base + [[300.0, 5.0]], base)
        extra_predicted = self.scores(base, base + [[300.0, 5.0]])
        self.assertAlmostEqual(extra_measured["reverse_cosine"][0], 1.0, places=5)
        self.assertLess(extra_measured["cosine"][0], 0.5)
        self.assertLess(extra_predicted["reverse_cosine"][0], 0.5)
        self.assertAlmostEqual(extra_predicted["peak_intensity_coverage_measured"][0], 1.0, places=5)

    def test_ppm_tolerance_is_inclusive_and_one_to_one(self):
        inside = self.scores([[100.0, 1.0]], [[100.0019, 1.0]])
        outside = self.scores([[100.0, 1.0]], [[100.0021, 1.0]])
        self.assertAlmostEqual(inside["cosine"][0], 1.0, places=5)
        self.assertEqual(outside["cosine"][0], 0.0)
        # two measured peaks near one predicted peak: only the closer one pairs
        one_to_one = self.scores([[100.0, 1.0], [100.001, 1.0]], [[100.0012, 1.0]])
        self.assertAlmostEqual(one_to_one["cosine"][0], 1 / math.sqrt(2), places=5)

    @staticmethod
    def reference_entropy(measured, predicted, weighted):
        """Li et al. (2021) on spectra whose peaks are far apart (so the pairing is unambiguous)."""
        def probabilities(peaks):
            p = {mz: i for mz, i in peaks}
            total = sum(p.values())
            p = {mz: i / total for mz, i in p.items()}
            h = -sum(v * math.log(v) for v in p.values())
            if weighted and h < 3:
                p = {mz: v ** (0.25 + 0.25 * h) for mz, v in p.items()}
                total = sum(p.values())
                p = {mz: v / total for mz, v in p.items()}
            return p

        def f(x):
            return x * math.log2(x) if x > 0 else 0.0

        a, b = probabilities(measured), probabilities(predicted)
        return sum((f(a[mz] + b[mz]) - f(a[mz]) - f(b[mz])) / 2 for mz in a if mz in b)

    def test_entropy_similarity(self):
        measured = [[100.0, 9.0], [150.0, 1.0], [210.0, 3.0]]
        predicted = [[100.0, 1.0], [210.0, 2.0], [300.0, 4.0]]
        scores = self.scores(measured, predicted, methods=("entropy", "unweighted_entropy"))
        self.assertAlmostEqual(scores["entropy"][0], self.reference_entropy(measured, predicted, True), places=5)
        self.assertAlmostEqual(scores["unweighted_entropy"][0],
                               self.reference_entropy(measured, predicted, False), places=5)
        same = self.scores([[100.0, 1.0], [150.0, 2.0]], [[100.0, 1.0], [150.0, 2.0]], methods=("entropy",))
        disjoint = self.scores([[100.0, 1.0]], [[200.0, 1.0]], methods=("entropy",))
        self.assertAlmostEqual(same["entropy"][0], 1.0, places=5)
        self.assertEqual(disjoint["entropy"][0], 0.0)

    def test_old_entropy_options_are_ignored(self):
        settings = {"library_dir": ".", "libraries": {"[M-H]-": "x.msds"}, "similarities": ["entropy"],
                    "entropy": {"clean_spectra": True, "noise_threshold": 0.01}}
        with self.assertLogs("msemblator.glacier.config", level="WARNING"):
            config = config_from_dict(settings)
        self.assertEqual(config.similarities, ("entropy",))
        self.assertEqual(config.method_label("unweighted_entropy"), "UnweightedEntropy")


if __name__ == "__main__":
    unittest.main()
