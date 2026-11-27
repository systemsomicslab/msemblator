"""Tests for the SIRIUS-free annotation runners (no external tool needed).

Run from the repository root:
    python -m unittest discover -s tests -t .
Set MSEMBLATOR_RUN_TOOL_TESTS=1 to also run MSBuddy / MS-FINDER on the demo MSP.
"""
import csv
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from msemblator.sirius_free import msp_io  # noqa: E402
from msemblator.sirius_free.annotators import fiddle_runner, msfinder_runner  # noqa: E402

DEMO_MSP = ROOT / "tests" / "fixtures" / "sirius_free_demo.msp"


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class MspReaderTest(unittest.TestCase):
    def setUp(self):
        self.records = msp_io.read_msp_records(DEMO_MSP)
        self.by_name = {r.name: r for r in self.records}

    def test_query_ids_are_sequential(self):
        self.assertEqual(len(self.records), 31)
        self.assertEqual([r.query_id for r in self.records], [str(i) for i in range(1, 32)])

    def test_query_ids_match_default_workflow_numbering(self):
        from msemblator.formats.msp_format_change import convert_name_to_peakid, modify_msp_data_type
        _, name_df = convert_name_to_peakid(modify_msp_data_type(str(DEMO_MSP)))
        expected = dict(zip(name_df["Original_NAME"], name_df["Updated_NAME"]))
        self.assertEqual({r.name: r.query_id for r in self.records}, expected)

    def test_metadata_needed_by_fiddle_is_kept(self):
        record = self.by_name["SYN_pos_na"]
        self.assertEqual(record.collision_energy, "NCE=35%")
        self.assertEqual(record.instrument_type, "HCD")
        self.assertEqual(record.ion_mode, "positive")
        self.assertEqual(self.by_name["SYN_no_ce"].collision_energy, "")
        self.assertEqual(self.by_name["SYN_few_peaks"].n_peaks, 3)

    def test_adduct_canonicalisation(self):
        self.assertEqual(self.by_name["SYN_formate"].adduct, "[M+FA-H]-")
        self.assertIsNone(self.by_name["SYN_unknown_adduct"].adduct)
        self.assertEqual(msp_io.canonical_adduct("[M + CHO2]-"), "[M+FA-H]-")
        self.assertEqual(msp_io.canonical_adduct("[M+H+NH3]+"), "[M+NH4]+")
        # [M+CO2]- is a different ion; it must not be mapped to formate.
        self.assertIsNone(msp_io.canonical_adduct("[M+CO2]-"))
        self.assertIsNone(msp_io.canonical_adduct("[2M-H]-"))
        self.assertIsNone(msp_io.canonical_adduct(""))

    def test_ion_mode(self):
        self.assertEqual(msp_io.ion_mode_of("[M+Cl]-"), "negative")
        self.assertEqual(msp_io.ion_mode_of("[M+H]+"), "positive")
        self.assertEqual(msp_io.ion_mode_of("", "N"), "negative")
        self.assertIsNone(msp_io.ion_mode_of("", ""))


class InstrumentRuleTest(unittest.TestCase):
    def model(self, instrument_type, instrument="", default=None):
        record = msp_io.SpectrumRecord("1", "x", 100.0, "[M+H]+", "[M+H]+", "positive", "35",
                                       instrument_type, instrument, "", "", "")
        return fiddle_runner.instrument_model(record, fiddle_runner.DEFAULT_PARAMETERS["instrument_rules"], default)

    def test_rules(self):
        self.assertEqual(self.model("HCD", "Thermo Finnigan Elite Orbitrap"), "orbitrap")
        self.assertEqual(self.model("IT-FT/ion trap with FTMS"), "orbitrap")
        self.assertEqual(self.model("Q-TOF"), "qtof")
        self.assertEqual(self.model("LC-ESI-QTOF"), "qtof")
        self.assertEqual(self.model("timsTOF Pro"), "timstof")
        self.assertIsNone(self.model(""))
        self.assertEqual(self.model("", default="qtof"), "qtof")


class _FakePredictor:
    """Stands in for msfiddle.MsFiddlePredictor; records what it receives."""
    seen = []
    energies = {}

    def __init__(self, *, instrument_type, device, batch_size):
        self.instrument_type = instrument_type
        self.resume_path = Path("missing_tcn.pt")
        self.rescore_resume_path = Path("missing_rescore.pt")

    def _encode_spectrum(self, *, mz_array, intensity_array, precursor_mz, adduct, collision_energy, spec_id):
        if len(mz_array) < 5:
            raise ValueError("Spectrum must contain at least 5 peaks")
        if "," in str(collision_energy):
            raise ValueError("Stepped collision energy is not supported")

    def predict_batch(self, spectra, top_k):
        out = []
        for s in spectra:
            _FakePredictor.seen.append((s["id"], s["adduct"], self.instrument_type))
            _FakePredictor.energies[s["id"]] = s["collision_energy"]
            candidates = [] if s["id"] == "4" else [
                {"formula": "C7H10O5", "score": 0.9, "mass": 174.05, "metadata": {"raw_rescore": 0.9}},
                {"formula": "C6H10N2O4", "score": 0.1, "mass": 174.06, "metadata": {"raw_rescore": 0.1}},
            ]
            out.append({"id": s["id"], "candidates": candidates, "metadata": {}})
        return out


class FiddleRunnerTest(unittest.TestCase):
    def test_statuses_and_adducts(self):
        _FakePredictor.seen = []
        fake = types.ModuleType("msfiddle")
        fake.MsFiddlePredictor = _FakePredictor
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"msfiddle": fake}):
            manifest = fiddle_runner.run(DEMO_MSP, tmp)
            status = {r["query_id"]: r for r in read_csv(Path(tmp) / "fiddle" / "fiddle_status.csv")}
            candidates = read_csv(Path(tmp) / "fiddle" / "fiddle_candidates.csv")
        names = {r.name: r.query_id for r in msp_io.read_msp_records(DEMO_MSP)}

        def state(name):
            return status[names[name]]["status"]

        self.assertEqual(state("SYN_no_ce"), "missing_collision_energy")
        self.assertEqual(state("SYN_cl"), "unsupported_adduct")
        self.assertEqual(state("SYN_unknown_adduct"), "unsupported_adduct")
        self.assertEqual(state("EMBL_MCF_2_0_HRMS_Library000473"), "unsupported_adduct")  # [2M-H]-
        self.assertEqual(state("SYN_few_peaks"), "rejected_by_tool")
        self.assertEqual(state("SYN_stepped_ce"), "rejected_by_tool")
        self.assertEqual(state("SYN_no_instrument"), "unknown_instrument")
        self.assertEqual(status["4"]["status"], "no_candidate")
        self.assertEqual(state("SYN_formate"), "ok")
        self.assertEqual(len(status), 31)

        sent = {qid: (adduct, model) for qid, adduct, model in _FakePredictor.seen}
        self.assertEqual(sent[names["SYN_formate"]], ("[M+FA-H]-", "orbitrap"))
        self.assertEqual(sent[names["SYN_pos_nh4"]], ("[M+NH4]+", "qtof"))
        self.assertNotIn(names["SYN_cl"], sent)  # never converted, never sent
        self.assertNotIn(names["SYN_no_ce"], sent)  # policy A: no default collision energy

        ranks = [(r["query_id"], r["rank"]) for r in candidates if r["query_id"] == names["SYN_formate"]]
        self.assertEqual(ranks, [(names["SYN_formate"], "1"), (names["SYN_formate"], "2")])
        self.assertEqual(manifest["n_spectra"], 31)
        self.assertEqual(sum(manifest["status_counts"].values()), 31)


class FiddleCollisionEnergyTest(unittest.TestCase):
    def test_energy_choice(self):
        choose = fiddle_runner.collision_energy_for
        self.assertEqual(choose("35", None), ("35", "msp"))
        self.assertEqual(choose("", None), (None, ""))                        # policy A
        self.assertEqual(choose("", 30), ("30", "default"))
        self.assertEqual(choose("35", 30), ("35", "msp"))                     # the MSP value wins ...
        self.assertEqual(choose("35", 30, override=True), ("30", "override"))  # ... unless overridden
        self.assertEqual(choose("NCE=27% 30V", None), ("NCE=27% 30eV", "msp"))
        self.assertEqual(choose("", 22.5), ("22.5", "default"))

    def run_demo(self, **parameters):
        _FakePredictor.seen, _FakePredictor.energies = [], {}
        fake = types.ModuleType("msfiddle")
        fake.MsFiddlePredictor = _FakePredictor
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"msfiddle": fake}):
            manifest = fiddle_runner.run(DEMO_MSP, tmp, parameters)
            status = {r["query_id"]: r for r in read_csv(Path(tmp) / "fiddle" / "fiddle_status.csv")}
        names = {r.name: r.query_id for r in msp_io.read_msp_records(DEMO_MSP)}
        return manifest, status, names

    def test_default_energy_for_spectra_without_one(self):
        manifest, status, names = self.run_demo(default_collision_energy=30)
        no_ce = names["SYN_no_ce"]
        self.assertEqual(status[no_ce]["status"], "ok")
        self.assertIn("30 eV from the settings (none in the MSP)", status[no_ce]["message"])
        self.assertEqual(_FakePredictor.energies[no_ce], "30")
        formate = names["SYN_formate"]
        self.assertNotEqual(_FakePredictor.energies[formate], "30")           # MSP value kept
        self.assertEqual(manifest["parameters"]["collision_energy_from_settings"], {"default": 1, "override": 0})

    def test_override_for_every_spectrum(self):
        _, status, names = self.run_demo(default_collision_energy=25, override_collision_energy=True)
        self.assertTrue(_FakePredictor.energies)
        self.assertEqual(set(_FakePredictor.energies.values()), {"25"})
        # the stepped energy of the MSP is replaced, so FIDDLE accepts the spectrum
        self.assertEqual(status[names["SYN_stepped_ce"]]["status"], "ok")
        self.assertIn("replaces the MSP value", status[names["SYN_formate"]]["message"])


class FiddleReferenceTest(unittest.TestCase):
    def test_reference_matches_demo_statuses(self):
        reference = read_csv(ROOT / "envs" / "fiddle" / "reference" / "fiddle_status.csv")
        self.assertEqual(len(reference), 31)
        names = {r.query_id: r.name for r in msp_io.read_msp_records(DEMO_MSP)}
        by_name = {names[r["query_id"]]: r["status"] for r in reference}
        self.assertEqual(by_name["SYN_cl"], "unsupported_adduct")
        self.assertEqual(by_name["SYN_no_ce"], "missing_collision_energy")


class MsfinderParserTest(unittest.TestCase):
    def test_rank_follows_file_order_per_spectrum(self):
        header = ("File path\tFile name\tTitle\tMS1 count\tMSMS count\tPrecursor mz\tPrecursor type\t"
                  "Formula\tTheoretical mass\tMass error\tFormula score\tDatabases\n")
        rows = [
            "p\t1\t1\t0\t5\t175.06\t[M-H]-\tC7H10O5\t174.05\t0.0001\t4.1\tChEBI\n",
            "p\t1\t1\t0\t5\t175.06\t[M-H]-\tC6H10N2O4\t174.06\t-0.002\t3.2\t\n",
            "p\t2\t2\t0\t5\t131.07\t[M-H]-\tC6H12O3\t132.07\t0.0003\t3.9\tHMDB\n",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "Formula result-0001.txt").write_text(header + "".join(rows), encoding="utf-8")
            df = msfinder_runner.parse_formula_results(str(Path(tmp) / "Formula result*.txt"), top_n=100)
        self.assertEqual(df["query_id"].tolist(), ["1", "1", "2"])
        self.assertEqual(df["rank"].tolist(), [1, 2, 1])
        self.assertEqual(df["score"].tolist(), [4.1, 3.2, 3.9])

    def test_spectrum_file_has_no_answer_fields(self):
        record = msp_io.read_msp_records(DEMO_MSP)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1.msp"
            msfinder_runner._write_spectrum_msp(record, "[M-H]-", path)
            text = path.read_text(encoding="utf-8")
        self.assertNotIn("FORMULA", text.upper())
        self.assertNotIn("SMILES", text.upper())
        self.assertIn("PRECURSORTYPE: [M-H]-", text)


@unittest.skipUnless(os.environ.get("MSEMBLATOR_RUN_TOOL_TESTS") == "1", "set MSEMBLATOR_RUN_TOOL_TESTS=1")
class ToolIntegrationTest(unittest.TestCase):
    def test_msbuddy_on_demo(self):
        from msemblator.sirius_free.annotators import msbuddy_runner
        with tempfile.TemporaryDirectory() as tmp:
            manifest = msbuddy_runner.run(DEMO_MSP, tmp)
        self.assertEqual(manifest["status_counts"].get("unsupported_adduct"), 3)
        self.assertEqual(manifest["n_spectra"], 31)

    def test_msfinder_on_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = msfinder_runner.run(DEMO_MSP, tmp)
        self.assertEqual(manifest["status_counts"].get("unsupported_adduct"), 3)
        self.assertGreater(manifest["status_counts"].get("ok", 0), 20)


if __name__ == "__main__":
    unittest.main()
