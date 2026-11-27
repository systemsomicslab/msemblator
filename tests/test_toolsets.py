"""Tests for the annotation tool selection (msemblator.toolsets) and its use in the CLI.

Run from the repository root:
    python -m unittest discover -s tests -t .
"""
import io
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msemblator import paths, toolsets  # noqa: E402
from msemblator.scoring import calculating_score, struc_score_calc  # noqa: E402

FORMULA, STRUCTURE = toolsets.FORMULA, toolsets.STRUCTURE


class ResolveSelectionTest(unittest.TestCase):
    def test_default_is_the_current_workflow(self):
        formula = toolsets.resolve_selection(FORMULA)
        structure = toolsets.resolve_selection(STRUCTURE)
        self.assertEqual(set(formula.tools), {"sirius", "msfinder", "msbuddy"})
        self.assertEqual(set(structure.tools), {"msfinder", "sirius", "metfrag"})
        self.assertEqual(formula.model_dir, paths.FORMULA_MODEL_DIR)
        self.assertEqual(structure.model_dir, paths.STRUCTURE_MODEL_DIR)

    def test_preset_names(self):
        self.assertEqual(toolsets.resolve_selection(FORMULA, "fiddle").tools, ("msfinder", "msbuddy", "fiddle"))
        self.assertEqual(toolsets.resolve_selection(FORMULA, "all").tools, ("msfinder", "sirius", "msbuddy", "fiddle"))
        self.assertEqual(toolsets.resolve_selection(STRUCTURE, "glacier").tools, ("msfinder", "metfrag", "glacier"))
        self.assertEqual(toolsets.resolve_selection(STRUCTURE, "ALL").tools,
                         ("msfinder", "sirius", "metfrag", "glacier"))

    def test_tool_lists_in_any_order_and_spelling(self):
        for spec in ("msfinder,msbuddy,fiddle", "FIDDLE, MS-FINDER, msbuddy", "fiddle ms_finder MSBuddy"):
            with self.subTest(spec=spec):
                selection = toolsets.resolve_selection(FORMULA, spec)
                self.assertEqual(selection.name, "fiddle")
        self.assertEqual(toolsets.resolve_selection(STRUCTURE, "MetFrag,GLACIER,MS-FINDER").name, "glacier")

    def test_unknown_tool_and_unregistered_combination(self):
        with self.assertRaisesRegex(ValueError, "Unknown formula tool"):
            toolsets.resolve_selection(FORMULA, "msfinder,glacier")
        with self.assertRaisesRegex(ValueError, "No formula model is registered"):
            toolsets.resolve_selection(FORMULA, "msfinder,fiddle")

    def test_model_dirs_of_new_combinations(self):
        fiddle = toolsets.resolve_selection(FORMULA, "fiddle")
        glacier = toolsets.resolve_selection(STRUCTURE, "glacier")
        self.assertEqual(fiddle.model_dir, paths.FORMULA_MODEL_DIR / "msfinder_msbuddy_fiddle")
        self.assertEqual(glacier.model_dir, paths.STRUCTURE_MODEL_DIR / "msfinder_glacier_metfrag")

    def test_config_file_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toolsets.yaml"
            path.write_text(
                "formula:\n  default: mine\n  combinations:\n"
                "    mine: {tools: [msfinder, fiddle], model_dir: 'C:/models/mine'}\n", encoding="utf-8")
            settings = toolsets.load_toolsets(path)
            selection = toolsets.resolve_selection(FORMULA, None, settings)
            self.assertEqual(selection.tools, ("msfinder", "fiddle"))
            self.assertEqual(selection.model_dir, Path("C:/models/mine"))
            # The structure block falls back to the built-in combinations.
            self.assertEqual(toolsets.resolve_selection(STRUCTURE, None, settings).name, "sirius")
            path.write_text("formula:\n  default: x\n  combinations:\n    x: {tools: [msfinder, nope]}\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown tools"):
                toolsets.load_toolsets(path)


class AssetCheckTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.dirs = {
            "PARAMETER_FILE": root / "config" / "msemblator_parameter_file.yaml",
            "CONFIG_DIR": root / "config",
            "SIRIUS_DIR": root / "external" / "sirius",
            "MSFINDER_DIR": root / "external" / "msfinder",
            "MSFINDER_CONFIG_DIR": root / "config" / "msfinder",
            "METFRAG_DIR": root / "external" / "metfrag",
            "METFRAG_CONFIG_DIR": root / "config" / "metfrag",
            "LIBRARY_DIR": root / "library",
            "FORMULA_MODEL_DIR": root / "models" / "formula",
            "STRUCTURE_MODEL_DIR": root / "models" / "structure",
        }
        patcher = mock.patch.multiple(paths, **self.dirs)
        patcher.start()
        self.addCleanup(patcher.stop)
        # FIDDLE environment: a python.exe given in the parameter file.
        self.python = root / "fiddle" / "python.exe"
        self.touch(self.python)
        self.touch(self.dirs["PARAMETER_FILE"])
        self.dirs["PARAMETER_FILE"].write_text(
            f"formula_prediction:\n  fiddle:\n    python: '{self.python.as_posix()}'\n", encoding="utf-8")

    def touch(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def formula_tools_in_place(self):
        d = self.dirs
        self.touch(d["SIRIUS_DIR"] / "sirius.exe")
        self.touch(d["MSFINDER_DIR"] / "MSFINDER ver 3.61" / "MsfinderConsoleApp.exe")
        self.touch(d["MSFINDER_CONFIG_DIR"] / "MsfinderConsoleApp_Param_formula.txt")

    def test_new_combination_needs_its_own_ranker(self):
        self.formula_tools_in_place()
        base = self.dirs["FORMULA_MODEL_DIR"]
        for tool in ("msfinder", "msbuddy", "fiddle"):
            for name in toolsets.TOOLS[FORMULA][tool].pipelines:
                self.touch(base / name)
        self.touch(base / "xgbranker_final_all.pkl")       # ranker of the default combination
        selection = toolsets.resolve_selection(FORMULA, "fiddle")
        missing = toolsets.missing_assets(selection)
        self.assertEqual(len(missing), 1)
        self.assertIn(str(base / "msfinder_msbuddy_fiddle" / "xgbranker_final_all.pkl"), missing[0])
        self.touch(base / "msfinder_msbuddy_fiddle" / "catboostranker_final_all.pkl")
        self.assertEqual(toolsets.missing_assets(selection), [])
        # SIRIUS is not needed for this combination.
        (self.dirs["SIRIUS_DIR"] / "sirius.exe").unlink()
        self.assertEqual(toolsets.missing_assets(selection), [])

    def test_pipelines_in_the_combination_folder_win(self):
        selection = toolsets.resolve_selection(FORMULA, "fiddle")
        self.assertEqual(selection.pipeline_dir("fiddle"), str(self.dirs["FORMULA_MODEL_DIR"]))
        for name in toolsets.TOOLS[FORMULA]["fiddle"].pipelines:
            self.touch(selection.model_dir / name)
        self.assertEqual(selection.pipeline_dir("fiddle"), str(selection.model_dir))

    def test_missing_fiddle_environment_is_reported(self):
        self.python.unlink()
        missing = toolsets.tool_assets_missing(toolsets.resolve_selection(FORMULA, "fiddle"))
        self.assertTrue(any("formula_prediction.fiddle.python" in m for m in missing), missing)

    def test_glacier_needs_settings_and_a_library(self):
        selection = toolsets.resolve_selection(STRUCTURE, "glacier")
        missing = toolsets.tool_assets_missing(selection)
        self.assertTrue(any("glacier.yaml" in m for m in missing), missing)
        library_dir = Path(self.dirs["CONFIG_DIR"]).parent / "glacier_libs"
        (self.dirs["CONFIG_DIR"] / "glacier.yaml").write_text(
            f"glacier:\n  library_dir: '{library_dir.as_posix()}'\n"
            "  libraries: {'[M-H]-': 'minus.msds'}\n  similarities: [cosine]\n", encoding="utf-8")
        self.assertTrue(any("GLACIER libraries" in m for m in toolsets.tool_assets_missing(selection)))
        self.touch(library_dir / "minus.msds")
        self.assertFalse(any("GLACIER" in m or "glacier" in m for m in toolsets.tool_assets_missing(selection)))
        # SIRIUS files are not needed without SIRIUS.
        self.assertFalse(any("sirius" in m.lower() for m in toolsets.missing_assets(selection)))

    def test_validate_runtime_uses_the_selection(self):
        self.formula_tools_in_place()
        fiddle = toolsets.resolve_selection(FORMULA, "fiddle")
        with self.assertRaises(FileNotFoundError) as ctx:
            paths.validate_runtime(1, fiddle)
        self.assertIn("msfinder_msbuddy_fiddle", str(ctx.exception))
        self.assertNotIn("csi_fingerid", str(ctx.exception))


class ModelFeatureTest(unittest.TestCase):
    def test_default_feature_lists_match_the_released_models(self):
        formula = toolsets.ranking_features(toolsets.resolve_selection(FORMULA))
        structure = toolsets.ranking_features(toolsets.resolve_selection(STRUCTURE))
        self.assertEqual(formula[:3], ["normalization_score_msbuddy", "normalization_score_msfinder",
                                       "normalization_score_sirius"])
        self.assertEqual(formula[6:9], ["normalization_rank_msbuddy", "normalization_rank_msfinder",
                                        "normalization_rank_sirius"])
        self.assertEqual(structure[6:9], ["normalized_rank_metfrag", "normalized_rank_msfinder",
                                          "normalized_rank_sirius"])
        self.assertEqual(len(formula), 15)
        self.assertEqual(len(structure), 15)

    def test_model_of_another_combination_is_refused(self):
        fiddle = toolsets.resolve_selection(FORMULA, "fiddle")
        toolsets.check_model_features(["normalization_score_fiddle", "adduct_MplusHplus"], fiddle, "m.pkl")
        with self.assertRaisesRegex(ValueError, "normalization_score_sirius"):
            toolsets.check_model_features(["normalization_score_sirius", "normalization_score_fiddle"],
                                          fiddle, "m.pkl")
        default_structure = toolsets.resolve_selection(STRUCTURE)
        with self.assertRaisesRegex(ValueError, "GLACIER_Entropy_Max"):
            toolsets.check_model_features(["GLACIER_Entropy_Max"], default_structure, "m.pkl")

    def _predict(self, module, predict, model, df, selection):
        with mock.patch.object(module, "find_ranker_model", return_value=Path("xgbranker_final_all.pkl")), \
                mock.patch.object(module.joblib, "load", return_value=model):
            return predict(df, "models", selection)

    @staticmethod
    def _model(*features):
        """A ranker that records its features and scores a row by their sum."""
        return mock.Mock(feature_names_in_=np.array(features),
                         predict=mock.Mock(side_effect=lambda X: X.sum(axis=1).to_numpy()))

    def test_ranker_gets_the_features_named_in_the_model(self):
        df = pd.DataFrame({"normalization_score_fiddle": [0.5, np.nan], "other": [9, 9]})
        out = self._predict(calculating_score, calculating_score.predict_and_append,
                            self._model("normalization_score_fiddle"), df,
                            toolsets.resolve_selection(FORMULA, "fiddle"))
        self.assertEqual(out["rank_score"].tolist(), [0.5, 0.0])      # missing value -> 0

        glacier_df = pd.DataFrame({"GLACIER_Entropy_Max": [0.25], "normalization_score_fiddle": [1.0]})
        out = self._predict(struc_score_calc, struc_score_calc.predict_and_append_rank,
                            self._model("GLACIER_Entropy_Max"), glacier_df,
                            toolsets.resolve_selection(STRUCTURE, "glacier"))
        self.assertEqual(out["rank_score"].tolist(), [0.25])


class GlacierJoinTest(unittest.TestCase):
    def test_glacier_candidates_join_the_other_tools(self):
        from msemblator.summaries.glacier_struc_summary import add_glacier_features, glacier_summary_smiles

        wide = pd.DataFrame({
            "filename": ["1", "1", "2"], "adduct": ["[M-H]-", "[M-H]-", "[M+H]+"],
            "Short_InChIKey": ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC"],
            "SMILES": ["CA", "CB", "CC"], "Used_tools": ["MetFrag_Rank:1", "MS-FINDER_Rank:1", "MetFrag_Rank:1"],
            "normalization_score_metfrag": [0.9, 0.0, 0.8],
        })
        features = pd.DataFrame({
            "query_index": [1, 1], "filename": ["1", "1"], "adduct": ["[M-H]-", "[M-H]-"],
            "Short_InChIKey": ["AAAAAAAAAAAAAA", "DDDDDDDDDDDDDD"], "SMILES": ["CA_g", "CD"],
            "GLACIER_Entropy_Max": [0.7, 0.4], "GLACIER_Entropy_Rank": [1, 2],
            "GLACIER_Candidate_Count": [2, 2], "adduct_MminusHminus": [1, 1],
        })
        name_adduct = pd.DataFrame({"filename": ["1", "2"], "adduct": ["[M-H]-", "[M+H]+"]})
        merged = add_glacier_features(wide, features, "GLACIER_Entropy_Rank", name_adduct).set_index("Short_InChIKey")

        self.assertEqual(len(merged), 4)
        self.assertEqual(merged.loc["AAAAAAAAAAAAAA", "Used_tools"], "MetFrag_Rank:1,GLACIER_Rank:1")
        self.assertEqual(merged.loc["AAAAAAAAAAAAAA", "SMILES"], "CA")          # existing SMILES kept
        self.assertEqual(merged.loc["DDDDDDDDDDDDDD", "SMILES"], "CD")          # GLACIER-only candidate
        self.assertEqual(merged.loc["DDDDDDDDDDDDDD", "adduct"], "[M-H]-")
        self.assertEqual(merged.loc["DDDDDDDDDDDDDD", "adduct_MminusHminus"], 1)
        self.assertEqual(merged.loc["DDDDDDDDDDDDDD", "Used_tools"], "GLACIER_Rank:2")
        # The candidate count describes the query: also set for candidates GLACIER did not score.
        self.assertEqual(merged.loc["BBBBBBBBBBBBBB", "GLACIER_Candidate_Count"], 2)
        self.assertTrue(pd.isna(merged.loc["BBBBBBBBBBBBBB", "GLACIER_Entropy_Max"]))
        self.assertTrue(pd.isna(merged.loc["CCCCCCCCCCCCCC", "GLACIER_Candidate_Count"]))

        summary = glacier_summary_smiles(features, "GLACIER_Entropy_Rank", name_adduct)
        row = summary.set_index("filename").loc["1"]
        self.assertEqual((row["glacier_structure_1"], row["glacier_structure_2"]), ("CA_g", "CD"))

        # A failed similarity leaves the rank empty; it is left out of the summary.
        features["GLACIER_Entropy_Rank"] = pd.array([1, pd.NA], dtype="Int64")
        summary = glacier_summary_smiles(features, "GLACIER_Entropy_Rank", name_adduct)
        self.assertNotIn("glacier_structure_2", summary.columns)
        merged = add_glacier_features(wide, features, "GLACIER_Entropy_Rank", name_adduct).set_index("Short_InChIKey")
        self.assertEqual(merged.loc["DDDDDDDDDDDDDD", "Used_tools"], "")


class CliToolOptionTest(unittest.TestCase):
    def run_cli(self, argv, **patches):
        from msemblator.cli import msemblator as cli

        out = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, "argv", ["msemblator"] + argv))
            stack.enter_context(redirect_stdout(out))
            if patches:
                self.patched = stack.enter_context(mock.patch.multiple(cli, **patches))
            code = cli.main()
        return code, out.getvalue()

    def test_list_tools_needs_no_input(self):
        code, out = self.run_cli(["--list_tools"])
        self.assertEqual(code, 0)
        self.assertIn("fiddle", out)
        self.assertIn("glacier", out)

    def test_unregistered_combination_is_an_error(self):
        code, out = self.run_cli(["--input", "x.msp", "--output", "o", "--mode", "1",
                                  "--formula_tools", "sirius,fiddle"])
        self.assertEqual(code, 2)
        self.assertIn("No formula model is registered", out)

    def test_selection_is_passed_and_sirius_login_skipped(self):
        from msemblator.cli import msemblator as cli

        with tempfile.TemporaryDirectory() as tmp:
            calls = {}

            def formula(msp, out, name_df, failed, selection):
                calls["formula"] = selection.name
                return pd.DataFrame({"filename": [], "formula": []})

            def structure(msp, out, name_df, selection):
                calls["structure"] = selection.name
                return []

            code, _ = self.run_cli(
                ["--input", "in.msp", "--output", str(Path(tmp) / "out"), "--mode", "2",
                 "--formula_tools", "fiddle", "--structure_tools", "glacier", "--sirius_user", "someone"],
                WORK_DIR=tmp, validate_runtime=mock.DEFAULT, ensure_runtime_directories=mock.DEFAULT,
                modify_msp_data_type=mock.DEFAULT, save_updated_msp=mock.DEFAULT, msp_formula_changer=mock.DEFAULT,
                convert_name_to_peakid=mock.Mock(return_value=("", pd.DataFrame())),
                formula_elucidation=formula, structure_elucidation=structure,
                sirius_login=mock.Mock(side_effect=AssertionError("SIRIUS is not selected")),
                getpass=mock.Mock(getpass=mock.Mock(side_effect=AssertionError("no prompt expected"))),
            )
            self.assertEqual(code, 0)
            self.assertEqual(calls, {"formula": "fiddle", "structure": "glacier"})
            mode, formula_tools, structure_tools = self.patched["validate_runtime"].call_args.args
            self.assertEqual((mode, formula_tools.name, structure_tools.name), (2, "fiddle", "glacier"))


if __name__ == "__main__":
    unittest.main()
