"""Tests for the GUI prototype: the helpers in msemblator.gui.backend, and one run
through the main window with a stand-in for the msemblator command.

Run from the repository root:
    python -m unittest discover -s tests -t .
"""
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msemblator import toolsets  # noqa: E402
from msemblator.gui import backend  # noqa: E402

DEMO_MSP = ROOT / "tests" / "fixtures" / "sirius_free_demo.msp"
PARAMETERS = """formula_prediction:
  msfinder:
    MS1_ppm: 10
    halogen: True #True or False

  fiddle:
    conda_env: msemblator-fiddle
    python: null   # or the FIDDLE environment's python.exe
  msemblator_output_records: 100

structure_prediction:
  msfinder:
    MS1_ppm: 5
"""


class BackendTest(unittest.TestCase):
    def test_read_spectra(self):
        spectra = backend.read_spectra(DEMO_MSP)
        self.assertEqual(len(spectra), 31)
        self.assertEqual(spectra[0].index, 1)
        self.assertTrue(spectra[0].mz)

    def test_command_keeps_the_password_out_of_argv(self):
        request = backend.RunRequest("in put.msp", "out", 2, "fiddle", "glacier", sirius_user="me@example.org")
        command = backend.build_command(request, python="python")
        self.assertEqual(command[:4], ["python", "-u", "-m", "msemblator.cli.msemblator"])
        self.assertIn("--formula_tools", command)
        self.assertNotIn("secret", " ".join(command))
        env = backend.run_environment(request, "secret", base={})
        self.assertEqual((env["SIRIUS_USER"], env["SIRIUS_PASS"]), ("me@example.org", "secret"))
        self.assertIn(str(ROOT / "src"), env["PYTHONPATH"])
        self.assertIn('"in put.msp"', backend.display_command(request))
        # Tools of a step that the mode does not run are left out.
        only_structure = backend.cli_arguments(backend.RunRequest("a.msp", "o", 3, "fiddle", "glacier"))
        self.assertNotIn("--formula_tools", only_structure)

    def test_step_and_decoding(self):
        self.assertEqual(backend.step_of("MetFrag processing start"), "MetFrag")
        self.assertIsNone(backend.step_of("MetFrag processing complete"))
        self.assertEqual(backend.step_of("Running formula elucidation: fiddle (...)"), "Formula elucidation")
        self.assertEqual(backend.decode_line("警告\r\n".encode("cp932")), "警告")
        self.assertEqual(backend.decode_line("ok\n".encode("utf-8")), "ok")

    def test_parameters_keep_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "params.yaml"
            path.write_text(PARAMETERS, encoding="utf-8")
            changed = backend.update_parameters({
                ("formula_prediction", "msfinder", "MS1_ppm"): 5,
                ("formula_prediction", "msfinder", "halogen"): False,
                ("formula_prediction", "fiddle", "python"): "C:/envs/fiddle python/python.exe",
                ("formula_prediction", "msemblator_output_records"): 50,
                ("structure_prediction", "msfinder", "MS1_ppm"): 5,      # unchanged
            }, path)
            self.assertEqual(changed, 4)
            text = path.read_text(encoding="utf-8")
            self.assertIn("    MS1_ppm: 5\n    halogen: False #True or False\n", text)
            self.assertIn("    python: 'C:/envs/fiddle python/python.exe'   # or the FIDDLE", text)
            self.assertIn("  msemblator_output_records: 50\n", text)
            settings = backend.read_parameters(path)
            self.assertEqual(settings["formula_prediction"]["fiddle"]["python"], "C:/envs/fiddle python/python.exe")
            self.assertEqual(settings["structure_prediction"]["msfinder"]["MS1_ppm"], 5)

    def test_parse_entry(self):
        self.assertEqual(backend.parse_entry("20", 10), 20)
        self.assertEqual(backend.parse_entry("0.01", 0.02), 0.01)
        self.assertIsNone(backend.parse_entry("", "x"))
        self.assertEqual(backend.parse_entry("qtof", "orbitrap"), "qtof")

    def test_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            scores = pd.DataFrame({"filename": ["a", "a", "b"], "adduct": ["[M+H]+"] * 3,
                                   "formula": ["C2", "C1", "C3"], "rank": [2, 1, 1],
                                   "rank_score": [0.1, 0.9, 0.5], "softmax_score": [0.2, 0.8, 1.0],
                                   "Used_Tool": ["SIRIUS_Rank:2", "MS-FINDER_Rank:1,SIRIUS_Rank:1", None]})
            scores.to_csv(Path(tmp) / "formula_score.csv", index=False)
            time.sleep(0.05)
            scores.to_csv(Path(tmp) / "formula_score_1.csv", index=False)
            files = backend.latest_results(tmp)
            self.assertEqual(files["formula_score"].name, "formula_score_1.csv")
            rows = backend.candidates_for(backend.load_results(files)["formula_score"], "a", "formula")
            self.assertEqual([r["candidate"] for r in rows], ["C1", "C2"])
            self.assertEqual(rows[0]["tools"], "MS-FINDER #1, SIRIUS #1")
            self.assertEqual(backend.candidates_for(None, "a", "formula"), [])


class SettingsAndDrawingTest(unittest.TestCase):
    def test_settings_round_trip(self):
        from msemblator.gui import settings

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ", {"MSEMBLATOR_GUI_SETTINGS": str(Path(tmp) / "s.json")}):
            self.assertEqual(settings.load_settings()["mode"], 2)          # defaults without a file
            stored = settings.load_settings()
            stored.update(mode=3, output_dir="D:/out", sirius_password="secret")
            stored["recent_inputs"] = settings.add_recent(["b.msp", "a.msp"], "a.msp")
            settings.save_settings(stored)
            again = settings.load_settings()
            self.assertEqual((again["mode"], again["output_dir"]), (3, "D:/out"))
            self.assertEqual(again["recent_inputs"], ["a.msp", "b.msp"])
            self.assertNotIn("secret", (Path(tmp) / "s.json").read_text(encoding="utf-8"))

    def test_structure_drawing(self):
        png = backend.structure_png("CCO")
        self.assertTrue(png.startswith(b"\x89PNG"))
        self.assertIsNone(backend.structure_png("not a smiles"))
        self.assertEqual(backend.short_inchikey("CCO"), "LFQSCWFLJHTTHZ")

    def test_glacier_spectra_and_peak_matching(self):
        table = pd.DataFrame({
            "filename": ["s1", "s1", "s1"], "adduct": ["[M-H]-"] * 3,
            "Short_InChIKey": ["LFQSCWFLJHTTHZ"] * 3, "SMILES": ["CCO"] * 3,
            "collision_energy": [10.0, 40.0, 40.0], "entropy": [0.5, 0.7, 0.2],
            "mz": ["45.03;27.0", "45.03", "17.0"], "intensity": ["1;0.2", "1", "1"],
        })
        found = backend.glacier_spectra_for(table, "s1", "OCC")           # same molecule, other SMILES
        self.assertEqual([p["collision_energy"] for p in found], [10.0, 40.0])
        self.assertEqual(found[1]["scores"]["entropy"], 0.7)               # best spectrum at 40 eV
        self.assertEqual(found[0]["mz"], [45.03, 27.0])
        self.assertEqual(backend.glacier_spectra_for(table, "s2", "CCO"), [])
        hit_a, hit_b = backend.matched_peaks([100.0, 200.0], [100.0015, 150.0], ppm=20)
        self.assertEqual([list(hit_a), list(hit_b)], [[True, False], [True, False]])


@unittest.skipUnless(sys.platform == "win32", "Windows shortcuts")
class ShortcutTest(unittest.TestCase):
    def test_shortcut_starts_the_gui_with_the_icon(self):
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            created = backend.create_shortcuts([tmp])
            self.assertEqual(created, [str(Path(tmp) / "MS-Emblator.lnk")])
            read = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{created[0]}'); "
                 "$s.TargetPath; $s.IconLocation; $s.WorkingDirectory"],
                capture_output=True, text=True)
        target, icon, folder = read.stdout.strip().splitlines()
        self.assertEqual(Path(target), ROOT / "msemblator_gui.cmd")
        self.assertTrue(icon.endswith(r"src\msemblator\gui\assets\msemblator.ico,0"), icon)
        self.assertEqual(Path(folder), ROOT)

    def test_icon_files_are_in_the_package(self):
        from PIL import Image

        from msemblator.gui import app

        for path in (app.ICON_ICO, app.ICON_PNG, app.ICON_128):
            self.assertTrue(path.is_file(), path)
        for name in ("msemblator_logo.svg", "msemblator_icon.svg", "msemblator_icon_small.svg"):
            self.assertTrue((app.ASSETS / name).is_file(), name)     # the vector masters
        with Image.open(app.ICON_ICO) as ico:
            self.assertEqual(sorted(ico.ico.sizes()),
                             [(s, s) for s in (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)])
        with Image.open(app.ICON_128) as about:
            self.assertEqual(about.size, (128, 128))


class MainWindowTest(unittest.TestCase):
    """Open the window, run a fake analysis and check that its results are shown."""

    def setUp(self):
        try:
            import tkinter as tk
            self.root = tk.Tk()
        except Exception as exc:  # no display
            self.skipTest(f"Tk is not available: {exc}")
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        ready = {step: [toolsets.SelectionStatus(s, s.name == "sirius", [])
                        for s in toolsets.all_selections(step)] for step in toolsets.STEPS}
        patches = [mock.patch.object(toolsets, "available_selections", return_value=ready)]
        for name in ("showinfo", "showwarning", "showerror"):
            patches.append(mock.patch(f"tkinter.messagebox.{name}"))
        patches.append(mock.patch("tkinter.messagebox.askyesno", return_value=True))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_run_shows_log_and_results(self):
        from msemblator.gui import app as gui

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fake = Path(tmp.name) / "fake_msemblator.py"
        fake.write_text(
            "import sys, pathlib\n"
            "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1]); out.mkdir(parents=True, exist_ok=True)\n"
            "print('Running formula elucidation: fiddle'); print('FIDDLE processing start')\n"
            "(out / 'formula_score.csv').write_text('filename,adduct,formula,rank_score,Used_Tool,rank,softmax_score\\n'\n"
            "    'EMBL_MCF_2_0_HRMS_Library000529,[M-H]-,C4H4N2O2,2.0,FIDDLE_Rank:1,1,0.9\\n')\n"
            "(out / 'formula_summary.csv').write_text('filename,adduct,Top_score_formula\\n'\n"
            "    'EMBL_MCF_2_0_HRMS_Library000529,[M-H]-,C4H4N2O2\\n')\n",
            encoding="utf-8")
        window = gui.App(self.root, remember=False)
        window.load_msp(str(DEMO_MSP))
        window.output_dir.set(str(Path(tmp.name) / "out"))
        window.mode.set(1)
        window.formula_choice.set("fiddle")
        with mock.patch.object(backend, "build_command", return_value=[sys.executable, str(fake), "--output",
                                                                        window.output_dir.get()]):
            window.run()
            deadline = time.time() + 20
            while window.process is not None and time.time() < deadline:
                self.root.update()
                time.sleep(0.05)
        self.assertIsNone(window.process)
        log = window.log.get("1.0", "end")
        self.assertIn("--formula_tools fiddle", log)
        self.assertIn("FIDDLE processing start", log)
        self.assertEqual(window.step_text.get(), "Finished")
        window.spectra_tree.selection_set("1")
        self.root.update()
        rows = [window.formula_table.item(i, "values") for i in window.formula_table.get_children()]
        self.assertEqual(rows[0][1], "C4H4N2O2")
        self.assertEqual(rows[0][4], "FIDDLE #1")
        self.assertEqual(window.spectra_tree.item("1", "values")[5], "C4H4N2O2")

    def test_download_worker_reports_progress_and_failures(self):
        import threading

        from msemblator import downloads
        from msemblator.gui.dialogs import SetupDialog

        dialog = SetupDialog(self.root, {}, lambda: {})
        self.addCleanup(lambda: dialog.winfo_exists() and dialog.destroy())
        assets = [downloads.Asset("a", "Item A", "tools", "https://x/a", "file", "external/a"),
                  downloads.Asset("b", "Item B", "tools", "https://x/b", "file", "external/b")]

        def fake_install(asset, progress, cancel):
            progress(5, 10, "a.bin")
            if asset.id == "b":
                raise downloads.DownloadError("checksum differs")

        with mock.patch.object(downloads, "install", side_effect=fake_install):
            dialog._download_worker(assets, threading.Event())
        messages = []
        while not dialog.messages.empty():
            messages.append(dialog.messages.get())
        self.assertEqual(messages[-1], ("done", ["Item B: checksum differs"]))
        self.assertIn(("progress", "[1/2] Item A", 5, 10, "a.bin"), messages)

    def test_drop_files_and_folders(self):
        from msemblator.gui import app as gui

        window = gui.App(self.root, remember=False)
        window.handle_dropped([str(DEMO_MSP)])
        self.assertEqual(len(window.spectra), 31)
        with tempfile.TemporaryDirectory() as tmp:
            window.handle_dropped([tmp])                       # empty folder -> output folder
            self.assertEqual(window.output_dir.get(), tmp)
            pd.DataFrame({"filename": [window.spectra[0].name], "adduct": ["[M-H]-"], "Canonical_SMILES": ["CCO"],
                          "rank": [1], "rank_score": [1.0], "softmax_score": [1.0], "Used_Tool": ["GLACIER_Rank:1"]}
                         ).to_csv(Path(tmp) / "structure_score.csv", index=False)
            pd.DataFrame({"filename": [window.spectra[0].name], "adduct": ["[M-H]-"],
                          "Short_InChIKey": ["LFQSCWFLJHTTHZ"], "SMILES": ["CCO"], "collision_energy": [40.0],
                          "entropy": [0.8], "mz": ["45.03"], "intensity": ["1"]}
                         ).to_csv(Path(tmp) / "glacier_spectra.csv.gz", index=False, compression="gzip")
            window.handle_dropped([tmp])                       # folder with results -> results are loaded
            self.root.update()
        self.assertIn("glacier_spectra", window.results)
        window.spectra_tree.selection_set("1")
        self.root.update()
        self.assertEqual(window.energy_box.cget("values"), ("40 eV",))
        self.assertIsNotNone(window.structure_image)
        self.assertEqual(window.mirror.predicted["scores"]["entropy"], 0.8)


if __name__ == "__main__":
    unittest.main()
