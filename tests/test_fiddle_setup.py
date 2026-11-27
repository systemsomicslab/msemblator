"""Tests for msemblator.fiddle_setup (no conda or network needed).

Run from the repository root:
    python -m unittest discover -s tests -t .
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msemblator import fiddle_setup as fs  # noqa: E402


class FiddleSetupTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        patcher = mock.patch.object(fs.Path, "home", return_value=self.home)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_version_and_weights(self):
        self.assertEqual(fs.msfiddle_version(), "2.3.0")
        self.assertEqual(fs.weights_dir(), self.home / ".msfiddle" / "check_point" / "v2.3.0")
        self.assertEqual(len(fs.missing_weights()), 6)
        fs.weights_dir().mkdir(parents=True)
        for name in fs.WEIGHT_FILES:
            (fs.weights_dir() / name).write_bytes(b"w")
        self.assertEqual(fs.missing_weights(), [])

    def test_status_follows_the_parameter_file(self):
        with mock.patch.object(fs, "fiddle_settings", return_value={"conda_env": None, "python": None}), \
                mock.patch.object(fs, "installed_here", return_value=True):
            self.assertEqual(fs.status().where, fs.HERE)
        with mock.patch.object(fs, "fiddle_settings", return_value={"python": "C:/x/python.exe"}):
            self.assertEqual(fs.status().where, fs.OTHER_PYTHON)
        with mock.patch.object(fs, "fiddle_settings", return_value={"conda_env": "old-fiddle"}), \
                mock.patch.object(fs, "find_conda", return_value=None):
            current = fs.status()
        self.assertEqual((current.where, current.env_name, current.environment_ready),
                         (fs.OTHER_CONDA, "old-fiddle", False))

    def test_steps_in_the_msemblator_environment(self):
        missing = fs.FiddleStatus(fs.HERE, installed=False, weights_missing=["w.pt"])
        steps = fs.setup_steps(missing)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0].command[1:5], ["-m", "pip", "install", "-r"])
        self.assertEqual(steps[0].command[5], str(fs.EXTRAS_FILE))
        self.assertEqual(steps[1].command[-3:], ["-m", "msfiddle", "download"])
        ready = fs.FiddleStatus(fs.HERE, installed=True)
        self.assertEqual([s.title for s in fs.setup_steps(ready)], ["Check that msfiddle and torch can be imported"])

    def test_steps_for_another_environment(self):
        python = self.home / "python.exe"
        python.write_bytes(b"")
        steps = fs.setup_steps(fs.FiddleStatus(fs.OTHER_PYTHON, python=str(python), weights_missing=["w.pt"]))
        self.assertEqual(steps[0].command[:4], [str(python), "-m", "msfiddle", "download"])
        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            fs.setup_steps(fs.FiddleStatus(fs.OTHER_PYTHON, python=str(self.home / "missing.exe")))
        with self.assertRaisesRegex(RuntimeError, "was not found"):
            fs.setup_steps(fs.FiddleStatus(fs.OTHER_CONDA, env_name="old-fiddle", conda="conda.exe"))
        conda_steps = fs.setup_steps(fs.FiddleStatus(fs.OTHER_CONDA, env_name="old-fiddle", conda="conda.exe",
                                                     env_path=Path("C:/envs/old-fiddle")))
        self.assertEqual(conda_steps[0].command[:6], ["conda.exe", "run", "--no-capture-output", "-n", "old-fiddle",
                                                      "python"])

    def test_console_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "python.exe").write_bytes(b"")
            with mock.patch.object(fs.sys, "executable", str(Path(tmp) / "pythonw.exe")):
                self.assertEqual(fs.console_python(), str(Path(tmp) / "python.exe"))

    def test_run_steps(self):
        lines = []
        ok_step = fs.Step("print", [sys.executable, "-c", "print('hello')"])
        bad_step = fs.Step("fail", [sys.executable, "-c", "import sys; sys.exit(3)"])
        self.assertEqual(fs.run_steps([ok_step], lambda text, is_step: lines.append(text)), (True, ""))
        self.assertIn("hello", lines)
        self.assertEqual(fs.run_steps([ok_step, bad_step, ok_step]), (False, "fail"))
        cancel = threading.Event()
        cancel.set()
        self.assertEqual(fs.run_steps([ok_step], cancel=cancel), (False, "print"))


class LauncherTest(unittest.TestCase):
    """scripts/msemblator_env.py (the launcher behind msemblator_gui.cmd / msemblator.cmd)."""

    def setUp(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("msemblator_env", ROOT / "scripts" / "msemblator_env.py")
        self.launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.launcher)

    def test_names_fingerprint_and_clean_environment(self):
        launcher = self.launcher
        self.assertEqual(launcher.CondaRuntime("conda.exe").name, "msemblator")
        with mock.patch.dict("os.environ", {"MSEMBLATOR_ENV_NAME": "other", "MSEMBLATOR_VENV_DIR": "D:/venv"}):
            self.assertEqual(launcher.CondaRuntime("conda.exe").name, "other")
            self.assertEqual(launcher.VenvRuntime().folder, Path("D:/venv"))
        self.assertEqual(launcher.VenvRuntime().folder, ROOT / ".venv")
        self.assertEqual(len(launcher.fingerprint()), 64)
        with mock.patch.dict("os.environ", {"PYTHONPATH": "C:/elsewhere", "PYTHONHOME": "C:/x"}):
            env = launcher.clean_environment()
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")

    def test_conda_create_only_when_missing_and_update_only_after_changes(self):
        launcher = self.launcher
        runtime = launcher.CondaRuntime("conda.exe")
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp)
            calls = []
            record = mock.patch.object(launcher, "run", side_effect=lambda c: calls.append(c[1:3]) or 0)
            with mock.patch.object(runtime, "find", side_effect=[None, env]), record:
                runtime.ensure()
            self.assertEqual(calls, [["env", "create"]])
            calls.clear()
            with mock.patch.object(runtime, "find", return_value=env), record:
                runtime.ensure()                                   # nothing changed
                self.assertEqual(calls, [])
                with mock.patch.object(launcher, "fingerprint", return_value="changed"):
                    runtime.ensure()                               # environment files changed
            self.assertEqual(calls, [["env", "update"]])

    def test_venv_create_install_and_update(self):
        launcher = self.launcher
        with tempfile.TemporaryDirectory() as tmp,                 mock.patch.dict("os.environ", {"MSEMBLATOR_VENV_DIR": str(Path(tmp) / "venv")}):
            runtime = launcher.VenvRuntime()
            calls = []

            def fake_run(command):
                calls.append(command)
                if command[1:3] == ["-m", "venv"]:                 # creating the venv makes its python.exe
                    python = runtime._bin("python")
                    python.parent.mkdir(parents=True)
                    python.write_bytes(b"")
                return 0

            with mock.patch.object(launcher, "run", side_effect=fake_run),                     mock.patch.object(launcher.VenvRuntime, "_check_java"):
                runtime.ensure()
                self.assertEqual([c[1:3] for c in calls], [["-m", "venv"], ["-m", "pip"], ["-m", "pip"]])
                self.assertIn(str(launcher.EXTRAS_FILE), calls[-1])
                self.assertIn("-e", calls[-1])
                calls.clear()
                runtime.ensure()                                   # nothing changed
                self.assertEqual(calls, [])
                with mock.patch.object(launcher, "fingerprint", return_value="changed"):
                    runtime.ensure()
                self.assertEqual([c[1:3] for c in calls], [["-m", "pip"], ["-m", "pip"]])
            self.assertEqual(runtime.python("-m", "x")[0], str(runtime._bin("python")))

    def test_venv_needs_python_312_or_older(self):
        launcher = self.launcher
        with mock.patch.object(launcher, "VENV_PYTHONS", ()), self.assertRaisesRegex(SystemExit, "Python 3.12"):
            launcher.VenvRuntime().ensure()

    def test_arguments(self):
        launcher = self.launcher
        self.assertEqual(launcher.main(["--nothing"]), 2)
        with mock.patch.object(launcher, "status", return_value=0) as status:
            self.assertEqual(launcher.main(["--venv", "status"]), 0)
            self.assertIsInstance(status.call_args.args[0], launcher.VenvRuntime)
            self.assertEqual(launcher.main(["--conda", "C:/conda.exe", "status"]), 0)
            self.assertEqual(status.call_args.args[0].conda, "C:/conda.exe")


if __name__ == "__main__":
    unittest.main()
