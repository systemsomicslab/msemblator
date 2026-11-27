"""Tests for msemblator.downloads with a fake web server (no network needed).

Run from the repository root:
    python -m unittest discover -s tests -t .
"""
import hashlib
import io
import os
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msemblator import downloads  # noqa: E402


class FakeResponse(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


class FakeServer:
    """Serves bytes by URL and supports Range requests, like GitHub / Zenodo."""

    def __init__(self, files):
        self.files = files
        self.requests = []

    def open(self, url, start=None, end=None, timeout=60):
        self.requests.append((url, start, end))
        if "zenodo.org/api/records" in url:
            raise OSError("no API in tests")
        data = self.files[url]
        if start is None:
            return FakeResponse(data, 200, {"Content-Length": str(len(data))})
        part = data[start:end + 1]
        return FakeResponse(part, 206, {"Content-Range": f"bytes {start}-{start + len(part) - 1}/{len(data)}"})


def make_zip(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


MANIFEST = """
variables:
  record: '42'
  root: 'bundle'
  base: 'https://example.org/{record}'
assets:
  tool:
    title: 'Tool {record}'
    group: tools
    url: '{base}/tool.zip'
    type: zip
    destination: 'external'
    installed: ['external/tool/run.exe']
    license: 'MIT'
  jar:
    url: '{base}/x.jar'
    type: file
    destination: 'external/x.jar'
    installed: ['external/x.jar']
  models:
    url: '{base}/bundle.zip'
    type: zip_members
    members: '{root}/models/'
    strip: '{root}/'
    destination: '.'
    installed: ['models/a.pkl']
  later:
    url: ''
    type: file
    destination: 'external/later.bin'
    installed: ['external/later.bin']
"""


class DownloadsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.manifest = self.root / "downloads.yaml"
        self.manifest.write_text(MANIFEST, encoding="utf-8")
        self.state = self.root / "state.json"
        self.jar = b"jar-bytes" * 1000
        self.server = FakeServer({
            "https://example.org/42/tool.zip": make_zip({"tool/run.exe": b"exe", "tool/lib/a.dll": b"dll"}),
            "https://example.org/42/x.jar": self.jar,
            "https://example.org/42/bundle.zip": make_zip({"bundle/models/a.pkl": b"A" * 50000,
                                                           "bundle/models/b.pkl": b"B",
                                                           "bundle/library/big.db": os.urandom(400000)}),
        })
        patcher = mock.patch.object(downloads, "_open", self.server.open)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.assets = {a.id: a for a in downloads.load_manifest(self.manifest)}

    def install(self, asset_id, **kwargs):
        return downloads.install(self.assets[asset_id], root=self.root, state_path=self.state, **kwargs)

    def test_manifest_variables(self):
        tool = self.assets["tool"]
        self.assertEqual(tool.title, "Tool 42")
        self.assertEqual(tool.url, "https://example.org/42/tool.zip")
        self.assertEqual(self.assets["models"].members, "bundle/models/")
        bad = self.root / "bad.yaml"
        bad.write_text("assets:\n  x: {url: 'http://plain.org/a', type: file, destination: a}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "https"):
            downloads.load_manifest(bad)

    def test_zip_file_and_members(self):
        self.install("tool")
        self.assertEqual((self.root / "external/tool/run.exe").read_bytes(), b"exe")
        self.install("jar")
        self.assertEqual((self.root / "external/x.jar").read_bytes(), self.jar)
        with mock.patch.object(downloads.RemoteFile, "BLOCK", 16384):
            files = self.install("models")
        self.assertEqual(sorted(Path(f).name for f in files), ["a.pkl", "b.pkl"])
        self.assertFalse((self.root / "library").exists())          # only the requested members
        # The large entry that was not asked for is not downloaded (range requests).
        whole = len(self.server.files["https://example.org/42/bundle.zip"])
        fetched = sum(end - start + 1 for url, start, end in self.server.requests
                      if url.endswith("bundle.zip") and start is not None)
        self.assertLess(fetched, whole / 4)

    def test_status_and_updates(self):
        state = downloads.load_state(self.state)
        self.assertEqual(downloads.asset_status(self.assets["jar"], state, self.root), downloads.MISSING)
        self.assertEqual(downloads.asset_status(self.assets["later"], state, self.root), downloads.UNPUBLISHED)
        self.install("jar")
        state = downloads.load_state(self.state)
        self.assertEqual(downloads.asset_status(self.assets["jar"], state, self.root), downloads.INSTALLED)
        # A new address in the manifest (e.g. a new Zenodo version) is offered as an update.
        self.manifest.write_text(MANIFEST.replace("record: '42'", "record: '43'"), encoding="utf-8")
        moved = {a.id: a for a in downloads.load_manifest(self.manifest)}["jar"]
        self.assertEqual(downloads.asset_status(moved, state, self.root), downloads.UPDATE)

    def test_checksum_mismatch_keeps_the_old_file(self):
        target = self.root / "external/x.jar"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"old")
        self.assets["jar"].sha256 = "0" * 64
        with self.assertRaisesRegex(downloads.DownloadError, "SHA-256"):
            self.install("jar")
        self.assertEqual(target.read_bytes(), b"old")
        self.assertFalse(target.with_name("x.jar.part").exists())
        self.assets["jar"].sha256 = hashlib.sha256(self.jar).hexdigest()
        self.install("jar")
        self.assertEqual(target.read_bytes(), self.jar)

    def test_archive_entries_cannot_leave_the_destination(self):
        self.server.files["https://example.org/42/tool.zip"] = make_zip({"../../evil.txt": b"x"})
        with self.assertRaisesRegex(downloads.DownloadError, "outside"):
            self.install("tool")
        self.assertFalse((self.root.parent / "evil.txt").exists())

    def test_cancel(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(downloads.DownloadCancelled):
            self.install("jar", cancel=cancel)
        self.assertFalse((self.root / "external/x.jar").exists())

    def test_unpublished_asset(self):
        with self.assertRaisesRegex(downloads.DownloadError, "no download address"):
            self.install("later")

    def test_repository_manifest_is_valid(self):
        assets = downloads.load_manifest()
        ids = {a.id for a in assets}
        self.assertTrue({"sirius", "msfinder", "metfrag", "models_formula", "models_structure"} <= ids)
        for asset in assets:
            self.assertNotIn("{", asset.url + asset.destination + asset.members, asset.id)


if __name__ == "__main__":
    unittest.main()
