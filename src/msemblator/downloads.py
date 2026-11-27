"""Download the runtime assets listed in config/downloads.yaml (programs, models, libraries).

    python -m msemblator.downloads --list
    python -m msemblator.downloads --install sirius models_formula

The GUI uses the same functions (Settings > Tool setup > Downloads). What is
installed, and from which address, is recorded in external/downloads_installed.json,
so a changed address in the manifest (for example a new Zenodo version) shows up
as an available update.

Safety: downloads use the addresses in the manifest only; files are checked with
the SHA-256 in the manifest, the MD5 published by Zenodo, or the zip CRC-32;
archive entries can only be written inside their destination folder; a file is
replaced only after it was downloaded completely.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from msemblator import paths

MANIFEST_FILE = paths.CONFIG_DIR / "downloads.yaml"
STATE_FILE = paths.EXTERNAL_DIR / "downloads_installed.json"
GROUPS = ("tools", "models", "libraries")
TYPES = ("file", "zip", "zip_members")
CHUNK = 1 << 20
USER_AGENT = "msemblator-downloader"

INSTALLED, UPDATE, MISSING, UNPUBLISHED = "installed", "update available", "not installed", "not published"


class DownloadError(RuntimeError):
    pass


class DownloadCancelled(DownloadError):
    pass


@dataclass
class Asset:
    id: str
    title: str
    group: str
    url: str
    type: str
    destination: str
    members: str = ""
    strip: str = ""
    sha256: str = ""
    installed: list = field(default_factory=list)
    size_mb: float | None = None
    license: str = ""
    license_url: str = ""
    note: str = ""

    @property
    def source(self) -> str:
        """What the state file records: an asset changes when its source changes."""
        return f"{self.url}#{self.members}" if self.members else self.url


# --- manifest -------------------------------------------------------------------

def _expand(text, variables, depth=0):
    if not isinstance(text, str) or "{" not in text or depth > 5:
        return text
    expanded = re.sub(r"\{(\w+)\}", lambda m: str(variables.get(m.group(1), m.group(0))), text)
    return _expand(expanded, variables, depth + 1) if expanded != text else expanded


def load_manifest(path=None) -> list[Asset]:
    import yaml

    path = Path(path or MANIFEST_FILE)
    with open(path, encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}
    variables = {k: str(v) for k, v in (settings.get("variables") or {}).items()}
    assets = []
    for asset_id, entry in (settings.get("assets") or {}).items():
        entry = {k: (_expand(v, variables) if not isinstance(v, list) else [_expand(x, variables) for x in v])
                 for k, v in (entry or {}).items()}
        asset = Asset(id=str(asset_id), title=entry.get("title") or str(asset_id), group=entry.get("group", "tools"),
                      url=(entry.get("url") or "").strip(), type=entry.get("type", "file"),
                      destination=entry.get("destination") or "", members=entry.get("members") or "",
                      strip=entry.get("strip") or "", sha256=(entry.get("sha256") or "").lower(),
                      installed=list(entry.get("installed") or []), size_mb=entry.get("size_mb"),
                      license=entry.get("license") or "", license_url=entry.get("license_url") or "",
                      note=entry.get("note") or "")
        if asset.type not in TYPES:
            raise ValueError(f"{path}: {asset_id}: type must be one of {', '.join(TYPES)}")
        if not asset.destination:
            raise ValueError(f"{path}: {asset_id}: destination is missing")
        if asset.type == "zip_members" and not asset.members:
            raise ValueError(f"{path}: {asset_id}: zip_members needs members")
        if asset.url and not asset.url.lower().startswith("https://"):
            raise ValueError(f"{path}: {asset_id}: only https:// addresses are allowed")
        assets.append(asset)
    return assets


# --- state ------------------------------------------------------------------------

def load_state(path=None) -> dict:
    path = Path(path or STATE_FILE)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state, path=None):
    path = Path(path or STATE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def asset_status(asset: Asset, state: dict, root=None) -> str:
    root = Path(root or paths.PROJECT_ROOT)
    present = bool(asset.installed) and all((root / p).exists() for p in asset.installed)
    if present:
        recorded = state.get(asset.id, {}).get("source")
        return UPDATE if (asset.url and recorded and recorded != asset.source) else INSTALLED
    return MISSING if asset.url else UNPUBLISHED


# --- download ---------------------------------------------------------------------

def _open(url, start=None, end=None, timeout=60):
    headers = {"User-Agent": USER_AGENT}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)


class RemoteFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP range requests (8 MiB blocks)."""

    BLOCK = 8 << 20

    def __init__(self, url, cancel=None):
        self.url, self.pos, self.cancel = url, 0, cancel
        with _open(url, 0, 0) as response:
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or "/" not in content_range:
                raise DownloadError(f"The server does not support partial downloads: {url}")
            self.size = int(content_range.rsplit("/", 1)[1])
        self.block_start, self.block = -1, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset)
        return self.pos

    def _fetch(self, start):
        if self.cancel is not None and self.cancel.is_set():
            raise DownloadCancelled("Cancelled")
        end = min(start + self.BLOCK, self.size) - 1
        for attempt in range(3):
            try:
                with _open(self.url, start, end) as response:
                    self.block = response.read()
                self.block_start = start
                return
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            if not (self.block_start <= self.pos < self.block_start + len(self.block)):
                self._fetch(self.pos)
            offset = self.pos - self.block_start
            piece = self.block[offset:offset + n]
            out += piece
            self.pos += len(piece)
            n -= len(piece)
        return bytes(out)


def zenodo_md5(url: str) -> str | None:
    """MD5 that Zenodo publishes for https://zenodo.org/records/<id>/files/<name>."""
    match = re.match(r"https://zenodo\.org/records/(\d+)/files/([^/?#]+)", url)
    if not match:
        return None
    record, name = match.groups()
    try:
        with _open(f"https://zenodo.org/api/records/{record}", timeout=30) as response:
            files = json.load(response).get("files", [])
    except (OSError, ValueError):
        return None
    for entry in files:
        if entry.get("key") == urllib.request.unquote(name) and str(entry.get("checksum", "")).startswith("md5:"):
            return entry["checksum"][4:]
    return None


def _download(url, target: Path, progress, cancel, total_hint=None) -> dict:
    """Stream url to target; returns {'sha256': ..., 'md5': ...} of the bytes received."""
    sha256, md5 = hashlib.sha256(), hashlib.md5()
    with _open(url) as response, open(target, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0) or total_hint
        done = 0
        while True:
            if cancel is not None and cancel.is_set():
                raise DownloadCancelled("Cancelled")
            chunk = response.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
            sha256.update(chunk)
            md5.update(chunk)
            done += len(chunk)
            if progress:
                progress(done, total, target.name)
    return {"sha256": sha256.hexdigest(), "md5": md5.hexdigest()}


def _verify(asset: Asset, digests: dict):
    if asset.sha256:
        if digests["sha256"] != asset.sha256:
            raise DownloadError(f"{asset.title}: SHA-256 does not match the manifest "
                                f"(got {digests['sha256']}, expected {asset.sha256})")
        return "sha256"
    md5 = zenodo_md5(asset.url)
    if md5:
        if digests["md5"] != md5:
            raise DownloadError(f"{asset.title}: MD5 does not match the value published by Zenodo")
        return "zenodo md5"
    return "none"


def _safe_target(destination: Path, name: str) -> Path | None:
    """Path of an archive entry inside destination, or None for folders; refuses to leave it."""
    if not name or name.endswith("/"):
        return None
    target = (destination / name).resolve()
    if target != destination and destination not in target.parents:
        raise DownloadError(f"Archive entry outside the destination folder: {name}")
    return target


def _strip(name: str, prefix: str) -> str:
    return name[len(prefix):] if prefix and name.startswith(prefix) else name


def _extract(archive: zipfile.ZipFile, infos, destination: Path, strip: str, progress, cancel) -> list[str]:
    total = sum(i.file_size for i in infos) or None
    done, written = 0, []
    for info in infos:
        target = _safe_target(destination, _strip(info.filename, strip))
        if target is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        with archive.open(info) as source, open(partial, "wb") as out:   # CRC-32 checked at the end
            while True:
                if cancel is not None and cancel.is_set():
                    raise DownloadCancelled("Cancelled")
                chunk = source.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total, Path(info.filename).name)
        partial.replace(target)
        written.append(str(target))
    return written


def install(asset: Asset, progress=None, cancel=None, root=None, state_path=None) -> list[str]:
    """Download and install one asset; returns the files written."""
    if not asset.url:
        raise DownloadError(f"{asset.title}: no download address yet (config/downloads.yaml)")
    root = Path(root or paths.PROJECT_ROOT).resolve()
    destination = (root / asset.destination).resolve()
    if destination != root and root not in destination.parents:
        raise DownloadError(f"{asset.title}: destination must be inside {root}")

    if asset.type == "file":
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        try:
            check = _verify(asset, _download(asset.url, partial, progress, cancel))
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
        written = [str(destination)]
    elif asset.type == "zip":
        destination.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(suffix=".zip", dir=destination)
        temp = Path(temp_name)
        try:
            os.close(handle)
            check = _verify(asset, _download(asset.url, temp, progress, cancel))
            with zipfile.ZipFile(temp) as archive:
                written = _extract(archive, archive.infolist(), destination, asset.strip, progress, cancel)
        finally:
            temp.unlink(missing_ok=True)
    else:   # zip_members
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(RemoteFile(asset.url, cancel)) as archive:
            prefix = asset.members
            infos = [i for i in archive.infolist()
                     if (i.filename.startswith(prefix) if prefix.endswith("/") else i.filename == prefix)]
            if not [i for i in infos if not i.filename.endswith("/")]:
                raise DownloadError(f"{asset.title}: no entry '{prefix}' in the archive")
            written = _extract(archive, infos, destination, asset.strip, progress, cancel)
        check = "zip crc32"

    state = load_state(state_path)
    state[asset.id] = {"source": asset.source, "title": asset.title, "check": check,
                       "installed_at": datetime.now().isoformat(timespec="seconds"),
                       "files": [str(Path(f).relative_to(root)) for f in written]}
    save_state(state, state_path)
    return written


# --- command line -----------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show every asset and whether it is installed")
    parser.add_argument("--install", nargs="+", metavar="ID", help="download these assets")
    parser.add_argument("--manifest", help="download list (default: config/downloads.yaml)")
    parser.add_argument("--yes", action="store_true", help="accept the licenses without asking")
    args = parser.parse_args(argv)
    assets = {a.id: a for a in load_manifest(args.manifest)}
    state = load_state()
    if args.list or not args.install:
        for asset in assets.values():
            size = f"{asset.size_mb:g} MB" if asset.size_mb else ""
            print(f"{asset.id:<28} {asset_status(asset, state):<17} {size:>9}  {asset.title}")
        return 0
    unknown = [i for i in args.install if i not in assets]
    if unknown:
        print(f"Unknown asset(s): {', '.join(unknown)}")
        return 2
    for asset_id in args.install:
        asset = assets[asset_id]
        print(f"\n{asset.title}\n  from {asset.url}\n  license: {asset.license or '-'} {asset.license_url}")
        if not args.yes and input("  Accept the license and download? [y/N] ").strip().lower() != "y":
            print("  skipped")
            continue

        def progress(done, total, name):
            text = f"{done / 1e6:,.1f} MB" + (f" / {total / 1e6:,.1f} MB" if total else "")
            print(f"\r  {name}: {text}   ", end="", flush=True)

        files = install(asset, progress)
        print(f"\n  installed {len(files)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
