#!/usr/bin/env python3
"""Render the MS-Emblator icon files from the SVG masters in src/msemblator/gui/assets.

    python -m pip install resvg_py          # SVG renderer, only needed to rebuild the icons
    python scripts/build_icons.py

The SVG files are the source of the logo; edit them, then run this script.
Windows icons and Tk need bitmaps, so every size is rendered from the vector
drawing (no scaling of a bitmap):

    msemblator_icon_small.svg   16-24 px   frame and four peaks
    msemblator_icon.svg         32-96 px   frame, spectrum and molecule, no text
    msemblator_logo.svg         128-256 px the full logo with "MS-Emblator"

Outputs: msemblator.ico (all sizes), msemblator.png (256 px) and
msemblator_128.png (About dialog). The text of msemblator_logo.svg is rendered
with the system font Segoe UI (Arial elsewhere).
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

ASSETS = Path(__file__).resolve().parents[1] / "src" / "msemblator" / "gui" / "assets"
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)


def source_for(size: int) -> Path:
    if size <= 24:
        return ASSETS / "msemblator_icon_small.svg"
    if size <= 96:
        return ASSETS / "msemblator_icon.svg"
    return ASSETS / "msemblator_logo.svg"


def render(svg: Path, size: int) -> Image.Image:
    import resvg_py

    data = resvg_py.svg_to_bytes(svg_path=str(svg), width=size, height=size)
    return Image.open(io.BytesIO(bytes(data))).convert("RGBA")


def main() -> int:
    images = {size: render(source_for(size), size) for size in ICO_SIZES}
    largest = images[max(ICO_SIZES)]
    largest.save(ASSETS / "msemblator.ico", sizes=[(s, s) for s in ICO_SIZES],
                 append_images=[images[s] for s in ICO_SIZES if s != max(ICO_SIZES)])
    largest.save(ASSETS / "msemblator.png", optimize=True)
    images[128].save(ASSETS / "msemblator_128.png", optimize=True)
    for name in ("msemblator.ico", "msemblator.png", "msemblator_128.png"):
        print(f"{ASSETS / name}  {(ASSETS / name).stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
