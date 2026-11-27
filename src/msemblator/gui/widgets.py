"""Drawing widgets of the GUI: MS/MS spectrum, GLACIER mirror plot, tool chooser."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from msemblator.gui import backend

ACCENT = "#1f5fa8"
PREDICTED = "#d9822b"
UNMATCHED = "#b8b8b8"
OK_COLOR = "#2e7d32"
NG_COLOR = "#c62828"
MUTED = "#6b6b6b"
TOOL_LABELS = {"msfinder": "MS-FINDER", "sirius": "SIRIUS", "msbuddy": "msbuddy", "fiddle": "FIDDLE",
               "metfrag": "MetFrag", "glacier": "GLACIER"}
SCORE_LABELS = {"cosine": "cosine", "reverse_cosine": "reverse cosine", "entropy": "entropy"}
SMALL = ("Segoe UI", 8)


def _axis_range(*mz_lists, precursor=None):
    values = [v for mzs in mz_lists for v in mzs]
    if precursor:
        values.append(precursor)
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.05, 5.0)
    return max(lo - pad, 0.0), hi + pad


def _x_ticks(canvas, x_of, lo, hi, y, upward=False):
    for i in range(6):
        mz = lo + (hi - lo) * i / 5
        x = x_of(mz)
        canvas.create_line(x, y, x, y + 4, fill="#888888")
        canvas.create_text(x, y + 6, anchor="n", text=f"{mz:.0f}", fill=MUTED, font=SMALL)


def _label_peaks(canvas, peaks, x_of, y_of, anchor, limit=6, min_share=0.05):
    """Label the largest peaks (at least min_share of the base peak), keeping labels apart."""
    top = max((i for _, i in peaks), default=0) or 1.0
    labelled = []
    for mz, intensity in sorted(peaks, key=lambda p: p[1], reverse=True):
        if intensity / top < min_share or len(labelled) >= limit:
            break
        x = x_of(mz)
        if any(abs(x - other) < 46 for other in labelled):
            continue
        labelled.append(x)
        canvas.create_text(x, y_of(intensity / top), anchor=anchor, text=f"{mz:.4f}", fill="#333333", font=SMALL)


class SpectrumView(tk.Canvas):
    """Stick plot of an MS/MS spectrum, relative intensity in %."""

    def __init__(self, master, height=210):
        super().__init__(master, background="white", highlightthickness=0, height=height)
        self.spectrum = None
        self.bind("<Configure>", lambda _event: self.redraw())

    def show(self, spectrum):
        self.spectrum = spectrum
        self.redraw()

    def redraw(self):
        self.delete("all")
        width, height = self.winfo_width(), self.winfo_height()
        spectrum = self.spectrum
        if not spectrum or not spectrum.mz:
            self.create_text(width / 2, height / 2, text="Select a spectrum", fill=MUTED)
            return
        left, right, top, bottom = 48, 14, 40, 28
        plot_w, plot_h = max(width - left - right, 10), max(height - top - bottom, 10)
        lo, hi = _axis_range(spectrum.mz, precursor=spectrum.precursor_mz)
        top_intensity = max(spectrum.intensity) or 1.0

        def x_of(mz):
            return left + (mz - lo) / (hi - lo) * plot_w

        base = top + plot_h
        self.create_text(left, 6, anchor="nw", fill="#333333", font=("Segoe UI", 9, "bold"),
                         text=f"{spectrum.name}   {spectrum.adduct}")
        self.create_line(left, top, left, base, left + plot_w, base, fill="#888888")
        for percent in (0, 50, 100):
            y = base - percent / 100 * plot_h
            self.create_line(left - 4, y, left, y, fill="#888888")
            self.create_text(left - 6, y, anchor="e", text=str(percent), fill=MUTED, font=SMALL)
        _x_ticks(self, x_of, lo, hi, base)
        if spectrum.precursor_mz:
            x = x_of(spectrum.precursor_mz)
            self.create_line(x, top, x, base, fill=NG_COLOR, dash=(3, 3))
            self.create_text(left + plot_w, 6, anchor="ne", fill=NG_COLOR, font=SMALL,
                             text=f"- - precursor m/z {spectrum.precursor_mz:.4f}")
        peaks = list(zip(spectrum.mz, spectrum.intensity))
        for mz, intensity in peaks:
            self.create_line(x_of(mz), base, x_of(mz), base - intensity / top_intensity * plot_h, fill=ACCENT, width=2)
        _label_peaks(self, peaks, x_of, lambda share: base - share * plot_h - 2, "s")


class MirrorPlot(tk.Canvas):
    """Measured spectrum (up) against a GLACIER predicted spectrum (down).

    Peaks with a partner within the matching tolerance are coloured; the others are grey.
    """

    def __init__(self, master, ppm=20.0):
        super().__init__(master, background="white", highlightthickness=0, height=230)
        self.ppm = ppm
        self.measured = None
        self.predicted = None
        self.message = "Select a structure candidate"
        self.bind("<Configure>", lambda _event: self.redraw())

    def show(self, measured, predicted=None, message=""):
        self.measured, self.predicted, self.message = measured, predicted, message
        self.redraw()

    def redraw(self):
        self.delete("all")
        width, height = self.winfo_width(), self.winfo_height()
        measured, predicted = self.measured, self.predicted
        if not measured or not measured.mz or not predicted or not predicted["mz"]:
            self.create_text(width / 2, height / 2, text=self.message, fill=MUTED, width=max(width - 40, 100),
                             justify="center")
            return
        left, right, top, bottom = 48, 14, 50, 30
        plot_w = max(width - left - right, 10)
        half = max((height - top - bottom) / 2, 10)
        middle = top + half
        lo, hi = _axis_range(measured.mz, predicted["mz"], precursor=measured.precursor_mz)

        def x_of(mz):
            return left + (mz - lo) / (hi - lo) * plot_w

        energy = predicted.get("collision_energy")
        scores = ", ".join(f"{SCORE_LABELS.get(k, k)} {v:.3f}" for k, v in predicted["scores"].items())
        self.create_text(left, 4, anchor="nw", font=("Segoe UI", 9, "bold"), fill="#333333",
                         text=f"Measured (top)  vs  GLACIER prediction{f' at {energy:g} eV' if energy else ''} (bottom)")
        self.create_text(left, 19, anchor="nw", font=SMALL, fill=MUTED,
                         text=f"{scores}     (coloured: matched within {self.ppm:g} ppm)")
        self.create_line(left, top, left, top + 2 * half, fill="#888888")
        self.create_line(left, middle, left + plot_w, middle, fill="#888888")
        for share, label in ((1, "100"), (0, "0"), (-1, "100")):
            y = middle - share * half
            self.create_line(left - 4, y, left, y, fill="#888888")
            self.create_text(left - 6, y, anchor="e", text=label, fill=MUTED, font=SMALL)
        _x_ticks(self, x_of, lo, hi, top + 2 * half)
        if measured.precursor_mz:
            x = x_of(measured.precursor_mz)
            self.create_line(x, top, x, top + 2 * half, fill=NG_COLOR, dash=(3, 3))

        hit_measured, hit_predicted = backend.matched_peaks(measured.mz, predicted["mz"], self.ppm)
        top_measured = max(measured.intensity) or 1.0
        top_predicted = max(predicted["intensity"]) or 1.0
        for mz, intensity, hit in zip(measured.mz, measured.intensity, hit_measured):
            self.create_line(x_of(mz), middle, x_of(mz), middle - intensity / top_measured * half,
                             fill=ACCENT if hit else UNMATCHED, width=2)
        for mz, intensity, hit in zip(predicted["mz"], predicted["intensity"], hit_predicted):
            self.create_line(x_of(mz), middle, x_of(mz), middle + intensity / top_predicted * half,
                             fill=PREDICTED if hit else UNMATCHED, width=2)
        _label_peaks(self, list(zip(measured.mz, measured.intensity)), x_of,
                     lambda share: middle - share * half - 2, "s", limit=4)
        _label_peaks(self, list(zip(predicted["mz"], predicted["intensity"])), x_of,
                     lambda share: middle + share * half + 2, "n", limit=4)


class ToolChooser(ttk.LabelFrame):
    """Radio buttons for the tool combinations of one step, with a ready / not ready mark."""

    def __init__(self, master, title, variable, on_details, style=None):
        super().__init__(master, text=title, padding=(10, 6), **({"style": style} if style else {}))
        self.variable = variable
        self.on_details = on_details
        self.rows = []

    def refresh(self, statuses):
        for child in self.winfo_children():
            child.destroy()
        self.rows = []
        self.columnconfigure(1, weight=1)
        for row, status in enumerate(statuses):
            selection = status.selection
            tools = ", ".join(TOOL_LABELS[t] for t in selection.tools)
            radio = ttk.Radiobutton(self, text=tools, value=selection.name, variable=self.variable)
            radio.grid(row=row, column=0, sticky="w", pady=1)
            note = ttk.Label(self, text="default" if status.is_default else "", foreground=MUTED)
            note.grid(row=row, column=1, sticky="w", padx=(6, 6))
            mark = ttk.Label(self, text="✓ Ready" if status.ready else "✗ Not ready",
                             foreground=OK_COLOR if status.ready else NG_COLOR, cursor="hand2")
            mark.grid(row=row, column=2, sticky="e")
            mark.bind("<Button-1>", lambda _e, s=status: self.on_details(s))
            self.rows.append((radio, note, mark))

    def set_enabled(self, enabled):
        state = "!disabled" if enabled else "disabled"
        for radio, note, mark in self.rows:
            for widget in (radio, note, mark):
                widget.state([state])
