"""MS-Emblator desktop GUI (prototype).

    python -m msemblator.gui          (or msemblator_gui.cmd / msemblator-gui)

Layout follows MS-FINDER: the spectra of the opened MSP on the left with an
MS/MS viewer below, and on the right one tab to set up and run the analysis and
one tab with the ranked formula and structure candidates of the selected
spectrum (structure drawing, and a mirror plot against the GLACIER prediction
when GLACIER was used). The analysis itself is the msemblator command line,
started as a subprocess, so results are the same as from the command line.

Files can be dropped on the window: an MSP file opens it, a folder becomes the
output folder (or opens the results in it). The last files and choices are
remembered (gui/settings.py).
"""
from __future__ import annotations

import base64
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from msemblator import paths, toolsets
from msemblator.gui import backend, settings as gui_settings
from msemblator.gui.dialogs import FiddleSetupDialog, ParametersDialog, SetupDialog
from msemblator.gui.widgets import ACCENT, MUTED, NG_COLOR, TOOL_LABELS, MirrorPlot, SpectrumView, ToolChooser

try:   # drag and drop of files (optional package)
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:   # pragma: no cover - the GUI works without it
    DND_FILES, TkinterDnD = None, None

APP_TITLE = "MS-Emblator"


class App:
    def __init__(self, root, remember=True):
        self.root = root
        self.remember = remember
        self.spectra: list[backend.SpectrumInfo] = []
        self.results: dict = {}
        self.result_files: dict = {}
        self.statuses: dict = {}
        self.process = None
        self.messages: queue.Queue = queue.Queue()
        self.run_started = 0.0
        self.structure_image = None
        self.predictions: list = []
        self.stored = gui_settings.load_settings() if remember else dict(gui_settings.DEFAULTS)

        root.title(APP_TITLE)
        root.geometry(self.stored.get("geometry") or "1320x860")
        root.minsize(1080, 700)
        self._style()

        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=self.stored.get("output_dir", ""))
        self.mode = tk.IntVar(value=self.stored.get("mode", 2) if self.stored.get("mode") in (1, 2, 3) else 2)
        self.formula_choice = tk.StringVar(value=self.stored.get("formula_tools", ""))
        self.structure_choice = tk.StringVar(value=self.stored.get("structure_tools", ""))
        self.sirius_user = tk.StringVar(value=self.stored.get("sirius_user") or os.environ.get("SIRIUS_USER", ""))
        self.sirius_password = tk.StringVar()
        self.status_text = tk.StringVar(value="Open or drop an MSP file to start.")
        self.step_text = tk.StringVar(value="Idle")
        self.energy_choice = tk.StringVar()

        self._menu()
        self._toolbar()
        self._body()
        self._statusbar()
        self._enable_drop()
        self.refresh_tool_status()
        for variable in (self.mode, self.formula_choice, self.structure_choice):
            variable.trace_add("write", lambda *_: self.update_states())
        self.update_states()
        last = self.stored.get("input_path")
        if last and Path(last).is_file():
            root.after(50, lambda: self.load_msp(last, quiet=True))
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll_job = root.after(100, self.poll_messages)
        root.bind("<Destroy>", self._on_destroy, add="+")

    def _on_destroy(self, event):
        if event.widget is self.root and self._poll_job:
            self.root.after_cancel(self._poll_job)
            self._poll_job = None

    # layout -----------------------------------------------------------------------

    def _style(self):
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Step.TLabelframe.Label", foreground=ACCENT, font=("Segoe UI", 10, "bold"))
        style.configure("Run.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 4))
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Treeview", rowheight=22)

    def _menu(self):
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Open MSP file…", accelerator="Ctrl+O", command=self.open_msp)
        self.recent_menu = tk.Menu(file_menu, tearoff=False)
        file_menu.add_cascade(label="Recent MSP files", menu=self.recent_menu)
        file_menu.add_command(label="Choose output folder…", command=self.choose_output)
        file_menu.add_command(label="Load results…", command=self.load_results_dialog)
        file_menu.add_command(label="Open output folder", command=self.open_output)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        menu.add_cascade(label="File", menu=file_menu)
        run_menu = tk.Menu(menu, tearoff=False)
        run_menu.add_command(label="Run", accelerator="F5", command=self.run)
        run_menu.add_command(label="Stop", command=self.stop)
        run_menu.add_command(label="Show command line…", command=self.show_command)
        menu.add_cascade(label="Analysis", menu=run_menu)
        settings_menu = tk.Menu(menu, tearoff=False)
        settings_menu.add_command(label="Analysis parameters…", command=lambda: ParametersDialog(self.root))
        settings_menu.add_command(label="Tool setup…", command=self.open_setup)
        settings_menu.add_command(label="Download tools, models and libraries…",
                                  command=lambda: self.open_setup(downloads=True))
        settings_menu.add_command(label="Set up FIDDLE…", command=self.open_fiddle_setup)
        menu.add_cascade(label="Settings", menu=settings_menu)
        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="Create desktop and Start menu shortcuts", command=self.create_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About MS-Emblator", command=self.about)
        menu.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menu)
        self.root.bind("<Control-o>", lambda _e: self.open_msp())
        self.root.bind("<F5>", lambda _e: self.run())
        self.fill_recent_menu()

    def _toolbar(self):
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(fill="x")
        for text, command in (("📂  Open MSP", self.open_msp), ("▶  Run", self.run), ("■  Stop", self.stop),
                              ("📁  Output folder", self.open_output),
                              ("⚙  Parameters", lambda: ParametersDialog(self.root)),
                              ("🧰  Tool setup", self.open_setup),
                              ("⬇  Downloads", lambda: self.open_setup(downloads=True))):
            button = ttk.Button(bar, text=text, command=command)
            button.pack(side="left", padx=(0, 6))
            if text.endswith("Run"):
                self.toolbar_run = button
            elif text.endswith("Stop"):
                self.toolbar_stop = button
        hint = "Drop an MSP file or a results folder anywhere on the window" if TkinterDnD else \
            "Drag and drop needs the tkinterdnd2 package"
        ttk.Label(bar, text=hint, foreground=MUTED).pack(side="right")
        ttk.Separator(self.root).pack(fill="x")

    def _body(self):
        panes = ttk.Panedwindow(self.root, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Panedwindow(panes, orient="vertical")
        panes.add(left, weight=2)
        list_frame = ttk.LabelFrame(left, text="Spectra", padding=4)
        self.spectra_frame = list_frame
        columns = ("no", "name", "mz", "adduct", "formula", "top")
        self.spectra_tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")
        for column, title, width, anchor in (("no", "#", 34, "e"), ("name", "Name", 150, "w"),
                                             ("mz", "Precursor m/z", 92, "e"), ("adduct", "Adduct", 66, "w"),
                                             ("formula", "Formula (MSP)", 96, "w"),
                                             ("top", "Top formula", 96, "w")):
            self.spectra_tree.heading(column, text=title)
            self.spectra_tree.column(column, width=width, minwidth=30, anchor=anchor, stretch=column == "name")
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.spectra_tree.yview)
        hscroll = ttk.Scrollbar(list_frame, orient="horizontal", command=self.spectra_tree.xview)
        self.spectra_tree.configure(yscrollcommand=scroll.set, xscrollcommand=hscroll.set)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        self.spectra_tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        hscroll.grid(row=1, column=0, sticky="ew")
        self.drop_hint = ttk.Label(list_frame, foreground=MUTED, justify="center",
                                   text="Open an MSP file\n(File > Open, Ctrl+O, or drop it here)")
        self.drop_hint.place(relx=0.5, rely=0.45, anchor="center")
        self.spectra_tree.bind("<<TreeviewSelect>>", lambda _e: self.on_select_spectrum())
        left.add(list_frame, weight=3)
        viewer = ttk.LabelFrame(left, text="MS/MS spectrum", padding=4)
        self.spectrum_view = SpectrumView(viewer)
        self.spectrum_view.pack(fill="both", expand=True)
        left.add(viewer, weight=2)

        self.notebook = ttk.Notebook(panes)
        panes.add(self.notebook, weight=3)
        self.notebook.add(self._analysis_tab(self.notebook), text="  Analysis  ")
        self.notebook.add(self._results_tab(self.notebook), text="  Results  ")

    def _analysis_tab(self, parent):
        tab = ttk.Frame(parent, padding=10)
        tab.columnconfigure(0, weight=1, uniform="half")
        tab.columnconfigure(1, weight=1, uniform="half")

        files = ttk.LabelFrame(tab, text="1   Files", style="Step.TLabelframe", padding=(10, 6))
        files.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        files.columnconfigure(1, weight=1)
        ttk.Label(files, text="Input MSP").grid(row=0, column=0, sticky="w")
        self.input_entry = ttk.Entry(files, textvariable=self.input_path)
        self.input_entry.grid(row=0, column=1, sticky="ew", padx=6, pady=2)
        ttk.Button(files, text="Browse…", command=self.open_msp).grid(row=0, column=2)
        ttk.Label(files, text="Output folder").grid(row=1, column=0, sticky="w")
        self.output_entry = ttk.Entry(files, textvariable=self.output_dir)
        self.output_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=2)
        ttk.Button(files, text="Browse…", command=self.choose_output).grid(row=1, column=2)

        mode_box = ttk.LabelFrame(tab, text="2   Analysis", style="Step.TLabelframe", padding=(10, 6))
        mode_box.grid(row=1, column=0, sticky="nsew", padx=(0, 4), pady=(0, 8))
        for mode, (title, description) in backend.MODES.items():
            ttk.Radiobutton(mode_box, text=title, value=mode, variable=self.mode).grid(row=mode, column=0, sticky="w")
            ttk.Label(mode_box, text=description, foreground=MUTED).grid(row=mode, column=1, sticky="w", padx=8)

        sirius_box = ttk.LabelFrame(tab, text="SIRIUS account", padding=(10, 6))
        sirius_box.grid(row=1, column=1, sticky="nsew", padx=(4, 0), pady=(0, 8))
        sirius_box.columnconfigure(1, weight=1)
        self.sirius_widgets = []
        for row, (label, variable, show) in enumerate((("User (e-mail)", self.sirius_user, ""),
                                                       ("Password", self.sirius_password, "•"))):
            text = ttk.Label(sirius_box, text=label)
            text.grid(row=row, column=0, sticky="w")
            entry = ttk.Entry(sirius_box, textvariable=variable, show=show)
            entry.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
            self.sirius_widgets += [text, entry]
        self.sirius_note = ttk.Label(sirius_box, foreground=MUTED, wraplength=330, justify="left")
        self.sirius_note.grid(row=2, column=0, columnspan=2, sticky="w")

        self.formula_chooser = ToolChooser(tab, "3   Formula tools", self.formula_choice, self.show_tool_details,
                                           style="Step.TLabelframe")
        self.formula_chooser.grid(row=2, column=0, sticky="nsew", padx=(0, 4), pady=(0, 8))
        self.structure_chooser = ToolChooser(tab, "4   Structure tools", self.structure_choice,
                                             self.show_tool_details, style="Step.TLabelframe")
        self.structure_chooser.grid(row=2, column=1, sticky="nsew", padx=(4, 0), pady=(0, 8))

        run_bar = ttk.Frame(tab)
        run_bar.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.run_button = ttk.Button(run_bar, text="▶  Run analysis", style="Run.TButton", command=self.run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(run_bar, text="■  Stop", command=self.stop)
        self.stop_button.pack(side="left", padx=6)
        ttk.Button(run_bar, text="Command line…", command=self.show_command).pack(side="left")
        ttk.Label(run_bar, textvariable=self.step_text, foreground=ACCENT, width=26).pack(side="right")
        self.progress = ttk.Progressbar(run_bar, mode="determinate", length=180)
        self.progress.pack(side="right", padx=8)

        log_box = ttk.LabelFrame(tab, text="Log", padding=4)
        log_box.grid(row=4, column=0, columnspan=2, sticky="nsew")
        tab.rowconfigure(4, weight=1)
        self.log = tk.Text(log_box, height=10, wrap="none", font=("Consolas", 9), background="#fafafa",
                           state="disabled")
        log_scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.tag_configure("error", foreground=NG_COLOR)
        self.log.tag_configure("step", foreground=ACCENT)
        self.log.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")
        return tab

    def _results_tab(self, parent):
        tab = ttk.Frame(parent, padding=10)
        self.result_title = ttk.Label(tab, text="No results loaded", style="Title.TLabel")
        self.result_title.pack(anchor="w")
        self.result_info = ttk.Label(tab, foreground=MUTED,
                                     text="Run an analysis, or use File > Load results… for an earlier run.")
        self.result_info.pack(anchor="w", pady=(0, 6))
        panes = ttk.Panedwindow(tab, orient="vertical")
        panes.pack(fill="both", expand=True)
        self.formula_table = self._candidate_table(panes, "Formula candidates", "Formula", 150, weight=2)
        self.structure_table = self._candidate_table(panes, "Structure candidates", "SMILES", 260, weight=3)
        self.structure_table.bind("<<TreeviewSelect>>", lambda _e: self.on_select_structure())

        detail = ttk.LabelFrame(panes, text="Selected structure", padding=4)
        panes.add(detail, weight=4)
        image_box = ttk.Frame(detail)
        image_box.pack(side="left", fill="y")
        # Fixed pixel size: a tk.Label without an image would size itself in characters.
        image_frame = tk.Frame(image_box, width=270, height=210, background="white")
        image_frame.pack_propagate(False)
        image_frame.pack(anchor="n")
        self.structure_canvas = tk.Label(image_frame, background="white", foreground=MUTED, relief="flat")
        self.structure_canvas.pack(fill="both", expand=True)
        self.structure_caption = ttk.Label(image_box, foreground=MUTED, wraplength=260, justify="left")
        self.structure_caption.pack(anchor="w", pady=(2, 0))
        mirror_box = ttk.Frame(detail)
        mirror_box.pack(side="left", fill="both", expand=True, padx=(8, 0))
        energy_row = ttk.Frame(mirror_box)
        energy_row.pack(fill="x")
        ttk.Label(energy_row, text="GLACIER prediction").pack(side="left")
        self.energy_box = ttk.Combobox(energy_row, textvariable=self.energy_choice, state="readonly", width=12)
        self.energy_box.pack(side="left", padx=6)
        self.energy_box.bind("<<ComboboxSelected>>", lambda _e: self.show_mirror())
        self.mirror = MirrorPlot(mirror_box)
        self.mirror.pack(fill="both", expand=True, pady=(4, 0))

        self.result_files_label = ttk.Label(tab, foreground=MUTED)
        self.result_files_label.pack(anchor="w", pady=(6, 0))
        return tab

    def _candidate_table(self, panes, title, value_title, value_width, weight=1):
        box = ttk.LabelFrame(panes, text=title, padding=4)
        columns = ("rank", "value", "probability", "score", "tools")
        tree = ttk.Treeview(box, columns=columns, show="headings", selectmode="browse", height=4)
        for column, heading, width, anchor in (("rank", "Rank", 50, "e"), ("value", value_title, value_width, "w"),
                                               ("probability", "Probability", 90, "e"),
                                               ("score", "Score", 70, "e"), ("tools", "Supported by", 300, "w")):
            tree.heading(column, text=heading)
            tree.column(column, width=width, anchor=anchor, stretch=column in ("value", "tools"))
        tree.tag_configure("top", font=("Segoe UI", 9, "bold"))
        scroll = ttk.Scrollbar(box, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        panes.add(box, weight=weight)
        return tree

    def _statusbar(self):
        bar = ttk.Frame(self.root, padding=(8, 2))
        bar.pack(fill="x", side="bottom")
        ttk.Label(bar, textvariable=self.status_text).pack(side="left")

    # drag and drop --------------------------------------------------------------------

    def _enable_drop(self):
        if TkinterDnD is None or not hasattr(self.root, "drop_target_register"):
            return
        for widget in (self.root, self.spectra_tree, self.notebook, self.input_entry, self.output_entry,
                       self.spectrum_view, self.log):
            try:
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self.on_drop)
            except Exception:
                pass

    def on_drop(self, event):
        self.handle_dropped(self.root.tk.splitlist(event.data))
        return event.action

    def handle_dropped(self, items):
        """MSP file -> open it; folder -> load its results, or use it as the output folder."""
        for item in items:
            path = Path(item)
            if path.is_file() and path.suffix.lower() == ".msp":
                self.load_msp(str(path))
                return
            if path.is_dir():
                if backend.latest_results(path):
                    self.output_dir.set(str(path))
                    self.load_results(str(path))
                else:
                    self.output_dir.set(str(path))
                    self.status_text.set(f"Output folder: {path}")
                return
        messagebox.showinfo(APP_TITLE, "Drop an .msp file, or a folder for the results.")

    # remembered settings ----------------------------------------------------------------

    def fill_recent_menu(self):
        self.recent_menu.delete(0, "end")
        recent = [p for p in self.stored.get("recent_inputs", []) if p]
        if not recent:
            self.recent_menu.add_command(label="(none)", state="disabled")
        for path in recent:
            self.recent_menu.add_command(label=path, command=lambda p=path: self.load_msp(p))

    def save_settings(self):
        if not self.remember:
            return
        self.stored.update({
            "input_path": self.input_path.get(), "output_dir": self.output_dir.get(), "mode": self.mode.get(),
            "formula_tools": self.formula_choice.get(), "structure_tools": self.structure_choice.get(),
            "sirius_user": self.sirius_user.get().strip(), "geometry": self.root.geometry(),
        })
        gui_settings.save_settings(self.stored)

    # tool status ------------------------------------------------------------------

    def refresh_tool_status(self):
        try:
            self.statuses = toolsets.available_selections(toolsets.load_toolsets())
        except Exception as exc:  # a broken toolsets.yaml must not stop the GUI
            messagebox.showerror(APP_TITLE, f"Cannot read the tool combinations:\n{exc}")
            self.statuses = {toolsets.FORMULA: [], toolsets.STRUCTURE: []}
        for step, chooser, variable in ((toolsets.FORMULA, self.formula_chooser, self.formula_choice),
                                        (toolsets.STRUCTURE, self.structure_chooser, self.structure_choice)):
            entries = self.statuses.get(step, [])
            chooser.refresh(entries)
            names = [s.selection.name for s in entries]
            if variable.get() not in names:
                default = next((s.selection.name for s in entries if s.is_default), names[0] if names else "")
                variable.set(default)
        self.update_states()
        return self.statuses

    def selected_status(self, step):
        variable = self.formula_choice if step == toolsets.FORMULA else self.structure_choice
        return next((s for s in self.statuses.get(step, []) if s.selection.name == variable.get()), None)

    def active_statuses(self):
        mode = self.mode.get()
        steps = [toolsets.FORMULA] * (mode in (1, 2)) + [toolsets.STRUCTURE] * (mode in (2, 3))
        return [s for s in (self.selected_status(step) for step in steps) if s]

    def show_tool_details(self, status):
        selection = status.selection
        tools = ", ".join(TOOL_LABELS[t] for t in selection.tools)
        if status.ready:
            messagebox.showinfo(f"{selection.step.capitalize()} tools: {selection.name}",
                                f"{tools}\n\nEverything is in place.\nModel folder: {selection.model_dir}",
                                parent=self.root)
        else:
            fiddle_missing = any("FIDDLE" in m for m in status.missing)
            others = [m for m in status.missing if "FIDDLE" not in m]
            question = "\n\nSet up FIDDLE now?" if fiddle_missing and not others else "\n\nOpen the downloads?"
            if messagebox.askyesno(f"{selection.step.capitalize()} tools: {selection.name}",
                                   f"{tools}\n\nMissing:\n" + "\n".join(f"• {m}" for m in status.missing) + question,
                                   parent=self.root):
                if fiddle_missing and not others:
                    self.open_fiddle_setup()
                else:
                    self.open_setup(downloads=True)

    def open_setup(self, downloads=False):
        SetupDialog(self.root, self.statuses, self.refresh_tool_status, select_downloads=downloads)

    def open_fiddle_setup(self):
        FiddleSetupDialog(self.root, on_done=self.refresh_tool_status)

    def update_states(self):
        mode = self.mode.get()
        running = self.process is not None
        self.formula_chooser.set_enabled(mode in (1, 2) and not running)
        self.structure_chooser.set_enabled(mode in (2, 3) and not running)
        uses_sirius = any(s.selection.uses("sirius") for s in self.active_statuses())
        for widget in self.sirius_widgets:
            widget.state(["!disabled" if uses_sirius and not running else "disabled"])
        self.sirius_note.configure(
            text="Leave empty to use the login stored by 'sirius.exe login'. The password is never saved."
            if uses_sirius else "Not needed: SIRIUS is not among the selected tools.")
        for button in (self.run_button, self.toolbar_run):
            button.state(["disabled" if running else "!disabled"])
        for button in (self.stop_button, self.toolbar_stop):
            button.state(["!disabled" if running else "disabled"])

    # files --------------------------------------------------------------------------

    def open_msp(self):
        start = Path(self.input_path.get()).parent if self.input_path.get() else None
        path = filedialog.askopenfilename(title="Open MSP file", initialdir=str(start) if start else None,
                                          filetypes=[("MSP files", "*.msp"), ("All files", "*.*")])
        if path:
            self.load_msp(path)

    def load_msp(self, path, quiet=False):
        try:
            spectra = backend.read_spectra(path)
        except Exception as exc:
            if not quiet:
                messagebox.showerror(APP_TITLE, f"Cannot read the MSP file:\n{exc}")
            return
        self.input_path.set(path)
        if not self.output_dir.get():
            self.output_dir.set(str(Path(path).parent / f"{Path(path).stem}_msemblator"))
        self.spectra = spectra
        self.results, self.result_files = {}, {}
        self.fill_spectra()
        self.clear_results()
        self.status_text.set(f"{len(spectra)} spectra in {Path(path).name}")
        self.root.title(f"{APP_TITLE} — {Path(path).name}")
        self.stored["recent_inputs"] = gui_settings.add_recent(self.stored.get("recent_inputs", []), path)
        self.fill_recent_menu()
        self.save_settings()

    def fill_spectra(self):
        tops = {}
        summary = self.results.get("formula_summary")
        if summary is not None and "Top_score_formula" in summary:
            tops = dict(zip(summary["filename"].astype(str), summary["Top_score_formula"].astype(str)))
        self.spectra_tree.delete(*self.spectra_tree.get_children())
        for spectrum in self.spectra:
            self.spectra_tree.insert("", "end", iid=str(spectrum.index), values=(
                spectrum.index, spectrum.name,
                f"{spectrum.precursor_mz:.4f}" if spectrum.precursor_mz else "",
                spectrum.adduct, spectrum.formula, tops.get(spectrum.name, "")))
        self.spectra_frame.configure(text=f"Spectra ({len(self.spectra)})")
        if self.spectra:
            self.drop_hint.place_forget()
            self.spectra_tree.selection_set(str(self.spectra[0].index))
        else:
            self.drop_hint.place(relx=0.5, rely=0.45, anchor="center")

    def choose_output(self):
        path = filedialog.askdirectory(title="Output folder", initialdir=self.output_dir.get() or None)
        if path:
            self.output_dir.set(path)

    def open_output(self):
        folder = self.output_dir.get()
        if folder and Path(folder).is_dir():
            os.startfile(folder)  # noqa: S606 - Windows only, opens Explorer
        else:
            messagebox.showinfo(APP_TITLE, "The output folder does not exist yet.")

    # results ------------------------------------------------------------------------

    def load_results_dialog(self):
        folder = filedialog.askdirectory(title="Folder with formula_summary.csv / structure_summary.csv",
                                         initialdir=self.output_dir.get() or None)
        if folder:
            self.output_dir.set(folder)
            if not self.load_results(folder):
                messagebox.showinfo(APP_TITLE, "No MS-Emblator result files in this folder.")

    def load_results(self, folder, since=None):
        files = backend.latest_results(folder, since)
        if not files:
            return False
        try:
            self.results = backend.load_results(files)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Cannot read the results:\n{exc}")
            return False
        self.result_files = files
        if not self.spectra:   # results opened without the MSP: list the spectra from the summary
            summary = self.results.get("formula_summary", self.results.get("structure_summary"))
            if summary is not None:
                self.spectra = [backend.SpectrumInfo(i + 1, str(r.filename), None, str(r.adduct), "", [], [])
                                for i, r in enumerate(summary[["filename", "adduct"]].itertuples())]
        self.fill_spectra()
        self.result_files_label.configure(text="Files: " + ", ".join(p.name for p in files.values())
                                          + f"   (in {Path(folder)})")
        self.notebook.select(1)
        self.save_settings()
        return True

    def current_spectrum(self):
        selected = self.spectra_tree.selection()
        return next((s for s in self.spectra if selected and str(s.index) == selected[0]), None)

    def clear_results(self):
        for tree in (self.formula_table, self.structure_table):
            tree.delete(*tree.get_children())
        self.show_structure(None)

    def on_select_spectrum(self):
        spectrum = self.current_spectrum()
        if spectrum is None:
            return
        self.spectrum_view.show(spectrum)
        self.result_title.configure(text=spectrum.name)
        details = [f"Adduct {spectrum.adduct}"]
        if spectrum.precursor_mz:
            details.insert(0, f"Precursor m/z {spectrum.precursor_mz:.4f}")
        if spectrum.formula:
            details.append(f"Formula in the MSP {spectrum.formula}")
        self.result_info.configure(text="   |   ".join(details))
        for tree, name, column in ((self.formula_table, "formula_score", "formula"),
                                   (self.structure_table, "structure_score", "Canonical_SMILES")):
            tree.delete(*tree.get_children())
            for row in backend.candidates_for(self.results.get(name), spectrum.name, column):
                tree.insert("", "end", tags=("top",) if row["rank"] == 1 else (), values=(
                    row["rank"], row["candidate"], f"{row['probability'] * 100:.1f} %",
                    f"{row['score']:.3f}", row["tools"]))
        first = self.structure_table.get_children()
        if first:
            self.structure_table.selection_set(first[0])
        else:
            self.show_structure(None)

    def on_select_structure(self):
        selected = self.structure_table.selection()
        if not selected:
            return
        values = self.structure_table.item(selected[0], "values")
        self.show_structure(values[1], rank=values[0])

    def show_structure(self, smiles, rank=None):
        """Structure drawing and, when GLACIER predicted the candidate, the mirror plot."""
        if not smiles:
            self.structure_image = None
            self.structure_canvas.configure(image="", text="")
            self.structure_caption.configure(text="")
            self.predictions = []
            self.energy_box.configure(values=[])
            self.energy_choice.set("")
            self.mirror.show(None, message="Select a structure candidate")
            return
        png = backend.structure_png(smiles, 270, 210)
        if png:
            self.structure_image = tk.PhotoImage(data=base64.b64encode(png).decode("ascii"))
            self.structure_canvas.configure(image=self.structure_image, text="")
        else:
            self.structure_image = None
            self.structure_canvas.configure(image="", text="(cannot draw this SMILES)")
        self.structure_caption.configure(text=f"Rank {rank}: {smiles}" if rank else smiles)

        spectrum = self.current_spectrum()
        spectra_table = self.results.get("glacier_spectra")
        self.predictions = backend.glacier_spectra_for(spectra_table, spectrum.name, smiles) if spectrum else []
        labels = [f"{p['collision_energy']:g} eV" if p["collision_energy"] is not None else "unknown eV"
                  for p in self.predictions]
        self.energy_box.configure(values=labels)
        if self.predictions:
            best = max(range(len(self.predictions)),
                       key=lambda i: next(iter(self.predictions[i]["scores"].values()), 0.0))
            self.energy_choice.set(labels[best])
            self.show_mirror()
        else:
            self.energy_choice.set("")
            message = ("No GLACIER prediction for this candidate." if spectra_table is not None and not spectra_table.empty
                       else "The mirror plot is shown when the analysis used GLACIER (structure tools "
                            "'glacier' or 'all').")
            self.mirror.show(spectrum, None, message)

    def show_mirror(self):
        labels = list(self.energy_box.cget("values"))
        choice = self.energy_choice.get()
        if choice in labels and self.predictions:
            self.mirror.show(self.current_spectrum(), self.predictions[labels.index(choice)])

    # running ------------------------------------------------------------------------

    def current_request(self):
        user = self.sirius_user.get().strip()
        uses_sirius = any(s.selection.uses("sirius") for s in self.active_statuses())
        return backend.RunRequest(
            input_msp=self.input_path.get().strip(), output_dir=self.output_dir.get().strip(),
            mode=self.mode.get(), formula_tools=self.formula_choice.get() or None,
            structure_tools=self.structure_choice.get() or None,
            sirius_user=user if (uses_sirius and user) else None)

    def show_command(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Command line")
        dialog.transient(self.root)
        ttk.Label(dialog, padding=(10, 8, 10, 0),
                  text="The same analysis from PowerShell in the repository folder:").pack(anchor="w")
        text = tk.Text(dialog, height=4, width=100, font=("Consolas", 9), wrap="word")
        text.insert("1.0", backend.display_command(self.current_request()))
        text.pack(fill="both", expand=True, padx=10, pady=8)
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(anchor="e", padx=10, pady=(0, 10))

    def run(self):
        if self.process is not None:
            return
        request = self.current_request()
        problems = []
        if not request.input_msp or not Path(request.input_msp).is_file():
            problems.append("Choose an input MSP file.")
        if not request.output_dir:
            problems.append("Choose an output folder.")
        password = self.sirius_password.get()
        if request.sirius_user and not password:
            problems.append("Enter the SIRIUS password, or leave the user empty to use the stored login.")
        not_ready = [s for s in self.active_statuses() if not s.ready]
        for status in not_ready:
            problems.append(f"{status.selection.step.capitalize()} tools '{status.selection.name}' are not "
                            f"ready ({len(status.missing)} file(s) missing).")
        if problems:
            if not_ready and messagebox.askyesno(APP_TITLE, "\n\n".join(problems) + "\n\nOpen Tool setup?"):
                self.open_setup(downloads=True)
            elif not not_ready:
                messagebox.showwarning(APP_TITLE, "\n\n".join(problems))
            return
        if request.mode == 3 and self.spectra:
            without = sum(1 for s in self.spectra if not s.formula)
            if without and not messagebox.askyesno(
                    APP_TITLE, f"{without} of {len(self.spectra)} spectra have no formula and get no structure "
                               "result in 'Structure only'.\n\nRun anyway?"):
                return

        self.save_settings()
        self.clear_log()
        self.append_log("> " + backend.display_command(request) + "\n", "step")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.process = subprocess.Popen(
                backend.build_command(request), cwd=str(paths.PROJECT_ROOT), env=backend.run_environment(request, password),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"Cannot start MS-Emblator:\n{exc}")
            return
        self.run_started = time.time()
        threading.Thread(target=self._read_output, args=(self.process,), daemon=True).start()
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.step_text.set("Starting…")
        self.status_text.set("Running…")
        self.update_states()

    def _read_output(self, process):
        for raw in iter(process.stdout.readline, b""):
            self.messages.put(("line", backend.decode_line(raw)))
        self.messages.put(("exit", process.wait()))

    def poll_messages(self):
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == "line":
                    step = backend.step_of(value)
                    if step:
                        self.step_text.set(f"{step}…")
                        self.status_text.set(f"Running: {step}")
                    tag = "error" if any(w in value for w in ("Error", "ERROR", "Traceback", "failed")) else \
                        ("step" if step else None)
                    self.append_log(value + "\n", tag)
                else:
                    self.finish(value)
        except queue.Empty:
            pass
        self._poll_job = self.root.after(100, self.poll_messages)

    def finish(self, code):
        self.process = None
        self.progress.stop()
        self.progress.configure(mode="determinate", value=100 if code == 0 else 0)
        self.update_states()
        minutes = (time.time() - self.run_started) / 60
        loaded = self.load_results(self.output_dir.get(), since=self.run_started)
        if code == 0:
            self.step_text.set("Finished")
            self.status_text.set(f"Finished in {minutes:.1f} min. Results: {self.output_dir.get()}")
        elif loaded:
            self.step_text.set("Finished with errors")
            self.status_text.set("Some tools failed; see the log.")
            messagebox.showwarning(APP_TITLE, "The analysis finished, but some tools failed.\n"
                                              "Results were written; the log shows which tools failed.")
        else:
            self.step_text.set("Failed")
            self.status_text.set(f"Stopped (exit code {code}); see the log.")
            messagebox.showerror(APP_TITLE, f"MS-Emblator stopped with exit code {code}.\nSee the log for details.")

    def stop(self):
        process = self.process
        if process is None:
            return
        if not messagebox.askyesno(APP_TITLE, "Stop the running analysis?"):
            return
        self._kill(process)
        self.append_log("Stopped by the user.\n", "error")

    @staticmethod
    def _kill(process):
        if os.name == "nt":   # also stop SIRIUS, MS-FINDER and MetFrag started by it
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            process.terminate()

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def append_log(self, text, tag=None):
        self.log.configure(state="normal")
        self.log.insert("end", text, (tag,) if tag else ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def create_shortcuts(self):
        try:
            created = backend.create_shortcuts()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"The shortcuts could not be created:\n{exc}\n\n"
                                            "scripts\\create_shortcut.cmd does the same from Explorer.")
            return
        messagebox.showinfo(APP_TITLE, "Shortcuts with the MS-Emblator icon:\n\n" + "\n".join(created)
                            + "\n\nThey start this copy of MS-Emblator; create them again after moving it.")

    def about(self):
        dialog = tk.Toplevel(self.root)
        dialog.title(f"About {APP_TITLE}")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=16)
        frame.pack()
        try:
            logo = tk.PhotoImage(file=str(ICON_128))
            dialog.logo = logo
            ttk.Label(frame, image=logo).grid(row=0, column=0, rowspan=3, padx=(0, 16))
        except tk.TclError:
            pass
        ttk.Label(frame, text=APP_TITLE, style="Title.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(frame, text="Ensemble annotation of MS/MS spectra").grid(row=1, column=1, sticky="w")
        ttk.Label(frame, foreground=MUTED, justify="left",
                  text=f"Program folder: {paths.PROJECT_ROOT}\nSettings: {gui_settings.settings_path()}"
                  ).grid(row=2, column=1, sticky="w", pady=(8, 0))
        ttk.Button(frame, text="Close", command=dialog.destroy).grid(row=3, column=1, sticky="e", pady=(12, 0))

    def on_close(self):
        if self.process is not None:
            if not messagebox.askyesno(APP_TITLE, "An analysis is running. Stop it and exit?"):
                return
            self._kill(self.process)
        self.save_settings()
        self.root.destroy()


ASSETS = Path(__file__).resolve().parent / "assets"
ICON_ICO = ASSETS / "msemblator.ico"
ICON_PNG = ASSETS / "msemblator.png"            # 256 px
ICON_128 = ASSETS / "msemblator_128.png"        # About dialog (rendered at this size, not scaled)


def set_icon(root):
    """MS-Emblator logo for the main window, its dialogs and the taskbar."""
    try:
        if os.name == "nt" and ICON_ICO.is_file():
            root.iconbitmap(default=str(ICON_ICO))      # default: also every dialog
        elif ICON_PNG.is_file():
            root.icon_image = tk.PhotoImage(file=str(ICON_PNG))
            root.iconphoto(True, root.icon_image)
    except tk.TclError:
        pass   # an icon is cosmetic; never stop the GUI for it


def create_root():
    """Tk root window, with drag and drop when tkinterdnd2 is installed."""
    root = None
    if TkinterDnD is not None:
        try:
            root = TkinterDnD.Tk()
        except Exception:
            root = None
    root = root or tk.Tk()
    set_icon(root)
    return root


def main():
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)   # sharp text on high-DPI screens
            # Own taskbar entry with the MS-Emblator icon, instead of being grouped under Python.
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MS-Emblator.GUI")
        except Exception:
            pass
    root = create_root()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
