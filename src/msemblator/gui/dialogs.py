"""Dialogs of the GUI: analysis parameters, tool setup (combinations and downloads), license consent."""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk

from msemblator import downloads, paths
from msemblator.gui import backend
from msemblator.gui.widgets import MUTED, NG_COLOR, OK_COLOR, TOOL_LABELS

PARAMETER_LABELS = {
    "MS1_ppm": "MS1 tolerance (ppm)", "MS2_ppm": "MS2 tolerance (ppm)", "MS2_Da": "MS2 tolerance (Da)",
    "MS1": "Instrument (qtof / orbitrap)", "halogen": "Consider halogens",
    "msemblator_output_records": "Candidates kept in the output",
    "conda_env": "Conda environment", "python": "Python executable (optional)", "device": "Device (cpu / cuda)",
    "batch_size": "Batch size", "chunk_size": "Chunk size", "top_k": "Candidates per spectrum",
    "default_instrument": "Instrument when unknown",
    "default_collision_energy": "Collision energy (eV) if the MSP has none",
    "override_collision_energy": "Use this energy for every spectrum (ignore the MSP)",
}
SECTION_LABELS = {"formula_prediction": "Formula prediction", "structure_prediction": "Structure prediction"}
STATUS_TAGS = {downloads.INSTALLED: "ok", downloads.UPDATE: "update", downloads.MISSING: "missing",
               downloads.UNPUBLISHED: "muted"}


def open_in_editor(path):
    """Open a text file in the user's editor (the .yaml association, else Notepad)."""
    try:
        os.startfile(str(path))  # noqa: S606 - Windows
    except OSError:
        subprocess.Popen(["notepad.exe", str(path)])


class ParametersDialog(tk.Toplevel):
    """Edit config/msemblator_parameter_file.yaml (comments in the file are kept)."""

    def __init__(self, master):
        super().__init__(master)
        self.title("Analysis parameters")
        self.transient(master)
        self.resizable(False, False)
        try:
            self.settings = backend.read_parameters()
        except OSError as exc:
            messagebox.showerror("Analysis parameters", f"Cannot read {paths.PARAMETER_FILE}:\n{exc}", parent=master)
            self.destroy()
            return
        self.fields = {}
        notebook = ttk.Notebook(self, padding=8)
        notebook.pack(fill="both", expand=True)
        for section, title in SECTION_LABELS.items():
            notebook.add(self._section_tab(notebook, section), text=title)
        buttons = ttk.Frame(self, padding=(8, 0, 8, 8))
        buttons.pack(fill="x")
        ttk.Label(buttons, text=f"Saved to {paths.PARAMETER_FILE.name}", foreground=MUTED).pack(side="left")
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="Save", command=self.save).pack(side="right", padx=6)
        self.grab_set()

    def _section_tab(self, notebook, section):
        frame = ttk.Frame(notebook, padding=8)
        values = self.settings.get(section) or {}
        general = {k: v for k, v in values.items() if not isinstance(v, dict)}
        column = 0
        for tool, entries in values.items():
            if not isinstance(entries, dict):
                continue
            box = ttk.LabelFrame(frame, text=TOOL_LABELS.get(tool, tool), padding=8)
            box.grid(row=0, column=column, sticky="nsew", padx=4, pady=4)
            self._add_fields(box, (section, tool), entries)
            column += 1
        if general:
            box = ttk.LabelFrame(frame, text="Output", padding=8)
            box.grid(row=1, column=0, columnspan=max(column, 1), sticky="ew", padx=4, pady=4)
            self._add_fields(box, (section,), general)
        return frame

    def _add_fields(self, box, prefix, entries):
        for row, (key, value) in enumerate(entries.items()):
            if isinstance(value, dict):
                continue
            ttk.Label(box, text=PARAMETER_LABELS.get(key, key)).grid(row=row, column=0, sticky="w", pady=2)
            if isinstance(value, bool):
                variable = tk.BooleanVar(value=value)
                ttk.Checkbutton(box, variable=variable).grid(row=row, column=1, sticky="w", padx=6)
            else:
                variable = tk.StringVar(value="" if value is None else str(value))
                ttk.Entry(box, textvariable=variable, width=14).grid(row=row, column=1, sticky="w", padx=6)
            self.fields[prefix + (key,)] = (variable, value)

    def save(self):
        values = {}
        for key_path, (variable, original) in self.fields.items():
            raw = variable.get()
            values[key_path] = raw if isinstance(raw, bool) else backend.parse_entry(raw, original)
        try:
            changed = backend.update_parameters(values)
        except OSError as exc:
            messagebox.showerror("Analysis parameters", f"Cannot save:\n{exc}", parent=self)
            return
        messagebox.showinfo("Analysis parameters", f"Saved ({changed} value(s) changed).", parent=self)
        self.destroy()


class ConsentDialog(tk.Toplevel):
    """Show what will be downloaded, from where and under which license; ask for agreement."""

    def __init__(self, master, assets):
        super().__init__(master)
        self.title("Download — licenses and terms")
        self.transient(master)
        self.accepted = False
        total = sum(a.size_mb or 0 for a in assets)
        ttk.Label(self, padding=(12, 10, 12, 4), font=("Segoe UI", 10, "bold"),
                  text=f"{len(assets)} item(s), about {total:,.0f} MB").pack(anchor="w")
        text = tk.Text(self, width=96, height=min(6 + 6 * len(assets), 24), wrap="word", font=("Segoe UI", 9),
                       background="#fafafa", relief="flat", padx=8, pady=6)
        text.tag_configure("title", font=("Segoe UI", 9, "bold"))
        text.tag_configure("muted", foreground=MUTED)
        text.tag_configure("link", foreground="#1f5fa8", underline=True)
        for asset in assets:
            text.insert("end", f"{asset.title}", "title")
            text.insert("end", f"   {asset.size_mb:g} MB\n" if asset.size_mb else "\n", "muted")
            text.insert("end", f"From: {asset.url}\n", "muted")
            text.insert("end", f"License: {asset.license or 'see the source'}\n")
            if asset.license_url:
                tag = f"link{id(asset)}"
                text.insert("end", asset.license_url + "\n", ("link", tag))
                text.tag_bind(tag, "<Button-1>", lambda _e, u=asset.license_url: webbrowser.open(u))
                text.tag_bind(tag, "<Enter>", lambda _e: text.configure(cursor="hand2"))
                text.tag_bind(tag, "<Leave>", lambda _e: text.configure(cursor=""))
            if asset.note:
                text.insert("end", f"Note: {asset.note}\n", "muted")
            text.insert("end", "\n")
        text.configure(state="disabled")
        text.pack(fill="both", expand=True, padx=12)
        self.agree = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, variable=self.agree, padding=(12, 8),
                        text="I have read these licenses and terms and accept them.",
                        command=lambda: self.ok.state(["!disabled" if self.agree.get() else "disabled"])).pack(anchor="w")
        buttons = ttk.Frame(self, padding=(12, 0, 12, 12))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right")
        self.ok = ttk.Button(buttons, text="Download", command=self._accept)
        self.ok.pack(side="right", padx=6)
        self.ok.state(["disabled"])
        self.grab_set()

    def _accept(self):
        self.accepted = True
        self.destroy()


def kill_tree(process):
    """Stop a process and the programs it started (conda, pip, python)."""
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        process.terminate()


class FiddleSetupDialog(tk.Toplevel):
    """Install FIDDLE into the MS-Emblator environment and download the msfiddle weights."""

    PACKAGES_SIZE_MB = 800  # CPU PyTorch, msfiddle and their packages on disk, approximately
    WEIGHTS_SIZE_MB = 250

    def __init__(self, master, on_done=None):
        from msemblator import fiddle_setup

        super().__init__(master)
        self.fiddle_setup = fiddle_setup
        self.title("Set up FIDDLE")
        self.transient(master)
        self.geometry("820x560")
        self.on_done = on_done
        self.messages: queue.Queue = queue.Queue()
        self.cancel = None
        self.process = None
        version = fiddle_setup.msfiddle_version()
        ttk.Label(self, padding=(12, 10, 12, 2), font=("Segoe UI", 10, "bold"),
                  text=f"FIDDLE (msfiddle {version})").pack(anchor="w")
        ttk.Label(self, padding=(12, 0, 12, 6), foreground=MUTED, wraplength=780, justify="left",
                  text="FIDDLE runs in the MS-Emblator environment. Environments created with the current "
                       "environment.yml (or by msemblator_gui.cmd) already contain it; this installs it into an "
                       "older environment if needed (CPU PyTorch, msfiddle) and downloads the FIDDLE model "
                       "weights, once. It takes a few minutes.").pack(anchor="w")
        self.status_box = ttk.LabelFrame(self, text="Current state", padding=8)
        self.status_box.pack(fill="x", padx=12)
        self.status_labels = ttk.Frame(self.status_box)
        self.status_labels.pack(fill="x")
        log_box = ttk.LabelFrame(self, text="Progress", padding=4)
        log_box.pack(fill="both", expand=True, padx=12, pady=8)
        self.log = tk.Text(log_box, height=12, wrap="none", font=("Consolas", 9), background="#fafafa",
                           state="disabled")
        scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.tag_configure("step", foreground="#1f5fa8", font=("Consolas", 9, "bold"))
        self.log.tag_configure("error", foreground=NG_COLOR)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        row = ttk.Frame(self, padding=(12, 0, 12, 12))
        row.pack(fill="x")
        self.progress = ttk.Progressbar(row, mode="determinate", length=200)
        self.progress.pack(side="left")
        self.step_text = tk.StringVar()
        ttk.Label(row, textvariable=self.step_text).pack(side="left", padx=8)
        ttk.Button(row, text="Close", command=self.close).pack(side="right")
        self.cancel_button = ttk.Button(row, text="Cancel", command=self.stop)
        self.cancel_button.pack(side="right", padx=6)
        self.start_button = ttk.Button(row, text="⬇  Set up FIDDLE", command=self.start)
        self.start_button.pack(side="right")
        self.cancel_button.state(["disabled"])
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()
        self.after(150, self._poll)

    def refresh(self):
        self.current = self.fiddle_setup.status()
        for child in self.status_labels.winfo_children():
            child.destroy()
        for line, good in self.current.describe():
            ttk.Label(self.status_labels, text=("✓ " if good else "✗ ") + line,
                      foreground=OK_COLOR if good else NG_COLOR).pack(anchor="w")
        if self.current.ready:
            self.step_text.set("FIDDLE is ready.")
            self.start_button.configure(text="Check again")
        else:
            self.start_button.configure(text="⬇  Set up FIDDLE")

    def _append(self, text, tag=None):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", (tag,) if tag else ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self):
        if self.cancel is not None:
            return
        try:
            steps = self.fiddle_setup.setup_steps(self.current)
        except RuntimeError as exc:
            messagebox.showerror("Set up FIDDLE", str(exc), parent=self)
            return
        version = self.fiddle_setup.msfiddle_version()
        items = []
        if any(s.title.startswith("Install FIDDLE") for s in steps):
            items.append(downloads.Asset(
                "fiddle_packages", f"FIDDLE packages (msfiddle {version}, CPU PyTorch) for this environment",
                "tools", "PyPI and download.pytorch.org (requirements-extras.txt)", "file",
                "environment", size_mb=self.PACKAGES_SIZE_MB,
                license="msfiddle: Apache-2.0. PyTorch: BSD-3-Clause. The other packages are open source "
                        "(MIT, BSD, Apache-2.0).",
                license_url="https://github.com/JosieHong/FIDDLE"))
        if any("weights" in s.title for s in steps):
            items.append(downloads.Asset(
                "fiddle_weights", f"FIDDLE model weights v{version}", "models", self.fiddle_setup.WEIGHTS_URL,
                "file", str(self.fiddle_setup.weights_dir()), size_mb=self.WEIGHTS_SIZE_MB,
                license="Apache-2.0 (FIDDLE).", license_url="https://github.com/JosieHong/FIDDLE"))
        if items:
            consent = ConsentDialog(self, items)
            self.wait_window(consent)
            if not consent.accepted:
                return
        self.cancel = threading.Event()
        self.start_button.state(["disabled"])
        self.cancel_button.state(["!disabled"])
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        threading.Thread(target=self._worker, args=(steps, self.cancel), daemon=True).start()

    def _worker(self, steps, cancel):
        def on_process(process):
            self.process = process

        def on_line(text, is_step):
            self.messages.put(("line", text, is_step))

        ok, failed = self.fiddle_setup.run_steps(steps, on_line, cancel, on_process)
        self.messages.put(("done", ok, "cancelled" if cancel.is_set() else failed))

    def _poll(self):
        try:
            while True:
                message = self.messages.get_nowait()
                if message[0] == "line":
                    _, text, is_step = message
                    if is_step:
                        self.step_text.set(text)
                    tag = "step" if is_step else ("error" if any(w in text for w in ("Error", "ERROR")) else None)
                    self._append(text, tag)
                else:
                    _, ok, failed = message
                    self.cancel, self.process = None, None
                    self.progress.stop()
                    self.progress.configure(mode="determinate", value=100 if ok else 0)
                    self.start_button.state(["!disabled"])
                    self.cancel_button.state(["disabled"])
                    self.refresh()
                    if ok:
                        self._append("FIDDLE is set up.", "step")
                        if self.on_done:
                            self.on_done()
                    else:
                        self.step_text.set("Cancelled" if failed == "cancelled" else f"Failed: {failed}")
                        self._append(f"Stopped: {failed}", "error")
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(150, self._poll)

    def stop(self):
        if self.cancel is not None:
            self.cancel.set()
            kill_tree(self.process)
            self.step_text.set("Cancelling…")

    def close(self):
        if self.cancel is not None:
            if not messagebox.askyesno("Set up FIDDLE", "The setup is running. Stop it and close?", parent=self):
                return
            self.stop()
        self.destroy()


class SetupDialog(tk.Toplevel):
    """Tool combinations (what can run) and downloads of the programs, models and libraries."""

    def __init__(self, master, statuses, on_refresh, select_downloads=False):
        super().__init__(master)
        self.title("Tool setup")
        self.transient(master)
        self.geometry("940x560")
        self.on_refresh = on_refresh
        self.messages: queue.Queue = queue.Queue()
        self.cancel = None
        notebook = ttk.Notebook(self, padding=8)
        notebook.pack(fill="both", expand=True)
        notebook.add(self._combinations_tab(notebook), text="  Tool combinations  ")
        notebook.add(self._downloads_tab(notebook), text="  Downloads  ")
        if select_downloads:
            notebook.select(1)
        self.fill_combinations(statuses)
        self.fill_downloads()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(100, self._poll)

    # combinations -------------------------------------------------------------------

    def _combinations_tab(self, parent):
        tab = ttk.Frame(parent, padding=6)
        ttk.Label(tab, foreground=MUTED, text="Each combination needs its tools, libraries and a trained ranking "
                                             "model (config/toolsets.yaml).").pack(anchor="w")
        self.tree = ttk.Treeview(tab, columns=("step", "name", "tools", "status"), show="headings", height=7)
        for column, title, width in (("step", "Step", 90), ("name", "Name", 80), ("tools", "Tools", 400),
                                     ("status", "Status", 120)):
            self.tree.heading(column, text=title)
            self.tree.column(column, width=width, anchor="w")
        self.tree.tag_configure("ready", foreground=OK_COLOR)
        self.tree.tag_configure("missing", foreground=NG_COLOR)
        self.tree.pack(fill="x", pady=6)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self.show_missing())
        ttk.Label(tab, text="Missing files").pack(anchor="w")
        self.missing_text = tk.Text(tab, height=8, wrap="none", font=("Consolas", 9), background="#fafafa")
        self.missing_text.pack(fill="both", expand=True, pady=(2, 6))
        buttons = ttk.Frame(tab)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Check again", command=lambda: self.fill_combinations(self.on_refresh())).pack(side="right")
        return tab

    def fill_combinations(self, statuses):
        self.items = {}
        self.tree.delete(*self.tree.get_children())
        for step, entries in statuses.items():
            for status in entries:
                selection = status.selection
                item = self.tree.insert("", "end", tags=("ready" if status.ready else "missing",), values=(
                    step, selection.name + (" *" if status.is_default else ""),
                    ", ".join(TOOL_LABELS[t] for t in selection.tools),
                    "✓ Ready" if status.ready else f"✗ {len(status.missing)} missing"))
                self.items[item] = status
        first = next(iter(self.items), None)
        if first:
            self.tree.selection_set(first)
        self.show_missing()

    def show_missing(self):
        self.missing_text.delete("1.0", "end")
        selected = self.tree.selection()
        if selected:
            status = self.items[selected[0]]
            self.missing_text.insert("1.0", "\n".join(status.missing) if status.missing else "Nothing is missing.")

    # downloads ----------------------------------------------------------------------

    def _downloads_tab(self, parent):
        tab = ttk.Frame(parent, padding=6)
        ttk.Label(tab, foreground=MUTED, text="Programs come from their official releases; models and libraries "
                  "from the MS-Emblator Zenodo record. The addresses are in config/downloads.yaml.").pack(anchor="w")
        frame = ttk.Frame(tab)
        frame.pack(fill="both", expand=True, pady=6)
        self.download_tree = ttk.Treeview(frame, columns=("title", "group", "size", "status"), show="headings",
                                          selectmode="extended", height=11)
        for column, title, width, anchor in (("title", "Item", 430, "w"), ("group", "Kind", 90, "w"),
                                             ("size", "Size", 90, "e"), ("status", "Status", 140, "w")):
            self.download_tree.heading(column, text=title)
            self.download_tree.column(column, width=width, anchor=anchor)
        for tag, color in (("ok", OK_COLOR), ("update", "#b26a00"), ("missing", NG_COLOR), ("muted", MUTED)):
            self.download_tree.tag_configure(tag, foreground=color)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.download_tree.yview)
        self.download_tree.configure(yscrollcommand=scroll.set)
        self.download_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.download_tree.bind("<<TreeviewSelect>>", lambda _e: self.show_asset())
        self.asset_info = ttk.Label(tab, foreground=MUTED, wraplength=880, justify="left")
        self.asset_info.pack(fill="x")
        progress_row = ttk.Frame(tab)
        progress_row.pack(fill="x", pady=(6, 4))
        self.download_progress = ttk.Progressbar(progress_row, mode="determinate", length=320)
        self.download_progress.pack(side="left")
        self.download_text = tk.StringVar(value="")
        ttk.Label(progress_row, textvariable=self.download_text).pack(side="left", padx=8)
        buttons = ttk.Frame(tab)
        buttons.pack(fill="x")
        self.download_button = ttk.Button(buttons, text="⬇  Download selected", command=self.start_download)
        self.download_button.pack(side="left")
        ttk.Button(buttons, text="Select missing", command=self.select_missing).pack(side="left", padx=6)
        self.cancel_button = ttk.Button(buttons, text="Cancel", command=self.cancel_download)
        self.cancel_button.pack(side="left")
        self.cancel_button.state(["disabled"])
        ttk.Separator(buttons, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(buttons, text="Set up FIDDLE…", command=self.open_fiddle_setup).pack(side="left")
        self.fiddle_state = ttk.Label(buttons)
        self.fiddle_state.pack(side="left", padx=6)
        self.show_fiddle_state()
        ttk.Button(buttons, text="Close", command=self.close).pack(side="right")
        ttk.Button(buttons, text="Reload list", command=self.fill_downloads).pack(side="right", padx=6)
        ttk.Button(buttons, text="Edit download addresses…",
                   command=lambda: open_in_editor(downloads.MANIFEST_FILE)).pack(side="right")
        return tab

    def show_fiddle_state(self):
        from msemblator import fiddle_setup

        current = fiddle_setup.status(ask_conda=False)
        self.fiddle_state.configure(text="FIDDLE ready" if current.ready else "FIDDLE not set up",
                                    foreground=OK_COLOR if current.ready else NG_COLOR)

    def open_fiddle_setup(self):
        def done():
            self.show_fiddle_state()
            self.fill_combinations(self.on_refresh())

        FiddleSetupDialog(self, on_done=done)

    def fill_downloads(self):
        self.download_tree.delete(*self.download_tree.get_children())
        try:
            self.assets = {a.id: a for a in downloads.load_manifest()}
        except Exception as exc:
            self.assets = {}
            self.asset_info.configure(text=f"Cannot read {downloads.MANIFEST_FILE}: {exc}", foreground=NG_COLOR)
            return
        state = downloads.load_state()
        self.asset_status = {}
        for asset in self.assets.values():
            status = downloads.asset_status(asset, state)
            self.asset_status[asset.id] = status
            size = f"{asset.size_mb:,.0f} MB" if asset.size_mb else ""
            self.download_tree.insert("", "end", iid=asset.id, tags=(STATUS_TAGS[status],),
                                      values=(asset.title, asset.group, size, status))

    def show_asset(self):
        selected = self.download_tree.selection()
        if len(selected) != 1:
            self.asset_info.configure(text=f"{len(selected)} items selected." if selected else "")
            return
        asset = self.assets[selected[0]]
        lines = [f"From: {asset.url or '(no address yet — add it in config/downloads.yaml)'}",
                 f"To: {asset.destination}", f"License: {asset.license or '-'}"]
        if asset.note:
            lines.append(f"Note: {asset.note}")
        self.asset_info.configure(text="\n".join(lines), foreground=MUTED)

    def select_missing(self):
        wanted = [i for i, s in self.asset_status.items() if s in (downloads.MISSING, downloads.UPDATE)]
        self.download_tree.selection_set(wanted)

    def start_download(self):
        if self.cancel is not None:
            return
        chosen = [self.assets[i] for i in self.download_tree.selection()]
        unpublished = [a.title for a in chosen if not a.url]
        chosen = [a for a in chosen if a.url]
        if unpublished:
            messagebox.showinfo("Downloads", "No download address yet for:\n" + "\n".join(unpublished)
                                + "\n\nAdd it with 'Edit download addresses…'.", parent=self)
        if not chosen:
            return
        installed = [a.title for a in chosen if self.asset_status[a.id] == downloads.INSTALLED]
        if installed and not messagebox.askyesno(
                "Downloads", "Already installed (the files are replaced):\n" + "\n".join(installed) + "\n\nContinue?",
                parent=self):
            return
        consent = ConsentDialog(self, chosen)
        self.wait_window(consent)
        if not consent.accepted:
            return
        self.cancel = threading.Event()
        self.download_button.state(["disabled"])
        self.cancel_button.state(["!disabled"])
        threading.Thread(target=self._download_worker, args=(chosen, self.cancel), daemon=True).start()

    def _download_worker(self, assets, cancel):
        failures = []
        for number, asset in enumerate(assets, 1):
            prefix = f"[{number}/{len(assets)}] {asset.title}"

            def progress(done, total, name, prefix=prefix):
                self.messages.put(("progress", prefix, done, total, name))

            self.messages.put(("progress", prefix, 0, None, "connecting"))
            try:
                downloads.install(asset, progress, cancel)
            except downloads.DownloadCancelled:
                failures.append(f"{asset.title}: cancelled")
                break
            except Exception as exc:
                failures.append(f"{asset.title}: {exc}")
        self.messages.put(("done", failures))

    def cancel_download(self):
        if self.cancel is not None:
            self.cancel.set()
            self.download_text.set("Cancelling…")

    def _poll(self):
        try:
            while True:
                message = self.messages.get_nowait()
                if message[0] == "progress":
                    _, prefix, done, total, name = message
                    if total:
                        self.download_progress.configure(mode="determinate", maximum=total, value=done)
                        self.download_text.set(f"{prefix}: {done / 1e6:,.1f} / {total / 1e6:,.1f} MB")
                    else:
                        self.download_progress.configure(mode="determinate", maximum=1, value=0)
                        self.download_text.set(f"{prefix}: {name} {done / 1e6:,.1f} MB" if done else f"{prefix}: {name}")
                else:
                    failures = message[1]
                    self.cancel = None
                    self.download_button.state(["!disabled"])
                    self.cancel_button.state(["disabled"])
                    self.download_text.set("Finished" if not failures else "Finished with problems")
                    self.fill_downloads()
                    self.fill_combinations(self.on_refresh())
                    if failures:
                        messagebox.showwarning("Downloads", "\n\n".join(failures), parent=self)
                    else:
                        messagebox.showinfo("Downloads", "Download finished.", parent=self)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(150, self._poll)

    def close(self):
        if self.cancel is not None:
            if not messagebox.askyesno("Downloads", "A download is running. Cancel it and close?", parent=self):
                return
            self.cancel.set()
        self.destroy()
