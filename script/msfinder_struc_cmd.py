import subprocess
import os
import sys
from convert_struc_data_type import modify_msfinder_config_in_place
from pathlib import Path
import re

def run_msfinder(msfinder_directory, input_path, output_path, method_path, library_path, config):

    # Modify config before running
    modify_msfinder_config_in_place(method_path, library_path, config)

    # Full path to MSFinder executable
    msfinder_exe = os.path.join(msfinder_directory, "MsfinderConsoleApp.exe")

    if not os.path.exists(msfinder_exe):
        print(f"Error: Executable not found at {msfinder_exe}")
        return

    # Prepare command
    command = [
        msfinder_exe,
        "predict",
        "-i", input_path,
        "-o", output_path,
        "-m", method_path
    ]

    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end="")
            proc.wait()

    except Exception as e:
        print(f"An error occurred during MS-FINDER execution: {e}")

def extract_formulas_from_msp(msp_path):
    msp_path = Path(msp_path)
    txt = msp_path.read_text(encoding="utf-8", errors="ignore").strip()
    entries = re.split(r"\r?\n\r?\n+", txt) if txt else []
    formulas = set()

    for ent in entries:
        m = re.search(r"(?m)^FORMULA:\s*(.+?)\s*$", ent)
        if m:
            formulas.add(m.group(1).strip())

    return formulas

def parse_fgt_records(fgt_path):
    fgt_path = Path(fgt_path)
    txt = fgt_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"(?m)^NAME:", txt)
    if not m:
        return "", []  

    header = txt[:m.start()].strip("\r\n")
    body = txt[m.start():]

    starts = [m.start() for m in re.finditer(r"(?m)^NAME:", body)]
    records = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(body)
        rec = body[s:e].strip("\r\n")
        name_val = rec.splitlines()[0].split("NAME:", 1)[1].strip()
        records.append((name_val, rec))

    return header, records

def process_folder(folder):
    folder = Path(folder)
    for msp_path in folder.glob("*.msp"):
        fgt_path = msp_path.with_suffix(".fgt")
        if not fgt_path.exists():
            print(f"[SKIP] FGT not found: {fgt_path.name}")
            continue

        formulas = extract_formulas_from_msp(msp_path)
        header, records = parse_fgt_records(fgt_path)

        matched_blocks = [rec for name, rec in records if name in formulas]

        out_path = fgt_path
        with out_path.open("w", encoding="utf-8", newline="\n") as f:
            if header:
                f.write(header + "\n")
            if matched_blocks:
                f.write("\n\n".join(matched_blocks) + "\n")
            else:
                f.write(f"\n# No matching NAME blocks found for formulas: {', '.join(sorted(formulas))}\n")