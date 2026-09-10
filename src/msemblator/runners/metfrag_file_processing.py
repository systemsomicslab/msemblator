import os
import csv
import logging
from tqdm import tqdm
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from msemblator.chemistry.chem_data import formula_to_dict, calc_exact_mass

logging.basicConfig(level=logging.ERROR)

# Cache formula mass calculations to avoid redundant work
@lru_cache(maxsize=None)
def safe_calc_exact_mass(formula):
    try:
        elements = formula_to_dict(formula)
        return calc_exact_mass(elements)
    except Exception as e:
        logging.error(f"Error calculating exact mass for formula {formula}: {e}")
        return None
    
def filtering_library_by_formula_index(library_index, target_formula):
    headers, index = library_index
    return [headers] + index.get(target_formula, [])

def load_library(library_path, target_formulas=None):
    """Index library rows, optionally retaining only requested formulas."""
    with open(library_path, "r") as f:
        reader = csv.reader(f, delimiter="|")
        headers = next(reader)
        formula_idx = headers.index("MolecularFormula") 

        index = defaultdict(list)
        for row in reader:
            if not row:
                continue
            formula = row[formula_idx]
            if target_formulas is None or formula in target_formulas:
                index[formula].append(row)

    return headers, index


def process_spectrum(spectrum, parameter_file, output_dir, library, params=None):
    """Process one spectrum: write peak list, filtered library, and parameter file."""
    try:
        # Write peak list file
        if "PeakListPath" in spectrum and "m/z" in spectrum:
            peak_list_file = os.path.join(output_dir, f"{spectrum['PeakListPath']}_peaklist.txt")
            with open(peak_list_file, "w") as f:
                f.write("\n".join(spectrum["m/z"]))

        # Write filtered library
        if "FORMULA" in spectrum:
            filtered = filtering_library_by_formula_index(library, spectrum.get("FORMULA"))
            library_file = os.path.join(output_dir, f"{spectrum['PeakListPath']}_library.txt")
            with open(library_file, "w") as f:
                writer = csv.writer(f, delimiter="|")
                writer.writerows(filtered)

        # Write parameter file
        if params is None:
            with open(parameter_file, "r") as f:
                params = f.readlines()

        param_output_file = os.path.join(output_dir, f"parameter_{spectrum['PeakListPath']}.txt")
        with open(param_output_file, "w") as f:
            for line in params:
                lower = line.lower()
                if lower.startswith("neutralprecursormolecularformula"):
                    line = f"NeutralPrecursorMolecularFormula = {spectrum.get('FORMULA', '')}\n"
                elif lower.startswith("neutralprecursormass"):
                    line = f"NeutralPrecursorMass = {spectrum.get('NeutralPrecursorMass', '')}\n"
                elif lower.startswith("precursorionmode"):
                    line = f"PrecursorIonMode = {spectrum['PrecursorIonMode']}\n"
                elif lower.startswith("ispositiveionmode"):
                    line = f"IsPositiveIonMode = {spectrum['IsPositiveIonMode']}\n"
                elif lower.startswith("peaklistpath"):
                    line = f"PeakListPath = {spectrum['PeakListPath']}_peaklist.txt\n"
                elif line.startswith("SampleName"):
                    line = f"SampleName = {spectrum['PeakListPath']}\n"
                elif line.startswith("LocalDatabasePath"):
                    line = f"LocalDatabasePath = {spectrum['PeakListPath']}_library.txt\n"
                f.write(line)

    except Exception as e:
        logging.error(f"Error processing spectrum {spectrum.get('PeakListPath', 'Unknown')}: {e}")


# Keep the existing tuple-based entry point available.
def process_wrapper(args):
    spectrum, parameter_file, output_dir, library = args
    return process_spectrum(spectrum, parameter_file, output_dir, library)


def creat_metfrag_file(msp_file, parameter_file, output_dir, library_path, *, max_workers=4):
    """Generate files with a shared library and template; tune I/O via max_workers."""
    if not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    spectra = []
    spectrum = {}
    is_in_peaks = False

    # Parse MSP file line by line 
    with open(msp_file, "r") as f:
        for line in tqdm(f, desc="Reading MSP file lines", unit="line"):
            stripped_line = line.strip().lower()
            if not stripped_line:
                if spectrum:
                    spectra.append(spectrum)
                    spectrum = {}
                    is_in_peaks = False
            elif "name:" in stripped_line:
                spectrum["PeakListPath"] = line.split(":", 1)[1].strip()
            elif "precursormz:" in stripped_line:
                spectrum["PRECURSORMZ"] = line.split(":", 1)[1].strip()
            elif "precursortype:" in stripped_line:
                adduct = line.split(":", 1)[1].strip()
                spectrum["ADDUCT"] = adduct
                spectrum["PrecursorIonMode"] = {"[M+H]+": "1", "[M-H]-": "-1"}.get(adduct, "1")
                spectrum["IsPositiveIonMode"] = "True" if "+" in adduct else "False"
            elif "formula:" in stripped_line:
                formula = line.split(":", 1)[1].strip()
                spectrum["FORMULA"] = formula
                spectrum["NeutralPrecursorMass"] = safe_calc_exact_mass(formula)
            elif "num peaks:" in stripped_line:
                is_in_peaks = True
            elif is_in_peaks:
                spectrum.setdefault("m/z", []).append(line.strip())
        if spectrum:
            spectra.append(spectrum)

    target_formulas = {s["FORMULA"] for s in spectra if "FORMULA" in s}
    library = load_library(library_path, target_formulas)
    with open(parameter_file, "r") as f:
        params = tuple(f.readlines())
    os.makedirs(output_dir, exist_ok=True)

    def write_spectrum(spectrum):
        process_spectrum(spectrum, parameter_file, output_dir, library, params)

    # Threads share the index instead of serializing it for every spectrum.
    # Bound pending work for Python versions where map has no buffersize option.
    with tqdm(total=len(spectra), desc="Processing spectra", unit="spectrum") as progress:
        if max_workers == 1:
            for spectrum in spectra:
                write_spectrum(spectrum)
                progress.update()
        else:
            batch_size = max_workers * 16
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for start in range(0, len(spectra), batch_size):
                    for _ in executor.map(write_spectrum, spectra[start:start + batch_size]):
                        progress.update()
