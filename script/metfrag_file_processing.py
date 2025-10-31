import os
import csv
import logging
from tqdm import tqdm
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor

from chem_data import formula_to_dict, calc_exact_mass

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


def load_library(library_path):
    """Load the library once into memory and sort by MonoisotopicMass."""
    with open(library_path, "r") as f:
        reader = csv.reader(f, delimiter="|")
        headers = next(reader)
        mass_index = headers.index("MonoisotopicMass")
        rows = [(float(row[mass_index]), row) for row in reader if row]
    rows.sort(key=lambda x: x[0])  # sort by mass
    return headers, rows


def filtering_library_preloaded(library, target_mass, tolerance=0.02):
    """Filter preloaded library rows by mass range."""
    headers, rows = library
    lo, hi = target_mass - tolerance, target_mass + tolerance
    return [headers] + [row for mass, row in rows if lo <= mass <= hi]


def process_spectrum(spectrum, parameter_file, output_dir, library):
    """Process one spectrum: write peak list, filtered library, and parameter file."""
    try:
        # Write peak list file
        if "PeakListPath" in spectrum and "m/z" in spectrum:
            peak_list_file = os.path.join(output_dir, f"{spectrum['PeakListPath']}_peaklist.txt")
            with open(peak_list_file, "w") as f:
                f.write("\n".join(spectrum["m/z"]))

        # Write filtered library
        if "NeutralPrecursorMass" in spectrum:
            filtered = filtering_library_preloaded(library, spectrum["NeutralPrecursorMass"], tolerance=0.01)
            library_file = os.path.join(output_dir, f"{spectrum['PeakListPath']}_library.txt")
            with open(library_file, "w") as f:
                writer = csv.writer(f, delimiter="|")
                writer.writerows(filtered)

        # Write parameter file
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


# Wrapper for multiprocessing (must be top-level, not lambda)
def process_wrapper(args):
    spectrum, parameter_file, output_dir, library = args
    return process_spectrum(spectrum, parameter_file, output_dir, library)


def creat_metfrag_file(msp_file, parameter_file, output_dir, library_path):
    """Main function: parse MSP, load library once, and process spectra in parallel."""
    spectra = []
    spectrum = {}
    is_in_peaks = False

    # Parse MSP file line by line (memory efficient)
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

    # Load library once
    library = load_library(library_path)

    # Prepare arguments for parallel processing
    tasks = [(s, parameter_file, output_dir, library) for s in spectra]

    # Process spectra in parallel
    with ProcessPoolExecutor() as executor:
        list(tqdm(
            executor.map(process_wrapper, tasks),
            total=len(spectra),
            desc="Processing spectra",
            unit="spectrum"
        ))

