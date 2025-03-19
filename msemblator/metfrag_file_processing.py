import os
import csv
import logging
from tqdm import tqdm
from chem_data import formula_to_dict, calc_exact_mass

# Setup logging configuration
logging.basicConfig(level=logging.ERROR)

def creat_metfrag_file(msp_file, parameter_file, output_dir, library_path):
    """
    Process an MSP file to create necessary MetFrag files (peak list, parameter files, and filtered library).

    Args:
        msp_file (str): Path to the input MSP file.
        parameter_file (str): Path to the MetFrag parameter template file.
        output_dir (str): Directory for output files.
        library_path (str): Path to the library file for filtering.

    Returns:
        None
    """
    # Read MSP file
    with open(msp_file, 'r') as file:
        lines = file.readlines()

    spectra = []
    spectrum = {}
    is_in_peaks = False

    # Parse the MSP file
    with tqdm(total=len(lines), desc="Reading MSP file lines", unit="line") as pbar:
        for line in lines:
            stripped_line = line.strip().lower()

            if not stripped_line:  # Empty line indicates the end of a spectrum
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
                try:
                    elements = formula_to_dict(formula)
                    spectrum["NeutralPrecursorMass"] = calc_exact_mass(elements)
                except Exception as e:
                    logging.error(f"Error calculating exact mass for formula {formula}: {e}")
            elif "num peaks:" in stripped_line:
                is_in_peaks = True
            elif is_in_peaks:
                spectrum.setdefault("m/z", []).append(line.strip())

            pbar.update(1)

        if spectrum:  # Add the last spectrum
            spectra.append(spectrum)

    # Write peak lists, parameter files, and filtered libraries
    with tqdm(total=len(spectra), desc="Creating library file", unit="spectrum") as pbar:
        for spectrum in spectra:
            try:
                # Write peak list file
                if "PeakListPath" in spectrum and "m/z" in spectrum:
                    peak_list_file = os.path.join(output_dir, f"{spectrum['PeakListPath']}_peaklist.txt")
                    with open(peak_list_file, 'w') as peak_file:
                        peak_file.write("\n".join(spectrum["m/z"]))

                # Filter the library
                if "NeutralPrecursorMass" in spectrum:
                    filtered_library = filtering_library(library_path, spectrum["NeutralPrecursorMass"], tolerance=0.01)
                    library_file = os.path.join(output_dir, f"{spectrum['PeakListPath']}_library.txt")
                    with open(library_file, 'w') as lib_file:
                        writer = csv.writer(lib_file, delimiter="|")
                        writer.writerows(filtered_library)

                # Create and save parameter file
                with open(parameter_file, 'r') as param_file:
                    params = param_file.readlines()

                param_output_file = os.path.join(output_dir, f"parameter_{spectrum['PeakListPath']}.txt")
                with open(param_output_file, 'w') as param_file:
                    for line in params:
                        if line.lower().startswith("neutralprecursormolecularformula"):
                            line = f"NeutralPrecursorMolecularFormula = {spectrum.get('FORMULA', '')}\n"
                        elif line.lower().startswith("neutralprecursormass"):
                            line = f"NeutralPrecursorMass = {spectrum.get('NeutralPrecursorMass', '')}\n"
                        elif line.lower().startswith("precursorionmode"):
                            line = f"PrecursorIonMode = {spectrum['PrecursorIonMode']}\n"
                        elif line.lower().startswith("ispositiveionmode"):
                            line = f"IsPositiveIonMode = {spectrum['IsPositiveIonMode']}\n"
                        elif line.lower().startswith("peaklistpath"):
                            line = f"PeakListPath = {spectrum['PeakListPath']}_peaklist.txt\n"
                        elif line.startswith("SampleName"):
                            line = f"SampleName = {spectrum['PeakListPath']}\n"
                        elif line.startswith("LocalDatabasePath"):
                            line = f"LocalDatabasePath = {spectrum['PeakListPath']}_library.txt\n"
                        param_file.write(line)

                pbar.update(1)
            except Exception as e:
                logging.error(f"Error processing spectrum {spectrum.get('PeakListPath', 'Unknown')}: {e}")


def filtering_library(library_path, target_mass, tolerance=0.02):
    """
    Filter the library file to find rows matching the target mass within a given tolerance.

    Args:
        library_path (str): Path to the library file.
        target_mass (float): Target mass for filtering.
        tolerance (float): Allowed mass deviation.

    Returns:
        list: Matching rows from the library.
    """
    matching_rows = []
    with open(library_path, 'r') as file:
        reader = csv.reader(file, delimiter="|")
        headers = next(reader)
        mass_index = headers.index("MonoisotopicMass")

        for row in reader:
            try:
                mass = float(row[mass_index])
                if abs(mass - target_mass) <= tolerance:
                    matching_rows.append(row)
            except (ValueError, IndexError):
                continue

    return [headers] + matching_rows  # Include headers at the top
