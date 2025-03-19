import logging
import os

# Configure logging for error messages
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_msp_to_mgf(file_path, output_file_path=None):
    """
    Convert an MSP file to MGF format.
    
    Args:
        file_path (str): Path to the input MSP file.
        output_file_path (str, optional): Path to save the output MGF file. Defaults to None.
        
    Returns:
        str: MGF formatted content as a string.
    """
    try:
        # Ensure the file exists before reading
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input MSP file not found: {file_path}")

        with open(file_path, 'r') as file:
            lines = file.readlines()

        spectra = []
        spectrum = {}
        is_in_peaks = False
        for line in lines:
            try:
                line = line.strip()
                if line == '':
                    if spectrum:
                        spectra.append(spectrum)
                        spectrum = {}
                        is_in_peaks = False
                elif 'NAME:' in line.upper():
                    spectrum['TITLE'] = line.split(':', 1)[1].strip()
                elif 'PRECURSORMZ:' in line.upper():
                    spectrum['PEPMASS'] = line.split(':', 1)[1].strip()
                elif 'PRECURSORTYPE:' in line.upper():
                    value = line.split(':', 1)[1].strip()
                    spectrum['ADDUCT'] = value
                    spectrum['CHARGE'] = '1' if '+' in value else '1'
                    spectrum["IONMODE"] = "POSITIVE" if '+' in value else 'NEGATIVE'
                elif 'NUM PEAKS:' in line.upper():
                    is_in_peaks = True
                elif is_in_peaks:
                    spectrum.setdefault('m/z', []).append(line)
            except Exception as e:
                logging.error(f"Error processing spectrum line: {line} - {e}")
                # Continue to the next line in case of an error
                continue

        if spectrum:
            spectra.append(spectrum)

        mgf_data = []
        for spectrum in spectra:
            try:
                mgf_data.append("BEGIN IONS")
                if 'TITLE' in spectrum:
                    mgf_data.append(f"TITLE={spectrum['TITLE']}")
                if 'PEPMASS' in spectrum:
                    mgf_data.append(f"PEPMASS={spectrum['PEPMASS']}")
                if 'CHARGE' in spectrum:
                    mgf_data.append(f"CHARGE={spectrum['CHARGE']}")
                    mgf_data.append("MSLEVEL=2")
                    mgf_data.append(f"IONMODE={spectrum['IONMODE']}")
                if 'ADDUCT' in spectrum:
                    mgf_data.append(f"ADDUCT={spectrum['ADDUCT']}")
                mgf_data.extend(spectrum.get('m/z', []))
                mgf_data.append("END IONS\n")
            except Exception as e:
                logging.error(f"Error formatting spectrum: {spectrum.get('TITLE', 'Unknown Title')} - {e}")
                # Skip this spectrum if an error occurs
                continue

        mgf_content = "\n".join(mgf_data)

        if output_file_path:
            with open(output_file_path, 'w') as file:
                file.write(mgf_content)

        return mgf_content

    except FileNotFoundError as e:
        logging.error(e)
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while converting MSP to MGF: {e}")
        raise


def split_mgf_by_adduct_in_memory(mgf_content):
    """
    Split MGF content by adduct type.

    Args:
        mgf_content (str): MGF formatted content as a string.

    Returns:
        dict: A dictionary where keys are adduct types, and values are spectra lists.
    """
    try:
        if not mgf_content.strip():
            raise ValueError("The MGF content is empty or invalid.")

        lines = mgf_content.splitlines()
        adduct_data = {}
        current_adduct = None
        current_data = []
        current_title = None

        for line in lines:
            try:
                if line.startswith('BEGIN IONS'):
                    current_data = [line]
                elif line.startswith('END IONS'):
                    current_data.append(line)
                    if current_adduct:
                        adduct_data.setdefault(current_adduct, []).append("\n".join(current_data))
                    else:
                        logging.warning(f"Skipped spectrum: {current_title if current_title else 'Unknown Title'} (No ADDUCT found)")
                    current_adduct = None
                else:
                    current_data.append(line)
                    if line.startswith("ADDUCT="):
                        current_adduct = line.split('=')[1].strip()
                    if line.startswith("TITLE="):
                        current_title = line.split('=')[1].strip()
            except Exception as e:
                logging.error(f"Error processing line in MGF: {line} - {e}")
                # Continue to the next line if an error occurs
                continue

        return adduct_data

    except ValueError as e:
        logging.error(e)
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while splitting MGF by adduct: {e}")
        raise
