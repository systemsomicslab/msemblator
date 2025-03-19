import logging

# Configure logging for error messages
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def read_msp_file(file_path):
    """
    Read the content of an MSP file.
    
    Args:
        file_path (str): Path to the MSP file.
        
    Returns:
        str: Content of the MSP file.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For other unexpected errors.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while reading the MSP file: {e}")
        raise

def convert_msp_to_ms(msp_data):
    """
    Convert MSP formatted data to MS formatted data.
    
    Args:
        msp_data (str): The MSP data as a string.
        
    Returns:
        str: MS formatted data.
        
    Raises:
        ValueError: If the input data is improperly formatted.
        Exception: For other unexpected errors.
    """
    try:
        lines = msp_data.strip().splitlines()
        ms_data = []
        spectrum = {}
        is_in_peaks = False

        for line in lines:
            line = line.strip()
            if line == '':
                # Handle the end of a spectrum
                if spectrum and "ms2" in spectrum:
                    try:
                        ms_data.append(format_ms(spectrum))
                    except Exception as e:
                        logging.error(f"Error formatting spectrum: {spectrum.get('compound', 'Unknown')} - {e}")
                    spectrum = {}
                is_in_peaks = False
                continue
            
            if line.casefold().startswith("name:"):
                # Handle a new spectrum
                if spectrum and "ms2" in spectrum:
                    try:
                        ms_data.append(format_ms(spectrum))
                    except Exception as e:
                        logging.error(f"Error formatting spectrum: {spectrum.get('compound', 'Unknown')} - {e}")
                    spectrum = {}
                spectrum = {
                    "compound": "",
                    "formula": "",
                    "parentmass": "",
                    "ionization": "",
                    "ms2": []
                }
                spectrum["compound"] = line.split(':', 1)[1].strip()
            elif line.casefold().startswith("formula:"):
                spectrum["formula"] = line.split(':', 1)[1].strip()
            elif line.casefold().startswith("precursormz:"):
                spectrum["parentmass"] = line.split(':', 1)[1].strip()
            elif line.casefold().startswith("precursortype:"):
                spectrum["ionization"] = line.split(':', 1)[1].strip()
                if spectrum["ionization"] == '[M+NH4]+':
                    spectrum["ionization"] = '[M+H3N+H]+'
            elif line.casefold().startswith("num peaks:"):
                is_in_peaks = True
            elif is_in_peaks:
                mz_intensity = line.split()
                if len(mz_intensity) == 2:
                    spectrum["ms2"].append(f"{mz_intensity[0]}\t{mz_intensity[1]}")

        # Process the last spectrum
        if spectrum and "ms2" in spectrum:
            try:
                ms_data.append(format_ms(spectrum))
            except Exception as e:
                logging.error(f"Error formatting spectrum: {spectrum.get('compound', 'Unknown')} - {e}")

        return "\n\n".join(ms_data)
    except Exception as e:
        logging.error(f"An unexpected error occurred during MSP to MS conversion: {e}")
        raise

def format_ms(spectrum):
    """
    Format a single spectrum in MS format.
    
    Args:
        spectrum (dict): A dictionary containing spectrum details.
        
    Returns:
        str: Formatted spectrum in MS format.
        
    Raises:
        Exception: If required fields are missing.
    """
    try:
        ms2_data = "\n".join(spectrum["ms2"])
        formatted = (
            f">compound {spectrum['compound']}\n"
            f">formula {spectrum['formula']}\n"
            f">parentmass {spectrum['parentmass']}\n"
            f">ionization {spectrum['ionization']}\n"
            f">ms2\n{ms2_data}"
        )
        return formatted
    except KeyError as e:
        raise Exception(f"Missing required field: {e}")

def convert_msp_file_to_ms(msp_file_path):

    try:
        # Read the MSP file content
        msp_data = read_msp_file(msp_file_path)
        
        # Convert MSP data to MS format
        ms_file_content = convert_msp_to_ms(msp_data)
        
        return ms_file_content
    except Exception as e:
        logging.error(f"An error occurred during MSP to MS conversion: {e}")
        raise
