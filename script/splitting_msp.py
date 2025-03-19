import re

def sanitize_filename(name, max_length=150):
    """
    Sanitize a file name by replacing invalid characters and truncating if necessary.

    Args:
        name (str): Original file name.
        max_length (int): Maximum allowed length for the file name.

    Returns:
        str: Sanitized file name.
    """
    sanitized = re.sub(r'[\/:*?"<>|]', '_', name)
    return sanitized[:max_length]

def format_msp_entry(name, content):
    """
    Format the MSP entry to match the desired structure and fill missing values.

    Args:
        name (str): Compound name.
        content (list of str): List of lines in the compound section.

    Returns:
        str: Formatted MSP entry as a string.
    """
    required_fields = {
        "PRECURSORMZ": "",
        "PRECURSORTYPE": "",
        "RETENTIONTIME": "",
        "FORMULA": "",
        "SMILES": "",
        "INCHIKEY": "",
        "COLLISIONENERGY": "",
        "IONMODE": "",
        "COMMENT": ""
    }

    formatted_entry = []
    formatted_entry.append(f"NAME: {name}")

    peaks = []
    for line in content:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().upper()
            value = value.strip()
            formatted_entry.append(f"{key}: {value}")
            if key in required_fields:
                required_fields[key] = value  # Override default values if present
        elif line.strip():  # Handling peak data
            peaks.append(line.strip())

    # Ensure all required fields are included
    for key, default_value in required_fields.items():
        if not any(key in entry for entry in formatted_entry):
            formatted_entry.append(f"{key}: {default_value}")

    # Adding peaks section
    if peaks:
        formatted_entry.append("Num Peaks: " + str(len(peaks)))
        formatted_entry.extend(peaks)

    return "\n".join(formatted_entry)

def read_msp(file_path):
    """
    Read and process an MSP file, formatting it according to the specified structure.

    Args:
        file_path (str): Path to the input MSP file.

    Returns:
        dict: A dictionary where keys are sanitized file names and values are the formatted MSP content.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    split_files = {}
    processed_lines = []
    current_name = None
    is_in_peaks = False  # Flag to track if the parser is within a compound's section

    for line in lines:
        stripped_line = line.strip()

        if stripped_line.casefold().startswith("name:"):
            # Save the previous compound's data if available
            if current_name and processed_lines:
                filename = sanitize_filename(current_name)
                split_files[filename] = format_msp_entry(current_name, processed_lines)

            # Start a new compound section
            current_name = stripped_line.split(":", 1)[1].strip()
            processed_lines = [line]
            is_in_peaks = True  # Mark that we're processing a compound section
        elif is_in_peaks:
            # Add lines within the compound section
            processed_lines.append(line)

    # Save the last compound's data
    if processed_lines and current_name:
        filename = sanitize_filename(current_name)
        split_files[filename] = format_msp_entry(current_name, processed_lines)

    return split_files
