import pandas as pd
import re

def msp_formula_changer(input_msp_path, formula_summary, msp_output_path):
    formula_summary = formula_summary[["filename","formula"]]

    name_list_df = formula_summary.astype(str)

    # Remove NaN values from 'match_formula' and create a dictionary
    name_list_df = name_list_df.dropna(subset=["formula"])  # Remove rows where 'match_formula' is NaN
    rename = name_list_df.set_index("filename")["formula"].to_dict()

    # Debug: Print the keys in the rename dictionary
    print("Loaded compounds:", rename.keys())

    # Process the MSP file and replace names and related fields
    with open(input_msp_path, 'r') as msps:
        lines = msps.readlines()

    with open(msp_output_path, 'w') as output_msp:
        current_name = None  # To track the current compound name
        add_entry = False  # Flag to determine if the current compound should be added
        temp_entry = []  # Temporary storage for the current entry lines
        has_valid_formula = False  # Flag to check if a valid FORMULA exists

        for line in lines:
            if line.startswith("NAME:"):
                # If a new compound starts, check if the previous entry should be written
                if add_entry and temp_entry and has_valid_formula:
                    output_msp.writelines(temp_entry)  # Write the valid entry
                elif not has_valid_formula:
                    print(f"Skipping entire entry: {current_name} due to missing or 'nan' FORMULA.")  # Debug

                # Start a new compound
                temp_entry = [line]
                has_valid_formula = False  # Reset for new entry
                current_name = line.split(":", 1)[1].strip()  # Extract the compound name
                print(f"Processing compound: {current_name}")  # Debug: Show the name being processed

                # Check if the compound exists in the dictionary
                if current_name in rename and pd.notna(rename[current_name]):  # Ensure it's not NaN
                    add_entry = True
                    print(f"Compound {current_name} found. Checking FORMULA.")  # Debug
                else:
                    add_entry = False
                    print(f"Compound {current_name} not found or has NaN FORMULA. Skipping.")  # Debug

            elif current_name and add_entry:
                # Modify the FORMULA field if necessary
                if line.startswith("FORMULA:"):
                    new_formula = rename.get(current_name, "").strip()
                    if new_formula and new_formula.lower() != "nan":  # Ensure FORMULA is valid
                        line = f"FORMULA: {new_formula}\n"
                        has_valid_formula = True  # Mark that the entry has a valid FORMULA
                        print(f"Updated FORMULA for {current_name}: {new_formula}")  # Debug
                    else:
                        has_valid_formula = False  # Ensure we skip this entry if FORMULA is invalid
                        print(f"Skipping {current_name} due to missing or 'nan' FORMULA.")  # Debug

                temp_entry.append(line)

        # Write the last entry if applicable and contains a valid FORMULA
        if add_entry and temp_entry and has_valid_formula:
            output_msp.writelines(temp_entry)
        elif not has_valid_formula:
            print(f"Skipping entire entry: {current_name} due to missing or 'nan' FORMULA.")  # Debug

    print("Processing complete. Check the output file for results.")



def convert_name_to_peakid(msp_data):
    """
    Convert the NAME field to PEAKID for each spectrum block in the MSP data.
    If a COMMENT line contains '|PEAKID=...', the function extracts the PEAKID,
    replaces the NAME with the PEAKID, and appends the original NAME as an ORIGNAME tag
    in the COMMENT line.

    Parameters:
        msp_data (str): The original MSP data as a string.

    Returns:
        tuple:
            - str: The updated MSP data with NAME replaced by PEAKID.
            - pd.DataFrame: A DataFrame containing the original and updated NAMEs for each spectrum block.
              The DataFrame has the columns 'Original_NAME' and 'Updated_NAME'.
    """
    # Split the MSP data into blocks (each separated by one or more blank lines)
    blocks = re.split(r'\n\s*\n', msp_data.strip())
    updated_blocks = []
    records = []  # For storing the NAME pairs for each block
    
    for block in blocks:
        lines = block.splitlines()
        original_name = None
        new_name = None
        new_lines = []
        
        for line in lines:
            # Process the NAME line.
            if line.lower().startswith("name:"):
                original_name = line.split(":", 1)[1].strip()
                new_name = original_name  # default to original
                new_lines.append(line)
                continue
            
            # Process the COMMENT line.
            if line.lower().startswith("comment:"):
                comment_line = line
                if "|PEAKID=" in comment_line:
                    # Extract PEAKID using regex.
                    match = re.search(r'\|PEAKID=([^|]+)\|', comment_line)
                    if match:
                        peakid = match.group(1).strip()
                        new_name = peakid
                        # Append ORIGNAME tag if not already present.
                        if "|ORIGNAME=" not in comment_line and original_name:
                            comment_line += f"|ORIGNAME={original_name}|"
                new_lines.append(comment_line)
                continue
            
            new_lines.append(line)
        
        # Replace the NAME line with the new_name.
        if original_name is not None and new_name is not None:
            for i, line in enumerate(new_lines):
                if line.lower().startswith("name:"):
                    new_lines[i] = f"NAME: {new_name}"
                    break
        
        updated_block = "\n".join(new_lines)
        updated_blocks.append(updated_block)
        records.append({
            "Original_NAME": original_name,
            "Updated_NAME": new_name
        })
    
    updated_msp_data = "\n\n".join(updated_blocks)
    df = pd.DataFrame(records)

    return updated_msp_data, df


def revert_name_from_peakid(msp_data):
    """
    Revert the NAME field to the original value for each spectrum block in the MSP data.
    The function searches for the ORIGNAME tag in the COMMENT line, replaces the NAME field
    with the original name, and removes the ORIGNAME tag from the COMMENT.

    Parameters:
        msp_data (str): The MSP data (with NAME replaced by PEAKID) as a string.

    Returns:
        str: The MSP data with the original NAME restored.
    """
    blocks = re.split(r'\n\s*\n', msp_data.strip())
    reverted_blocks = []
    
    for block in blocks:
        lines = block.splitlines()
        original_name = None
        new_lines = []
        
        for line in lines:
            # Check the COMMENT line for the ORIGNAME tag.
            if line.lower().startswith("comment:"):
                match = re.search(r'\|ORIGNAME=([^|]+)\|', line)
                if match:
                    original_name = match.group(1).strip()
                    # Remove the ORIGNAME tag from the COMMENT line.
                    line = re.sub(r'\|ORIGNAME=[^|]+\|', '', line)
                new_lines.append(line)
                continue
            
            new_lines.append(line)
        
        # Replace the NAME line with the original_name if found.
        if original_name:
            for i, line in enumerate(new_lines):
                if line.lower().startswith("name:"):
                    new_lines[i] = f"NAME: {original_name}"
                    break
        
        reverted_blocks.append("\n".join(new_lines))
    
    return "\n\n".join(reverted_blocks)


def read_msp_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    

def save_updated_msp(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(data)

