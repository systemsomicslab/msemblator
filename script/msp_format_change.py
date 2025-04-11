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

                # Start a new compound
                temp_entry = [line]
                has_valid_formula = False  # Reset for new entry
                current_name = line.split(":", 1)[1].strip()  # Extract the compound name
                print(f"Processing compound: {current_name}")  # Debug: Show the name being processed

                # Check if the compound exists in the dictionary
                if current_name in rename and pd.notna(rename[current_name]):  # Ensure it's not NaN
                    add_entry = True
                else:
                    add_entry = False

            elif current_name and add_entry:
                # Modify the FORMULA field if necessary
                if line.startswith("FORMULA:"):
                    new_formula = rename.get(current_name, "").strip()
                    if new_formula and new_formula.lower() != "nan":  # Ensure FORMULA is valid
                        line = f"FORMULA: {new_formula}\n"
                        has_valid_formula = True  # Mark that the entry has a valid FORMULA
                    else:
                        has_valid_formula = False  # Ensure we skip this entry if FORMULA is invalid

                temp_entry.append(line)

        # Write the last entry if applicable and contains a valid FORMULA
        if add_entry and temp_entry and has_valid_formula:
            output_msp.writelines(temp_entry)
        elif not has_valid_formula:
            print(f"Skipping entire entry: {current_name} due to missing or 'nan' FORMULA.")  

    print("Processing complete. Check the output file for results.")


def convert_name_to_peakid(msp_data):
    """
    Replace the NAME field with sequential numbers in each spectrum block.
    Optionally adds |ORIGNAME=...| to COMMENT line.
    """
    blocks = re.split(r'\n\s*\n', msp_data.strip())
    updated_blocks = []
    records = []

    for i, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        original_name = None
        new_lines = []
        comment_updated = False

        for line in lines:
            if line.lower().startswith("name:"):
                original_name = line.split(":", 1)[1].strip()
                new_lines.append(f"NAME: {i}")
                continue

            if line.lower().startswith("comment:"):
                comment = line
                # Remove existing ORIGNAME if any
                comment = re.sub(r'\|ORIGNAME=[^|]+\|', '', comment)
                comment = comment.rstrip('|')  # avoid trailing |
                if original_name:
                    comment += f"|ORIGNAME={original_name}|"
                new_lines.append(comment)
                comment_updated = True
                continue

            new_lines.append(line)

        # Add a COMMENT if not already present
        if not comment_updated and original_name:
            new_lines.append(f"COMMENT: |ORIGNAME={original_name}|")

        updated_blocks.append("\n".join(new_lines))
        records.append({
            "Original_NAME": original_name,
            "Updated_NAME": str(i)
        })

    updated_msp_data = "\n\n".join(updated_blocks)
    df = pd.DataFrame(records)
    return updated_msp_data, df


def read_msp_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    

def save_updated_msp(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(data)

