import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Function to read MSP file
def read_msp_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def read_msp_as_list(file_path):
    with open(file_path, "r") as file:
        return file.readlines()

# Function to extract compound and ionization information from MSP data
def extract_compound_and_ionization(msp_data):
    lines = msp_data.strip().splitlines()
    compound_ionization_list = []  # To store tuples of (compound, ionization)
    spectrum = {"compound": "", "ionization": ""}

    for line in lines:
        line = line.strip()
        if line == '':
            spectrum = {"compound": "", "ionization": ""}
            continue
        
        if line.casefold().startswith("name:"):
            spectrum["compound"] = line.split(':', 1)[1].strip()

        elif line.casefold().startswith("precursortype:"):
            spectrum["ionization"] = line.split(':', 1)[1].strip()
            # Append the (compound, ionization) pair to the list
            compound_ionization_list.append((spectrum["compound"], spectrum["ionization"]))

    return compound_ionization_list

def smiles_to_inchikeys(smiles_series, msfinder_library):
    # Load the library file
    librarys = pd.read_table(msfinder_library)
    
    # Ensure necessary columns exist
    if 'SMILES' not in librarys.columns or 'InChIkey' not in librarys.columns or 'Short InChIKey' not in librarys.columns:
        print("Error: Required columns ('SMILES', 'InChIkey', 'Short InChIKey') are not found in the library file.")
        return pd.DataFrame({'shortInChiKey': [None] * len(smiles_series), 'InChIKey': [None] * len(smiles_series)})
    
    # Create dictionary mappings for both Short InChIKey and Full InChIKey
    smiles_short_inchikey_dict = librarys.set_index(librarys['SMILES'].str.strip())['Short InChIKey'].to_dict()
    smiles_full_inchikey_dict = librarys.set_index(librarys['SMILES'].str.strip())['InChIkey'].to_dict()

    # Map SMILES values to both Short and Full InChIKeys
    short_inchikey_series = smiles_series.str.strip().map(smiles_short_inchikey_dict)
    full_inchikey_series = smiles_series.str.strip().map(smiles_full_inchikey_dict)
    
    # Return as DataFrame with both keys
    return pd.DataFrame({'shortInChiKey': short_inchikey_series, 'InChIKey': full_inchikey_series})

def modify_msfinder_config_in_place(method_path, librarypath):
    with open(method_path, 'r') as file:
        lines = file.readlines()
    
    with open(method_path, 'w') as file:
        for line in lines:
            if line.startswith("UserDefinedDbFilePath="):
                line = f"UserDefinedDbFilePath={librarypath}\n"
            file.write(line)

def convert_to_canonical_smiles(df, column_name, new_column_name="Canonical_SMILES"):
    """
    Converts SMILES in a specified column of a DataFrame to Canonical SMILES using RDKit.
    
    - If conversion fails, the original SMILES is retained.
    - NaN (missing values) are converted to None.
    - If RDKit cannot process a SMILES, a warning message is displayed, and the original SMILES is used.

    :param df: Pandas DataFrame containing a column with SMILES strings.
    :param column_name: Name of the column containing SMILES strings.
    :param new_column_name: Name of the new column for Canonical SMILES.
    :return: DataFrame with an additional column containing Canonical SMILES.
    """
    def safe_convert(smiles):
        try:
            if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
                return None  # Convert NaN or empty strings to None
            
            mol = Chem.MolFromSmiles(smiles)  # Generate a molecule using RDKit
            
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)  # Convert to Canonical SMILES
            else:
                return smiles  # If conversion fails, return the original SMILES
        except Exception as e:
            return smiles  # If an error occurs, return the original SMILES

    # Convert SMILES (use the original value if conversion fails)
    df[new_column_name] = df[column_name].apply(lambda x: safe_convert(x))
    return df

def normalize_rank(df):
    scaler=MinMaxScaler()
    df["normalized_rank"]= scaler.fit_transform(df[["rank"]])
    
def normalize_rank_n(df):
    scaler=MinMaxScaler()
    df["normalized_rank"]= 1 - scaler.fit_transform(df[["rank"]])

def normalize_rank_score(df):
    df["normalized_rank"] = 1 - np.log(df["rank"]) / (np.log(df["rank"].max()+1))

def smiles_list_to_inchikeys(smiles_list):
    """
    Convert a list of SMILES strings to their corresponding InChIKeys.
    """
    inchikeys = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            inchikeys.append(Chem.MolToInchiKey(mol))
        else:
            inchikeys.append(None)
    return inchikeys

def convert_to_shortinchikey(df, column_name, new_column_name="Short_InChIKey"):
    """
    Converts SMILES in a specified column of a DataFrame to Short InChIKeys using RDKit.
    
    """
    def safe_convert(smiles):
        try:
            if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
                return None
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                inchikey = inchi.MolToInchiKey(mol)
                return inchikey[:14]  # Short InChIKey (first 14 chars)
            else:
                return None
        except Exception as e:
            return None

    df[new_column_name] = df[column_name].apply(safe_convert)
    return df