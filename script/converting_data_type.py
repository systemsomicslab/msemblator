import pandas as pd
from rdkit import Chem
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import os

class ClippingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, q_low=None, q_high=None):
        self.q_low = q_low
        self.q_high = q_high
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.q_low is None or self.q_high is None:
            raise ValueError("q_low and q_high must be specified manually before using transform().")
        
        clipped = np.clip(X.to_numpy().flatten(), self.q_low, self.q_high)
        return clipped.reshape(-1, 1)


def normalize_rank(df):
    scaler=MinMaxScaler()
    df["normalized_rank"]= 1-scaler.fit_transform(df[["rank"]])
    

def read_msp_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

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

def generate_unique_filename(directory, filename):
    base_name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base_name}_{counter}{ext}"
        counter += 1
    return new_filename    

def extract_used_tooks(row):
    tools = []
    if row.get("tool_name_buddy", 0) == 1:
        tools.append(f"msbuddy(rank={int(row['rank'])})")
    if row.get("tool_name_msfinder", 0) == 1:
        tools.append(f"MS-FINDER(rank={int(row['rank'])})")
    if row.get("tool_name_sirius", 0) == 1:
        tools.append(f"SIRIUS(rank={int(row['rank'])})")
    return ", ".join(tools)