import yaml
import os
import re

# Load atomic data from YAML file
current_directory = os.path.dirname(os.path.abspath(__file__))
atomic_data_path = os.path.join(current_directory, 'atomic_data.yaml')

try:
    with open(atomic_data_path, 'r') as f:
        atomic_data = yaml.safe_load(f)
except (FileNotFoundError, yaml.YAMLError) as e:
    raise RuntimeError(f"Error loading 'atomic_data.yaml': {e}")

# Prepare atomic mass data
atomic_to_exact_mass = {}
try:
    for element, data in atomic_data.items():
        data['data'].sort(key=lambda x: x["Composition"], reverse=True)
        atomic_to_exact_mass[data['isotope']] = data['data'][0]['ExactMass']
except KeyError as e:
    raise KeyError(f"Missing required key in atomic data: {e}")

def calc_exact_mass(elements):
    """Calculate exact mass based on element composition."""
    try:
        return sum(atomic_to_exact_mass[el] * count for el, count in elements.items())
    except KeyError as e:
        print(f"Warning: Element '{e.args[0]}' not found. Skipping.")
        return None

def read_aduct_type_data():
    """Load adduct type data from a YAML file."""
    aduct_data_path = os.path.join(current_directory, 'aduct_type_data.yaml')
    try:
        with open(aduct_data_path, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise RuntimeError(f"Error loading 'aduct_type_data.yaml': {e}")

def formula_to_dict(formula):
    """Convert a chemical formula string to a dictionary of elements and their counts."""
    try:
        return {el: int(count) if count else 1 for el, count in re.findall(r'([A-Z][a-z]?)(\d*)', formula)}
    except Exception as e:
        raise ValueError(f"Invalid formula '{formula}': {e}")

def dict_to_formula(element_counts):
    """Convert a dictionary of elements and counts to a chemical formula string."""
    formula = []
    if 'C' in element_counts:
        formula.append(f"C{element_counts['C']}" if element_counts['C'] > 1 else "C")
    if 'H' in element_counts:
        formula.append(f"H{element_counts['H']}" if element_counts['H'] > 1 else "H")
    for el in sorted(el for el in element_counts if el not in ['C', 'H']):
        formula.append(f"{el}{element_counts[el]}" if element_counts[el] > 1 else el)
    return ''.join(formula)

