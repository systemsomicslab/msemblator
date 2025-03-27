# Msemblator: A reliable annotation tool for metabolomics data
## Overview
Msemblator is a metabolomics annotation tool that integrates results from multiple in-silico annotation tools and applies ensemble learning-based scoring to provide highly reliable annotations.

Users can input **MSP files generated by MS-DIAL** and perform **formula prediction, structure prediction, or both**.

# Usage examples
## 1. Formula elucidation only
If you only want to perform formula prediction (no SIRIUS account required):

``` PowerShell
python main.py \
  --input data/example.msp \
  --output results/formula_only \
  --mode 1
```

## 2. Formula + structure elucidation (Recommended)
Run **both formula and structure prediction**, including ensemble scoring. Requires SIRIUS credentials:

``` PowerShell
python main.py \
  --input data/example.msp \
  --output results/formula_and_structure \
  --mode 2 \
  --sirius_user your_email@example.com \
  --sirius_pass your_password
```

## 3. Structure elucidation only
If your MSP file already contains predicted formulas and you want to perform **only structure annotation**. Requires SIRIUS credentials:

``` PowerShell
python main.py \
  --input data/formula_predicted.msp \
  --output results/structure_only \
  --mode 3 \
  --sirius_user your_email@example.com \
  --sirius_pass your_password
```

### Notes
・ `--input`: Path to the input MSP file (recommended: exported from MS-DIAL)
・ `--output`: Output folder to save results
・ `--mode`: 
   ・ 1 = Formula elucidation only
   ・ 2 = Both formula and structure elucidation
   ・ 3 = Structure elucidation only
・ --sirius_user and --sirius_pass : Required only for modes 2 and 3

## Input file preparation
Msemblator does not support raw data as input. Instead, **MSP files processed with MS-DIAL 5** are strongly recommended. The application utilizes MS-DIAL's MSP output to perform **formula and structure predictions**.

・ Formula prediction requires at least **m/z, MS2, Peak ID, and adduct type**.
・ **Structure prediction** requires the above information plus a **predicted molecular formula**.
・ If formula information is missing, structure prediction will be skipped. However, this limitation can be resolved by performing both formula and structure predictions simultaneously.

## Output files
Msemblator generates two output files:
1. A file containing **the top 3 predictions** from each annotation tool for both formula and structure, along with **the highest-ranked annotation based on the ensemble scoring model**.
2. A file summarizing **the top 3 ranked predictions from the scoring model**.

This structured approach ensures that users obtain high-confidence annotations from their metabolomics data.

## Environment setup
### 1. Python version:
This tool has been tested with:
```
Python 3.12+
```

Check your version:
``` PowerShell
python --version
```

### 2. Install required python modules
Install required Python packages using pip:
``` PowerShell
cd msemblator 
pip install -r requirements.txt
```

### 3. External Tool Placement
Please download the required external tools and place them in the following recommended structure:
<pre><code>```text D:. ├─ buddy/ # Output folder for msbuddy ├─ data/ # Input files and example datasets ├─ machine/ # Structure scoring models ├─ metfrag/ # MetFrag integration │ ├─ example_parameter.txt # Configuration file for MetFrag │ ├─ library_psv_v2.txt # Custom database for MetFrag │ └─ MetFragCommandLine-2.5.0.jar # MetFrag CLI JAR ├─ model/ # Formula scoring models ├─ msfinder/ # MS-FINDER integration │ ├─ MSFINDER ver 3.61/ # MS-FINDER executable and resources │ ├─ msp/ # Input MSP files for MS-FINDER │ ├─ output/ # MS-FINDER output (structures) │ └─ output_formula/ # MS-FINDER output (formulas only) ├─ save_folder/ # Intermediate files during execution │ ├─ buddy_mgf/ # MGF files for BUDDY input │ ├─ formula_fixed_msp/ # MSP files with updated formula information │ ├─ msfinder_msp/ # Reformatted MSP files for MS-FINDER │ └─ sirius_ms/ # Reformatted files for SIRIUS input ├─ sirius4/ # SIRIUS v4+ installation folder │ ├─ app/ │ ├─ database/ │ ├─ ExplorerLicTester/ │ ├─ ms/ │ ├─ output/ │ ├─ ... (other resources) │ ├─ sirius.exe # Main SIRIUS executable (CLI) │ └─ sirius-gui.exe # GUI version of SIRIUS (optional) └─ __pycache__/ # Python cache (auto-generated) ``` </code></pre>



**・MS-DIAL(for MSP generation)**
Download:[MS-DIAL5](https://systemsomicslab.github.io/compms/msdial/main.html)
Export your data as .msp using MS-DIAL 5.
These MSP files will be used as input for this tool.

**・ SIRIUS (required for formula and structure elucidation)**
Download:[SIRIUS 5.8.6](https://github.com/sirius-ms/sirius/releases/tag/v5.8.6)
You will need:
・ The SIRIUS executable
・ A SIRIUS web service account

**・ MS-FINDER (required for formula and structure elucidation)**
Download:[MSFINDER3.61](https://github.com/systemsomicslab/MsdialWorkbench/releases/tag/MSFINDER-v3.61)

**・ MetFrag (required for structure elucidation)**
Download:[MetFragCommandLine-2.5.0.jar](https://github.com/ipb-halle/MetFragRelaunched/releases/tag/v2.5.0)

**・ Required compound library and scoring model**
Download:[structure_scoring_model](https://github.com/ipb-halle/MetFragRelaunched/releases/tag/v2.5.0)
Download:[formula_scoring_model](https://github.com/ipb-halle/MetFragRelaunched/releases/tag/v2.5.0)
Download:[sirius_database](https://github.com/ipb-halle/MetFragRelaunched/releases/tag/v2.5.0)
Download:[msfinder_database](https://github.com/ipb-halle/MetFragRelaunched/releases/tag/v2.5.0)
Download:[MetFrag_database](https://github.com/ipb-halle/MetFragRelaunched/releases/tag/v2.5.0)









