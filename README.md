# MS-Emblator 2026.7.23

MS-Emblator is a metabolomics annotation pipeline that integrates predictions
from multiple in-silico tools and applies ensemble-learning models to produce
reliable molecular formula and structure annotations.

The pipeline accepts spectra in MSP format, runs the appropriate combination of
SIRIUS, MS-FINDER, msbuddy, and MetFrag, and summarizes their candidates using
trained scoring models.

> [!IMPORTANT]
> MS-Emblator is currently designed for Windows. Raw mass spectrometry data
> cannot be used directly; prepare an MSP file before running the pipeline.

## Contents

- [Workflow](#workflow)
- [Requirements](#requirements)
- [Installation](#installation)
- [Runtime assets](#runtime-assets)
- [Preparing input data](#preparing-input-data)
- [Usage](#usage)
- [Output files](#output-files)
- [Parameter configuration](#parameter-configuration)
- [Troubleshooting](#troubleshooting)

## Workflow

MS-Emblator provides three analysis modes:

| Mode | Analysis | Integrated tools | Formula required in input | SIRIUS account |
| --- | --- | --- | --- | --- |
| `1` | Formula elucidation | SIRIUS, MS-FINDER, msbuddy | No | No |
| `2` | Formula and structure elucidation | SIRIUS, MS-FINDER, msbuddy, MetFrag | No | Yes |
| `3` | Structure elucidation only | SIRIUS, MS-FINDER, MetFrag | Yes | Yes |

Mode 2 is recommended when the input MSP file does not already contain reliable
molecular formula annotations.

## Requirements

### Operating system

- Windows 10 or Windows 11

### Python

- Python 3.10 or newer
- Python 3.12 is recommended

Check the installed version:

```powershell
python --version
```

### Java

MetFrag requires Java 21 or newer.

```powershell
java --version
```

### External software

MS-Emblator uses the following third-party tools:

- [MS-DIAL 5](https://systemsomicslab.github.io/compms/msdial/main.html)
  for preparing MSP input files
- [SIRIUS 6.3.2](https://github.com/sirius-ms/sirius/releases/tag/v6.3.2)
  for formula and structure prediction
- [MS-FINDER 3.61](https://github.com/systemsomicslab/MsdialWorkbench/releases/tag/MSFINDER-v3.61)
  for formula and structure prediction
- [MetFrag command line 2.5.0](https://github.com/ipb-halle/MetFragRelaunched/releases/tag/v2.5.0)
  for structure prediction

The external programs, compound libraries, and trained scoring models are not
stored in this Git repository. They are distributed separately through
[Zenodo](https://zenodo.org/records/17490421).

## Installation

Clone the repository and move into its directory:

```powershell
git clone https://github.com/systemsomicslab/msemblator.git
cd msemblator
```

Creating an isolated Python environment is recommended:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install MS-Emblator and its Python dependencies:

```powershell
python -m pip install --upgrade pip
pip install -e .
```

Confirm that the command is available:

```powershell
msemblator --help
```

If you do not want to install the command, the repository also provides a
Windows launcher:

```powershell
.\msemblator.cmd --help
```

## Runtime assets

Download the runtime asset archive from
[Zenodo](https://zenodo.org/records/17490421), extract it, and arrange the
repository as shown below. Keep the directory and file names unchanged.

```text
msemblator/
├─ config/
│  ├─ msemblator_parameter_file.yaml
│  ├─ metfrag/
│  │  └─ example_paramater.txt
│  └─ msfinder/
│     ├─ MsfinderConsoleApp_Param_formula.txt
│     ├─ MsfinderConsoleApp-Param2_structure.txt
│     └─ MsfinderConsoleApp-Param_all_processing.txt
├─ external/
│  ├─ metfrag/
│  │  └─ MetFragCommandLine-2.5.0.jar
│  ├─ msfinder/
│  │  ├─ MSFINDER ver 3.61/
│  │  │  └─ MsfinderConsoleApp.exe
│  │  ├─ MsfinderConsoleApp_Param_formula.txt
│  │  └─ MsfinderConsoleApp-Param2_structure.txt
│  └─ sirius/
│     └─ sirius.exe
├─ models/
│  ├─ formula/
│  │  └─ *.pkl
│  └─ structure/
│     └─ *.pkl
├─ src/
├─library/
│  ├─sirius_structure_db.siriusdb
│  ├─MsfinderStructureDB_all.txt
│  └─metfrag_StructureDB.txt
├─ msemblator.cmd
└─ pyproject.toml
```

The `external/`, `models/`, and generated `work/` directories are intentionally
excluded from Git because they contain large or machine-local files.

At startup, MS-Emblator checks for the assets required by the selected mode. A
`FileNotFoundError` listing one or more paths means that the Zenodo archive was
not extracted into the expected location.

## Preparing input data

MS-Emblator accepts an MSP file, not a raw instrument file. Exporting an MSP
file from MS-DIAL 5 is recommended.

Each spectrum should contain:

- a unique compound or feature name;
- precursor m/z;
- an adduct or ion type;
- MS/MS peaks;
- a molecular formula when using mode 3.

Formula information is not required for modes 1 and 2. In mode 2, MS-Emblator
first predicts formulas and then passes the top formula results to the structure
elucidation workflow.

Before processing a large dataset, test the installation with a small MSP file.
This makes missing executables, unsupported adducts, and formatting problems
easier to identify.

## Usage

General command syntax:

```powershell
msemblator --input <input.msp> --output <output-directory> --mode <1|2|3>
```

### Mode 1: formula elucidation only

Use mode 1 to predict molecular formulas without running structure annotation.
A SIRIUS account is not required.

```powershell
msemblator `
  --input .\data\example.msp `
  --output .\results\formula_only `
  --mode 1
```

### Mode 2: formula and structure elucidation

Mode 2 performs the complete workflow and is recommended for MSP files without
trusted molecular formulas. SIRIUS credentials are required.

```powershell
msemblator `
  --input .\data\example.msp `
  --output .\results\formula_and_structure `
  --mode 2 `
  --sirius_user "your_email@example.com" `
  --sirius_pass "your_password"
```

### Mode 3: structure elucidation only

Use mode 3 when the input MSP file already contains molecular formulas. SIRIUS
credentials are required.

```powershell
msemblator `
  --input .\data\formula_predicted.msp `
  --output .\results\structure_only `
  --mode 3 `
  --sirius_user "your_email@example.com" `
  --sirius_pass "your_password"
```

The same arguments can be passed to the bundled launcher:

```powershell
.\msemblator.cmd --input .\data\example.msp --output .\results --mode 1
```

### Command-line arguments

| Argument | Required | Description |
| --- | --- | --- |
| `--input` | Yes | Path to the input MSP file |
| `--output` | Yes | Directory in which summary CSV files are saved |
| `--mode` | Yes | `1`: formula, `2`: formula and structure, `3`: structure |
| `--sirius_user` | Modes 2 and 3 | SIRIUS account email or username |
| `--sirius_pass` | Modes 2 and 3 | SIRIUS account password |

> [!CAUTION]
> Supplying a password directly on the command line can expose it in shell
> history or process information. Use these commands only on a trusted machine
> and clear sensitive command history when appropriate.

## Output files

MS-Emblator creates the requested output directory if it does not exist. When a
file with the same name already exists, a unique name is generated instead of
overwriting the existing result.

Formula elucidation produces:

| File | Description |
| --- | --- |
| `formula_summary.csv` | Candidates reported by the individual tools together with the top ensemble result |
| `formula_score.csv` | Formula candidates ranked by the ensemble scoring model |

Structure elucidation produces:

| File | Description |
| --- | --- |
| `structure_summary.csv` | Structure candidates reported by the tools together with the top ensemble result |
| `structure_score.csv` | Structure candidates ranked by the ensemble scoring model |

Mode 2 produces both formula and structure output files. Intermediate converted
spectra and tool-specific results are stored under `work/`; they are intended
for runtime use and troubleshooting rather than as final results.

The number of ensemble-ranked records written per feature is controlled by
`msemblator_output_records` in the parameter file.

## Parameter configuration

The main user-editable configuration file is:

```text
config/msemblator_parameter_file.yaml
```

Default configuration:

```yaml
formula_prediction:
  msfinder:
    MS1_ppm: 10
    MS2_ppm: 20
    halogen: true

  sirius:
    # Supported values: orbitrap, qtof
    MS1: qtof
    MS2_ppm: 20
    halogen: true

  msbuddy:
    MS1_ppm: 10
    MS2_ppm: 20
    halogen: true

  msemblator_output_records: 100

structure_prediction:
  msfinder:
    MS1_ppm: 5
    MS2_ppm: 20
    halogen: true

  sirius:
    MS2_ppm: 20

  metfrag:
    # MetFrag uses the larger of the absolute and relative tolerances.
    MS2_Da: 0.01
    MS2_ppm: 20

  msemblator_output_records: 100
```

### Formula parameters

- `MS1_ppm`: precursor mass tolerance in ppm.
- `MS2_ppm`: fragment mass tolerance in ppm.
- `halogen`: whether halogen-containing molecular formulas are considered.
- `sirius.MS1`: SIRIUS instrument profile (`qtof` or `orbitrap`).
- `msemblator_output_records`: maximum number of ensemble-ranked candidates
  retained per feature.

### Structure parameters

- `msfinder.MS1_ppm` and `msfinder.MS2_ppm`: MS-FINDER tolerances.
- `sirius.MS2_ppm`: SIRIUS fragment mass tolerance.
- `metfrag.MS2_Da`: MetFrag absolute fragment tolerance in daltons.
- `metfrag.MS2_ppm`: MetFrag relative fragment tolerance in ppm.

Additional tool-specific templates are located under `config/metfrag/` and
`config/msfinder/`. Back up a template before making advanced changes.

## Troubleshooting

### `msemblator` is not recognized

Activate the virtual environment and reinstall the package:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Alternatively, run the bundled launcher:

```powershell
.\msemblator.cmd --help
```

### Missing runtime assets

If startup reports `Missing runtime assets`, compare every reported path with
the [runtime asset layout](#runtime-assets). In particular, check:

- that the Zenodo archive was extracted inside the repository;
- that no extra top-level directory was introduced during extraction;
- that executable and database file names were not changed;
- that the required `.pkl` model files exist under `models/formula/` or
  `models/structure/`.

### MetFrag does not start

Run `java --version` and confirm that Java 21 or newer is available on `PATH`.
Also confirm that `MetFragCommandLine-2.5.0.jar` is in `external/metfrag/`.

### Modes 2 and 3 stop at SIRIUS

Confirm that:

- the SIRIUS username and password are correct;
- the machine can reach the SIRIUS web service;
- `sirius.exe` and `sirius_structure_db.siriusdb` are in `external/sirius/`.

### Structure prediction is skipped or incomplete

Mode 3 requires formulas in the input MSP records. If the formulas are missing
or uncertain, use mode 2 so that formula elucidation runs first.

## License and citation

Before redistributing MS-Emblator or its separately downloaded runtime assets,
review the licenses of SIRIUS, MS-FINDER, MetFrag, msbuddy, the compound
databases, and the trained models. Citation information for MS-Emblator should
be added here when the corresponding publication or software record is
available.
