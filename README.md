# MS-Emblator 2026.10.28

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
| `1` | Formula elucidation | SIRIUS, MS-FINDER, msbuddy | No | Yes |
| `2` | Formula and structure elucidation | SIRIUS, MS-FINDER, msbuddy, MetFrag | No | Yes |
| `3` | Structure elucidation only | SIRIUS, MS-FINDER, MetFrag | Yes | Yes |

Mode 2 is recommended when the input MSP file does not already contain reliable
molecular formula annotations.

The tools above are the default combination. FIDDLE (formula) and GLACIER
(structure) can be selected instead of, or in addition to, SIRIUS; see
[Selecting annotation tools](#selecting-annotation-tools).

> [!IMPORTANT]
> SIRIUS 6 requires a login for every command-line run, including formula
> prediction. A SIRIUS account is therefore needed whenever SIRIUS is among
> the selected tools (the default). See [SIRIUS login](#sirius-login).

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
[Zenodo](https://zenodo.org/records/23721007).

## Installation

Clone the repository and move into its directory:

```powershell
git clone https://github.com/systemsomicslab/msemblator.git
cd msemblator
```

MS-Emblator runs in one environment that holds everything: MS-Emblator,
msbuddy, FIDDLE (CPU PyTorch) and the GLACIER packages. The launchers create and
maintain it:

```powershell
.\msemblator_gui.cmd                 # the GUI (or double-click the file)
.\msemblator.cmd --help              # the command line
```

The environment is made in one of two ways:

| | conda (recommended) | Python venv (no conda) |
| --- | --- | --- |
| Needs | [Miniconda](https://docs.conda.io/projects/miniconda/) or Anaconda | [Python 3.12](https://www.python.org/downloads/) (3.10 and 3.11 also work; 3.13 does not), and Java 21 or newer for MetFrag (e.g. [Adoptium](https://adoptium.net/)) |
| Used when | conda is installed | conda is not installed, or `MSEMBLATOR_USE_VENV=1` is set |
| Environment | conda environment `msemblator` from `environment.yml` (includes Java 21) | `.venv` in the repository, with `pip install -e . -r requirements-extras.txt` |

- The first start creates the environment (about 10 minutes, 2–3 GB) and
  downloads the FIDDLE weights.
- Later starts check `environment.yml`, `requirements.txt`,
  `requirements-extras.txt`, `envs/glacier/requirements-glacier.txt` and
  `pyproject.toml`; when one of them changed (e.g. after `git pull`), the
  environment is updated first. Close other MS-Emblator windows before that.
- The launchers run MS-Emblator inside the environment, without `PYTHONPATH`
  or user site-packages from other programs.
- `MSEMBLATOR_ENV_NAME` (conda) or `MSEMBLATOR_VENV_DIR` (venv) selects another
  environment.

By hand, the same environment is, with conda:

```powershell
conda env create -f environment.yml          # or: conda env update -f environment.yml
conda activate msemblator
python -m msfiddle download                  # FIDDLE weights, once
msemblator --help
```

or without conda:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e . -r requirements-extras.txt
python -m msfiddle download                  # FIDDLE weights, once
msemblator --help
```

`-h` is a short form of `--help`. The help lists the modes, every argument, the
SIRIUS login requirement, and example commands.

The environment installs MS-Emblator in editable mode from this folder, so keep
the repository where it is. After moving it, delete the environment
(`conda env remove -n msemblator`, or the `.venv` folder) and start a launcher
again.

## Runtime assets

Download the runtime asset archive from
[Zenodo](https://zenodo.org/records/23721007), extract it, and arrange the
repository as shown below. Keep the directory and file names unchanged.

```text
msemblator/
├─ config/
│  ├─ msemblator_parameter_file.yaml
│  ├─ toolsets.yaml          (tool combinations; in Git)
│  ├─ glacier.yaml           (GLACIER settings; in Git)
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
│  ├─ sirius/
│  │  └─ sirius.exe
│  └─ glacier/               (only for GLACIER)
│     └─ *_predictions.msds
├─ models/
│  ├─ formula/
│  │  ├─ *.pkl
│  │  └─ msfinder_msbuddy_fiddle/       (only for --formula_tools fiddle)
│  └─ structure/
│     ├─ *.pkl
│     └─ msfinder_glacier_metfrag/      (only for --structure_tools glacier)
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

At startup, MS-Emblator checks for the assets required by the selected mode and
tools (`--list_tools` shows them for every tool combination). A
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
.\msemblator.cmd --input <input.msp> --output <output-directory> --mode <1|2|3>
```

To show the available options and examples:

```powershell
.\msemblator.cmd --help    # or -h
```

### SIRIUS login

SIRIUS 6 refuses to run from the command line until you are logged in, also
for formula prediction in mode 1. Without a login, SIRIUS stops with
`Login ERROR: Please Login to use the SIRIUS command line tool!` and MS-Emblator
reports that `formula_identifications_top-100.tsv` was not written.

There are two ways to log in. Both work in every mode.

- **Recommended:** log in once with SIRIUS itself. The login is stored in your
  user profile and is reused by later runs, so no credentials are needed on the
  MS-Emblator command line:

  ```powershell
  .\external\sirius\sirius.exe login -u "your_email@example.com" -p
  .\external\sirius\sirius.exe login --show   # check the current login
  ```

- Give your account to MS-Emblator, which then logs in before the first SIRIUS
  step. Pass `--sirius_user` (or set `SIRIUS_USER`). The password is taken from
  `SIRIUS_PASS` if set; otherwise MS-Emblator asks for it at the prompt:

  ```powershell
  $env:SIRIUS_USER = "your_email@example.com"
  $env:SIRIUS_PASS = "your_password"   # optional; omit to be asked instead
  ```

  `--sirius_pass` is still accepted, but a password typed on the command line
  can remain in shell history.

### Mode 1: formula elucidation only

Use mode 1 to predict molecular formulas without running structure annotation.
The examples below assume that you have logged in with `sirius.exe login`; if
not, add `--sirius_user "your_email@example.com"` (see
[SIRIUS login](#sirius-login)).

```powershell
.\msemblator.cmd `
  --input .\data\example.msp `
  --output .\results\formula_only `
  --mode 1
```

### Mode 2: formula and structure elucidation

Mode 2 performs the complete workflow and is recommended for MSP files without
trusted molecular formulas.

```powershell
.\msemblator.cmd `
  --input .\data\example.msp `
  --output .\results\formula_and_structure `
  --mode 2
```

### Mode 3: structure elucidation only

Use mode 3 when the input MSP file already contains molecular formulas.

```powershell
.\msemblator.cmd `
  --input .\data\formula_predicted.msp `
  --output .\results\structure_only `
  --mode 3
```

The same arguments can be passed to the bundled launcher:

```powershell
.\msemblator.cmd --input .\data\example.msp --output .\results --mode 1
```

### GUI (prototype)

A desktop window runs the same analysis without typing commands:

```powershell
.\msemblator_gui.cmd          # inside the activated environment also: python -m msemblator.gui
```

- **Left:** the spectra of the opened MSP, with the MS/MS spectrum of the
  selected one below.
- **Analysis tab:** choose the files, the mode and the formula / structure tools
  (each combination is marked *Ready* or *Not ready*; click the mark to see what
  is missing), then **Run analysis**. The log shows the progress; **Stop** ends
  the run and the tools it started.
- **Results tab:** the ranked formula and structure candidates of the selected
  spectrum, with the tools that support each one, and a drawing of the selected
  structure. When GLACIER was used, a mirror plot compares the measured spectrum
  with the GLACIER prediction of the candidate (10 or 40 eV; peaks matched
  within 20 ppm are coloured). **File > Load results…** opens the CSV files of
  an earlier run.
- **Drag and drop:** drop an `.msp` file on the window to open it, or a folder
  to use it as the output folder (a folder with results opens them).
- The last input file, output folder, mode, tools, SIRIUS user and window size
  are remembered, and **File > Recent MSP files** lists recent inputs. They are
  kept in `%APPDATA%\MS-Emblator\gui_settings.json`; the password is never saved.
- **Settings > Analysis parameters** edits `config/msemblator_parameter_file.yaml`
  (comments in the file are kept); **Settings > Tool setup** lists every tool
  combination and the files it still needs.
- **Downloads** (toolbar, or Settings > Download tools, models and libraries)
  installs SIRIUS, MS-FINDER and MetFrag from their official releases, and the
  models and compound libraries from the MS-Emblator Zenodo record. The license
  and terms of each item are shown first and must be accepted. See
  [Downloading the runtime assets](#downloading-the-runtime-assets).
- **Set up FIDDLE** (Downloads tab, or Settings > Set up FIDDLE) checks FIDDLE
  in the MS-Emblator environment: it installs msfiddle and CPU PyTorch into an
  environment created before they were part of `environment.yml`, and
  downloads the FIDDLE weights if they are missing. The launchers do the same
  at start. Without the GUI: `python -m msemblator.fiddle_setup --run`.

The window starts the `msemblator` command line in the background, so the
results are the same. **Analysis > Show command line** shows the equivalent
command. The SIRIUS password is passed through the environment, not on the
command line.

### Downloading the runtime assets

The download addresses are in `config/downloads.yaml`; edit that file to
change them. A new Zenodo version needs only `zenodo_record` (and
`zenodo_file` / `zenodo_root` if the archive was renamed); installed models and
libraries then show *update available*. Items without an address yet (FIDDLE
and GLACIER models, GLACIER libraries) are listed as *not published*.

```powershell
python -m msemblator.downloads --list                        # what is installed
python -m msemblator.downloads --install sirius models_formula
```

- Models and libraries are read directly out of the Zenodo archive with HTTP
  range requests, so updating the models does not download the whole 2.4 GB
  archive.
- Files are checked before they replace anything: with the SHA-256 in the
  manifest (given for SIRIUS), the MD5 that Zenodo publishes, or the zip CRC-32.
  Archive entries can only be written inside their destination folder.
- What was installed, and from which address, is recorded in
  `external/downloads_installed.json`.
- The official MetFrag 2.5.0 release jar is a later build than the jar used to
  validate MS-Emblator (see the note in `config/downloads.yaml`).

### Selecting annotation tools

`--formula_tools` and `--structure_tools` choose the tools. Give a combination
name or a list of tools (comma separated, any order, case ignored):

| Step | Name | Tools | Ranking model folder |
| --- | --- | --- | --- |
| formula | `sirius` (default) | SIRIUS, MS-FINDER, msbuddy | `models/formula/` |
| formula | `fiddle` | MS-FINDER, msbuddy, FIDDLE | `models/formula/msfinder_msbuddy_fiddle/` |
| formula | `all` | SIRIUS, MS-FINDER, msbuddy, FIDDLE | `models/formula/sirius_msfinder_msbuddy_fiddle/` |
| structure | `sirius` (default) | MS-FINDER, SIRIUS, MetFrag | `models/structure/` |
| structure | `glacier` | MS-FINDER, GLACIER, MetFrag | `models/structure/msfinder_glacier_metfrag/` |
| structure | `all` | MS-FINDER, SIRIUS, MetFrag, GLACIER | `models/structure/msfinder_sirius_metfrag_glacier/` |

```powershell
# SIRIUS-free formula and structure elucidation
.\msemblator.cmd --input .\data\example.msp --output .\results --mode 2 `
  --formula_tools fiddle --structure_tools glacier

# The same formula tools, written as a list
.\msemblator.cmd --input .\data\example.msp --output .\results --mode 1 `
  --formula_tools msfinder,msbuddy,fiddle

# Which combinations can run on this computer, and what each one still needs
.\msemblator.cmd --list_tools
```

How it works:

- Each combination is ranked by its own trained model, so only the
  combinations in `config/toolsets.yaml` can be selected. A combination
  without its ranking model stops at startup with the missing path. The
  models for FIDDLE and GLACIER combinations still have to be trained.
- Only the selected tools run. SIRIUS, its login and `sirius.exe` are needed
  only when SIRIUS is selected.
- The model folder holds the ranking model (`xgbranker_final_all.pkl`, or the
  CatBoost / LightGBM file). Each tool's normalization pipelines
  (`pipeline_<tool>_*.pkl`) are read from that folder if present, otherwise
  from `models/formula/` or `models/structure/`.
- The ranking model's own feature list is used. A model that needs features of
  a tool that is not selected is refused.
- FIDDLE runs in the MS-Emblator environment (msfiddle and CPU PyTorch come
  from `requirements-extras.txt`). Its weights are downloaded once to
  `%USERPROFILE%\.msfiddle`. It needs `pipeline_fiddle_score.pkl` and
  `pipeline_fiddle_score_diff.pkl`. To run it in another environment instead,
  set `formula_prediction.fiddle.conda_env` or `.python` in
  `config/msemblator_parameter_file.yaml`.
- FIDDLE needs a collision energy for each spectrum. The order of priority is:
  1. `--fiddle_ce <eV>` on the command line: that energy for every spectrum;
  2. `override_collision_energy: true` in `formula_prediction.fiddle` (in the
     GUI: Settings > Analysis parameters > FIDDLE, *Use this energy for every
     spectrum*): `default_collision_energy` for every spectrum;
  3. otherwise the collision energy of the MSP record, and
     `default_collision_energy` (30 eV) for records without one (`null`: such
     spectra are skipped by FIDDLE).

  Energies taken from the settings are noted in FIDDLE's status file.
- GLACIER needs its packages (also in `requirements-extras.txt`) and the
  predicted spectral libraries in `external/glacier/` (see `README_glacier.md`).
  Its features are given to the ranking model as raw values. Spectra whose
  adduct has no GLACIER library are ranked from the other tools.

To add a combination, train a ranking model for it, put the model folder in
place and add an entry to `config/toolsets.yaml`. `--toolsets` reads another
file in the same format.

### Command-line arguments

| Argument | Required | Description |
| --- | --- | --- |
| `--input` | Yes | Path to the input MSP file |
| `--output` | Yes | Directory in which summary CSV files are saved |
| `--mode` | Yes | `1`: formula, `2`: formula and structure, `3`: structure |
| `--formula_tools` | No | Formula tools: `sirius` (default), `fiddle`, `all`, or a list such as `msfinder,msbuddy,fiddle` |
| `--structure_tools` | No | Structure tools: `sirius` (default), `glacier`, `all`, or a list such as `msfinder,glacier,metfrag` |
| `--fiddle_ce` | No | FIDDLE collision energy (eV) for every spectrum; takes priority over the MSP and the parameter file |
| `--toolsets` | No | Tool combination file (default `config/toolsets.yaml`) |
| `--list_tools` | No | Show the tool combinations and the files each one still needs, then exit |
| `--sirius_user` | No | SIRIUS account email or username. Defaults to `SIRIUS_USER`; without either, the login stored by `sirius.exe login` is used |
| `--sirius_pass` | No | SIRIUS account password. Defaults to `SIRIUS_PASS`; without either, you are asked for it |
| `-h`, `--help` | No | Show the help message and exit |

> [!CAUTION]
> Supplying a password directly on the command line with `--sirius_pass` can
> expose it in shell history or process information. Prefer `sirius.exe login`,
> the password prompt, or `SIRIUS_PASS`.

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

### SIRIUS stops with `Login ERROR`

SIRIUS is not logged in, or the stored login has expired. Log in again with
`sirius.exe login`, or pass `--sirius_user`, as described in
[SIRIUS login](#sirius-login). This applies to every mode.

### SIRIUS fails in any mode

Confirm that:

- you are logged in to SIRIUS (see [SIRIUS login](#sirius-login));
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
