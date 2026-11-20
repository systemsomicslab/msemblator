# SIRIUS-free formula model: annotation setup

This document covers the first step of the SIRIUS-free formula model: running
MSBuddy, MS-FINDER and FIDDLE on an MSP file in a common output format. Training
and inference with the new ranker are added in later steps. The default
MS-Emblator workflow (modes 1–3, `models/formula`, `models/structure`) is not
changed by anything described here.

## Environments

Two environments are used:

| Environment | Python | Used for | Defined in |
| --- | --- | --- | --- |
| `msemblator` | 3.12 (≥3.10) | MSBuddy, MS-FINDER, running the script | `environment.yml`, `requirements.txt` |
| `msemblator-fiddle` | 3.11 | FIDDLE (msfiddle 2.3.0, CPU torch) | `envs/fiddle/environment.yml`, `envs/fiddle/requirements-fiddle.txt` |

FIDDLE has its own environment because it needs torch and newer
pandas/scikit-learn than the main environment.

### Setting up on a new machine (PowerShell)

```powershell
# 1. Main environment (skip if it already exists)
conda env create -f environment.yml

# 2. FIDDLE environment
conda env create -f envs/fiddle/environment.yml

# 3. Download the FIDDLE weights (~0.3 GB, saved to %USERPROFILE%\.msfiddle\check_point\v2.3.0)
conda run -n msemblator-fiddle msfiddle download

# 4. Check that FIDDLE reproduces the reference output
conda run -n msemblator-fiddle python scripts/sirius_free/check_fiddle_install.py
#    -> "OK: FIDDLE output is identical to the reference."
```

MS-FINDER uses the same files as the default workflow:
`external/msfinder/MSFINDER*/MsfinderConsoleApp.exe` and
`config/msfinder/MsfinderConsoleApp_Param_formula.txt`. The template is copied
before tolerances are applied, so it is never modified.

### Checking the whole set-up with the demo data

```powershell
conda activate msemblator
python scripts/sirius_free/run_formula_annotation.py `
  --input tests/fixtures/sirius_free_demo.msp --output work/sirius_free_demo

# Unit tests; MSEMBLATOR_RUN_TOOL_TESTS=1 also runs MSBuddy and MS-FINDER on the demo
$env:MSEMBLATOR_RUN_TOOL_TESTS = "1"
python -m unittest discover -s tests -t .
```

`tests/fixtures/sirius_free_demo.msp` has 31 spectra:

- 21 real spectra are the msfiddle demo spectra (EMBL-MCF 2.0).
- 10 synthetic records exercise every status. They cover a missing collision
  energy, `[M+Cl]-`, formate, positive adducts, fewer than 5 peaks, a stepped
  collision energy, an unsupported adduct and an unknown instrument.
- `scripts/sirius_free/make_demo_msp.py` rebuilds the file.

Expected status counts:

| Tool | ok | no_candidate | unsupported_adduct | missing_collision_energy | rejected_by_tool | unknown_instrument |
| --- | --- | --- | --- | --- | --- | --- |
| msbuddy | 28 | 0 | 3 | – | 0 | – |
| msfinder | 28 | 0 | 3 | – | 0 | – |
| fiddle | 20 | 3 | 4 | 1 | 2 | 1 |

## Running the annotation

```powershell
conda activate msemblator
python scripts/sirius_free/run_formula_annotation.py --input <data.msp> --output <out_dir>
```

- `--tools msbuddy msfinder fiddle` selects tools. The default comes from
  `annotation.tools` in `config/sirius_free_formula.yaml`.
- FIDDLE is started automatically in `msemblator-fiddle`, through `conda run`.
  If conda is not on `PATH`, set `annotation.fiddle.python` to the
  environment's `python.exe`.
- To run only FIDDLE, use the FIDDLE environment directly:

  ```powershell
  conda run -n msemblator-fiddle python scripts/sirius_free/run_formula_annotation.py --input <data.msp> --output <out_dir> --tools fiddle
  ```

- Each tool writes its own sub-folder, so the tools can be run at different
  times or on different machines into the same `<out_dir>`.

### Input requirements

The script reads the MSP directly and keeps every field. It does not use the
reduced MSP of the default workflow, which drops the collision energy.

| Field | MSBuddy | MS-FINDER | FIDDLE |
| --- | --- | --- | --- |
| `PRECURSORMZ` / `PrecursorMZ` | required | required | required |
| `PRECURSORTYPE` / `Precursor_type` | required | required | required |
| `Collision_energy` | – | – | required (single value, eV or `NCE=..%`) |
| `Instrument_type` / `Instrument` | – | – | required to choose the model (Q-TOF → `qtof`, HCD / Orbitrap / FT → `orbitrap`, timsTOF → `timstof`) |
| peaks | – | – | at least 5 peaks within m/z 50–1500 |

- Spectra without a collision energy are not sent to FIDDLE. No default
  energy is substituted; this is option A.
- For NIST26, use the original `NIST26_ML_data.msp`, which has
  `Collision_energy` and `Instrument_type`. `NIST26_id_ML.msp` does not have
  them.

### Supported adducts

| Adduct | MSBuddy | MS-FINDER | FIDDLE |
| --- | --- | --- | --- |
| [M+H]+ | yes | yes | yes |
| [M+NH4]+ | yes | yes | yes (FIDDLE converts it to [M+H+NH3]+) |
| [M+Na]+ | yes | yes | yes |
| [M-H]- | yes | yes | yes |
| [M+Cl]- | yes | yes | **no** → `unsupported_adduct` |
| [M+FA-H]- | yes | yes | yes (FIDDLE folds it into [M-H]-) |

Notes on adduct names:

- `[M+CHO2]-`, `[M+HCOO]-` and `[M+HCO2]-` are read as `[M+FA-H]-`.
- `[M+CO2]-` is a different ion and is not mapped.
- Any other adduct gets the status `unsupported_adduct` for every tool.

### Output

```text
<out_dir>/
  spectra_index.csv            query_id, name, adduct, instrument, collision energy, reference formula, ...
  annotation.log
  msbuddy/  msbuddy_candidates.csv   msbuddy_status.csv   manifest.json
  msfinder/ msfinder_candidates.csv  msfinder_status.csv  manifest.json  (+ input_msp/, raw_output/)
  fiddle/   fiddle_candidates.csv    fiddle_status.csv    manifest.json
```

`query_id` is the position of the spectrum in the MSP, counting from 1. This is
the same numbering as the default workflow.

All `*_candidates.csv` files have the columns `query_id, tool, rank, formula,
score`, followed by tool-specific columns:

| Tool | `score` | Other columns |
| --- | --- | --- |
| msbuddy | `1 - estimated_fdr`, the same score as the default model | `estimated_fdr, estimated_prob, normed_estimated_prob, mz_error, ms1_isotope_similarity, db_existed` |
| msfinder | `Formula score` | `mass_error, precursor_type_reported, databases` |
| fiddle | Rescore clipped to 0–1 | `raw_rescore, candidate_mass, pred_formula, instrument_model` |

The scores of different tools are not on a common scale.

Every spectrum has exactly one row in each `*_status.csv`:

| status | Meaning |
| --- | --- |
| `ok` | The tool returned at least one candidate |
| `no_candidate` | The tool ran but returned no candidate |
| `unsupported_adduct` | The adduct is not supported by the tool; it is not sent, and never converted to another adduct |
| `missing_collision_energy` | FIDDLE only: no collision energy in the MSP |
| `unknown_instrument` | FIDDLE only: the instrument cannot be mapped to a released model |
| `rejected_by_tool` | FIDDLE rejected the spectrum (fewer than 5 peaks, stepped or unparseable energy); its message is kept |
| `failed` | The tool raised an error; the message is kept |

`manifest.json` records the following:

- versions of the tool, Python and packages, and the platform;
- SHA-256 of the input MSP and of the FIDDLE checkpoints;
- parameters and run time;
- the number of spectra in each status.

## Differences from the default workflow's tool runners

These differences apply only to the new scripts. The default workflow keeps its
behaviour.

- **MSBuddy candidates are matched to spectra by `query_id`.** The default
  runner matches by precursor m/z, so two spectra with the same m/z can get
  each other's candidates.
- **MSBuddy adducts are parsed with `report_invalid=True`.** An adduct it
  cannot parse is reported. In the default runner it is silently replaced by
  [M+H]+ or [M-H]-.
- **MSBuddy keeps more fields.** Mass error, ML probability and isotope
  similarity are saved; the default runner saves only the FDR.
- **The MS-FINDER input has no answer fields.** FORMULA, SMILES and INCHIKEY
  are not written to the per-spectrum MSP files, so they cannot influence the
  prediction.

## Licences (summary; details in the licence report)

| Component | Code | Weights / data | Notes |
| --- | --- | --- | --- |
| MSBuddy | Apache-2.0 | No separate terms stated for its formula database or ML models | Needs confirmation |
| FIDDLE | Apache-2.0 | **No licence stated for the weights.** Q-TOF/Orbitrap models are trained on NIST20/23 (paid); the Q-TOF model also on Agilent PCDL | Needs confirmation before commercial use |
| MS-FINDER | LGPL-3 | Binaries CC-BY 4.0 | Attribution required; LGPL obligations apply on redistribution |
| CatBoost / Optuna | Apache-2.0 / MIT | – | – |

FIDDLE's released models were trained on NIST20/NIST23. Evaluating them on
NIST spectra overestimates FIDDLE's accuracy. Evaluation must therefore also be
reported on compounds that are not in NIST23.
