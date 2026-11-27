# GLACIER structure features

This module compares measured MS/MS spectra with GLACIER predicted spectra and
turns the similarities into ranker features. It is the planned replacement for
SIRIUS in structure annotation (MetFrag + GLACIER + MS-FINDER).

What it does:

- For each measured spectrum, it takes every library compound with the same
  molecular formula and adduct as the query. No MS1 or precursor m/z filter is
  used to pick candidates.
- It scores the measured spectrum against each candidate's predicted spectra
  (10 and 40 eV) with cosine, reverse cosine and spectral entropy.
- It writes one feature row per candidate structure (Short InChIKey), plus
  optional labels and accuracy.

What it does not do yet:

- train a ranker or normalize the features
- run inside the default workflow

The default MS-Emblator workflow (MetFrag, SIRIUS, MS-FINDER, `models/structure`,
`struc_main.py`) is unchanged. GLACIER features must not be given to the
existing models.

## Quick start

```powershell
conda activate msemblator

# 1. Extra packages (once)
python -m pip install -r envs/glacier/requirements-glacier.txt

# 2. Unit tests (synthetic libraries; no real data needed)
python -m unittest discover -s tests/glacier -t .

# 3. Features for an MSP file
python scripts/glacier/run_glacier_features.py `
  --input path\to\queries.msp `
  --output path\to\output_folder
```

Before step 3, put the predicted libraries in `external/glacier/` (see
[Libraries](#libraries)).

## Set-up

| Package | Version | Used for | License |
| --- | --- | --- | --- |
| msentity | 1.2.2 | reading MSP and MSDS | MIT |
| msentity-similarity | 2.0.1 | cosine and reverse cosine (also modified cosine, BONANZA) | MIT |
| ms_entropy | 1.5.3 | weighted spectral entropy (only needed when `entropy` is selected) | Apache-2.0 |

- msentity-similarity 2.0.1 requires exactly msentity 1.2.2.
- The `msemblator` environment sets `PYTHONNOUSERSITE=1`. The packages must be
  installed into the environment itself; a copy in the user site-packages
  (`%APPDATA%\Python`) is ignored.

## Libraries

Put one MSDS file per adduct in `external/glacier/`. The folder is in
`.gitignore`. The file names are set in `config/glacier.yaml`:

| Adduct | File name |
| --- | --- |
| `[M-H]-` | `M_minus_H_predictions.msds` |
| `[M-H2O-H]-` | `M_minus_H2O_minus_H_predictions.msds` |
| `[M-H2O+H]+` | `M_minus_H2O_plus_H_predictions.msds` |
| `[M+H]+` | `M_plus_H_predictions.msds` |
| `[M+Na]+` | `M_plus_Na_predictions.msds` |
| `[M+NH4]+` | `M_plus_NH4_predictions.msds` |

A file with one of these names is used as soon as it is in the folder. No code
change is needed. If an adduct's file is not there yet, its queries get the
status `library_not_available` and the other adducts are still processed.

Only one library is held in memory at a time. Queries are grouped by adduct,
and each library is released after use. The `[M-H]-` library holds 1.05 M
spectra and takes about 1.3 GB.

### Library columns

Each library is read with msentity. The first column present is used:

| Field | Columns |
| --- | --- |
| spectrum ID | `SPECID`, `SpecID`, `SCANS`, `TITLE` |
| structure | `INCHIKEY` / `InChIKey`, `SMILES` (Short InChIKey falls back to RDKit from SMILES) |
| formula | `FORMULA`, `Formula` |
| adduct | `ION`, `AdductType`, `PredictionAdduct` |
| collision energy (eV) | `PredictionCollisionEnergyEV`, `CollisionEnergy`, `COLLISIONENERGY` |
| precursor m/z | `PrecursorMZ`, `PEPMASS`, `PredictionPrecursorMZ` |

### Adding or renaming a library

1. Put the MSDS in `external/glacier/` under the name in the table above. To use
   another name or folder, change `libraries` or `library_dir` in
   `config/glacier.yaml` (an absolute path is also accepted).
2. Check the adduct spelling in the library (`ION` column). Spellings in
   `adduct_aliases` are accepted, for example `[M+H-H2O]+` for `[M-H2O+H]+`.
   Spaces are ignored, and `1+` / `1-` is read as `+` / `-`.
   - If no spectrum of the file matches the adduct, the run reports the library
     as unusable. The message lists the spellings found in the file; add them
     to `adduct_aliases`.
3. Leave `formula_basis` on `auto`. The module detects whether `FORMULA` is the
   neutral molecule M or already includes the neutral part of the adduct (for
   example M − H2O for `[M-H2O+H]+`). See [Formula basis](#formula-basis).
4. Run with `--limit 100` first and check `glacier.log`. Each library logs a
   line like
   `[M+H]+ M_plus_H_predictions.msds: 199816 of 199816 spectra usable, 14550 formulas, formula basis neutral`.
5. If the adduct should get a one-hot feature column, add it to
   `adduct_features`. The default list holds the six adducts of the current
   structure models, so `[M-H2O+H]+` and `[M-H2O-H]-` rows have all one-hot
   columns at 0.

A new adduct that is not in the table (for example `[M+Cl]-`) works the same
way: add a line to `libraries` with its file name.

## Running

`scripts/glacier/run_glacier_features.py` processes every adduct in the input
that has a library. Run it from the repository root in the `msemblator`
environment.

### NIST26, correct-formula condition, with labels and the train/test split

```powershell
python scripts/glacier/run_glacier_features.py `
  --input E:\MS-Emblator\data\nist26\NIST26_id_ML.msp `
  --reference E:\MS-Emblator\data\nist26\NIST26_id_ML.msp `
  --output E:\MS-Emblator\data\nist26\tools_result\structure\glacier\M_minus_H `
  --adducts "[M-H]-" `
  --split train=E:\MS-Emblator\data\nist26\NIST26_id_train.msp `
  --split test=E:\MS-Emblator\data\nist26\NIST26_id_test.msp
```

Leave out `--adducts` to process every adduct that has a library, into one
output folder. The `adduct` column tells the rows apart.

### Other common runs

```powershell
# Quick check of a new library: first 100 spectra of one adduct
python scripts/glacier/run_glacier_features.py --input queries.msp --output out_test `
  --adducts "[M+H]+" --limit 100

# Predicted-formula condition: formulas from a CSV instead of the MSP FORMULA field
python scripts/glacier/run_glacier_features.py --input queries.msp --output out_pred `
  --formula-table predicted_formulas.csv

# Without precursor peak removal (scores the spectra as they are)
python scripts/glacier/run_glacier_features.py --input queries.msp --output out_raw `
  --remove-precursor off

# Another settings file
python scripts/glacier/run_glacier_features.py --input queries.msp --output out `
  --config my_glacier.yaml
```

Existing output files are not replaced unless `--overwrite` is given. An output
file may not replace an input file.

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--input` | required | MSP with the measured spectra. |
| `--output` | required | Output folder (created if needed). |
| `--config` | `config/glacier.yaml` | Settings file. |
| `--reference` | none | Answers for the label `TF` and the accuracy table: an MSP with InChIKey/SMILES, or a CSV such as `nist26_answer_id.csv`. Answers are used for nothing else. |
| `--formula-table` | none | CSV with `formula` and `filename` (MSP NAME) or `query_index`. Replaces the MSP formula field. |
| `--adducts` | all | Process only these adducts; aliases are accepted. Other spectra get `not_selected`. |
| `--limit N` | none | Process only the first N eligible spectra. |
| `--split NAME=MSP` | none | Adds a `split` column: spectra whose NAME is in MSP get NAME. Repeatable. |
| `--ms2-tolerance`, `--ms2-tolerance-unit` | 20, ppm | Fragment peak matching tolerance. |
| `--similarities` | `cosine reverse_cosine entropy` | Similarities to calculate. |
| `--remove-precursor {off,measured,both}` | `measured` | Where the precursor peak is removed. |
| `--precursor-tolerance`, `--precursor-tolerance-unit` | 20, ppm | Window for the precursor peak. |
| `--overwrite` | off | Replace existing output files. |

The command-line options override the settings file for that run only.

### Input MSP

The query MSP is read with msentity. Each spectrum needs:

| Field | Use |
| --- | --- |
| `NAME` | Becomes `filename`, the key shared with the MetFrag / MS-FINDER tables. |
| `PRECURSORTYPE` (adduct) | Chooses the library. Adducts without a library get `unsupported_adduct`. |
| `FORMULA` | The neutral molecular formula used to pick candidates (or `--formula-table`). |
| `PRECURSORMZ` | Only used for precursor peak removal. Spectra without it are scored unchanged. |
| peaks | The MS2 spectrum. |

Answer fields in the query MSP (InChIKey, SMILES) are not read. Answers come
only from `--reference`.

### Formula used for each query

The query formula is the neutral molecular formula decided before structure
annotation. By default it is read from the MSP `FORMULA` field. This is what the
default workflow writes into the MSP: the top formula in mode 2, or the user's
formula in mode 3.

- **Correct-formula condition:** run on `NIST26_id_ML.msp`, whose `FORMULA` is
  the answer.
- **Predicted-formula condition:** pass the predicted formulas with
  `--formula-table`.

With `--reference`, `glacier_summary.json` reports in `formula_condition` where
the formulas came from and how many equal the reference formula.

### Using it from Python

```python
from msemblator.glacier import load_config, run_glacier
from msemblator.glacier.integration import merge_with_candidates

config = load_config()                      # config/glacier.yaml
result = run_glacier("queries.msp", config, adducts=["[M-H]-"])
result.features   # one row per (query, Short InChIKey)
result.raw        # one row per (query, predicted spectrum)
result.queries    # one row per input spectrum, with status

# Join with a MetFrag / MS-FINDER table keyed by (filename, adduct, Short_InChIKey)
merged = merge_with_candidates(existing_candidates, result.features)
```

`load_config(path, overrides)` accepts a dict of single settings to replace, for
example `{"similarities": ["cosine"]}`.

## Settings (`config/glacier.yaml`)

| Setting | Default | Meaning |
| --- | --- | --- |
| `library_dir` | `external/glacier` | Library folder, relative to the repository root. |
| `libraries` | six adducts | Adduct → MSDS file. |
| `adduct_aliases` | dehydrated and NH4 spellings | Other spellings, used for the query MSP and the libraries. |
| `formula_basis` | `{default: auto}` | `auto`, `neutral` or `modified`, per adduct. |
| `collision_energies` | `[10, 40]` | Energies (eV) that become features. |
| `ms2_tolerance`, `ms2_tolerance_unit` | 20, ppm | Fragment matching tolerance. |
| `intensity_exponent` | 1.0 | Intensities are used as they are. |
| `similarities` | `[cosine, reverse_cosine, entropy]` | Also `modified_cosine`, `bonanza`. Each adds five features. |
| `precursor_peak_removal` | on, `measured`, 20 ppm | See [Precursor peak removal](#precursor-peak-removal). |
| `entropy` | `clean_spectra: true`, `noise_threshold: 0.01` | MSEntropy options. |
| `threads` | 0 | Threads inside msentity-similarity (0 = its default). |
| `adduct_features` | six MS-Emblator adducts | One-hot columns, named as in the existing structure features. |
| `query_columns`, `library_columns` | see the file | Metadata column names. |

Use single quotes for Windows paths in YAML. In double quotes, `\n`, `\t` and so
on are read as escapes.

## How the scores are calculated

### Candidates

Candidates are all library spectra of the query's adduct whose formula has the
same element counts as the query formula (a Hill-order key).

- No precursor m/z or MS1 tolerance is used.
- The formula is never changed or re-ranked.
- Spectra of other adducts are never compared.
- Formulas with elements that have no mass in `atomic_data.yaml` (As, Se, Ge)
  are still matched; they are only left out of the formula-basis mass check.

### Formula basis

The query formula is always the neutral molecule M. When a library is loaded,
the module computes the precursor m/z from `FORMULA` in two ways:

- **neutral:** `FORMULA` is the neutral molecule M.
- **modified:** `FORMULA` already includes the neutral part of the adduct, for
  example M − H2O for `[M-H2O+H]+`.

It compares both with the library precursor m/z on up to 2,000 spectra and uses
whichever matches at least 95% of them. For "modified", the query formula is
converted once (M → M − H2O); it is never converted twice. If neither definition
fits, the library is reported as unusable. To force one definition, set for
example `formula_basis: {'[M-H2O+H]+': modified}`.

The `[M-H]-` library uses the neutral formula (all checked spectra matched).

### Peaks

Measured peaks that are not finite, or have m/z ≤ 0 or intensity ≤ 0, are
dropped and counted in `n_invalid_peaks`. By default the precursor peak is also
removed from the measured spectra (below). No other peak removal or intensity
transform is applied. Predicted spectra are used as stored, with base peak = 1.

### Precursor peak removal

`precursor_peak_removal` is on by default with `spectra: measured`. The
precursor peak is removed from the measured MS2 peak list before any similarity
is calculated. The precursor is the base peak of most measured spectra; keeping
it raised the scores of wrong candidates (see
[the comparison](#precursor-peak-removal-comparison)). Use
`--remove-precursor off`, or `enabled: false` in the settings, to score the
spectra as they are.

- A peak is the precursor peak when `|peak m/z − precursor m/z| ≤ tolerance`.
  In ppm, the tolerance is relative to the precursor m/z.
- The precursor m/z is the spectrum's own metadata value: `PRECURSORMZ` of the
  query MSP, and `PrecursorMZ` of the library.
- This tolerance is separate from `ms2_tolerance`. Neither is used to select
  candidates.

| `spectra` | Peaks removed from |
| --- | --- |
| `measured` (default) | measured spectra only |
| `both` | measured and GLACIER predicted spectra |

How removal is recorded:

- `glacier_query_status.csv` has `precursor_peak_removal`
  (`removed` / `not_found` / `no_precursor_mz` / `off`) and
  `n_precursor_peaks_removed`.
- A measured spectrum with no peak left gets the status `no_valid_peaks`.
- `n_peaks_measured` and `n_peaks_predicted` in the raw output count the peaks
  actually used.
- The totals are in `glacier_summary.json` under `precursor_peak_removal`.

### Matching (msentity-similarity)

- Tolerance: 20 ppm by default, measured on the mean of the two m/z values,
  bounds included.
- Each peak pairs at most once, closest pairs first.
- The measured spectrum is the **query** (first argument) and the predicted
  spectrum is the **reference**:

```text
cosine         = Σ a·r (matched pairs) / (‖a‖ · ‖r‖)
reverse_cosine = Σ a·r (matched pairs) / (‖a matched‖ · ‖r‖)
```

Measured peaks that GLACIER did not predict do not lower `reverse_cosine`.
Predicted peaks without a measured partner still do.

The raw output keeps the share of matched intensity:
`peak_intensity_coverage_measured` and `peak_intensity_coverage_predicted`.
Each is Σx² over the matched peaks divided by Σx² over all peaks.
msentity-similarity does not report a matched peak count.

### Entropy (ms_entropy)

This is the weighted spectral entropy similarity with MSEntropy's own cleaning:

- peaks below 1% of the base peak are removed
- peaks closer than 2 × tolerance are merged
- intensities are normalized to sum to 1

The ppm tolerance is relative to the m/z of the measured peak. The cleaning
applies to entropy only.

### Value range

Scores are between 0 and 1. A pair with no matched peaks scores 0. A failed
calculation is NaN, with status `similarity_failed`, and is never 0.

## Features (`glacier_features.csv`)

There is one row per (query, Short InChIKey). Each similarity S (Cosine,
Reverse, Entropy) adds five raw, unnormalized features:

| Column | Meaning |
| --- | --- |
| `GLACIER_S_CE10`, `GLACIER_S_CE40` | Best score of the candidate's predicted spectra at that energy. NaN if the library has none. |
| `GLACIER_S_Max` | Maximum over the configured energies, taken for each similarity separately. |
| `GLACIER_S_Rank` | 1 = highest Max within the query. Ties are ordered by Short InChIKey. |
| `GLACIER_S_ScoreDiff` | Max minus the Max of the next rank. 0 for the last rank. |

One more feature describes the query rather than the candidate:

| Column | Meaning |
| --- | --- |
| `GLACIER_Candidate_Count` | Number of candidate structures (Short InChIKeys) GLACIER scored for the query. Same value on every row of the query. |

With the default three similarities there are 3 × 5 + 1 = 16 features.

- Rank and ScoreDiff follow the existing MetFrag / MS-FINDER definitions in
  `metfrag_summary.py` and `msfinder_struc_summary.py`: rank by order within the
  query, and the difference to the next candidate.
- The candidate count matters because rank and score mean different things
  depending on it. In NIST26 [M-H]-, 815 queries had a single candidate, and
  99% of them had the answer at rank 1. Queries with more than 100 candidates
  had it at rank 1 in 2–7% of cases, depending on the similarity.
- Missing values are left as NaN. The existing pipeline fills missing tool
  scores with 0; that choice, and any normalization, belong to the training
  step.
- If a Short InChIKey has several spectra at one energy (for example
  stereoisomers), the raw output keeps all of them and the feature takes the
  maximum.
- Library spectra without an energy are kept in the raw output but are not used
  for features. Spectra at an energy that is not configured are skipped.

Other columns:

| Column | Meaning |
| --- | --- |
| `filename`, `adduct`, `Short_InChIKey`, `SMILES` | Same keys as the existing structure tables. `filename` is the MSP NAME. |
| `query_index` | 1-based position in the MSP. |
| `formula`, `InChIKey` | Query formula, and the first InChIKey of the candidate. |
| `n_spectra`, `n_inchikeys`, `n_collision_energies` | Number of predicted spectra behind the candidate. Use these to check for bias from unequal spectrum counts. |
| `adduct_*` | The existing one-hot adduct columns. |
| `reference_Short_InChIKey`, `TF` | Only with `--reference`. |
| `split` | Only with `--split`. |

### Joining with MetFrag / MS-FINDER

`integration.merge_with_candidates()` joins the features with a MetFrag /
MS-FINDER table on (`filename`, `adduct`, `Short_InChIKey`):

- The join is an outer join, so candidates from either side are kept, and
  GLACIER-only candidates are added.
- Queries without GLACIER output keep their candidates.
- `candidate_source` records where each candidate came from: `existing`,
  `glacier` or `both`.
- `GLACIER_Candidate_Count` is filled for every candidate of a query GLACIER
  processed. The other GLACIER columns stay NaN for candidates GLACIER did not
  score.

## Outputs

| File | Content |
| --- | --- |
| `glacier_features.csv` | Features (above). This is the file to use for training. |
| `glacier_similarity_raw.csv` | One row per (query, predicted spectrum): energy, the scores, peak counts, coverages, status. |
| `glacier_query_status.csv` | One row per input spectrum: adduct, formula, peak counts, precursor removal, candidate counts, status and message. |
| `glacier_accuracy.csv` | Only with `--reference`. Top-1/5/10 per similarity and adduct, with two denominators: all queries, and queries with the answer among the candidates. |
| `glacier_spectra_count_check.csv` | Mean Max by number of predicted spectra per candidate. |
| `glacier_summary.json` | Counts, library checks, precursor removal, timings, settings, package versions, the command line. |
| `glacier.log` | The run log. |

CSV files are written as UTF-8 with BOM, so Excel opens them correctly.

### Query status

| Status | Meaning |
| --- | --- |
| `ok` | Scored. |
| `no_candidate` | The library has no compound with this formula. |
| `unsupported_adduct` | The adduct has no entry in `libraries` (for example `[M+Cl]-`). |
| `library_not_available` | The adduct's library is missing or unusable; `message` says why. |
| `missing_formula`, `invalid_formula` | No formula, or one that cannot be read. |
| `no_valid_peaks` | No peak left (for example, only the precursor peak). |
| `similarity_failed` | The calculation raised an error; scores are NaN. |
| `not_selected` | Left out by `--adducts` or `--limit`. |

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `ModuleNotFoundError: msentity` (or `msentity_similarity`, `ms_entropy`) | Install `envs/glacier/requirements-glacier.txt` into the `msemblator` environment. |
| `Output files already exist ... Use --overwrite` | Add `--overwrite`, or choose another `--output`. |
| All queries of an adduct are `library_not_available` with `file_missing` | The MSDS is not in `library_dir` under the configured name. |
| `... no usable spectrum for [M+X]+ ...; adduct values in the file: [...]` | The library writes the adduct in another spelling. Add it to `adduct_aliases`. |
| `... cannot tell how FORMULA is defined ...` | The formula basis check failed. Set `formula_basis` for that adduct. |
| Many `unsupported_adduct` | Expected for adducts without a GLACIER library (`[M+Cl]-`, `[M+CHO2]-` in NIST26). |
| Many `no_candidate` | The query formulas are not in the library (the library is built from the MS-FINDER structure DB). |

Do not call `run_glacier` from several Python threads at once. msentity
datasets are not thread-safe, and parallel use crashed the process.

## NIST26 [M-H]- results (correct-formula condition)

This run compared 4,417 [M-H]- spectra from `NIST26_id_ML.msp` with
`M_minus_H_predictions.msds` (527,288 compounds at 10 and 40 eV), with the
default settings (precursor peak removed from the measured spectra).

- 4,383 queries had candidates (median 9, maximum 1,761 structures).
- 33 queries had no candidate, and 1 query had no peak left after precursor
  peak removal (`no_valid_peaks`).
- For 4,333 queries, the answer was among the candidates.
- 290,380 query × predicted-spectrum pairs were scored.

| Similarity | Top-1 | Top-5 | Top-10 | Top-1 (answer among candidates) |
| --- | --- | --- | --- | --- |
| cosine | 47.3% | 72.4% | 80.7% | 48.2% |
| reverse cosine | 44.6% | 70.5% | 79.2% | 45.5% |
| entropy | 50.2% | 75.8% | 82.5% | 51.1% |

The first three columns use all 4,417 queries as the denominator.

### Run time

Run times are in seconds, on a 20-thread Windows machine. The default column is
the run above; the other columns change one setting.

| Step | Default (measured removal, 3 similarities) | No removal | No removal, cosine and reverse only | Removal on both sides |
| --- | --- | --- | --- | --- |
| Read the query MSP (17,401 spectra) | 0.8 | 0.8 | 0.7 | 0.9 |
| Load the library and build the index | 10.2 | 9.5 | 9.9 | 10.9 |
| Score the pairs (about 290,000) | 16.4 | 15.3 | 11.0 | 18.6 |
| Build the features | 1.6 | 1.4 | 1.4 | 1.4 |
| Total, including writing the CSVs | 33.8 | 31.7 | 26.9 | 35.8 |

- Entropy adds about 4 s.
- The library is loaded once per adduct.
- Scoring time grows with the number of pairs: the sum, over queries, of the
  candidate spectra per query. `[M+H]+` has about twice as many NIST26 queries
  as `[M-H]-`.

### Precursor peak removal comparison

In 92.5% of the measured [M-H]- spectra, the base peak is the precursor peak.
That is true for only 27% of the predicted spectra.

4,328 of the 4,417 measured spectra had exactly one peak within the window, and
89 had none. Top-1 (all queries):

| Similarity | Off | Measured, 20 ppm (default) | Measured, 0.01 Da | Both, 20 ppm |
| --- | --- | --- | --- | --- |
| cosine | 31.4% | 47.3% | 47.3% | 46.9% |
| reverse cosine | 30.0% | 44.6% | 44.6% | 44.3% |
| entropy | 40.0% | 50.2% | 50.2% | 50.5% |

- With "Both, 20 ppm", the peak was also removed from 117,480 of the
  1,054,576 predicted spectra.
- The 20 ppm and 0.01 Da windows differed in only 4 spectra.
- "Measured, 20 ppm" is the default. It is about as accurate as "Both", faster,
  and it leaves the predicted spectra as GLACIER wrote them.

### Other adducts

The other five libraries have not been generated yet. The loading, formula
basis detection and adduct handling were checked with libraries derived from
the `[M-H]-` file (same columns, with `ION`, `FORMULA` and `PrecursorMZ`
rewritten for each adduct, including the `[M+H-H2O]+` spelling and a
dehydrated `FORMULA`). All five loaded, the formula basis was detected
correctly, and no rows were compared across adducts. Accuracy for these
adducts needs the real libraries.

## Package layout

| File | Role |
| --- | --- |
| `config.py` | Settings (`config/glacier.yaml`) and their checks. |
| `formula.py` | Formula parsing, Hill-order keys, masses. |
| `adducts.py` | Adduct spellings, parsing, ion m/z, the neutral part of an adduct. |
| `queries.py` | Measured spectra from the MSP (no answer fields). |
| `preprocessing.py` | Precursor peak removal. |
| `library.py` | Predicted libraries: file lookup, loading, checks, formula index. |
| `similarity.py` | Cosine / reverse cosine (msentity-similarity), entropy (ms_entropy). |
| `features.py` | Energy and Short InChIKey aggregation; Max, Rank, ScoreDiff, candidate count. |
| `pipeline.py` | `run_glacier()`: the whole flow, shared by training and inference. |
| `labels.py` | Answers and the `TF` label (training only). |
| `evaluation.py` | Run summary and top-k accuracy. |
| `integration.py` | Join with MetFrag / MS-FINDER candidates. |

The command-line script is `scripts/glacier/run_glacier_features.py`, and the
tests are in `tests/glacier/`.

## Not implemented yet

- A matched peak count; msentity-similarity does not report one.
- A cache of the formula index. Loading the library and building the index takes
  about 10 s per adduct, so a cache was not needed.
- Scoring several formula groups in parallel (msentity is not thread-safe).
  msentity-similarity still uses its own threads inside one call.
- Ranker training, feature normalization, and use in `struc_main.py`.

## Replacing SIRIUS: places to change later

- `workflows/struc_main.py`: run GLACIER (`msemblator.glacier.run_glacier`)
  instead of the SIRIUS block. The input is the same formula-fixed MSP.
- `summaries/creating_struc_summary.py`: replace `process_sirius_output` with
  the GLACIER features, joined with `integration.merge_with_candidates`.
- `scoring/struc_score_calc.py`:
  - `predict_and_append_rank`: change the feature list.
  - `machine_input_generation`: change the tool list; it currently has
    `["metfrag", "sirius", "msfinder"]`.
- `models/structure`: needs a ranker trained on the new features. Decide the
  GLACIER feature normalization when training; the current pipelines are
  `pipeline_csi_fingerid*.pkl`. The existing models must not be given GLACIER
  features.
- `analysis/make_training_data.py`:
  - Add GLACIER to `STRUCTURE_FEATURES` and the readers.
  - The script imports `msemblator.summaries.sirius_summary`, which does not
    exist on this branch.
- `cli/msemblator.py` and `paths.validate_runtime`: modes 2 and 3 still need
  SIRIUS (login and `sirius.exe`).
- GLACIER libraries for `[M+Cl]-` and `[M+FA-H]-` (MS-Emblator adducts that
  GLACIER does not cover yet), and one-hot columns for the dehydrated adducts.
