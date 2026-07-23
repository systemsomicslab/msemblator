# MS-Emblator

MS-Emblator is organized as an installable Python package using the `src` layout.

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

External tools, databases, and trained models are stored locally under `external/` and `models/`. These large runtime assets are excluded from Git. Generated intermediate files are written under `work/`.

## Command line

```powershell
msemblator --input sample.msp --output results --mode 1
```

The repository also includes a no-install Windows launcher:

```powershell
.\msemblator.cmd --input sample.msp --output results --mode 1
```

Use modes 2 and 3 with `--sirius_user` and `--sirius_pass`.
