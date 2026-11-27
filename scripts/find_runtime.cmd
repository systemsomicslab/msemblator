@echo off
rem Used by msemblator_gui.cmd and msemblator.cmd. Chooses how MS-Emblator's environment is made:
rem   conda (Miniconda / Anaconda installed)  -> the conda environment of environment.yml
rem   otherwise, or MSEMBLATOR_USE_VENV=1     -> a Python venv in .venv (Python 3.12 from python.org)
rem Sets MSEMBLATOR_PYTHON (the Python that runs scripts\msemblator_env.py) and
rem MSEMBLATOR_MODE (--conda "<conda.exe>" or --venv).
set "MSEMBLATOR_PYTHON="
set "MSEMBLATOR_MODE="
if /i not "%MSEMBLATOR_USE_VENV%"=="1" call :find_conda
if defined MSEMBLATOR_MODE exit /b 0
call :find_python
if defined MSEMBLATOR_MODE exit /b 0
echo Neither conda nor Python 3.12 was found. Install one of them and start again:
echo   conda  : https://docs.conda.io/projects/miniconda/
echo   Python : https://www.python.org/downloads/  (3.12; MetFrag also needs Java 21 or newer)
exit /b 1

:find_conda
set "MSEMBLATOR_CONDA="
if defined CONDA_EXE if exist "%CONDA_EXE%" set "MSEMBLATOR_CONDA=%CONDA_EXE%"
if not defined MSEMBLATOR_CONDA for %%C in (conda.exe) do if not "%%~$PATH:C"=="" set "MSEMBLATOR_CONDA=%%~$PATH:C"
for %%R in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniforge3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "C:\anaconda" "C:\anaconda3" "C:\miniconda3") do (
  if not defined MSEMBLATOR_CONDA if exist "%%~R\Scripts\conda.exe" set "MSEMBLATOR_CONDA=%%~R\Scripts\conda.exe"
)
if not defined MSEMBLATOR_CONDA exit /b 0
for %%I in ("%MSEMBLATOR_CONDA%\..\..") do set "MSEMBLATOR_PYTHON=%%~fI\python.exe"
if not exist "%MSEMBLATOR_PYTHON%" (
  set "MSEMBLATOR_PYTHON="
  exit /b 0
)
set "MSEMBLATOR_MODE=--conda "%MSEMBLATOR_CONDA%""
exit /b 0

:find_python
rem The Python launcher first (it knows every installed version), then python on PATH.
for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "MSEMBLATOR_PYTHON=%%P"
if not defined MSEMBLATOR_PYTHON for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "MSEMBLATOR_PYTHON=%%P"
if defined MSEMBLATOR_PYTHON set "MSEMBLATOR_MODE=--venv"
exit /b 0
