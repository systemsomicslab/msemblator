@echo off
rem MS-Emblator command line, run inside the MS-Emblator environment, e.g.
rem   msemblator.cmd --input data\example.msp --output results --mode 2
rem The environment is created (first use) or updated (after a change of the
rem environment files) before the analysis starts. Uses conda when it is installed,
rem otherwise a Python 3.12 venv in .venv (MSEMBLATOR_USE_VENV=1 forces the venv).
setlocal
call "%~dp0scripts\find_runtime.cmd" || exit /b 1
"%MSEMBLATOR_PYTHON%" "%~dp0scripts\msemblator_env.py" %MSEMBLATOR_MODE% cli %*
exit /b %ERRORLEVEL%
