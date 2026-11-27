@echo off
rem MS-Emblator GUI. Double-click this file.
rem The first start creates the MS-Emblator environment (about 10 minutes); later
rem starts update it when the environment files change.
rem The GUI then runs inside that environment. Uses conda (Miniconda / Anaconda) when it
rem is installed, otherwise a Python 3.12 venv in .venv (MSEMBLATOR_USE_VENV=1 forces the venv).
setlocal
call "%~dp0scripts\find_runtime.cmd" || goto :failed
"%MSEMBLATOR_PYTHON%" "%~dp0scripts\msemblator_env.py" %MSEMBLATOR_MODE% gui || goto :failed
exit /b 0

:failed
echo.
echo MS-Emblator could not be started. See the messages above.
pause
exit /b 1
