@echo off
rem Creates "MS-Emblator" shortcuts (desktop and Start menu) with the MS-Emblator icon.
rem Options: -Desktop, -StartMenu, -Folder <folder>. See create_shortcut.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_shortcut.ps1" %*
if errorlevel 1 (
  echo.
  echo The shortcut could not be created. See the message above.
  pause
  exit /b 1
)
echo Shortcut created. Start MS-Emblator from the desktop or the Start menu.
