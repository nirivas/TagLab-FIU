@echo off
rem Update TagLab (FIU lab version) from GitHub. Close TagLab first.
cd /d "%~dp0"
set "PY=python"
if exist "%~dp0venv\Scripts\python.exe" set "PY=%~dp0venv\Scripts\python.exe"
"%PY%" update.py %*
pause
