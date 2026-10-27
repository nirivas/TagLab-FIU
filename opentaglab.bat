@echo off
rem Start TagLab from the folder that contains this file, with its own Python environment (venv) when it exists.
cd /d "%~dp0"
set "PY=python"
if exist "%~dp0venv\Scripts\python.exe" set "PY=%~dp0venv\Scripts\python.exe"
"%PY%" TagLab.py
pause
