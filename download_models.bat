@echo off
rem Download the AI models used by TagLab (they are not stored on GitHub).
cd /d "%~dp0"
set "PY=python"
if exist "%~dp0venv\Scripts\python.exe" set "PY=%~dp0venv\Scripts\python.exe"
"%PY%" download_models.py %*
pause
