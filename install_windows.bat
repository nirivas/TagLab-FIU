@echo off
rem Install TagLab (FIU lab version) on this computer, in its own Python environment: the venv folder.
cd /d "%~dp0"
echo ==== TagLab installation ====
echo.

rem Python 3.11 is used to create the environment. The "py" launcher finds it even if other versions are installed.
set "BASEPY="
py -3.11 --version >nul 2>&1
if not errorlevel 1 set "BASEPY=py -3.11"
if not defined BASEPY (
    python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
    if not errorlevel 1 set "BASEPY=python"
)
if not defined BASEPY (
    echo Python 3.11 was not found. Install Python 3.11 64-bit from python.org and tick "Add python.exe to PATH".
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo Creating the Python environment of TagLab in the venv folder..
    %BASEPY% -m venv venv
)
if not exist "venv\Scripts\python.exe" (
    echo The Python environment could not be created, see the messages above.
    pause
    exit /b 1
)
set "PY=%~dp0venv\Scripts\python.exe"
echo Using the Python environment: %PY%
echo.

"%PY%" -m pip install --upgrade pip

rem packages, PyTorch (GPU version if an NVIDIA GPU is found), GDAL/rasterio and the AI models except SAM
"%PY%" install.py no-sam
if errorlevel 1 (
    echo.
    echo The installation failed, see the messages above.
    pause
    exit /b 1
)

echo.
set /p SAM="Download also the SAM model, about 2.5 GB, optional? [y/N] "
if /i "%SAM%"=="y" "%PY%" download_models.py --sam
if /i "%SAM%"=="yes" "%PY%" download_models.py --sam

echo.
rem record which GitHub version is installed, used to tell when an update is available
"%PY%" update.py --sync-state

echo.
echo Installation completed. Start TagLab with opentaglab.bat - update it later with update.bat
pause
