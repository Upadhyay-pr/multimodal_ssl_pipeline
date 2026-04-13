@echo off
REM =============================================================================
REM setup.bat  —  Windows equivalent of setup_data.sh
REM Multimodal SSL Preprocessing Pipeline — MED05664
REM
REM Tested on:  Windows 10/11, Python 3.8+
REM
REM What this does:
REM   1. Checks Python version (needs 3.8+)
REM   2. Creates directory structure
REM   3. Creates Python virtual environment (venv\)
REM   4. Installs requirements.txt
REM   5. Calls download_datasets.py to download all raw data
REM
REM Usage:
REM   Double-click this file, OR open Command Prompt and run:
REM     setup.bat
REM     setup.bat --skip-eeg
REM     setup.bat --skip-ecg
REM     setup.bat --no-mhealth
REM     setup.bat --dry-run
REM =============================================================================

setlocal EnableDelayedExpansion

REM --- Parse arguments ---
set SKIP_EEG=
set SKIP_ECG=
set NO_MHEALTH=
set DRY_RUN=

:parse_args
if "%~1"=="" goto done_parse
if /I "%~1"=="--skip-eeg"   set SKIP_EEG=--skip-eeg
if /I "%~1"=="--skip-ecg"   set SKIP_ECG=--skip-ecg
if /I "%~1"=="--no-mhealth" set NO_MHEALTH=--no-mhealth
if /I "%~1"=="--dry-run"    set DRY_RUN=--dry-run
if /I "%~1"=="--help" (
    echo.
    echo  Usage: setup.bat [options]
    echo.
    echo  Options:
    echo    --skip-eeg     Skip EEGMMIDB download (~3 GB^)
    echo    --skip-ecg     Skip PTB-XL download (~1.7 GB^)
    echo    --no-mhealth   Skip bonus mHealth dataset
    echo    --dry-run      Print actions without downloading
    echo    --help         Show this help
    echo.
    echo  Quick test (skips large downloads^):
    echo    setup.bat --skip-eeg --skip-ecg --no-mhealth
    echo.
    goto end
)
shift
goto parse_args
:done_parse

echo.
echo ============================================================
echo  Multimodal SSL Preprocessing Pipeline — Windows Setup
echo ============================================================
echo.

REM --- Change to script directory ---
cd /d "%~dp0"
echo Working directory: %CD%

REM --- Check Python availability ---
echo.
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH.
    echo.
    echo Solutions:
    echo   1. Download Python 3.10+ from https://python.org/downloads/
    echo   2. During installation, tick "Add Python to PATH"
    echo   3. Restart this batch file after installation
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VERSION=%%v
echo Found Python %PY_VERSION%

REM Check version >= 3.8
for /f "tokens=1,2 delims=." %%a in ("%PY_VERSION%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if !PY_MAJOR! LSS 3 (
    echo ERROR: Python 3.8+ required, found %PY_VERSION%.
    pause
    exit /b 1
)
if !PY_MAJOR! EQU 3 if !PY_MINOR! LSS 8 (
    echo ERROR: Python 3.8+ required, found %PY_VERSION%.
    pause
    exit /b 1
)

REM --- Create directory structure ---
echo.
echo [2/4] Creating directory structure...

for %%d in (
    data\raw\pamap2
    data\raw\wisdm
    data\raw\mhealth
    data\raw\eegmmidb
    data\raw\ptbxl
    data\interim\har
    data\interim\eeg
    data\interim\ecg
    data\processed\har
    data\processed\eeg
    data\processed\ecg
    reports
    submission_sample\har
    submission_sample\eeg
    submission_sample\ecg
    logs
) do (
    if not exist "%%d\" (
        mkdir "%%d" >nul 2>&1
    )
)
echo Directory tree created.

REM --- Create virtual environment ---
echo.
echo [3/4] Setting up Python virtual environment...

if not exist "venv\" (
    echo Creating venv\...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Could not create virtual environment.
        echo Try: python -m pip install virtualenv
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM Activate and install requirements
echo Installing requirements into venv...
call venv\Scripts\activate.bat

python -m pip install --upgrade pip --quiet
if exist "requirements.txt" (
    python -m pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo WARNING: Some packages failed to install.
        echo Trying core packages only...
        python -m pip install numpy pandas scipy pyyaml psutil mne wfdb matplotlib pytest
    )
) else (
    python -m pip install numpy pandas scipy pyyaml psutil mne wfdb matplotlib pytest --quiet
)
echo Requirements installed.

REM --- Download datasets ---
echo.
echo [4/4] Downloading datasets...
echo   This may take a while depending on internet speed.
echo   Downloads can be interrupted and resumed safely.
echo.

set DOWNLOAD_ARGS=%SKIP_EEG% %SKIP_ECG% %NO_MHEALTH% %DRY_RUN%
python download_datasets.py %DOWNLOAD_ARGS%

if errorlevel 1 (
    echo.
    echo WARNING: Some downloads may have failed.
    echo Re-run setup.bat to retry (already-downloaded files are skipped).
)

REM --- Done ---
echo.
echo ============================================================
echo  Setup complete!
echo ============================================================
echo.
echo  Next steps:
echo.
echo  1. Activate the virtual environment:
echo       venv\Scripts\activate.bat
echo.
echo  2. Run preprocessing:
echo       python preprocess.py --config configs\pipeline_config.yaml
echo.
echo  3. Validate outputs:
echo       python validate_outputs.py --config configs\pipeline_config.yaml
echo.
echo  4. Run tests:
echo       python -m pytest tests\ -v
echo.

:end
pause
endlocal
