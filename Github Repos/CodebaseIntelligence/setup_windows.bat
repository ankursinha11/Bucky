@echo off
REM Windows Installation Script for CodebaseIntelligence
REM Works on Windows 10/11 and Azure Virtual Desktop (AVD)

echo ============================================================
echo CodebaseIntelligence - Windows Installation
echo ============================================================
echo.

REM Check Python installation
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

python --version
echo ✓ Python is installed
echo.

REM Check pip
echo [2/5] Checking pip installation...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not installed
    echo Installing pip...
    python -m ensurepip --upgrade
)
echo ✓ pip is available
echo.

REM Create virtual environment
echo [3/5] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists
    choice /C YN /M "Do you want to recreate it"
    if errorlevel 2 goto skip_venv
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment created
echo.

:skip_venv

REM Activate virtual environment and install dependencies
echo [4/5] Installing dependencies...
call venv\Scripts\activate.bat

REM Check which requirements file to use
if exist requirements-minimal.txt (
    echo Installing minimal requirements ^(local-only, no cloud dependencies^)...
    python -m pip install --upgrade pip
    python -m pip install -r requirements-minimal.txt
) else if exist requirements.txt (
    echo Installing from requirements.txt...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
) else (
    echo ERROR: No requirements file found
    echo Please ensure requirements.txt or requirements-minimal.txt exists
    pause
    exit /b 1
)

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo.
    echo Common issues:
    echo - Network connectivity problems
    echo - Insufficient disk space
    echo - Permission issues
    echo.
    pause
    exit /b 1
)
echo ✓ Dependencies installed
echo.

REM Verify parser files exist
echo [5/5] Verifying parser files...
if not exist parsers\hadoop\parser.py (
    echo ERROR: Parser files not found
    echo Please ensure all parser files are copied to the CodebaseIntelligence directory
    pause
    exit /b 1
)
echo ✓ Parser files verified
echo.

echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo Next steps:
echo.
echo 1. Verify installation:
echo    python verify_parser_version.py
echo.
echo 2. Index a codebase:
echo    python index_codebase.py --parser hadoop --source C:\path\to\Hadoop
echo.
echo 3. Query the chatbot:
echo    python chatbot_cli.py
echo.
echo For Azure Virtual Desktop users:
echo - Ensure you have network access for downloading packages
echo - All data is stored locally, no cloud services required
echo - Large codebases may require sufficient disk space
echo.
echo ============================================================

pause
