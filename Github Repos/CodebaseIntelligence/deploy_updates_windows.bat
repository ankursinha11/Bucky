@echo off
REM Deploy updated parser files to CodebaseIntelligence
REM Use this script to copy updated files from updated_files_for_windows folder

echo ============================================================
echo Deploying Updated Parser Files
echo ============================================================
echo.

REM Check if updated files directory exists
if not exist updated_files_for_windows (
    echo ERROR: updated_files_for_windows directory not found
    echo.
    echo Please ensure you have copied the updated_files_for_windows folder
    echo to this directory before running this script.
    pause
    exit /b 1
)

echo Found updated_files_for_windows directory
echo.

REM Backup existing files
echo Creating backup of current files...
set BACKUP_DIR=backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set BACKUP_DIR=%BACKUP_DIR: =0%

mkdir "%BACKUP_DIR%" 2>nul
mkdir "%BACKUP_DIR%\parsers" 2>nul
mkdir "%BACKUP_DIR%\core" 2>nul
mkdir "%BACKUP_DIR%\services" 2>nul

REM Backup parsers
if exist parsers (
    echo Backing up parsers...
    xcopy parsers "%BACKUP_DIR%\parsers" /E /I /Q >nul 2>&1
)

REM Backup models
if exist core\models (
    echo Backing up core\models...
    xcopy core\models "%BACKUP_DIR%\core\models" /E /I /Q >nul 2>&1
)

REM Backup services
if exist services (
    echo Backing up services...
    xcopy services "%BACKUP_DIR%\services" /E /I /Q >nul 2>&1
)

echo ✓ Backup created in: %BACKUP_DIR%
echo.

REM Deploy updated files
echo Deploying updated files...

REM Copy parsers
echo - Copying parsers...
xcopy updated_files_for_windows\parsers parsers /E /I /Y >nul
if errorlevel 1 (
    echo ERROR: Failed to copy parsers
    pause
    exit /b 1
)

REM Copy models
echo - Copying core\models...
xcopy updated_files_for_windows\core\models core\models /E /I /Y >nul
if errorlevel 1 (
    echo ERROR: Failed to copy models
    pause
    exit /b 1
)

REM Copy services
echo - Copying services...
xcopy updated_files_for_windows\services services /E /I /Y >nul
if errorlevel 1 (
    echo ERROR: Failed to copy services
    pause
    exit /b 1
)

echo.
echo ✓ Files deployed successfully!
echo.

REM Run verification
echo ============================================================
echo Running Verification...
echo ============================================================
echo.

if exist verify_parser_version.py (
    python verify_parser_version.py
) else if exist updated_files_for_windows\verify_parser_version.py (
    copy updated_files_for_windows\verify_parser_version.py . >nul
    python verify_parser_version.py
) else (
    echo WARNING: verify_parser_version.py not found
    echo Please run verification manually
)

echo.
echo ============================================================
echo Deployment Complete!
echo ============================================================
echo.
echo Backup location: %BACKUP_DIR%
echo.
echo You can now run:
echo   python index_codebase.py --parser hadoop --source C:\path\to\Hadoop
echo.

pause
