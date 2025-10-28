@echo off
REM Package updated files for transfer to Windows
REM This is the Windows version of package_for_windows.sh

echo ==================================
echo Packaging Updated Files for Windows
echo ==================================

set OUTPUT_DIR=updated_files_for_windows

REM Remove old package if exists
if exist "%OUTPUT_DIR%" rmdir /s /q "%OUTPUT_DIR%"

REM Create directory structure
mkdir "%OUTPUT_DIR%\parsers\hadoop"
mkdir "%OUTPUT_DIR%\parsers\abinitio"
mkdir "%OUTPUT_DIR%\parsers\databricks"
mkdir "%OUTPUT_DIR%\core\models"
mkdir "%OUTPUT_DIR%\services"

REM Copy updated parser files
echo Copying parsers...
copy parsers\__init__.py "%OUTPUT_DIR%\parsers\" >nul
copy parsers\hadoop\__init__.py "%OUTPUT_DIR%\parsers\hadoop\" >nul
copy parsers\hadoop\parser.py "%OUTPUT_DIR%\parsers\hadoop\" >nul
copy parsers\hadoop\oozie_parser.py "%OUTPUT_DIR%\parsers\hadoop\" >nul
copy parsers\abinitio\__init__.py "%OUTPUT_DIR%\parsers\abinitio\" >nul
copy parsers\abinitio\parser.py "%OUTPUT_DIR%\parsers\abinitio\" >nul
copy parsers\abinitio\mp_file_parser.py "%OUTPUT_DIR%\parsers\abinitio\" >nul
copy parsers\abinitio\patterns.py "%OUTPUT_DIR%\parsers\abinitio\" >nul
copy parsers\databricks\__init__.py "%OUTPUT_DIR%\parsers\databricks\" >nul
copy parsers\databricks\parser.py "%OUTPUT_DIR%\parsers\databricks\" >nul

REM Copy updated model files
echo Copying models...
copy core\models\component.py "%OUTPUT_DIR%\core\models\" >nul

REM Copy updated service files
echo Copying services...
copy services\codebase_indexer.py "%OUTPUT_DIR%\services\" >nul

REM Copy verification script
echo Copying verification script...
copy verify_parser_version.py "%OUTPUT_DIR%\" >nul

REM Create instructions file
echo Creating instructions...
(
echo INSTRUCTIONS FOR WINDOWS
echo ========================
echo.
echo 1. Copy all files from this directory to your Windows CodebaseIntelligence folder
echo 2. Maintain the same directory structure ^(parsers\, core\models\, services\^)
echo 3. Run the verification script:
echo.
echo    cd CodebaseIntelligence
echo    python verify_parser_version.py
echo.
echo 4. You should see all checkmarks ^(✓^) if files were copied correctly
echo 5. Then try running the indexer again
echo.
echo Files included:
echo - parsers\hadoop\parser.py ^(CRITICAL - unique ID fix^)
echo - parsers\hadoop\oozie_parser.py ^(coordinator support^)
echo - parsers\abinitio\* ^(all Ab Initio parsers^)
echo - parsers\databricks\* ^(all Databricks parsers^)
echo - core\models\component.py ^(OOZIE_COORDINATOR type^)
echo - services\codebase_indexer.py ^(document ID structure fix^)
echo - verify_parser_version.py ^(verification script^)
echo.
echo Key Changes:
echo - Process IDs now include file hash for uniqueness
echo - Component IDs now include index for uniqueness
echo - Parser finds all workflow and coordinator files ^(not just workflow.xml^)
echo - Coordinators properly supported and tagged
echo - Document IDs at top level ^(not in metadata^)
echo.
echo Windows Path Compatibility:
echo - All path handling uses os.path and pathlib.Path
echo - Works with both forward slashes and backslashes
echo - Tested on Windows 10/11 and Azure Virtual Desktop
) > "%OUTPUT_DIR%\COPY_INSTRUCTIONS.txt"

echo.
echo ✓ Packaged files to: %OUTPUT_DIR%\
echo.

REM Count files
dir /s /b "%OUTPUT_DIR%\*.py" | find /c ".py"

echo.
echo Next steps:
echo 1. Copy the '%OUTPUT_DIR%' folder to Windows
echo 2. Follow instructions in COPY_INSTRUCTIONS.txt

pause
