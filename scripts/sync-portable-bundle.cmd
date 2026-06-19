@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT=%%~fI"
set "OUTPUT_DIR=dist\Haze_UAP-Windows-x86_64-Portable"
set "REVERSE=0"
set "DRY_RUN=0"

:parse
if "%~1"=="" goto parsed
if /I "%~1"=="--output-dir" (
    if "%~2"=="" goto usage
    set "OUTPUT_DIR=%~2"
    shift
    shift
    goto parse
)
if /I "%~1"=="--reverse" (
    set "REVERSE=1"
    shift
    goto parse
)
if /I "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
    goto parse
)
if /I "%~1"=="-h" goto usage
if /I "%~1"=="--help" goto usage
echo Unknown option: %~1 1>&2
goto usage_error

:usage
echo Usage: scripts\sync-portable-bundle.cmd [options]
echo.
echo Options:
echo   --output-dir DIR   Portable output directory under dist\ ^(default: %OUTPUT_DIR%^)
echo   --reverse          Sync portable bundle files back into repo bundle\
echo   --dry-run          Preview changes without copying or deleting
echo   -h, --help         Show this help
exit /b 0

:usage_error
echo.
echo Usage: scripts\sync-portable-bundle.cmd [--output-dir DIR] [--reverse] [--dry-run] 1>&2
exit /b 2

:parsed
for %%I in ("%ROOT%\%OUTPUT_DIR%") do set "OUT_FULL=%%~fI"
for %%I in ("%ROOT%\dist") do set "DIST_FULL=%%~fI"

set "CHECK=%OUT_FULL%\"
set "DIST_CHECK=%DIST_FULL%\"
if /I not "%OUT_FULL%"=="%DIST_FULL%" if /I "!CHECK:%DIST_CHECK%=!"=="!CHECK!" (
    echo Refusing to sync outside the dist directory: %OUT_FULL% 1>&2
    exit /b 1
)

if not exist "%OUT_FULL%" (
    echo Portable directory does not exist: %OUT_FULL% 1>&2
    exit /b 1
)

if "%REVERSE%"=="1" (
    set "FROM_ROOT=%OUT_FULL%"
    set "TO_ROOT=%ROOT%\bundle"
    echo Syncing portable bundle files back into repo bundle...
) else (
    set "FROM_ROOT=%ROOT%\bundle"
    set "TO_ROOT=%OUT_FULL%"
    echo Syncing repo bundle files into portable bundle...
)

set "ROBO_FLAGS=/MIR /R:2 /W:1 /NFL /NDL /NP /XF *.onnx"
if "%DRY_RUN%"=="1" (
    set "ROBO_FLAGS=%ROBO_FLAGS% /L"
    echo Dry run only. No files will be changed.
)

for %%D in (webroot managed audio) do (
    set "SRC=!FROM_ROOT!\%%D"
    set "DST=!TO_ROOT!\%%D"
    if not exist "!SRC!" (
        echo Skipping missing source: !SRC!
    ) else (
        echo %%D: !SRC! --^> !DST!
        robocopy "!SRC!" "!DST!" %ROBO_FLAGS%
        set "RC=!ERRORLEVEL!"
        if !RC! GEQ 8 (
            echo robocopy failed for %%D with exit code !RC! 1>&2
            exit /b !RC!
        )
    )
)

echo Bundle sync complete.
exit /b 0
