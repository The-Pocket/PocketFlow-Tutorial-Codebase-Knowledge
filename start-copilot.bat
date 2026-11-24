@echo off
REM ======================================================================
REM Generates a tutorial for a local code base with the GitHub Copilot CLI.
REM No API key is needed. Run "copilot login" once before the first use.
REM
REM Usage:
REM   start-copilot.bat [directory] [options]
REM
REM   directory             Project to analyze. Asked for if omitted.
REM                         Required as first argument if options are given.
REM
REM Options (passed on to main.py, they override the defaults of this script):
REM   --language german     Language of the tutorial (default: english)
REM   --no-cache            Ignore cached answers. The cache only knows the prompt, not the
REM                         model: use this to compare models on the same project.
REM   --max-abstractions 15 Maximum number of chapters (default: 10)
REM   --max-size 200000     Skip files larger than this, in bytes (default: 100000)
REM   --include "*.cs"      Files to read (default here: "*.cs" "*.ts" "*.htm*")
REM   --exclude "*Tests*"   Files to skip. REPLACES the built-in list (bin, obj, tests,
REM                         node_modules, ...), so repeat the entries you still need.
REM   -n, --name MyProject  Project name (default: name of the directory)
REM   -o, --output C:\Docs  Output directory (default: output)
REM
REM Environment variables (set them before the call or in the section below):
REM   COPILOT_CLI_MODEL             Model (default: claude-sonnet-5.5), e.g. gpt-6-luna
REM   COPILOT_CLI_CONTEXT           Context window: default or long_context
REM                                 (long_context is preset for the default model only)
REM   COPILOT_CLI_REASONING_EFFORT  Reasoning effort, e.g. medium or high
REM   COPILOT_CLI_TIMEOUT           Timeout per call in seconds (default: 1200)
REM   COPILOT_CLI_PATH              Copilot executable, if it is not in PATH
REM   COPILOT_GITHUB_TOKEN          Token instead of "copilot login". GITHUB_TOKEN and
REM                                 GH_TOKEN are not passed on to Copilot.
REM
REM Examples:
REM   start-copilot.bat C:\Projects\App
REM   start-copilot.bat C:\Projects\App --language german --no-cache
REM   start-copilot.bat C:\Projects\App --max-abstractions 15 --include "*.cs" "*.xaml"
REM
REM Notes:
REM   Every call consumes AI credits. A run needs one call per chapter plus three.
REM   The first prompt contains all files and has to fit into the context window.
REM   Prompts (source code) are stored in logs, llm_cache.json and the Copilot sessions.
REM ======================================================================

setlocal

set "SCRIPT_DIR=%~dp0"

title PocketFlow - Codebase Analyzer (GitHub Copilot)

REM ==================================================
REM Configuration
REM ==================================================

set "LLM_PROVIDER=COPILOT_CLI"

REM Uncomment to change the defaults
REM set "COPILOT_CLI_MODEL=gpt-6-luna"
REM set "COPILOT_CLI_CONTEXT=long_context"
REM set "COPILOT_CLI_REASONING_EFFORT=high"
REM set "COPILOT_CLI_TIMEOUT=2400"

REM Print umlauts and other special characters correctly
set "PYTHONUTF8=1"

REM ==================================================
REM Target directory
REM ==================================================

REM Resolved before changing the directory, so that relative paths stay valid
set "TARGET_DIR=%~1"
if not defined TARGET_DIR set /p "TARGET_DIR=Directory of the project to analyze: "
if not defined TARGET_DIR (
    echo.
    echo ERROR: No directory given.
    goto :error
)

REM Remove quotes (drag and drop into the prompt)
set "TARGET_DIR=%TARGET_DIR:"=%"

REM Absolute path without trailing backslash, which would escape the closing quote
for %%I in ("%TARGET_DIR%\.") do set "TARGET_DIR=%%~fI"

if not exist "%TARGET_DIR%\" (
    echo.
    echo ERROR: Directory does not exist:
    echo "%TARGET_DIR%"
    goto :error
)

REM Pass the remaining arguments on to main.py
set "EXTRA_ARGS="
:collect_args
shift
if "%~1"=="" goto :args_done
set EXTRA_ARGS=%EXTRA_ARGS% %1
goto :collect_args
:args_done

REM Always run in the directory of this script
REM (stored at the beginning because shift also moves %%0)
cd /d "%SCRIPT_DIR%"

REM ==================================================
REM Prerequisites
REM ==================================================

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

if defined COPILOT_CLI_PATH goto :copilot_found
where copilot >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: GitHub Copilot CLI not found.
    echo Install it or set COPILOT_CLI_PATH, then run "copilot login" once.
    goto :error
)
:copilot_found

REM ==================================================
REM Run
REM ==================================================

echo.
echo ==========================================
echo PocketFlow Codebase Analyzer
echo ==========================================
echo.
echo Analyzing: "%TARGET_DIR%"
if defined COPILOT_CLI_MODEL (echo Model:     %COPILOT_CLI_MODEL%) else (echo Model:     claude-sonnet-5.5)
echo.

"%PYTHON%" main.py --dir "%TARGET_DIR%" --include "*.cs" "*.ts" "*.htm*" %EXTRA_ARGS%
if errorlevel 1 (
    echo.
    echo ERROR: The run was aborted. See the messages above and "%CD%\logs".
    goto :error
)

echo.
echo ==========================================
echo Finished.
echo ==========================================

pause
exit /b 0

:error
echo.
pause
exit /b 1
