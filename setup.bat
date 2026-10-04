@echo off
setlocal
cd /d "%~dp0"
echo === TwitchCut setup ===
echo.

rem ---------- 1. Python ----------
set "PY="
call :findpy
if not defined PY (
  echo Python not found - installing Python 3.12 with winget, please wait...
  winget install -e --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
  call :findpy
)
if not defined PY (
  echo.
  echo [!] Could not install Python automatically.
  echo     Download Python 3.12 from https://www.python.org/downloads/
  echo     During installation tick "Add python.exe to PATH", then run setup.bat again.
  pause
  exit /b 1
)
echo Using Python: %PY%

rem ---------- 2. Virtual environment + packages ----------
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  "%PY%" -m venv .venv
  if errorlevel 1 (
    echo [!] Failed to create virtual environment.
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [!] Failed to install packages. Check your internet connection and run setup.bat again.
  pause
  exit /b 1
)

rem ---------- 3. FFmpeg ----------
where ffmpeg >nul 2>nul
if errorlevel 1 (
  if not exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe" (
    echo FFmpeg not found - installing with winget...
    winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
  )
)

rem ---------- 4. Claude Code (analysis via your Claude subscription) ----------
call :findclaude
if not defined CLAUDE (
  echo Installing Claude Code - official installer...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://claude.ai/install.ps1 | iex"
  call :findclaude
)
if not defined CLAUDE (
  echo Trying winget...
  winget install -e --id Anthropic.ClaudeCode --accept-source-agreements --accept-package-agreements
  call :findclaude
)
if defined CLAUDE (
  echo Claude Code: %CLAUDE%
  "%CLAUDE%" auth status >nul 2>nul
  if errorlevel 1 (
    echo.
    echo Sign in to Claude with your Pro/Max subscription. A browser window will open...
    "%CLAUDE%" auth login
  ) else (
    echo Claude Code: already signed in.
  )
) else (
  echo [!] Claude Code could not be installed automatically.
  echo     Open PowerShell and run:  irm https://claude.ai/install.ps1 ^| iex
  echo     then run login_claude.bat
)

if not exist config.yaml copy config.example.yaml config.yaml >nul
echo.
echo ===============================================
echo  Done! Start TwitchCut with start.bat
echo ===============================================
pause
exit /b 0

:findclaude
set "CLAUDE="
for /f "delims=" %%P in ('where claude 2^>nul') do if not defined CLAUDE set "CLAUDE=%%P"
if not defined CLAUDE if exist "%USERPROFILE%\.local\bin\claude.exe" set "CLAUDE=%USERPROFILE%\.local\bin\claude.exe"
if not defined CLAUDE if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\claude.exe" set "CLAUDE=%LOCALAPPDATA%\Microsoft\WinGet\Links\claude.exe"
exit /b 0

:findpy
for %%V in (3.12 3.11 3.13) do (
  if not defined PY (
    for /f "delims=" %%P in ('py -%%V -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%P"
  )
)
if not defined PY (
  for /f "delims=" %%P in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%P"
)
for %%D in (Python312 Python311 Python313) do (
  if not defined PY (
    if exist "%LOCALAPPDATA%\Programs\Python\%%D\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\%%D\python.exe"
  )
)
exit /b 0
