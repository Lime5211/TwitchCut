@echo off
cd /d "%~dp0"
set "CLAUDE="
for /f "delims=" %%P in ('where claude 2^>nul') do if not defined CLAUDE set "CLAUDE=%%P"
if not defined CLAUDE if exist "%USERPROFILE%\.local\bin\claude.exe" set "CLAUDE=%USERPROFILE%\.local\bin\claude.exe"
if not defined CLAUDE if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\claude.exe" set "CLAUDE=%LOCALAPPDATA%\Microsoft\WinGet\Links\claude.exe"
if not defined CLAUDE (
  echo [!] Claude Code not found. Run setup.bat first.
  pause
  exit /b 1
)
echo Sign in with your Claude Pro/Max subscription. A browser window will open...
"%CLAUDE%" auth login
"%CLAUDE%" auth status --text
pause
