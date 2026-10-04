@echo off
setlocal
cd /d "%~dp0"
title TwitchCut - push to GitHub

set REPO=https://github.com/Lime5211/TwitchCut.git

where git >nul 2>nul
if errorlevel 1 (
  echo Git is not installed. Installing it with winget...
  winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
  echo.
  echo Git installed. Close this window and run push_github.bat again.
  pause
  exit /b 1
)

if not exist ".git" (
  git init
  git branch -M main
)

git remote get-url origin >nul 2>nul
if errorlevel 1 (
  git remote add origin %REPO%
) else (
  git remote set-url origin %REPO%
)

rem commit author for this folder only, if not set globally
git config user.name >nul 2>nul || git config user.name "Lime5211"
git config user.email >nul 2>nul || git config user.email "tlime5211@gmail.com"

echo.
echo Files going to GitHub (config.yaml, workspace, tools, cookies and logs are excluded by .gitignore):
git add -A
git status --short
echo.

git diff --cached --quiet
if errorlevel 1 git commit -q -m "TwitchCut update" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"

echo Pushing to GitHub. If a GitHub sign-in window opens, sign in as Lime5211.
git push -u origin main
if not errorlevel 1 goto ok

echo.
echo GitHub already has files (for example a README). Merging and trying again...
git pull origin main --allow-unrelated-histories --no-edit -X ours
git push -u origin main
if not errorlevel 1 goto ok

echo.
echo Push failed. Copy the error text above and send it to Claude.
pause
exit /b 1

:ok
echo.
echo Done: https://github.com/Lime5211/TwitchCut
pause
