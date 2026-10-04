@echo off
rem Usage: cut.bat https://www.twitch.tv/videos/123456789 [--clips 5] [--llm none]
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m twitchcut %*
pause
