@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo  TwitchCut: speech recognition on the graphics card
echo ============================================================
echo.

where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo [*] NVIDIA card found: installing CUDA libraries for faster-whisper...
  if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
  )
  echo.
)

rem ---- whisper.cpp with Vulkan: works on AMD / Intel / NVIDIA, including integrated graphics
set "WC=tools\whispercpp"
set "ZIP=%TEMP%\whispercpp-vulkan.zip"
set "URL=https://github.com/jerryshell/whisper.cpp-windows-vulkan-bin/releases/download/v1.0.0/whisper.cpp-windows-vulkan.zip"
set "SHA=A5D408C72E460433B39875F74A0B6E27E60A3724301D478FE9873DB7FF4098E0"
if not exist "%WC%\models" mkdir "%WC%\models"

if exist "%WC%\whisper-cli.exe" (
  echo [ok] whisper.cpp ^(Vulkan^) is already installed
) else (
  echo [*] Downloading whisper.cpp ^(Vulkan build, ~18 MB^)...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing -Uri '%URL%' -OutFile '%ZIP%'"
  if not exist "%ZIP%" goto :dlfail
  powershell -NoProfile -ExecutionPolicy Bypass -Command "if ((Get-FileHash '%ZIP%' -Algorithm SHA256).Hash -ne '%SHA%') { exit 1 }"
  if errorlevel 1 (
    echo [!] Checksum mismatch: the downloaded file is not the expected one. Aborting.
    del "%ZIP%"
    goto :end
  )
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force '%ZIP%' '%WC%'"
  del "%ZIP%"
)

set "MODEL=%WC%\models\ggml-large-v3-turbo-q5_0.bin"
if exist "%MODEL%" (
  echo [ok] Speech model is already downloaded
) else (
  echo [*] Downloading speech model large-v3-turbo ^(~550 MB, one time^)...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing -Uri 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin' -OutFile '%MODEL%.part'"
  if not exist "%MODEL%.part" goto :dlfail
  move /y "%MODEL%.part" "%MODEL%" >nul
)

set "VAD=%WC%\models\ggml-silero-v5.1.2.bin"
if not exist "%VAD%" (
  echo [*] Downloading voice activity model ^(~1 MB^)...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; try { Invoke-WebRequest -UseBasicParsing -Uri 'https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin' -OutFile '%VAD%' } catch { }"
)

echo.
echo [*] Test run on the graphics card...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$f='%TEMP%\tc_test.wav'; & ffmpeg -v error -y -f lavfi -i 'sine=f=440:d=3' -ar 16000 -ac 1 $f 2>$null; & '%WC%\whisper-cli.exe' -m '%MODEL%' -f $f -l ru 2>&1 | Select-String 'ggml_vulkan: [0-9]|no GPU found|error' | Select-Object -First 3"
echo.
echo Done. Restart start.bat - speech will be recognized by the graphics card and the processor together.
echo The engine status is shown on the "Status" page.
goto :end

:dlfail
echo [!] Download failed. Check the internet connection and run setup_gpu.bat again.

:end
echo.
pause
