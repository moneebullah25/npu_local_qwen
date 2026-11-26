@echo off
setlocal
title Qwen 2.5 Coder 7B OpenVINO NPU API for Open WebUI

cd /d "%~dp0"

set "PYTHON=%~dp0npu_env\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo ERROR: Python virtual environment not found:
    echo %PYTHON%
    pause
    exit /b 1
)

set "QWEN_MODEL_PATH=%~dp0models\qwen2.5-coder-7b-npu-ov"
if not exist "%QWEN_MODEL_PATH%" (
    echo ERROR: Qwen 7B OpenVINO model folder not found:
    echo %QWEN_MODEL_PATH%
    pause
    exit /b 1
)

rem ---- API security and model identity ----
set "ANTHROPIC_AUTH_TOKEN=local-dev-key"
set "ANTHROPIC_MODEL=qwen-local"
set "HOST=0.0.0.0"
set "PORT=8000"

rem ---- OpenVINO NPU settings ----
rem Start with a conservative prompt length. Increase to 4096 or 8192 only
rem after confirming that the 7B model loads reliably on this machine.
set "OPENVINO_DEVICE=NPU"
set "MAX_PROMPT_LEN=16384"
set "MIN_RESPONSE_LEN=512"

rem ---- Generation ----
rem Greedy decoding is intentional because NPU sampling caused unstable text
rem in earlier tests with this setup.
set "TEMPERATURE=0"
set "REPETITION_PENALTY=1.0"
set "TOP_P=0.8"
set "TOP_K=20"

rem ---- Chat behavior ----
set "SYSTEM_MODE=full"
set "MAX_SYSTEM_CHARS=6000"
set "STRIP_REMINDERS=0"

rem ---- Optional function calling ----
set "ALLOWED_TOOLS=*"
set "MAX_TOOL_DESC_CHARS=400"
set "MAX_SCHEMA_DESC_CHARS=120"
set "STREAM_WITH_TOOLS=0"

rem ---- Dedicated NPU compilation cache ----
set "NPU_CACHE_DIR=%~dp0npu_cache_7b_openwebui_2048"
set "LOG_LEVEL=INFO"

echo.
echo ================================================
echo   QWEN 2.5 CODER 7B - OPEN WEBUI API
echo ================================================
echo Model:  %QWEN_MODEL_PATH%
echo Device: %OPENVINO_DEVICE%
echo Prompt: %MAX_PROMPT_LEN%
echo Reply:  %MIN_RESPONSE_LEN%
echo API:    http://127.0.0.1:%PORT%
echo.
echo Open WebUI base URL:
echo http://host.docker.internal:%PORT%/v1
echo.
echo Keep this window open while chatting.
echo Press Ctrl+C to stop the server.
echo.

"%PYTHON%" -m uvicorn qwen_api_openwebui:app --host 0.0.0.0 --port %PORT%

echo.
echo Qwen API server stopped.
pause
