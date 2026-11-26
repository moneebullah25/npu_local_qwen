
@echo off
title Qwen 7B NPU FastAPI Server

cd /d "%~dp0"

set "PYTHON=%~dp0npu_env\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: Virtual environment not found.
    pause
    exit /b 1
)

rem ---- Model ----
set "QWEN_MODEL_PATH=%~dp0models\qwen2.5-coder-7b-npu-ov"
set "OPENVINO_DEVICE=NPU"

rem ---- NPU memory settings ----
set "MAX_PROMPT_LEN=16384"
set "MIN_RESPONSE_LEN=512"

rem ---- Generation ----
set "REPETITION_PENALTY=1.0"
set "TEMPERATURE=0"

rem ---- Claude Code tools ----
set "ALLOWED_TOOLS=Bash,Read,Edit,Write,Glob,Grep"
set "MAX_SYSTEM_CHARS=6000"
set "SYSTEM_MODE=lite"
set "STRIP_REMINDERS=1"
set "MAX_TOOL_DESC_CHARS=400"

rem ---- NPU compilation cache ----
set "NPU_CACHE_DIR=%~dp0npu_cache_7b"
set "LOG_LEVEL=INFO"

echo ==========================================
echo       QWEN 7B NPU API SERVER
echo ==========================================
echo Model: %QWEN_MODEL_PATH%
echo Device: %OPENVINO_DEVICE%
echo Context: %MAX_PROMPT_LEN% + %MIN_RESPONSE_LEN%
echo.

"%PYTHON%" -m uvicorn qwen_api:app --host 127.0.0.1 --port 8000

echo.
echo Server stopped.
pause
