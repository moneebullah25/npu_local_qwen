@echo off
title Qwen FastAPI Server

cd /d "%~dp0"

set "PYTHON=%~dp0npu_env\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: Virtual environment not found:
    echo %PYTHON%
    pause
    exit /b 1
)

rem ---- Settings (edit these) ----
rem Device: NPU or GPU. Try both and compare speed.
set "OPENVINO_DEVICE=NPU"

rem Context size. On the NPU the cache has a FIXED size (prompt + reply); bigger means
rem more memory and a slower first compile. 8192 was tested OK; 16384 is the default here.
rem Model limit is 32768 total; for that use MAX_PROMPT_LEN=32256 (slow, needs lots of free RAM).
set "MAX_PROMPT_LEN=16384"
set "MIN_RESPONSE_LEN=512"

rem Decoding. NPU sampling was broken in testing, so the server uses greedy on the NPU.
rem Repetition penalty: 1.1 worked; try 1.0 or 1.05 if you see stray "1" characters.
set "REPETITION_PENALTY=1.1"

rem Speed: send fewer tools and a shorter system prompt (biggest win with Claude Code).
rem ALLOWED_TOOLS=*  -> all tools.  MAX_SYSTEM_CHARS=0 -> full system prompt.
set "ALLOWED_TOOLS=Bash,Read,Edit,Write,Glob,Grep"
set "MAX_SYSTEM_CHARS=6000"
rem SYSTEM_MODE=lite: short system prompt (best for a 3B model). full: Claude Code's own.
set "SYSTEM_MODE=lite"
rem Remove Claude Code's <system-reminder> blocks (the model parrots them). 0 = keep.
set "STRIP_REMINDERS=1"
set "MAX_TOOL_DESC_CHARS=400"

set "NPU_CACHE_DIR=%~dp0npu_cache"
set "LOG_LEVEL=INFO"

echo Starting Qwen API server on http://127.0.0.1:8000  (device: %OPENVINO_DEVICE%)
echo Press Ctrl+C to stop.
echo.

"%PYTHON%" -m uvicorn qwen_api:app --host 127.0.0.1 --port 8000

echo.
echo Server stopped.
pause
