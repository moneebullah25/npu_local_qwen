@echo off
rem Launch Claude Code against the local Qwen server (start run_qwen_api.cmd first).

set "ANTHROPIC_BASE_URL=http://127.0.0.1:8000"
set "ANTHROPIC_AUTH_TOKEN=local-dev-key"
set "ANTHROPIC_MODEL=qwen-local"

rem Background tasks also go to the local model (the server ignores the name anyway).
set "ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen-local"
set "ANTHROPIC_DEFAULT_SONNET_MODEL=qwen-local"
set "ANTHROPIC_DEFAULT_OPUS_MODEL=qwen-local"

rem Tell Claude Code the real context size so it compacts instead of overflowing.
set "CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768"
rem Compaction window has a 100000 minimum, so lower the trigger with the percentage.
set "CLAUDE_CODE_AUTO_COMPACT_WINDOW=100000"
set "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=25"

rem Max reply length per request (server clamps to the room that is left).
set "CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192"

set "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"

claude %*
