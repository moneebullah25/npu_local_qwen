"""Local Qwen -> Anthropic Messages API bridge (OpenVINO GenAI, NPU).

Run:  python server.py
Env:  QWEN_MODEL_PATH, OPENVINO_DEVICE, MAX_PROMPT_LEN, MIN_RESPONSE_LEN,
      MAX_TOOL_DESC_CHARS, TEMPERATURE, NPU_CACHE_DIR, PORT, LOG_LEVEL
"""
import asyncio
import hmac
import json
import logging
import os
import re
import threading
import uuid
from typing import Any, Callable

import openvino_genai as ov_genai
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# MODEL_PATH = os.getenv("QWEN_MODEL_PATH", r"D:\npu_local_qwen\models\qwen2.5-coder-3b-npu-ov")

MODEL_PATH = os.getenv(
    "QWEN_MODEL_PATH",
    r"D:\npu_local_qwen\models\qwen2.5-coder-7b-npu-ov"
)

DEVICE = os.getenv("OPENVINO_DEVICE", "NPU")
API_KEY = os.getenv("ANTHROPIC_AUTH_TOKEN", "local-dev-key")
MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "qwen-local")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = _env_int("PORT", 8000)

MODEL_CONTEXT_TOKENS = 32768

# On the NPU these are STATIC shapes: the KV cache is sized
# MAX_PROMPT_LEN + MIN_RESPONSE_LEN. Bigger = slower init, more RAM, slower prefill.
# 16384 is a safer default for a 16 GB machine; raise it if it initializes fine.
MAX_PROMPT_LEN = _env_int("MAX_PROMPT_LEN", 16384)
MIN_RESPONSE_LEN = _env_int("MIN_RESPONSE_LEN", 512)


# MAX_PROMPT_LEN = _env_int("MAX_PROMPT_LEN", 2048)
# MIN_RESPONSE_LEN = _env_int("MIN_RESPONSE_LEN", 512)

IS_NPU = DEVICE.upper().startswith("NPU")
# NPU: fixed KV-cache size. GPU/CPU: dynamic, so the full model context is usable.
TOTAL_CAPACITY = (
    min(MODEL_CONTEXT_TOKENS, MAX_PROMPT_LEN + MIN_RESPONSE_LEN) if IS_NPU else MODEL_CONTEXT_TOKENS
)

MAX_OUTPUT_TOKENS = TOTAL_CAPACITY
DEFAULT_OUTPUT_TOKENS = 4096
MAX_REQUEST_BYTES = 32 * 1024 * 1024

# Claude Code sends huge tool descriptions; shortening them cuts prefill time a lot.
# Set to 0 to disable trimming.
MAX_TOOL_DESC_CHARS = _env_int("MAX_TOOL_DESC_CHARS", 400)
MAX_SCHEMA_DESC_CHARS = _env_int("MAX_SCHEMA_DESC_CHARS", 120)

# Claude Code sends ~20 tools plus a very long system prompt on EVERY request, and this
# server re-processes the whole prompt each turn. Keeping only the core tools and a
# shortened system prompt is the biggest speed-up. Use ALLOWED_TOOLS=* for all tools and
# MAX_SYSTEM_CHARS=0 for the full system prompt.
_tools_env = os.getenv("ALLOWED_TOOLS", "Bash,Read,Edit,Write,Glob,Grep").strip()
ALLOWED_TOOLS = set() if _tools_env in ("", "*") else {t.strip() for t in _tools_env.split(",") if t.strip()}
MAX_SYSTEM_CHARS = _env_int("MAX_SYSTEM_CHARS", 6000)

# "lite": replace Claude Code's long system prompt with a short one plus just the
# environment facts (working directory, OS). A 3B model gets lost in the full prompt.
# "full": send Claude Code's system prompt (shortened to MAX_SYSTEM_CHARS).
SYSTEM_MODE = os.getenv("SYSTEM_MODE", "lite").strip().lower()
# Claude Code injects <system-reminder> blocks (agent lists, skills, etc.) into messages.
# A small model tends to parrot them back, so they are removed by default.
STRIP_REMINDERS = os.getenv("STRIP_REMINDERS", "1") != "0"
STREAM_WITH_TOOLS = os.getenv("STREAM_WITH_TOOLS", "0") == "1"
_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_ENV_LINE_RE = re.compile(
    r"^\s*[-*]?\s*(Primary working directory|Platform|Shell|OS Version|Is a git repository)\s*:.*$",
    re.MULTILINE | re.IGNORECASE,
)

# On this NPU, sampling (temperature/top-k/top-p) produced garbage in testing (diag.py),
# while greedy + repetition penalty was fine. So NPU defaults to greedy (TEMPERATURE=0);
# GPU/CPU default to light sampling. Override with the TEMPERATURE env var.
TEMPERATURE = _env_float("TEMPERATURE", 0.0 if IS_NPU else 0.3)
TOP_P = _env_float("TOP_P", 0.8)
TOP_K = _env_int("TOP_K", 20)
# NOTE: this penalty also applies to every token in the PROMPT, so with long prompts it
# pushes greedy decoding away from natural words (junk first tokens). Keep it at 1.0 and
# let the loop detector handle repetition; raise to 1.05 only if loops return.
REPETITION_PENALTY = _env_float("REPETITION_PENALTY", 1.0)

NPU_CACHE_DIR = os.getenv("NPU_CACHE_DIR", r".\npu_cache")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("qwen-anthropic-api")


# ---------------------------------------------------------------------------
# Load tokenizer and pipeline
# ---------------------------------------------------------------------------

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        f"Model directory not found: {MODEL_PATH}\nSet QWEN_MODEL_PATH to your OpenVINO model directory."
    )

logger.info("Loading tokenizer from %s", MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

logger.info(
    "Loading pipeline on %s (MAX_PROMPT_LEN=%s, MIN_RESPONSE_LEN=%s)",
    DEVICE, MAX_PROMPT_LEN, MIN_RESPONSE_LEN,
)

# Static-shape keys are NPU-only; the GPU/CPU plugins use dynamic shapes and may reject them.
_base_config: dict[str, Any] = (
    {"MAX_PROMPT_LEN": MAX_PROMPT_LEN, "MIN_RESPONSE_LEN": MIN_RESPONSE_LEN} if IS_NPU else {}
)

pipe = None
if NPU_CACHE_DIR:
    # Caches the compiled model blob so later start-ups are much faster.
    try:
        pipe = ov_genai.LLMPipeline(MODEL_PATH, DEVICE, {**_base_config, "CACHE_DIR": NPU_CACHE_DIR})
        logger.info("NPU compile cache enabled at %s", NPU_CACHE_DIR)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CACHE_DIR not accepted (%s); loading without cache", exc)
if pipe is None:
    pipe = ov_genai.LLMPipeline(MODEL_PATH, DEVICE, _base_config)

# A single NPU InferRequest cannot run overlapping generate() calls.
inference_lock = threading.Lock()

app = FastAPI(title="Local Qwen Anthropic API", version="2.0.0")


@app.exception_handler(HTTPException)
async def _http_error(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": "error",
            "error": {
                "type": "invalid_request_error" if exc.status_code < 500 else "api_error",
                "message": str(exc.detail),
            },
        },
    )


# ---------------------------------------------------------------------------
# Anthropic -> Qwen conversion
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM = (
    "You are Qwen, an expert AI coding assistant.\n"
    "- Reply in clear, professional English unless asked otherwise.\n"
    "- Think before answering; check edge cases; give concise explanations of key decisions.\n"
    "- Write correct, complete, runnable code that follows the user's project conventions.\n"
    "- Never invent APIs or libraries; mention dependencies.\n"
    "- Follow tool schemas exactly and use tools only when needed. "
    "Never claim a tool ran or invent file contents or results; the host executes tools."
)


def _system_to_text(system: Any) -> str:
    if isinstance(system, str):
        text = system
    elif isinstance(system, list):
        text = "\n".join(
            str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in system
        )
    else:
        text = str(system) if system else ""
    if SYSTEM_MODE == "lite":
        env = [m.group(0).strip() for m in _ENV_LINE_RE.finditer(text)]
        return "Environment:\n" + "\n".join(env) if env else ""
    if MAX_SYSTEM_CHARS > 0 and len(text) > MAX_SYSTEM_CHARS:
        # Keep the start (identity/rules) and the end (working directory, environment).
        head = int(MAX_SYSTEM_CHARS * 0.4)
        text = text[:head].rstrip() + "\n...\n" + text[-(MAX_SYSTEM_CHARS - head):].lstrip()
    return text


def _tool_result_text(block: dict[str, Any]) -> str:
    content = block.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            str(b.get("text", "")) if isinstance(b, dict) and b.get("type") == "text"
            else ("[Image omitted]" if isinstance(b, dict) and b.get("type") == "image" else str(b))
            for b in content
        )
    content = str(content)
    return f"Error: {content}" if block.get("is_error") else content


def _clean_text(s: str) -> str:
    return _REMINDER_RE.sub("", s).strip() if STRIP_REMINDERS else s


def _convert_message(role: str, content: Any) -> list[dict[str, Any]]:
    """One Anthropic message -> one or more Qwen-template messages (native tool roles)."""
    if isinstance(content, str):
        content = _clean_text(content)
        return [{"role": role, "content": content}] if content else []
    if not isinstance(content, list):
        return [{"role": role, "content": "" if content is None else str(content)}]

    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_msgs: list[dict[str, Any]] = []

    for block in content:
        if not isinstance(block, dict):
            texts.append(str(block))
            continue
        btype = block.get("type")
        if btype == "text":
            texts.append(_clean_text(str(block.get("text", ""))))
        elif btype == "tool_use":
            tool_calls.append({
                "type": "function",
                "function": {"name": str(block.get("name", "")), "arguments": block.get("input") or {}},
            })
        elif btype == "tool_result":
            tool_msgs.append({"role": "tool", "content": _tool_result_text(block)})
        elif btype == "image":
            texts.append("[Image omitted: this local endpoint is text-only.]")
        elif btype in ("thinking", "redacted_thinking"):
            continue
        else:
            texts.append(json.dumps(block, ensure_ascii=False))

    out = list(tool_msgs)  # tool results come first in a user turn
    text = "\n".join(t for t in texts if t)
    if text or tool_calls:
        msg: dict[str, Any] = {"role": role, "content": text}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        out.append(msg)
    return out


def convert_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in body.get("messages", []):
        if not isinstance(item, dict):
            continue
        role = item.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        for msg in _convert_message(role, item.get("content", "")):
            prev = messages[-1] if messages else None
            mergeable = (
                prev is not None
                and prev["role"] == msg["role"]
                and msg["role"] in ("user", "assistant")
                and "tool_calls" not in prev
                and "tool_calls" not in msg
            )
            if mergeable:
                prev["content"] = (prev["content"] + "\n" + msg["content"]).strip()
            else:
                messages.append(msg)
    return messages or [{"role": "user", "content": "Hello"}]


def _shrink_schema(node: Any) -> Any:
    """Drop noise and trim long descriptions inside a JSON schema."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in ("$schema", "additionalProperties"):
                continue
            if k == "description" and isinstance(v, str) and MAX_SCHEMA_DESC_CHARS > 0:
                v = v if len(v) <= MAX_SCHEMA_DESC_CHARS else v[:MAX_SCHEMA_DESC_CHARS].rstrip() + "..."
            out[k] = _shrink_schema(v)
        return out
    if isinstance(node, list):
        return [_shrink_schema(v) for v in node]
    return node


def convert_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Anthropic tool defs -> OpenAI/Qwen function format (what Qwen's template expects)."""
    out = []
    for tool in tools:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        if ALLOWED_TOOLS and tool["name"] not in ALLOWED_TOOLS:
            continue
        desc = str(tool.get("description", ""))
        if MAX_TOOL_DESC_CHARS > 0 and len(desc) > MAX_TOOL_DESC_CHARS:
            desc = desc[:MAX_TOOL_DESC_CHARS].rstrip() + "..."
        out.append({
            "type": "function",
            "function": {
                "name": str(tool["name"]),
                "description": desc,
                "parameters": _shrink_schema(tool.get("input_schema") or {"type": "object", "properties": {}}),
            },
        })
    return out


_TOOL_EXAMPLES = [
    ("Bash", {"command": "ls"}, "list files or run a command"),
    ("Read", {"file_path": "README.md"}, "read a file"),
    ("Glob", {"pattern": "**/*.py"}, "find files by name"),
    ("Grep", {"pattern": "TODO"}, "search inside files"),
]


def tool_guidance(tools: list[dict[str, Any]]) -> str:
    """Explicit rules + one-shot examples; small models need to be shown the exact format."""
    by_name = {t["function"]["name"]: t["function"] for t in tools}
    lines = [
        "TOOL USE RULES:",
        "- You CAN read the user's project files and run commands through tools. When the user "
        "asks about files, folders, code or to run something, CALL A TOOL. Never just explain "
        "which command they could run, and never say you cannot access files.",
        "- To call a tool, reply with ONLY this block and no other text:",
        '<tool_call>\n{"name": "TOOL_NAME", "arguments": {ARGUMENTS}}\n</tool_call>',
        "- When the tool result arrives, answer using that result.",
        "Examples:",
    ]
    for name, args, purpose in _TOOL_EXAMPLES:
        fn = by_name.get(name)
        props = ((fn or {}).get("parameters") or {}).get("properties", {})
        if fn and all(k in props for k in args):
            lines.append(f"- {purpose}: <tool_call>{json.dumps({'name': name, 'arguments': args})}</tool_call>")
    return "\n".join(lines)


def allowed_tool_names(body: dict[str, Any]) -> dict[str, Any]:
    """Map of available tool name -> JSON parameter schema."""
    raw = body.get("tools")
    if not isinstance(raw, list):
        return {}
    return {t["function"]["name"]: t["function"].get("parameters") or {} for t in convert_tools(raw)}


def _few_shot(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A fake exchange showing a real tool call; small models imitate turns far better than rules."""
    bash = next((t["function"] for t in tools if t["function"]["name"] == "Bash"), None)
    if not bash or "command" not in ((bash.get("parameters") or {}).get("properties") or {}):
        return []
    return [
        {"role": "user", "content": "What files are in the current folder?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "Bash", "arguments": {"command": "ls"}}}],
        },
        {"role": "tool", "content": "README.md\nmain.py"},
        {"role": "assistant", "content": "The folder contains README.md and main.py."},
    ]


def render_prompt(messages: list[dict[str, Any]], system: str, tools: list[dict[str, Any]]) -> str:
    """Single system message; tools are injected once by Qwen's own chat template."""
    system_text = DEFAULT_SYSTEM
    if tools:
        system_text += "\n\n" + tool_guidance(tools)
    if system.strip():
        system_text += "\n\nAdditional instructions:\n" + system
    full = [{"role": "system", "content": system_text}, *_few_shot(tools), *messages]
    kwargs = {"tools": tools} if tools else {}
    return tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=True, **kwargs)


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def fit_prompt(
    messages: list[dict[str, Any]], system: str, tools: list[dict[str, Any]], budget: int
) -> tuple[str, int, int]:
    """Render the prompt; if too long, drop the oldest turns instead of failing."""
    prompt = render_prompt(messages, system, tools)
    tokens = count_tokens(prompt)
    dropped = 0
    while tokens > budget and len(messages) > 1:
        drop = max(1, int(len(messages) * (tokens - budget) / tokens))
        drop = min(drop, len(messages) - 1)
        messages = messages[drop:]
        dropped += drop
        # Never start with an orphan assistant turn or tool result.
        while len(messages) > 1 and messages[0]["role"] in ("assistant", "tool"):
            messages = messages[1:]
            dropped += 1
        prompt = render_prompt(messages, system, tools)
        tokens = count_tokens(prompt)
    if tokens > budget:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prompt has {tokens} tokens even after trimming history; limit is {budget}. "
                "The system prompt + tools are too large: lower MAX_TOOL_DESC_CHARS "
                "or raise MAX_PROMPT_LEN."
            ),
        )
    if dropped:
        logger.warning("Dropped %d old message(s) to fit %d-token budget", dropped, budget)
    return prompt, tokens, dropped


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------

TOOL_TAG = "<tool_call>"
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", flags=re.DOTALL)


def parse_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """Return (tool_use blocks, remaining text)."""
    calls: list[dict[str, Any]] = []

    def _sub(match: re.Match) -> str:
        try:
            call = json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.warning("Could not parse tool call JSON: %s", match.group(1)[:200])
            return match.group(0)
        name = call.get("name") if isinstance(call, dict) else None
        if not name:
            return match.group(0)
        args = call.get("arguments", call.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        if not isinstance(args, dict):
            args = {"input": args}
        calls.append({
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:24]}",
            "name": str(name),
            "input": args,
        })
        return ""

    remainder = TOOL_CALL_PATTERN.sub(_sub, text).strip()
    return calls, remainder


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_JSON_DEC = json.JSONDecoder()
_NAME_KEYS = ("name", "tool", "tool_name", "function")
_ARG_KEYS = ("arguments", "parameters", "input", "args")


def _infer_tool(args: Any, tools: dict[str, Any]) -> str | None:
    """Pick the single tool whose parameters fit these argument keys (model forgot the name)."""
    if not isinstance(args, dict) or not args:
        return None
    keys = set(args)
    hits = []
    for name, params in tools.items():
        props = set((params or {}).get("properties", {}))
        required = set((params or {}).get("required", []))
        if keys <= props and required <= keys:
            hits.append(name)
    return hits[0] if len(hits) == 1 else None


def _call_from_obj(obj: Any, tools: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):  # OpenAI style {"function": {...}}
        obj = obj["function"]
    name = next((obj[k] for k in _NAME_KEYS if isinstance(obj.get(k), str)), None)
    if name and name not in tools:
        return None
    if not name:
        # e.g. {"func": {"command": "ls"}}: no tool name, so infer it from the arguments
        for cand in [v for v in obj.values() if isinstance(v, dict)] + [obj]:
            inferred = _infer_tool(cand, tools)
            if inferred:
                return {
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": inferred,
                    "input": cand,
                }
        return None
    args = next((obj[k] for k in _ARG_KEYS if k in obj), {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:24]}", "name": name, "input": args}


def extract_tool_calls(text: str, tool_names: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    # tool_names maps each available tool name to its JSON parameter schema.
    """Strict <tool_call> tags first; then tolerate unclosed, untagged or fenced JSON,
    which small models often produce. Only known tool names are accepted."""
    if not tool_names:
        return [], text.strip()
    calls, remainder = parse_tool_calls(text)
    valid = [c for c in calls if c["name"] in tool_names]
    if valid:
        return valid, remainder
    body = text.strip()
    start = body.find("{")
    if start != -1:
        try:
            obj, end = _JSON_DEC.raw_decode(body[start:])
        except json.JSONDecodeError:
            return [], body
        # Only treat it as a tool call when the JSON is most of the reply, not a snippet in prose.
        if end >= 0.5 * len(body):
            call = _call_from_obj(obj, tool_names)
            if call:
                return [call], ""
    return [], body


def build_gen_config(max_new_tokens: int, stop_sequences: list[str] | None):
    cfg = pipe.get_generation_config()  # keeps EOS / stop ids from the model dir
    cfg.max_new_tokens = max_new_tokens
    # Prompt is already rendered with the chat template; don't let GenAI apply it again.
    if hasattr(cfg, "apply_chat_template"):
        cfg.apply_chat_template = False
    cfg.repetition_penalty = REPETITION_PENALTY
    if TEMPERATURE > 0:
        cfg.do_sample = True
        cfg.temperature = TEMPERATURE
        cfg.top_p = TOP_P
        cfg.top_k = TOP_K
    else:
        cfg.do_sample = False
    if stop_sequences:
        cfg.stop_strings = set(stop_sequences)
        cfg.include_stop_str_in_output = False
    return cfg


def _is_looping(text: str) -> bool:
    """True if the last 160 chars are one short unit (1-40 chars) repeated over and over."""
    if len(text) < 160:
        return False
    tail = text[-160:]
    for p in range(1, 41):
        n = len(tail) // p
        if tail[-p:] * n == tail[len(tail) - n * p:]:
            return True
    return False


def _is_repeating_phrases(text: str) -> bool:
    """True if the recent output is mostly the same few phrases over and over."""
    words = text.split()
    if len(words) < 120:
        return False
    grams = [tuple(words[i:i + 5]) for i in range(len(words) - 4)]
    return len(set(grams)) / len(grams) < 0.35


def run_generation(
    prompt: str,
    cfg: Any,
    on_text: Callable[[str], None] | None = None,
    cancel: threading.Event | None = None,
) -> str:
    recent = ""

    def streamer(chunk: str) -> bool:
        nonlocal recent
        if on_text:
            on_text(chunk)
        recent = (recent + chunk)[-1200:]
        if _is_looping(recent) or _is_repeating_phrases(recent):
            logger.warning("Repetition loop detected; stopping generation early")
            return True
        return bool(cancel and cancel.is_set())  # True = stop generating

    try:
        with inference_lock:
            if cancel and cancel.is_set():
                return ""
            result = pipe.generate(prompt, cfg, streamer)
    except Exception as exc:
        logger.exception("OpenVINO generation failed")
        raise HTTPException(status_code=500, detail=f"OpenVINO generation failed: {exc}") from exc

    if isinstance(result, str):
        return result
    texts = getattr(result, "texts", None)
    return texts[0] if texts else str(result)


# ---------------------------------------------------------------------------
# Anthropic response helpers
# ---------------------------------------------------------------------------

def _stop_reason(blocks: list[dict[str, Any]], output_tokens: int, max_new: int) -> str:
    if any(b.get("type") == "tool_use" for b in blocks):
        return "tool_use"
    return "max_tokens" if output_tokens >= max_new else "end_turn"


def make_response(model: str, content: list[dict[str, Any]], in_tok: int, out_tok: int, stop: str) -> dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _partial_tag_suffix(text: str) -> int:
    """Length of the longest suffix of text that is a proper prefix of <tool_call>."""
    for k in range(min(len(TOOL_TAG) - 1, len(text)), 0, -1):
        if text.endswith(TOOL_TAG[:k]):
            return k
    return 0


def stream_response(
    prompt: str, cfg: Any, in_tok: int, model: str, max_new: int, tool_names: dict[str, Any]
) -> StreamingResponse:
    """True token streaming. Plain text is streamed live; once '<tool_call>' appears the
    rest is buffered and emitted as tool_use blocks when generation finishes."""

    async def events():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancel = threading.Event()

        def put(item: tuple[str, str]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def worker() -> None:
            try:
                put(("done", run_generation(prompt, cfg, lambda c: put(("text", c)), cancel)))
            except HTTPException as exc:
                put(("error", str(exc.detail)))
            except Exception as exc:  # noqa: BLE001
                put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

        yield sse("message_start", {
            "type": "message_start",
            "message": {
                "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant",
                "model": model, "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": in_tok, "output_tokens": 0},
            },
        })

        index = 0
        text_open = False
        # With tools available, hold the whole reply until it ends so a tool call can be
        # recognised cleanly (otherwise junk text is shown before the call). Costs live streaming.
        in_tool = bool(tool_names) and not STREAM_WITH_TOOLS
        pending = ""
        streamed_len = 0

        def text_delta(s: str) -> list[str]:
            nonlocal text_open, streamed_len
            out = []
            if not s:
                return out
            if not text_open:
                out.append(sse("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "text", "text": ""},
                }))
                text_open = True
            out.append(sse("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "text_delta", "text": s},
            }))
            streamed_len += len(s)
            return out

        try:
            while True:
                kind, payload = await queue.get()

                if kind == "text" and not in_tool:
                    pending += payload
                    if not streamed_len and pending.lstrip().startswith(("{", "```json")):
                        # Likely an untagged tool call: buffer everything until generation ends.
                        in_tool = True
                        pending = ""
                        continue
                    pos = pending.find(TOOL_TAG)
                    if pos != -1:
                        emit, pending, in_tool = pending[:pos], "", True
                    else:
                        keep = _partial_tag_suffix(pending)
                        emit, pending = pending[: len(pending) - keep], pending[len(pending) - keep:]
                    for chunk in text_delta(emit):
                        yield chunk

                elif kind == "error":
                    yield sse("error", {"type": "error", "error": {"type": "api_error", "message": payload}})
                    return

                elif kind == "done":
                    full = payload
                    tool_blocks, _ = extract_tool_calls(full, tool_names)
                    if not tool_blocks:
                        for chunk in text_delta(full[streamed_len:]):
                            yield chunk
                    if text_open:
                        yield sse("content_block_stop", {"type": "content_block_stop", "index": index})
                        index += 1
                    for block in tool_blocks:
                        yield sse("content_block_start", {
                            "type": "content_block_start", "index": index,
                            "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}},
                        })
                        yield sse("content_block_delta", {
                            "type": "content_block_delta", "index": index,
                            "delta": {"type": "input_json_delta",
                                      "partial_json": json.dumps(block["input"], ensure_ascii=False)},
                        })
                        yield sse("content_block_stop", {"type": "content_block_stop", "index": index})
                        index += 1

                    out_tok = count_tokens(full)
                    logger.info("Output (%d tokens, %d tool call(s)): %r", out_tok, len(tool_blocks), full[:400])
                    yield sse("message_delta", {
                        "type": "message_delta",
                        "delta": {"stop_reason": _stop_reason(tool_blocks, out_tok, max_new), "stop_sequence": None},
                        "usage": {"output_tokens": out_tok},
                    })
                    yield sse("message_stop", {"type": "message_stop"})
                    return
        finally:
            cancel.set()  # client disconnected or finished: stop the NPU early

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

def check_auth(x_api_key: str | None, authorization: str | None) -> None:
    supplied = x_api_key
    if not supplied and authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if API_KEY and not hmac.compare_digest((supplied or "").encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid API key")


async def read_body(request: Request) -> dict[str, Any]:
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_REQUEST_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")
    raw = await request.body()
    if len(raw) > MAX_REQUEST_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    return body


def build_prompt(body: dict[str, Any]) -> tuple[str, int]:
    messages = convert_messages(body)
    system = _system_to_text(body.get("system", ""))
    raw_tools = body.get("tools")
    tools = convert_tools(raw_tools) if isinstance(raw_tools, list) else []
    prompt, tokens, _ = fit_prompt(messages, system, tools, MAX_PROMPT_LEN)
    return prompt, tokens


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "model_context_tokens": MODEL_CONTEXT_TOKENS,
        "max_prompt_len": MAX_PROMPT_LEN,
        "min_response_len": MIN_RESPONSE_LEN,
        "total_capacity": TOTAL_CAPACITY,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


@app.get("/")
async def root():
    return {
        "service": "Local Qwen Anthropic API",
        "messages_endpoint": "/v1/messages",
        "count_tokens_endpoint": "/v1/messages/count_tokens",
        "health_endpoint": "/health",
    }


@app.post("/v1/messages/count_tokens")
async def count_tokens_endpoint(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    check_auth(x_api_key, authorization)
    body = await read_body(request)
    messages = convert_messages(body)
    raw_tools = body.get("tools")
    tools = convert_tools(raw_tools) if isinstance(raw_tools, list) else []
    prompt = render_prompt(messages, _system_to_text(body.get("system", "")), tools)
    return {"input_tokens": count_tokens(prompt)}


@app.post("/v1/messages")
async def messages(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    check_auth(x_api_key, authorization)
    body = await read_body(request)

    # Tokenizing a big prompt is CPU work; keep it off the event loop.
    prompt, in_tok = await run_in_threadpool(build_prompt, body)

    try:
        requested = int(body.get("max_tokens", DEFAULT_OUTPUT_TOKENS))
    except (TypeError, ValueError):
        requested = DEFAULT_OUTPUT_TOKENS
    max_new = max(1, min(requested, MAX_OUTPUT_TOKENS, TOTAL_CAPACITY - in_tok))

    stops = body.get("stop_sequences")
    stops = [s for s in stops if isinstance(s, str) and s] if isinstance(stops, list) else None
    cfg = build_gen_config(max_new, stops)
    model = str(body.get("model", MODEL_NAME))
    tool_names = allowed_tool_names(body)

    logger.info("Generating: prompt_tokens=%d max_new_tokens=%d stream=%s", in_tok, max_new, bool(body.get("stream")))

    if body.get("stream", False):
        return stream_response(prompt, cfg, in_tok, model, max_new, tool_names)

    text = await run_in_threadpool(run_generation, prompt, cfg)
    out_tok = count_tokens(text)
    tool_blocks, remainder = extract_tool_calls(text, tool_names)
    logger.info("Output (%d tokens, %d tool call(s)): %r", out_tok, len(tool_blocks), text[:400])
    content =([{"type": "text", "text": remainder}] if remainder else []) + tool_blocks
    if not content:
        content = [{"type": "text", "text": text}]

    return JSONResponse(make_response(model, content, in_tok, out_tok, _stop_reason(tool_blocks, out_tok, max_new)))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
