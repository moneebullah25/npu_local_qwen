"""Standalone sanity test for the Qwen OpenVINO model (no server, no Claude Code).

Usage (stop the server first, it holds the NPU):
    python diag.py NPU 2048      # NPU with a small static context (compiles faster)
    python diag.py GPU           # GPU for comparison
Tries 4 ways of prompting and prints each answer with its speed.
"""
import os
import sys
import time

import openvino_genai as ov_genai
from transformers import AutoTokenizer

MODEL = os.getenv("QWEN_MODEL_PATH", r"D:\npu_local_qwen\models\qwen2.5-coder-3b-npu-ov")
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "NPU"
MAX_PROMPT = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
HERE = os.path.dirname(os.path.abspath(__file__))

tok = AutoTokenizer.from_pretrained(MODEL)

config = {"CACHE_DIR": os.path.join(HERE, f"cache_diag_{DEVICE.lower()}_{MAX_PROMPT}")}
if DEVICE.upper().startswith("NPU"):
    config.update({"MAX_PROMPT_LEN": MAX_PROMPT, "MIN_RESPONSE_LEN": 512})

print(f"Loading on {DEVICE} with {config} ...")
t0 = time.time()
pipe = ov_genai.LLMPipeline(MODEL, DEVICE, config)
print(f"Loaded in {time.time() - t0:.1f}s")

QUESTION = "Write a Python function that reverses a string."
RENDERED = tok.apply_chat_template(
    [{"role": "user", "content": QUESTION}], tokenize=False, add_generation_prompt=True
)


FILLER = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # optional: long-prompt test size


def make_cfg(greedy=False, template=None, max_new=120):
    cfg = pipe.get_generation_config()
    cfg.max_new_tokens = max_new
    if template is not None and hasattr(cfg, "apply_chat_template"):
        cfg.apply_chat_template = template
    if greedy:
        cfg.do_sample = False
        cfg.repetition_penalty = 1.1
    return cfg


def run(label, fn):
    print(f"\n=== {label} ===")
    try:
        t = time.time()
        out = fn()
        text = out if isinstance(out, str) else (out.texts[0] if hasattr(out, "texts") else str(out))
        dt = time.time() - t
        n = len(tok.encode(text, add_special_tokens=False))
        print(f"{n} tokens in {dt:.1f}s ({n / max(dt, 0.01):.1f} tok/s)")
        print(text[:600])
    except Exception as exc:  # noqa: BLE001
        print("FAILED:", exc)


run("A: plain question, GenAI applies the chat template itself",
    lambda: pipe.generate(QUESTION, make_cfg()))

run("B: pre-rendered template, template OFF, greedy + repetition penalty",
    lambda: pipe.generate(RENDERED, make_cfg(greedy=True, template=False)))

run("C: pre-rendered template, template OFF, model's default sampling",
    lambda: pipe.generate(RENDERED, make_cfg(template=False)))


def chat_mode():
    pipe.start_chat()
    try:
        return pipe.generate(QUESTION, make_cfg())
    finally:
        pipe.finish_chat()


run("D: chat mode (start_chat)", chat_mode)

if FILLER > 0:
    src_path = os.path.join(HERE, "qwen_api.py")
    if not os.path.exists(src_path):
        print("\n(E skipped: qwen_api.py not found next to diag.py)")
    else:
        with open(src_path, encoding="utf-8") as fh:
            ids = tok.encode(fh.read(), add_special_tokens=False)[:FILLER]
        long_prompt = tok.apply_chat_template(
            [{"role": "user", "content": tok.decode(ids) + "\n\nIn two sentences, what does the code above do?"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        n_prompt = len(tok.encode(long_prompt, add_special_tokens=False))
        run(
            f"E: long prompt ({n_prompt} tokens of code), greedy (time includes prompt processing)",
            lambda: pipe.generate(long_prompt, make_cfg(greedy=True, template=False, max_new=80)),
        )

print("\nDone.")
