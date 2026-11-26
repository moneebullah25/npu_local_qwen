"""Compare repetition_penalty values with greedy decoding (looks for stray '1' glitches).

Usage (stop the server first):  python diag_rp.py
Reuses the 8192-token NPU cache that diag.py already built, so it loads quickly.
"""
import os
import time

import openvino_genai as ov_genai
from transformers import AutoTokenizer

MODEL = os.getenv("QWEN_MODEL_PATH", r"D:\npu_local_qwen\models\qwen2.5-coder-3b-npu-ov")
HERE = os.path.dirname(os.path.abspath(__file__))
MAX_PROMPT = 8192

tok = AutoTokenizer.from_pretrained(MODEL)
pipe = ov_genai.LLMPipeline(
    MODEL,
    "NPU",
    CACHE_DIR=os.path.join(HERE, f"cache_diag_npu_{MAX_PROMPT}"),
    MAX_PROMPT_LEN=MAX_PROMPT,
    MIN_RESPONSE_LEN=512,
)

short = tok.apply_chat_template(
    [{"role": "user", "content": "Write a Python function that reverses a string."}],
    tokenize=False, add_generation_prompt=True,
)

with open(os.path.join(HERE, "qwen_api.py"), encoding="utf-8") as fh:
    code = tok.decode(tok.encode(fh.read(), add_special_tokens=False)[:6000])
long_ = tok.apply_chat_template(
    [{"role": "user", "content": code + "\n\nIn two sentences, what does the code above do?"}],
    tokenize=False, add_generation_prompt=True,
)

for rp in (1.0, 1.05, 1.1):
    for label, prompt, n in (("short", short, 120), ("long", long_, 80)):
        cfg = pipe.get_generation_config()
        cfg.max_new_tokens = n
        cfg.do_sample = False
        cfg.repetition_penalty = rp
        if hasattr(cfg, "apply_chat_template"):
            cfg.apply_chat_template = False
        t = time.time()
        out = pipe.generate(prompt, cfg)
        text = out if isinstance(out, str) else out.texts[0]
        print(f"\n=== repetition_penalty={rp} | {label} prompt | {time.time() - t:.1f}s ===")
        print(text[:500])

print("\nDone.")
