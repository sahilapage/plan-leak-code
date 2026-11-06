"""Shared paths and helpers for the Imagine-Then-Drive project."""
import json
import os
import re
from pathlib import Path

DROOT = Path(os.environ.get("DROOT", "<data>"))
DATA = DROOT / "data"
NUSC = DATA / "nuscenes"
ADV = DATA / "advnusc"
CKPT = DROOT / "ckpt"
RUNS = DROOT / "runs"
IMPROMPTU_3B = CKPT / "impromptu" / "3B_AD_finetune"

# Impromptu / LLaMA-Factory qwen2_vl settings (train/inference_scripts/sglang_infer.py)
IMAGE_MAX_PIXELS = 262144
IMAGE_MIN_PIXELS = 1024
SYSTEM_PROMPT = "You are a helpful assistant."
VISION_TOKENS = "<|vision_start|><|image_pad|><|vision_end|>"

HORIZON = 6  # waypoints, 0.5 s apart -> 3 s
DT = 0.5

_PAIR = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def build_prompt(user_content: str) -> str:
    """Raw chat prompt identical to LLaMA-Factory's qwen2_vl template with <image> replaced in place."""
    body = user_content.replace("<image>", VISION_TOKENS)
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n")


def parse_waypoints(text: str):
    """Return list of 6 (x, y) waypoints (x forward, y left, metres) or None if unparseable."""
    tail = text.split("formatted as [x, y]:")[-1]
    pts = [(float(a), float(b)) for a, b in _PAIR.findall(tail)]
    return pts[:HORIZON] if len(pts) >= HORIZON else None


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
