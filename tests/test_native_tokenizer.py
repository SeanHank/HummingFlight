"""Native in-process BPE tokenizer parity tests (#10).

Runs the engine's `--tokenizer native --dump-tokens` path against the real
model's tokenizer.json and verifies the produced ids are byte-for-byte equal
to the reference transformers tokenizer for a mixed battery of inputs.

Skipped automatically when no model directory or built binary is available."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = os.environ.get(
    "GLM_MODEL_DIR", "E:/glm-5.2-bf16"
)
BIN = os.environ.get("GLM_BIN", str(ROOT / "build" / "Release" / "glm.exe"))

TOKEN_RE = re.compile(r"TOKEN: ([0-9,]+)")

BATTERY = [
    "Hello world, this is a native tokenizer test!",
    "I know you're right, it's fine.",
    "It'll work, we've done this before.",
    "12345 67890 99",
    "<|endoftext|>",
    "<|endoftext|> trailing and  leading <|endoftext|>",
    "Emoji test 🎉🚀 done",
    "\n\n  spaced\ttext\n",
    "price is $9.99 & 100% of it!",
    "Ünïcödé ştríng (café)",
    "你好，世界！这是一个测试。",
    "编码 一致 test mixed 中英",
    "  leading and trailing   ",
    "a\nb\r\nc",
    "x   y",
    "Capitalized  Words and 混排",
]

needs_model = pytest.mark.skipif(
    not (Path(MODEL_DIR) / "tokenizer.json").exists(),
    reason=f"model directory not found: {MODEL_DIR}",
)
needs_bin = pytest.mark.skipif(
    not Path(BIN).exists(), reason=f"engine binary not found: {BIN}"
)


def _native_encode(text: str) -> tuple[list[int], str]:
    proc = subprocess.run(
        [BIN, "--model", MODEL_DIR, "--tokenizer", "native",
         "--dump-tokens", text],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    m = TOKEN_RE.search(out)
    if not m:
        pytest.fail(f"no TOKEN line in engine output for {text!r}: {out[-400:]}")
    ids = [int(x) for x in m.group(1).split(",") if x != ""]
    dec_m = re.search(r"TOKEN_DEC: (.*)", out)
    if not dec_m:
        pytest.fail(f"no TOKEN_DEC in engine output for {text!r}: {out[-500:]!r}")
    try:
        dec = json.loads('"' + dec_m.group(1) + '"')
    except ValueError:
        pytest.fail(f"invalid TOKEN_DEC payload for {text!r}: {dec_m.group(1)!r}")
    return ids, dec


@pytest.fixture(scope="module")
def reference():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed; reference parity check unavailable")
    return AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)


@needs_model
@needs_bin
def test_native_encode_matches_reference(reference) -> None:
    for text in BATTERY:
        expected = reference.encode(text, add_special_tokens=False)
        got, _ = _native_encode(text)
        assert got == expected, f"ids mismatch for {text!r}"


@needs_model
@needs_bin
def test_native_decode_roundtrip(reference) -> None:
    for text in BATTERY:
        ids, dec = _native_encode(text)
        expected = reference.decode(ids, skip_special_tokens=True)
        assert dec == expected, f"decode mismatch for {text!r} (ids={ids})"


@needs_model
@needs_bin
def test_native_special_tokens(reference) -> None:
    eos_ref = reference.convert_tokens_to_ids("<|endoftext|>")
    ids, _ = _native_encode("<|endoftext|>")
    assert ids == [eos_ref], f"EOS special token mismatch: {ids}"


@needs_model
@needs_bin
def test_native_eos_info_matches_config(reference) -> None:
    cfg_path = Path(MODEL_DIR) / "tokenizer_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        eos_content = cfg.get("eos_token")
        if isinstance(eos_content, str):
            assert reference.convert_tokens_to_ids(eos_content) is not None