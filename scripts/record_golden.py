#!/usr/bin/env python3
"""record_golden.py - record deterministic golden inference outputs.

Runs the engine in greedy mode on a small set of fixed prompts and stores the
resulting token ids in tests/golden/*.json. The recorded files are pinned by
commit so that validate.py can reproduce and diff them on any machine (with the
model available), giving an externally verifiable, reproducible correctness
check of the full forward pass.

WARNING: this harness must be run on the machine that owns the model weights.
Each prompt triggers a full forward pass, which on HDD hardware takes hours.
Record small prompts *once*; afterwards only validate.py is needed to verify.
No timeout is applied: a single recording run always completes and returns a result.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from common import (
    TESTS,
    build_exe_path,
    executable_name,
    find_model_dir,
    model_dna,
    project_version,
    run_capture,
)

GOLDEN_DIR = TESTS / "golden"

# Fixed prompts + token counts. Keep tokens small (verify cheaply).
# The first entry is the canonical smoke prompt used everywhere (validate.py,
# package.py, CI). Chat-template encoding (no --raw) exercises the full stack
# including the model's default system prompt, so the model emits a real
# completion instead of an immediate EOS on a bare raw prompt.
DEFAULT_CASES = [
    {"name": "smoke", "prompt": "hi", "max_tokens": 1, "raw": False},
    {"name": "greeting", "prompt": "Hello", "max_tokens": 4, "raw": False},
]


def record(args: argparse.Namespace) -> int:
    model_dir = find_model_dir(args.model)
    if not model_dir:
        print("ERROR: no model directory found; pass --model (GLM_MODEL_DIR) ")
        return 1
    exe = build_exe_path()
    if exe is None:
        print("ERROR: engine binary not built")
        return 1

    cases = json.loads(args.cases) if args.cases else DEFAULT_CASES
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    entries = []
    for case in cases:
        prompt = case["prompt"]
        max_tokens = int(case.get("max_tokens", 1))
        raw = bool(case.get("raw", False))
        prompt_tokens = case.get("prompt_tokens")
        if prompt_tokens:
            cmd = [str(exe), "--model", model_dir, "--prompt-tokens",
                   ",".join(str(x) for x in prompt_tokens),
                   "--max-tokens", str(max_tokens)]
        else:
            cmd = [str(exe), "--model", model_dir, "--prompt", prompt,
                   "--max-tokens", str(max_tokens)]
            if raw:
                cmd += ["--raw"]
        print(f"Recording golden [{case['name']}] prompt={prompt!r} tokens={max_tokens} "
              f"raw={raw} ...")
        code, out = run_capture(cmd + ["--python", args.python])
        if code != 0:
            print(f"ERROR: engine exited with {code}:\n{out[-800:]}")
            return 1
        tokens: list[int] = []
        for line in out.splitlines():
            marker = "] token="
            if marker in line:
                try:
                    tokens.append(int(line.split(marker, 1)[1].split(" (")[0]))
                except ValueError:
                    continue
        if not tokens:
            print(f"ERROR: no tokens parsed from output:\n{out[-800:]}")
            return 1
        record = {
            "name": case["name"],
            "prompt": prompt,
            "max_tokens": max_tokens,
            "raw": raw,
            "version": project_version(),
            "signature": model_dna(model_dir),
            "tokens": tokens,
        }
        if prompt_tokens:
            record["prompt_tokens"] = list(prompt_tokens)
        out_path = GOLDEN_DIR / f"{case['name']}.json"
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"  recorded {len(tokens)} tokens -> {out_path}")
        entries.append(out_path.name)

    manifest = GOLDEN_DIR / "manifest.json"
    manifest.write_text(
        json.dumps({"version": project_version(), "cases": entries}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Golden manifest: {manifest}")
    print("Next: run `python scripts/validate.py --model <dir>` to verify.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Record golden inference outputs")
    ap.add_argument("--model", default=None, help="model directory")
    ap.add_argument("--cases", default=None,
                    help='JSON array of {"name","prompt","max_tokens"} (default: smoke cases)')
    ap.add_argument("--python", default=sys.executable,
                    help="python interpreter for the tokenizer bridge")
    args = ap.parse_args()
    return record(args)


if __name__ == "__main__":
    sys.exit(main())