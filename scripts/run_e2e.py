#!/usr/bin/env python3
"""run_e2e.py - L5 end-to-end quality gate (real-model only).

Verifies the engine binary + real-model checkpoint with a fast, deterministic
pass: weight structural load (--check-weights) and tokenizer round-trip via the
Python tokenizer server. No forward generation is performed.

Usage:
  python scripts/run_e2e.py --model <dir> [--python <interp>] [--report]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

from common import ROOT, build_exe_path, find_model_dir, project_version, \
    write_report, MarkdownReport  # noqa: E402

PY = sys.executable


def run(cmd: list[str], cwd: pathlib.Path | None = None) -> subprocess.CompletedProcess:
    """Run without a subprocess timeout (the load/tokenizer checks must always finish)."""
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=cwd or ROOT)


def check_weight_loading(exe: pathlib.Path, model_dir: str, report: MarkdownReport) -> bool:
    """Verify weight index + safetensors headers parse (fast structural check)."""
    p = run([str(exe), "--check-weights", "--model", model_dir])
    combined = p.stdout + p.stderr
    ok = p.returncode == 0 and "0 failures" in combined
    report.check("L5 real-model weight index & tensor loading",
                 "PASS" if ok else "FAIL",
                 combined.strip().splitlines()[-1] if combined.strip() else "")
    if not ok:
        report.text("```\n" + combined[-1200:] + "\n```")
    return ok


def check_tokenizer_roundtrip(model_dir: str, python: str, report: MarkdownReport) -> bool:
    """Encode / decode a fixed string via the tokenizer server (no forward pass)."""
    script = ROOT / "tools" / "tokenizer_server.py"
    if not script.exists():
        report.check("L5 tokenizer round-trip", "FAIL", "tools/tokenizer_server.py missing")
        return False
    try:
        proc = subprocess.Popen(
            [python, str(script), model_dir],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        report.check("L5 tokenizer round-trip", "FAIL", f"cannot start: {exc}")
        return False
    ready = ""
    try:
        ready = proc.stdout.readline().strip()  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        ready = ""
    if not ready:
        report.check("L5 tokenizer round-trip", "FAIL", "tokenizer did not become ready")
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        return False
    # encode
    encode_cmd = {"cmd": "encode", "text": "Hello, world!", "add_special": False}
    try:
        proc.stdin.write(json.dumps(encode_cmd) + "\n")  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]
        ids_resp = json.loads(proc.stdout.readline())  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        ids_resp = None
    if not (ids_resp and isinstance(ids_resp.get("ids"), list) and ids_resp["ids"]):
        report.check("L5 tokenizer encode", "FAIL", "no ids returned")
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        return False
    report.check("L5 tokenizer encode", "PASS", f"{len(ids_resp['ids'])} ids")
    # decode
    decode_cmd = {"cmd": "decode", "ids": list(ids_resp["ids"])}
    try:
        proc.stdin.write(json.dumps(decode_cmd) + "\n")  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]
        dec = json.loads(proc.stdout.readline())  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        dec = {}
    try:
        proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass
    ok = isinstance(dec, dict) and "text" in dec
    report.check("L5 tokenizer decode", "PASS" if ok else "FAIL",
                 repr(dec.get("text", ""))[:80])
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="L5 end-to-end quality gate (real-model, fast)")
    ap.add_argument("--model", default=None,
                    help="real model directory (required; set GLM_MODEL_DIR or pass --model)")
    ap.add_argument("--python", default=PY,
                    help="python interpreter for the tokenizer server")
    ap.add_argument("--report", action="store_true",
                    help="write reports/e2e_report.md")
    args = ap.parse_args()

    exe = build_exe_path()
    version = project_version()
    report = MarkdownReport("HummingFlight End-to-End Report (L5)", version).header()

    if exe is None:
        report.check("L5 build artifacts", "FAIL",
                     "engine binary not built; run cmake --build")
        report.summary()
        print(report.render())
        print("E2E FAILED: engine binary missing")
        return 1
    report.check("L5 build artifacts", "PASS", str(exe))

    model_dir = find_model_dir(args.model)
    if not model_dir:
        report.check("L5 real-model directory", "FAIL",
                     "no real model directory found; set GLM_MODEL_DIR or pass --model")
        report.summary()
        print(report.render())
        print("E2E FAILED: no real model directory supplied")
        return 1
    report.check("L5 real-model directory", "PASS", model_dir)

    ok = True
    ok = check_weight_loading(exe, model_dir, report) and ok
    ok = check_tokenizer_roundtrip(model_dir, args.python, report) and ok

    report.summary()
    if args.report:
        out = write_report(report, "e2e_report.md")
        print(f"Report written to {out}")
    else:
        print(report.render())

    if ok:
        print("E2E PASSED (L5)")
        return 0
    print("E2E FAILED (L5)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
