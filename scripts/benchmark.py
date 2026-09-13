#!/usr/bin/env python3
"""benchmark.py - reproducible performance benchmark (L4 quality gate).

Runs a fixed workload (default 4 tokens from the real model) and validates that
the engine actually produced the requested work with measurable throughput.
Since hardware varies across machines, the PASS bar is hardware-independent but
real: the run must complete, report aggregate timing metrics, and emit a
positive generation speed. No "informational" skip path exists anymore: an
incomplete or failed run is treated as a FAILURE (design.md §10.4).

  python scripts/benchmark.py --model <dir> --report
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

from common import (
    REPORTS,
    build_exe_path,
    find_model_dir,
    project_version,
    write_report,
    MarkdownReport,
)

SPEED_RE = re.compile(r"Speed:\s*([\d.]+)\s*tok/s", re.IGNORECASE)


def run_engine(args: argparse.Namespace, model_dir: str) -> tuple[int, str]:
    exe = build_exe_path()
    if exe is None:
        return 1, "engine binary not built"
    cmd = [str(exe), "--model", model_dir, "--no-gpu",
           "--prompt-tokens", args.prompt_tokens_csv,
           "--max-tokens", str(args.max_tokens)]
    # No subprocess timeout: real-model multi-token runs take minutes to hours
    # on HDD checkpoints and a single run must always yield a result.
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def parse_metrics(output: str) -> dict:
    """Extract aggregate timing metrics printed by main.cpp."""
    metrics: dict = {}
    total = re.search(r"Generated\s+(\d+)\s+tokens,\s+time\s+([\d.]+)\s+s", output, re.IGNORECASE)
    if total:
        metrics["generated_tokens"] = int(total.group(1))
        metrics["total_sec"] = float(total.group(2))
    per_token = re.search(r"avg\s+per[-\s]token\s*[:=]?\s*([\d.]+)\s*ms", output, re.IGNORECASE)
    if per_token:
        metrics["avg_per_token_ms"] = float(per_token.group(1))
    speed = SPEED_RE.search(output)
    if speed:
        metrics["speed_tok_per_s"] = float(speed.group(1))
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="HummingFlight performance benchmark (L4 gate)")
    ap.add_argument("--model", default=None,
                    help="model directory (required; set GLM_MODEL_DIR or pass --model)")
    ap.add_argument("--prompt-tokens-csv", default="3,7,12",
                    help="raw input token ids for the headless benchmark (default: 3,7,12)")
    ap.add_argument("--max-tokens", type=int, default=4,
                    help="tokens to generate")
    ap.add_argument("--report", action="store_true",
                    help="write reports/benchmark_report.md + JSON")
    args = ap.parse_args()

    model_dir = find_model_dir(args.model)
    if not model_dir:
        print("ERROR: no model directory found; set GLM_MODEL_DIR or pass --model")
        return 1

    version = project_version()
    code, out = run_engine(args, model_dir)
    metrics = parse_metrics(out)
    generated = metrics.get("generated_tokens", 0)
    speed = metrics.get("speed_tok_per_s", 0.0)

    report = MarkdownReport("HummingFlight Benchmark Report (L4)", version).header()
    report.section("Config")
    report.table(["Item", "Value"])
    report.row(["Model directory", model_dir])
    report.row(["Prompt tokens", args.prompt_tokens_csv])
    report.row(["Max tokens", str(args.max_tokens)])
    report.text("")

    report.section("Results")
    report.table(["Metric", "Value"])
    for key, unit in (("generated_tokens", ""), ("total_sec", "s"),
                      ("avg_per_token_ms", "ms"), ("speed_tok_per_s", "tok/s")):
        if key in metrics:
            report.row([key, f"{metrics[key]} {unit}".strip()])
    report.text("")
    report.text(out[-1200:])
    report.section("L4 gate decision")
    if code != 0:
        reason = f"engine exited {code}"
        report.check("L4 performance benchmark", "FAIL", reason)
    elif generated != args.max_tokens:
        report.check("L4 performance benchmark", "FAIL",
                     f"expected {args.max_tokens} tokens, engine produced {generated}")
    elif "total_sec" not in metrics:
        report.check("L4 performance benchmark", "FAIL", "no aggregate timing metrics emitted")
    elif speed <= 0.0:
        report.check("L4 performance benchmark", "FAIL", "no positive generation speed reported")
    else:
        report.check("L4 performance benchmark", "PASS",
                     f"{generated} tokens @ {speed:.2f} tok/s")
    report.summary()

    if args.report:
        out_path = write_report(report, "benchmark_report.md")
        raw_path = REPORTS / "benchmark_raw.json"
        raw_path.write_text(
            json.dumps({"version": version, "model_dir": model_dir,
                        "metrics": metrics}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(f"Report written to {out_path}")
        print(f"Raw metrics written to {raw_path}")
    else:
        print(report.render())

    if report.failed:
        print("BENCHMARK FAILED (L4)")
        return 1
    print("BENCHMARK PASSED (L4)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
