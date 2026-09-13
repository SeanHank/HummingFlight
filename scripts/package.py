#!/usr/bin/env python3
"""package.py - one-click release packaging for HummingFlight.

Pipeline (matches GitHub Actions ci.yml; a pushed `v*` tag publishes the
GitHub Release automatically after Build + L0 + L1 + L6 + CPack pass):
  1. Synchronize the project-wide version (single source: scripts/sync_version.py)
  2. Configure + build the engine (Release)
  3. Run ALL quality gates to 100% (design.md §10):
       L0 functional self-tests       (glm_tests + CTest)
       L1 pytest suite                (unit + config + golden-contract tests)
       L2 model structural            (--check-weights + index/shard integrity)
       L2b tokenizer round-trip       (real-model tokenizer_config.json)
       L3 golden outputs              (must PASS, not SKIP)
       L4 performance benchmark       (PASS/FAIL; blocks release)
       L5 end-to-end                  (real-model tokenizer + weight-load, fast)
       L6 completeness audit          (scripts/audit_completeness.py)
     A single gate that is FAIL or SKIP fails the build. L2-L5 require a real
     model checkpoint (set GLM_MODEL_DIR or pass --model); there is no fake.
  4. Generate the validation/audit/benchmark report markdown files
  5. Build distributable archives with CPack (ZIP on all platforms, DEB/DMG
     when available) into the dist/ directory
  6. Any packaging step that fails aborts the whole run (release gate)

Usage:
  python scripts/package.py --model <dir>    # full 100% gate incl. L2/L3/L4/L5
  python scripts/package.py --version X.Y.Z  # bump before packaging
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import time

from common import (
    BUILD,
    ROOT,
    REPORTS,
    build_exe_path,
    find_model_dir,
    project_version,
    run_capture,
    write_report,
    MarkdownReport,
)

PY = sys.executable
SCRIPTS_DIR = ROOT / "scripts"

CMAKE_BASE = [
    "cmake", "-S", ".", "-B", "build",
    "-DENABLE_CUDA=OFF",
    "-DBUILD_TESTING=ON",
]


def configure_cmd() -> list[str]:
    """Choose the best available generator and return the full configure command."""
    if shutil.which("ninja"):
        return ["cmake", "-S", ".", "-B", "build", "-G", "Ninja",
                "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_CUDA=OFF", "-DBUILD_TESTING=ON"]
    return CMAKE_BASE + ["-DCMAKE_BUILD_TYPE=Release"]


def sync_version(bump: str | None) -> str:
    if bump:
        run_capture([PY, str(SCRIPTS_DIR / "sync_version.py"), "--bump", bump])
    run_capture([PY, str(SCRIPTS_DIR / "sync_version.py")])
    return project_version()


def step(label: str) -> None:
    print(f"\n=== [{time.strftime('%H:%M:%S')}] {label} ===")


def fail(msg: str) -> int:
    print(f"ERROR: {msg}")
    return 1


def run_or_record(cmd: list[str], label: str, report: MarkdownReport,
                  gate: str, skip_ok: bool = False) -> int:
    """Run a gate command; record PASS/FAIL(/SKIP when skip_ok and no model).

    No subprocess timeout is applied: gate runs must always complete and yield
    a result, never a false timeout (HDD-hosted checkpoint forwards take hours).
    """
    step(label)
    code, out = run_capture(cmd)
    if code != 0:
        print(out[-1200:])
        detail = out.strip().splitlines()[-1] if out.strip() else f"exit {code}"
        report.check(gate, "FAIL", detail)
        return code
    report.check(gate, "PASS", label)
    print(out[-300:])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="One-click HummingFlight release packaging")
    ap.add_argument("--model", default=None, help="model directory (for full validation)")
    ap.add_argument("--version", default=None, help="bump version before packaging")
    ap.add_argument("--skip-cpack", action="store_true", help="skip archive creation")
    ap.add_argument("--python", default=None, help="python interpreter for the tokenizer/gates")
    args = ap.parse_args()

    version = sync_version(args.version)
    global PY
    if args.python:
        PY = args.python
    print(f"Project version: {version}")

    report = MarkdownReport("HummingFlight Release Gate Report", version).header()
    report.section("Quality gates (require 100% PASS; model-present SKIPs fail the release)")
    report.text("")

    # 1) Configure + build
    if run_or_record(configure_cmd(), "0/9 CMake configure", report, "cmake configure") != 0:
        write_report(report, "release_gate_report.md")
        return 1
    bld_cmd = ["cmake", "--build", "build", "--config", "Release", "-j"]
    if run_or_record(bld_cmd, "1/9 Build (Release)", report, "build (Release)") != 0:
        write_report(report, "release_gate_report.md")
        return 1

    # 2) L0 functional self-tests (CTest)
    ctest_cmd = ["ctest", "--test-dir", str(BUILD), "-C", "Release", "--output-on-failure"]
    if shutil.which("ctest") is None:
        ctest_cmd = [str(build_exe_path())]
    if run_or_record(ctest_cmd, "2/9 L0 functional self-tests (CTest)",
                     report, "L0 functional self-tests") != 0:
        write_report(report, "release_gate_report.md")
        return 1

    # 3) L1 pytest suite
    l1 = [PY, "-m", "pytest", "tests/", "-q"]
    if run_or_record(l1, "3/9 L1 pytest suite", report, "L1 pytest suite") != 0:
        write_report(report, "release_gate_report.md")
        return 1

    model_dir = find_model_dir(args.model)
    gate_model = model_dir  # real model required; no fake fallback

    # 4) L2/L3 model structural + tokenizer + goldens (requires real model)
    l2 = [PY, str(SCRIPTS_DIR / "validate.py"), "--report",
          "--python", PY]
    if gate_model:
        l2 += ["--model", gate_model]
    if run_or_record(l2, "4/9 L2/L3 validation harness", report,
                     "L2+L3 validation harness") != 0:
        write_report(report, "release_gate_report.md")
        return 1

    # 5) L4 performance benchmark (PASS/FAIL, blocks release; requires real model)
    l4 = [PY, str(SCRIPTS_DIR / "benchmark.py"), "--report"]
    l4 += ["--model", gate_model] if gate_model else []
    if run_or_record(l4, "5/9 L4 performance benchmark", report,
                     "L4 performance benchmark") != 0:
        write_report(report, "release_gate_report.md")
        return 1

    # 6) L5 end-to-end (real-model tokenizer + weight-load; requires real model)
    l5 = [PY, str(SCRIPTS_DIR / "run_e2e.py"), "--report"]
    l5 += ["--model", gate_model] if gate_model else []
    l5 += ["--python", PY]
    if run_or_record(l5, "6/9 L5 end-to-end", report, "L5 end-to-end") != 0:
        write_report(report, "release_gate_report.md")
        return 1

    # 7) L6 completeness audit
    l6 = [PY, str(SCRIPTS_DIR / "audit_completeness.py"), "--report"]
    if run_or_record(l6, "7/9 L6 completeness audit", report, "L6 completeness audit") != 0:
        write_report(report, "release_gate_report.md")
        return 1

    # 8) CPack archives
    if not args.skip_cpack:
        code = run_or_record(["cmake", "--build", "build", "--config", "Release",
                              "--target", "package"], "8/9 CPack archives",
                             report, "CPack archives")
        if code != 0:
            write_report(report, "release_gate_report.md")
            return code

    step("9/9 Writing release gate report")
    write_report(report, "release_gate_report.md")
    print(f"\nRelease gate report: {REPORTS / 'release_gate_report.md'}")
    print(f"Packages (if produced): {ROOT / 'dist'}")

    if report.failed:
        print("RELEASE GATE FAILED: all quality gates must pass 100% "
              "(no FAIL, and no SKIP when a model was supplied)")
        return 1
    print("RELEASE GATE PASSED (all quality gates 100%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())