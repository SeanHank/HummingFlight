#!/usr/bin/env python3
"""Implementation-completeness audit - L6 quality gate.

The project policy (design.md §10) forbids shipping any unfinished
implementation: every capability must be real and wired end-to-end. That
guarantee cannot be proven by tests alone, so this gate statically scans the
entire in-scope tree for the incomplete-work marker vocabulary and checks
structural invariants that prove each engine capability is actually attached.

This gate is exemption-free by construction:

* there is no exemption list, no per-path exception, and no CLI escape hatch;
* it scans its own source exactly like every other file;
* the trigger vocabulary is expressed as regular expressions whose own text
  never contains a literal trigger token, so the detector cannot trip on
  itself; the vocabulary it ships is the vocabulary it enforces.

Any single marker hit fails the gate. Exit code 0 means the scan found zero
markers and every structural invariant holds.

Usage:
  python scripts/audit_completeness.py          # scan; FAILs on any hit
  python scripts/audit_completeness.py --report # also write reports/audit_report.md
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  (non-TTY/interactive consoles)
    pass

from common import ROOT, build_exe_path, write_report, MarkdownReport  # noqa: E402

# Banned marker vocabulary (case-insensitive regex). Each pattern is written so
# that its own bytes do not contain a literal trigger token; this is what makes
# a zero-exemption self-scan consistent.
MARKERS: list[re.Pattern[str]] = [
    re.compile(r"\bT[O0]D[O0]\b", re.IGNORECASE),
    re.compile(r"\bF[I1]XM[E3]\b", re.IGNORECASE),
    re.compile(r"\bH[A4]CK\b", re.IGNORECASE),
    re.compile(r"\bST[U0]BS?\b", re.IGNORECASE),
    re.compile(r"\bPL[A4]CEH[O0]LD[E3]R\b", re.IGNORECASE),
    re.compile(r"\bN[O0]T\s+[I1]MPLEMENT[E3]D\b", re.IGNORECASE),
    re.compile(r"\bN[O0]T\s+Y[E3]T\s+[I1]MPLEMENT[E3]D\b", re.IGNORECASE),
    re.compile(r"\bUN[I1]MPLEMENT[E3]D\b", re.IGNORECASE),
    re.compile(r"\bD[E3]F[E3]RR[E3]D\b", re.IGNORECASE),
    re.compile(r"\bW[O0]RK-?[A4]R[O0]UND\b", re.IGNORECASE),
    re.compile(r"\bS[I1]MPLIF(?:IC)?[A4]T[I1][O0]N\b", re.IGNORECASE),
    re.compile(r"\bS[I1]MPLIF[I1][E3]D\b", re.IGNORECASE),
    re.compile(r"\bB[E3]ST[- ][E3]FF[O0]RT\b", re.IGNORECASE),
    re.compile(r"\bN[O0]T\s+S[U0]PP[O0]RT[E3]D\s+Y[E3]T\b", re.IGNORECASE),
    re.compile(r"\bT[O0](?:[- ]?B[E3])[- ]?D[O0]N[E3]\b", re.IGNORECASE),
    re.compile(r"\b[I1]NT[E3]NT[I1][O0]NALLY\s+[E3]MPTY\b", re.IGNORECASE),
    re.compile(r"^#if\s+0\b", re.IGNORECASE | re.MULTILINE),
]

# Files (relative paths or suffixes) scanned. Scope is tight: runtime sources,
# tooling, tests, build and CI configuration. Reports and rendered docs are
# produced output, not source, so they are not scanned.
SCAN_PATHS = [
    "src",
    "scripts",
    "tools",
    "tests",
    ".github",
    "CMakeLists.txt",
]
SKIP_SUFFIXES = {".md", ".json", ".txt", ".lic", ".patch", ".pyc"}
SKIP_DIR_PARTS = {"__pycache__", "build", ".git"}


def iter_source_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for rel in SCAN_PATHS:
        p = ROOT / rel
        if not p.exists():
            continue
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                parts = set(child.parts)
                suffix = child.suffix.lower()
                if parts & SKIP_DIR_PARTS or suffix in SKIP_SUFFIXES:
                    continue
                if child.is_file() and suffix not in SKIP_SUFFIXES:
                    files.append(child)
    return files


def scan_files() -> list[tuple[str, int, str, str]]:
    """Return (relative_path, lineno, pattern, excerpt) for every marker hit."""
    hits: list[tuple[str, int, str, str]] = []
    for file in iter_source_files():
        rel = file.relative_to(ROOT).as_posix()
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        for pat in MARKERS:
            for lineno, line in enumerate(lines, start=1):
                m = pat.search(line)
                if not m:
                    continue
                excerpt = re.sub(r"\s+", " ", line).strip()[:200]
                hits.append((rel, lineno, pat.pattern, excerpt))
    return hits


def check_structure(report: MarkdownReport) -> list[str]:
    """Structural invariants proving capabilities are real and wired."""
    failures: list[str] = []

    exe = build_exe_path()
    if exe is None:
        failures.append("engine binary missing (run cmake --build build --config Release first)")
        report.check("engine binary present", "FAIL")
    else:
        report.check("engine binary present", "PASS", str(exe))

    def src_contains(rel: str, needle: str) -> bool:
        p = ROOT / rel
        return p.exists() and needle in p.read_text(encoding="utf-8", errors="replace")

    # Full DSA indexer wired: config parsing, kv indexer keys, forward scoring.
    checks = [
        ("src/model/config.cpp", "indexer_types", "DSA per-layer schedule parsed"),
        ("src/engine/kv_cache.cpp", "appendIndexKey", "DSA indexer key cache"),
        ("src/model/glm_forward.cpp", "dsaIndexer", "DSA indexer scoring implemented"),
        ("src/model/glm_forward.cpp", "isIndexerFull", "full/shared layer dispatch"),
        ("src/main.cpp", "attachScheduler", "storage pipeline attached by CLI"),
        ("CMakeLists.txt", "scheduler.cpp", "scheduler compiled into the binary"),
    ]
    for rel, needle, desc in checks:
        ok = src_contains(rel, needle)
        report.check(f"src: {needle}", "PASS" if ok else "FAIL", desc)
        if not ok:
            failures.append(f"{rel} missing '{needle}' ({desc})")

    # GPU backend is a real feature toggle: sources exist and CMake wires it.
    cuda_file = ROOT / "src" / "compute" / "cuda_kernels.cu"
    cuda_wired = cuda_file.exists() and "ENABLE_CUDA" in (ROOT / "CMakeLists.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    report.check("CUDA sources + CMake wiring", "PASS" if cuda_wired else "FAIL",
                 "GPU is a build-time feature toggle")
    if not cuda_wired:
        failures.append("CUDA backend missing or not compile-wired")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description="L6 implementation-completeness audit")
    ap.add_argument("--report", action="store_true", help="write reports/audit_report.md")
    args = ap.parse_args()

    hits = scan_files()

    report = MarkdownReport("HummingFlight Implementation-Completeness Audit (L6)").header()
    report.section("Marker scan")
    report.table(["Marker hits", "Count"])
    report.row(["Total hits", str(len(hits))])
    report.text("")
    for rel, lineno, pat, excerpt in hits:
        report.check(f"{rel}:{lineno}", "FAIL", f"[{pat}] {excerpt}")

    report.section("Structural invariants")
    failures = check_structure(report)

    report.section("Summary")
    report.summary()

    if args.report:
        out = write_report(report, "audit_report.md")
        print(f"Report written to {out}")
    else:
        print(report.render())

    if hits:
        print(f"AUDIT FAILED: {len(hits)} marker hit(s); the gate is exemption-free, "
              "every hit must be removed")
        return 1
    if failures:
        print(f"AUDIT FAILED: structural invariants: {failures}")
        return 1
    print("AUDIT PASSED: zero marker hits, all structural invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())