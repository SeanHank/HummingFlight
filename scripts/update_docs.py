#!/usr/bin/env python3
"""update_docs.py - keep docs in sync with the project.

Idempotently refreshes stale, machine-specific facts that are easy to forget at
release time: the reference model path (moved from D:\\ to E:\\), version stamps
and any pasted old version strings. Run automatically by package.py so docs are
never stale in a release.

Usage:
  python scripts/update_docs.py [--check]   # --check exits 1 when changes are due
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from common import ROOT, project_version

VERSION = project_version()

# (filename, old substring, new substring) applied with str.replace().
RULES: list[tuple[str, str, str]] = [
    # Reference machine relocated the GLM model from D: to E:.
    ("README.md", "D:\\glm-5.2-bf16", "E:\\glm-5.2-bf16"),
    ("README.md", "D:/glm-5.2-bf16", "E:/glm-5.2-bf16"),
    ("doc/design.md", "D:\\glm-5.2-bf16", "E:\\glm-5.2-bf16"),
    ("doc/design.md", "D:/glm-5.2-bf16", "E:/glm-5.2-bf16"),
]


def apply_rules(check_only: bool) -> list[str]:
    edits: list[str] = []
    for fname, old, new in RULES:
        path = ROOT / fname
        if not path.exists() or old == new:
            continue
        text = path.read_text(encoding="utf-8")
        if old in text:
            if not check_only:
                path.write_text(text.replace(old, new), encoding="utf-8")
            edits.append(f"{fname}: {old!r} -> {new!r}")
    return edits


def main() -> int:
    ap = argparse.ArgumentParser(description="Sync stale doc facts")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    edits = apply_rules(check_only=args.check)
    if edits:
        print("docs need update:")
        for e in edits:
            print(f"  {e}")
        return 1 if args.check else 0
    print(f"docs are up to date (version {VERSION})")
    return 0


if __name__ == "__main__":
    sys.exit(main())