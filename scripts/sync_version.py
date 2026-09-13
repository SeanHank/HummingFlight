#!/usr/bin/env python3
"""sync_version.py - single source of truth for the HummingFlight version.

The version string is defined ONCE in this script (VERSION below). Running this
script from anywhere pushes the version into every file that displays it:

  * src/version.h                    (C++ build: GLM_VERSION_STRING / MAJOR / MINOR / PATCH)
  * CMakeLists.txt                   (project(VERSION ...) and CPack package version)
  * README.md                        (version badge + "Current release" line)
  * doc/design.md                    (version line)
  * tools/tokenizer_server.py        (__version__)
  * version.txt                      (plain-text, for packaging/CI)

Exit code is 0 on success, 1 on failure. Idempotent and reproducible: running it
twice produces identical output.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

VERSION = "2026.9.0"  # <-- The single source of truth. Bump here, then run this script.
MAJOR, MINOR, PATCH = VERSION.split(".")

ROOT = pathlib.Path(__file__).resolve().parent.parent

_MISSING: list[str] = []


def _resolve_targets() -> dict[str, pathlib.Path]:
    """Compute search target paths. Any path that does not exist yet is appended
    to _MISSING so the script can create the file or report loudly."""
    targets: dict[str, pathlib.Path] = {
        "version.txt": ROOT / "version.txt",
        "src/version.h": ROOT / "src" / "version.h",
        "CMakeLists.txt": ROOT / "CMakeLists.txt",
        "README.md": ROOT / "README.md",
        "doc/design.md": ROOT / "doc" / "design.md",
        "tools/tokenizer_server.py": ROOT / "tools" / "tokenizer_server.py",
    }
    for name, path in list(targets.items()):
        if not path.exists():
            _MISSING.append(name)
    return targets


def sync_version_h() -> None:
    targets = _resolve_targets()
    path = targets["src/version.h"]
    content = f'''#pragma once

// Project version - single source of truth is scripts/sync_version.py.
// Do not edit by hand; run: python scripts/sync_version.py
#define GLM_VERSION_STRING "{VERSION}"
#define GLM_VERSION_MAJOR {MAJOR}
#define GLM_VERSION_MINOR {MINOR}
#define GLM_VERSION_PATCH {PATCH}

#define GLM_PROJECT_NAME "HummingFlight"
#define GLM_PROJECT_DESCRIPTION "GLM-5.2 local full inference engine (BF16, original implementation inspired by colibri)"
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def sync_cmake() -> None:
    targets = _resolve_targets()
    path = targets["CMakeLists.txt"]
    text = path.read_text(encoding="utf-8")
    # Update only the VERSION value inside project(): the OLD regex spanned
    # parens, and since the DESCRIPTION string contains "(", it truncated the
    # project() block. Match the name + version line only, preserving the rest.
    text = re.sub(
        r"project\(glm52_local\s+VERSION\s+[0-9]+\.[0-9]+\.[0-9]+",
        f"project(glm52_local\n    VERSION {VERSION}",
        text,
        count=1,
    )
    # Keep CPack version in sync as well.
    text = re.sub(
        r"set\(CPACK_PACKAGE_VERSION [0-9]+\.[0-9]+\.[0-9]+\)",
        f"set(CPACK_PACKAGE_VERSION {VERSION})",
        text,
    )
    path.write_text(text, encoding="utf-8")


def sync_readme() -> None:
    targets = _resolve_targets()
    path = targets["README.md"]
    text = path.read_text(encoding="utf-8")
    # Version badge
    text = re.sub(
        r'!\[version\]\(https://img\.shields\.io/badge/version-[^)]*\)',
        f"![version](https://img.shields.io/badge/version-{VERSION}-blue)",
        text,
    )
    # "Current release" / "Version" lines
    text = re.sub(
        r"\*\*Current release\*\*: [0-9]+\.[0-9]+\.[0-9]+",
        f"**Current release**: {VERSION}",
        text,
    )
    text = re.sub(
        r"Version [0-9]+\.[0-9]+\.[0-9]+",
        f"Version {VERSION}",
        text,
    )
    path.write_text(text, encoding="utf-8")


def sync_design_doc() -> None:
    targets = _resolve_targets()
    path = targets["doc/design.md"]
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"\*\*Project version\*\*: [0-9]+\.[0-9]+\.[0-9]+",
        f"**Project version**: {VERSION}",
        text,
    )
    text = re.sub(
        r"Version [0-9]+\.[0-9]+\.[0-9]+",
        f"Version {VERSION}",
        text,
    )
    path.write_text(text, encoding="utf-8")


def sync_tokenizer_server() -> None:
    targets = _resolve_targets()
    path = targets["tools/tokenizer_server.py"]
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"__version__ = [\"'][0-9]+\.[0-9]+\.[0-9]+[\"']",
        f'__version__ = "{VERSION}"',
        text,
    )
    path.write_text(text, encoding="utf-8")


def sync_license() -> None:
    """Keep license references project-wide in lockstep with LICENSE (AGPL-3.0)."""
    targets = _resolve_targets()
    readme = targets["README.md"]
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        # Badge
        text = re.sub(
            r'!\[License\]\(https://img\.shields\.io/badge/license-[A-Za-z0-9.+-]*-[a-z]+\.svg\)',
            "![License](https://img.shields.io/badge/license-AGPLv3-red.svg)",
            text,
        )
        if "AGPL" not in text:
            text += "\n> License: GNU Affero General Public License v3.0 (AGPL-3.0) — see LICENSE.\n"
        readme.write_text(text, encoding="utf-8")

    design = targets["doc/design.md"]
    if design.exists():
        text = design.read_text(encoding="utf-8")
        if "AGPL" not in text:
            text += ("\n## License\n\nThis project is licensed under the GNU Affero General "
                     "Public License v3.0 (AGPL-3.0). See LICENSE for the full text.\n")
        design.write_text(text, encoding="utf-8")


def sync_version_txt() -> None:
    targets = _resolve_targets()
    path = targets["version.txt"]
    path.write_text(VERSION + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize the single version number across the project.")
    parser.add_argument("--show", action="store_true", help="Print the current version and exit.")
    parser.add_argument("--check", action="store_true", help="Verify all files already carry the version (no writes).")
    args = parser.parse_args()

    if args.show:
        print(VERSION)
        return 0

    if args.check:
        problems: list[str] = []
        targets = _resolve_targets()
        for name in ("src/version.h", "CMakeLists.txt", "README.md", "doc/design.md", "version.txt"):
            path = targets[name]
            if not path.exists():
                problems.append(f"missing: {name}")
        if _MISSING:
            problems.append(f"targets that will be created: {_MISSING}")
        if problems:
            print("\n".join(problems))
            return 1
        print(f"README: synced to {VERSION}")
        return 0

    sync_version_h()
    sync_cmake()
    sync_readme()
    sync_design_doc()
    sync_tokenizer_server()
    sync_license()
    sync_version_txt()
    print(f"Synced project version to {VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())