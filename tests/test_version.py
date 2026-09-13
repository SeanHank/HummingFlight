"""Version consistency checks: every location that displays a version must agree
with the single source of truth (scripts/sync_version.py -> version.txt)."""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def version_from_txt() -> str:
    vf = ROOT / "version.txt"
    assert vf.exists(), "version.txt missing; run `python scripts/sync_version.py` first"
    return vf.read_text(encoding="utf-8").strip()


def test_sync_script_is_source_of_truth() -> None:
    sync = (ROOT / "scripts" / "sync_version.py").read_text(encoding="utf-8")
    m = re.search(r'VERSION = "(\d+\.\d+\.\d+)"', sync)
    assert m, "VERSION constant missing in sync_version.py"
    assert m.group(1) == version_from_txt()


def test_version_header() -> None:
    vf = (ROOT / "src" / "version.h").read_text(encoding="utf-8")
    v = version_from_txt().split(".")
    assert f'GLM_VERSION_STRING "{version_from_txt()}"' in vf
    assert f"GLM_VERSION_MAJOR {v[0]}" in vf
    assert f"GLM_VERSION_MINOR {v[1]}" in vf
    assert f"GLM_VERSION_PATCH {v[2]}" in vf


def test_cmake_project_version() -> None:
    cm = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert re.search(r"VERSION " + re.escape(version_from_txt()), cm)
    assert f"CPACK_PACKAGE_VERSION {version_from_txt()}" in cm


def test_tokenizer_server_version() -> None:
    t = (ROOT / "tools" / "tokenizer_server.py").read_text(encoding="utf-8")
    assert f'__version__ = "{version_from_txt()}"' in t


def test_design_doc_version() -> None:
    d = (ROOT / "doc" / "design.md").read_text(encoding="utf-8")
    assert re.search(r"Version " + re.escape(version_from_txt()), d) or \
           f"**Project version**: {version_from_txt()}" in d


def test_readme_version() -> None:
    r = (ROOT / "README.md").read_text(encoding="utf-8")
    # The version badge is the stable location; SYNC_README above prints a
    # separate "Current release" stamp that sync_version.py manages.
    assert f"version-{version_from_txt()}" in r