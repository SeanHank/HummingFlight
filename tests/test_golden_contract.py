"""Goldens: the recorded inference outputs must stay readable and carry enough
metadata for validate.py to reproduce them faithfully. This test does NOT require
the model; it only pins the recording contract so recordings never drift into
unreproducible shapes."""
from __future__ import annotations

import json
import pathlib

GOLDEN = pathlib.Path(__file__).parent / "golden"


def _cases() -> list[dict]:
    if not GOLDEN.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(GOLDEN.glob("*.json")) if p.name != "manifest.json"]


def test_golden_records_have_required_fields() -> None:
    for case in _cases():
        for field in ("name", "prompt", "max_tokens", "raw", "version", "tokens"):
            assert field in case, f"{field} missing in golden {case.get('name')}"
        assert isinstance(case["tokens"], list) and case["tokens"], \
            f"golden {case['name']} has no tokens"
        assert all(isinstance(t, int) for t in case["tokens"])


def test_golden_tokens_in_vocab_range() -> None:
    # GLM-5.2 vocabulary is 154,880 entries; recorded ids must be valid.
    for case in _cases():
        for tok in case["tokens"]:
            assert 0 <= tok < 154880


def test_golden_manifest_matches_files() -> None:
    manifest = GOLDEN / "manifest.json"
    if not manifest.exists():
        return
    data = json.loads(manifest.read_text(encoding="utf-8"))
    names = set(data.get("cases", []))
    files = {p.name for p in GOLDEN.glob("*.json") if p.name != "manifest.json"}
    assert names == files, f"manifest out of sync: {names ^ files}"