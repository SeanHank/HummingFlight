"""STATS/telemetry parser tests for scripts/benchmark.py.

Covers extraction of the engine's `STATS: {...}` JSON line and the aggregate
metric regexes that feed the L4 benchmark report (design.md §10.4)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import benchmark  # noqa: E402


def test_parse_metrics_basic() -> None:
    out = (
        "=== Generation complete ===\n"
        "Generated 4 tokens, time 12.5 s\n"
        "Speed: 0.32 tok/s\n"
    )
    m = benchmark.parse_metrics(out)
    assert m["generated_tokens"] == 4
    assert m["total_sec"] == 12.5
    assert m["speed_tok_per_s"] == 0.32


def test_parse_stats_line() -> None:
    stats = (
        '{"generated_sec":12.5,"cpu_utilization":1.73,'
        '"expert_lookups":100,"expert_lru_hits":90,"expert_lru_misses":10,'
        '"expert_cache_hit_rate":0.9,"prefetch_delivery_rate":0.8,'
        '"prefetch_waste_rate":0.05,"random_io_count":7,"io_read_count":31,'
        '"disk_bytes_read":1048576,"sequential_bytes":524288,'
        '"bytes_read_per_token":1.5,"hdd_latency_ms":3.2,'
        '"ram_bandwidth_mbps":12.3,"expert_reuse_distance_tok":42.0}'
    )
    out = f"[12:00:00] STATS: {stats}\n"
    data = benchmark.parse_stats(out)
    assert data is not None
    assert data["expert_lru_hits"] == 90
    assert data["expert_cache_hit_rate"] == 0.9
    assert data["disk_bytes_read"] == 1048576


def test_parse_metrics_includes_telemetry() -> None:
    out = (
        "Generated 4 tokens, time 1 s\n"
        "Speed: 4 tok/s\n"
        'STATS: {"expert_lru_hits":5,"expert_lru_misses":1}\n'
    )
    m = benchmark.parse_metrics(out)
    assert m["telemetry"]["expert_lru_misses"] == 1


def test_no_stats_line_is_ok() -> None:
    assert benchmark.parse_stats("no stats here\n") is None
    assert benchmark.parse_stats('STATS: {broken json') is None


def test_human_bytes() -> None:
    assert benchmark._human_bytes(512) == "512.00 B"
    assert benchmark._human_bytes(2048) == "2.00 KiB"
    assert benchmark._human_bytes(1048576) == "1.00 MiB"