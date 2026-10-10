"""STATS/telemetry parser tests for scripts/benchmark.py.

Covers extraction of the engine's `STATS: {...}` JSON line and the aggregate
metric regexes that feed the L4 benchmark report (design.md §10.4)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import benchmark  # noqa: E402
from common import MarkdownReport  # noqa: E402


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


def test_parse_output_ids_prefers_summary_line() -> None:
    out = (
        "[12:36:54.857] [INFO ] [1/4] token=16 (3257.1 s)\n"
        "[12:36:55.000] [INFO ] [2/4] token=29661 (12.0 s)\n"
        "[12:36:55.100] [INFO ] Output ids: 16,29661,90,77\n"
    )
    assert benchmark.parse_output_ids(out) == [16, 29661, 90, 77]


def test_parse_output_ids_falls_back_to_token_markers() -> None:
    out = ("[1/2] token=7 (1.0 s)\n[2/2] token=9 (2.0 s)\n")
    assert benchmark.parse_output_ids(out) == [7, 9]


def test_history_roundtrip_and_comparison_table(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "HISTORY_PATH", tmp_path / "benchmark_history.json")
    config = "--prompt-tokens 3,7,12 --max-tokens 4"
    metrics = {"total_sec": 100.0, "speed_tok_per_s": 0.04,
               "telemetry": {"disk_bytes_read": 1024, "sync_bytes": 0,
                            "prefetch_waste_rate": 0.0}}
    benchmark.record_history("R1", config, metrics, [1, 2, 3, 4])
    benchmark.record_history("R2", config, {"total_sec": 80.0, "speed_tok_per_s": 0.05,
                                            "telemetry": {}}, [1, 2, 3, 4])
    records = benchmark._load_history()["records"]
    assert len(records) == 2
    # re-using a label overwrites that round instead of appending a duplicate
    benchmark.record_history("R1", config, {"total_sec": 90.0}, [1, 2, 3, 4])
    assert len(benchmark._load_history()["records"]) == 2

    report = MarkdownReport("t", "v")
    benchmark.render_comparison(report, config)
    table = "\n".join(report.lines[2:])
    assert "R1" in table and "R2" in table
    assert "90.0 s" in table and "80.0 s" in table
    # fastest run is emphasised, and the shared output is reported identical
    assert "**80.0 s**" in table
    assert "bit-identical" in table


def test_comparison_skips_other_configs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "HISTORY_PATH", tmp_path / "benchmark_history.json")
    benchmark.record_history("R1", "--prompt-tokens 1 --max-tokens 1",
                             {"total_sec": 10.0}, [5])
    report = MarkdownReport("t", "v")
    benchmark.render_comparison(report, "--prompt-tokens 3,7,12 --max-tokens 4")
    assert "R1" not in "\n".join(report.lines[2:])