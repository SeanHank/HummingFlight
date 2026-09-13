"""common.py - shared helpers for the HummingFlight toolchain.

Centralizes project paths, version, subprocess helpers, and the markdown
report writer used by validate.py, benchmark.py and package.py.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
TOOLS = ROOT / "tools"
TESTS = ROOT / "tests"
DOC = ROOT / "doc"
BUILD = ROOT / "build"
REPORTS = ROOT / "reports"

MODEL_DIR_ENV = "GLM_MODEL_DIR"
# Default candidate model directories, checked in order when --model is absent.
MODEL_CANDIDATES = [
    os.environ.get(MODEL_DIR_ENV, ""),
    r"E:\glm-5.2-bf16",
    r"D:\glm-5.2-bf16",
    "/opt/glm-5.2-bf16",
    "~/glm-5.2-bf16",
]


def project_version() -> str:
    """Reads the version from the single source of truth (version.txt is
    written by scripts/sync_version.py)."""
    vf = ROOT / "version.txt"
    if vf.exists():
        v = vf.read_text(encoding="utf-8").strip()
        if v:
            return v
    # Fallback: import the constant without executing side effects.
    sys.path.insert(0, str(SCRIPTS))
    try:
        import sync_version  # type: ignore[import-not-found]
        return sync_version.VERSION
    except Exception:
        return "2026.9.0"


def find_model_dir(explicit: str | None = None) -> str | None:
    """Locate a usable GLM model directory."""
    candidates = [explicit] if explicit else MODEL_CANDIDATES
    for cand in candidates:
        if not cand:
            continue
        p = pathlib.Path(cand).expanduser()
        if (p / "config.json").exists():
            return str(p)
    return None


def which(name: str) -> str | None:
    return shutil.which(name)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess with sane defaults. Raises on non-zero exit unless
    check=False is passed."""
    check = kwargs.pop("check", True)
    capture = kwargs.pop("capture", True)
    if capture and "stdout" not in kwargs and "stderr" not in kwargs:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    proc = subprocess.run(cmd, check=check, **kwargs)
    return proc


def run_capture(cmd: list[str]) -> tuple[int, str]:
    """Run and return (returncode, combined stderr/stdout).

    No subprocess timeout: the gate pipeline (validate/benchmark/e2e/record)
    drives real-model forward passes that last hours on HDD checkpoints, so a
    single end-to-end run must always be allowed to complete and yield a result.
    """
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode, (proc.stdout or "")


def executable_name() -> str:
    """Platform-specific engine binary name."""
    return "glm.exe" if platform.system() == "Windows" else "glm"


def build_exe_path() -> pathlib.Path | None:
    """Locate the freshly built engine binary for the current platform."""
    exe = executable_name()
    for cand in [
        BUILD / "Release" / exe,
        BUILD / exe,
        BUILD / "bin" / exe,
    ]:
        if cand.exists():
            return cand
    return None


# ---- Model signature (golden-file scoping) ----
# Topology fields that determine inference outputs. Goldens recorded from one
# checkpoint are diffed only against models with the same signature; a
# mismatch is reported as SKIP (never FAIL) by validate.py.
MODEL_DNA_FIELDS = (
    ("H", "hidden_size"),
    ("L", "num_hidden_layers"),
    ("V", "vocab_size"),
    ("Hd", "num_attention_heads"),
    ("Kv", "num_key_value_heads"),
    ("E", "n_routed_experts"),
    ("T", "num_experts_per_tok"),
    ("I", "intermediate_size"),
    ("M", "moe_intermediate_size"),
    ("S", "n_shared_experts"),
)


def model_dna(model_dir: str | None) -> str | None:
    """Deterministic model topology signature, e.g. 'H6144_L78_...'."""
    if not model_dir:
        return None
    config_path = pathlib.Path(model_dir) / "config.json"
    if not config_path.exists():
        return None
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    parts = [f"{label}{cfg.get(key, '?')}" for label, key in MODEL_DNA_FIELDS]
    return "_".join(parts)


# ---- ---- Hash helpers (reproducibility) ---- ----
def sha256_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---- ---- Simple markdown report builder ---- ----
class MarkdownReport:
    def __init__(self, title: str, version: str | None = None):
        self.title = title
        self.version = version or project_version()
        self.lines: list[str] = []
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self._table_open = False

    # ---- sections ----
    def header(self) -> "MarkdownReport":
        self.lines.append(f"# {self.title}")
        self.lines.append("")
        self.lines.append("| Field | Value |")
        self.lines.append("|---|---|")
        self.lines.append(f"| Project version | `{self.version}` |")
        self.lines.append(f"| Platform | `{platform.platform()}` |")
        self.lines.append(f"| Python | `{platform.python_version()}` |")
        self.lines.append(f"| Timestamp (UTC) | `{utc_now()}` |")
        self.lines.append("")
        return self

    def section(self, name: str) -> "MarkdownReport":
        self.close_table()
        self.lines.append(f"## {name}")
        self.lines.append("")
        return self

    # ---- result rows ----
    def table(self, headers: list[str]) -> "MarkdownReport":
        self.close_table()
        self._table_open = True
        self.lines.append("| " + " | ".join(headers) + " |")
        self.lines.append("|" + "---|" * len(headers))
        return self

    def row(self, cells: list[str]) -> "MarkdownReport":
        self.lines.append("| " + " | ".join(cells) + " |")
        return self

    def check(self, name: str, status: str, detail: str = "") -> "MarkdownReport":
        if status == "PASS":
            self.passed += 1
        elif status == "FAIL":
            self.failed += 1
        else:
            self.skipped += 1
        icon = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}.get(status, status)
        self.lines.append(f"- `{icon}` **{name}**{(' -- ' + detail) if detail else ''}")
        return self

    def text(self, content: str) -> "MarkdownReport":
        self.lines.append(content)
        return self

    def close_table(self) -> None:
        self._table_open = False

    def summary(self) -> "MarkdownReport":
        self.close_table()
        self.section("Summary")
        total = self.passed + self.failed + self.skipped
        pct = (self.passed / total * 100.0) if total else 0.0
        self.text(
            f"- **Total checks**: {total}\n"
            f"- **Passed**: {self.passed}\n"
            f"- **Failed**: {self.failed}\n"
            f"- **Skipped**: {self.skipped}\n"
            f"- **Verification rate**: `{pct:.2f}%` of runnable checks passed"
        )
        return self

    def render(self) -> str:
        self.close_table()
        return "\n".join(self.lines) + "\n"


def write_report(report: MarkdownReport, rel_path: str) -> pathlib.Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / rel_path
    out.write_text(report.render(), encoding="utf-8")
    return out