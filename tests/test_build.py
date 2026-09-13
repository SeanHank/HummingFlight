"""Black-box checks against the compiled engine binary. Requires the build;
skips cleanly when the binary is absent (e.g. on a freshly cloned machine).
Runs the same binary the CI matrix builds, so this is the link between the
"externally verifiable" requirement and the actual artifacts."""
from __future__ import annotations

import pathlib
import platform
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import build_exe_path  # noqa: E402

_exe = build_exe_path()


def _binary() -> pathlib.Path:
    if _exe is None:
        pytest = __import__("pytest").skip  # keeps import light
        pytest("engine binary not built; run cmake --build first")
    assert _exe is not None
    return _exe


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(_binary()), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=ROOT,
    )


def test_version_exits_zero_and_prints_v() -> None:
    proc = _run("--version")
    assert proc.returncode == 0, proc.stderr
    assert "v" in proc.stdout


def test_usage_lists_new_flags() -> None:
    # Usage is printed to stderr when --help is passed.
    proc = _run("--help")
    combined = proc.stdout + proc.stderr
    assert "--self-test" in combined
    assert "--version" in combined
    assert "--check-weights" in combined


def test_functional_selftest_binary_passes() -> None:
    """Run the full functional suite through its dedicated test binary."""
    exe = _binary()
    tests = exe.parent / "glm_tests.exe" if platform.system() == "Windows" else exe.parent / "glm_tests"
    if tests.exists():
        proc = subprocess.run(
            [str(tests)], capture_output=True, text=True,
            encoding="utf-8", errors="replace", cwd=ROOT,
        )
        assert proc.returncode == 0, proc.stderr
        assert "0 failures" in proc.stdout, proc.stdout


def test_selftest_flag_passes() -> None:
    proc = _run("--self-test")
    assert proc.returncode == 0, proc.stderr
    assert "Self-tests:" in proc.stdout


def test_check_weights_skips_without_model() -> None:
    proc = _run("--check-weights", "--model", "NONEXISTENT_DIR")
    assert proc.returncode != 0  # must fail loudly when the model is missing