# Contributing to HummingFlight

Thank you for your interest in HummingFlight. This project runs an unquantized,
1.5 TB GLM-5.2 in BF16 on a single laptop — every PR and every release must keep
that promise: **bit-exact fidelity, zero unfinished work, zero unverified claims.**

## Ground rules

1. **No unfinished work ships.** The repository policy (see `doc/design.md`
   §10) forbids incomplete markers of any kind in `src/`, `tests/`, `scripts/`,
   `tools/`, build and CI files. This is enforced by the
   **implementation-completeness audit** (`scripts/audit_completeness.py`),
   which is **exemption-free by construction**: it has no exemption list, no
   per-file exception, and no CLI escape hatch — it even scans its own source.
   If you add a marker that signals unfinished work, the audit fails; the fix is
   to finish the work, not to relax the gate.
2. **100% is the only pass.** The aggregate gate (L0–L6 in `doc/design.md` §10)
   must report 100% PASS. A skip is only acceptable for a level that genuinely
   cannot run without the 1.5 TB checkpoint; once the checkpoint is supplied,
   zero skips are allowed.
3. **Fidelity over speed.** Never change numerical behavior to make the code
   faster. Optimization must preserve the bit-exact output contract that
   `tests/golden/` recordings and the functional self-tests pin down.
4. **Everything is verified.** New features need (a) an L0 functional
   self-test in `src/self_test.cpp` / `tests/run_tests.cpp`, (b) a pytest
   contract test when scripting/CLI behavior is involved, and (c) a structural
   invariant in the audit when the feature crosses engine/CLI boundaries.

## Getting started

Prerequisites (Windows 10/11 x64, or macOS 14+, or Ubuntu):

| Component | Requirement |
|---|---|
| CMake | >= 3.20 |
| Compiler | MSVC (VS 2022), AppleClang, or GCC/Clang |
| Python | 3.10+ with `transformers` and `jinja2` for the tokenizer bridge |
| Model | optional — `E:\glm-5.2-bf16` or any GLM-5.2 BF16 checkpoint for L2–L5 |

Build and run the full test stack:

```powershell
python scripts/sync_version.py              # resolve the single version source
cmake -B build -G "Visual Studio 17 2022" -A x64 -DENABLE_CUDA=OFF
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure   # L0
python -m pytest tests/ -q                              # L1
python scripts/audit_completeness.py --report           # L6, exemption-free
```

For CUDA builds set `-DENABLE_CUDA=ON` (requires the CUDA toolkit; without a
CUDA toolset the engine still builds CPU-only — that is a documented feature
toggle, not a reduced engine).

On macOS and Linux the same stack runs with the platform generator (make /
Ninja): configure single-config with `-DCMAKE_BUILD_TYPE=Release`, then
`cmake --build build -j` and `ctest --test-dir build --output-on-failure`.
The CI matrix (`.github/workflows/ci.yml`) builds and runs L0 + L1 + L6 on
`windows-latest`, `macos-latest` and `ubuntu-latest` on every push.

## How to add a quality gate

Gates are defined in `doc/design.md` §10 and orchestrated by:

| Script | Role |
|---|---|
| `scripts/audit_completeness.py` | L6 marker scan + structural invariants |
| `scripts/validate.py` | L0–L6 aggregate harness (`--report`) |
| `scripts/benchmark.py` | L4 pass/fail benchmark gate |
| `scripts/run_e2e.py` | L5 end-to-end (real-model tokenizer + weight load, fast; requires GLM_MODEL_DIR or `--model`) |
| `scripts/package.py` | one-click release: version → build → all gates → archive |

To add a gate level: document it in §10, add the executor, hook it into
`validate.py` (and `package.py` when it blocks releases), add a pytest contract
test, and keep the README table in sync.

## Commits and pull requests

- Keep changes scoped; a PR should solve one problem.
- Run the full L0 + L1 + L6 stack before opening a PR (CI runs the same gates).
- Do not add generated artifacts (`reports/`, `build/`, `dist/`) unless the PR
  explicitly updates pinned goldens or release reports.
- Commit messages follow conventional style: `area: summary`. Pinned goldens
  (`tests/golden/`) are part of the change when behavior is re-verified, and
  only when the diff reproduces exactly on the reference machine.
- Reports and release artifacts land with the tag/release, not with code edits.

## Reporting issues

Bug reports should include: the OS/compiler/build flags, the exact command,
the tail of the engine log, and whether the failure reproduces from a clean
build. Performance reports should include storage, RAM, and CPU model.
Security issues: report privately and do not include model weights in any
attachment.

## Licensing

HummingFlight is licensed under the **GNU Affero General Public License v3.0**
(`LICENSE`); contributions are accepted under the same license. The GLM-5.2
model weights are separate and must be obtained from their owners; see
`DISCLAIMER.md` and `CREDITS.md`. By contributing you agree your changes are
published under AGPL-3.0.