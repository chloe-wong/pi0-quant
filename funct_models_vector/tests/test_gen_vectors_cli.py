"""Phase 1f byte-identity gate for `funct_models_vector.gen_vectors`.

Asserts that:

1. `gen_vectors.main(...)` is fully reproducible — two runs with the same
   seed produce identical output. This is a regression guard against the
   old `gen_vpu_vectors.py` bug where numpy's global RNG was un-seeded.
2. The default invocation (seed 12345, num 58) reproduces the committed
   `src/test/resources/vpu_vectors.txt` byte-for-byte.
3. The csum-only invocation (seed 12345, num 50, ops=csum) reproduces the
   committed `src/test/resources/vpu_vectors_csum.txt` byte-for-byte.
4. The `scripts/gen_vpu_vectors.py` shim produces output identical to
   calling `gen_vectors.main` directly — confirming the historical CLI
   path that the Scala drivers print in their error messages still works.

These are the plan's §4.6 gate. Once green, the reference files are the
new golden; any intentional regen also passes through this test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from funct_models_vector.gen_vectors import main


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_SHIM = REPO_ROOT / "scripts" / "gen_vpu_vectors.py"
REF_VECTORS = REPO_ROOT / "src" / "test" / "resources" / "vpu_vectors.txt"
REF_CSUM = REPO_ROOT / "src" / "test" / "resources" / "vpu_vectors_csum.txt"

# These match the invocations that regenerated the committed files.
DEFAULT_SEED = 12345
DEFAULT_NUM = 58  # committed vpu_vectors.txt has 2 static + 58 dynamic = 60 cases
CSUM_NUM = 50     # committed vpu_vectors_csum.txt has 50 cases


def _run_main(tmp_path: Path, argv: list[str]) -> Path:
    out = tmp_path / "vectors.txt"
    rc = main(["--out", str(out), *argv])
    assert rc == 0, f"main() returned non-zero: {rc}"
    assert out.is_file(), f"main() did not write {out}"
    return out


def test_ref_files_exist() -> None:
    assert REF_VECTORS.is_file(), (
        f"Reference file {REF_VECTORS} is missing. Regenerate with:\n"
        f"  python -m funct_models_vector.gen_vectors "
        f"--out {REF_VECTORS} --seed {DEFAULT_SEED} --num {DEFAULT_NUM}"
    )
    assert REF_CSUM.is_file(), (
        f"Reference file {REF_CSUM} is missing. Regenerate with:\n"
        f"  python -m funct_models_vector.gen_vectors "
        f"--out {REF_CSUM} --seed {DEFAULT_SEED} --num {CSUM_NUM} --ops csum"
    )


def test_main_is_reproducible(tmp_path: Path) -> None:
    """Two runs with the same seed MUST produce identical output.

    Regression guard: the old `scripts/gen_vpu_vectors.py` only seeded
    Python's stdlib `random`, not `numpy.random`, so sin/cos/tanh cases
    drifted between runs.
    """
    a = _run_main(tmp_path / "a", ["--seed", str(DEFAULT_SEED), "--num", "20"])
    b = _run_main(tmp_path / "b", ["--seed", str(DEFAULT_SEED), "--num", "20"])
    assert a.read_bytes() == b.read_bytes(), (
        "gen_vectors.main() is non-deterministic under a fixed seed. "
        "Check that np.random.seed(args.seed) is still being called."
    )


def test_default_matches_committed_reference(tmp_path: Path) -> None:
    """seed=12345 num=58 (all ops) must reproduce src/test/resources/vpu_vectors.txt."""
    out = _run_main(tmp_path, ["--seed", str(DEFAULT_SEED), "--num", str(DEFAULT_NUM)])
    got = out.read_bytes()
    want = REF_VECTORS.read_bytes()
    if got != want:
        pytest.fail(
            f"gen_vectors.main() output drifted from {REF_VECTORS.name}. "
            "If this is intentional (e.g. a lane_box changed its LUT), regen with:\n"
            f"  python -m funct_models_vector.gen_vectors "
            f"--out {REF_VECTORS} --seed {DEFAULT_SEED} --num {DEFAULT_NUM}\n"
            f"len(got)={len(got)}, len(want)={len(want)}"
        )


def test_csum_only_matches_committed_reference(tmp_path: Path) -> None:
    """seed=12345 num=50 ops=csum must reproduce src/test/resources/vpu_vectors_csum.txt."""
    out = _run_main(
        tmp_path,
        ["--seed", str(DEFAULT_SEED), "--num", str(CSUM_NUM), "--ops", "csum"],
    )
    got = out.read_bytes()
    want = REF_CSUM.read_bytes()
    if got != want:
        pytest.fail(
            f"csum gen_vectors output drifted from {REF_CSUM.name}. "
            "If this is intentional (e.g. ColAddVec changed), regen with:\n"
            f"  python -m funct_models_vector.gen_vectors "
            f"--out {REF_CSUM} --seed {DEFAULT_SEED} --num {CSUM_NUM} --ops csum\n"
            f"len(got)={len(got)}, len(want)={len(want)}"
        )


def test_scripts_shim_matches_main(tmp_path: Path) -> None:
    """`python3 scripts/gen_vpu_vectors.py` must produce output identical to
    calling `main` directly. The Scala test drivers' error messages still
    reference the scripts/ path, so that invocation has to keep working."""
    assert SCRIPTS_SHIM.is_file(), f"Shim missing: {SCRIPTS_SHIM}"

    out_via_main = _run_main(tmp_path / "via_main", ["--seed", str(DEFAULT_SEED), "--num", "12"])

    out_via_shim = tmp_path / "via_shim" / "vectors.txt"
    out_via_shim.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_SHIM),
            "--out", str(out_via_shim),
            "--seed", str(DEFAULT_SEED),
            "--num", "12",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"shim exited non-zero. stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert out_via_shim.is_file()

    assert out_via_main.read_bytes() == out_via_shim.read_bytes(), (
        "scripts/gen_vpu_vectors.py shim diverged from "
        "funct_models_vector.gen_vectors.main(). Check the shim's sys.path "
        "plumbing."
    )


def test_default_invocation_emits_static_add_cases(tmp_path: Path) -> None:
    """Sanity: id 0 and id 1 are the hardcoded add cases from _static_cases()."""
    out = _run_main(tmp_path, ["--seed", str(DEFAULT_SEED), "--num", "3"])
    lines = out.read_text().splitlines()
    assert lines[0].startswith('# 0 - "1.0 + 0.0 = 1.0')
    assert any(l.startswith('# 1 - "1.0 + 2.0 = 3.0') for l in lines[:20])
