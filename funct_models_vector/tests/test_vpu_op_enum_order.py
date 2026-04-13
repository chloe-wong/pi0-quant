"""Tier-0 canary: Python VPUOp enum must match the Scala ChiselEnum 1:1.

This test parses src/main/scala/atlas/vector/VectorIO.scala and verifies that
the ChiselEnum's `val add, sub, …` declaration order matches
`funct_models_vector.vpu_op.VPUOp`'s integer values. Any reorder (on either
side) fails the test before any downstream laneBox bug can mask it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from funct_models_vector.vpu_op import SCALA_ENUM_ORDER, VPUOp


REPO_ROOT = Path(__file__).resolve().parents[3]
SCALA_VPU_OP_PATH = REPO_ROOT / "src" / "main" / "scala" / "atlas" / "vector" / "VectorIO.scala"


def _parse_scala_enum() -> tuple[str, ...]:
    text = SCALA_VPU_OP_PATH.read_text()
    obj_match = re.search(
        r"object\s+VPUOp\s+extends\s+ChiselEnum\s*\{(.+?)\}",
        text,
        re.DOTALL,
    )
    assert obj_match, f"Could not find VPUOp object in {SCALA_VPU_OP_PATH}"
    body = obj_match.group(1)

    val_match = re.search(
        r"val\s+(.+?)\s*=\s*Value",
        body,
        re.DOTALL,
    )
    assert val_match, f"Could not find `val ... = Value` in VPUOp body"
    raw = val_match.group(1)

    names = [tok.strip() for tok in raw.replace("\n", " ").split(",")]
    names = [n for n in names if n]
    return tuple(names)


def test_scala_vpu_op_file_exists() -> None:
    assert SCALA_VPU_OP_PATH.is_file(), f"Expected {SCALA_VPU_OP_PATH} to exist"


def test_python_enum_matches_scala_order() -> None:
    scala_names = _parse_scala_enum()
    python_names = tuple(op.name for op in VPUOp)
    assert python_names == scala_names, (
        "Python VPUOp order drifted from VectorIO.scala.\n"
        f"  Python: {python_names}\n"
        f"  Scala:  {scala_names}"
    )


def test_static_order_literal_matches_live_enum() -> None:
    # SCALA_ENUM_ORDER is a static tuple used by downstream code. Keep it in
    # sync with the live VPUOp enum so that grep-for-string tools can find
    # op names without importing the enum.
    assert SCALA_ENUM_ORDER == tuple(op.name for op in VPUOp)


def test_enum_values_are_zero_indexed_and_contiguous() -> None:
    for expected_idx, op in enumerate(VPUOp):
        assert op.value == expected_idx, (
            f"VPUOp.{op.name} has value {op.value}, expected {expected_idx}"
        )


@pytest.mark.parametrize("op_name", ["add", "fp8pack", "fp8unpack", "csum", "vliAll"])
def test_critical_ops_are_reachable_by_name(op_name: str) -> None:
    assert hasattr(VPUOp, op_name)
