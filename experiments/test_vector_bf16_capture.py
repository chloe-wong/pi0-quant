"""
Regression test for vector-op reference capture.

The VPU functional model receives BF16-quantized operands. Capture mode must
run the PyTorch reference op on those same BF16-quantized values, otherwise
large-angle trig ops compare cos(fp32_input) against VPU_cos(bf16_input).
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")

from pi0_inout.quant_vector import VectorQuantMode
from pi0_inout.quant_types import QuantFormat, quant
from pi0_inout.reference_store import ReferenceStore


def test_capture_mode_reference_uses_bf16_replay_args() -> None:
    store = ReferenceStore()
    mode = VectorQuantMode(capture_mode=True, clean_input_store=store)
    x = torch.tensor([515.0, 58.119140625], dtype=torch.float32)

    with torch.no_grad(), mode:
        y = torch.cos(x)

    stored_args = store.get("unattributed.cos.2")
    stored_out = store.get("unattributed.cos.2.__out__")

    assert stored_args is not None
    assert stored_out is not None

    replay_x = stored_args[0]
    ref_y = stored_out[0]

    assert replay_x.dtype == x.dtype
    assert ref_y.dtype == x.dtype
    assert y.dtype == x.dtype

    # 515 and 58.119... are the exact cases that made the old logs confusing:
    # the saved input is BF16-quantized before reference and FM computation.
    expected_x = quant(x, QuantFormat.BFLOAT16)
    torch.testing.assert_close(
        replay_x,
        expected_x,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(ref_y, torch.cos(replay_x))
    torch.testing.assert_close(y, ref_y)
