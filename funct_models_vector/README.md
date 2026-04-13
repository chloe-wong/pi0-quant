# `funct_models_vector`

Pure-Python functional model of the Atlas vector processing unit,
mirroring the Scala RTL in `src/main/scala/atlas/vector/` op-by-op.
One call, one result. The cycle-accurate driver
(`VectorEngineModel.cycle_step`) is intentionally still a stub.

## Op coverage

- **Bit-exact against RTL**: `add`, `sub`, `mul`, `pairmax`, `pairmin`,
  `relu`, `square`, `cube`, `mov`, `rsum`, `rmax`, `rmin`, `csum`,
  `cmax`, `cmin`, the `vli` family (`vliOne`, `vliCol`, `vliRow`,
  `vliAll`), `fp8pack`, `fp8unpack`, and the LUT-backed transcendentals
  `rcp`, `sqrt`, `log`, `sin`, `cos`. Tier 3 verifies the snapshot-
  covered ops against a snapshot of Scala RTL output; `fp8pack` and
  `fp8unpack` are verified separately by `test_fp8_pack.py` /
  `test_fp8_unpack.py` and the torch adapter tests, since they are
  phased 2-to-1 / 1-to-2 ops and are not part of the single-pulse
  `vpu_vectors.txt` corpus that Tier 3 reads.
- **Python math fallback**: `tanh`, `exp`, `exp2`. These route through
  `_legacy_math_fallback` in `vector_engine_model.py` because the
  Group D HardFloat lane boxes are not yet implemented.
  Tier 3 marks `tanh` xfail; `exp` and `exp2` happen to match the
  committed corpus today but that is incidental, not guaranteed.

## Torch API (most common)

`VectorRTLFunctions` is a thin torch wrapper. It handles BF16
quantization, lane chunking, and zero-padding for you.

```python
import torch
from funct_models_vector.vector_rtl_forward import VectorRTLFunctions

vpu = VectorRTLFunctions(num_lanes=16)

a = torch.randn(48)
b = torch.randn(48)

# Pointwise ops preserve the input shape. Length need not divide
# num_lanes; the adapter zero-pads internally.
c = vpu.add(a, b)              # (48,)
d = vpu.mul(a, b)              # (48,)
e = vpu.relu(a)                # (48,)
f = vpu.sqrt(torch.abs(a))     # (48,)

# Row reductions: last dim must be a multiple of num_lanes.
rows = torch.randn(4, 16)
row_sums = vpu.rsum(rows)      # (4,)

# Column reductions: input (N, num_lanes), output (num_lanes,).
col_sums = vpu.csum(rows)      # (16,)
col_max  = vpu.cmax(rows)      # (16,)

# FP8 pack/unpack (E4M3 payload, E8M0 scale byte).
pair = torch.randn(2, 16)
packed = vpu.fp8_pack(pair, scale_e8m0=127)        # (16,) int32
unpacked = vpu.fp8_unpack(packed, scale_e8m0=127)  # (2, 16) float
```

Available methods: `add`, `sub`, `mul`, `pairwise_max`, `pairwise_min`,
`relu`, `rcp`, `sqrt`, `sin`, `cos`, `log2`, `tanh`, `exp`, `exp2`,
`square`, `cube`, `rsum`, `rmax`, `rmin`, `csum`, `cmax`, `cmin`,
`fp8_pack`, `fp8_unpack`. The torch method is spelled `log2()` because
the lane box computes base-2 logarithm; the underlying raw op is `log`.

## Raw bit-level API

`VectorEngineModel.execute()` is the underlying dispatcher. It takes
and returns lists of 16-bit BF16 bit patterns (`list[int]`), which is
what you want for hex-exact control or for mirroring the Scala test
driver directly.

```python
from funct_models_vector.vector_engine_model import VectorEngineModel
from funct_models_vector.vector_params import VectorParams

model = VectorEngineModel(VectorParams())

a_bits = [0x4040] * 16    # 16 lanes of BF16 3.0
b_bits = [0x3F80] * 16    # 16 lanes of BF16 1.0
out = model.execute("add", a_vec=a_bits, b_vec=b_bits)
# out is a list[int] of 16-bit BF16 words
```

Op names match the VPU ISA: `add`, `sub`, `mul`, `rcp`, `sqrt`, `sin`,
`cos`, `log`, `exp`, `exp2`, `tanh`, `square`, `cube`, `rmax`, `rmin`,
`rsum`, `mov`, `relu`, `cmax`, `cmin`, `csum`, `pairmax`, `pairmin`,
`vliOne`, `vliCol`, `vliRow`, `vliAll`, `fp8pack`, `fp8unpack`.

A few wrinkles worth knowing:

- **VLI ops** take `imm` plus an optional `row_idx: int = 0`. `vliRow`
  and `vliOne` only fire on row 0 and return all-zero lanes for
  `row_idx > 0`, mirroring the `rowIdx == 0` guard in
  `VectorLoadImm.scala`. `vliAll` and `vliCol` ignore `row_idx`.
- **`fp8pack`** takes two 16-lane BF16 rows (`a_vec` = low, `b_vec` =
  high) plus an E8M0 `scale_e8m0` byte and returns 16 UInt16 packed
  slots. **`fp8unpack`** takes 16 packed slots in `a_vec` and returns
  32 BF16 bits (low row followed by high row). Both are phased
  2-to-1 / 1-to-2 ops and are deliberately kept out of the
  single-pulse `vpu_vectors.txt` flow.
- The `fp8` enum sitting between `csum` and `fp8pack` in `VPUOp` is
  unwired in the RTL.
  `execute("fp8", ...)` raises `NotImplementedError`.
- `execute()` is the only functional entry point. `cycle_step()` is a
  stub and raises if called.

## Tests

```bash
python3 -m pytest funct_models_vector/tests/ -q
```

Test tiers:

- `test_<lane_box>.py`: per-op lane-box unit tests against a Python
  tree reference.
- `test_vpu_vectors_file.py`: model vs. the committed `vpu_vectors.txt`
  golden.
- `test_vector_engine_model.py`: dispatcher-level tests (currently the
  VLI `row_idx` matrix).
- `test_exp_tanh_placeholders.py`: sanity checks on the Group D
  placeholder lane boxes: they must construct, raise on `compute_now`,
  and stay safe during a drain cycle.
- `test_vector_rtl_forward.py` and `test_vector_rtl_forward_fuzz.py`:
  torch adapter smoke and fuzz tests.
- `test_rtl_actual_outputs.py`: model vs. a snapshot of Scala RTL
  output (`src/test/resources/rtl_actual_outputs.txt`). `tanh` is the
  only remaining expected divergence (xfail); everything else is
  strict pass. Note that the snapshot parser treats the `inA`/`inB`
  input columns as optional, since the Scala driver only emits them on
  failing rows, so a fully-green snapshot has those columns absent and
  the per-lane input cross-check becomes a no-op.

  To regenerate the snapshot, first rebuild the vector golden from the
  current funct model, then re-run the Scala driver (both from
  `sp26-atlas-acc/`):

  ```bash
  python3 scripts/gen_vpu_vectors.py \
      --out src/test/resources/vpu_vectors.txt --num 58 --seed 0
  mill atlas.test.testOnly atlas.vector.VectorEngineTopAllOpsTest
  ```
