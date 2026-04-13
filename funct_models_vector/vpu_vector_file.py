"""Reader/writer for the `vpu_vectors.txt` block format consumed by
`VectorEngineTestDriver.scala`, `VectorEngineTopAllOpsTest.scala`,
and `VectorEngineTestCsum.scala`.

Format per case (byte-exact reproduction of `scripts/gen_vpu_vectors.py`):

    # <id> - "<desc>"
    vpuOp <op>
    numLanes <int>
    vecA <HEX> <HEX> ...
    vecB <HEX> <HEX> ...
    [scaleExp <HEX>]                 # fp8pack/fp8unpack only
    [leftAlign <0|1>]                # fp8pack/fp8unpack only
    exp  <HEX> <HEX> ...             # two spaces after `exp` (intentional)
                                     # trailing blank line after every case

Hex tokens are uppercase, zero-padded to 4 chars (BF16 / packed FP8 slot)
or 2 chars (`scaleExp`).

The writer never invents whitespace and never drops a line, so the round
trip `parse(open(f).read())` → `write_cases(g, cases)` produces byte-equal
output for any well-formed input. This matters because Phase 1f's success
gate (`tests/test_gen_vectors_cli.py`) is byte-identity against the
committed `src/test/resources/vpu_vectors.txt`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, TextIO


@dataclass
class VPUTestCase:
    case_id: int
    desc: str
    vpu_op: str
    num_lanes: int
    vec_a: list[str]                                 # hex strings, 4 chars each
    vec_b: list[str] = field(default_factory=list)
    scale_exp: list[str] = field(default_factory=list)   # 2-char hex, fp8 only
    left_align: list[str] = field(default_factory=list)  # ["0"] or ["1"], fp8 only
    exp: list[str] = field(default_factory=list)


# ---------------------------------------------------------------
#  Writer
# ---------------------------------------------------------------

def write_case(f: TextIO, case: VPUTestCase) -> None:
    """Write one case in the canonical block format. Bytes match the
    historical `scripts/gen_vpu_vectors.py:write_case` exactly, including
    the double space after `exp` and the trailing blank line."""
    f.write(f"# {case.case_id} - \"{case.desc}\"\n")
    f.write(f"vpuOp {case.vpu_op}\n")
    f.write(f"numLanes {case.num_lanes}\n")
    f.write(f"vecA {' '.join(case.vec_a)}\n")
    f.write(f"vecB {' '.join(case.vec_b)}\n")
    if case.scale_exp:
        f.write(f"scaleExp {' '.join(case.scale_exp)}\n")
    if case.left_align:
        f.write(f"leftAlign {' '.join(case.left_align)}\n")
    f.write(f"exp  {' '.join(case.exp)}\n\n")


def write_cases(f: TextIO, cases: Iterable[VPUTestCase]) -> None:
    for c in cases:
        write_case(f, c)


# ---------------------------------------------------------------
#  Reader
# ---------------------------------------------------------------

def parse_cases(f: TextIO) -> Iterator[VPUTestCase]:
    """Yield one VPUTestCase per block. Tolerant of leading whitespace
    inside a block but strict about block boundaries (a `# id - "desc"`
    header starts a new case, a blank line ends one)."""
    case: VPUTestCase | None = None
    for raw in f:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            if case is not None:
                yield case
                case = None
            continue
        if stripped.startswith("#"):
            if case is not None:
                yield case
            header = stripped[1:].strip()
            cid_str, _, rest = header.partition(" - ")
            case = VPUTestCase(
                case_id=int(cid_str),
                desc=rest.strip().strip('"'),
                vpu_op="",
                num_lanes=0,
                vec_a=[],
            )
            continue
        if case is None:
            continue   # stray comment / preamble before first block
        key, _, rest = stripped.partition(" ")
        toks = rest.split()
        if key == "vpuOp":
            case.vpu_op = rest.strip()
        elif key == "numLanes":
            case.num_lanes = int(rest)
        elif key == "vecA":
            case.vec_a = toks
        elif key == "vecB":
            case.vec_b = toks
        elif key == "scaleExp":
            case.scale_exp = toks
        elif key == "leftAlign":
            case.left_align = toks
        elif key == "exp":
            case.exp = toks
        # unknown keys silently ignored — forward-compatible with new fields
    if case is not None:
        yield case


def read_cases(path: str) -> list[VPUTestCase]:
    with open(path, "r") as f:
        return list(parse_cases(f))
