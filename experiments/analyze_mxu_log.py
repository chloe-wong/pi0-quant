#!/usr/bin/env python3
"""
analyze_mxu_log.py — Numerical analysis of decoded MXU linear-layer log files.

Parses *_call*.log files produced by decode_mxu_npz.py and reports, per log file,
any sign flips, high relative errors, NaN/Inf, or absolute outliers between
unpatched_y (reference) and patched_y (quantized output), together with the worst
offending elements (showing unpatched_x input values, ref, patched, diff, and
relative error at each position).

Usage
-----
    python experiments/analyze_mxu_log.py <path>  [options]

    <path>  a single *_call*.log file  OR  a directory containing *_call*.log files

Options
-------
    --rel-err-threshold  FLOAT   (default 0.5)   |diff|/|ref| > this → high_rel_err
    --outlier-k          FLOAT   (default 10)    |diff| > k×rms(diff) → abs_outlier
    --sign-flip-min      FLOAT   (default 1e-4)  minimum |ref| to count a sign flip
    --top-n              INT     (default 5)     worst elements shown per check
    --only-analysis              skip files with no findings
    --csv                FILE    append one CSV row per log file to FILE
    --log-dir            DIR     save full text output to DIR/<stem>_analysis.log
    --max-files          INT     stop after N log files (default: all)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Log file regexes
# ---------------------------------------------------------------------------

# Tensor section header — field name + shape; trailing scale_exp info is ignored
FIELD_RE  = re.compile(r'^=== (\S+)\s+shape=\(([^)]*)\)')
FLOAT_RE  = re.compile(r'-?(?:inf|nan|\d+\.?\d*(?:[eE][+-]?\d+)?)', re.IGNORECASE)
_DTYPE_RE = re.compile(r',?\s*dtype=\S+')


def _parse_shape(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(',') if x.strip())


def _parse_tensor_lines(lines: list[str]) -> np.ndarray:
    """Extract all float values from a list of tensor-dump lines."""
    nums: list[str] = []
    for line in lines:
        clean = _DTYPE_RE.sub('', line)
        nums.extend(FLOAT_RE.findall(clean))
    return np.array(nums, dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-file log parser
# ---------------------------------------------------------------------------

def parse_log(path: Path) -> dict[str, dict]:
    """
    Parse one *_call*.log file.
    Returns {field_name: {'shape': tuple, 'lines': [str]}}.
    Skips header lines (File:, Calls:, blank lines before first === block).
    """
    sections: dict = {}
    current_field: Optional[str] = None
    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.rstrip('\n')
            m = FIELD_RE.match(line)
            if m:
                current_field = m.group(1)
                shape = _parse_shape(m.group(2))
                sections[current_field] = {'shape': shape, 'lines': []}
                continue
            if current_field is not None and line.strip():
                sections[current_field]['lines'].append(line)
    return sections


# ---------------------------------------------------------------------------
# Analysis (copied verbatim from analyze_vec_log.py)
# ---------------------------------------------------------------------------

@dataclass
class Config:
    rel_err_threshold: float = 0.5
    outlier_k:         float = 10.0
    sign_flip_min:     float = 1e-4
    top_n:             int   = 5


def _top_n_rows(
    mask:      np.ndarray,
    sort_vals: np.ndarray,
    inputs:    list[np.ndarray],
    ref:       np.ndarray,
    fm:        np.ndarray,
    diff:      np.ndarray,
    rel_err:   np.ndarray,
    shape:     tuple,
    top_n:     int,
) -> list[dict]:
    flat_mask    = mask.ravel()
    flat_indices = np.where(flat_mask)[0]
    if len(flat_indices) == 0:
        return []
    flat_sort = sort_vals.ravel()[flat_indices]
    order     = np.argsort(flat_sort)[::-1][:top_n]
    rows = []
    for flat_idx in flat_indices[order]:
        nd_idx = np.unravel_index(int(flat_idx), shape)
        rows.append({
            'idx':    nd_idx,
            'inputs': [float(inp.ravel()[min(int(flat_idx), inp.size - 1)])
                       for inp in inputs],
            'ref':    float(ref.ravel()[flat_idx]),
            'fm':     float(fm.ravel()[flat_idx]),
            'diff':   float(diff.ravel()[flat_idx]),
            'rel':    float(rel_err.ravel()[flat_idx]),
        })
    return rows


def check_analysis(
    inputs:      list[np.ndarray],
    ref:         np.ndarray,
    fm:          np.ndarray,
    stored_rmse: Optional[float],
    shape:       tuple,
    cfg:         Config,
) -> dict:
    diff     = fm - ref
    abs_diff = np.abs(diff)
    rel_err  = abs_diff / (np.abs(ref) + 1e-6)
    rms_diff = float(np.sqrt(np.mean(diff ** 2)))
    n        = ref.size

    nan_inf_mask   = np.isnan(fm) | np.isinf(fm)
    sign_flip_mask = (np.sign(fm) != np.sign(ref)) & (np.abs(ref) > cfg.sign_flip_min)
    high_rel_mask  = rel_err > cfg.rel_err_threshold
    if rms_diff > 1e-12:
        abs_out_mask = abs_diff > cfg.outlier_k * rms_diff
    else:
        abs_out_mask = np.zeros(shape, dtype=bool)

    kw = dict(inputs=inputs, ref=ref, fm=fm, diff=diff,
              rel_err=rel_err, shape=shape, top_n=cfg.top_n)

    sr       = stored_rmse if stored_rmse is not None else 0.0
    mismatch = abs(rms_diff - sr) / max(sr, 1e-12) > 0.01

    return {
        'n':                n,
        'shape':            shape,
        'nan_inf':          int(nan_inf_mask.sum()),
        'nan_inf_top':      _top_n_rows(nan_inf_mask,  abs_diff, **kw),
        'sign_flip':        int(sign_flip_mask.sum()),
        'sign_flip_top':    _top_n_rows(sign_flip_mask, abs_diff, **kw),
        'high_rel_err':     int(high_rel_mask.sum()),
        'high_rel_top':     _top_n_rows(high_rel_mask, rel_err,  **kw),
        'abs_outlier':      int(abs_out_mask.sum()),
        'abs_outlier_top':  _top_n_rows(abs_out_mask,  abs_diff, **kw),
        'stored_rmse':      sr,
        'recomputed_rmse':  rms_diff,
        'rmse_mismatch':    mismatch,
    }


def has_any_findings(result: dict) -> bool:
    # No rmse_mismatch check — MXU logs have no stored RMSE
    return (result['nan_inf']      > 0 or
            result['sign_flip']    > 0 or
            result['high_rel_err'] > 0 or
            result['abs_outlier']  > 0)


def _fmt_row(row: dict) -> str:
    idx_str = '[' + ', '.join(f'{i:3d}' for i in row['idx']) + ']'
    rel_val = row['rel']
    rel_str = f'{rel_val:10.4g}' if np.isfinite(rel_val) else '       inf'
    return (
        f'      {idx_str}'
        f'  ref={row["ref"]:11.4g}'
        f'  fm={row["fm"]:11.4g}'
        f'  diff={row["diff"]:11.4g}'
        f'  rel={rel_str}'
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def report_file(path: Path, result: dict, emit=print) -> None:
    shape_str = '(' + ', '.join(str(s) for s in result['shape']) + ')'
    n = result['n']
    emit(f'\n{path.name}  shape={shape_str}  n={n}')

    analysis_checks = [
        ('nan_inf',     'nan_inf     ', 'nan_inf_top'),
        ('sign_flip',   'sign_flip   ', 'sign_flip_top'),
        ('high_rel_err','high_rel_err', 'high_rel_top'),
        ('abs_outlier', 'abs_outlier ', 'abs_outlier_top'),
    ]
    for key, label, top_key in analysis_checks:
        count = result[key]
        frac  = 100.0 * count / n if n > 0 else 0.0
        emit(f'  {label}: {count:7d} / {n:7d}  ({frac:6.2f}%)')
        for row in result[top_key]:
            emit(_fmt_row(row))

    emit(f'  rmse        :  {result["recomputed_rmse"]:.8f}')


# ---------------------------------------------------------------------------
# Component / layer helpers
# ---------------------------------------------------------------------------

_COMPONENT_RULES: list[tuple[str, str]] = [
    ("action_in_proj",     "action_head"),
    ("action_out_proj",    "action_head"),
    ("action_time_mlp",    "action_head"),
    ("state_proj",         "action_head"),
    ("time_mlp",           "action_head"),
    ("gemma_expert",       "action_expert"),
    ("vision_tower",       "vision"),
    ("language_model",     "language"),
    ("paligemma",          "language"),
]


def _infer_component(stem: str) -> str:
    for substr, comp in _COMPONENT_RULES:
        if substr in stem:
            return comp
    return "unknown"


def _stem_to_layer(stem: str) -> str:
    return re.sub(r'_call\d+$', '', stem)


# ---------------------------------------------------------------------------
# Per-file driver
# ---------------------------------------------------------------------------

def analyze_file(
    path:          Path,
    cfg:           Config,
    only_analysis: bool,
    csv_writer,
    emit=print,
) -> None:
    sections = parse_log(path)

    ref_sec = sections.get('unpatched_y')
    fm_sec  = sections.get('patched_y')
    if ref_sec is None or fm_sec is None:
        emit(f'\n{path.name}  [skipped — missing unpatched_y or patched_y]')
        return

    shape = ref_sec['shape']
    ref   = _parse_tensor_lines(ref_sec['lines'])
    fm    = _parse_tensor_lines(fm_sec['lines'])

    expected = 1
    for s in shape:
        expected *= s

    if ref.size != expected or fm.size != expected:
        emit(f'\n{path.name}  [skipped — parsed {ref.size}/{fm.size} values, '
             f'expected {expected} for shape {shape}]')
        return

    ref = ref.reshape(shape)
    fm  = fm.reshape(shape)

    inputs: list[np.ndarray] = []
    x_sec = sections.get('unpatched_x')
    if x_sec is not None:
        x_arr = _parse_tensor_lines(x_sec['lines'])
        try:
            x_arr = x_arr.reshape(x_sec['shape'])
        except ValueError:
            pass
        inputs = [x_arr]

    result = check_analysis(inputs, ref, fm, stored_rmse=None, shape=shape, cfg=cfg)

    if only_analysis and not has_any_findings(result):
        return

    report_file(path, result, emit=emit)

    if csv_writer is not None:
        n = result['n']
        csv_writer.writerow({
            'file':              path.name,
            'layer':             _stem_to_layer(path.stem),
            'component':         _infer_component(path.stem),
            'n_elem':            n,
            'nan_inf':           result['nan_inf'],
            'sign_flip':         result['sign_flip'],
            'sign_flip_frac':    result['sign_flip']    / max(n, 1),
            'high_rel_err':      result['high_rel_err'],
            'high_rel_err_frac': result['high_rel_err'] / max(n, 1),
            'abs_outlier':       result['abs_outlier'],
            'abs_outlier_frac':  result['abs_outlier']  / max(n, 1),
            'rmse':              result['recomputed_rmse'],
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    'file', 'layer', 'component', 'n_elem',
    'nan_inf',
    'sign_flip',    'sign_flip_frac',
    'high_rel_err', 'high_rel_err_frac',
    'abs_outlier',  'abs_outlier_frac',
    'rmse',
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Numerical analysis of decoded MXU linear-layer log files.')
    ap.add_argument('path', help='*_call*.log file or directory of *_call*.log files')
    ap.add_argument('--rel-err-threshold', type=float, default=0.5,
                    metavar='F',
                    help='|diff|/|ref| threshold for high_rel_err (default: 0.5)')
    ap.add_argument('--outlier-k', type=float, default=10.0,
                    metavar='K',
                    help='k for |diff| > k×rms(diff) abs_outlier test (default: 10)')
    ap.add_argument('--sign-flip-min', type=float, default=1e-4,
                    metavar='M',
                    help='minimum |ref| to count a sign flip (default: 1e-4)')
    ap.add_argument('--top-n', type=int, default=5,
                    metavar='N',
                    help='worst elements to show per check (default: 5)')
    ap.add_argument('--only-analysis', action='store_true',
                    help='skip files with no findings')
    ap.add_argument('--csv', metavar='FILE',
                    help='append CSV rows to FILE (one row per log file)')
    ap.add_argument('--log-dir', metavar='DIR',
                    help='save full text output to DIR/<stem>_analysis.log per file')
    ap.add_argument('--max-files', type=int, default=None,
                    metavar='N',
                    help='stop after N log files (default: all)')
    args = ap.parse_args()

    cfg = Config(
        rel_err_threshold=args.rel_err_threshold,
        outlier_k=args.outlier_k,
        sign_flip_min=args.sign_flip_min,
        top_n=args.top_n,
    )

    p = Path(args.path)
    if p.is_dir():
        log_files = sorted(p.glob('*_call*.log'))
        if not log_files:
            print(f'No *_call*.log files found in {p}', file=sys.stderr)
            sys.exit(1)
    elif p.is_file():
        log_files = [p]
    else:
        print(f'Path not found: {p}', file=sys.stderr)
        sys.exit(1)

    if args.max_files is not None:
        log_files = log_files[:args.max_files]

    csv_fh = csv_writer = None
    if args.csv:
        csv_path  = Path(args.csv)
        write_hdr = not csv_path.exists()
        csv_fh    = open(csv_path, 'a', newline='')
        csv_writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
        if write_hdr:
            csv_writer.writeheader()

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    try:
        for log_file in log_files:
            out_fh = None
            if log_dir is not None:
                out_path = log_dir / f'{log_file.stem}_analysis.log'
                out_fh   = open(out_path, 'w')
                print(f'Logging {log_file.name} → {out_path}')

            def emit(*a, _fh=out_fh, **kw):
                print(*a, **kw)
                if _fh is not None:
                    print(*a, **kw, file=_fh)

            try:
                analyze_file(log_file, cfg,
                             only_analysis=args.only_analysis,
                             csv_writer=csv_writer,
                             emit=emit)
            finally:
                if out_fh is not None:
                    out_fh.close()
    finally:
        if csv_fh is not None:
            csv_fh.close()


if __name__ == '__main__':
    main()
