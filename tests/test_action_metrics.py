"""
test_action_metrics.py
----------------------
Correctness tests for action_metrics.compute_action_metrics.

Uses deterministic, hand-crafted inputs so expected values can be verified
analytically.  Run with:
    python -m pytest tests/test_action_metrics.py -v
"""

import math
import torch
import pytest

from pi0_inout.action_metrics import (
    ActionMetrics,
    ActionThresholds,
    CheckResult,
    compute_action_metrics,
    BaselineVariance,
    compute_baseline_variance_from_actions,
)


# ---------------------------------------------------------------------------
# 1. Verify overall RMSE matches hand-computed value
# ---------------------------------------------------------------------------

def test_overall_rmse_known_error():
    """Constant error of 0.1 everywhere => RMSE should be exactly 0.1."""
    ref = [torch.ones(4, 8)]          # 1 obs, horizon=4, dim=8
    quant = [torch.ones(4, 8) + 0.1]  # every element off by +0.1

    m = compute_action_metrics(ref, quant)

    assert abs(m.overall_rmse - 0.1) < 1e-6, f"expected 0.1, got {m.overall_rmse}"
    assert abs(m.overall_mae - 0.1) < 1e-6, f"expected MAE 0.1, got {m.overall_mae}"


def test_overall_rmse_matches_old_implementation():
    """
    Verify overall_rmse matches the old _compute_action_rmse logic:
    flatten everything, sum of squared errors, divide by numel, sqrt.
    """
    torch.manual_seed(42)
    ref = [torch.randn(10, 32) for _ in range(5)]
    quant = [r + 0.01 * torch.randn_like(r) for r in ref]

    m = compute_action_metrics(ref, quant)

    # Recompute with the old approach
    total_se, total_n = 0.0, 0
    for r, q in zip(ref, quant):
        diff = r.float() - q.float()
        total_se += diff.pow(2).sum().item()
        total_n += diff.numel()
    old_rmse = math.sqrt(total_se / total_n)

    assert abs(m.overall_rmse - old_rmse) < 1e-6, (
        f"overall_rmse={m.overall_rmse} != old_rmse={old_rmse}"
    )


# ---------------------------------------------------------------------------
# 2. Verify per-step RMSE
# ---------------------------------------------------------------------------

def test_per_step_rmse_concentrated_error():
    """
    Error only at step 0: per_step_rmse[0] should be large,
    all other steps should be ~0.
    """
    ref = torch.zeros(4, 8)
    quant = torch.zeros(4, 8)
    quant[0, :] = 0.5  # error of 0.5 at step 0 only

    m = compute_action_metrics([ref], [quant])

    assert abs(m.per_step_rmse[0] - 0.5) < 1e-6
    for t in range(1, 4):
        assert abs(m.per_step_rmse[t]) < 1e-6, f"step {t} should be 0"


# ---------------------------------------------------------------------------
# 3. Verify per-dimension RMSE
# ---------------------------------------------------------------------------

def test_per_dim_rmse_single_dim_error():
    """
    Error only in dim 3: per_dim_rmse[3] should be large, others ~0.
    """
    ref = torch.zeros(4, 8)
    quant = torch.zeros(4, 8)
    quant[:, 3] = 0.2  # error of 0.2 in dim 3 at all steps

    m = compute_action_metrics([ref], [quant])

    assert abs(m.per_dim_rmse[3] - 0.2) < 1e-6
    for d in range(8):
        if d != 3:
            assert abs(m.per_dim_rmse[d]) < 1e-6, f"dim {d} should be 0"


# ---------------------------------------------------------------------------
# 4. Verify per-sample RMSE
# ---------------------------------------------------------------------------

def test_per_sample_rmse():
    """Two samples: one clean, one with error 0.3 everywhere."""
    ref_clean = torch.zeros(4, 8)
    ref_noisy = torch.zeros(4, 8)
    quant_clean = torch.zeros(4, 8)
    quant_noisy = torch.full((4, 8), 0.3)

    m = compute_action_metrics(
        [ref_clean, ref_noisy],
        [quant_clean, quant_noisy],
    )

    assert abs(m.per_sample_rmse[0]) < 1e-6, "clean sample should have ~0 RMSE"
    assert abs(m.per_sample_rmse[1] - 0.3) < 1e-6, f"noisy sample should have 0.3 RMSE"


# ---------------------------------------------------------------------------
# 5. Verify percentiles
# ---------------------------------------------------------------------------

def test_percentiles_uniform_error():
    """With constant |error|, all percentiles should equal that value."""
    ref = [torch.zeros(10, 32)]
    quant = [torch.full((10, 32), 0.05)]

    m = compute_action_metrics(ref, quant)

    for k in ("p50", "p90", "p95", "p99"):
        assert abs(m.percentiles[k] - 0.05) < 1e-6, (
            f"{k} should be 0.05, got {m.percentiles[k]}"
        )


# ---------------------------------------------------------------------------
# 6. Verify max_abs_error location
# ---------------------------------------------------------------------------

def test_max_abs_error_location():
    """Place one large error at a known location and verify it's found."""
    ref = torch.zeros(4, 8)
    quant = torch.zeros(4, 8)
    quant[2, 5] = 1.0  # largest error at (step=2, dim=5)

    m = compute_action_metrics([ref], [quant])

    assert abs(m.max_abs_error - 1.0) < 1e-6
    assert m.max_abs_error_location == (0, 2, 5), (
        f"expected (0, 2, 5), got {m.max_abs_error_location}"
    )


# ---------------------------------------------------------------------------
# 7. Verify relative RMSE
# ---------------------------------------------------------------------------

def test_relative_rmse():
    """
    ref = 10.0 everywhere, error = 0.1 everywhere.
    rel_rmse = rmse / rms(ref) = 0.1 / 10.0 = 0.01
    """
    ref = [torch.full((4, 8), 10.0)]
    quant = [torch.full((4, 8), 10.1)]

    m = compute_action_metrics(ref, quant)

    assert abs(m.overall_rel_rmse - 0.01) < 1e-5, (
        f"expected 0.01, got {m.overall_rel_rmse}"
    )


# ---------------------------------------------------------------------------
# 8. Verify check() pass/fail
# ---------------------------------------------------------------------------

def test_check_pass():
    ref = [torch.zeros(4, 8)]
    quant = [torch.full((4, 8), 0.01)]

    m = compute_action_metrics(ref, quant)
    thresholds = ActionThresholds(max_rmse=0.1, max_p99_abs_error=0.1)
    result = m.check(thresholds)

    assert result.passed, f"expected PASS, got: {result}"
    assert len(result.failures) == 0


def test_check_fail_rmse():
    ref = [torch.zeros(4, 8)]
    quant = [torch.full((4, 8), 0.5)]

    m = compute_action_metrics(ref, quant)
    thresholds = ActionThresholds(max_rmse=0.1)
    result = m.check(thresholds)

    assert not result.passed
    assert any("overall_rmse" in f for f in result.failures)


def test_check_fail_per_dim():
    ref = torch.zeros(4, 8)
    quant = torch.zeros(4, 8)
    quant[:, 0] = 0.5  # dim 0 has large error

    m = compute_action_metrics([ref], [quant])
    thresholds = ActionThresholds(max_per_dim_rmse=0.1)
    result = m.check(thresholds)

    assert not result.passed
    assert any("per_dim_rmse" in f for f in result.failures)


def test_check_none_thresholds_always_pass():
    """All-None thresholds should always pass."""
    ref = [torch.randn(4, 8)]
    quant = [torch.randn(4, 8)]  # totally different

    m = compute_action_metrics(ref, quant)
    result = m.check(ActionThresholds())

    assert result.passed


# ---------------------------------------------------------------------------
# 9. Verify print_report doesn't crash
# ---------------------------------------------------------------------------

def test_print_report_runs(capsys):
    torch.manual_seed(0)
    ref = [torch.randn(10, 32) for _ in range(3)]
    quant = [r + 0.01 * torch.randn_like(r) for r in ref]

    m = compute_action_metrics(ref, quant)
    m.print_report()

    captured = capsys.readouterr()
    assert "ACTION METRICS REPORT" in captured.out
    assert "Per-Timestep RMSE" in captured.out
    assert "Per-Dimension RMSE" in captured.out
    assert "Error Percentiles" in captured.out


# ---------------------------------------------------------------------------
# 10. Verify to_dict / serialization
# ---------------------------------------------------------------------------

def test_to_dict_has_all_keys():
    ref = [torch.zeros(4, 8)]
    quant = [torch.full((4, 8), 0.01)]

    m = compute_action_metrics(ref, quant)
    d = m.to_dict()

    expected_keys = {
        "overall_rmse", "overall_rel_rmse", "overall_mae",
        "per_step_rmse", "per_step_rel_rmse",
        "per_dim_rmse", "per_dim_rel_rmse",
        "per_sample_rmse", "percentiles",
        "max_abs_error", "max_abs_error_location",
        "n_observations", "action_horizon", "action_dim", "dim_labels",
    }
    assert set(d.keys()) == expected_keys


def test_to_dict_is_json_serializable():
    import json

    ref = [torch.zeros(4, 8)]
    quant = [torch.full((4, 8), 0.01)]

    m = compute_action_metrics(ref, quant)
    serialized = json.dumps(m.to_dict())
    assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# 11. Batch dim squeeze
# ---------------------------------------------------------------------------

def test_batch_dim_squeeze():
    """Tensors with leading batch dim [1, H, D] should work."""
    ref = [torch.zeros(1, 4, 8)]
    quant = [torch.full((1, 4, 8), 0.1)]

    m = compute_action_metrics(ref, quant)
    assert abs(m.overall_rmse - 0.1) < 1e-6
    assert m.action_horizon == 4
    assert m.action_dim == 8


# ===========================================================================
# Baseline variance tests
# ===========================================================================

# ---------------------------------------------------------------------------
# 12. Basic shape and metadata
# ---------------------------------------------------------------------------

def test_baseline_variance_shape():
    """Verify output shapes and metadata match input dimensions."""
    # 3 obs, 5 seeds each, action_horizon=4, action_dim=8
    actions_per_obs = [
        [torch.randn(4, 8) for _ in range(5)]
        for _ in range(3)
    ]
    bv = compute_baseline_variance_from_actions(actions_per_obs)

    assert bv.n_observations == 3
    assert bv.n_seeds == 5
    assert bv.action_horizon == 4
    assert bv.action_dim == 8
    assert len(bv.per_step_std) == 4
    assert len(bv.per_dim_std) == 8
    assert len(bv.per_sample_std) == 3


# ---------------------------------------------------------------------------
# 13. Zero variance when seeds produce identical actions
# ---------------------------------------------------------------------------

def test_baseline_variance_zero_when_identical():
    """If every seed produces the same tensor, variance should be 0."""
    base = torch.ones(4, 8)
    actions_per_obs = [
        [base.clone() for _ in range(5)]
        for _ in range(2)
    ]
    bv = compute_baseline_variance_from_actions(actions_per_obs)

    assert abs(bv.overall_std) < 1e-6
    for s in bv.per_step_std:
        assert abs(s) < 1e-6
    for d in bv.per_dim_std:
        assert abs(d) < 1e-6


# ---------------------------------------------------------------------------
# 14. Known std from controlled noise
# ---------------------------------------------------------------------------

def test_baseline_variance_known_std():
    """
    Seeds produce base + offset where offset is {-1, 0, +1}.
    std of [-1, 0, 1] = 1.0 (sample std).
    """
    base = torch.zeros(1, 1)  # simplest shape: horizon=1, dim=1
    actions_per_obs = [
        [base - 1.0, base.clone(), base + 1.0]  # 3 seeds
    ]
    bv = compute_baseline_variance_from_actions(actions_per_obs)

    assert abs(bv.overall_std - 1.0) < 1e-5, f"expected 1.0, got {bv.overall_std}"


# ---------------------------------------------------------------------------
# 15. Variance concentrated in one dim
# ---------------------------------------------------------------------------

def test_baseline_variance_single_dim_noisy():
    """Only dim 0 varies across seeds; others are constant."""
    n_seeds = 10
    torch.manual_seed(99)
    actions_per_obs = []
    for _ in range(2):  # 2 obs
        seed_actions = []
        for _ in range(n_seeds):
            a = torch.zeros(4, 8)
            a[:, 0] = torch.randn(4)  # only dim 0 is noisy
            seed_actions.append(a)
        actions_per_obs.append(seed_actions)

    bv = compute_baseline_variance_from_actions(actions_per_obs)

    assert bv.per_dim_std[0] > 0.1, "dim 0 should have non-trivial std"
    for d in range(1, 8):
        assert abs(bv.per_dim_std[d]) < 1e-6, f"dim {d} should have ~0 std"


# ---------------------------------------------------------------------------
# 16. to_thresholds scaling
# ---------------------------------------------------------------------------

def test_to_thresholds_scaling():
    """to_thresholds(k) should scale all limits by k."""
    torch.manual_seed(0)
    actions_per_obs = [
        [torch.randn(4, 8) for _ in range(5)]
        for _ in range(3)
    ]
    bv = compute_baseline_variance_from_actions(actions_per_obs)

    t1 = bv.to_thresholds(k=1.0)
    t2 = bv.to_thresholds(k=2.0)

    assert t1.max_rmse is not None and t2.max_rmse is not None
    assert t1.max_per_step_rmse is not None and t2.max_per_step_rmse is not None
    assert t1.max_per_dim_rmse is not None and t2.max_per_dim_rmse is not None
    assert t1.max_p99_abs_error is not None and t2.max_p99_abs_error is not None

    assert abs(t2.max_rmse - 2.0 * t1.max_rmse) < 1e-10
    assert abs(t2.max_per_step_rmse - 2.0 * t1.max_per_step_rmse) < 1e-10
    assert abs(t2.max_per_dim_rmse - 2.0 * t1.max_per_dim_rmse) < 1e-10
    assert abs(t2.max_p99_abs_error - 2.0 * t1.max_p99_abs_error) < 1e-10


def test_to_thresholds_no_rel_rmse():
    """to_thresholds should leave max_rel_rmse as None."""
    actions_per_obs = [[torch.randn(4, 8) for _ in range(3)]]
    bv = compute_baseline_variance_from_actions(actions_per_obs)
    t = bv.to_thresholds(k=1.0)

    assert t.max_rel_rmse is None


# ---------------------------------------------------------------------------
# 17. Baseline + check integration
# ---------------------------------------------------------------------------

def test_baseline_check_pass_small_error():
    """Quant error smaller than baseline noise should PASS at k=1."""
    torch.manual_seed(42)
    # Baseline: actions vary by ~1.0 std across seeds
    actions_per_obs = [
        [torch.randn(4, 8) for _ in range(10)]
        for _ in range(3)
    ]
    bv = compute_baseline_variance_from_actions(actions_per_obs)
    thresholds = bv.to_thresholds(k=1.0)

    # Quant error: tiny (0.001)
    ref = [torch.randn(4, 8) for _ in range(3)]
    quant = [r + 0.001 * torch.randn_like(r) for r in ref]
    metrics = compute_action_metrics(ref, quant)

    result = metrics.check(thresholds)
    assert result.passed, f"expected PASS, got: {result}"


def test_baseline_check_fail_large_error():
    """Quant error much larger than baseline noise should FAIL."""
    # Baseline: actions are constant (zero variance)
    base = torch.ones(4, 8)
    actions_per_obs = [
        [base.clone() for _ in range(5)]
        for _ in range(2)
    ]
    bv = compute_baseline_variance_from_actions(actions_per_obs)
    thresholds = bv.to_thresholds(k=1.0)

    # With zero baseline std, any error should fail
    ref = [torch.zeros(4, 8)]
    quant = [torch.full((4, 8), 0.1)]
    metrics = compute_action_metrics(ref, quant)

    result = metrics.check(thresholds)
    assert not result.passed


# ---------------------------------------------------------------------------
# 18. print_report doesn't crash
# ---------------------------------------------------------------------------

def test_baseline_print_report_runs(capsys):
    torch.manual_seed(0)
    actions_per_obs = [
        [torch.randn(10, 32) for _ in range(5)]
        for _ in range(3)
    ]
    bv = compute_baseline_variance_from_actions(actions_per_obs)
    bv.print_report()

    captured = capsys.readouterr()
    assert "BASELINE VARIANCE REPORT" in captured.out
    assert "Suggested thresholds" in captured.out


# ---------------------------------------------------------------------------
# 19. to_dict serialization
# ---------------------------------------------------------------------------

def test_baseline_to_dict():
    import json

    actions_per_obs = [[torch.randn(4, 8) for _ in range(3)]]
    bv = compute_baseline_variance_from_actions(actions_per_obs)
    d = bv.to_dict()

    expected_keys = {
        "overall_std", "per_step_std", "per_dim_std", "per_sample_std",
        "percentiles", "n_observations", "n_seeds",
        "action_horizon", "action_dim", "dim_labels",
    }
    assert set(d.keys()) == expected_keys
    assert isinstance(json.dumps(d), str)


# ---------------------------------------------------------------------------
# 20. Batch dim squeeze for baseline
# ---------------------------------------------------------------------------

def test_baseline_batch_dim_squeeze():
    """Tensors with leading batch dim [1, H, D] should work."""
    actions_per_obs = [
        [torch.randn(1, 4, 8) for _ in range(5)]
    ]
    bv = compute_baseline_variance_from_actions(actions_per_obs)
    assert bv.action_horizon == 4
    assert bv.action_dim == 8
