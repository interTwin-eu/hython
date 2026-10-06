"""What the loop stops on (`converge_on`).

Two defensible signals. The clean wflow run is ground truth but costs a
simulation per scored cycle. Calibration's own validation metric is free,
available every cycle, and is the quantity the search actually minimises - so
its plateau is the optimiser's own convergence, even though it mixes parameter
quality with surrogate quality.

Tsai's released code has no convergence test at all (`nEpoch=500`, fixed), so
neither choice is inherited from it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
run_dpl_cycle = pytest.importorskip("run_dpl_cycle")

CycleConfig = run_dpl_cycle.CycleConfig
CycleState = run_dpl_cycle.CycleState
converged = run_dpl_cycle.converged
has_converged = run_dpl_cycle.has_converged

# wflow scores (H12: per-cell KGE on the validation period) ...
FLAT = [{"kge_valid": 0.230, "rmse": 0.10}, {"kge_valid": 0.2301, "rmse": 0.12}]
MOVING = [{"kge_valid": 0.180, "rmse": 0.10}, {"kge_valid": 0.230, "rmse": 0.10}]
# ... and what calibration logged (KGECellMetric)
CAL_FLAT = [{"kgecell": 0.40, "rmse": 0.10}, {"kgecell": 0.4001, "rmse": 0.2}]
CAL_MOVING = [{"kgecell": 0.30, "rmse": 0.10}, {"kgecell": 0.40, "rmse": 0.10}]


def state(history=(), cal=()):
    return CycleState(history=list(history), cal_metrics=list(cal))


def cfg(**kw):
    """Patience 1 unless a test asks otherwise, so the two-entry fixtures test
    the plain rule; patience has its own tests below."""
    import dataclasses
    kw.setdefault("converge_patience", 1)
    return dataclasses.replace(CycleConfig(), **kw)


# ==== the rule itself


def test_a_plateau_converges():
    assert converged(FLAT, 0.01) is True


def test_continued_improvement_does_not():
    assert converged(MOVING, 0.01) is False


def test_one_entry_is_never_convergence():
    assert converged(FLAT[:1], 0.01) is False


def test_a_missing_key_is_skipped_not_assumed():
    """A key absent from the history must not count as 'converged'."""
    h = [{"rmse": 0.100}, {"rmse": 0.080}]
    assert converged(h, 0.01, keys=("rmse", "bias")) is False


def test_rmse_no_longer_decides():
    """H12: RMSE moving does not stop a KGE plateau from converging."""
    assert converged(FLAT, 0.01) is True
    assert has_converged(state(cal=CAL_FLAT), cfg(converge_on="surrogate")) is True


def test_no_usable_keys_means_not_converged():
    assert converged([{"kge": 1.0}, {"kge": 1.0}], 0.01, keys=("rmse",)) is False


# ==== choosing the signal


def test_wflow_is_the_default():
    """Changed 2026-09-25: the surrogate signal stopped production after cycle
    1 while the wflow score was still rising."""
    assert CycleConfig().converge_on == "wflow"


def test_default_threshold_is_absolute_0_02_patience_1():
    """Set 2026-09-28 after the clean rerun never stopped on rel_tol 0.01
    with patience 2."""
    assert CycleConfig().abs_tol == 0.02
    assert CycleConfig().converge_patience == 1


def test_rel_tol_zero_disables_stopping():
    """Any change clears a zero threshold, so the loop runs to n_cycles."""
    assert converged(FLAT, 0.0) is False


def test_wflow_ignores_the_surrogate():
    st = state(history=FLAT, cal=CAL_MOVING)
    assert has_converged(st, cfg(converge_on="wflow")) is True


def test_surrogate_ignores_wflow():
    st = state(history=MOVING, cal=CAL_FLAT)
    assert has_converged(st, cfg(converge_on="surrogate")) is True


def test_both_requires_agreement():
    st = state(history=FLAT, cal=CAL_MOVING)
    assert has_converged(st, cfg(converge_on="both")) is False
    st = state(history=FLAT, cal=CAL_FLAT)
    assert has_converged(st, cfg(converge_on="both")) is True


def test_surrogate_can_stop_the_loop_with_no_wflow_scores_at_all():
    """The point of the option: ground truth is paid for only where wanted,
    not to decide when to stop."""
    st = state(history=[], cal=CAL_FLAT)
    assert has_converged(st, cfg(converge_on="surrogate", score_every=0)) is True
    assert has_converged(st, cfg(converge_on="wflow", score_every=0)) is False


def test_an_unknown_setting_is_refused():
    with pytest.raises(ValueError, match="converge_on"):
        has_converged(state(history=FLAT), cfg(converge_on="magic"))


# ==== patience


FLAT3 = [{"kge_valid": 0.230}, {"kge_valid": 0.2301}, {"kge_valid": 0.2302}]
# a big step, then one flat step: the production cycle 0 -> 1 -> 2 shape
JUMP_THEN_FLAT = [{"kge_valid": 0.347}, {"kge_valid": 0.367}, {"kge_valid": 0.368}]


def test_patience_two_needs_three_entries():
    assert converged(FLAT, 0.01, patience=2) is False
    assert converged(FLAT3, 0.01, patience=2) is True


def test_patience_two_needs_both_steps_flat():
    """One flat step after a real improvement is not enough."""
    assert converged(JUMP_THEN_FLAT, 0.01, patience=1) is True
    assert converged(JUMP_THEN_FLAT, 0.01, patience=2) is False


def test_patience_looks_only_at_the_last_steps():
    """An early jump does not block a later plateau."""
    h = [{"kge_valid": 0.30}] + FLAT3
    assert converged(h, 0.01, patience=2) is True


def test_patience_missing_key_in_one_step_is_not_converged():
    h = [{"kge_valid": 0.230}, {"rmse": 0.1}, {"kge_valid": 0.230}]
    assert converged(h, 0.01, patience=2) is False


def test_patience_below_one_is_an_error():
    with pytest.raises(ValueError):
        converged(FLAT3, 0.01, patience=0)


def test_has_converged_uses_the_config_patience():
    st = state(history=JUMP_THEN_FLAT)
    assert has_converged(st, cfg(converge_on="wflow", converge_patience=1)) is True
    assert has_converged(st, cfg(converge_on="wflow", converge_patience=2)) is False


# ==== absolute threshold


def test_abs_tol_replaces_rel_tol():
    """0.004 is 2% relative but under 0.02 absolute."""
    h = [{"kge_valid": 0.200}, {"kge_valid": 0.204}]
    assert converged(h, 0.01) is False
    assert converged(h, 0.01, abs_tol=0.02) is True


def test_abs_tol_is_strict():
    h = [{"kge_valid": 0.30}, {"kge_valid": 0.33}]
    assert converged(h, 0.01, abs_tol=0.02) is False


def test_abs_tol_counts_drops_as_change():
    h = [{"kge_valid": 0.33}, {"kge_valid": 0.30}]
    assert converged(h, 0.01, abs_tol=0.02) is False


def test_abs_tol_zero_disables_stopping():
    assert converged(FLAT, 0.01, abs_tol=0.0) is False


def test_abs_tol_none_falls_back_to_rel_tol():
    assert has_converged(state(history=FLAT), cfg(abs_tol=None, rel_tol=0.01)) is True
    assert has_converged(state(history=FLAT), cfg(abs_tol=None, rel_tol=0.0)) is False


def test_abs_tol_applies_to_the_surrogate_signal():
    st = state(cal=[{"kgecell": 0.440}, {"kgecell": 0.450}])
    assert has_converged(st, cfg(converge_on="surrogate", abs_tol=0.02)) is True
    assert has_converged(st, cfg(converge_on="surrogate", abs_tol=0.005)) is False


# Clean rerun 2026-09-26/28, median per-cell KGE on the validation period.
CLEAN_RERUN = [0.3689, 0.3792, 0.3720, 0.3713, 0.3620, 0.3764, 0.3719]


def test_clean_rerun_stops_after_cycle_1_with_the_defaults():
    """Replay: with the defaults the loop would have stopped after cycle 1.
    The old setting (rel_tol 0.01, patience 2) never stopped it."""
    c = CycleConfig()
    hist = [{"kge_valid": k} for k in CLEAN_RERUN]
    first = next(n for n in range(1, len(hist) + 1)
                 if has_converged(state(history=hist[:n]), c))
    assert first - 1 == 1
    old = cfg(abs_tol=None, rel_tol=0.01, converge_patience=2)
    assert not any(has_converged(state(history=hist[:n]), old)
                   for n in range(1, len(hist) + 1))
