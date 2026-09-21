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

FLAT = [{"rmse": 0.100, "bias": 0.010}, {"rmse": 0.1001, "bias": 0.0100}]
MOVING = [{"rmse": 0.100, "bias": 0.010}, {"rmse": 0.080, "bias": 0.005}]


def state(history=(), cal=()):
    return CycleState(history=list(history), cal_metrics=list(cal))


def cfg(**kw):
    import dataclasses
    return dataclasses.replace(CycleConfig(), **kw)


# ==== the rule itself


def test_a_plateau_converges():
    assert converged(FLAT, 0.01) is True


def test_continued_improvement_does_not():
    assert converged(MOVING, 0.01) is False


def test_one_entry_is_never_convergence():
    assert converged(FLAT[:1], 0.01) is False


def test_a_missing_key_is_skipped_not_assumed():
    """Calibration metrics have no `bias`, so the rule must not treat its
    absence as 'converged'."""
    h = [{"rmse": 0.100}, {"rmse": 0.080}]
    assert converged(h, 0.01, keys=("rmse", "bias")) is False


def test_no_usable_keys_means_not_converged():
    assert converged([{"kge": 1.0}, {"kge": 1.0}], 0.01, keys=("rmse",)) is False


# ==== choosing the signal


def test_surrogate_is_the_default():
    """Free, available every cycle, and the quantity the search minimises."""
    assert CycleConfig().converge_on == "surrogate"


def test_rel_tol_zero_disables_stopping():
    """Any change clears a zero threshold, so the loop runs to n_cycles."""
    assert converged(FLAT, 0.0) is False


def test_wflow_ignores_the_surrogate():
    st = state(history=FLAT, cal=MOVING)
    assert has_converged(st, cfg(converge_on="wflow")) is True


def test_surrogate_ignores_wflow():
    st = state(history=MOVING, cal=FLAT)
    assert has_converged(st, cfg(converge_on="surrogate")) is True


def test_both_requires_agreement():
    st = state(history=FLAT, cal=MOVING)
    assert has_converged(st, cfg(converge_on="both")) is False
    st = state(history=FLAT, cal=FLAT)
    assert has_converged(st, cfg(converge_on="both")) is True


def test_surrogate_can_stop_the_loop_with_no_wflow_scores_at_all():
    """The point of the option: ground truth is paid for only where wanted,
    not to decide when to stop."""
    st = state(history=[], cal=FLAT)
    assert has_converged(st, cfg(converge_on="surrogate", score_every=0)) is True
    assert has_converged(st, cfg(converge_on="wflow", score_every=0)) is False


def test_an_unknown_setting_is_refused():
    with pytest.raises(ValueError, match="converge_on"):
        has_converged(state(history=FLAT), cfg(converge_on="magic"))
