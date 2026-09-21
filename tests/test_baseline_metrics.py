"""Baseline reference and calibration diagnostics.

`converged()` only compares consecutive cycles, so on its own the loop can
settle on a result that is worse than not calibrating and report success. The
baseline is one wflow run for the whole campaign - the uncalibrated parameters
never change - and it makes every cycle's number readable.

The calibration metrics are surrogate(theta) vs satellite, not wflow(theta) vs
satellite, so they cannot replace the score. They are kept because they answer
a different question: when a cycle goes badly, was it the parameters or the
imitation?
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
run_dpl_cycle = pytest.importorskip("run_dpl_cycle")

rel = run_dpl_cycle.relative_to_baseline
parse = run_dpl_cycle.parse_cal_metrics


# ==== comparing against the uncalibrated run


BASE = {"rmse": 0.1007, "bias": -0.0524}


def test_an_improvement_is_reported_as_positive():
    out = rel({"rmse": 0.0900, "bias": -0.0100}, BASE)
    assert out["rmse_vs_baseline_pct"] > 0
    assert out["better_than_uncalibrated"] is True


def test_a_regression_is_caught():
    """The failure `converged()` cannot see: tidy convergence on something
    worse than the starting point."""
    out = rel({"rmse": 0.1078, "bias": 0.0029}, BASE)
    assert out["rmse_vs_baseline_pct"] < 0
    assert out["better_than_uncalibrated"] is False


def test_bias_is_compared_on_magnitude():
    """-0.0524 to +0.0029 is a large improvement, despite the sign flip."""
    out = rel({"rmse": 0.1078, "bias": 0.0029}, BASE)
    assert out["bias_vs_baseline_pct"] > 90


def test_no_baseline_means_no_comparison():
    assert rel({"rmse": 0.1}, {}) == {}


def test_a_zero_baseline_does_not_divide_by_zero():
    out = rel({"rmse": 0.1, "bias": 0.0}, {"rmse": 0.0, "bias": 0.0})
    assert "rmse_vs_baseline_pct" not in out


# ==== reading calibration's own metrics


LOG = """
ConsoleLogger: train_ssm_rmse_epoch = 0.4289756417274475
ConsoleLogger: val_ssm_rmse_epoch = 0.2488086223602295
ConsoleLogger: val_ssm_kge_epoch = -0.17318064003108424
ConsoleLogger: val_ssm_nse_epoch = -0.3366605043411255
ConsoleLogger: val_ssm_pearson_epoch = 0.016072312369942665
"""


def test_validation_metrics_are_read():
    got = parse(LOG)
    assert got["rmse"] == pytest.approx(0.2488, abs=1e-4)
    assert got["pearson"] == pytest.approx(0.01607, abs=1e-5)


def test_training_metrics_are_ignored():
    """Only the validation numbers are kept."""
    assert parse(LOG)["rmse"] != pytest.approx(0.4290, abs=1e-4)


def test_negative_values_survive():
    """KGE and NSE are routinely negative and must not be dropped."""
    got = parse(LOG)
    assert got["kge"] < 0 and got["nse"] < 0


def test_the_last_epoch_wins():
    got = parse(LOG + "ConsoleLogger: val_ssm_rmse_epoch = 0.1\n")
    assert got["rmse"] == pytest.approx(0.1)


def test_an_unparseable_log_is_not_fatal():
    """A diagnostic must never stop a cycle."""
    assert parse("nothing here") == {}
    assert parse("") == {}


def test_state_carries_both_new_fields():
    st = run_dpl_cycle.CycleState()
    assert st.baseline == {} and st.cal_metrics == []
