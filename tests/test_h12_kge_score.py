"""H12: one success metric - per-cell KGE - for the loss, the logged metric
and the wflow score."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from hython.losses import CellKGELoss, compute_kge_per_cell
from hython.metrics import KGECellMetric, MetricCollection, RMSEMetric
from hython.metrics.custom import compute_kge2, compute_kge_per_cell_np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
run_dpl_cycle = pytest.importorskip("run_dpl_cycle")


def _series(n_cells=6, n_days=200, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6 * np.pi, n_days)
    true = 0.2 + 0.05 * np.sin(t)[None] + 0.1 * rng.random((n_cells, 1))
    pred = 0.8 * true + 0.03 + 0.01 * rng.standard_normal((n_cells, n_days))
    return true, pred


# ==== the definition


def test_numpy_matches_the_pooled_kge_of_each_cell():
    true, pred = _series()
    out = compute_kge_per_cell_np(true, pred, min_obs=2)
    for c in range(true.shape[0]):
        k, r, a, b = compute_kge2(true[c], pred[c], return_all=True)
        assert out["kge"][c] == pytest.approx(k)
        assert out["r"][c] == pytest.approx(r)
        assert out["alpha"][c] == pytest.approx(a)
        assert out["beta"][c] == pytest.approx(b)


def test_numpy_matches_the_torch_loss():
    """The score and the loss are the same KGE, with or without beta."""
    true, pred = _series()
    for use_beta in (True, False):
        ref = compute_kge_per_cell_np(true, pred, min_obs=2, use_beta=use_beta)["kge"]
        got, _ = compute_kge_per_cell(
            torch.tensor(true), torch.tensor(pred), use_beta=use_beta
        )
        np.testing.assert_allclose(got.numpy(), ref, atol=1e-5)


def test_dropping_beta_ignores_a_constant_offset():
    true, _ = _series()
    shifted = true + 0.1  # right timing and amplitude, wrong level
    with_beta = compute_kge_per_cell_np(true, shifted, min_obs=2)["kge"]
    without = compute_kge_per_cell_np(true, shifted, min_obs=2, use_beta=False)["kge"]
    assert np.all(with_beta < 0.9)
    np.testing.assert_allclose(without, 1.0)
    loss = CellKGELoss(use_beta=False)(torch.tensor(true), torch.tensor(shifted))
    assert loss.item() == pytest.approx(-1.0, abs=1e-4)


def test_cells_below_min_obs_are_nan():
    true, pred = _series(n_cells=3, n_days=50)
    true[1, 15:] = np.nan  # 15 observations
    out = compute_kge_per_cell_np(true, pred, min_obs=20)
    assert np.isfinite(out["kge"][[0, 2]]).all()
    assert np.isnan(out["kge"][1])
    assert out["n"].tolist() == [50, 15, 50]


# ==== the logged metric


def test_logged_metric_is_the_median_per_cell_kge():
    true, pred = _series()
    y, p = true[..., None], pred[..., None]
    got = KGECellMetric(min_obs=2)(y, p, ["vwc"], ~np.isnan(y))["vwc"]
    assert got == pytest.approx(np.median(compute_kge_per_cell_np(true, pred, 2)["kge"]))


def test_metric_collection_on_sequences_uses_every_day():
    """`metric_decorator` took day `idx` of an (N, T, C) array instead of
    target `idx`: calibration's logged metrics were day 0 only."""
    rng = np.random.default_rng(0)
    y = rng.random((50, 300, 1))
    p = y.copy()
    p[:, 0, 0] += 1.0  # an error on day 0 only
    got = MetricCollection([RMSEMetric()])(y, p, ["vwc"], ~np.isnan(y))
    assert got["vwc"]["RMSEMetric"] == pytest.approx(np.sqrt(1 / 300))


def test_metric_decorator_still_handles_samples_by_target():
    rng = np.random.default_rng(1)
    y = rng.random((100, 2))
    p = y.copy()
    p[:, 1] += 0.5
    got = MetricCollection([RMSEMetric()])(y, p, ["a", "b"], ~np.isnan(y))
    assert got["a"]["RMSEMetric"] == pytest.approx(0.0)
    assert got["b"]["RMSEMetric"] == pytest.approx(0.5)


def test_the_logged_key_is_parsed():
    log = "ConsoleLogger: val_ssm_kgecell_epoch = 0.2345\n"
    assert run_dpl_cycle.parse_cal_metrics(log)["kgecell"] == pytest.approx(0.2345)
    assert run_dpl_cycle.SURROGATE_KEYS == ("kgecell",)


# ==== the wflow score


@pytest.fixture
def wflow_pair(tmp_path, monkeypatch):
    """A tiny wflow output and observation file over 2017-2018."""
    days = pd.date_range("2017-01-01", "2018-12-31")
    lat, lon = np.arange(3.0), np.arange(4.0)
    true, pred = _series(n_cells=12, n_days=len(days))
    obs = true.reshape(3, 4, -1).copy()
    obs[0, 0, :] = np.nan  # a cell with no observations
    obs[0, 1, 30:] = np.nan  # a cell with 30, all in the first period
    xr.Dataset(
        {"theta": (("lat", "lon", "time"), obs)},
        coords={"lat": lat, "lon": lon, "time": days},
    ).to_netcdf(tmp_path / "obs.nc")
    # wflow writes lat descending, and vwc with a layer axis
    sim = pred.reshape(3, 4, -1)[::-1][None]
    xr.Dataset(
        {"vwc": (("layer", "lat", "lon", "time"), sim)},
        coords={"layer": [1], "lat": lat[::-1], "lon": lon, "time": days},
    ).to_netcdf(tmp_path / "out.nc")
    monkeypatch.setattr(run_dpl_cycle, "OBS", tmp_path / "obs.nc")
    monkeypatch.setattr(run_dpl_cycle, "_unpack_layer", lambda ds, var: ds.isel(layer=0))
    return tmp_path / "out.nc", obs, pred.reshape(3, 4, -1), days


def _cfg(**kw):
    ranges = {
        "train_temporal_range": ["2017-01-01", "2017-12-31"],
        "valid_temporal_range": ["2018-01-01", "2018-12-31"],
        "test_temporal_range": ["2021-01-01", "2022-12-31"],  # outside the window
        # the real mask is on the RT0 grid; test_dynamic_mask.py covers it
        "data_source.file.target_variables_dynamic_mask": None,
    }
    # the standard KGE, so the references below hold whatever the default
    # weights are (1, 0.25, 0.75 since 2026-09-22)
    kw.setdefault("kge_weights", (1.0, 1.0, 1.0))
    return run_dpl_cycle.CycleConfig(config_overrides=ranges, **kw)


def test_score_per_period(wflow_pair):
    out_nc, obs, sim, days = wflow_pair
    got = run_dpl_cycle.score(out_nc, _cfg())

    valid = days.year == 2018
    ref = compute_kge_per_cell_np(
        obs[..., valid].reshape(12, -1), sim[..., valid].reshape(12, -1), min_obs=20
    )
    counted = np.isfinite(ref["kge"])
    assert got["cells_valid"] == counted.sum() == 10
    for part in ("kge", "r", "alpha", "beta"):
        assert got[f"{part}_valid"] == pytest.approx(np.median(ref[part][counted]))
    assert got["cells_train"] == 11  # the 30-observation cell counts in 2017
    assert "kge_test" not in got and "cells_test" not in got  # not simulated
    assert "rmse" in got and "bias" in got  # kept for reference


def test_score_follows_the_switches(wflow_pair):
    out_nc, *_ = wflow_pair
    base = run_dpl_cycle.score(out_nc, _cfg())
    no_beta = run_dpl_cycle.score(out_nc, _cfg(kge_use_beta=False))
    assert no_beta["kge_valid"] != base["kge_valid"]
    assert no_beta["r_valid"] == base["r_valid"]
    strict = run_dpl_cycle.score(out_nc, _cfg(min_obs_per_cell=31))
    assert strict["cells_train"] == 10


def test_score_periods_come_from_the_calibration_config():
    periods = run_dpl_cycle.score_periods(run_dpl_cycle.CycleConfig())
    assert periods["test"] == ("2021-01-01", "2022-12-31")
    assert periods["valid"] == ("2020-01-01", "2020-12-31")


def test_calibration_config_carries_the_metric():
    i = run_dpl_cycle._metric_index("config_calibration_loop", "hython.metrics.KGECellMetric")
    assert isinstance(i, int)


def test_weights_scale_the_terms():
    true, pred = _series()
    ref = compute_kge_per_cell_np(true, pred, min_obs=2)
    unit = compute_kge_per_cell_np(true, pred, min_obs=2, weights=(1, 1, 1))["kge"]
    np.testing.assert_allclose(unit, ref["kge"])
    no_alpha = compute_kge_per_cell_np(true, pred, min_obs=2, weights=(1, 0, 1))["kge"]
    expected = 1 - np.sqrt((ref["r"] - 1) ** 2 + (ref["beta"] - 1) ** 2)
    np.testing.assert_allclose(no_alpha, expected)
    got, _ = compute_kge_per_cell(torch.tensor(true), torch.tensor(pred), weights=(1, 0, 1))
    np.testing.assert_allclose(got.numpy(), no_alpha, atol=1e-5)
    loss = CellKGELoss(weights=[1, 0, 1])(torch.tensor(true), torch.tensor(pred))
    assert np.isfinite(loss.item())
