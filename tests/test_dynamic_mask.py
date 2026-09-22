"""Dynamic RT0 mask (D4): invalid days become NaN in the calibration target and
in the wflow score, only in the configured months."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from omegaconf import OmegaConf

from hython.datasets.wflow_sbm import WflowSBMCal
from hython.utils import apply_dynamic_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
run_dpl_cycle = pytest.importorskip("run_dpl_cycle")

DAYS = pd.date_range("2017-01-01", "2018-12-31")
LAT, LON = np.arange(2.0), np.arange(3.0)
DJFMA = [12, 1, 2, 3, 4]


def _grid(values, days=DAYS, lat=LAT, name="valid"):
    return xr.Dataset(
        {name: (("time", "lat", "lon"), values)},
        coords={"time": days, "lat": lat, "lon": LON},
    )


@pytest.fixture
def files(tmp_path):
    """Seasonal observations; the mask is invalid on even days, and
    always invalid in cell (1, 2)."""
    shape = (len(DAYS), len(LAT), len(LON))
    rng = np.random.default_rng(1)
    season = 0.25 + 0.05 * np.sin(np.arange(len(DAYS)) * 2 * np.pi / 365)[:, None, None]
    obs = _grid((season + 0.02 * rng.random(shape)).astype("float32"), name="ssm")
    obs.to_netcdf(tmp_path / "obs.nc")
    valid = np.ones(shape, "uint8")
    valid[::2] = 0
    valid[:, 1, 2] = 0
    _grid(valid).assign_coords(layer=1).to_netcdf(tmp_path / "mask.nc")
    return tmp_path, obs, valid


# ==== the helper


def test_the_mask_applies_only_in_its_months(files):
    d, obs, valid = files
    got = apply_dynamic_mask(obs, d / "mask.nc", DJFMA).ssm.values
    in_months = DAYS.month.isin(DJFMA)
    np.testing.assert_array_equal(np.isnan(got[in_months]), valid[in_months] == 0)
    assert not np.isnan(got[~in_months]).any()


def test_no_months_applies_the_mask_all_year(files):
    d, obs, valid = files
    got = apply_dynamic_mask(obs, d / "mask.nc", None).ssm.values
    np.testing.assert_array_equal(np.isnan(got), valid == 0)


def test_a_mask_on_another_grid_raises(files, tmp_path):
    d, obs, valid = files
    _grid(valid, lat=LAT + 0.5).to_netcdf(tmp_path / "shifted.nc")
    with pytest.raises(ValueError):
        apply_dynamic_mask(obs, tmp_path / "shifted.nc", DJFMA)


def test_a_mask_missing_days_raises(files, tmp_path):
    d, obs, valid = files
    _grid(valid[:100], days=DAYS[:100]).to_netcdf(tmp_path / "short.nc")
    with pytest.raises(KeyError):
        apply_dynamic_mask(obs, tmp_path / "short.nc", DJFMA)


# ==== the calibration loader


class StubScaler:
    use_cached = False
    flag_stats_computed = False

    def load_or_compute(self, data, type, is_train, axes=None, **kwargs):
        pass

    def transform(self, data, type):
        return data

    def write(self, type):
        pass


def _cal_cfg(d, **changes):
    xr.Dataset(
        {"precip": (("lat", "lon", "time"), np.ones((2, 3, len(DAYS)), "float32"))},
        coords={"lat": LAT, "lon": LON, "time": DAYS},
    ).to_zarr(d / "dynamic.zarr", mode="w")
    grid = {"lat": LAT, "lon": LON}
    xr.Dataset({"s1": (("lat", "lon"), np.ones((2, 3)))}, coords=grid).to_zarr(
        d / "static.zarr", mode="w")
    xr.Dataset(
        {
            "p1": (("lat", "lon"), np.ones((2, 3))),
            "mask_missing": (("lat", "lon"), np.zeros((2, 3), bool)),
            "mask_lake": (("lat", "lon"), np.zeros((2, 3), bool)),
        },
        coords=grid,
    ).to_zarr(d / "params.zarr", mode="w")
    cfg = {
        "data_source": {"file": {
            "dynamic_inputs": str(d / "dynamic.zarr"),
            "static_inputs": str(d / "static.zarr"),
            "target_variables": str(d / "obs.nc"),
            "static_parameter_inputs": str(d / "params.zarr"),
            "mask_variables": str(d / "params.zarr"),
            "target_variables_dynamic_mask": str(d / "mask.nc"),
        }},
        "dynamic_inputs": ["precip"],
        "static_inputs": ["s1"],
        "target_variables": ["ssm"],
        "head_model_inputs": {"aux_feat": ["p1"], "cal_param": ["x"]},
        "mask_variables": ["mask_missing", "mask_lake"],
        "static_inputs_mask": ["s1"],
        "train_temporal_range": ["2017-01-01", "2017-12-31"],
        "valid_temporal_range": ["2018-01-01", "2018-12-31"],
        "train_downsampler": None,
        "valid_downsampler": None,
        "target_has_missing_dates": True,
        "data_lazy_load": False,
        "warmup_steps": 0,
        "target_dynamic_mask_months": DJFMA,
    }
    cfg.update(changes)
    return OmegaConf.create(cfg)


def test_the_calibration_target_is_masked(files):
    d, _, valid = files
    ds = WflowSBMCal(_cal_cfg(d), StubScaler(), is_train=True, period="train")
    y = ds.y.ssm.isel(lat=0, lon=0).to_series()
    days = DAYS[DAYS.year == 2017]
    in_months = days.month.isin(DJFMA)
    ref = valid[: len(days), 0, 0] == 0
    np.testing.assert_array_equal(y.isna().values[in_months], ref[in_months])
    assert y[~in_months].notna().all()


def test_the_mask_also_reaches_the_warm_up_sequence(files):
    d, _, valid = files
    ds = WflowSBMCal(_cal_cfg(d, warmup_steps=30), StubScaler(), is_train=True, period="valid")
    y = ds.y.ssm.isel(lat=0, lon=0).to_series()
    jan = DAYS.slice_indexer("2018-01-01", "2018-01-31")
    np.testing.assert_array_equal(y["2018-01"].isna().values, valid[jan, 0, 0] == 0)
    assert y["2018-06-01":"2018-08-31"].notna().all()


def test_a_cell_the_mask_empties_is_dropped(files):
    """No static target mask: the cells kept are those with an observation left."""
    d, _, _ = files
    ds = WflowSBMCal(_cal_cfg(d, target_dynamic_mask_months=None), StubScaler(),
                     is_train=True, period="train")
    cells = {tuple(c) for c in ds.coord_cells}
    assert (1, 2) not in cells and len(cells) == 5


def test_without_the_key_nothing_is_masked(files):
    d, _, _ = files
    cfg = _cal_cfg(d)
    del cfg.data_source.file["target_variables_dynamic_mask"]
    ds = WflowSBMCal(cfg, StubScaler(), is_train=True, period="train")
    assert ds.y.ssm.notnull().all()


# ==== the wflow score


def _sim(d, obs):
    rng = np.random.default_rng(0)
    vwc = obs.ssm.values + 0.1 * rng.standard_normal(obs.ssm.shape)
    # wflow writes lat descending, and vwc with a layer axis
    xr.Dataset(
        {"vwc": (("layer", "time", "lat", "lon"), vwc[None, :, ::-1])},
        coords={"layer": [1], "time": DAYS, "lat": LAT[::-1], "lon": LON},
    ).to_netcdf(d / "out.nc")
    return d / "out.nc"


def _score_cfg(d, mask=True, months=DJFMA):
    over = {
        "train_temporal_range": ["2017-01-01", "2017-12-31"],
        "valid_temporal_range": ["2018-01-01", "2018-12-31"],
        "test_temporal_range": ["2021-01-01", "2022-12-31"],
        "data_source.file.target_variables_dynamic_mask": str(d / "mask.nc") if mask else None,
        "target_dynamic_mask_months": months,
    }
    return run_dpl_cycle.CycleConfig(config_overrides=over, kge_weights=(1.0, 1.0, 1.0))


def test_score_sees_the_masked_observations(files, monkeypatch):
    d, obs, valid = files
    monkeypatch.setattr(run_dpl_cycle, "OBS", d / "obs.nc")
    monkeypatch.setattr(run_dpl_cycle, "_unpack_layer", lambda ds, var: ds.isel(layer=0))
    out = _sim(d, obs)
    masked = run_dpl_cycle.score(out, _score_cfg(d))
    plain = run_dpl_cycle.score(out, _score_cfg(d, mask=False))
    assert plain["cells_train"] == 6
    assert masked["cells_train"] == 6  # cell (1, 2) keeps its May-Nov days
    all_year = run_dpl_cycle.score(out, _score_cfg(d, months=None))
    assert all_year["cells_train"] == 5  # cell (1, 2) has no observation left
    assert masked["rmse"] != plain["rmse"]


def test_score_reads_the_mask_from_the_calibration_config():
    path, months = run_dpl_cycle.score_mask(run_dpl_cycle.CycleConfig())
    assert path.endswith("alps_rt0old_dynamic_mask_2017-2022.nc")
    assert list(months) == DJFMA


def test_the_calibration_config_has_no_static_target_mask():
    cfg = OmegaConf.load(run_dpl_cycle.WD_CONFIG / "config_calibration_loop.yaml")
    assert "target_variables_mask" not in cfg.data_source.file
    step = next(s for s in cfg.training_pipeline.steps if "target_dynamic_mask_months" in s)
    assert list(step["target_dynamic_mask_months"]) == DJFMA  # resolved from the top level
