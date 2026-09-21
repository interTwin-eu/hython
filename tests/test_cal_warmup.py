"""H10: calibration sequences start `warmup_steps` before the period, and the
warm-up is not scored."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from omegaconf import OmegaConf

from hython.datasets.wflow_sbm import WflowSBMCal, warmup_window

DAYS = pd.date_range("2017-01-01", "2017-12-31")
MISSING = [pd.Timestamp("2017-07-10"), pd.Timestamp("2017-08-20")]  # like RT0's gaps


class StubScaler:
    """Identity scaler that records the data it computes statistics on."""

    use_cached = False
    flag_stats_computed = False

    def __init__(self):
        self.seen = {}

    def load_or_compute(self, data, type, is_train, axes=None, **kwargs):
        self.seen[type] = data

    def transform(self, data, type):
        return data

    def write(self, type):
        pass


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    d = tmp_path_factory.mktemp("cal")
    lat, lon = np.arange(2.0), np.arange(3.0)
    shape = (len(lat), len(lon), len(DAYS))
    step = np.broadcast_to(np.arange(len(DAYS), dtype="float32"), shape)

    # forcing value = day index, so a test can tell which days it got
    xr.Dataset(
        {"precip": (("lat", "lon", "time"), step.copy())},
        coords={"lat": lat, "lon": lon, "time": DAYS},
    ).to_zarr(d / "dynamic.zarr")

    target = xr.Dataset(
        {"vwc": (("lat", "lon", "time"), 1000 + step.copy())},
        coords={"lat": lat, "lon": lon, "time": DAYS},
    ).drop_sel(time=MISSING)
    target.to_netcdf(d / "target.nc")

    grid = {"lat": lat, "lon": lon}
    xr.Dataset({"s1": (("lat", "lon"), np.ones((2, 3)))}, coords=grid).to_zarr(d / "static.zarr")
    xr.Dataset(
        {
            "p1": (("lat", "lon"), np.ones((2, 3))),
            "mask_missing": (("lat", "lon"), np.zeros((2, 3), bool)),
            "mask_lake": (("lat", "lon"), np.zeros((2, 3), bool)),
        },
        coords=grid,
    ).to_zarr(d / "params.zarr")

    return {
        "data_source": {
            "file": {
                "dynamic_inputs": str(d / "dynamic.zarr"),
                "static_inputs": str(d / "static.zarr"),
                "target_variables": str(d / "target.nc"),
                "static_parameter_inputs": str(d / "params.zarr"),
                "mask_variables": str(d / "params.zarr"),
            }
        },
        "dynamic_inputs": ["precip"],
        "static_inputs": ["s1"],
        "target_variables": ["vwc"],
        "head_model_inputs": {"aux_feat": ["p1"], "cal_param": ["x"]},
        "mask_variables": ["mask_missing", "mask_lake"],
        "static_inputs_mask": ["s1"],
        "train_temporal_range": ["2017-01-01", "2017-06-30"],
        "valid_temporal_range": ["2017-07-01", "2017-12-31"],
        "train_downsampler": None,
        "valid_downsampler": None,
        "target_has_missing_dates": True,
        "data_lazy_load": False,
        "warmup_steps": 120,
    }


def _dataset(cfg, period, **changes):
    scaler = StubScaler()
    ds = WflowSBMCal(OmegaConf.create({**cfg, **changes}), scaler, is_train=True, period=period)
    return ds, scaler


def test_train_starts_scoring_120_days_after_the_forcing(cfg):
    ds, _ = _dataset(cfg, "train")
    t = pd.DatetimeIndex(ds.xd.time.values)
    assert t[0] == pd.Timestamp("2017-01-01") and t[-1] == pd.Timestamp("2017-06-30")
    y = ds.y.vwc.isel(lat=0, lon=0).to_series()
    assert y[: "2017-04-30"].isna().all()
    assert y["2017-05-01":].notna().all()
    assert y.first_valid_index() == pd.Timestamp("2017-05-01")


def test_valid_takes_its_warm_up_from_the_days_before(cfg):
    ds, _ = _dataset(cfg, "valid")
    t = pd.DatetimeIndex(ds.xd.time.values)
    assert t[0] == pd.Timestamp("2017-07-01") - pd.Timedelta(days=120)
    y = ds.y.vwc.isel(lat=0, lon=0).to_series()
    assert y[:"2017-06-30"].isna().all()
    assert y.first_valid_index() == pd.Timestamp("2017-07-01")


def test_forcing_stays_daily_and_missing_target_days_are_nan(cfg):
    ds, _ = _dataset(cfg, "valid")
    t = pd.DatetimeIndex(ds.xd.time.values)
    assert (np.diff(t) == np.timedelta64(1, "D")).all()
    y = ds.y.vwc.isel(lat=0, lon=0).to_series()
    for day in MISSING:
        assert np.isnan(y[day])
    # target and forcing line up: target = 1000 + the forcing's day index
    x = ds.xd.precip.isel(lat=0, lon=0).to_series()
    scored = y.dropna().index
    np.testing.assert_array_equal(y[scored].values, 1000 + x[scored].values)


def test_scaling_statistics_see_only_the_period(cfg):
    _, scaler = _dataset(cfg, "valid")
    t = pd.DatetimeIndex(scaler.seen["dynamic_inputs"].time.values)
    assert t[0] == pd.Timestamp("2017-07-01")
    t = pd.DatetimeIndex(scaler.seen["target_variables"].time.values)
    assert t[0] == pd.Timestamp("2017-07-01")


def test_samples_carry_the_warm_up(cfg):
    ds, _ = _dataset(cfg, "valid")
    s = ds[0]
    assert s["xd"].shape[0] == 184 + 120
    assert s["y"].shape[0] == 184 + 120
    assert s["y"][:120].isnan().all()


def test_no_warm_up_keeps_the_old_behaviour(cfg):
    ds, _ = _dataset(cfg, "valid", warmup_steps=0)
    t = pd.DatetimeIndex(ds.xd.time.values)
    assert t[0] == pd.Timestamp("2017-07-01")
    assert len(t) == 184 - len(MISSING)  # forcing cut to the target's days, as before


def test_warmup_window_indices():
    times = DAYS.values
    assert warmup_window(times, slice("2017-01-01", "2017-06-30"), 120) == (0, 120, 180)
    assert warmup_window(times, slice("2017-07-01", "2017-12-31"), 120) == (61, 181, 364)
    assert warmup_window(times, slice("2017-07-01", "2017-12-31"), 0) == (181, 181, 364)


def test_warmup_window_raises_when_nothing_is_left_to_score():
    with pytest.raises(ValueError, match="no step left"):
        warmup_window(DAYS.values, slice("2017-01-01", "2017-03-31"), 120)
