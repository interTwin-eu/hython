"""Scaled data keeps the float type of its input.

Statistics read back from the YAML cache are float64. Without a cast, float32
data scaled with them came out float64, which doubled the memory of every
dataset built with `is_train=False`. On the 2017-2022 calibration window that
was +10 GB and helped push the calibration of 2026-09-24 out of memory.
"""

import numpy as np
import pytest
import xarray as xr
from hydra.utils import instantiate
from omegaconf import OmegaConf

from hython.scaler import Scaler, _match_dtype

VARS = ["precip", "temp"]


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return xr.Dataset({
        v: (("time", "cell"), rng.uniform(-5, 50, (30, 40)).astype("float32"))
        for v in VARS
    })


def make_scaler(tmp_path, use_cached=False):
    cfg = OmegaConf.create({
        "experiment_name": "t", "experiment_run": "t", "work_dir": str(tmp_path),
        "dynamic_inputs": VARS,
        "scaler": {"dynamic_inputs": {"lazy": False, "variant": [
            {"_target_": "hython.scaler.MinMax01Scaler", "variable": VARS}
        ]}},
    })
    return Scaler(instantiate(cfg), use_cached=use_cached)


def test_cached_statistics_keep_float32(tmp_path, data):
    """The valid/test path: statistics loaded from YAML, not computed."""
    train = make_scaler(tmp_path)
    train.load_or_compute(data, "dynamic_inputs", is_train=True, axes=("time", "cell"))
    train.write("dynamic_inputs")
    want = train.transform(data, "dynamic_inputs")

    valid = make_scaler(tmp_path)
    valid.load_or_compute(data, "dynamic_inputs", is_train=False, axes=("time", "cell"))
    assert valid.archive["dynamic_inputs"]["center"]["precip"].dtype == np.float64
    got = valid.transform(data, "dynamic_inputs")

    for v in VARS:
        assert got[v].dtype == np.float32
        np.testing.assert_allclose(got[v].values, want[v].values, rtol=1e-6, atol=1e-6)


def test_values_match_the_float64_path(tmp_path, data):
    """Only rounding changes: compare against scaling done in float64."""
    sc = make_scaler(tmp_path)
    sc.load_or_compute(data, "dynamic_inputs", is_train=True, axes=("time", "cell"))
    got = sc.transform(data, "dynamic_inputs")
    center = sc.archive["dynamic_inputs"]["center"]
    scale = sc.archive["dynamic_inputs"]["scale"]
    for v in VARS:
        ref = (data[v].values.astype("float64") - float(center[v])) / float(scale[v])
        np.testing.assert_allclose(got[v].values, ref, rtol=1e-6, atol=1e-6)


def test_integer_data_is_not_cast():
    stats = xr.Dataset({"landuse": ((), 1.5), "temp": ((), 2.5)})
    data = xr.Dataset({
        "landuse": ("cell", np.arange(4, dtype="int32")),
        "temp": ("cell", np.ones(4, dtype="float32")),
    })
    out = _match_dtype(stats, data)
    assert out["landuse"].dtype == np.float64
    assert float(out["landuse"]) == 1.5
    assert out["temp"].dtype == np.float32
