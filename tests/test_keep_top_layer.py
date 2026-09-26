"""Clean wflow outputs are cut to the top soil layer after the run.

Wflow 0.8.1 writes vwc for all 4 layers in gridded output and cannot select
one, so `keep_top_layer` does it afterwards. Scoring must read the cut file
exactly as it read the full one.
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
run_dpl_cycle = pytest.importorskip("run_dpl_cycle")

keep_top_layer = run_dpl_cycle.keep_top_layer
_unpack_layer = run_dpl_cycle._unpack_layer
LAYER = run_dpl_cycle.LAYER


def cfg(**kw):
    return dataclasses.replace(run_dpl_cycle.CycleConfig(), **kw)


def write_output(path, n_layer=4, fill=-9999.0):
    rng = np.random.default_rng(0)
    vwc = rng.random((6, n_layer, 3, 4)).astype("float32")
    vwc[0, :, 0, 0] = np.nan
    ds = xr.Dataset(
        {"vwc": (("time", "layer", "lat", "lon"), vwc)},
        coords={
            "time": np.arange("2021-01-01", "2021-01-07", dtype="datetime64[D]"),
            "layer": np.arange(1.0, n_layer + 1),
            "lat": [44.0, 44.1, 44.2],
            "lon": [5.0, 5.1, 5.2, 5.3],
        },
    )
    ds.to_netcdf(path, engine="netcdf4", encoding={"vwc": {"_FillValue": fill, "zlib": True}})
    return ds


def test_keeps_only_the_top_layer(tmp_path):
    out = tmp_path / "output_cycle1_cal.nc"
    full = write_output(out)
    keep_top_layer(out, cfg())
    with xr.open_dataset(out, engine="netcdf4") as cut:
        assert cut.sizes["layer"] == 1
        assert cut["layer"].values.tolist() == [full["layer"].values[LAYER]]
        np.testing.assert_array_equal(
            cut["vwc"].isel(layer=0).values, full["vwc"].isel(layer=LAYER).values
        )


def test_scoring_reads_the_same_values(tmp_path):
    """`score()` goes through `_unpack_layer`; it must see identical data."""
    out = tmp_path / "out.nc"
    full = write_output(out)
    expected = _unpack_layer(full.copy(), "vwc")["vwc"]
    keep_top_layer(out, cfg())
    with xr.open_dataset(out, engine="netcdf4") as cut:
        got = _unpack_layer(cut.load(), "vwc")["vwc"]
    xr.testing.assert_identical(got.drop_encoding(), expected.drop_encoding())


def test_fill_value_and_nan_survive(tmp_path):
    out = tmp_path / "out.nc"
    write_output(out, fill=-9999.0)
    keep_top_layer(out, cfg())
    with xr.open_dataset(out, engine="netcdf4") as cut:
        assert cut["vwc"].encoding["_FillValue"] == -9999.0
        assert np.isnan(cut["vwc"].values[0, 0, 0, 0])


def test_no_temporary_file_left(tmp_path):
    out = tmp_path / "out.nc"
    write_output(out)
    keep_top_layer(out, cfg())
    assert [p.name for p in tmp_path.iterdir()] == ["out.nc"]


def test_single_layer_file_is_left_alone(tmp_path):
    out = tmp_path / "out.nc"
    write_output(out, n_layer=1)
    before = out.stat().st_mtime_ns
    keep_top_layer(out, cfg())
    assert out.stat().st_mtime_ns == before


def test_on_by_default():
    assert run_dpl_cycle.CycleConfig().top_layer_only is True
