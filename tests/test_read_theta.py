"""Gap filling for calibrated parameters (`read_theta`).

dPL leaves ~6.7% of land cells empty. `interpolate_na` closes almost all of
it; what is left sits at the ends of rows, where the nearest valid value on
that row is the sensible fill. That is `ffill().bfill()` semantics, done in
numpy because xarray routes those through `bottleneck`, which is not installed
and is not worth a dependency for the ~49 cells measured on a real cycle-0
`theta_cal.nc`.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
run_dpl_cycle = pytest.importorskip("run_dpl_cycle")

fill = run_dpl_cycle._fill_nearest_along_last
NAN = np.nan


def test_trailing_gap_copies_the_last_valid_value():
    got = fill(np.array([[1.0, 2.0, NAN, NAN]]))
    np.testing.assert_array_equal(got, [[1.0, 2.0, 2.0, 2.0]])


def test_leading_gap_copies_the_first_valid_value():
    got = fill(np.array([[NAN, NAN, 3.0, 4.0]]))
    np.testing.assert_array_equal(got, [[3.0, 3.0, 3.0, 4.0]])


def test_interior_gap_takes_the_value_on_its_left():
    """ffill runs before bfill, as in the original `.ffill().bfill()`."""
    got = fill(np.array([[1.0, NAN, 5.0]]))
    np.testing.assert_array_equal(got, [[1.0, 1.0, 5.0]])


def test_rows_are_filled_independently():
    got = fill(np.array([[1.0, NAN], [NAN, 9.0]]))
    np.testing.assert_array_equal(got, [[1.0, 1.0], [9.0, 9.0]])


def test_an_entirely_empty_row_stays_empty():
    """Same as ffill/bfill. `read_theta` warns rather than inventing a value."""
    got = fill(np.array([[NAN, NAN, NAN]]))
    assert np.isnan(got).all()


def test_a_full_row_is_untouched():
    a = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(fill(a.copy()), a)


def test_works_on_more_than_two_dimensions():
    a = np.array([[[1.0, NAN]], [[NAN, 4.0]]])
    np.testing.assert_array_equal(fill(a), [[[1.0, 1.0]], [[4.0, 4.0]]])


def test_the_input_is_not_modified():
    a = np.array([[1.0, NAN]])
    fill(a)
    assert np.isnan(a[0, 1])


# ==== read_theta end to end


@pytest.fixture
def theta_file(tmp_path):
    """lat/lon named as dPL writes them, with an interior and a trailing gap."""
    a = np.arange(12, dtype=float).reshape(3, 4)
    a[0, 1] = NAN      # interior: interpolate_na closes this
    a[1, 3] = NAN      # trailing: only the nearest-value fill closes this
    ds = xr.Dataset({"KsatVer": (("lat", "lon"), a)},
                    coords={"lat": np.arange(3), "lon": np.arange(4)})
    path = tmp_path / "theta_cal.nc"
    ds.to_netcdf(path)
    return path


def test_read_theta_renames_onto_the_wflow_grid(theta_file):
    out = run_dpl_cycle.read_theta(theta_file)
    assert set(out.dims) == {"latitude", "longitude"}


def test_read_theta_leaves_no_gaps(theta_file):
    out = run_dpl_cycle.read_theta(theta_file)
    assert not bool(out["KsatVer"].isnull().any())


def test_read_theta_interpolates_interior_gaps(theta_file):
    """Between 0 and 2, so 1 - not a copy of a neighbour."""
    out = run_dpl_cycle.read_theta(theta_file)
    assert float(out["KsatVer"][0, 1]) == pytest.approx(1.0)


def test_read_theta_copies_a_neighbour_at_a_row_end(theta_file):
    """Row 1 is 4,5,6,NaN - the end takes 6, not a global mean."""
    out = run_dpl_cycle.read_theta(theta_file)
    assert float(out["KsatVer"][1, 3]) == pytest.approx(6.0)


def test_read_theta_does_not_need_bottleneck():
    with pytest.raises(ModuleNotFoundError):
        __import__("bottleneck")


# ==== clamping to physical bounds


@pytest.fixture
def out_of_bounds_theta():
    """A theta map that breaks both ends of KsatVer's [1.0, 8000.0]."""
    import xarray as xr
    base = xr.open_dataset(
        "/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/staticmaps.nc"
    )
    shape = base["thetaS"].shape
    ds = xr.Dataset(coords=base["thetaS"].coords)
    for name, (lo, hi) in run_dpl_cycle.CAL_PARAMS.items():
        a = np.full(shape, (lo + hi) / 2, dtype=float)
        a[0, 0] = -2.6        # below - the negative conductivity seen for real
        a[0, 1] = hi * 10     # above
        dims = base["thetaS"].dims
        ds[name] = (dims[-2:], a) if len(dims) == 2 else (dims, a)
    return ds


@pytest.mark.skipif(
    not Path("/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/staticmaps.nc").exists(),
    reason="needs the wflow staticmaps",
)
def test_write_staticmaps_clamps_to_the_declared_bounds(out_of_bounds_theta, tmp_path):
    """`perturb` keeps members in bounds by construction, but the clean
    theta_cal map comes straight from calibration and is bounded by nothing."""
    import xarray as xr
    dest = tmp_path / "staticmaps.nc"
    run_dpl_cycle.write_staticmaps(out_of_bounds_theta, dest)

    written = xr.open_dataset(dest)
    for name, (lo, hi) in run_dpl_cycle.CAL_PARAMS.items():
        a = written[name]
        a = a.isel(layer=0) if "layer" in a.dims else a
        valid = a.values[np.isfinite(a.values)]
        assert valid.min() >= lo, f"{name} below {lo}: {valid.min()}"
        assert valid.max() <= hi, f"{name} above {hi}: {valid.max()}"
