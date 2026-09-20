"""Fast integration tests for the multicycle pool archive (A1) and H1/H3.

Everything here is synthetic and hermetic: a tiny archive is built in tmp_path
with the same contract `pool_archive` writes, so the suite needs no CEPH data,
no wflow run and no GPU, and finishes in about a second.

The archive is built with run 1 = run 0 with `KsatVer` scaled *and* `vwc`
genuinely shifted. That is the pairing H1 has to read correctly, and
synthesising it here means the mechanical half of the H1 test does not wait on
a 54-minute wflow run. The real wflow run is still needed to show the
*surrogate* learns d(vwc)/d(theta); it is not needed to show the dataset pairs
the right rows.
"""

import json

import numpy as np
import pytest
import xarray as xr

POOL_SIZE = 20
RUNS = 2
NTIME = 60
KSAT_FACTOR = 3.0
VWC_SHIFT = 0.05

STATIC_VARS = ["KsatVer", "c", "f", "RootingDepth", "Sl", "thetaS", "thetaR",
               "SoilThickness", "M", "Kext", "Swood", "wflow_uparea",
               "wflow_landuse", "wflow_dem", "Slope", "WaterFrac"]
FORCING_VARS = ["precip", "pet", "temp"]


def _run(run, cycle, member, pool, time):
    """One run's worth of rows, in the layout pool_archive writes.

    Every run is generated from the *same* seeds, so the base cell values are
    identical across runs and only the two intended differences remain:
    `KsatVer` scaled, `vwc` shifted.
    """
    n = len(pool)
    sr = np.random.default_rng(1)
    ksat = sr.uniform(10, 500, n).astype("float32") * (KSAT_FACTOR ** run)

    static = xr.Dataset(
        {v: ("cell", sr.uniform(0.1, 10, n).astype("float32")) for v in STATIC_VARS}
        | {"KsatVer": ("cell", ksat),
           "mask_missing": ("cell", np.zeros(n, bool)),
           "mask_lake": ("cell", np.zeros(n, bool))}
    )
    # the forcing is identical in every run; only vwc moves
    fr = np.random.default_rng(0)
    dynamic = xr.Dataset(
        {v: (("time", "cell"), fr.uniform(0, 5, (len(time), n)).astype("float32"))
         for v in FORCING_VARS}
        | {"vwc": (("time", "cell"),
                   (fr.uniform(0.1, 0.5, (len(time), n)) + VWC_SHIFT * run).astype("float32"))}
    )
    for ds in (static, dynamic):
        ds.coords.update({
            "lat": ("cell", pool[:, 0] * 0.01 + 45.0),
            "lon": ("cell", pool[:, 1] * 0.01 + 10.0),
            "lat_i": ("cell", pool[:, 0].astype("int32")),
            "lon_i": ("cell", pool[:, 1].astype("int32")),
            "run": ("cell", np.full(n, run, "int32")),
            "cycle": ("cell", np.full(n, cycle, "int32")),
            "member": ("cell", np.full(n, member, "int32")),
        })
    dynamic = dynamic.assign_coords(time=time)
    return static, dynamic


@pytest.fixture(scope="module")
def archive_dir(tmp_path_factory):
    """A two-run archive: static.zarr + dynamic.zarr, forcing beside the target."""
    tmp = tmp_path_factory.mktemp("archive")
    pool = np.stack([np.arange(POOL_SIZE) % 7, np.arange(POOL_SIZE) % 11], axis=1)
    time = xr.date_range("2017-01-01", periods=NTIME, freq="D")

    manifest = []
    for r in range(RUNS):
        cycle, member = (0, r)
        static, dynamic = _run(r, cycle, member, pool, time)
        manifest.append(dict(run=r, cycle=cycle, member=member,
                             static=f"s{r}.nc", output=f"o{r}.nc"))
        attrs = dict(pool_seed=42, pool_size=POOL_SIZE, runs=json.dumps(manifest))
        static.attrs.update(attrs); dynamic.attrs.update(attrs)
        mode = dict(mode="w") if r == 0 else dict(mode="a", append_dim="cell")
        static.chunk({"cell": POOL_SIZE}).to_zarr(tmp / "static.zarr", **mode)
        dynamic.chunk({"cell": POOL_SIZE, "time": -1}).to_zarr(tmp / "dynamic.zarr", **mode)

    return tmp


@pytest.fixture(scope="module")
def archive(archive_dir):
    return archive_dir / "static.zarr", archive_dir / "dynamic.zarr"


@pytest.fixture(scope="module")
def cfg(archive_dir):
    """The smallest config WflowSBM_Pool accepts, built through hydra.

    Written as YAML rather than a dict because `scaler.variant` holds
    instantiated objects, which OmegaConf will not carry in a plain dict.
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    return instantiate(OmegaConf.create(f"""
experiment_name: t
experiment_run: t
run_dir: {archive_dir}
seq_length: 10
data_lazy_load: false
train_temporal_range: ["2017-01-01", "2017-02-20"]
train_downsampler: null
dynamic_downsampler: null
static_inputs: {STATIC_VARS}
dynamic_inputs: {FORCING_VARS}
target_variables: [vwc]
mask_variables: [mask_missing, mask_lake]
scaling_use_cached: false
scaling_static_range: null
preprocessor: null
data_source:
  file:
    static_inputs: {archive_dir}/static.zarr
    dynamic_inputs: {archive_dir}/dynamic.zarr
    target_variables: {archive_dir}/dynamic.zarr
scaler:
  static_inputs:
    lazy: false
    variant:
      - _target_: hython.scaler.MinMax01Scaler
        variable: {STATIC_VARS}
  dynamic_inputs:
    lazy: false
    variant:
      - _target_: hython.scaler.MinMax01Scaler
        variable: {FORCING_VARS}
  target_variables: null
"""))


# ==== the archive contract (A1) -- these pass today


def test_shape_and_provenance(archive):
    s, d = (xr.open_zarr(p) for p in archive)
    assert s.sizes["cell"] == d.sizes["cell"] == POOL_SIZE * RUNS
    assert s.attrs["pool_size"] == POOL_SIZE
    assert len(json.loads(s.attrs["runs"])) == RUNS
    # each run is pool_size contiguous rows, in order
    np.testing.assert_array_equal(
        s.run.values, np.repeat(np.arange(RUNS, dtype="int32"), POOL_SIZE)
    )
    np.testing.assert_array_equal(s.run.values, d.run.values)


def test_cell_is_not_indexed(archive):
    for p in archive:
        ds = xr.open_zarr(p)
        assert "cell" not in ds.coords
        assert "cell" not in ds.indexes


def test_coords_agree_between_stores(archive):
    s, d = (xr.open_zarr(p) for p in archive)
    for k in ("lat", "lon", "lat_i", "lon_i"):
        np.testing.assert_array_equal(s[k].values, d[k].values)


def test_forcing_identical_across_runs(archive):
    _, d = archive
    d = xr.open_zarr(d)
    a = d.precip.isel(cell=slice(0, POOL_SIZE)).values
    b = d.precip.isel(cell=slice(POOL_SIZE, None)).values
    np.testing.assert_array_equal(a, b)


def test_runs_differ_in_theta_and_target(archive):
    """The pairing the surrogate learns from: same cell, same weather, different
    theta -> different vwc. Without this the archive teaches d(vwc)/d(theta)=0."""
    s, d = (xr.open_zarr(p) for p in archive)
    lo, hi = slice(0, POOL_SIZE), slice(POOL_SIZE, None)
    np.testing.assert_allclose(
        s.KsatVer.isel(cell=hi).values, s.KsatVer.isel(cell=lo).values * KSAT_FACTOR,
        rtol=1e-5,
    )
    assert not np.allclose(d.vwc.isel(cell=hi).values, d.vwc.isel(cell=lo).values)
    # and the base cell is the same cell on the map
    np.testing.assert_array_equal(s.lat_i.isel(cell=lo).values, s.lat_i.isel(cell=hi).values)


# ==== the dataset reading that layout (H1) -- red until H1 lands


def test_dataset_reads_stacked_layout(cfg):
    """The mechanical half of the H1 test, with no wflow run needed.

    A sample must pair the static of row i with the target of row i and the
    forcing of row i. The archive is built so that two rows of the same base
    cell differ in KsatVer and vwc but share their forcing exactly, so a
    dataset that mispaired rows could not produce this pattern by accident.
    """
    from hython.datasets import WflowSBM_Pool
    from hython.scaler import Scaler

    ds = WflowSBM_Pool(cfg, Scaler(cfg), True, "train")

    # one sample per (cell, start time). `time_size` is captured before the
    # xarray objects are replaced by tensors.
    n_time = ds.time_size - cfg.seq_length
    assert len(ds) == POOL_SIZE * RUNS * n_time

    # the same base cell in run 0 and run 1, at the same start time
    lo = ds[0]
    hi = ds[POOL_SIZE * n_time]

    ksat = STATIC_VARS.index("KsatVer")
    np.testing.assert_allclose(lo["xd"], hi["xd"], rtol=1e-6)   # same weather
    assert not np.allclose(lo["xs"][ksat], hi["xs"][ksat])      # different theta
    assert not np.allclose(lo["y"], hi["y"])                    # different vwc


# ==== back to xarray -- hython.utils.pool_to_xarray / scatter_pool_to_map


def test_flat_prediction_round_trips_through_the_cell_axis(archive):
    """A prediction over the archive comes back flat and cell-major.

    `pool_to_xarray` must reshape it the way `WflowSBM_Pool` laid the samples
    out - `itertools.product(cells, times)`, so cell-major - and carry
    `lat_i`/`lon_i` through, because `scatter_pool_to_map` needs them.
    """
    from hython.utils import pool_to_xarray

    _, dynamic = archive
    ds = xr.open_zarr(dynamic)
    n_cell, n_time = ds.sizes["cell"], ds.sizes["time"]

    variables = ["vwc", "q"]
    flat = np.arange(n_cell * n_time * len(variables), dtype="float32")
    out = pool_to_xarray(flat, ds, variables)

    assert out.sizes == {"cell": n_cell, "time": n_time}
    assert set(out.data_vars) == set(variables)
    for k in ("lat", "lon", "lat_i", "lon_i"):
        np.testing.assert_array_equal(out[k].values, ds[k].values)

    # cell-major: row i's series is the i-th block, not every n_cell-th element
    block = flat.reshape(n_cell, n_time, len(variables))
    np.testing.assert_allclose(out.vwc.isel(cell=0).values, block[0, :, 0])
    np.testing.assert_allclose(out.q.isel(cell=3).values, block[3, :, 1])


def test_scatter_refuses_a_multi_run_input(archive):
    """Runs are stacked along `cell`, so a base cell maps to the same pixel in
    every run. Scattering all of them would keep whichever run numpy wrote last
    and silently drop the rest, so it must raise instead."""
    from hython.utils import pool_to_xarray, scatter_pool_to_map

    _, dynamic = archive
    ds = xr.open_zarr(dynamic)
    n_cell, n_time = ds.sizes["cell"], ds.sizes["time"]
    out = pool_to_xarray(np.arange(n_cell * n_time, dtype="float32"), ds, ["vwc"])

    grid = _grid_for(ds)
    with pytest.raises(ValueError, match="ambiguous"):
        scatter_pool_to_map(out, grid, crs=None)


def _grid_for(ds):
    return xr.Dataset(coords={
        "latitude": np.arange(int(ds.lat_i.max()) + 3, dtype="float64"),
        "longitude": np.arange(int(ds.lon_i.max()) + 3, dtype="float64"),
    })


def test_scatter_fills_pool_cells_and_leaves_the_rest_nan(archive):
    """One run scatters cleanly: its pool cells carry values, the rest is NaN.

    `grid` is passed in rather than read from a wflow path, which is what let
    these two move out of the script and into hython.
    """
    from hython.utils import pool_to_xarray, scatter_pool_to_map

    _, dynamic = archive
    run0 = xr.open_zarr(dynamic).isel(cell=slice(0, POOL_SIZE))
    n_time = run0.sizes["time"]

    out = pool_to_xarray(
        np.arange(POOL_SIZE * n_time, dtype="float32"), run0, ["vwc"]
    )

    grid = _grid_for(run0)
    n_lat, n_lon = grid.sizes["latitude"], grid.sizes["longitude"]
    m = scatter_pool_to_map(out, grid, crs=None)

    assert m.sizes == {"time": n_time, "lat": n_lat, "lon": n_lon}

    got = m.vwc.values[:, run0.lat_i.values, run0.lon_i.values]
    np.testing.assert_allclose(got, out.vwc.transpose("time", "cell").values)
    assert np.isnan(m.vwc.values).sum() == (n_lat * n_lon - POOL_SIZE) * n_time
