"""Archive alignment checks (H5).

The stores are opened separately and paired by position. A difference in
length or order marries one run's parameters to another run's soil moisture,
and nothing downstream notices - the shapes still work and training still
converges. These tests corrupt an archive in each of the ways that can happen
and assert the dataset refuses it at startup.

`check_alignment` is called on a stub rather than a built dataset: the point is
the index arithmetic, and a real `__init__` would need zarr stores on disk.
"""

import numpy as np
import pytest
import xarray as xr

from hython.datasets.wflow_sbm import WflowSBM_Pool

POOL = 10
RUNS = 3
N = POOL * RUNS


def store(n_cell, runs=RUNS, pool=POOL, lat=None, run_coord=None, lat_i=None):
    lat = np.arange(n_cell, dtype="float64") if lat is None else lat
    ds = xr.Dataset(coords={"cell": np.arange(n_cell)})
    ds = ds.assign_coords(lat=("cell", lat), lon=("cell", lat * 2))
    if run_coord is not None:
        ds = ds.assign_coords(run=("cell", run_coord))
    if lat_i is not None:
        ds = ds.assign_coords(lat_i=("cell", lat_i))
    return ds


def make(xs=None, xd=None, y=None, pool=POOL, runs=RUNS):
    ds = object.__new__(WflowSBM_Pool)
    ds.pool_size = pool
    ds.n_runs = runs
    good_run = np.repeat(np.arange(runs, dtype="int32"), pool)
    good_lat_i = np.tile(np.arange(pool, dtype="int32"), runs)
    ds.xs = store(N, run_coord=good_run, lat_i=good_lat_i) if xs is None else xs
    ds.xd = store(N) if xd is None else xd
    ds.y = store(N) if y is None else y
    return ds


def test_a_correct_archive_passes():
    make().check_alignment()


def test_a_shorter_dynamic_store_is_refused():
    with pytest.raises(ValueError, match="stores disagree"):
        make(xd=store(N - 1)).check_alignment()


def test_a_shorter_target_is_refused():
    with pytest.raises(ValueError, match="stores disagree"):
        make(y=store(N - 5)).check_alignment()


def test_a_partial_run_is_refused():
    with pytest.raises(ValueError, match="whole number of runs"):
        make(xs=store(N + 3), xd=store(N + 3), y=store(N + 3)).check_alignment()


def test_uneven_run_lengths_are_refused():
    """The case `n % pool_size == 0` cannot catch: one 10-row run followed by
    two 5-row runs is 20 rows, which divides by 5 cleanly."""
    ragged = np.concatenate([
        np.zeros(20, "int32"), np.ones(5, "int32"), np.full(5, 2, "int32"),
    ])
    xs = store(N, run_coord=ragged, lat_i=np.tile(np.arange(POOL, dtype="int32"), RUNS))
    with pytest.raises(ValueError, match="contiguous"):
        make(xs=xs).check_alignment()


def test_stores_pointing_at_different_cells_are_refused():
    """Same length, same run layout, different cells - the silent one."""
    shuffled = np.arange(N, dtype="float64")[::-1]
    with pytest.raises(ValueError, match="disagree on `lat`"):
        make(xd=store(N, lat=shuffled)).check_alignment()


def test_runs_covering_different_base_cells_are_refused():
    """Every run must list the same base cells in the same order, or
    `idx % pool_size` stops identifying a base cell."""
    bad = np.concatenate([
        np.arange(POOL, dtype="int32"),
        np.arange(POOL, dtype="int32")[::-1],
        np.arange(POOL, dtype="int32"),
    ])
    xs = store(N, run_coord=np.repeat(np.arange(RUNS, dtype="int32"), POOL), lat_i=bad)
    with pytest.raises(ValueError, match="same base cells"):
        make(xs=xs).check_alignment()


def test_a_single_run_archive_passes():
    ds = make(
        xs=store(POOL, run_coord=np.zeros(POOL, "int32"),
                 lat_i=np.arange(POOL, dtype="int32")),
        xd=store(POOL), y=store(POOL), runs=1,
    )
    ds.check_alignment()
