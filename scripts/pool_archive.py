"""A1 - build the pooled surrogate archive.

Each wflow run writes ~6 GB over the full 568x1220 map, of which training reads
a few thousand cells. Instead of storing the map, we fix one random set of
cells at cycle 0 - the *pool* - and store only those.

Runs are stacked along a single `cell` axis rather than given a `cycle`
dimension, which is how Tsai's released data is laid out and means hython
barely changes:

    emo1_dynamic_cycle.zarr  (time, cell)  forcing + vwc, runs x pool_size rows
    emo1_static_cycle.zarr   (cell,)       theta,         runs x pool_size rows

Row `i` is base cell `i % pool_size` from run `i // pool_size`. Both stores
carry every run, so they stay the same length and are read positionally.

**The forcing lives in the dynamic store beside the target**, repeated once per
run, because that is the dataset's contract: `WflowSBM_Pool.__init__` opens
`urls["dynamic_inputs"]` once and takes both out of it
(`self.xd = data_dynamic[dynamic_inputs]`, `self.y = data_dynamic[
target_variables]`, wflow_sbm.py:30-34), and `urls["target_variables"]` is
never opened. Storing the forcing once in a third file and looking it up with
`i % pool_size` would save 1.46 GB over the 11-run archive and cost a second
file open, a modulo in `__getitem__`, and two stores of unequal length. Not
worth it.

**The archive describes itself. There is no pool sidecar file.** Each row
carries `lat`/`lon`, the integer grid indices `lat_i`/`lon_i`, and the run it
came from as `run`/`cycle`/`member`; each store records `pool_seed`,
`pool_size` and a JSON `runs` manifest in its attrs. So the pool is read
back from the archive, run boundaries are `n // pool_size`, and results go
back onto the map through `lat_i`/`lon_i`. An earlier draft kept a separate
`emo1_pool_coords.npz`; its `lat`/`lon` duplicated the archive and its `idx`
was exactly derivable from them, which is a second source of truth for one
fact - the drift H5 exists to catch.

`run` is derivable from position (`i // pool_size`) but is stored anyway: it
costs 2.4 MB at 11 runs, it cannot drift because it is written in the same
`to_zarr` call as the data, and it lets `verify()` assert the run layout rather
than trust the arithmetic. `cycle` and `member` are **not** derivable - A2
gives cycle 0 four runs and later cycles one each, so `run -> cycle` is
`[0,0,0,0,1,2,3,...]`, not a division.

These are per-cell coordinates rather than a separate `run` dimension because
zarr cannot append along two dimensions in one call, which would break the
both-stores-or-neither write.

`cell` is deliberately left without a coordinate. Giving it one would make the
index repeat (0 1 2 0 1 2 ...) and `.sel()` ambiguous.

`lat`, `lon`, `run`, `cycle` and `member` are attached but non-index, so
**`.sel()` does not work on them** in the emulator environment: xarray 2024.3
raises `KeyError: no index found for coordinate 'run'`. Newer xarray (2026.2)
builds an index on demand and would accept it, but the environment that trains
the surrogate does not. Use position instead, which works everywhere:

    ds.isel(cell=slice(m * pool_size, (m + 1) * pool_size))   # run m
    ds.where(ds.run == m, drop=True)                          # same, by value

For the cell nearest a point, use `argmin` on the squared distance - even where
`.sel()` is available, `method="nearest"` cannot work here, because `lat`
repeats across runs and within a run.

**Run this with the emulator environment**, not a bare `python`:

    EMU=/home/iferrario/.local/miniforge/envs/emulator/bin/python

A bare `python` here resolves to a different project's env (xarray 2026 /
zarr 3) and writes a zarr **v3** store, which the emulator environment
(xarray 2024.3 / zarr 2.13) cannot open at all. `_write` pins v2, but run it in
the right place anyway - that is where `hython`, `itwinai` and `toml` live.

Run directly to build the archive and check it:

    $EMU pool_archive.py --build            # draw pool, write run 0
    $EMU pool_archive.py --append-test      # run 1 = run 0 with KsatVer x 3
    $EMU pool_archive.py --verify
"""

import argparse
import inspect
import json
import logging
import shutil
import subprocess as sp
import time
from pathlib import Path

import numpy as np
import xarray as xr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

WD_WFLOW = Path("/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1")
WD_SURROGATE = Path("/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_input")

DYNAMIC_ARCHIVE = WD_SURROGATE / "emo1_dynamic_cycle.zarr"
STATIC_ARCHIVE = WD_SURROGATE / "emo1_static_cycle.zarr"

# Full-map sources the pool is cut from.
STATICMAPS = WD_WFLOW / "staticmaps.nc"
FORCING_STORE = WD_SURROGATE / "emo1_dynamic.zarr"

# 5% of the 336,466 usable cells, rounded so it divides evenly into zarr chunks.
# Fixed once: growing the pool later means re-running wflow, cutting it is free.
POOL_SIZE = 16800
CELL_CHUNK = 2100
POOL_SEED = 20250919

# Must match `static_inputs` in config_training_calibration_loop.yaml.
STATIC_VARS = [
    "KsatVer", "c", "f", "RootingDepth", "Sl", "thetaS", "thetaR",
    "SoilThickness", "M", "Kext", "Swood", "wflow_uparea", "wflow_landuse",
    "wflow_dem", "Slope", "WaterFrac",
]
FORCING_VARS = ["precip", "pet", "temp"]
TARGET_VARS = ["vwc"]

# `c` and `vwc` carry a leading layer axis; the surrogate uses layer index 1.
LAYER = 1


# ==== POOL


def draw_pool(seed: int | None = None, size: int | None = None) -> np.ndarray:
    """A plain random draw over the usable cells, returned as (size, 2) indices.

    Not stratified, on purpose: only the pool is fixed, so stratification can
    still happen later where training picks its cells out of it.

    Indices point into the staticmaps grid (latitude descending), which is the
    orientation emo1_dynamic.zarr already uses. Deterministic in `seed`, so the
    archive stores the seed rather than the draw.
    """
    # Resolved here, not in the signature: a default argument binds at
    # definition time, so `POOL_SIZE = n` after import would silently not reach
    # this function while `_describe` picked the new value up.
    seed = POOL_SEED if seed is None else seed
    size = POOL_SIZE if size is None else size

    static = xr.open_dataset(STATICMAPS)
    usable = ~(
        static["thetaS"].isnull().values
        | (static["wflow_lakeareas"].values > 0)
    )
    cells = np.argwhere(usable)
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(cells), size=size, replace=False)
    return cells[np.sort(picked)]


def load_pool() -> np.ndarray:
    """The pool, read back off the archive's own first run."""
    s = xr.open_zarr(STATIC_ARCHIVE)
    n = s.attrs["pool_size"]
    return np.stack([s.lat_i.values[:n], s.lon_i.values[:n]], axis=1)


def _selector(coords: np.ndarray) -> dict:
    """Pointwise (lat, lon) -> cell selection, for use with `.isel()`."""
    return dict(
        lat=xr.DataArray(coords[:, 0], dims="cell"),
        lon=xr.DataArray(coords[:, 1], dims="cell"),
    )


def _describe(ds: xr.Dataset, coords: np.ndarray, prov: dict) -> xr.Dataset:
    """Attach the grid indices and the run's provenance, so no sidecar is needed.

    `prov` names the wflow run these rows came from: {run, cycle, member,
    static, output}. `run` could be recomputed as `i // pool_size`, but `cycle`
    and `member` could not - runs per cycle vary (A2).
    """
    n = ds.sizes["cell"]
    prior = _attrs()
    if prior and (prior["pool_size"], prior["pool_seed"]) != (POOL_SIZE, POOL_SEED):
        raise ValueError(
            f"archive was built with pool_size={prior['pool_size']} "
            f"seed={prior['pool_seed']}, but this module has {POOL_SIZE}/"
            f"{POOL_SEED}. Appending would mix two different pools."
        )
    if n != POOL_SIZE:
        raise ValueError(f"run has {n} rows, expected pool_size={POOL_SIZE}")
    ds = ds.assign_coords(
        lat_i=("cell", coords[:, 0].astype("int32")),
        lon_i=("cell", coords[:, 1].astype("int32")),
        run=("cell", np.full(n, prov["run"], "int32")),
        cycle=("cell", np.full(n, prov["cycle"], "int32")),
        member=("cell", np.full(n, prov["member"], "int32")),
    )
    ds.attrs["pool_seed"] = int(POOL_SEED)
    ds.attrs["pool_size"] = int(POOL_SIZE)
    ds.attrs["runs"] = json.dumps(_manifest() + [prov])
    return ds


def _attrs() -> dict:
    """The existing archive's attrs, or empty if there is no archive yet."""
    if not STATIC_ARCHIVE.exists():
        return {}
    return dict(xr.open_zarr(STATIC_ARCHIVE).attrs)


def _manifest() -> list[dict]:
    """The run manifest already in the archive, or empty if there is none."""
    return json.loads(_attrs().get("runs", "[]"))


def next_run_index() -> int:
    return len(_manifest())


# ==== CUTTING A RUN DOWN TO THE POOL


def _unpack_layer(ds: xr.Dataset, var: str, layer: int = LAYER) -> xr.Dataset:
    if var in ds and "layer" in ds[var].dims:
        ds[var] = ds[var].isel(layer=layer, drop=True)
    return ds.drop_dims("layer", errors="ignore")


def cut_static(static_nc: Path, coords: np.ndarray, prov: dict) -> xr.Dataset:
    """One staticmaps file -> (cell,) archive rows."""
    ds = xr.open_dataset(static_nc)
    ds = ds.rename_dims({"latitude": "lat", "longitude": "lon"})
    try:
        ds = ds.rename_vars({"latitude": "lat", "longitude": "lon"})
    except ValueError:
        pass
    ds = _unpack_layer(ds, "c")
    ds = ds[STATIC_VARS].isel(**_selector(coords)).load()

    # The pool is already masked, so these are all-False. Kept so the existing
    # `mask_variables` config keeps working unchanged.
    zeros = xr.zeros_like(ds["thetaS"], dtype=bool)
    ds["mask_missing"] = zeros
    ds["mask_lake"] = zeros
    return _describe(_tidy(ds), coords, prov)


def cut_dynamic(output_nc: Path, coords: np.ndarray, prov: dict) -> xr.Dataset:
    """One wflow output + the forcing -> (time, cell) archive rows.

    Both go into one dataset because that is what the dataset reads: the
    forcing is identical in every run, but it is repeated rather than looked up,
    to keep `dynamic_inputs` and `target_variables` on the same store.
    """
    target = xr.open_dataset(output_nc, chunks={"time": 256})
    target = target.sel(lat=slice(None, None, -1))  # wflow writes lat ascending
    target = _unpack_layer(target, "vwc")
    target = target[TARGET_VARS].isel(**_selector(coords)).load()

    forcing = xr.open_zarr(FORCING_STORE)[FORCING_VARS]
    forcing = forcing.sel(time=target.time).isel(**_selector(coords)).load()

    # compat/join are explicit and strict on purpose: this merge is the one
    # place the forcing cut and the target cut meet, so a disagreement over
    # lat/lon/time must raise here rather than be silently overridden.
    merged = xr.merge([forcing, target], compat="equals", join="exact")
    return _describe(_tidy(merged), coords, prov)


def _tidy(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.drop_vars(["spatial_ref", "layer"], errors="ignore")
    ds = ds.map(lambda x: x.astype("float32") if x.dtype == "float64" else x)
    ds.attrs.pop("_FillValue", None)
    for v in ds.variables:
        ds[v].encoding.clear()
    return ds


# Every other store in WD_SURROGATE is zarr v2, and the emulator environment
# that trains the surrogate has zarr 2.13, which cannot open a v3 store at all
# (it looks for `.zgroup`; v3 writes `zarr.json`). Newer environments default
# to v3, so the format is pinned here rather than left to whoever runs this.
_ZARR_V2 = {"zarr_format": 2} if "zarr_format" in inspect.signature(
    xr.Dataset.to_zarr
).parameters else {}


def _write(ds: xr.Dataset, path: Path, **mode) -> None:
    """`to_zarr`, pinned to zarr v2 when the installed xarray can express it."""
    ds.to_zarr(path, **mode, **_ZARR_V2)


def _chunked(ds: xr.Dataset) -> xr.Dataset:
    chunks = {"cell": CELL_CHUNK}
    if "time" in ds.dims:
        chunks["time"] = -1
    return ds.chunk(chunks)


# ==== WRITING


def write_run(static: xr.Dataset, dynamic: xr.Dataset, first: bool) -> None:
    """Append one run to both archives, or to neither.

    A partial append permanently offsets the two stores, and because the dataset
    pairs them by position that silently marries theta from one run to vwc from
    another. Stage to temporary stores, promote only once both succeed.
    """
    tmp_static = STATIC_ARCHIVE.with_suffix(".zarr.tmp")
    tmp_dynamic = DYNAMIC_ARCHIVE.with_suffix(".zarr.tmp")
    pairs = ((tmp_static, STATIC_ARCHIVE), (tmp_dynamic, DYNAMIC_ARCHIVE))

    for tmp, final in pairs:
        if tmp.exists():
            shutil.rmtree(tmp)
        if not first:
            shutil.copytree(final, tmp)

    try:
        mode = dict(mode="w") if first else dict(mode="a", append_dim="cell")
        _write(_chunked(static), tmp_static, **mode)
        _write(_chunked(dynamic), tmp_dynamic, **mode)
    except Exception:
        for tmp, _ in pairs:
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    for tmp, final in pairs:
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)


def ingest_run(
    static_nc: Path, output_nc: Path, first: bool, cycle: int = 0, member: int = 0
) -> None:
    """Cut one wflow run down to the pool and add it to the end of the archive."""
    t0 = time.perf_counter()
    coords = draw_pool() if first else load_pool()
    prov = dict(
        run=0 if first else next_run_index(), cycle=cycle, member=member,
        static=static_nc.name, output=output_nc.name,
    )

    static = cut_static(static_nc, coords, prov)
    t_static = time.perf_counter() - t0
    dynamic = cut_dynamic(output_nc, coords, prov)
    t_dynamic = time.perf_counter() - t0 - t_static

    t1 = time.perf_counter()
    write_run(static, dynamic, first=first)
    logger.info(
        f"ingest: static {t_static:.1f}s, dynamic {t_dynamic:.1f}s, "
        f"write {time.perf_counter() - t1:.1f}s, total {time.perf_counter() - t0:.1f}s"
    )


def du(path: Path) -> str:
    return sp.run(["du", "-sh", str(path)], capture_output=True, text=True).stdout.split()[0]


# ==== BACK TO XARRAY


# These moved into `hython.utils` so the training and inference paths can use
# them without importing this script. `scatter_pool_to_map` takes the grid as
# an argument there - it cannot carry STATICMAPS, which is a wflow path - so
# the wrapper below supplies it and this module's surface is unchanged.

from hython.utils import pool_to_xarray as to_xarray, scatter_pool_to_map  # noqa: E402


def scatter_to_map(ds: xr.Dataset, crs: int | None = 4326) -> xr.Dataset:
    """`hython.utils.scatter_pool_to_map` against this model's staticmaps."""
    return scatter_pool_to_map(ds, xr.open_dataset(STATICMAPS), crs)


# ==== CHECKS


def verify() -> None:
    """Every check A1 owes: shape, provenance, coordinate agreement, repetition."""
    s = xr.open_zarr(STATIC_ARCHIVE)
    d = xr.open_zarr(DYNAMIC_ARCHIVE)
    grid = xr.open_dataset(STATICMAPS)

    # the stores describe themselves, and agree about what they describe
    for key in ("pool_seed", "pool_size"):
        assert s.attrs[key] == d.attrs[key], f"{key} differs between stores"
    pool_size = s.attrs["pool_size"]

    n = s.sizes["cell"]
    assert d.sizes["cell"] == n, f"static {n} vs dynamic {d.sizes['cell']}"
    assert n % pool_size == 0, f"{n} rows is not a whole number of runs"
    runs = n // pool_size
    logger.info(
        f"rows={n}  runs={runs}  pool_size={pool_size}  pool_seed={s.attrs['pool_seed']}"
    )

    # The run layout is asserted first, because it is the check that localises
    # a bad append. `n % pool_size == 0` does NOT prove every run is whole: a
    # 16800-row run followed by two 300-row runs also divides by 300.
    manifest = json.loads(s.attrs["runs"])
    assert len(manifest) == runs, (
        f"manifest lists {len(manifest)} runs but the archive holds {runs} - "
        "some run was written with a different pool_size"
    )
    expected = np.repeat(np.arange(runs, dtype="int32"), pool_size)
    for name, ds in (("static", s), ("dynamic", d)):
        np.testing.assert_array_equal(ds.run.values, expected, err_msg=f"{name} run")
        for key in ("cycle", "member"):
            np.testing.assert_array_equal(
                ds[key].values,
                np.repeat([r[key] for r in manifest], pool_size).astype("int32"),
                err_msg=f"{name} {key}",
            )
    np.testing.assert_array_equal(s.run.values, d.run.values)
    for r in manifest:
        logger.info(
            f"  run {r['run']}: cycle {r['cycle']} member {r['member']}"
            f"  <- {r['static']} + {r['output']}"
        )
    logger.info(f"run layout asserted: {runs} x {pool_size} contiguous rows")

    # forcing and target must share one store, or the dataset cannot read them
    for v in FORCING_VARS + TARGET_VARS:
        assert v in d, f"{v} missing from the dynamic store"
    logger.info(f"dynamic store holds {FORCING_VARS + TARGET_VARS}")

    # the pool is reproducible from the seed it records - this is what lets the
    # sidecar go away
    pool = np.stack([s.lat_i.values[:pool_size], s.lon_i.values[:pool_size]], axis=1)
    np.testing.assert_array_equal(
        pool, draw_pool(s.attrs["pool_seed"], pool_size),
        err_msg="archive rows do not match a redraw from the recorded seed",
    )
    logger.info("pool reproducible: redrawing from the stored seed gives the same cells")

    # grid indices and float coords must describe the same cell
    tiled = np.tile(pool, (runs, 1))
    for name, ds in (("static", s), ("dynamic", d)):
        np.testing.assert_array_equal(
            np.stack([ds.lat_i.values, ds.lon_i.values], axis=1), tiled,
            err_msg=f"{name} grid indices",
        )
        np.testing.assert_array_equal(
            ds.lat.values, grid.latitude.values[tiled[:, 0]], err_msg=f"{name} lat"
        )
        np.testing.assert_array_equal(
            ds.lon.values, grid.longitude.values[tiled[:, 1]], err_msg=f"{name} lon"
        )
    np.testing.assert_array_equal(s.lat.values, d.lat.values)
    np.testing.assert_array_equal(s.lon.values, d.lon.values)
    logger.info("coordinates agree: static and dynamic sit on the same cells of the map")

    # the forcing must be identical in every run, or `i % pool_size` would have
    # been the better layout after all
    if runs > 1:
        base = d["precip"].isel(cell=slice(0, pool_size)).values
        for m in range(1, runs):
            other = d["precip"].isel(cell=slice(m * pool_size, (m + 1) * pool_size)).values
            np.testing.assert_array_equal(base, other, err_msg=f"forcing differs in run {m}")
        logger.info(f"forcing identical across all {runs} runs")

    # `cell` must stay a plain position, or .sel() becomes ambiguous
    for name, ds in (("static", s), ("dynamic", d)):
        assert "cell" not in ds.coords, f"{name} has a cell coordinate; drop it"
        assert "cell" not in ds.indexes, f"{name} indexes cell; drop it"

    for name, path in (("static", STATIC_ARCHIVE), ("dynamic", DYNAMIC_ARCHIVE)):
        logger.info(f"{name}: {du(path)}  ({runs} run(s))")


def verify_append(factor: float) -> None:
    """Run 1 is run 0 with KsatVer scaled - the check H1 will reuse."""
    s = xr.open_zarr(STATIC_ARCHIVE)
    pool_size = s.attrs["pool_size"]
    assert s.sizes["cell"] == 2 * pool_size, "expected exactly two runs"
    a = s["KsatVer"].isel(cell=slice(0, pool_size)).values
    b = s["KsatVer"].isel(cell=slice(pool_size, None)).values
    np.testing.assert_allclose(b, a * factor, rtol=1e-5)
    # everything else must be untouched
    for v in ("thetaS", "wflow_dem", "Slope"):
        np.testing.assert_array_equal(
            s[v].isel(cell=slice(0, pool_size)).values,
            s[v].isel(cell=slice(pool_size, None)).values,
        )
    logger.info(f"run 1 = run 0 with KsatVer x {factor}, all other statics identical")


# ==== DRIVER


def build(overwrite: bool = False) -> None:
    existing = [p for p in (DYNAMIC_ARCHIVE, STATIC_ARCHIVE) if p.exists()]
    if existing and not overwrite:
        raise SystemExit(
            "refusing to overwrite:\n  " + "\n  ".join(map(str, existing))
            + "\nrerun with --overwrite if that is what you want"
        )
    for p in existing:
        shutil.rmtree(p)

    coords = draw_pool()
    logger.info(f"pool: {len(coords)} cells, seed {POOL_SEED}")

    output_nc = WD_WFLOW / "run_default" / "output_calib_0.nc"
    logger.info(f"run 0 from {STATICMAPS.name} + {output_nc.name}")

    prov = dict(run=0, cycle=0, member=0,
                static=STATICMAPS.name, output=output_nc.name)

    t0 = time.perf_counter()
    static = cut_static(STATICMAPS, coords, prov)
    dynamic = cut_dynamic(output_nc, coords, prov)
    logger.info(f"cut in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    write_run(static, dynamic, first=True)
    logger.info(f"run 0 written in {time.perf_counter() - t0:.1f}s")
    for name, path in (("static", STATIC_ARCHIVE), ("dynamic", DYNAMIC_ARCHIVE)):
        logger.info(f"{name}: {du(path)}")


def append_test(factor: float = 3.0) -> None:
    """Append a synthetic run 1 to time the append and prove it lines up."""
    coords = load_pool()
    output_nc = WD_WFLOW / "run_default" / "output_calib_0.nc"

    prov = dict(run=next_run_index(), cycle=0, member=1,
                static=f"{STATICMAPS.name} (KsatVer x {factor})",
                output=output_nc.name)

    t0 = time.perf_counter()
    static = cut_static(STATICMAPS, coords, prov)
    static["KsatVer"] = static["KsatVer"] * factor
    dynamic = cut_dynamic(output_nc, coords, prov)
    logger.info(f"cut in {time.perf_counter() - t0:.1f}s")

    before = {p: du(p) for p in (STATIC_ARCHIVE, DYNAMIC_ARCHIVE)}
    t0 = time.perf_counter()
    write_run(static, dynamic, first=False)
    logger.info(f"append (both stores, staged) in {time.perf_counter() - t0:.1f}s")
    for p, was in before.items():
        logger.info(f"{p.name}: {was} -> {du(p)}")

    verify_append(factor)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--append-test", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.build:
        build(overwrite=args.overwrite)
    if args.append_test:
        append_test()
    if args.verify:
        verify()
