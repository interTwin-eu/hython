"""Inputs of the article C6.1 analysis (transfer error) for one dPL cycle.

The first submission analysed the article calibration (thetalai, seed 10532).
This script makes the same files for the calibration of dPL cycle `c`, so the
C6.1 notebook can read them in place of the old ones:

    surrogate_2022_apriori_c<c>.nc   replaces  article/surrogate_2022.nc
    surrogate_2022_cal_c<c>.nc       (the calibrated prediction, new)
    kge_surr_rt0_dpl_c<c>.nc         replaces  temp/kge_surr_rt0_thetalai_10532.nc
    kge_wflow_rt0_dpl_c<c>.nc        replaces  temp/kge_wflow_rt0_thetalai_10532.nc
    c62_response_fidelity_dpl_c<c>.nc replaces temp/c62_response_fidelity.nc

The methods are those of the old files, not those of the dPL scores:

- Surrogate: CudaLSTM_<c>.pt on the full map, each cell run as one sequence
  that starts `seq_length` days (from the cycle config) before 2022 and ends
  on 2022-12-31; only 2022 is kept. The model was trained on the last day of
  `seq_length`-day windows (predict_steps 0), so a sequence that starts cold
  on 2022-01-01 gives days the model never learned (user, 2026-10-06). The
  surrogate_2022.nc of the first submission had no warm-up. Inputs are scaled
  with the cached statistics of the dPL training run.
- Static inputs: taken from the staticmaps file that wflow ran with, `c` at
  soil layer 0. This is how the dPL archive builds them (pool_archive.py), so
  the surrogate sees the same values as in training. emo1_static.zarr holds
  `c` of layer 1 and is used only for the masks.
- KGE against RT0: standard KGE (weights 1, 1, 1), 2022, RT0 without the
  static mask (the notebook applies it later). On the old wflow run this
  code gives the old file again (r to 1e-15, KGE to 1e-5).
- Response fidelity, per cell over 2022: OLS slope (with intercept) and
  Pearson r of dS = surr(cal) - surr(apriori) on dW = wflow(cal) -
  wflow(apriori). dS outside (-0.3, 0.3) is NaN, and a cell with any NaN
  gets NaN, as in REV2_C6_2. A cell where dW does not change gets NaN (the
  old lstsq gave a meaningless number there).
- A priori wflow: emo1_dynamic.zarr (user, 2026-10-05).

Each step is skipped when its output exists; --force redoes it.

Two campaigns (see CAMPAIGNS): `c1` is the production dPL campaign, `c1ap`
the campaign with the a priori run in the cycle-0 archive
(run_apriori_campaign.py). Each one reads only its own cycles, surrogate and
wflow outputs, and writes to its own folder with its own file tag
(`_c1` or `_c1ap`). The a priori staticmaps and wflow run are the same for both.

Run with the `emulator` interpreter, from anywhere:

    python scripts/c6_transfer_inputs.py --cycle 1                     # c1
    python scripts/c6_transfer_inputs.py --cycle 1 --campaign c1ap     # c1ap
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pool_archive import LAYER, _unpack_layer  # noqa: E402

from hython.datasets import WflowSBM  # noqa: E402
from hython.evaluator import predict  # noqa: E402
from hython.models.cudnnLSTM import CudaLSTM  # noqa: E402
from hython.sampler import SamplerBuilder  # noqa: E402
from hython.scaler import Scaler  # noqa: E402
from hython.utils import create_xarray_data  # noqa: E402

EMO1 = Path("/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1")
APRIORI = Path("/mnt/CEPH_PROJECTS/A_DROP/pilot5/article/dpl_apriori")
# Per campaign: cycle configs, surrogate weights, wflow dir (calibrated
# staticmaps + run_default outputs), output dir, and the tag the file names get
# in place of the cycle number ("{c}" is filled with the cycle).
CAMPAIGNS = {
    "c1": dict(cyc=Path(__file__).resolve().parent / "outputs" / "dpl_cycles",
               seq=Path("/mnt/CEPH_PROJECTS/InterTwin/hython_model_run/loop_train_multicycle/model_sequence"),
               wflow=EMO1,
               out=Path("/mnt/CEPH_PROJECTS/A_DROP/pilot5/article/dpl_c1"),
               tag="c{c}"),
    "c1ap": dict(cyc=APRIORI / "cycles",
                 seq=APRIORI / "runs" / "loop_train_multicycle" / "model_sequence",
                 wflow=APRIORI / "wflow",
                 out=Path("/mnt/CEPH_PROJECTS/A_DROP/pilot5/article/dpl_c1ap"),
                 tag="c{c}ap"),
}
SURR_IN = Path("/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_input")
STATIC_ZARR = SURR_IN / "emo1_static.zarr"
DYNAMIC_ZARR = SURR_IN / "emo1_dynamic.zarr"
RT0 = Path("/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/SSM-RT0-SIG0-R-CRRL/processed"
           "/alps_rt0old_2017-2022_theta.nc")
YEAR = ("2022-01-01", "2022-12-31")
DELTA_CLIP = 0.3   # |dS| at or above this is NaN, as in REV2_C6_2
LAT_BLOCK = 71     # rows per block in the per-cell statistics (568 = 8 x 71)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- static inputs ----------------------------------------------------------

def static_from_staticmaps(nc: Path, cfg) -> xr.Dataset:
    """Static inputs as the dPL archive builds them, plus the masks."""
    ds = xr.open_dataset(nc)
    ds = ds.rename_dims({"latitude": "lat", "longitude": "lon"})
    try:
        ds = ds.rename_vars({"latitude": "lat", "longitude": "lon"})
    except ValueError:
        pass
    ds = _unpack_layer(ds, "c", LAYER)[list(cfg.static_inputs)].load()

    masks = xr.open_dataset(STATIC_ZARR, engine="zarr")[list(cfg.mask_variables)].load()
    assert np.allclose(ds.lat, masks.lat) and np.allclose(ds.lon, masks.lon), \
        f"{nc.name}: grid differs from {STATIC_ZARR.name}"
    ds = ds.assign_coords(lat=masks.lat, lon=masks.lon)
    out = xr.merge([ds, masks])
    for v in out.variables:
        out[v].encoding = {}
    return out


# ---- surrogate --------------------------------------------------------------

def surrogate_cfg(cyc: Path, cycle: int, static_zarr: Path):
    """Cycle `cycle` training config, pointed at the full map for 2022 with
    `seq_length` days of warm-up before it."""
    cfg = OmegaConf.load(cyc / f"cycle_{cycle}" / f"config_training_calibration_loop_c{cycle}.yaml")
    for key in ("training_pipeline", "evaluator", "model_logger",
                "loss_fn", "metric_fn", "regularization"):
        cfg.pop(key, None)
    cfg.dataset = "WflowSBM"
    cfg.data_source.file.static_inputs = str(static_zarr)
    cfg.data_source.file.dynamic_inputs = str(DYNAMIC_ZARR)
    cfg.data_source.file.target_variables = str(DYNAMIC_ZARR)
    warm = np.datetime64(YEAR[0]) - np.timedelta64(int(cfg.seq_length), "D")
    cfg.test_temporal_range = [str(warm), YEAR[1]]
    cfg.test_downsampler = None
    # WflowSBM makes one full-length sample per cell only when this is set;
    # the "sequential" sampler below does not draw from it.
    assert cfg.dynamic_downsampler is not None
    return instantiate(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))


def predict_surrogate(camp: dict, cycle: int, static_zarr: Path, dest: Path, device):
    cfg = surrogate_cfg(camp["cyc"], cycle, static_zarr)
    scaler = Scaler(cfg, use_cached=True)   # the dPL training statistics
    ds = WflowSBM(cfg, scaler, False, "test")
    log(f"{static_zarr.name}: {len(ds)} cells x {ds.xd.sizes['time']} days "
        f"(from {cfg.test_temporal_range[0]}, warm-up {cfg.seq_length} days)")
    sampler = SamplerBuilder(cfg, ds, sampling="sequential", processing="single-gpu").get_sampler()
    loader = DataLoader(ds, batch_size=cfg.batch * 4, sampler=sampler,
                        num_workers=cfg.num_workers_dataloader_eval)

    model = CudaLSTM(cfg).to(device)
    model.load_state_dict(torch.load(camp["seq"] / f"CudaLSTM_{cycle}.pt", map_location=device,
                                     weights_only=False))
    pred = predict(ds, loader, model, device)

    coords = xr.Coordinates({"lat": ds.y.lat, "lon": ds.y.lon, "time": ds.y.time,
                             "variable": list(cfg.target_variables)})
    shape = {"lat": ds.y.sizes["lat"], "lon": ds.y.sizes["lon"],
             "time": ds.y.sizes["time"], "variable": len(cfg.target_variables)}
    out = create_xarray_data(pred, coords, output_shape=shape).sel(time=slice(*YEAR))
    assert out.sizes["time"] == 365, out.sizes
    out.attrs.update(surrogate=str(camp["seq"] / f"CudaLSTM_{cycle}.pt"), static_inputs=str(static_zarr),
                     made_by=Path(__file__).name,
                     warmup_start=cfg.test_temporal_range[0], warmup_days=int(cfg.seq_length))
    out.to_netcdf(dest, encoding={v: {"zlib": True, "complevel": 1} for v in out.data_vars})


# ---- per-cell statistics ----------------------------------------------------

def kge_blocks(obs: np.ndarray, sim: np.ndarray):
    """Standard KGE and its parts per cell; (time, lat, lon) in, float64.

    Pairs with a NaN on either side are dropped, as keep_valid did. float64
    matters: in float32 the deviations of a near-constant series lose most of
    their digits and r comes out wrong.
    """
    out = {k: np.full(obs.shape[1:], np.nan) for k in ("kge", "r", "alpha", "beta")}
    for i in range(0, obs.shape[1], LAT_BLOCK):
        o = obs[:, i:i + LAT_BLOCK].astype("float64")
        s = sim[:, i:i + LAT_BLOCK].astype("float64")
        ok = np.isfinite(o) & np.isfinite(s)
        o[~ok] = np.nan
        s[~ok] = np.nan
        n = ok.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            do = o - np.nanmean(o, 0)
            ds = s - np.nanmean(s, 0)
            so = np.sqrt(np.nansum(do ** 2, 0) / (n - 1))
            ss = np.sqrt(np.nansum(ds ** 2, 0) / (n - 1))
            r = np.nansum(do * ds, 0) / ((n - 1) * so * ss)
            alpha = ss / so
            beta = np.nanmean(s, 0) / np.nanmean(o, 0)
            kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        bad = n < 2
        for k, v in (("kge", kge), ("r", r), ("alpha", alpha), ("beta", beta)):
            out[k][i:i + LAT_BLOCK] = np.where(bad, np.nan, v)
    return out


def fidelity_blocks(dw: np.ndarray, ds: np.ndarray):
    """OLS slope (with intercept) of ds on dw, and Pearson r, per cell."""
    slope = np.full(dw.shape[1:], np.nan)
    r = np.full(dw.shape[1:], np.nan)
    for i in range(0, dw.shape[1], LAT_BLOCK):
        x = dw[:, i:i + LAT_BLOCK].astype("float64")
        y = ds[:, i:i + LAT_BLOCK].astype("float64")
        ok = np.isfinite(x).all(0) & np.isfinite(y).all(0)   # any NaN -> NaN
        dx = x - x.mean(0)
        dy = y - y.mean(0)
        sxx = (dx ** 2).sum(0)
        syy = (dy ** 2).sum(0)
        sxy = (dx * dy).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            slope[i:i + LAT_BLOCK] = np.where(ok & (sxx > 0), sxy / sxx, np.nan)
            r[i:i + LAT_BLOCK] = np.where(ok & (sxx > 0) & (syy > 0),
                                          sxy / np.sqrt(sxx * syy), np.nan)
    return slope, r


def grid_like(ref: xr.DataArray, name_map: dict, layer) -> xr.Dataset:
    """Per-cell maps on the RT0 grid, with the scalar coordinates of the old
    files (`layer`, `spatial_ref`) so the notebook's .drop() calls still work."""
    ds = xr.Dataset({k: (("lat", "lon"), v) for k, v in name_map.items()},
                    coords={"lat": ref.lat, "lon": ref.lon})
    return ds.assign_coords(layer=layer, spatial_ref=0)


def on_rt0_grid(da: xr.DataArray, rt: xr.DataArray) -> np.ndarray:
    """(time, lat, lon) values of `da` on the RT0 grid and days."""
    da = da.sel(time=rt.time).sortby("lat", ascending=bool(rt.lat[0] < rt.lat[-1]))
    assert np.allclose(da.lat, rt.lat) and np.allclose(da.lon, rt.lon), "grid differs from RT0"
    return da.transpose("time", "lat", "lon").values


# ---- main -------------------------------------------------------------------

def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--cycle", type=int, default=1)
    a.add_argument("--campaign", choices=sorted(CAMPAIGNS), default="c1")
    a.add_argument("--out", type=Path, default=None, help="default: the campaign's folder")
    a.add_argument("--force", action="store_true", help="redo steps whose output exists")
    a = a.parse_args()
    c = a.cycle
    camp = CAMPAIGNS[a.campaign]
    a.out = a.out or camp["out"]
    t = camp["tag"].format(c=c)
    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"campaign {a.campaign}, cycle {c}, tag {t}, out {a.out}, device {device}")
    log(f"  cycles {camp['cyc']}, surrogate {camp['seq']}, wflow {camp['wflow']}")

    def todo(p: Path) -> bool:
        if p.exists() and not a.force:
            log(f"exists, skipped: {p.name}")
            return False
        return True

    cfg0 = surrogate_cfg(camp["cyc"], c, STATIC_ZARR)
    st_ap, st_cal = a.out / f"static_apriori_{t}.zarr", a.out / f"static_cal_{t}.zarr"
    for dest, nc in ((st_ap, EMO1 / "staticmaps.nc"),
                     (st_cal, camp["wflow"] / f"staticmaps_cycle{c}_cal.nc")):
        if todo(dest):
            static_from_staticmaps(nc, cfg0).to_zarr(dest, mode="w", consolidated=True)
            log(f"wrote {dest.name} from {nc.name}")

    surr_ap, surr_cal = a.out / f"surrogate_2022_apriori_{t}.nc", a.out / f"surrogate_2022_cal_{t}.nc"
    for dest, st in ((surr_ap, st_ap), (surr_cal, st_cal)):
        if todo(dest):
            predict_surrogate(camp, c, st, dest, device)
            log(f"wrote {dest.name}")

    rt = xr.open_dataset(RT0).ssm.sel(time=slice(*YEAR)).load()   # not masked, as the old files
    s_cal = xr.open_dataset(surr_cal).vwc
    w_cal = _unpack_layer(xr.open_dataset(camp["wflow"] / "run_default" / f"output_cycle{c}_cal.nc"), "vwc").vwc

    dest = a.out / f"kge_surr_rt0_dpl_{t}.nc"
    if todo(dest):
        k = kge_blocks(rt.values, on_rt0_grid(s_cal, rt))
        grid_like(rt, k, layer=1).to_netcdf(dest)
        log(f"wrote {dest.name}: median KGE {np.nanmedian(k['kge']):.3f}")

    dest = a.out / f"kge_wflow_rt0_dpl_{t}.nc"
    if todo(dest):
        k = kge_blocks(rt.values, on_rt0_grid(w_cal, rt))
        grid_like(rt, k, layer=1).to_netcdf(dest)
        log(f"wrote {dest.name}: median KGE {np.nanmedian(k['kge']):.3f}")

    dest = a.out / f"c62_response_fidelity_dpl_{t}.nc"
    if todo(dest):
        s_ap = xr.open_dataset(surr_ap).vwc
        w_ap = xr.open_dataset(DYNAMIC_ZARR, engine="zarr").vwc
        ds_ = on_rt0_grid(s_cal, rt) - on_rt0_grid(s_ap, rt)
        ds_[~((ds_ > -DELTA_CLIP) & (ds_ < DELTA_CLIP))] = np.nan
        dw = on_rt0_grid(w_cal, rt) - on_rt0_grid(w_ap, rt)
        slope, r = fidelity_blocks(dw, ds_)
        grid_like(rt, {"slope": slope, "r": r}, layer=1.0).to_netcdf(dest)
        log(f"wrote {dest.name}: median slope {np.nanmedian(slope):.3f}, r {np.nanmedian(r):.3f}")

    log("done")


if __name__ == "__main__":
    main()
