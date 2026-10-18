"""Test: does adding the a priori run to the archive fix the Po Plain failure?

The cycle-1 surrogate reproduces wflow_sbm well on its 5 training runs, also in
the Po Plain (median r 0.85), but fails there at the a priori parameters
(r 0.18). Clamping KsatVer and RootingDepth into the training-run range brings
r back to 0.75: the a priori point lies at the lower edge of the training data
(the dPL perturbation pushes both parameters up) and the surrogate extrapolates.

This script retrains the cycle-1 surrogate with one change - the a priori run
added to its training runs - and scores both surrogates on the same cells.

    1. archive  runs 0-4 of the dPL archive (the cycle-1 training set) plus the
                a priori run (staticmaps.nc + output_baseline_apriori.nc) as
                run 5, in new stores
    2. train    the cycle-1 training config with `runs: 6`; rows scaled to
                26,400 / 7,200 so each run gets the same places per epoch as
                in cycle 1; warm start from CudaLSTM_0.pt, as cycle 1
    3. score    median r and std ratio (alpha) of the old and the new surrogate
                against wflow_sbm, at the a priori parameters, on the 5
                training runs, and with the a priori parameters clamped into
                the training-run range; 2019 and 2022, one sequence per cell
                from a cold state (as the a priori map in the C6.1 notebook)

Everything is written under ROOT. The production archive, the dPL run folders,
the model weights and the scaling statistics are only read; `check_outputs`
stops the script before training if any output path is outside ROOT. Each step
is skipped when its output exists.

Run with the `emulator` interpreter:

    python scripts/apriori_in_archive_test.py            # all steps
    python scripts/apriori_in_archive_test.py --step score
"""
import argparse
import json
import os
import shutil
import subprocess as sp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pool_archive as pa  # noqa: E402

ROOT = Path("/mnt/CEPH_PROJECTS/A_DROP/pilot5/article/apriori_test")
ARCH = ROOT / "archive"
STATIC_TEST, DYNAMIC_TEST = ARCH / "static.zarr", ARCH / "dynamic.zarr"
WORK = ROOT / "run"                                   # work_dir of the test training
EXP_NAME, EXP_RUN = "loop", "train_c1_apriori_test"
RUN_DIR = WORK / f"{EXP_NAME}_{EXP_RUN}"              # scaler cache + weights (hython's run folder)
RES = ROOT / "results"

CYC = Path(__file__).resolve().parent / "outputs" / "dpl_cycles"
CFG_C1 = CYC / "cycle_1" / "config_training_calibration_loop_c1.yaml"
PROD_RUN = Path("/mnt/CEPH_PROJECTS/InterTwin/hython_model_run/loop_train_multicycle")
SEQ = PROD_RUN / "model_sequence"
BASELINE = pa.WD_WFLOW / "run_default" / "output_baseline_apriori.nc"
SURR_AP_C1 = Path("/mnt/CEPH_PROJECTS/A_DROP/pilot5/article/dpl_c1/surrogate_2022_apriori_c1.nc")
ENV_BIN = Path("/mnt/CEPH_PROJECTS/A_DROP/pilot5/dev/conda/emulator/bin")

RUNS_C1 = 5                                           # cycle 1 trained on runs 0-4
ROWS_TRAIN, ROWS_VALID = 26400, 7200                  # 22,000 / 6,000 x 6/5
PARAMS = ["KsatVer", "c", "f", "RootingDepth", "Sl"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def under_root(p) -> bool:
    return Path(p).resolve().is_relative_to(ROOT.resolve())


# ---- 1. archive -------------------------------------------------------------

def build_archive():
    if STATIC_TEST.exists() and DYNAMIC_TEST.exists():
        log("archive exists, skipped")
        return
    for p in (STATIC_TEST, DYNAMIC_TEST):
        assert under_root(p) and not p.exists(), f"refusing to write {p}"
    ARCH.mkdir(parents=True, exist_ok=True)
    n = pa.POOL_SIZE

    # runs 0-4, read from the production archive
    st = xr.open_zarr(pa.STATIC_ARCHIVE).isel(cell=slice(0, RUNS_C1 * n))
    dy = xr.open_zarr(pa.DYNAMIC_ARCHIVE).isel(cell=slice(0, RUNS_C1 * n))
    manifest = json.loads(st.attrs["runs"])[:RUNS_C1]
    assert [m["run"] for m in manifest] == list(range(RUNS_C1))
    assert np.array_equal(np.unique(st.run.values), np.arange(RUNS_C1))

    # the a priori run, cut like every other run (cut_* only read the production archive)
    coords = pa.load_pool()
    prov = dict(run=RUNS_C1, cycle=-1, member=0, static=pa.STATICMAPS.name, output=BASELINE.name)
    st_ap = pa.cut_static(pa.STATICMAPS, coords, prov)
    log("a priori static cut")
    dy_ap = pa.cut_dynamic(BASELINE, coords, prov)
    log("a priori dynamic cut")
    assert np.array_equal(dy_ap.time.values, dy.time.values), "time axis differs from the archive"

    attrs = dict(pool_seed=int(pa.POOL_SEED), pool_size=int(n), runs=json.dumps(manifest + [prov]))
    for ds_old, ds_new, dest in ((st, st_ap, STATIC_TEST), (dy, dy_ap, DYNAMIC_TEST)):
        ds_old = pa._tidy(ds_old.copy())
        ds_new = ds_new[list(ds_old.data_vars)]
        ds_old.attrs, ds_new.attrs = dict(attrs), dict(attrs)
        pa._write(pa._chunked(ds_old), dest, mode="w")
        pa._write(pa._chunked(ds_new), dest, mode="a", append_dim="cell")
        log(f"wrote {dest}")

    s = xr.open_zarr(STATIC_TEST)
    assert s.sizes["cell"] == (RUNS_C1 + 1) * n
    assert np.array_equal(s.run.values, np.repeat(np.arange(RUNS_C1 + 1), n))
    assert np.array_equal(s.lat_i.values, np.tile(s.lat_i.values[:n], RUNS_C1 + 1))
    log("archive checked: 6 runs, contiguous, same pool in every run")


# ---- 2. train ---------------------------------------------------------------

def test_config():
    """Cycle-1 config with the test archive, 6 runs and every output under ROOT."""
    cfg = OmegaConf.load(CFG_C1)
    cfg.work_dir = str(WORK)
    cfg.experiment_run = EXP_RUN
    for k in ("static_inputs", "dynamic_inputs", "target_variables"):
        cfg.data_source.file[k] = str(STATIC_TEST if k == "static_inputs" else DYNAMIC_TEST)
    cfg.train_downsampler.runs = RUNS_C1 + 1
    cfg.valid_downsampler.runs = RUNS_C1 + 1
    cfg.train_rows_target, cfg.valid_rows_target = ROWS_TRAIN, ROWS_VALID
    cfg.model_logger.CudaLSTM.model_uri = str(RUN_DIR / "CudaLSTM.pt")
    assert cfg.model_load_pretrained is True, "cycle 1 was warm-started; keep it the same"
    mlflow = [lg for lg in cfg.training_pipeline.steps[1].logger.loggers
              if lg._target_.endswith("MLFlowLogger")]
    assert len(mlflow) == 1
    mlflow[0].savedir = str(RES / "mllogs")
    return cfg


def check_outputs(cfg):
    """Stop unless every place the training can write is under ROOT."""
    r = OmegaConf.to_container(cfg, resolve=True)
    writes = {
        "work_dir": r["work_dir"],
        "run folder (scaler cache)": str(Path(r["work_dir"]) / f"{r['experiment_name']}_{r['experiment_run']}"),
        "model weights": r["model_logger"]["CudaLSTM"]["model_uri"],
        "mlflow": [lg["savedir"] for lg in r["training_pipeline"]["steps"][1]["logger"]["loggers"]
                   if "savedir" in lg][0],
        "evaluator": r["evaluator"]["dir_out"],
        "working folder (checkpoints, log)": str(RES),
    }
    bad = {k: v for k, v in writes.items() if not under_root(v)}
    assert not bad, f"outputs outside {ROOT}: {bad}"
    text = OmegaConf.to_yaml(cfg, resolve=True)
    assert "loop_train_multicycle" not in text and "scripts/config/mllogs" not in text, \
        "config still points at a production folder"
    for k, v in writes.items():
        log(f"  writes {k}: {v}")


def train():
    weights = RUN_DIR / "CudaLSTM.pt"
    done = RES / "train.done"
    if done.exists():
        log("training done before, skipped")
        return
    cfg = test_config()
    check_outputs(cfg)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    # warm start and scaler cache: copies, so the production files are only read
    shutil.copy2(SEQ / "CudaLSTM_0.pt", weights)
    for f in ("static_inputs.yaml", "dynamic_inputs.yaml"):
        shutil.copy2(PROD_RUN / f, RUN_DIR / f)
    name = "config_training_c1_apriori_test"
    OmegaConf.save(cfg, RES / f"{name}.yaml")
    log(f"training: config {RES / name}.yaml, log {RES / name}.log")
    env = dict(os.environ, PATH=f"{ENV_BIN}:{os.environ['PATH']}")
    with open(RES / f"{name}.log", "w") as fh:     # same call as run_dpl_cycle.exec_pipeline
        proc = sp.run(f"itwinai exec-pipeline --config-dir {RES} --config-name {name}",
                      shell=True, cwd=RES, env=env, stdout=fh, stderr=sp.STDOUT, text=True)
    if proc.returncode != 0:
        raise sp.CalledProcessError(proc.returncode, "itwinai exec-pipeline")
    done.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    log(f"training done, weights {weights}")


# ---- 3. score ---------------------------------------------------------------

def score():
    import torch
    from hydra.utils import instantiate
    from hython.models.cudnnLSTM import CudaLSTM
    from hython.scaler import Scaler

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    raw = OmegaConf.load(CFG_C1)
    for k in ("training_pipeline", "evaluator", "model_logger", "loss_fn", "metric_fn", "regularization"):
        raw.pop(k, None)
    raw.work_dir, raw.experiment_run = str(WORK), EXP_RUN     # scaler reads the copied cache
    cfg = instantiate(OmegaConf.create(OmegaConf.to_container(raw, resolve=True)))
    scaler = Scaler(cfg, use_cached=True)
    for t in ("static_inputs", "dynamic_inputs"):
        scaler.load_or_compute(None, t, is_train=False)
    s_vars, d_vars = list(cfg.static_inputs), list(cfg.dynamic_inputs)

    models = {"cycle 1": SEQ / "CudaLSTM_1.pt", "cycle 1 + a priori run": RUN_DIR / "CudaLSTM.pt"}
    nets = {}
    for k, p in models.items():
        m = CudaLSTM(cfg).to(dev)
        m.load_state_dict(torch.load(p, map_location=dev, weights_only=False))
        nets[k] = m.eval()

    st = xr.open_zarr(STATIC_TEST)
    dy = xr.open_zarr(DYNAMIC_TEST)
    n = int(st.attrs["pool_size"])
    ap_rows = np.arange(RUNS_C1 * n, (RUNS_C1 + 1) * n)          # the a priori run

    # cell groups at the a priori point, from the cycle-1 surrogate (as notebook cell 66)
    w22 = dy.vwc.sel(time=slice("2022-01-01", "2022-12-31")).isel(cell=ap_rows).values.astype("float64")
    s22 = xr.open_dataset(SURR_AP_C1).vwc.transpose("time", "lat", "lon").values
    s22 = s22[:, st.lat_i.values[:n], st.lon_i.values[:n]].astype("float64")
    r0 = corr(w22, s22)
    frozen = (np.abs(np.diff(w22, axis=0)) < 1e-3).mean(0)
    low = st.wflow_dem.values[:n] <= 200
    group = np.where(r0 > 0.85, np.where(low, "good, lowland", "good, higher"),
                     np.where((r0 < 0.5) & (frozen <= 0.3), "failure", "other"))
    rng = np.random.default_rng(0)
    hi = np.where(group == "good, higher")[0]
    cells = np.sort(np.concatenate([np.where(group == "failure")[0], np.where(group == "good, lowland")[0],
                                    rng.choice(hi, min(2000, len(hi)), replace=False)]))
    log(f"cells: {pd.Series(group[cells]).value_counts().to_dict()}")

    def xs_of(ds):
        ds = scaler.transform(ds[s_vars], "static_inputs")
        return np.stack([ds[v].values for v in s_vars], -1).astype("float32")

    def run(net, xd, xs):
        out = []
        with torch.no_grad():
            for i in range(0, len(xd), 256):
                d = torch.from_numpy(xd[i:i + 256]).to(dev)
                s = torch.from_numpy(xs[i:i + 256]).to(dev)[:, None, :].expand(-1, d.shape[1], -1)
                out.append(net(torch.cat([d, s], -1))["y_hat"][:, :, 0].cpu().numpy())
        return np.concatenate(out).astype("float64")

    st_runs = {p: np.stack([st[p].isel(cell=cells + k * n).values for k in range(RUNS_C1)]) for p in PARAMS}
    rows = []
    for year in ("2019", "2022"):
        tsel = slice(f"{year}-01-01", f"{year}-12-31")
        f = scaler.transform(dy[d_vars].sel(time=tsel).isel(cell=cells).load(), "dynamic_inputs")
        xd = np.stack([f[v].transpose("cell", "time").values for v in d_vars], -1).astype("float32")
        cases = {}
        for k in range(RUNS_C1):                                       # the 5 cycle-1 training runs
            idx = cells + k * n
            cases[f"run {k}"] = (st.isel(cell=idx).load(), dy.vwc.sel(time=tsel).isel(cell=idx).values.T)
        ap = st.isel(cell=cells + RUNS_C1 * n).load()
        obs_ap = dy.vwc.sel(time=tsel).isel(cell=cells + RUNS_C1 * n).values.T
        cases["a priori"] = (ap, obs_ap)
        cl = ap.copy(deep=True)
        for p in PARAMS:
            cl[p] = (cl[p].dims, np.clip(cl[p].values, st_runs[p].min(0), st_runs[p].max(0)))
        cases["a priori, clamped"] = (cl, obs_ap)
        for mname, net in nets.items():
            for case, (sds, obs) in cases.items():
                pred = run(net, xd, xs_of(sds))
                obs = obs.astype("float64")
                rows.append(pd.DataFrame(dict(model=mname, year=year, case=case, grp=group[cells],
                                              r=corr(obs.T, pred.T), alpha=pred.std(1) / obs.std(1))))
        log(f"{year} scored")

    res = pd.concat(rows)
    res["case"] = res.case.where(~res.case.str.startswith("run "), "training runs (0-4)")
    RES.mkdir(parents=True, exist_ok=True)
    res.to_csv(RES / "scores.csv", index=False)
    pd.set_option("display.width", 200)
    for stat in ("r", "alpha"):
        tab = res.pivot_table(index=["grp", "case"], columns=["year", "model"], values=stat, aggfunc="median")
        print(f"\nMedian {stat}:\n{tab.round(3).to_string()}")
    poor = res.assign(poor=res.r < 0.5).pivot_table(index=["grp", "case"], columns=["year", "model"],
                                                    values="poor", aggfunc="mean")
    print(f"\nShare with r < 0.5:\n{poor.round(3).to_string()}")
    log(f"scores written to {RES / 'scores.csv'}")


def corr(a, b):
    """Pearson r per column of two (time, cell) arrays."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--step", choices=["archive", "train", "score", "all"], default="all")
    ap.add_argument("--check", action="store_true", help="only print where training would write")
    a = ap.parse_args()
    if a.check:
        check_outputs(test_config())
        return
    if a.step in ("archive", "all"):
        build_archive()
    if a.step in ("train", "all"):
        train()
    if a.step in ("score", "all"):
        score()


if __name__ == "__main__":
    main()
