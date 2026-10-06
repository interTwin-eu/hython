"""dPL campaign with the a priori run in the cycle-0 archive, cycles 0 and 1 only.

The Po Plain test (`apriori_in_archive_test.py`) showed that the cycle-1
surrogate fails at the a priori parameters because they sit outside its
training runs, and that adding the a priori run to the archive fixes it.
This script reruns the campaign with that change:

    cycle 0  archive = production runs 0-3 (the cycle-0 LHS members, reused,
             no wflow) + the a priori run (staticmaps.nc +
             output_baseline_apriori.nc) as run 4. Train, calibrate, score.
    cycle 1  one perturbed member (wflow), retrain, calibrate, score. Stop.

The baseline score is copied from the production state.json (same a priori
run, same scoring). Everything is written under ROOT; the production archive,
cycles, weights and emo1 outputs are only read. The one exception is
`emo1/run_default/inference_parameter.nc`, which hython writes at a fixed path
and `calibrate()` moves into ROOT straight away.

Run with the `emulator` interpreter:

    python scripts/run_apriori_campaign.py --check    # print paths, write nothing
    python scripts/run_apriori_campaign.py            # archive + cycles 0-1
"""
import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import xarray as xr
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pool_archive as pa  # noqa: E402
import run_dpl_cycle as rdc  # noqa: E402

ROOT = Path("/mnt/CEPH_PROJECTS/A_DROP/pilot5/article/dpl_apriori")
STATIC, DYNAMIC = ROOT / "archive" / "static.zarr", ROOT / "archive" / "dynamic.zarr"
PROD_STATE = Path(__file__).resolve().parent / "outputs" / "dpl_cycles" / "state.json"
BASELINE = pa.WD_WFLOW / "run_default" / "output_baseline_apriori.nc"
ENV_BIN = Path("/home/iferrario/.local/miniforge/envs/emulator/bin")

N_LHS = 4  # production runs 0-3 are cycle 0 members 0-3

logger = logging.getLogger("apriori_campaign")


def under_root(p) -> bool:
    return Path(p).resolve().is_relative_to(ROOT.resolve())


def mlflow_index(template: str) -> int:
    """Position of the MLFlowLogger in a template's trainer step."""
    cfg = OmegaConf.load(rdc.WD_CONFIG / f"{template}.yaml")
    loggers = cfg.training_pipeline.steps[1].logger.loggers
    idx = [i for i, lg in enumerate(loggers) if lg._target_.endswith("MLFlowLogger")]
    assert len(idx) == 1, f"{template}: expected one MLFlowLogger, found {idx}"
    return idx[0]


def redirect() -> None:
    """Point every path the driver writes at ROOT."""
    rdc.WD_CYCLE = ROOT / "cycles"
    rdc.WD_RUN = ROOT / "runs"
    rdc.WD_OUT = ROOT / "wflow"
    for mod in (pa, rdc):
        mod.STATIC_ARCHIVE, mod.DYNAMIC_ARCHIVE = STATIC, DYNAMIC


def config() -> rdc.CycleConfig:
    i_train = mlflow_index("config_training_calibration_loop")
    i_cal = mlflow_index("config_calibration_loop")
    assert i_train == i_cal, "overrides are shared: the MLFlowLogger must sit at the same index"
    return rdc.CycleConfig(
        n_cycles=2,
        n_members=N_LHS,
        config_overrides={
            "work_dir": str(ROOT / "runs"),
            f"training_pipeline.steps.1.logger.loggers.{i_train}.savedir": str(ROOT / "mllogs"),
        },
    )


def check(cfg: rdc.CycleConfig) -> None:
    """Stop unless every write the driver makes lands under ROOT."""
    writes = {
        "cycles": rdc.WD_CYCLE, "runs": rdc.WD_RUN, "wflow outputs": rdc.WD_OUT,
        "static archive": pa.STATIC_ARCHIVE, "dynamic archive": pa.DYNAMIC_ARCHIVE,
        "surrogate weights": rdc.surrogate_weights(), "transfer weights": rdc.transfer_weights(),
        "work_dir": cfg.config_overrides["work_dir"],
        "mlflow": [v for k, v in cfg.config_overrides.items() if k.endswith("savedir")][0],
        "cwd (checkpoints)": ROOT,
    }
    for k, v in writes.items():
        logger.info(f"  writes {k}: {v}")
    bad = {k: v for k, v in writes.items() if not under_root(v)}
    assert not bad, f"outside {ROOT}: {bad}"
    # the training template's own output paths, resolved with the overrides
    tpl = OmegaConf.load(rdc.WD_CONFIG / "config_training_calibration_loop.yaml")
    for k, v in cfg.config_overrides.items():
        OmegaConf.update(tpl, k, v)
    text = OmegaConf.to_yaml(tpl, resolve=True)
    assert "/hython_model_run" not in text and "scripts/config/mllogs" not in text, \
        "training config still points at a production folder"
    logger.info("all writes under ROOT")


def build_archive() -> None:
    """Production runs 0-3 + the a priori run as run 4 (cycle 0, member 4)."""
    if STATIC.exists() and DYNAMIC.exists():
        logger.info("archive exists, skipped")
        return
    for p in (STATIC, DYNAMIC):
        assert under_root(p) and not p.exists(), f"refusing to write {p}"
    STATIC.parent.mkdir(parents=True, exist_ok=True)
    n = pa.POOL_SIZE
    prod_static = pa.WD_SURROGATE / "emo1_static_cycle.zarr"
    prod_dynamic = pa.WD_SURROGATE / "emo1_dynamic_cycle.zarr"

    st = xr.open_zarr(prod_static).isel(cell=slice(0, N_LHS * n))
    dy = xr.open_zarr(prod_dynamic).isel(cell=slice(0, N_LHS * n))
    manifest = json.loads(st.attrs["runs"])[:N_LHS]
    assert [(m["run"], m["cycle"], m["member"]) for m in manifest] == \
        [(k, 0, k) for k in range(N_LHS)], f"runs 0-3 are not cycle 0 members 0-3: {manifest}"

    coords = np.stack([st.lat_i.values[:n], st.lon_i.values[:n]], axis=1)
    np.testing.assert_array_equal(coords, pa.draw_pool(st.attrs["pool_seed"], n))
    prov = dict(run=N_LHS, cycle=0, member=N_LHS, static=pa.STATICMAPS.name, output=BASELINE.name)
    st_ap = pa.cut_static(pa.STATICMAPS, coords, prov)
    dy_ap = pa.cut_dynamic(BASELINE, coords, prov)
    assert np.array_equal(dy_ap.time.values, dy.time.values), "time axis differs from the archive"
    logger.info("a priori run cut")

    attrs = dict(pool_seed=int(st.attrs["pool_seed"]), pool_size=int(n),
                 runs=json.dumps(manifest + [prov]))
    for ds_old, ds_new, dest in ((st, st_ap, STATIC), (dy, dy_ap, DYNAMIC)):
        ds_old = pa._tidy(ds_old.copy())
        ds_new = ds_new[list(ds_old.data_vars)]
        ds_old.attrs, ds_new.attrs = dict(attrs), dict(attrs)
        pa._write(pa._chunked(ds_old), dest, mode="w")
        pa._write(pa._chunked(ds_new), dest, mode="a", append_dim="cell")
        logger.info(f"wrote {dest}")
    pa.verify()


def seed_state() -> None:
    """Cycle 0 with its 4 LHS members already ingested, baseline from production."""
    path = rdc.WD_CYCLE / "state.json"
    if path.exists():
        logger.info(f"state exists, resuming: {path}")
        return
    prod = json.loads(PROD_STATE.read_text())
    assert prod["baseline"], "production state has no baseline"
    state = rdc.CycleState(cycle=0, archive_runs=N_LHS + 1, baseline=prod["baseline"],
                           members_done=N_LHS)
    rdc.WD_CYCLE.mkdir(parents=True, exist_ok=True)
    state.save(path)
    logger.info(f"state seeded: {asdict(state)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print where everything goes, write nothing")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    redirect()
    cfg = config()
    check(cfg)
    if a.check:
        return 0

    # itwinai is started through a shell and must come from the emulator env;
    # its checkpoints folder is relative to the working directory
    os.environ["PATH"] = f"{ENV_BIN}:{os.environ['PATH']}"
    for d in (ROOT, rdc.WD_OUT / "run_default"):
        d.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)

    build_archive()
    seed_state()
    rdc.main(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
