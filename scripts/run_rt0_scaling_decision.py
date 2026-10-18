"""Which scaled RT0 to calibrate on: a 2-cycle dPL run per RT0 version.

WHY THIS WAS RUN (2026-10-18). Three scalings of the RT0 soil moisture exist,
`alps_rt0old_2017-2022_{theta,minmax,variance}.nc`. Each one is a different
calibration target. Before the final 8-cycle run (`run_final_campaign.py`)
we pick one, by comparing per version and cycle

    dKGE = KGE(wflow_cal, RT0_version) - KGE(wflow_apriori, RT0_version)

(median per cell; valid 2020, test 2021-2022; with r, alpha, beta) and the
share of cells with RootingDepth < 10 mm (the collapse seen in step3b).

`theta` needs no run: the c1ap campaign (`run_apriori_campaign.py`,
article/dpl_apriori) is exactly this run on `theta`. This script makes the
same run for `minmax` and `variance`, with every other setting as in c1ap.

    cycle 0  archive = 4 perturbed runs + the a priori run (copied from c1ap)
             surrogate = the c1ap cycle-0 surrogate (copied; RT0 is not used
             in surrogate training). Calibrate on this RT0, score.
    cycle 1  one perturbed member around this version's calibration (wflow),
             retrain, calibrate, score. Stop.

One setting, `run_dpl_cycle.OBS`, is the calibration target and the scoring
reference, so both follow the version. The dynamic RT0 mask is the same for
all versions. The a priori score (the baseline) is computed once in --seed,
from the existing a priori wflow output against this version's RT0: no wflow
run, and the driver never runs its own baseline (it would write into
emo1/run_default).

c1ap is only read. Run the versions ONE AT A TIME: calibration writes
`emo1/run_default/inference_parameter.nc` at a fixed path before it is moved.

Run with the `emulator` interpreter:

    python scripts/run_rt0_scaling_decision.py --rt0 minmax --seed    # set up the folder
    python scripts/run_rt0_scaling_decision.py --rt0 minmax --check   # print paths, write nothing
    python scripts/run_rt0_scaling_decision.py --rt0 minmax           # cycles 0-1
"""
import argparse
import dataclasses
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pool_archive as pa  # noqa: E402
import run_apriori_campaign as rac  # noqa: E402
import run_dpl_cycle as rdc  # noqa: E402

SRC = Path("/mnt/CEPH_PROJECTS/A_DROP/pilot5/article/dpl_apriori")
BASE = Path("/mnt/CEPH_PROJECTS/A_DROP/pilot5/article/rt0_scaling_decision")
RT0_DIR = Path("/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/SSM-RT0-SIG0-R-CRRL/processed")
VERSIONS = ("minmax", "variance")  # theta = the c1ap campaign
N_CYCLE0_RUNS = 5  # 4 perturbed + a priori; c1ap's run 5 is drawn around the theta calibration
N_CYCLES = 2

# Same julia and depot as the c1ap campaign (Wflow 0.8.1); see run_final_campaign.py.
JULIA_DEPOT = Path("/mnt/CEPH_PROJECTS/A_DROP/pilot5/dev/julia")
JULIA_BIN = JULIA_DEPOT / "juliaup" / "julia-1.11.9+0.x64.linux.gnu" / "bin"
JULIA_THREADS = 8

logger = logging.getLogger("rt0_scaling_decision")


def rt0_path(version: str) -> Path:
    return RT0_DIR / f"alps_rt0old_2017-2022_{version}.nc"


def point_at(version: str) -> Path:
    """Make the c1ap helpers write under this version's folder, and calibrate
    and score on this version's RT0."""
    root = BASE / version
    rac.ROOT = root
    rac.STATIC = root / "archive" / "static.zarr"
    rac.DYNAMIC = root / "archive" / "dynamic.zarr"
    rdc.OBS = rt0_path(version)
    return root


def config() -> rdc.CycleConfig:
    # n_cycles 2 already stops after cycle 1; the stopping rule is off so it cannot stop earlier
    return dataclasses.replace(
        rac.config(), n_cycles=N_CYCLES, rel_tol=0.0, abs_tol=None,
        julia_threads=JULIA_THREADS, score_baseline=False,
    )


def cut_archive(root: Path) -> None:
    """c1ap archive, first 5 runs (cycle 0) only."""
    n = pa.POOL_SIZE
    for name in ("static", "dynamic"):
        ds = xr.open_zarr(SRC / "archive" / f"{name}.zarr")
        runs = json.loads(ds.attrs["runs"])[:N_CYCLE0_RUNS]
        assert [r["cycle"] for r in runs] == [0] * N_CYCLE0_RUNS, f"not all cycle 0: {runs}"
        assert runs[-1]["output"] == rac.BASELINE.name, "run 4 is not the a priori run"
        ds = pa._tidy(ds.isel(cell=slice(0, N_CYCLE0_RUNS * n)).copy())
        ds.attrs = dict(pool_seed=int(ds.attrs["pool_seed"]), pool_size=int(n), runs=json.dumps(runs))
        pa._write(pa._chunked(ds), root / "archive" / f"{name}.zarr", mode="w")
        logger.info(f"archive {name}: {N_CYCLE0_RUNS} runs written")
    pa.verify()


def copy_surrogate(root: Path) -> None:
    """The c1ap cycle-0 surrogate and the input statistics it was trained with."""
    src, dst = SRC / "runs" / "loop_train_multicycle", root / "runs" / "loop_train_multicycle"
    (dst / "model_sequence").mkdir(parents=True)
    shutil.copy2(src / "model_sequence" / "CudaLSTM_0.pt", dst / "model_sequence" / "CudaLSTM_0.pt")
    shutil.copy2(src / "model_sequence" / "CudaLSTM_0.pt", dst / "CudaLSTM.pt")
    for f in ("static_inputs.yaml", "dynamic_inputs.yaml"):
        shutil.copy2(src / f, dst / f)
    # the cycle-0 training config and log, for the record
    (root / "cycles" / "cycle_0").mkdir(parents=True)
    for f in ("config_training_calibration_loop_c0.yaml", "config_training_calibration_loop_c0.log"):
        shutil.copy2(SRC / "cycles" / "cycle_0" / f, root / "cycles" / "cycle_0" / f)
    logger.info("cycle-0 surrogate copied")


def seed(version: str, root: Path, cfg: rdc.CycleConfig) -> None:
    assert not root.exists(), f"{root} exists; refusing to overwrite it"
    assert rdc.OBS.exists(), rdc.OBS
    root.mkdir(parents=True)
    (root / "checkpoints").mkdir()
    rac.redirect()
    cut_archive(root)
    copy_surrogate(root)

    logger.info(f"baseline: a priori wflow output vs {rdc.OBS.name} (no wflow run)")
    baseline = rdc.score(rac.BASELINE, cfg)
    state = rdc.CycleState(cycle=0, archive_runs=N_CYCLE0_RUNS, baseline=baseline,
                           members_done=N_CYCLE0_RUNS - 1, stages_done=["train"])
    state.save(root / "cycles" / "state.json")
    (root / "SEEDED_FROM.txt").write_text(
        f"RT0 scaling decision, version '{version}': calibration target and score = {rdc.OBS}.\n"
        f"Cycle-0 archive (runs 0-4) and cycle-0 surrogate copied from {SRC} (c1ap campaign, theta).\n"
        f"Baseline = {rac.BASELINE.name} scored against this RT0.\n"
    )
    logger.info(f"seeded {root}; baseline {baseline}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt0", choices=VERSIONS, required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--seed", action="store_true", help="set up this version's folder, then stop")
    g.add_argument("--check", action="store_true", help="print where everything goes, write nothing")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    root = point_at(a.rt0)
    cfg = config()
    if a.seed:
        seed(a.rt0, root, cfg)
        return 0

    rac.redirect()
    rac.check(cfg)
    state = json.loads((root / "cycles" / "state.json").read_text())
    logger.info(f"RT0 {rdc.OBS}")
    logger.info(f"state: cycle {state['cycle']}, archive_runs {state['archive_runs']}, "
                f"stages {state['stages_done']}, finished {state['finished']}; "
                f"n_cycles {cfg.n_cycles}, kge_weights {cfg.kge_weights}")
    assert state["baseline"], "no baseline in state.json; the driver would run its own in emo1"
    assert not state["finished"]
    if a.check:
        return 0

    os.environ["PATH"] = f"{rac.ENV_BIN}:{JULIA_BIN}:{os.environ['PATH']}"
    os.environ["JULIA_DEPOT_PATH"] = str(JULIA_DEPOT)
    os.chdir(root)
    rdc.main(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
