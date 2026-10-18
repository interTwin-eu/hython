"""How much does the seed change the learned transfer function? 10 calibrations, one surrogate.

WHY THIS WAS RUN (2026-10-18). After the final run (`run_final_campaign.py`)
one cycle N is chosen. With that cycle's surrogate fixed, the calibration is
repeated with different seeds, each starting from RANDOM initial weights of
the parameter network (no pretraining, not the cycle N-1 network). Then:

    similar surrogate skill, divergent parameter maps  -> equifinality
    different surrogate skill                          -> poor convergence

Skill is the surrogate's own KGE against RT0 (`val_<target>_kgecell`, the
calibration's validation metric, same cells, period and KGE weights as the
loss), read at the epoch the trainer saved (lowest validation loss), on the
training period 2017-2019 (what the optimiser minimises: the direct evidence
on convergence) and on validation 2020 (held out). No wflow runs. This is equifinality as the surrogate sees it: read each parameter
together with its per-parameter fidelity.

Note (2026-09-22): an untrained network gave ~0 in scaled space for every
parameter, with 3 seeds, so different random weights may still start from
nearly the same parameter maps (every parameter near its lower bound).

The seed is the calibration config's `seed`, which itwinai uses as
`random_seed`: it sets the initial weights and the order of the batches.
Everything else is the cycle-N calibration of the final run: the same
templates, CycleConfig defaults and RT0 file (read from that cycle's own
calibration config). The final run is only read.

Each seed writes to ROOT/seed_<s>/: its calibration config and log
(cycles/cycle_<N>/), theta_cal.nc, network weights (runs/). ROOT/summary.json
collects the scores; seeds already done are skipped. Calibration writes
`emo1/run_default/inference_parameter.nc` at a fixed path before it is moved,
so do not run this at the same time as another campaign.

Run with the `emulator` interpreter:

    python scripts/run_transfer_seed_sensitivity.py --cycle N --check   # paths, write nothing
    python scripts/run_transfer_seed_sensitivity.py --cycle N           # seeds 0-9
    python scripts/run_transfer_seed_sensitivity.py --cycle N --seeds 3 4
"""
import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_apriori_campaign as rac  # noqa: E402
import run_dpl_cycle as rdc  # noqa: E402

SRC = Path("/mnt/CEPH_PROJECTS/A_DROP/pilot5/article/dpl_final")
BASE = Path("/mnt/CEPH_PROJECTS/A_DROP/pilot5/article/transfer_seed_sensitivity")
DEFAULT_SEEDS = list(range(10))

logger = logging.getLogger("transfer_seed_sensitivity")


def src_cal_config(cycle: int):
    return OmegaConf.load(SRC / "cycles" / f"cycle_{cycle}" / f"config_calibration_loop_c{cycle}.yaml")


def point_at(seed_dir: Path) -> None:
    rdc.WD_CYCLE = seed_dir / "cycles"
    rdc.WD_RUN = seed_dir / "runs"


def config(seed_dir: Path, seed: int) -> rdc.CycleConfig:
    i = rac.mlflow_index("config_calibration_loop")
    return rdc.CycleConfig(config_overrides={
        "work_dir": str(seed_dir / "runs"),
        f"training_pipeline.steps.1.logger.loggers.{i}.savedir": str(seed_dir / "mllogs"),
        "seed": seed,
        # random initial weights: calibrate() would load the cycle N-1 network at N > 0
        "mt_load_pretrained": False,
        "model_logger.TransferNN.load": False,
    })


def prepare(cycle: int, seed_dir: Path) -> None:
    """The cycle-N surrogate and the calibration's cached input statistics."""
    tr, cal = seed_dir / "runs" / "loop_train_multicycle", seed_dir / "runs" / "loop_cal_multicycle"
    tr.mkdir(parents=True, exist_ok=True)
    cal.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC / "runs" / "loop_train_multicycle" / "model_sequence" / f"CudaLSTM_{cycle}.pt",
                 tr / "CudaLSTM.pt")
    for d, dst in (("loop_train_multicycle", tr), ("loop_cal_multicycle", cal)):
        for f in (SRC / "runs" / d).glob("*.yaml"):
            shutil.copy2(f, dst / f.name)
    (seed_dir / "checkpoints").mkdir(exist_ok=True)


def best_epoch_metrics(log: str) -> dict:
    """Train and validation metrics at the epoch with the lowest validation loss (the saved model).

    Train (2017-2019) is what the optimiser minimises, so it is the direct
    evidence on convergence; validation (2020) is the held-out check.
    """
    loss = [float(v) for v in re.findall(r"val_loss_per_epoch\s*=\s*([-\d.eE+]+)", log)]
    if not loss:
        return {}
    best = min(range(len(loss)), key=loss.__getitem__)
    out = {"epochs": len(loss), "best_epoch": best}
    for split in ("train", "val"):
        series = {"loss": [float(v) for v in re.findall(rf"{split}_loss_per_epoch\s*=\s*([-\d.eE+]+)", log)]}
        for name, v in re.findall(rf"{split}_\w+?_(\w+)_epoch\s*=\s*([-\d.eE+]+)", log):
            if name != "per":  # `<split>_loss_per_epoch` also matches; kept as "loss" above
                series.setdefault(name, []).append(float(v))
        for name, vals in series.items():
            if len(vals) == len(loss):
                out[f"{split}_{name}_best"] = vals[best]
                out[f"{split}_{name}_last"] = vals[-1]
    return out


def check(cycle: int, root: Path, seeds: list) -> None:
    surrogate = SRC / "runs" / "loop_train_multicycle" / "model_sequence" / f"CudaLSTM_{cycle}.pt"
    assert surrogate.exists(), f"no cycle-{cycle} surrogate: {surrogate}"
    assert (SRC / "cycles" / f"cycle_{cycle}" / "theta_cal.nc").exists(), f"cycle {cycle} not calibrated in {SRC}"
    assert not root.resolve().is_relative_to(SRC.resolve())
    logger.info(f"surrogate {surrogate}")
    logger.info(f"RT0 (from the cycle-{cycle} calibration config) {rdc.OBS}")
    for s in seeds:
        d = root / f"seed_{s}"
        point_at(d)
        cfg = config(d, s)
        writes = {"cycles": rdc.WD_CYCLE, "runs": rdc.WD_RUN, "work_dir": cfg.config_overrides["work_dir"],
                  "transfer weights": rdc.transfer_weights(), "surrogate": rdc.surrogate_weights()}
        bad = {k: v for k, v in writes.items() if not Path(v).resolve().is_relative_to(root.resolve())}
        assert not bad, f"seed {s} writes outside {root}: {bad}"
    logger.info(f"seeds {seeds}: all writes under {root}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, required=True, help="the cycle chosen from the final run")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--check", action="store_true", help="print where everything goes, write nothing")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    root = BASE / f"cycle_{a.cycle}"
    rdc.OBS = Path(src_cal_config(a.cycle).data_source.file.target_variables)
    check(a.cycle, root, a.seeds)
    if a.check:
        return 0

    os.environ["PATH"] = f"{rac.ENV_BIN}:{os.environ['PATH']}"
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {
        "source": str(SRC), "cycle": a.cycle, "rt0": str(rdc.OBS), "seeds": {}}

    for s in a.seeds:
        d = root / f"seed_{s}"
        theta = d / "cycles" / f"cycle_{a.cycle}" / "theta_cal.nc"
        if theta.exists():
            logger.info(f"seed {s}: done, skipped")
            continue
        logger.info(f"==== seed {s}")
        point_at(d)
        prepare(a.cycle, d)
        os.chdir(d)
        theta, _ = rdc.calibrate(a.cycle, config(d, s))
        log = (d / "cycles" / f"cycle_{a.cycle}" / f"config_calibration_loop_c{a.cycle}.log").read_text()
        summary["seeds"][str(s)] = {"theta_cal": str(theta)} | best_epoch_metrics(log)
        summary_path.write_text(json.dumps(summary, indent=1))
        logger.info(f"seed {s}: {summary['seeds'][str(s)]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
