"""dPL campaign "final": the c1ap campaign continued to 8 cycles, in its own folder.

Cycles 0 and 1 are not rerun. They are copied from the c1ap campaign
(`run_apriori_campaign.py`, SRC below): archive, cycle folders, network
weights, wflow outputs and logs. SRC is only read. The copied state.json is
set back to unfinished, so the driver continues at cycle 2.

    cycle 0  4 perturbed runs + the a priori run   (copied)
    cycle 1  + 1 perturbed member                  (copied)
    cycle 2-7  + 1 perturbed member each, retrain, calibrate, score

The stopping rule is off (`rel_tol: 0`, `abs_tol: None`), so all 8 cycles
run. Every cycle is scored with a clean wflow run and every calibrated output
is kept; the cycle to analyse is chosen afterwards with the convergence rule.
All other settings are the c1ap ones.

Run with the `emulator` interpreter:

    python scripts/run_final_campaign.py --seed     # copy SRC -> ROOT, nothing else
    python scripts/run_final_campaign.py --check    # print paths, write nothing
    python scripts/run_final_campaign.py            # cycles 2-7
"""
import argparse
import dataclasses
import filecmp
import json
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_apriori_campaign as rac  # noqa: E402
import run_dpl_cycle as rdc  # noqa: E402

SRC = Path("/mnt/CEPH_PROJECTS/A_DROP/pilot5/article/dpl_apriori")
ROOT = Path("/mnt/CEPH_PROJECTS/A_DROP/pilot5/article/dpl_final")
N_CYCLES = 8

# wflow runs through `julia` from a shell; on this VM neither is set up by the
# login profile. Same julia and depot as the c1ap campaign (Wflow 0.8.1).
JULIA_DEPOT = Path("/mnt/CEPH_PROJECTS/A_DROP/pilot5/dev/julia")
JULIA_BIN = JULIA_DEPOT / "juliaup" / "julia-1.11.9+0.x64.linux.gnu" / "bin"
# Measured: wflow speed plateaus at 8 threads (this VM has 16 cores).
JULIA_THREADS = 8

COPY_DIRS = ["archive", "cycles", "runs", "wflow", "mllogs", "outputs"]
COPY_FILES = ["campaign.log", "campaign_resume.log"]

logger = logging.getLogger("final_campaign")


def point_at_root() -> None:
    """Make the c1ap helpers (redirect, config, check) write under ROOT."""
    rac.ROOT = ROOT
    rac.STATIC = ROOT / "archive" / "static.zarr"
    rac.DYNAMIC = ROOT / "archive" / "dynamic.zarr"


def config() -> rdc.CycleConfig:
    return dataclasses.replace(
        rac.config(), n_cycles=N_CYCLES, rel_tol=0.0, abs_tol=None,
        julia_threads=JULIA_THREADS,
    )


def _same_tree(a: Path, b: Path) -> None:
    """Stop unless b holds every file of a with the same size."""
    for f in a.rglob("*"):
        if f.is_file():
            g = b / f.relative_to(a)
            assert g.is_file() and g.stat().st_size == f.stat().st_size, f"copy differs: {g}"


def seed() -> None:
    """Copy the c1ap campaign into ROOT and mark it unfinished. SRC is only read."""
    assert not ROOT.exists(), f"{ROOT} exists; refusing to overwrite it"
    assert not ROOT.resolve().is_relative_to(SRC.resolve())
    src_state = json.loads((SRC / "cycles" / "state.json").read_text())
    assert src_state["finished"] and src_state["cycle"] == 2 and src_state["archive_runs"] == 6, \
        f"c1ap state is not the expected end of cycle 1: {src_state}"

    ROOT.mkdir(parents=True)
    for d in COPY_DIRS:
        shutil.copytree(SRC / d, ROOT / d, copy_function=shutil.copy2)
        _same_tree(SRC / d, ROOT / d)
        logger.info(f"copied {d}/")
    for f in COPY_FILES:
        shutil.copy2(SRC / f, ROOT / f)
        assert filecmp.cmp(SRC / f, ROOT / f, shallow=False)
    (ROOT / "checkpoints").mkdir()

    state = json.loads((ROOT / "cycles" / "state.json").read_text())
    state["finished"] = False
    (ROOT / "cycles" / "state.json").write_text(json.dumps(state, indent=2))
    (ROOT / "SEEDED_FROM.txt").write_text(
        f"Cycles 0-1 copied from {SRC} (c1ap campaign); state.json set to finished=false.\n"
    )
    logger.info(f"seeded {ROOT} from {SRC}")


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--seed", action="store_true", help="copy the c1ap campaign into ROOT, then stop")
    g.add_argument("--check", action="store_true", help="print where everything goes, write nothing")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    point_at_root()
    if a.seed:
        seed()
        return 0

    rac.redirect()
    cfg = config()
    rac.check(cfg)
    state = json.loads((ROOT / "cycles" / "state.json").read_text())
    logger.info(f"state: cycle {state['cycle']}, archive_runs {state['archive_runs']}, "
                f"finished {state['finished']}; n_cycles {cfg.n_cycles}, rel_tol {cfg.rel_tol}, "
                f"abs_tol {cfg.abs_tol}, julia_threads {cfg.julia_threads}")
    assert not state["finished"], "state.json says finished; was --seed run?"
    if a.check:
        return 0

    os.environ["PATH"] = f"{rac.ENV_BIN}:{JULIA_BIN}:{os.environ['PATH']}"
    os.environ["JULIA_DEPOT_PATH"] = str(JULIA_DEPOT)
    os.chdir(ROOT)
    rdc.main(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
