"""End-to-end smoke test of the multicycle dPL loop.

Runs the real `run_dpl_cycle.main()` - the same wflow calls, the same archive
code, the same training and calibration pipelines - with everything scaled
down and every writable path redirected away from production:

    wflow            1 year instead of 6
    cycle 0          2 members instead of 4
    training         small row budget, low epoch ceiling
    archive          emo1_*_smoke.zarr, not emo1_*_cycle.zarr
    run folders      a scratch directory, not hython_model_run
    cycle outputs    a scratch directory, not scripts/outputs/dpl_cycles

The point is to exercise the steps that have never run together - especially
calibration, which touches `WflowSBMCal`, the Hybrid model and TransferNN -
before committing hours of wflow time to a real run.

    python smoke_multicycle.py --out /path/to/scratch [--cycles 2] [--parallel 2]

Nothing here belongs in a real run: the loop itself has no test mode, only the
hooks (`wflow_endtime`, `config_overrides`) this script fills in.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pool_archive
import run_dpl_cycle
from run_dpl_cycle import CycleConfig

logger = logging.getLogger("smoke")


def redirect(out: Path) -> None:
    """Point every writable path at `out`, leaving the read-only ones alone.

    wflow's model directory is *not* redirected: it holds the forcing, the
    staticmaps and the julia environment, all read-only here. What it does
    receive is the per-run staticmaps and outputs, which carry cycle numbers in
    their names and are pruned by the loop.
    """
    (out / "cycles").mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(parents=True, exist_ok=True)

    run_dpl_cycle.WD_CYCLE = out / "cycles"
    run_dpl_cycle.WD_RUN = out / "runs"

    # The archive lives beside the real ones but under its own name, so the
    # production stores are never opened, let alone appended to.
    static = pool_archive.WD_SURROGATE / "emo1_static_smoke.zarr"
    dynamic = pool_archive.WD_SURROGATE / "emo1_dynamic_smoke.zarr"
    for mod in (pool_archive, run_dpl_cycle):
        mod.STATIC_ARCHIVE = static
        mod.DYNAMIC_ARCHIVE = dynamic

    logger.info(f"cycles   -> {run_dpl_cycle.WD_CYCLE}")
    logger.info(f"runs     -> {run_dpl_cycle.WD_RUN}")
    logger.info(f"archive  -> {static.name}, {dynamic.name}")


def windows(years: int) -> dict:
    """Train/valid/test ranges inside a `years`-long wflow window.

    Each split must be longer than `seq_length`, or it yields no start points
    at all and the dataset comes out empty - a 1-year window cannot support
    the production `seq_length: 120`, which is why `--years` exists.
    """
    from datetime import date, timedelta

    start = date(2017, 1, 1)
    span = 365 * years - 2
    a = start + timedelta(days=int(span * 0.60))
    b = start + timedelta(days=int(span * 0.85))
    end = start + timedelta(days=span)
    iso = lambda d: d.isoformat()
    return {
        "train_temporal_range": [iso(start), iso(a)],
        "valid_temporal_range": [iso(a + timedelta(days=1)), iso(b)],
        "test_temporal_range": [iso(b + timedelta(days=1)), iso(end)],
    }


def config(out: Path, args) -> CycleConfig:
    from datetime import date, timedelta

    wflow_end = date(2017, 1, 1) + timedelta(days=365 * args.years)
    return CycleConfig(
        n_cycles=args.cycles,
        n_members=args.members,
        n_members_later=1,
        wflow_parallel=args.parallel,
        wflow_starttime="2016-12-31T00:00:00",
        wflow_endtime=f"{wflow_end.isoformat()}T00:00:00",
        prune_outputs=True,
        score_every=args.score_every,
        score_first_last=args.score_first_last,
        converge_on=args.converge_on,
        rel_tol=args.rel_tol,
        config_overrides={
            "work_dir": str(out / "runs"),
            "epochs": args.epochs,
            "train_rows_target": args.rows,
            "early_stopping_patience": args.patience,
            "seq_length": args.seq,
            **windows(args.years),
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--members", type=int, default=2, help="members at cycle 0")
    ap.add_argument("--years", type=int, default=1, help="length of the wflow window")
    ap.add_argument("--seq", type=int, default=30, help="LSTM sequence length")
    ap.add_argument("--rows", type=int, default=400, help="train_rows_target")
    ap.add_argument("--epochs", type=int, default=3, help="epoch ceiling")
    ap.add_argument("--patience", type=int, default=2, help="early stopping patience")
    ap.add_argument("--score-every", type=int, default=2, dest="score_every")
    ap.add_argument("--score-first-last", action="store_true",
                    dest="score_first_last",
                    help="score cycle 0 and the last cycle whatever --score-every says")
    ap.add_argument("--converge-on", default="surrogate", dest="converge_on",
                    choices=["wflow", "surrogate", "both"])
    ap.add_argument("--rel-tol", type=float, default=0.01, dest="rel_tol",
                    help="0 disables stopping, forcing the full n_cycles")
    ap.add_argument("--fresh", action="store_true",
                    help="delete the smoke archive and state before starting")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    redirect(args.out)

    if args.fresh:
        for store in (pool_archive.STATIC_ARCHIVE, pool_archive.DYNAMIC_ARCHIVE):
            if store.exists():
                shutil.rmtree(store)
                logger.info(f"removed {store.name}")
        state = run_dpl_cycle.WD_CYCLE / "state.json"
        if state.exists():
            state.unlink()

    cfg = config(args.out, args)
    w = windows(args.years)
    logger.info(f"{args.cycles} cycles | {cfg.n_members} members at cycle 0 | "
                f"wflow {args.years}y on {cfg.threads_per_run()} threads")
    logger.info(f"seq_length {args.seq} | rows {args.rows} | epochs<={args.epochs} "
                f"| patience {args.patience} | score_every {cfg.score_every}"
                f"{' + first/last' if cfg.score_first_last else ''}")
    logger.info(f"converge_on {cfg.converge_on} | rel_tol {cfg.rel_tol}"
                f"{' (stopping disabled)' if cfg.rel_tol == 0 else ''}")
    for k, v in w.items():
        logger.info(f"  {k:24s} {v[0]} -> {v[1]}")

    run_dpl_cycle.main(cfg)
    logger.info("smoke test finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
