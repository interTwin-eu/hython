"""One-parameter wflow_sbm runs at the c1ap calibrated parameters (article C6.2, panel vi).

For each calibrated parameter p, P_p = the a priori staticmaps with only p
taken from the c1ap cycle-1 calibrated staticmaps (`staticmaps_cycle1_cal.nc`).
The two files differ only in the five calibrated parameters (checked
2026-10-06), so P_p moves exactly one of them. Each P_p runs like the c1ap
calibrated run: same TOML template, window, warm state and compression
(`run_dpl_cycle.run_wflow`), then cut to the top layer (`keep_top_layer`).
The C6.2 notebook compares wflow(P_p) - wflow(a priori) with the same
difference of the c1ap surrogate. The a priori reference is
emo1/run_default/output_baseline_apriori.nc, made the same way.

The first submission made these runs with its own calibrated parameters
(emo1/review/P_<p>_*); those are only read here, never written.

Runs one after the other (parallel runs do not help, see the wflow
measurements). About 50-70 min each. Output is written only under OUT. A run
whose output exists is skipped.

Run with the `emulator` interpreter:

    python scripts/c62_param_runs_c1ap.py --check            # print the plan, write nothing
    python scripts/c62_param_runs_c1ap.py                    # all five
    python scripts/c62_param_runs_c1ap.py --params KsatVer c # some
"""
import argparse
import logging
import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_apriori_campaign as rac  # noqa: E402
import run_dpl_cycle as rdc  # noqa: E402

PARAMS = ("KsatVer", "c", "f", "RootingDepth", "Sl")
APRIORI = rdc.WD_WFLOW / "staticmaps.nc"
CAL = rac.ROOT / "wflow" / "staticmaps_cycle1_cal.nc"
OUT = rac.ROOT / "wflow" / "c62"

logger = logging.getLogger("c62_param_runs")


def write_static(p: str, dest: Path) -> None:
    """The a priori staticmaps with only parameter `p` from the calibrated file."""
    with xr.open_dataset(APRIORI) as ap, xr.open_dataset(CAL) as cal:
        ds = ap.load()
        assert ds[p].dims == cal[p].dims, f"{p}: dims differ"
        ds[p] = cal[p].load()
    tmp = dest.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp, mode="w")
    tmp.replace(dest)


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--params", nargs="+", choices=PARAMS, default=list(PARAMS))
    a.add_argument("--check", action="store_true", help="print the plan, write nothing")
    a = a.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg = rac.config()   # the c1ap campaign's settings (threads, window, compression)
    for f in (APRIORI, CAL):
        assert f.exists(), f"missing {f}"
    logger.info(f"a priori {APRIORI}\ncalibrated {CAL}\nout {OUT}")
    logger.info(f"threads {cfg.julia_threads}, window {cfg.wflow_starttime or 'template'}-"
                f"{cfg.wflow_endtime or 'template'}, top layer only {cfg.top_layer_only}")

    for p in a.params:
        static = OUT / f"staticmaps_P_{p}.nc"
        out_nc = OUT / "run_default" / f"output_P_{p}.nc"
        assert rac.under_root(static) and rac.under_root(out_nc)
        if out_nc.exists():
            logger.info(f"{p}: {out_nc.name} exists, skipped")
            continue
        logger.info(f"{p}: write {static.name}, run wflow -> {out_nc}")
        if a.check:
            continue
        out_nc.parent.mkdir(parents=True, exist_ok=True)
        if not static.exists():
            write_static(p, static)
        rdc.run_wflow(static, out_nc, cfg)
        if cfg.top_layer_only:
            rdc.keep_top_layer(out_nc, cfg)
        logger.info(f"{p}: done")


if __name__ == "__main__":
    main()
