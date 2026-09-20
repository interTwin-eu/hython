"""Multi-cycle dPL calibration of wflow_sbm.

Implements the iterated surrogate refinement of Tsai et al. (2021), as also used
by Ahmad et al. (2025): each cycle perturbs the *calibrated* parameters from the
previous cycle, runs wflow_sbm on those perturbed members, and appends the
resulting (theta, vwc) pairs to a growing surrogate training archive.

Two distinct artifacts hold theta, with opposite update semantics:

  staticmaps_cycle{n}_m{k}.nc  wflow_sbm input.      Replaced every member.
  emo1_static_cycle.zarr       surrogate train set.  Appended every run.

The archive must accumulate. If it held one theta per gridcell the surrogate
could fit the target from cell identity alone and never attribute variance to
the calibrated parameters, which makes d(vwc)/d(theta) - the quantity dPL
backpropagates through - meaningless.

Storage is delegated to `pool_archive` (A1). Each wflow run writes ~10 GB over
the full map, of which training reads a few thousand cells, so only a fixed
random 5% of cells - the *pool* - is kept, and runs are stacked along one
`cell` axis instead of gaining a `cycle` dimension. ~392 MB a run, measured.
See `pool_archive.py` for the layout and why the forcing sits in the same store
as the target.

How much is run each cycle (A2):

  cycle 0   4 perturbed members, spread across the physical range by a Latin
            hypercube, plus the clean theta_cal run.
  cycle 1+  1 perturbed member plus the clean run. After cycle 0 the members
            carry no level offset, only per-cell noise, and one run already
            gives thousands of (theta, vwc) pairs for 5 parameters.

Training stops on its own (H7, `early_stopping_patience`), so `epochs` in the
config is a ceiling rather than a target, and the row budget per epoch is fixed
by `train_rows_target` rather than a fraction of a growing archive.

Requires the hython changes in HYTHON_MULTICYCLE_PLAN.md (H1-H5, H7). Without
them the dataset cannot read the stacked layout.
"""

import json
import logging
import shutil
import subprocess as sp
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Iterator
from pathlib import Path

import numpy as np
import toml
import xarray as xr
from omegaconf import OmegaConf

import pool_archive
from pool_archive import DYNAMIC_ARCHIVE, STATIC_ARCHIVE, _unpack_layer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ==== CONFIGURATION

WD_WFLOW = Path("/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1")
WD_SURROGATE = Path("/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_input")
WD_RUN = Path("/mnt/CEPH_PROJECTS/InterTwin/hython_model_run")
WD_CONFIG = Path(__file__).resolve().parent / "config"
WD_CYCLE = Path(__file__).resolve().parent / "outputs" / "dpl_cycles"

# The two archive stores live in pool_archive, which owns their layout.
# `DYNAMIC_ARCHIVE` holds the forcing *and* the target, because that is what
# WflowSBM_Pool reads: it opens `dynamic_inputs` once and takes both out of it.

OBS = Path(
    "/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/SSM-RT0-SIG0-R-CRRL/processed"
    "/alps_rt0old_2017-2022_theta.nc"
)

# Calibrated parameters and their physical bounds. Must stay consistent with
# `scaling_static_range` in the training config, since BoundedScaler normalises
# with the same numbers and the perturbation happens in that normalised space.
CAL_PARAMS = {
    "KsatVer": (1.0, 8000.0),
    "c": (1.0, 20.0),
    "f": (1.0e-05, 0.01),
    "RootingDepth": (5.0, 3000.0),
    "Sl": (0.02, 0.2),
}

# `c` is stored with a leading layer axis in staticmaps.nc.
LAYERED_PARAMS = {"c"}


@dataclass
class CycleConfig:
    n_cycles: int = 8
    # A2(a). Cycle 0 is the only time the surrogate sees parameters far from
    # the current guess: `build_members` spreads its members across the whole
    # physical range with a Latin hypercube. After that the offsets are zero
    # and runs differ only by per-cell noise, which one run supplies just as
    # well - it already gives thousands of (theta, vwc) pairs for 5 parameters.
    # 4 + 1 per cycle is 40 runs and ~36 h; 4 then 1 is 19 runs and ~17 h.
    n_members: int = 4        # cycle 0, feeding the archive
    n_members_later: int = 1  # cycles 1+

    # Cycle 0 seeds the archive with a Latin hypercube over the full physical
    # range. Ahmad 2025 sec 2.4.1: training "designed to capture sensitivity to
    # the model parameters, by including a range of parameter values".
    seed_offset: float = 0.45  # max +/- LHS offset in normalised space
    seed_jitter: float = 0.05  # per-cell sigma at cycle 0

    # Later cycles perturb theta_cal locally, annealed as the search settles.
    # Tsai 2021: "as the search algorithms went near an optimum".
    jitter_start: float = 0.15
    # A2(a). Floored at 0.05-0.08, not 0.03: with one run per cycle the
    # per-cell noise is the *only* local signal the surrogate gets after cycle
    # 0, so annealing it away leaves nothing to learn the local gradient from.
    jitter_end: float = 0.06

    # Ahmad 2025 sec 2.4.1 stopping rule, on the clean theta_cal wflow run.
    rel_tol: float = 0.01

    seed: int = 42
    skip_wflow: bool = False
    skip_train: bool = False
    skip_calibration: bool = False
    # A2(f), measured: one wflow run plateaus at ~8 threads - 12 to 24 buys
    # 2.3%. So `julia_threads` is the **total core budget**, split across
    # concurrent runs, rather than a per-run count. `wflow_parallel: 1` leaves
    # 24 threads on one run, exactly as before.
    #
    #   wflow_parallel   threads each   30-day run each
    #   1                24             28.3 s
    #   3                8              28.9 s
    #   4                6              32.2 s
    #
    # 3 or 4 is the sweet spot on 24 cores. It only helps where a cycle has
    # several members to run, which after A2(a) means cycle 0.
    julia_threads: int = 24   # total, not per run
    wflow_parallel: int = 1   # concurrent wflow runs
    wflow_retries: int = 2  # A2(a): one run per cycle, so a failure costs a whole cycle

    # A2(f). Measured on a 30-day run: level 5 costs 20% of the simulation time
    # and level 1 gives 131 MB against level 5's 128 MB - 2.3% more disk for
    # 8.5% less time. Level 0 is not the alternative: it is 318 MB, 2.5x.
    # Set on the per-run TOML, never on the shared template.
    output_compression: int = 1

    # A full 8-cycle run leaves ~190 GB of full-map netCDFs for an archive that
    # reads ~4 GB of them. The pool cells are taken at ingestion, so the big
    # file has no reader afterwards. Kept: cycle 0 and the last cycle (for
    # figures and for checking the surrogate against the full map), and every
    # cycle's *score*, which lives in the state JSON and is never a netCDF.
    prune_outputs: bool = True

    def members_for(self, cycle: int) -> int:
        """How many perturbed wflow runs this cycle feeds the archive (A2 a)."""
        return self.n_members if cycle == 0 else self.n_members_later

    def threads_per_run(self) -> int:
        """Julia threads for one wflow run, dividing the total core budget."""
        return max(1, self.julia_threads // max(1, self.wflow_parallel))


@dataclass
class CycleState:
    """Persisted between stages so an interrupted run resumes cleanly."""

    cycle: int = 0
    archive_runs: int = 0  # wflow runs committed to BOTH stores
    history: list = field(default_factory=list)  # per-cycle {rmse, bias}

    @classmethod
    def load(cls, path: Path) -> "CycleState":
        if path.exists():
            return cls(**json.loads(path.read_text()))
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))


# ==== PARAMETER PERTURBATION


def _reflect_unit(u: np.ndarray) -> np.ndarray:
    """Fold values back into [0, 1] by reflection at the bounds.

    Reflection rather than clipping: when theta_cal sits on a bound, clipping
    stacks every member exactly on it and the surrogate sees no local variation
    in the one direction that matters.
    """
    u = np.mod(u, 2.0)
    return np.where(u > 1.0, 2.0 - u, u)


def perturb(
    da: xr.DataArray,
    lo: float,
    hi: float,
    jitter: float,
    offset: float,
    rng: np.random.Generator,
) -> xr.DataArray:
    """Perturb a parameter map inside its physical bounds.

    `offset` shifts the whole map (keeps the spatial pattern, moves the level),
    `jitter` decorrelates individual cells. Both act in normalised space, so
    bounds hold by construction and no negative-value repair is needed.
    """
    u = (da - lo) / (hi - lo)
    u = u + offset + rng.normal(0.0, jitter, size=u.shape)
    return xr.apply_ufunc(_reflect_unit, u) * (hi - lo) + lo


def latin_hypercube(n_members: int, n_params: int, rng: np.random.Generator) -> np.ndarray:
    """LHS on [0, 1)^n_params, returned as (n_members, n_params)."""
    cut = (np.arange(n_members)[:, None] + rng.random((n_members, n_params))) / n_members
    return np.take_along_axis(
        cut, rng.random((n_members, n_params)).argsort(axis=0), axis=0
    )


def build_members(
    cycle: int, cfg: CycleConfig, theta: xr.Dataset, rng: np.random.Generator
) -> list[xr.Dataset]:
    """Parameter maps for every wflow run of this cycle.

    Cycle 0 spans the physical range (LHS on the level offset). Later cycles
    perturb theta_cal locally with an annealed jitter.
    """
    names = list(CAL_PARAMS)
    n_members = cfg.members_for(cycle)

    if cycle == 0:
        lhs = latin_hypercube(n_members, len(names), rng)
        offsets = (lhs - 0.5) * 2.0 * cfg.seed_offset
        jitter = cfg.seed_jitter
    else:
        offsets = np.zeros((n_members, len(names)))
        frac = cycle / max(cfg.n_cycles - 1, 1)
        jitter = cfg.jitter_start + (cfg.jitter_end - cfg.jitter_start) * frac

    logger.info(f"cycle {cycle}: {n_members} member(s), jitter={jitter:.3f}")

    members = []
    for m in range(n_members):
        out = theta.copy(deep=True)
        for p, name in enumerate(names):
            lo, hi = CAL_PARAMS[name]
            out[name] = perturb(theta[name], lo, hi, jitter, offsets[m, p], rng)
        members.append(out)
    return members


# ==== STATICMAPS I/O


def read_theta(path: Path) -> xr.Dataset:
    """Calibrated parameters, gap-filled and renamed onto the staticmaps grid."""
    ds = xr.open_dataset(path).load()
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    # dPL leaves gaps where predictors are missing; fill along a single axis so
    # the result stays on the wflow grid.
    return ds.interpolate_na(dim="longitude", method="linear").ffill("longitude").bfill(
        "longitude"
    )


def write_staticmaps(theta: xr.Dataset, dest: Path) -> None:
    """Write theta into a copy of the original staticmaps, preserving the mask."""
    base = xr.open_dataset(WD_WFLOW / "staticmaps.nc").load()
    nodata = base["thetaS"].isnull()

    for name in CAL_PARAMS:
        filled = theta[name].where(~nodata)
        if name in LAYERED_PARAMS:
            base[name][0] = filled
        else:
            base[name] = filled

    if dest.exists():
        dest.unlink()
    base.to_netcdf(dest, mode="w")
    base.close()


def run_wflow(static: Path, out_nc: Path, cfg: CycleConfig) -> None:
    """One wflow_sbm forward run against the given staticmaps."""
    with open(WD_WFLOW / "wflow_sbm_workflow.toml", "rb") as f:
        data = tomllib.load(f)

    data["input"]["path_static"] = str(static)
    data["output"]["path"] = str(out_nc)
    data["output"]["compressionlevel"] = cfg.output_compression
    data["csv"]["path"] = str(out_nc.with_suffix(".csv"))

    run_toml = WD_WFLOW / f"wflow_sbm_{out_nc.stem}.toml"
    with open(run_toml, "w") as f:
        toml.dump(data, f)

    cmd = [
        "julia",
        f"--project={WD_WFLOW}",
        f"-t {cfg.julia_threads}",
        f'-e using Wflow;Wflow.run("{run_toml}")',
    ]

    # A2(a). With one run per cycle a failure costs the whole cycle - the
    # archive gains nothing and the surrogate sees no new parameters. `check`
    # used to stop the loop outright; retry first, and only then give up.
    attempts = max(1, cfg.wflow_retries + 1)
    for attempt in range(1, attempts + 1):
        result = sp.run(cmd)
        if result.returncode == 0:
            return
        if attempt < attempts:
            logger.warning(
                f"wflow failed (rc={result.returncode}) on {out_nc.name}, "
                f"attempt {attempt} of {attempts}; retrying"
            )

    raise sp.CalledProcessError(result.returncode, cmd)


# ==== SURROGATE ARCHIVE

# Cutting, stacking and the both-stores-or-neither write all live in
# `pool_archive`. What stays here is only the decision of *what* to ingest.


def ingest_member(static_nc: Path, output_nc: Path, cycle: int, member: int,
                  state: CycleState) -> None:
    """Add one wflow run to the archive and advance the run count.

    `pool_archive.ingest_run` stages both stores and promotes them together, so
    a crash leaves the archive at a whole number of runs rather than pairing
    theta from one run with vwc from another.
    """
    pool_archive.ingest_run(
        static_nc, output_nc,
        first=not STATIC_ARCHIVE.exists(),
        cycle=cycle, member=member,
    )
    state.archive_runs += 1


# ==== SURROGATE TRAINING AND CALIBRATION


def write_cycle_config(template: str, cycle: int, overrides: dict) -> tuple[Path, str]:
    """Materialise a per-cycle config instead of mutating the shared one.

    Mutating and re-saving the template leaves it in a cycle-n state whenever a
    run dies, so the next run starts from the wrong flags.
    """
    cfg = OmegaConf.load(WD_CONFIG / f"{template}.yaml")
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)

    out_dir = WD_CYCLE / f"cycle_{cycle}"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{template}_c{cycle}"
    OmegaConf.save(cfg, out_dir / f"{name}.yaml")
    return out_dir, name


def train_surrogate(cycle: int, cfg: CycleConfig) -> None:
    """Retrain on the full archive, warm-started from the previous cycle.

    Scaler statistics are computed once at cycle 0 and cached thereafter. The
    vwc distribution shifts as theta moves, so refitting MinMax01Scaler every
    cycle would redefine what the frozen output head means while the weights are
    warm-started.
    """
    overrides = {
        "experiment_run": "train_multicycle",
        "model_logger.CudaLSTM.load": cycle > 0,
        "scaling_use_cached": cycle > 0,
        "data_source.file.static_inputs": str(STATIC_ARCHIVE),
        # Both keys name the same store on purpose: WflowSBM_Pool opens
        # `dynamic_inputs` once and reads the target out of that same object,
        # and never opens `target_variables` at all.
        "data_source.file.dynamic_inputs": str(DYNAMIC_ARCHIVE),
        "data_source.file.target_variables": str(DYNAMIC_ARCHIVE),
        # H3. The downsamplers take every run of each chosen base cell, so they
        # have to know how many runs the archive holds now. This also shrinks
        # `places` as the archive grows (A2 b2), keeping rows per epoch fixed.
        "train_downsampler.runs": pool_archive.next_run_index(),
        "valid_downsampler.runs": pool_archive.next_run_index(),
    }
    out_dir, name = write_cycle_config("config_training_calibration_loop", cycle, overrides)

    sp.run(
        f"itwinai exec-pipeline --config-dir {out_dir} --config-name {name}",
        shell=True,
        check=True,
    )

    keep = WD_RUN / "loop_train_multicycle" / "model_sequence"
    keep.mkdir(parents=True, exist_ok=True)
    shutil.copy(WD_RUN / "loop_train_multicycle" / "CudaLSTM.pt", keep / f"CudaLSTM_{cycle}.pt")


def calibrate(cycle: int, cfg: CycleConfig) -> Path:
    """Run dPL against the observations, returning the calibrated parameter file."""
    overrides = {
        "experiment_run": "cal_multicycle",
        "model_logger.TransferNN.load": cycle > 0,
        "scaling_use_cached": cycle > 0,
        "data_source.file.target_variables": str(OBS),
    }
    out_dir, name = write_cycle_config("config_calibration_loop", cycle, overrides)

    sp.run(
        f"itwinai exec-pipeline --config-dir {out_dir} --config-name {name}",
        shell=True,
        check=True,
    )

    keep = WD_RUN / "loop_cal_multicycle" / "model_sequence"
    keep.mkdir(parents=True, exist_ok=True)
    shutil.copy(WD_RUN / "loop_cal_multicycle" / "Hybrid.pt", keep / f"Hybrid_{cycle}.pt")

    src = WD_WFLOW / "run_default" / "inference_parameter.nc"
    dest = WD_CYCLE / f"cycle_{cycle}" / "theta_cal.nc"
    shutil.move(src, dest)
    return dest


# ==== EVALUATION


def member_waves(members: list, cfg: CycleConfig) -> Iterator[list[tuple[int, object]]]:
    """Split members into groups that run at the same time.

    Yields `[(index, member), ...]` per wave. With `wflow_parallel: 1` every
    wave holds one member, which is the sequential behaviour.
    """
    width = max(1, cfg.wflow_parallel)
    indexed = list(enumerate(members))
    for start in range(0, len(indexed), width):
        yield indexed[start:start + width]


def run_wflow_concurrently(jobs: list[tuple[Path, Path]], cfg: CycleConfig) -> None:
    """Run several wflow jobs at once, each on `cfg.threads_per_run()` threads.

    Threads, not processes: every job spends its life inside `subprocess.run`,
    which releases the GIL, and the real work happens in separate julia
    processes anyway.

    If any job fails after its retries, the others are still waited for before
    raising - leaving a julia process running against a half-written output is
    worse than the delay.
    """
    if len(jobs) == 1:
        run_wflow(jobs[0][0], jobs[0][1], cfg)
        return

    logger.info(
        f"running {len(jobs)} wflow jobs at once, "
        f"{cfg.threads_per_run()} threads each"
    )
    errors = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(run_wflow, s, o, cfg): o for s, o in jobs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as err:  # noqa: BLE001 - re-raised below
                errors.append((futures[future], err))

    if errors:
        for out_nc, err in errors:
            logger.error(f"wflow failed for {out_nc.name}: {err}")
        raise errors[0][1]


def prune_cycle_outputs(cycle: int, cfg: CycleConfig) -> None:
    """Delete one finished cycle's full-map wflow outputs.

    Called with a one-cycle lag, so the cycle just finished always survives:
    the loop can stop on `rel_tol` at any point, and which cycle turns out to
    be the last is not known until it is. Cycle 0 is never pruned.

    Only the `.nc` files go. The `.csv` holds gauge discharge, is small, and
    the scores themselves are in the state JSON either way.
    """
    if cycle < 1 or not cfg.prune_outputs:
        return

    run_dir = WD_WFLOW / "run_default"
    targets = [run_dir / f"output_cycle{cycle}_m{m}.nc"
               for m in range(cfg.members_for(cycle))]
    targets.append(run_dir / f"output_cycle{cycle}_cal.nc")

    freed = 0
    for path in targets:
        if path.exists():
            freed += path.stat().st_size
            path.unlink()

    if freed:
        logger.info(f"pruned cycle {cycle} outputs: {freed / 2**30:.1f} GB freed")


def score(output_nc: Path) -> dict:
    """RMSE and bias of the clean theta_cal run against the observations.

    Scored on the unperturbed parameters. The perturbed members exist to train
    the surrogate; they are not the calibration result and must not drive the
    stopping rule.
    """
    sim = xr.open_dataset(output_nc).sel(lat=slice(None, None, -1))
    sim = _unpack_layer(sim, "vwc")["vwc"]
    obs = xr.open_dataset(OBS)
    obs = obs[list(obs.data_vars)[0]]

    sim, obs = xr.align(sim, obs, join="inner")
    diff = (sim - obs).values
    valid = np.isfinite(diff)
    return {
        "rmse": float(np.sqrt(np.nanmean(diff[valid] ** 2))),
        "bias": float(np.nanmean(diff[valid])),
    }


def converged(history: list, rel_tol: float) -> bool:
    """Ahmad 2025: stop when RMSE or bias improves by less than 1 percent."""
    if len(history) < 2:
        return False
    prev, curr = history[-2], history[-1]
    for key in ("rmse", "bias"):
        denom = abs(prev[key])
        if denom > 0 and abs(prev[key] - curr[key]) / denom >= rel_tol:
            return False
    return True


# ==== DRIVER


def main() -> None:
    cfg = CycleConfig()
    state_path = WD_CYCLE / "state.json"
    state = CycleState.load(state_path)
    rng = np.random.default_rng(cfg.seed + state.cycle)

    theta_path = WD_WFLOW / "staticmaps.nc"

    while state.cycle < cfg.n_cycles:
        n = state.cycle
        cycle_dir = WD_CYCLE / f"cycle_{n}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"==== CYCLE {n}")

        # 1. perturbed members: theta_cal(n-1) at n > 0, the PTF map at n == 0
        theta = xr.open_dataset(theta_path).load() if n == 0 else read_theta(theta_path)
        members = build_members(n, cfg, theta, rng)

        # 2/3. wflow forward runs, in waves of `wflow_parallel`, each wave
        # ingested before the next starts. Ingesting per wave rather than at
        # the end keeps one run's worth of pool cells in memory instead of
        # every member's full map, and means an interrupted cycle still leaves
        # a usable archive.
        #
        # `wflow_parallel: 1` gives one member per wave, which is exactly the
        # sequential loop this replaced.
        for wave in member_waves(members, cfg):
            jobs = []
            for m, member in wave:
                static_nc = WD_WFLOW / f"staticmaps_cycle{n}_m{m}.nc"
                out_nc = WD_WFLOW / "run_default" / f"output_cycle{n}_m{m}.nc"
                write_staticmaps(member, static_nc)
                jobs.append((m, static_nc, out_nc))

            if not cfg.skip_wflow:
                run_wflow_concurrently([(s, o) for _, s, o in jobs], cfg)

            # Sequential, and in member order, so the archive's run indices do
            # not depend on which wflow run happened to finish first.
            for m, static_nc, out_nc in jobs:
                ingest_member(static_nc, out_nc, cycle=n, member=m, state=state)
                state.save(state_path)

        pool_archive.verify()

        # 4/5. retrain the surrogate on the whole archive, then calibrate
        if not cfg.skip_train:
            train_surrogate(n, cfg)
        if not cfg.skip_calibration:
            theta_path = calibrate(n, cfg)

        # 6. clean theta_cal run: the evaluation, and the next cycle's centre
        eval_static = WD_WFLOW / f"staticmaps_cycle{n}_cal.nc"
        eval_out = WD_WFLOW / "run_default" / f"output_cycle{n}_cal.nc"
        write_staticmaps(read_theta(theta_path), eval_static)
        if not cfg.skip_wflow:
            run_wflow(eval_static, eval_out, cfg)

        state.history.append(score(eval_out))
        logger.info(f"cycle {n}: {state.history[-1]}")

        state.cycle += 1
        state.save(state_path)

        # One cycle behind, so the most recent cycle is always still on disk
        # whenever the loop stops.
        prune_cycle_outputs(n - 1, cfg)

        if converged(state.history, cfg.rel_tol):
            logger.info(f"converged after cycle {n}")
            break

    logger.info(f"history: {json.dumps(state.history, indent=2)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("interrupted; rerun to resume from state.json")
        sys.exit(1)
