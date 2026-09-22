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
import re
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
from pool_archive import DYNAMIC_ARCHIVE, LAYER, STATIC_ARCHIVE, _unpack_layer

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

    # Ahmad 2025 sec 2.4.1 stopping rule. `converge_on` chooses what it looks
    # at, because there are two defensible answers:
    #
    #   "wflow"      the clean theta_cal run scored against the satellite.
    #                Ground truth, but it costs a wflow run per scored cycle
    #                and is only available on cycles `score_every` allows.
    #   "surrogate"  what calibration itself reported - surrogate(theta) vs
    #                satellite, already in `cal_metrics`, free, available every
    #                cycle. It is the quantity the search actually minimises,
    #                so its plateau is the optimiser's own convergence. It
    #                mixes parameter quality with surrogate quality, but across
    #                cycles those improve jointly, and both flattening is a
    #                reasonable definition of the loop having converged.
    #   "both"       stop only when they agree. The conservative choice.
    #
    # With "surrogate" the loop can stop early even at `score_every: 0`, so
    # ground truth is paid for only where it is wanted, not to decide when to
    # stop. Tsai's released code has no convergence test at all - a fixed
    # `nEpoch=500` - so this is not inherited from there.
    rel_tol: float = 0.01

    # Chosen 2026-09-20: "surrogate". Note this signal is noisier than the
    # wflow score and is available from cycle 1, so `rel_tol` can fire early on
    # two numbers that happened to land close together. `rel_tol: 0` disables
    # stopping entirely (any change clears the threshold), which is the way to
    # force a full-length run.
    converge_on: str = "surrogate"

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

    # A member's full-map output (~10 GB on the 2017-2022 window) has no
    # reader once `ingest_member` has copied its pool cells into the archive.
    # False deletes it right after ingestion, cycle 0 included; the clean
    # `theta_cal` runs and the baseline are not affected. Any member can be
    # recreated from its kept staticmaps with one wflow run.
    keep_member_outputs: bool = False

    # The clean `theta_cal` run costs a full wflow simulation - as much as a
    # training member - and yields only the numbers in `score()`. It feeds
    # nothing else: the next cycle's centre comes from `calibrate()`, not from
    # this run. So it is optional.
    #
    #   1  score every cycle (what `rel_tol` was designed around)
    #   k  score every k-th cycle  <- 2 by default: half the clean runs, and
    #      `rel_tol` still works, just checked on alternate cycles
    #   0  never score - no clean run at all
    #
    # With 0 there is no history, so `converged()` can never fire and the loop
    # always runs the full `n_cycles`. With k > 1 convergence is still checked,
    # but only against the cycles that were scored.
    score_every: int = 2

    # Score the first and last cycle whatever `score_every` says. With
    # `score_every: 0` this gives exactly two scores - where the campaign
    # started and where it ended - for two wflow runs instead of eight.
    # `converged()` then has nothing to compare until the very end, so the loop
    # always runs the full `n_cycles`: this trades early stopping for wflow
    # time, and is only worth it if you expect to use every cycle anyway.
    score_first_last: bool = False

    # Score the uncalibrated parameters once, as a reference for every cycle.
    # One wflow run for the whole campaign.
    score_baseline: bool = True

    # H12: the success metric is per-cell KGE - the loss, the score and the
    # stopping rule all use it. `kge_use_beta: False` drops the mean-ratio
    # term everywhere at once (loss, logged metric, wflow score), for when the
    # systematic RT0-wflow level difference should not drive the calibration.
    # `min_obs_per_cell`: a cell counts in a score only with at least this
    # many observations in the scored period (D1-D3 used 20).
    kge_use_beta: bool = True
    min_obs_per_cell: int = 20
    # Scaled KGE: weights on the (r, alpha, beta) terms, for loss, logged
    # metric and score alike. (1, 1, 1) is the standard KGE.
    # Chosen 2026-09-22 (user): (1, 0.25, 0.75). Alpha is kept but weak -
    # with full weight (step 3b) calibration bought larger swings with ~5 mm
    # roots - and beta counts less than timing.
    kge_weights: tuple = (1.0, 0.25, 0.75)

    # The parameter network (TransferNN). Untrained, it outputs ~0 in scaled
    # space for every parameter, i.e. every parameter starts at its lower
    # bound (checked 2026-09-22: KsatVer 1 mm/d, RootingDepth 5 mm).
    # `pretrain_transfer` first fits it to the a priori maps, so cycle 0
    # calibration starts from the a priori parameters instead. Later cycles
    # start from the previous cycle's network either way.
    # `transfer_bias` adds bias terms to its linear layers; the pretrained
    # and the loaded weights must have been made with the same setting.
    # Chosen 2026-09-22 (user): pretrain. Steps 3e/3f: no ~5 mm roots, f and
    # Sl nearer their a priori values, same score. Bias terms changed nothing.
    pretrain_transfer: bool = True
    transfer_bias: bool = False

    # Test hooks. All None/empty in a real run, so nothing below changes.
    # `wflow_starttime`/`wflow_endtime` shorten the simulation window for a
    # smoke test; `config_overrides` is merged into both the training and the
    # calibration config, which is how a test redirects `work_dir` away from
    # the production run folder.
    wflow_starttime: str | None = None
    wflow_endtime: str | None = None
    config_overrides: dict = field(default_factory=dict)

    def members_for(self, cycle: int) -> int:
        """How many perturbed wflow runs this cycle feeds the archive (A2 a)."""
        return self.n_members if cycle == 0 else self.n_members_later

    def scores_cycle(self, cycle: int) -> bool:
        """Whether this cycle runs the clean `theta_cal` evaluation."""
        if self.score_first_last and cycle in (0, self.n_cycles - 1):
            return True
        if self.score_every <= 0:
            return False
        return cycle % self.score_every == 0

    def threads_per_run(self) -> int:
        """Julia threads for one wflow run, dividing the total core budget."""
        return max(1, self.julia_threads // max(1, self.wflow_parallel))


@dataclass
class CycleState:
    """Persisted between stages so an interrupted run resumes cleanly."""

    cycle: int = 0
    archive_runs: int = 0  # wflow runs committed to BOTH stores
    history: list = field(default_factory=list)  # per-cycle {rmse, bias}

    # One wflow run for the whole campaign: the *uncalibrated* parameters,
    # scored the same way. Without it a cycle's rmse has nothing to be read
    # against, and the loop can converge neatly on a result worse than not
    # calibrating at all - `converged()` only compares consecutive cycles.
    baseline: dict = field(default_factory=dict)

    # What calibration itself reported, per cycle. This is surrogate(theta) vs
    # satellite, NOT wflow(theta) vs satellite, so it mixes parameter quality
    # with surrogate quality and cannot replace `history`. Kept because it
    # answers a question the wflow score cannot: when a cycle goes badly, was
    # it the parameters or the imitation? A near-zero `pearson` says the
    # surrogate is not usable yet.
    cal_metrics: list = field(default_factory=list)

    # Progress inside the current cycle, reset when it ends. Without these a
    # crash mid-cycle reruns the cycle from the top and ingests its members a
    # second time. Members ingest in order, so a count is enough.
    members_done: int = 0
    stages_done: list = field(default_factory=list)  # "train", "calibrate"

    # Cycles whose `theta_cal` has had its clean wflow run and score.
    scored_cycles: list = field(default_factory=list)
    # Set when the loop has stopped - converged or at `n_cycles` - so a rerun
    # after a crash in the final scoring does not start another cycle.
    finished: bool = False

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


def _fill_nearest_along_last(a: np.ndarray) -> np.ndarray:
    """Forward then backward fill NaNs along the last axis.

    The semantics of `DataArray.ffill().bfill()`: take the nearest valid value
    along the axis. xarray routes those through `bottleneck`, which is not
    installed here and is not worth a dependency for what turns out to be ~49
    cells - so the same thing in numpy.

    A slice that is entirely NaN stays NaN, exactly as ffill/bfill leave it.
    `read_theta` reports that case rather than inventing a value for it.
    """
    def forward(x):
        idx = np.where(~np.isnan(x), np.arange(x.shape[-1]), 0)
        np.maximum.accumulate(idx, axis=-1, out=idx)
        return np.take_along_axis(x, idx, axis=-1)

    out = forward(a)
    return forward(out[..., ::-1])[..., ::-1]


def read_theta(path: Path) -> xr.Dataset:
    """Calibrated parameters, gap-filled and renamed onto the staticmaps grid.

    dPL leaves gaps where predictors are missing - about 6.7% of land cells.
    Interpolate along a single axis so the result stays on the wflow grid,
    then fill whatever interpolation could not reach with the nearest valid
    value on that row.

    Measured on a real cycle-0 `theta_cal.nc`: `interpolate_na` leaves only 49
    of 343,226 land cells, and no latitude row is entirely empty, so every
    remaining gap is a row end with a genuine neighbour to copy.
    """
    ds = xr.open_dataset(path).load()
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})

    out = ds.copy()
    for name in ds.data_vars:
        interp = ds[name].interpolate_na(dim="longitude", method="linear")
        interp = interp.transpose(..., "longitude")
        filled = xr.DataArray(
            _fill_nearest_along_last(interp.values),
            dims=interp.dims, coords=interp.coords, name=name,
        )
        remaining = int(filled.isnull().sum())
        if remaining:
            # Only possible where a whole row is empty, which the measurement
            # above did not find. Loud, because a NaN reaching wflow is not.
            logger.warning(
                f"{name}: {remaining} cells still empty after filling - "
                f"an entire latitude row had no calibrated value"
            )
        out[name] = filled
    return out


def write_staticmaps(theta: xr.Dataset, dest: Path) -> None:
    """Write theta into a copy of the original staticmaps, preserving the mask."""
    base = xr.open_dataset(WD_WFLOW / "staticmaps.nc").load()
    nodata = base["thetaS"].isnull()

    for name in CAL_PARAMS:
        lo, hi = CAL_PARAMS[name]

        # Clamp to the declared physical bounds. The perturbed members never
        # need this - `perturb` works in normalised space and reflects back
        # inside [0, 1], so their bounds hold by construction - but the clean
        # `theta_cal` map comes straight from calibration and is not bounded by
        # anything. Measured on a real cycle 0: 6 cells of KsatVer below 1.0
        # (down to -2.6), 190 of Sl below 0.02, 38 of f and 5 of c above their
        # maxima. 0.07% of land, but a negative conductivity is not a small
        # error, it is an invalid one, and wflow would be handed it.
        outside = int(((theta[name] < lo) | (theta[name] > hi)).sum())
        if outside:
            logger.warning(
                f"{name}: {outside} cells outside [{lo}, {hi}], clamped"
            )

        filled = theta[name].clip(lo, hi).where(~nodata)
        if name in LAYERED_PARAMS:
            # At cycle 0 `theta` is the raw staticmaps, so a layered parameter
            # still carries all 4 layers and cannot be written into the single
            # layer slot below. From cycle 1 on it comes back from calibration
            # already flat, which is why this only ever bit at cycle 0.
            if "layer" in filled.dims:
                filled = filled.isel(layer=LAYER, drop=True)
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
    if cfg.wflow_starttime:
        data["starttime"] = cfg.wflow_starttime
    if cfg.wflow_endtime:
        data["endtime"] = cfg.wflow_endtime
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


def exec_pipeline(out_dir: Path, name: str) -> str:
    """Run one itwinai pipeline and return its combined output.

    Captured rather than streamed so metrics can be read back. The output is
    echoed and also kept next to the cycle's config, so nothing is lost.
    """
    proc = sp.run(
        f"itwinai exec-pipeline --config-dir {out_dir} --config-name {name}",
        shell=True, capture_output=True, text=True,
    )
    log = (proc.stdout or "") + (proc.stderr or "")
    print(log, end="")
    (out_dir / f"{name}.log").write_text(log)
    if proc.returncode != 0:
        raise sp.CalledProcessError(proc.returncode, "itwinai exec-pipeline")
    return log


def surrogate_weights() -> Path:
    """Where `train_surrogate` leaves the weights, and `calibrate` reads them.

    Read at call time, not import time, so a redirected `WD_RUN` (the smoke
    test) is followed.
    """
    return WD_RUN / "loop_train_multicycle" / "CudaLSTM.pt"


def transfer_weights() -> Path:
    """Where calibration reads and writes the parameter network's weights."""
    return WD_RUN / "loop_cal_multicycle" / "TransferNN.pt"


def pretrain_transfer(dest: Path, bias: bool = False, epochs: int = 100,
                      holdout: float = 0.1) -> dict:
    """Fit the parameter network to the a priori maps and save it at `dest`.

    Inputs and targets are scaled the way calibration scales them: the
    predictors by MinMax01 over the full map (the same numbers calibration
    computes - checked against its cached statistics when they exist), the
    parameters linearly between their `CAL_PARAMS` bounds. The loss is MSE
    in that scaled space, on every cell where both are defined. Returns the
    fit on `holdout` of the cells, per parameter.
    """
    import torch
    import yaml
    from hython.models import ModelLogAPI
    from hython.models.transferNN import TransferNN

    ccfg = OmegaConf.load(WD_CONFIG / "config_calibration_loop.yaml")
    files = ccfg.data_source.file
    preds = list(ccfg.static_inputs)
    params = list(ccfg.head_model_inputs.cal_param)
    for p in params:
        if tuple(ccfg.scaling_static_range[p]) != CAL_PARAMS[p]:
            raise ValueError(f"{p}: calibration scales by {ccfg.scaling_static_range[p]}, CAL_PARAMS says {CAL_PARAMS[p]}")

    xds = xr.open_zarr(files.static_inputs)[preds].load()
    lo = xds.min().compute()
    span = xds.max().compute() - lo
    cached = WD_RUN / "loop_cal_multicycle" / "static_inputs.yaml"
    if cached.exists():
        stats = yaml.load(cached.read_text(), Loader=yaml.Loader)
        c, s = (xr.Dataset.from_dict(stats[k]) for k in ("center", "scale"))
        for v in preds:
            if not (np.isclose(float(c[v]), float(lo[v])) and np.isclose(float(s[v]), float(span[v]))):
                raise ValueError(f"{v}: full-map min/max differ from calibration's cached statistics")
    x = np.stack([((xds[v] - lo[v]) / span[v]).values.ravel() for v in preds], -1)

    tds = xr.open_zarr(files.static_parameter_inputs)[params].load()
    y = np.stack([((tds[p] - CAL_PARAMS[p][0]) / (CAL_PARAMS[p][1] - CAL_PARAMS[p][0])).values.ravel()
                  for p in params], -1)

    ok = np.isfinite(x).all(-1) & np.isfinite(y).all(-1)
    x, y = x[ok].astype("float32"), y[ok].astype("float32")

    torch.manual_seed(int(ccfg.seed))
    rng = np.random.default_rng(int(ccfg.seed))
    order = rng.permutation(len(x))
    n_val = int(len(x) * holdout)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt, yt = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)
    tr, va = torch.from_numpy(order[n_val:]).to(dev), torch.from_numpy(order[:n_val]).to(dev)

    model = TransferNN(params, len(preds), ccfg.mt_output_dim, ccfg.mt_hidden_dim,
                       ccfg.mt_n_layers, bias=bias).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    loss_fn = torch.nn.MSELoss()
    best, best_state = np.inf, None
    for epoch in range(epochs):
        model.train()
        for b in tr[torch.randperm(len(tr), device=dev)].split(1024):
            opt.zero_grad()
            loss_fn(model(xt[b]), yt[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(xt[va]), yt[va]).item()
        sched.step(val)
        if val < best:
            best, best_state = val, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            logger.info(f"pretrain_transfer: epoch {epoch}, holdout MSE {val:.5f}")
    model.load_state_dict(best_state)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # saved the way calibration's trainer saves and loads it
    ccfg.model_logger.TransferNN.model_uri = str(dest)
    ModelLogAPI(ccfg).log_model("transfernn", model.cpu())

    with torch.no_grad():
        pv = model(xt[va].cpu()).numpy()
    yv = y[order[:n_val]]
    fit = {"bias": bias, "cells": int(len(x)), "holdout_mse": best}
    for i, p in enumerate(params):
        lo_p, hi_p = CAL_PARAMS[p]
        fit[p] = {
            "r2": float(1 - np.mean((pv[:, i] - yv[:, i]) ** 2) / np.var(yv[:, i])),
            "apriori_p50": float(lo_p + np.median(yv[:, i]) * (hi_p - lo_p)),
            "fit_p50": float(lo_p + np.median(pv[:, i]) * (hi_p - lo_p)),
        }
    logger.info(f"pretrain_transfer: saved {dest}: {fit}")
    return fit


def train_surrogate(cycle: int, cfg: CycleConfig) -> None:
    """Retrain on the full archive, warm-started from the previous cycle.

    Until 2026-09-22 the warm start never happened: `CudaLSTM.load` only names
    the file, and the trainer reads it only with `model_load_pretrained`.

    Scaler statistics are computed once at cycle 0 and cached thereafter. The
    vwc distribution shifts as theta moves, so refitting MinMax01Scaler every
    cycle would redefine what the frozen output head means while the weights are
    warm-started.
    """
    overrides = {
        "experiment_run": "train_multicycle",
        "model_logger.CudaLSTM.load": cycle > 0,
        # the key the trainer actually reads; `.load` alone only names the file
        "model_load_pretrained": cycle > 0,
        "model_logger.CudaLSTM.model_uri": str(surrogate_weights()),
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
    overrides.update(cfg.config_overrides)
    out_dir, name = write_cycle_config("config_training_calibration_loop", cycle, overrides)

    exec_pipeline(out_dir, name)

    keep = surrogate_weights().parent / "model_sequence"
    keep.mkdir(parents=True, exist_ok=True)
    shutil.copy(surrogate_weights(), keep / f"CudaLSTM_{cycle}.pt")


def _metric_index(template: str, target: str) -> int:
    """Position of the metric `target` in a template's `metric_fn.metrics`."""
    metrics = OmegaConf.load(WD_CONFIG / f"{template}.yaml").metric_fn.metrics
    for i, m in enumerate(metrics):
        if m["_target_"] == target:
            return i
    raise KeyError(f"{target} is not in {template}.metric_fn.metrics")


def calibrate(cycle: int, cfg: CycleConfig) -> tuple[Path, dict]:
    """Run dPL against the observations, returning the calibrated parameter file."""
    # The calibration config names the surrogate through the path of the
    # training *template* (`model_logger.CudaLSTM.model_uri`), and `load_model`
    # reads the weights path out of that file - with the template's own
    # `work_dir`. Any `work_dir` override then goes unseen: the 2026-09-21
    # smoke calibrated all five cycles against a stale production surrogate
    # from 2026-09-19. Point at the weights `train_surrogate` just wrote.
    weights = surrogate_weights()
    if not weights.exists():
        raise FileNotFoundError(f"no trained surrogate at {weights}")
    # `model_logger.TransferNN.load` only names the file; the trainer reads it
    # only when `mt_load_pretrained` is true. Until 2026-09-22 that key was
    # never set, so every cycle started the network from scratch.
    pretrained = cycle > 0
    if cycle == 0 and cfg.pretrain_transfer:
        fit = pretrain_transfer(transfer_weights(), bias=cfg.transfer_bias)
        (WD_CYCLE / f"cycle_{cycle}").mkdir(parents=True, exist_ok=True)
        (WD_CYCLE / f"cycle_{cycle}" / "transfer_pretrain.json").write_text(json.dumps(fit, indent=1))
        pretrained = True
    overrides = {
        "model_logger.CudaLSTM.model_uri": str(weights),
        "experiment_run": "cal_multicycle",
        # explicit for the same reason as the surrogate's: pretraining writes
        # here, and a redirected `WD_RUN` must be followed
        "model_logger.TransferNN.model_uri": str(transfer_weights()),
        "model_logger.TransferNN.load": pretrained,
        "mt_load_pretrained": pretrained,
        "mt_bias": cfg.transfer_bias,
        "scaling_use_cached": cycle > 0,
        "data_source.file.target_variables": str(OBS),
    }
    # H12: loss and logged metric use the same KGE as score()
    overrides["loss_fn.use_beta"] = cfg.kge_use_beta
    i = _metric_index("config_calibration_loop", "hython.metrics.KGECellMetric")
    overrides[f"metric_fn.metrics.{i}.use_beta"] = cfg.kge_use_beta
    overrides[f"metric_fn.metrics.{i}.min_obs"] = cfg.min_obs_per_cell
    overrides["loss_fn.weights"] = list(cfg.kge_weights)
    overrides[f"metric_fn.metrics.{i}.weights"] = list(cfg.kge_weights)
    # H10: the warm-up is the surrogate's seq_length; keep them together
    if "seq_length" in cfg.config_overrides:
        overrides["warmup_steps"] = cfg.config_overrides["seq_length"]
    overrides.update(cfg.config_overrides)
    out_dir, name = write_cycle_config("config_calibration_loop", cycle, overrides)

    log = exec_pipeline(out_dir, name)

    keep = WD_RUN / "loop_cal_multicycle" / "model_sequence"
    keep.mkdir(parents=True, exist_ok=True)
    shutil.copy(WD_RUN / "loop_cal_multicycle" / "Hybrid.pt", keep / f"Hybrid_{cycle}.pt")

    src = WD_WFLOW / "run_default" / "inference_parameter.nc"
    dest = WD_CYCLE / f"cycle_{cycle}" / "theta_cal.nc"
    shutil.move(src, dest)
    return dest, parse_cal_metrics(log)


def parse_cal_metrics(log: str) -> dict:
    """The last validation metrics calibration reported, from its own output.

    `ConsoleLogger: val_<target>_<metric>_epoch = <value>`, one line per metric
    per epoch; the last of each wins. Returns {} if the format ever changes -
    these are a diagnostic, and a parsing miss must not stop a cycle.
    """
    found = {}
    for line in log.splitlines():
        m = re.search(r"val_\w+?_(\w+)_epoch\s*=\s*([-\d.eE+]+)", line)
        if m:
            try:
                found[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return found


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


# What the wflow score and the two stopping signals track (H12). The
# validation period, because the calibration period alone can hide overfitting.
WFLOW_KEYS = ("kge_valid",)
# Calibration's logged `val_<target>_kgecell_epoch` (KGECellMetric)
SURROGATE_KEYS = ("kgecell",)


def score_periods(cfg: CycleConfig) -> dict:
    """The calibration's train/valid/test periods, overrides applied."""
    tpl = OmegaConf.load(WD_CONFIG / "config_calibration_loop.yaml")
    out = {}
    for name in ("train", "valid", "test"):
        key = f"{name}_temporal_range"
        out[name] = tuple(cfg.config_overrides.get(key, tpl[key]))
    return out


MASK_KEYS = ("data_source.file.target_variables_dynamic_mask", "target_dynamic_mask_months")


def score_mask(cfg: CycleConfig) -> tuple:
    """The calibration's dynamic RT0 mask and its months, overrides applied.

    Read from the calibration config so the score masks the same days as the
    loss. (None, None) when the config has no dynamic mask.
    """
    tpl = OmegaConf.load(WD_CONFIG / "config_calibration_loop.yaml")
    return tuple(cfg.config_overrides.get(k, OmegaConf.select(tpl, k)) for k in MASK_KEYS)


def score(output_nc: Path, cfg: CycleConfig) -> dict:
    """Per-cell KGE of a wflow run against the observations, per period (H12).

    For each of the calibration's periods: the median over cells of the
    per-cell KGE and of its parts r, alpha and beta, so a change can be traced
    to timing, variability or level; and how many cells counted (those with
    at least `min_obs_per_cell` observations in the period). A period outside
    the simulated window is left out. RMSE and bias over the whole window are
    kept for reference; they decide nothing. The observations are masked with
    the calibration's dynamic RT0 mask (`score_mask`), as the loss sees them.

    Scored on the unperturbed parameters. The perturbed members exist to train
    the surrogate; they are not the calibration result and must not drive the
    stopping rule.
    """
    from hython.metrics.custom import compute_kge_per_cell_np
    from hython.utils import apply_dynamic_mask

    sim = xr.open_dataset(output_nc).sel(lat=slice(None, None, -1))
    sim = _unpack_layer(sim, "vwc")["vwc"]
    obs = xr.open_dataset(OBS)
    obs = obs[list(obs.data_vars)[0]]
    mask_path, months = score_mask(cfg)
    if mask_path:
        obs = apply_dynamic_mask(obs, mask_path, months)

    sim, obs = xr.align(sim, obs, join="inner")
    diff = (sim - obs).values
    valid = np.isfinite(diff)
    out = {
        "rmse": float(np.sqrt(np.nanmean(diff[valid] ** 2))),
        "bias": float(np.nanmean(diff[valid])),
    }

    for name, (start, end) in score_periods(cfg).items():
        s = sim.sel(time=slice(start, end))
        if s.sizes["time"] == 0:
            continue
        o = obs.sel(time=slice(start, end))
        n_t = s.sizes["time"]
        parts = compute_kge_per_cell_np(
            o.transpose(..., "time").values.reshape(-1, n_t),
            s.transpose(..., "time").values.reshape(-1, n_t),
            min_obs=cfg.min_obs_per_cell,
            use_beta=cfg.kge_use_beta,
            weights=cfg.kge_weights,
        )
        counted = np.isfinite(parts["kge"])
        out[f"cells_{name}"] = int(counted.sum())
        if counted.any():
            for part in ("kge", "r", "alpha", "beta"):
                out[f"{part}_{name}"] = float(np.nanmedian(parts[part][counted]))
    return out


def score_baseline(cfg: CycleConfig) -> dict:
    """Score the *uncalibrated* parameters, once, the same way a cycle is scored.

    One wflow run for the whole campaign - the default parameters never change.
    Warm-started and windowed exactly like a cycle's clean run, so the only
    difference between this and any cycle's score is the parameters themselves.
    """
    out_nc = WD_WFLOW / "run_default" / "output_baseline_apriori.nc"
    static = WD_WFLOW / "staticmaps.nc"

    if not cfg.skip_wflow:
        logger.info("baseline: scoring the uncalibrated parameters (one run)")
        run_wflow(static, out_nc, cfg)

    result = score(out_nc, cfg)
    logger.info(f"baseline (uncalibrated): {result}")
    return result


def relative_to_baseline(current: dict, baseline: dict) -> dict:
    """How a cycle compares with not calibrating at all.

    `converged()` only looks at consecutive cycles, so without this the loop
    can settle on a result worse than the starting point and report success.

    Decided on the validation per-cell KGE (H12); None when either side has
    no validation score. The KGE differences are absolute (KGE can be
    negative, so a percentage is meaningless); RMSE and bias stay as
    percentages, for reference.
    """
    if not baseline:
        return {}
    out = {}
    for key in ("rmse", "bias"):
        if key in current and key in baseline and baseline[key]:
            out[f"{key}_vs_baseline_pct"] = round(
                100.0 * (abs(baseline[key]) - abs(current[key])) / abs(baseline[key]), 2
            )
    for name in ("train", "valid", "test"):
        key = f"kge_{name}"
        if key in current and key in baseline:
            out[f"{key}_vs_baseline"] = round(current[key] - baseline[key], 4)
    key = WFLOW_KEYS[0]
    out["better_than_uncalibrated"] = (
        bool(current[key] >= baseline[key]) if key in current and key in baseline else None
    )
    return out


def converged(history: list, rel_tol: float, keys=WFLOW_KEYS) -> bool:
    """Ahmad 2025: stop when every tracked quantity improves by less than
    `rel_tol`. Needs two entries; a missing key is skipped rather than assumed
    converged, and if none of the keys are present the answer is False."""
    if len(history) < 2:
        return False
    prev, curr = history[-2], history[-1]
    seen = False
    for key in keys:
        if key not in prev or key not in curr:
            continue
        seen = True
        denom = abs(prev[key])
        if denom > 0 and abs(prev[key] - curr[key]) / denom >= rel_tol:
            return False
    return seen


def has_converged(state: "CycleState", cfg: CycleConfig) -> bool:
    """Whether the loop should stop, by whichever signal `converge_on` names."""
    wflow = converged(state.history, cfg.rel_tol, keys=WFLOW_KEYS)
    surrogate = converged(state.cal_metrics, cfg.rel_tol, keys=SURROGATE_KEYS)

    if cfg.converge_on == "surrogate":
        return surrogate
    if cfg.converge_on == "both":
        return wflow and surrogate
    if cfg.converge_on != "wflow":
        raise ValueError(
            f"converge_on must be 'wflow', 'surrogate' or 'both', "
            f"got {cfg.converge_on!r}"
        )
    return wflow


# ==== DRIVER


def score_cycle(n: int, theta_path: Path, state: "CycleState", cfg: CycleConfig,
                state_path: Path) -> dict:
    """Clean wflow run with cycle `n`'s calibrated parameters, scored.

    Scores the uncalibrated baseline first if that has not been done.
    """
    eval_static = WD_WFLOW / f"staticmaps_cycle{n}_cal.nc"
    eval_out = WD_WFLOW / "run_default" / f"output_cycle{n}_cal.nc"
    write_staticmaps(read_theta(theta_path), eval_static)
    if not cfg.skip_wflow:
        run_wflow(eval_static, eval_out, cfg)

    if cfg.score_baseline and not state.baseline:
        state.baseline = score_baseline(cfg)

    result = score(eval_out, cfg)
    state.history.append(result)
    state.scored_cycles.append(n)
    state.save(state_path)
    rel = relative_to_baseline(result, state.baseline)
    logger.info(f"cycle {n}: {result}{(' | vs uncalibrated: ' + str(rel)) if rel else ''}")
    if rel and rel["better_than_uncalibrated"] is False:
        key = WFLOW_KEYS[0]
        logger.warning(
            f"cycle {n} is WORSE than not calibrating "
            f"({key} {result[key]:.4f} vs baseline "
            f"{state.baseline[key]:.4f})"
        )
    return result


def main(cfg: CycleConfig | None = None) -> None:
    cfg = CycleConfig() if cfg is None else cfg
    state_path = WD_CYCLE / "state.json"
    state = CycleState.load(state_path)
    rng = np.random.default_rng(cfg.seed + state.cycle)

    theta_path = WD_WFLOW / "staticmaps.nc"
    if state.cycle > 0:
        # Resuming: centre on the last calibration, not on the PTF map.
        prev = WD_CYCLE / f"cycle_{state.cycle - 1}" / "theta_cal.nc"
        if prev.exists():
            theta_path = prev

    while not state.finished and state.cycle < cfg.n_cycles:
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
                if m < state.members_done:
                    continue  # ingested before a crash; see CycleState
                static_nc = WD_WFLOW / f"staticmaps_cycle{n}_m{m}.nc"
                out_nc = WD_WFLOW / "run_default" / f"output_cycle{n}_m{m}.nc"
                write_staticmaps(member, static_nc)
                jobs.append((m, static_nc, out_nc))

            if not jobs:
                continue
            if not cfg.skip_wflow:
                run_wflow_concurrently([(s, o) for _, s, o in jobs], cfg)

            # Sequential, and in member order, so the archive's run indices do
            # not depend on which wflow run happened to finish first.
            for m, static_nc, out_nc in jobs:
                ingest_member(static_nc, out_nc, cycle=n, member=m, state=state)
                state.members_done = m + 1
                state.save(state_path)
                if not cfg.keep_member_outputs and out_nc.exists():
                    size = out_nc.stat().st_size
                    out_nc.unlink()
                    logger.info(f"cycle {n} member {m}: output deleted after ingestion "
                                f"({size / 2**30:.1f} GB)")

        pool_archive.verify()

        # 4/5. retrain the surrogate on the whole archive, then calibrate
        if "train" in state.stages_done:
            logger.info(f"cycle {n}: surrogate already trained, skipping")
        elif not cfg.skip_train:
            train_surrogate(n, cfg)
            state.stages_done.append("train")
            state.save(state_path)
        if "calibrate" in state.stages_done:
            logger.info(f"cycle {n}: already calibrated, skipping")
            theta_path = WD_CYCLE / f"cycle_{n}" / "theta_cal.nc"
        elif not cfg.skip_calibration:
            theta_path, cal_metrics = calibrate(n, cfg)
            state.cal_metrics.append({"cycle": n} | cal_metrics)
            state.stages_done.append("calibrate")
            state.save(state_path)
            if cal_metrics:
                logger.info(f"cycle {n} calibration reported: {cal_metrics}")

        # 6. clean theta_cal run: the evaluation, and nothing else. The next
        # cycle's centre comes from `calibrate()` above, not from here, so
        # skipping this costs only the score. The cycle the loop stops at is
        # always scored, below.
        if cfg.scores_cycle(n) and n not in state.scored_cycles:
            score_cycle(n, theta_path, state, cfg, state_path)
        else:
            logger.info(
                f"cycle {n}: not scored (score_every={cfg.score_every}); "
                f"one wflow run saved"
            )

        state.cycle += 1
        state.members_done = 0
        state.stages_done = []
        state.save(state_path)

        # One cycle behind, so the most recent cycle is always still on disk
        # whenever the loop stops.
        prune_cycle_outputs(n - 1, cfg)

        if has_converged(state, cfg):
            logger.info(f"converged after cycle {n} (on {cfg.converge_on})")
            break

    state.finished = True
    state.save(state_path)

    # The last cycle's parameters are the result, so they always get a clean
    # wflow run - also when the loop stopped at a cycle `score_every` skips.
    last = state.cycle - 1
    last_theta = WD_CYCLE / f"cycle_{last}" / "theta_cal.nc"
    if last >= 0 and last not in state.scored_cycles and last_theta.exists():
        logger.info(f"final: scoring the last cycle ({last})")
        score_cycle(last, last_theta, state, cfg, state_path)

    logger.info(f"history: {json.dumps(state.history, indent=2)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("interrupted; rerun to resume from state.json")
        sys.exit(1)
