# Multi-cycle dPL — what to change before running it

What must change before `run_dpl_cycle.py` can run more than one cycle.
**A1 is built** (`scripts/pool_archive.py`); everything else is still to do.

Two places need work. **A1 and A2** are in `run_dpl_cycle.py`. **H1 to H7** are
in `hython` (`/home/iferrario/dev/hython`). A1 came first because it decides
what the data files look like, and the hython changes are written against
that.

---

## Next steps (written 2026-09-21, after the overnight smoke)

The 5-cycle smoke ran end to end, but its surrogate was too weak (100-300
cells per run) and calibration made wflow worse than not calibrating (RMSE
0.116-0.119 against 0.101 uncalibrated). A training-only test on the smoke
archive showed more cells double the surrogate's skill (H9). The work below
comes out of that. Do it in this order.

1. **Code, in hython.**
   - *NumPy sample-index build* (H9, prerequisite). Replace the Python tuple
     list in `WflowSBM_Pool.build_sample_index` with `np.repeat`/`np.tile`;
     test that the index is identical to the old one.
   - *Temporal samplers* (H8). Validation draws its days once, with its own
     row target and day fraction; training redraws each epoch from a seeded
     generator; no global `np.random.seed`; tests. This goes in with, or before,
     the larger budget: early stopping is only as good as the validation loss
     it reads.
2. **Config** (H9). `train_rows_target: 22000`, `frac_time: 0.1`, the new
   validation budget from H8. Keep `epochs: 100` and
   `early_stopping_patience: 20` - longer training is accepted because early
   stopping ends it.
3. **Check on the smoke archive** - no new wflow runs except one.
   - Train once with the new code and config; score with
     `smoke_runs/rows_test/eval_rows_test.py`. Confirms the fixes, and gives
     the real seconds per epoch.
   - Calibrate against that surrogate, then one clean wflow run to score it
     against the 0.101 baseline (~40 min). Shows whether a better surrogate
     fixes calibration before hours of production wflow time go in.
4. **Production.** Rebuild the archive from a real cycle 0. The old stores are
   renamed aside (`emo1_*_cycle_old.zarr`) and are not to be reused.

**Waiting on a decision, not blocking 1-3:**

- `compute_nse` (`hython/metrics/custom.py:127`) subtracts `y_pred.mean()`
  where NSE needs the mean of the observations. It makes the logged NSE a
  little high. Logged NSE/KGE are also pooled over the whole batch, not per
  cell.
- Turning off the clean/baseline wflow scoring runs at cycle 0 - deferred by
  the user. `smoke_multicycle.py` has no flag for `score_baseline`.

**Worth a look later:**

- Static ingest takes ~210 s per run against ~12 s for dynamic, although the
  static store grows by ~1 MB per run.
- Calibration takes 12-21 min per cycle (~1 min per epoch), now the second
  cost after wflow. Its validation loss was flat from epoch 1 in every cycle.
- Surrogate sensitivity slope is ~0.5 at best: it recovers about half of
  theta's effect on vwc, which weakens the gradient calibration gets.
- Calibration logs repeat `OSError: Directory not empty: /tmp/pymp-*` - the
  24 loader workers' temp dirs at shutdown. Harmless noise.

---

## Checklist

### 1. Build the archive — A1
No wflow run needed, no hython change. `staticmaps.nc` and
`run_default/output_calib_0.nc` already exist to test against.

Written as `scripts/pool_archive.py`. Nothing already in
`/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_input/` was touched: the
two new stores have new names and `--build` refuses to overwrite.

- [x] Decide the pool seed. The pool itself is **not** saved anywhere: it is
      reproducible from the seed, and each row carries its grid indices
      — seed `20250919`, `pool_size` 16800, both in the store attrs
- [x] Draw the pool: plain random, fixed seed, over the 336,466 unmasked cells
      — 16,800 cells (5%), rounded to divide the 2100-cell zarr chunks
- [x] Cut `staticmaps.nc` + `output_calib_0.nc` down to the pool, write run 0
- [x] Put the forcing in the **same** store as the target, repeated per run
      — `emo1_dynamic_cycle.zarr` holds `precip, pet, temp, vwc`

Then test the cycle-1 append. Run 1 = run 0 with `KsatVer` times a known
number, appended to **both** stores (this doubles as H1's test). Three things
to report on that append:

- [x] **Computation** — cut 305-309 s per run, append 2.5 s, so ~310 s a run.
      Almost all of it is reading the 10 GB wflow netCDF; the zarr write is
      seconds. Against wflow's own 54 min, ingestion adds ~10%. **Not yet in
      the A2 budget**
- [x] **Size** — static 1.1 M, dynamic 391 M per run. **~392 MB per
      run** as zarr v2, not the 147 MB the summary predicted: that figure
      assumed the forcing was stored once, and it is now repeated per run.
      4.3 GB at 11 runs
- [x] **Correctness** — `python pool_archive.py --verify` asserts all of it:
      rows = 2 x `pool_size`; the pool redraws from the stored seed onto the
      same cells; `lat`/`lon`/`lat_i`/`lon_i` agree with the map in both
      stores; `run`/`cycle`/`member` match the JSON manifest; the forcing is
      identical in every run; `cell` carries no coordinate or index. And
      `--append-test` checks run 1's `KsatVer` is exactly run 0's times the
      factor with every other static unchanged

Also done, not originally on this list:

- [x] `run`/`cycle`/`member` per-cell coords and a JSON run manifest, so a row
      can be traced to the wflow run that made it — `cycle` and `member` are
      not derivable from position once A2 varies runs per cycle
- [x] `to_xarray` / `scatter_to_map` in `pool_archive.py`, and the one-line
      `create_xarray_data` fix this needs, recorded under H2

- [x] Wire `run_dpl_cycle.py` to `pool_archive.ingest_run`. `prepare_member`,
      `append_members`, `assert_archives_aligned` and `_masks` are gone,
      replaced by one `ingest_member`; the docstring, the constants and the
      `train_surrogate` overrides all follow the new layout. Ingest now runs
      inside the member loop, so peak memory is one run's pool cells instead
      of every member's full map, and `state.save()` runs per member
- [x] Dry run over three runs across two cycles: provenance correct, every
      check green. It found two real bugs, both fixed — see H5

**Still open in A1's scope:**

- [ ] Rebuild from a real cycle 0. Run 1 in the archive today is the synthetic
      `KsatVer x 3` append test — the user is deleting both stores; the wired
      orchestrator rebuilds them by itself
- [x] **The real H1 test passed on 2026-09-20.** A two-run scratch archive was
      built from `emo1_static.zarr` + `emo1_dynamic.zarr` (run 0, apriori) and
      `review/P_KsatVer_*` (run 1, `KsatVer x 0.195`, every other static
      bit-identical). Trained 4 epochs on held-in cells, then compared the same
      held-out cell in both runs:

      | | predicted gap | true gap | r |
      |---|---|---|---|
      | untrained | -0.0011 | -0.0369 | +0.58 |
      | after 4 epochs | **-0.0430** | -0.0369 | **+0.82** |

      Sign agreement on individual cells 93%; 117% of the true mean gap
      recovered, so the response slightly overshoots. The untrained r is
      already +0.58 because `KsatVer` is the only input that differs, so even a
      random network orders the pairs - the evidence is the *magnitude*, which
      goes from ~0 to the right size. 19 s on the A100.

- [ ] (superseded, kept for the record) The real H1 test. **No new wflow run is
      needed** — an earlier draft said
      54 min for a fresh `KsatVer x 3` run, but
      `/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/review/` already holds
      seven matched (staticmaps, output) pairs from real wflow runs, over the
      same forcing and the same period (2016-01-02 to 2022-12-31):

      | pair | what moves |
      |---|---|
      | `apriori_*` | baseline |
      | `P_KsatVer_*` | `KsatVer x 0.2`, everything else bit-identical |
      | `P_c_*`, `P_f_*`, `P_Sl_*`, `P_RootingDepth_*` | one parameter each |
      | `cal_*` | the calibrated set |

      `apriori` + `P_KsatVer` is a cleaner H1 test than the one proposed here,
      because it is a controlled single-parameter perturbation rather than a
      whole-field rescale. Confirmed: `c`, `f`, `RootingDepth`, `Sl`, `thetaS`
      and `SoilThickness` are identical between the two staticmaps; the outputs
      hold `vwc` on 4 layers, 2556 days each.

      The forcing needs no work either. `cut_dynamic` reads it from
      `emo1_dynamic.zarr`, not from the output netCDF, and that store spans
      2000-01-01 to 2022-12-31. `emo1/forcings.nc` is the same 8401 steps if a
      raw source is ever wanted instead.

      So the real H1 test is an ingestion job over files that already exist.

### 2. Scaling numbers — H4
Needed before the first calibration, not before the first training.

- [x] Work out the five `MinMax01` numbers on the full map.
      `pool_archive.py --freeze-stats` writes them from `emo1_static.zarr`:
      `wflow_uparea` 67409, `wflow_landuse` 521, `wflow_dem` 4545.2,
      `Slope` 144.567, `WaterFrac` 0.904794.
- [x] Save them where both configs can name the same path.
      `scaling_frozen_stats` in both configs, pointing at
      `surrogate_input/scaling_frozen_fullmap.yaml`. `Scaler.apply_frozen`
      overrides by **variable name**, so one file serves `static_inputs` in
      training and `head_model_inputs.aux_feat` in calibration although they
      are different groups in different files.
- [x] `scaling_use_cached` works at all. It never did: `data.py:59` passed it
      as `Scaler.__init__`'s second positional, which is `is_train`, so
      `use_cached` stayed False for ever and `run_dpl_cycle.py`'s
      `cycle > 0` override was inert. Now passed by keyword. The template stays
      `false` on purpose - cycle 0 has nothing to load - and the per-cycle
      override does the rest.
- [x] Make `load_or_compute` raise instead of quietly recomputing. Also
      `Scaler.load` itself, which did not raise either: the `raise
      FileNotFoundError()` was commented out, so a missing cache left
      `self.archive` empty and failed somewhere far away. Groups with no
      scaler (`target_variables: null`) are skipped rather than demanded.
- [x] Stop the scaler writing statistics into the working directory. A bare
      `except:` in `Scaler.__init__` fell back to `Path(".")` silently; that is
      how a config without `work_dir` overwrote the tracked
      `static_inputs.yaml` in the repo root during this work. Still falls back,
      but logs a warning naming the directory.

### 3. The hython dataset — H1, H2, H3
These land together. Each is incomplete alone.

**H1 and H2 are done, in a new class.** The H2 section below names
`wflow_sbm.py` line numbers inside `WflowSBM_HPC`, but converting that class in
place would have broken the full-map training that still uses it. The pool
rewrite is `WflowSBM_Pool` instead; `WflowSBM_HPC` is untouched. The two cannot
share a path - the sample axis, the scaler axes and the reshape order all
differ.

Also done, not in this list: `to_xarray`/`scatter_to_map` moved out of
`pool_archive.py` into `hython.utils` as `pool_to_xarray`/`scatter_pool_to_map`.

Two things the H2 section below gets wrong, left in place as a record:
- `crs=None` applies only to the cell path. `inference.py:94` is a real lat/lon
  grid, so its `crs=4326` is correct and was not changed.
- it says `pool_archive.to_xarray`/`scatter_to_map` are "tested against the
  archive". They had no tests. They do now, and writing them found a bug:
  scattering rows from several runs kept only the last run and silently dropped
  the others, because every run repeats the same `lat_i`/`lon_i`.

- [x] New downsampler that knows `pool_size` and picks base cells before
      expanding to runs (H3 — `RandomDownsampler` cannot do this).
      `PoolDownsampler`, with tests.
- [x] Fixed train/valid base-cell split, drawn once (H3). Derived from
      `split_seed`, not written to disk - the per-cycle config *is* the
      persistence, and `tests/test_cycle_config.py` asserts `split_seed`,
      `valid_frac`, `pool_size` and `split` are identical in all 8 generated
      cycle configs. Verified disjoint on the real archive too.
- [x] Sampler that draws fresh cells every epoch (H3). Fires now, and
      measured on the real two-run archive: 800 cells an epoch, only ~50 shared
      with epoch 0, **2,913 distinct cells over 4 epochs**, row count constant
      at 1,600, train/valid overlap 0 throughout. This is the behaviour that
      makes storing the whole pool worth it.
- [x] Point the training config at `PoolDownsampler` (H3). Both blocks, with
      `pool_size`/`train_rows_target` as top-level keys and `runs` overridden
      per cycle by `run_dpl_cycle.py`.
- [x] Call `set_epoch` from the training loop (H3). `trainer.py:220` now
      forwards to `train_loader.dataset` and `val_loader.dataset` **outside**
      the `is_distributed` branch - that branch never runs on one GPU, which is
      why nothing fired before. Tested with stub loaders **and confirmed in a
      real `itwinai exec-pipeline` run** (38 s, 2 epochs, scratch archive): the
      probe recorded `train: 2 calls, 2 distinct cell sets` and `valid: 2
      calls, 1 distinct cell set` - training resamples, validation does not.
- [x] Read the stacked layout off the `cell` axis (H1)
- [x] Index cells directly; keep the whole pool in memory (~2 GB) instead of
      only the drawn rows (H2)
- [x] Let `test` read the pool; stop it rebuilding a full map (H2)
- [x] Add `"cell"` to the dim whitelist in `create_xarray_data` (H2)
- [x] Rename scaler axes to `("cell","time")` and `("cell",)` (H4). Done in
      `WflowSBM_Pool`; the lat/lon classes keep their own axes.

### 4. Safety
- [x] H5 — assert the files line up. `WflowSBM_Pool.check_alignment`, called
      from `__init__`, with `tests/test_alignment.py` corrupting an archive in
      each way it can go wrong. Checks: equal cell counts across the three
      stores, `n % pool_size == 0`, the `run` coordinate is `n_runs`
      contiguous blocks, `lat`/`lon` agree row-for-row between the static and
      dynamic stores, and every run lists the same base cells in the same
      order.

      **The plan's snippet above does not apply.** It compares `xs.lat`
      against `np.tile(xd.lat, runs)`, but the forcing is repeated per run in
      the archive, so `xd` already carries all `runs x pool_size` rows and the
      two arrays compare directly. Verified against the real archive.
- [ ] H6 — give `RandomDownsampler` its own random generator. **Deliberately
      deferred** (decided 2026-09-20), not an oversight: it is a legacy
      problem, and nothing in the multicycle path reaches it. Train and valid
      are `PoolDownsampler`, `test_downsampler` is null, and
      `dynamic_downsampler` is a plain dict read only by `WflowSBM`
      (`wflow_sbm.py:553,565`), never by `WflowSBM_Pool`. So
      `RandomDownsampler` is never instantiated by a multicycle run and its
      global `np.random.seed` cannot perturb one.

      Calibration does not reach it either, though for a different reason:
      `config_calibration_loop.yaml:80-83` sets all four downsamplers to
      `null`, and `WflowSBMCal` guards with `if self.downsampler is not None`.
      `RandomDownsampler` currently appears in **no config**.

      **Trip-wire, currently dormant.** The one thing that would wake H6 up is
      putting a downsampler into the calibration config. A2(d) would have done
      that, and it was **dropped** on 2026-09-20 partly for this reason. If
      anyone revives it, or adds sampling to calibration another way, do H6
      first or use `PoolDownsampler` there - otherwise the global
      `np.random.seed` starts running inside the loop.

      The two `xfail(strict=True)` tests in `tests/test_downsampler.py` stay as
      they are. They state the behaviour H6 owes and will turn into failures
      the moment it lands, which is the prompt to drop the marker.

### 5. Speed — A2, orchestrator only, any time
- [x] 4 runs at cycle 0, then 1 per cycle (a). `CycleConfig.members_for`,
      with `n_members: 4` and `n_members_later: 1`. Over 8 cycles that is 11
      archive runs instead of 32.
- [x] Floor `jitter_end` at 0.05-0.08, add a retry to `run_wflow` (a).
      `jitter_end` is 0.06. `run_wflow` retries `wflow_retries: 2` times and
      then still raises - retrying must not turn a real failure into a silent
      one, or the cycle would ingest a missing or stale output.
- [x] Expose `train_rows_target`; `places = rows // runs` (b2). Landed with the
      H3 wiring: top-level `train_rows_target: 6730` in the training config,
      and `PoolDownsampler._places` divides it by `runs`. Measured at 4 runs:
      `places=1682`, 6,728 rows, against the plan's predicted 1,683.
- [~] Fewer epochs after cycle 0 (c). **Superseded by H7**, which landed
      2026-09-20: each cycle now runs as long as it keeps improving rather than
      for a guessed number of epochs. Nothing to do unless early stopping turns
      out to let cycles run too long in practice.
- [~] Sample calibration cells in the middle cycles (d). **Dropped on
      2026-09-20** - considered and rejected, not forgotten. See (d) below.
- [x] Time wflow at 12 threads against 24 (f). **Measured 2026-09-20**, and
      the answer is not what this section assumed.

      | threads | 30-day run | | | 1-year run |
      |---|---|---|---|---|
      | 1 | 86.4 s | | 12 | 8m42s |
      | 2 | 55.1 s | | 24 | 8m30s |
      | 4 | 38.6 s | | | |
      | 6 | 32.2 s | | | |
      | 8 | 28.9 s | | | |
      | 12 | 28.3 s | | | |

      **One run cannot use 24 threads.** It plateaus around 8; 12 to 24 buys
      2.3%. Asking for more threads would not help a single run at all.

      **So run 3-4 wflow runs in parallel at 6-8 threads each**, not one at 24.
      Six threads is 14% slower than twelve on half the cores, so four
      concurrent runs give roughly 3.5x the throughput on the same 24 cores.
      With `n_members_later: 1` this only bites at cycle 0, where it turns four
      sequential runs into one wave.

      **Two things this ruled out.** CEPH is *not* the bottleneck: the same run
      writing to local disk finished within a tenth of a second of the CEPH
      one, with and without compression. And `compressionlevel = 5` costs 20%
      of the simulation time for nothing - level 1 gives 131 MB against 128 MB
      for 30 days, and is 8.5% faster than level 5. The rest is wflow's own
      serial computation.

      **Then measured, and it does not.** A smoke run with two concurrent
      1-year runs at 12 threads each took **16m32s and 17m10s**, against
      **8m42s** for the same run alone. Two at once is ~17 min of wall clock
      for two runs; sequentially it is ~17.4 min. **No benefit.**

      So the earlier reasoning here was wrong. "Nothing is saturated" did not
      follow from flat thread scaling - flat scaling meant the bottleneck was
      never the cores, and two processes contend for that same bottleneck
      (memory bandwidth, most likely). Keep `wflow_parallel: 1`.
- [x] Make parallel wflow a choice, not a rewrite (f). Measurement says leave
      it at 1 - see above - but the switch exists and works. `wflow_parallel`
      (default **1**, sequential) and `julia_threads` as a **total** core
      budget split by `threads_per_run()`. Members run in waves, each wave
      ingested before the next starts, so memory still holds one run at a time
      and an interrupted cycle still leaves a usable archive. At
      `wflow_parallel: 1` a wave is one member - the sequential loop exactly.
      Ingestion stays sequential and in member order, so archive run indices do
      not depend on which wflow finished first.

- [x] Choose what convergence is judged on. `CycleConfig.converge_on`:
      `"wflow"` (default, the clean run scored against satellite - ground
      truth, but costs a simulation per scored cycle), `"surrogate"` (what
      calibration itself reported, already in `cal_metrics`, free and available
      every cycle), or `"both"`.

      The surrogate signal is the quantity the search actually minimises, so
      its plateau is the optimiser's own convergence. It mixes parameter
      quality with surrogate quality, but across cycles those improve jointly.
      With `"surrogate"` the loop can stop early even at `score_every: 0`, so
      ground truth is paid for only where it is wanted rather than to decide
      when to stop.

      **Checked against the source:** Tsai's released code
      (`.../tsai/model_code/hydroDL/model/train.py`) has **no convergence test
      at all** - `for iEpoch in range(1, nEpoch + 1)` with `nEpoch=500`, and no
      `tol`/`early`/`stop` anywhere - and no outer round driver. So `rel_tol`
      is inherited from Ahmad 2025 sec 2.4.1 alone, and which quantity Ahmad
      stops on was not verifiable from here.
- [x] Make the clean `theta_cal` run optional. `CycleConfig.score_every`:
      1 every cycle (default), k every k-th, 0 never. It costs a **full wflow
      simulation per cycle** - as much as a training member - and yields only
      `score()`'s two numbers, after which `prune_cycle_outputs` deletes the
      ~10 GB output. Over 8 cycles that is 8 wflow runs, about half the whole
      wflow budget, spent on measurement rather than learning.

      **It feeds nothing else.** The code comment claimed it was "the next
      cycle's centre"; it is not - `theta` for cycle n+1 comes from
      `calibrate()` (`run_dpl_cycle.py:681`), not from this run. Corrected.

      The cost of switching it off is the stopping rule: `converged()` needs
      two scores, so with `score_every: 0` there is no history and the loop
      always runs the full `n_cycles`. `score_every: 2` keeps both - a real
      uncontaminated score and half the runs. Scoring a *member* instead was
      considered and rejected: its parameters carry deliberate per-cell jitter,
      so the stopping rule would be reacting to noise that was added on
      purpose.
- [x] Lower `compressionlevel` from 5 to 1 (f). `CycleConfig.output_compression`,
      applied to the per-run TOML `run_wflow` already writes - **never to the
      shared template**, which every future run reads. 8.5% off each run for
      2.3% more disk.
- [x] **Delete wflow outputs after ingestion.** `prune_cycle_outputs`, called
      with a **one-cycle lag** - the loop can stop on `rel_tol` at any point, so
      which cycle turns out to be the last is not known until it is, and
      pruning one behind means the most recent cycle is always still on disk.
      Cycle 0 is never pruned. Only `.nc` files go; the `.csv` is small and
      holds gauge discharge, and the scores live in the state JSON.

      Decision (2026-09-20): keep the scores plus cycle 0 and the last cycle,
      **not** all 8 clean `theta_cal` outputs. ~190 GB becomes ~20 GB. Switch
      off with `prune_outputs: false`.
- [x] Update `run_dpl_cycle.py` docstring. Now states the per-cycle run
      schedule, the measured ~392 MB a run rather than the old 256 MB estimate,
      that training stops on its own (H7), and that H1-H5 and H7 have landed.
      (was: still describes the old
      `cycle` dimension

### 6. Worth doing, not required
- [x] H7 — early stopping. `RNNDistributedTrainer` now counts epochs without
      improvement and leaves the loop. `early_stopping_patience: 20` and
      `early_stopping_min_delta: 0.0` in the training config; unset in the
      calibration config, where it stays off. `epochs: 100` is now a **ceiling**,
      not a target.

      Patience is deliberately larger than the scheduler's `patience: 10`, so a
      learning-rate drop gets a chance before the run is abandoned.

      **The decision is made on the main worker and shared** via
      `strategy.allgather_obj`, checked at the *top* of the loop where every
      worker reaches it. A worker breaking out of a collective on its own would
      hang the others. Under one process no collective is touched at all.

      Verified in a real run: with `patience: 1` and a `min_delta` nothing can
      beat, a 10-epoch ceiling stopped after 2 epochs and restored the best
      weights. `tests/test_early_stopping.py` covers the agreement logic - and
      found that assigning `trainer.strategy` silently replaces a stub via a
      property setter, so the distributed cases had to set `_strategy`
      directly or they would have passed without testing anything.
- [ ] H8 — fix the temporal samplers so validation uses the same start days
      every epoch. **Agreed 2026-09-21, not started.** See H8 below.
- [ ] H9 — more cells for the surrogate: `train_rows_target: 22000`,
      `frac_time: 0.1`, and a NumPy sample-index build. **Values decided
      2026-09-21, not applied yet.** See H9 below.

---

## Summary

**1. Do not store the full map.** Each wflow run writes 6 GB. Training only
reads a few thousand cells. Pick a fixed set of cells once — the **pool**, 5% of
the map, 16,800 cells — and store only those. **Measured: 392 MB per run**
instead of 6 GB.

**2. Stack runs on top of each other in one long list of cells.** No new
dimension. This is how Tsai's own data is laid out, and it means hython barely
changes. The weather is repeated once per run rather than stored once and
looked up, so the forcing and the target stay in one `dynamic` store - see A1.

**3. Work out the scaling numbers on the full map, not the pool.** Five inputs
get squeezed into 0–1 using the smallest and largest value. A 5% sample misses
the extremes. Training would use one set of numbers and calibration another.

**4. wflow is not the slow part. Training and calibration are.** One wflow run
is 54 min. Training is 3 h and calibration is 3 h, every cycle. Two fixes:
fewer runs (4 at cycle 0, then 1 each cycle), and a fixed number of training
rows instead of a fixed percentage. Without the second, training grows from
12 h to 33 h a cycle as runs pile up.

Time for the whole thing: **~33 h** if it stops at cycle 5, ~46 h for all 8.
Today's setup would take 84 h.

### Things decided along the way

**The pool covers the whole map, not just where calibration looks.**
Calibration only uses 157,817 cells — 47% of the 336,466 we draw from. Drawing
the pool only from those would double the useful density for free. We are not
doing it. The pool is fixed at cycle 0 but the masks are not: a new RT0
version, a longer time range, or a change to the predictors moves that area.
`static_mask` alone drops 95,999 cells for missing predictors, which is a dPL
detail, not physics. The surrogate stays a general wflow emulator.

**The pool is not stratified.** A plain random draw, to keep it simple. It can
be stratified later at the point where training picks its cells, because only
the pool is fixed.

### Two things earlier drafts got wrong

The `vwc` target scaler does **not** drift. `scaler.target_variables` is `null`
in both configs, and the target is never scaled at all. The risk is on the five
inputs in point 3.

`Hybrid.rescale_input` is `F.sigmoid(param)` (`hybrid.py:90-91`), not a
`scaling_static_range` lookup — **but it does not run.**
`config_calibration_loop.yaml:17` sets `scale_head_input_parameter: false`, and
`hybrid.py:52-53` only calls it when that is true. See the corrected note at
the end of H4. The conclusion is unchanged — no data-derived number is involved
— but the mechanism is not the sigmoid.

Also: `run_dpl_cycle.py`'s docstring (lines 18-23) still describes the old
design with a `cycle` dimension. It needs updating.

---

## Background

Each cycle perturbs the calibrated parameters and adds the resulting
(parameters, soil moisture) pairs to a training set that keeps growing. This is
Tsai et al. (2021) and Ahmad et al. (2025).

It has to keep growing. If every cell had one parameter set, the surrogate
could predict soil moisture from the cell alone and never learn how it responds
to the parameters. That response is what dPL needs.

`WflowSBM_HPC` (which the pool dataset `WflowSBM_Pool` replaces) today assumes one parameter set per cell, on the full map.

## What each change does

| id | where | change | blocker? |
|----|-------|--------|----------|
| A1 | `run_dpl_cycle.py` | store only pool cells; stack runs along the cell axis | yes |
| A2 | `run_dpl_cycle.py` | how many runs, rows and epochs per cycle | no, but it is 2/3 of the time |
| H1 | hython | read the stacked layout off the `cell` axis | yes |
| H2 | hython | index cells directly; stop loading the full map | yes |
| H3 | hython | pick the same cells in every run; keep train and valid apart | yes |
| H4 | hython | share scaling numbers with calibration | yes |
| H5 | hython | check the files line up | no |
| H6 | hython | give `RandomDownsampler` its own random generator | no |
| H7 | hython | early stopping | no, but it replaces A2's epoch guesses |

---

## A1 — store only the pool cells

**File:** `run_dpl_cycle.py:258-330` (`prepare_member`, `append_members`)

**Now.** `prepare_member` loads the whole wflow output. `append_members` writes
all of it, once per run.

The map is 568 x 1220 = 692,960 cells. 336,466 of them are usable. Training
reads about 6,700.

| what we store | per run | 11 runs |
|---|---|---|
| the full map | 6.07 GB | **67 GB** |
| pool cells only (measured) | 392 MB | 4.3 GB |

Nothing else reads this data. Calibration uses `OBS`
(`run_dpl_cycle.py:403`), not this archive.

### Step 1 — pick the pool once, at cycle 0

A plain random draw over the usable cells, from a fixed seed. It never changes
after that.

Not stratified, on purpose — see the summary.

**Do not save the pool to a sidecar file.** An earlier draft wrote
`emo1_pool_coords.npz` alongside the scaling numbers. That file held `lat`/`lon`
that duplicated the archive's own coordinates and an `idx` that was exactly
derivable from them — a second source of truth for one fact, which is the drift
H5 exists to catch. Instead the archive describes itself:

| where | what |
|---|---|
| `lat`, `lon` on `cell` | the map position of each row |
| `lat_i`, `lon_i` on `cell` | integer indices into the 568 x 1220 grid, for writing results back |
| `run`, `cycle`, `member` on `cell` | which wflow run produced the row |
| `attrs["pool_size"]` | run boundaries: `runs = n // pool_size` |
| `attrs["pool_seed"]` | lets the draw be reproduced and checked |
| `attrs["runs"]` | JSON manifest: one entry per run, with its source files |

`run` is derivable from position, but it is stored anyway: it costs 2.4 MB at
11 runs, it cannot drift (written in the same `to_zarr` call as the data), and
it lets `verify()` **assert** that each run is `pool_size` contiguous rows in
the same order - the invariant H3's `chosen + m * pool_size` depends on.
`cycle` and `member` are **not** derivable: A2 gives cycle 0 four runs and
later cycles one each, so `run -> cycle` is `[0,0,0,0,1,2,3,...]`, not a
division. These are per-cell coordinates rather than a `run` dimension because
zarr cannot append along two dimensions in one call, which would break the
both-stores-or-neither write.

`draw_pool(seed, size)` is deterministic, so `verify()` redraws from the stored
seed and asserts it lands on the same cells. The pool is recovered for later
cycles with `load_pool()`, which reads the archive's own first run.

The one thing that *does* need an external file is H3's train/valid split —
a different fact, decided once, about which pool cells go where.

### Step 2 — cut down the data inside `prepare_member`

Before anything is written:

```python
sel = dict(lat=xr.DataArray(pool[:, 0], dims="cell"),
           lon=xr.DataArray(pool[:, 1], dims="cell"))
target = target.isel(**sel)     # (time, cell)
static = static.isel(**sel)     # (cell,)
```

Then attach the grid indices and the provenance, so the rows stay traceable
without a sidecar (`_describe` in `pool_archive.py`):

```python
ds = ds.assign_coords(
    lat_i=("cell", pool[:, 0].astype("int32")),
    lon_i=("cell", pool[:, 1].astype("int32")),
    run=("cell", np.full(n, prov["run"], "int32")),
    cycle=("cell", np.full(n, prov["cycle"], "int32")),
    member=("cell", np.full(n, prov["member"], "int32")),
)
ds.attrs.update(pool_seed=POOL_SEED, pool_size=POOL_SIZE,
                runs=json.dumps(_manifest() + [prov]))
```

### Step 3 — add each run to the end of the cell axis

Every run adds `pool_size` rows to the same list:

```
emo1_dynamic_cycle.zarr  (time, cell)    forcing + vwc, runs x pool_size rows
emo1_static_cycle.zarr   (cell,)         theta,         runs x pool_size rows
```

Row `i` is base cell `i % pool_size` from run `i // pool_size`. No `cycle`
dimension. `lat` and `lon` travel with each row, so you can always find where a
row came from.

**Two stores, not three, and the forcing sits beside the target.** That is the
dataset's contract: `WflowSBM_HPC.__init__` opens `urls["dynamic_inputs"]` once
and takes both out of it (`self.xd = data_dynamic[dynamic_inputs]`,
`self.y = data_dynamic[target_variables]`, `wflow_sbm.py:30-34`), and
`urls["target_variables"]` is never opened - the existing config already points
both keys at `emo1_dynamic_calib.zarr`. An earlier draft of this section put
the forcing in a third file and looked it up with `i % pool_size`. That saves
1.46 GB across the 11-run archive and costs a second file open, a modulo in
`__getitem__`, and two stores of unequal length that can no longer be checked
against each other by position. Repeat the forcing instead.

`append_members` keeps writing to a temporary store and only promoting it when
both files succeed. Only `append_dim` changes, from `cycle` to `cell`.

**Tested and it works.** See `scripts/pool_archive.py` and the measurements
below.

**Do not keep `cell` as a coordinate.** If you do, the index becomes
`[0 1 2 3 4 0 1 2 3 4]` — repeated values. xarray allows it, but then `.sel()`
on `cell` is ambiguous. Leave `cell` as a plain position. Keep `lat` and `lon`
as attached coordinates.

Those attached coordinates are non-index, and **`.sel()` does not work on them
in the emulator environment** — xarray 2024.3 raises `KeyError: no index found
for coordinate 'run'`. Newer xarray builds an index on demand and accepts it,
but the environment that trains the surrogate does not, so use
`isel(cell=slice(...))` or `where(ds.run == m, drop=True)`. `method="nearest"`
cannot work either way, since `lat` repeats across runs and within a run.

### How big the pool should be

Once cycle 0 is stored, the pool is fixed. Making it bigger later means running
wflow again, which is the expensive part. Cutting cells is free, so leave room:

Sizes below are scaled from the measured 392 MB at 16,800 cells, written as
**zarr v2** by the emulator environment. They include the forcing, which is
repeated per run. (The same data written as zarr v3 is 256 MB — different
default compression — but v3 is unreadable by the emulator environment, so v2
is what counts.)

| pool | cells | per run | 11 runs |
|------|-------|---------|---------|
| 2% | 6,720 | 157 MB | 1.7 GB |
| **5% (chosen)** | 16,800 | 392 MB | 4.3 GB |
| 10% | 33,600 | 784 MB | 8.6 GB |

5% is 4.3 GB for everything, next to the 32 GB the weather file already uses.
It is 10x the 1,683 cells cycle 0 trains on, and 27x the 612 at cycle 7. That
room is what lets you raise `train_rows_target` later, or stratify the draw,
without running wflow again.

**Pool size costs disk, and now some memory.** With per-epoch resampling (H3)
the whole pool stays loaded — about 2 GB at 5%, against 62 GB available.

**5% is settled** — decided, not to be re-litigated. The earlier suggestion to
train on half the pool against all of it before committing has been dropped.

### What to keep on disk

Each wflow run writes ~6 GB. 19 runs is 114 GB you would almost never read.

Take the pool cells, then delete the big file. Two exceptions:

- **Cycle 0 and the last cycle** — keep the full map output for your own
  figures and for checking the surrogate. 12 GB.
- **The clean `theta_cal` runs** — these are scored against RT0 every cycle.
  Always keep the scores. Keeping the full output each time is optional, at
  6 GB each.

### Why stack along `cell` instead of adding a `cycle` dimension

| layout | weather cost | hython change |
|---|---|---|
| `cycle` dimension | stored once | new dimension through `__init__`, `__getitem__`, scaler |
| **stack, copy weather per run (chosen)** | 1.46 GB extra | almost none |
| stack, look up `i % pool_size` | stored once | a second file open, a modulo, unequal stores |

The second is chosen. The third saves 1.46 GB against a 32 GB weather file and
breaks the one-dynamic-store contract to do it. Both work because every run
holds the same cells in the same order.

---

## What Tsai's released code tells us

The Zenodo archive (`10.5281/zenodo.5227738`, unpacked at
`/mnt/CEPH_PROJECTS/A_DROP/pilot5/dev/iacopo/tsai`) answers two questions and
leaves one open.

**How many cells — answered.** The paper says they sampled one cell per 8x8
patch. The data confirms it: `CONUS_VICv8f1_PM/crd.csv` has **1,206 rows** for
the whole of CONUS. `surrogate_LSTM_example.py` trains on that set, and it is
the one behind `dPL_gA_s8_tr2_model.pt`. Sampling is part of their method.

Useful as scale, not as a target. Their 1,206 cells cover CONUS at ~12 km with
5 parameters and no flow between cells. We have 1 km Alpine terrain with
routing and 16 inputs. Matching their cell count tells us nothing.

**How the data is laid out — answered, and it is the stacked layout.** Cells
are their only axis:

```
CONUS_VICv8f1_PM/crd.csv      1206 rows — the cell list (lat, lon)
                 2015/*.csv    time series, one row per cell
                 const/*.csv   one value per cell — the VIC parameters
Subset/*.csv                   row numbers into crd.csv, or -1 for all
```

Parameters are constants per cell, joined to the weather at every timestep:
`nx = 9 weather + 5 parameters = 14`. Ours is 3 + 16 = 19. There is no cycle
dimension anywhere, so the only way to add runs is to add rows. That is step 3.

They calibrate **5** VIC parameters (`ds, dsmax, expt1, infilt, ws`). The
figure of 13 is Ahmad's Noah-MP, not Tsai.

**How the archive grows — not answered.** The released example is a single
training run on a single dataset. There are only three datasets, none per
cycle. The readme admits the loop is only described in the paper:

> Our experience has been, you don't get the perfect surrogate model using just
> one single run… Hence, **as described in the paper**, we take the parameters
> that dPL produced for each gridcell, also made perturbations… to retrain the
> surrogate model.

So step 3 is our decision. Their data layout constrains it; their code does not
demonstrate it.

**Ahmad et al. (2025) does not help here.** Their surrogate covers "all grid
cells", but their map is 1 degree over CONUS — about 1,620 cells, 428x smaller
than ours. Storage was never a problem for them. Their useful contribution is
the stopping rule (1% improvement).

**One difference from wflow.** VIC and Noah-MP treat each cell on its own, so
Tsai could skip cells in the wflow run itself. wflow_sbm has water moving
between cells, so we must run the full map every time. We can only choose what
to *store*. A1 is the half of their idea that we can use.

---

## A2 — how much to run each cycle

**Files:** `run_dpl_cycle.py:81-105` (`CycleConfig`), `357-420`

Measured on this machine (24 cores, 62 GB, one A100 40 GB): one wflow run
**54 min** (`nohup.out:414`), training **~3 h**, calibration **~3 h**. At
`n_cycles: 8` and `n_members: 4` that is **84 h**, and 48 h of it is training
and calibration.

### (a) 4 runs at cycle 0, then 1 per cycle

`build_members` gives runs two different jobs, and only one of them repeats:

```python
if cycle == 0:
    lhs = latin_hypercube(...); offsets = (lhs - 0.5) * 2.0 * cfg.seed_offset  # +/-0.45
else:
    offsets = np.zeros((cfg.n_members, len(names)))    # zero
```

At cycle 0 each run gets a different parameter **level**, spread across the
whole physical range. This is the only time the surrogate sees parameters far
from the current guess. No number of cells replaces it — it is the failure Tsai
warns about, where the search moves somewhere the surrogate has never been.

After cycle 0 the offsets are zero. Runs differ only by small per-cell noise,
which shrinks each cycle. What teaches the surrogate then is the path
`theta_cal` takes across cycles, and that builds up just as well with one run
per cycle. The noise is per cell, so one run already gives thousands of
parameter/soil-moisture pairs for 5 parameters.

| | cycle 0 | cycles 1-7 | runs | time |
|---|---|---|---|---|
| now | 4 + 1 | 4 + 1 | 40 | ~36 h |
| **chosen** | 4 + 1 | 1 + 1 | **19** | **~17 h** |

Cycle 0 keeps 4 runs rather than more. More runs would cover the parameter
range better, but every extra run stays in the archive forever, and under the
fixed row budget in (b2) that costs cells in every later cycle. 4 also matches
Tsai's four rounds.

Two warnings. The per-cell noise is now the only local signal, so do not let
`jitter_end` fall to 0.03 — keep it around 0.05-0.08. And with one run per
cycle, a failed wflow run means the cycle adds nothing, so add a retry in
`run_wflow` instead of letting `check=True` stop everything.

The clean `theta_cal` run stays one per cycle. It is the score, and it must use
the real model with no noise added.

### (b1) Training gets slower as the archive grows

`WflowSBM_HPC` builds its sample list as cells x time, and after A1 the cell
count is `places x runs`. **So every extra run makes an epoch slower.** This is
what an accumulating archive means, and it is why a fixed percentage fails.

Keeping today's 1% (3,365 cells):

| cycle | runs | rows | training |
|---|---|---|---|
| 0 | 4 | 13,460 | ~12 h |
| 3 | 7 | 23,555 | ~21 h |
| 7 | 11 | 37,015 | ~33 h |

That is ~54 h of training alone, and it would cancel out the saving in (c).

### (b2) Fix the number of rows, not the percentage

Draw a fixed number of rows each cycle and let the cell count follow:

```
places = train_rows_target // runs_in_archive
```

`train_rows_target` is a new setting on `CycleConfig`. **Expose it** — it is the
one knob that trades training time against how many places you cover.

**Chosen: `train_rows_target = 6730`** — twice today's 3,365 rows, about 6 h a
cycle.

| cycle | runs | places | % of pool | rows | training |
|---|---|---|---|---|---|
| 0 | 4 | 1,683 | 10.0% | 6,730 | ~6 h |
| 1 | 5 | 1,346 | 8.0% | 6,730 | ~6 h |
| 3 | 7 | 961 | 5.7% | 6,730 | ~6 h |
| 5 | 9 | 748 | 4.4% | 6,730 | ~6 h |
| 7 | 11 | 612 | 3.6% | 6,730 | ~6 h |

The steps follow 1 over the number of runs, not a straight line. A straight
line (1.0%, 0.9%, 0.8% …) would peak near 16,500 rows in the middle. Writing it
as a row target also means it fixes itself if the run count changes.

At 13,460 rows instead, every number doubles and training is ~12 h a cycle.

**The cost is places per epoch:** 1,683 at cycle 0 down to 612 at cycle 7. With
per-epoch resampling turned on (H3) this is much less of a worry than it looks
— the cells change every epoch, so over 100 epochs the model still covers the
whole train pool. With it turned off, 612 really is all the model ever sees,
and it is the number to watch if the surrogate stops improving.

**The places must be the same in every run** — see H3.

### (c) Do not retrain from scratch every cycle

Training runs `epochs: 100` and calibration `epochs: 60` every cycle, both
starting from the previous weights (`CudaLSTM.load`, `TransferNN.load`). With
one run per cycle, cycle 5 does 100 epochs to take in about 7% more data.

Cycle 0 needs the full amount. Later cycles are just tuning: about 25 and 20
takes 9 h a cycle down to ~2.5 h. Both are overrides in the dict
`write_cycle_config` already builds.

Those numbers are guesses. H7 replaces them with a rule. If H7 lands first,
skip this section.

### (d) Calibration does not need every cell while training — DROPPED

**Decision, 2026-09-20: not doing this.** The reasoning below still holds; it
is simply not worth what it costs.

The original idea: `train_downsampler` is `null` in the calibration config, so
the head trains on all 157,817 cells. `TransferNN` maps attributes to
parameters one cell at a time, so training it on a sample and then applying it
to the whole map is fine, and you still get a full map of parameters out.
Sample the middle cycles, run the last one at full size.

**Why it was dropped.** It buys about **6 h** - calibration goes from 3 h to
1 h in the middle cycles, but the `+2 h` full-size run at the end exists only
because the middle ones were sampled, so the net on a run stopping at cycle 5
is ~39 h instead of ~33 h. Against that:

- it is the only A2 item that needs **new code**. The rest are settings.
- it puts a sampler into the calibration path, which today has none. That is
  exactly where `RandomDownsampler`'s global `np.random.seed` would start to
  matter, so H6 would have to land first (see the H6 note above).
- each cycle would calibrate on a different subset, adding variance between
  cycles to the thing the whole loop is trying to converge.

The other A2 items - run counts per cycle (a), the row budget (b2), fewer
epochs after cycle 0 (c) - save more, are pure scheduling, and carry none of
this. If the loop turns out slower than the budget predicts, (d) is still
available as a last-resort lever.

### (e) `n_cycles: 8` is a limit, not a plan

The 1% rule may stop it at 4 or 5 (Ahmad ran five). Budget for ~5 cycles and
treat 8 as the cap.

### (f) Running wflow twice at once — measure first

`julia_threads: 24` on 24 cores means runs go one at a time. wflow rarely gets
24 threads' worth of speed, so two at 12 threads each may beat one at 24. It
only matters at cycle 0 (4 runs, 3.6 h). **Time one run at 12 threads against
one at 24 before deciding.** Do not assume it helps.

### Total

With (a) to (e), `train_rows_target = 6730`, full epochs only at cycle 0:

| | wflow | train | cal | total |
|---|---|---|---|---|
| cycle 0 (4 runs) | 4.5 h | 6 h | 3 h | 13.5 h |
| cycles 1-4 (1 run each) | 1.8 h | 1.5 h | 1 h | 4.3 h each = 17 h |
| last cycle, full-size calibration | | | +2 h | 2 h |
| **total, stopping at cycle 5** | | | | **~33 h** |
| all 8 cycles | | | | **~46 h** |

Against 84 h today. The saving splits about evenly between fewer wflow runs,
the row budget, and shorter epochs. No single change does it.

**One more idea, not counted.** A1 takes training off the ~49 GB memory limit,
so a wflow run could go at the same time as training. That means restructuring
`main()`.

---

## H1 — read the stacked layout

**File:** `hython/datasets/wflow_sbm.py:10-193`

**Now.** `__init__` opens two files and takes the target out of the weather
file:

```python
self.xd = data_dynamic[self.to_list(cfg.dynamic_inputs)]
self.xs = data_static[self.to_list(cfg.static_inputs)]
self.y  = data_dynamic[self.to_list(cfg.target_variables)]   # line 34
```

`urls["target_variables"]` is worked out by `get_source_url` but never opened.
A separate target file would therefore not be a config change but a code
change - which is why A1 does not use one.

**Change.**

- **nothing changes about how the files are opened.** `dynamic_inputs` and
  `target_variables` both point at `emo1_dynamic_cycle.zarr`, exactly as they
  both point at `emo1_dynamic_calib.zarr` today, so lines 30-34 stand
- the sample axis is the archive's `cell` axis, length `runs x pool_size`.
  `self.mask` and `np.argwhere` are no longer needed, because the pool is
  already masked and chosen
- `__getitem__` reads `xs[idx]`, `y[..., idx]` and `xd[..., idx]` - one index,
  no modulo, because the forcing is repeated per run
- read `pool_size` from the archive's `attrs`, do not guess it from the data

A sample is (run, base cell, time): same cell, same weather, different
parameters, different soil moisture. That difference is the only thing that
shows how the parameters act.

**The number of inputs does not change.** 16 statics and 3 weather variables,
so the LSTM still takes 19. Which run a row came from is never an input.

## H2 — index cells directly

**File:** `hython/datasets/wflow_sbm.py:41-44, 76-83, 137-166`

**Now.** With `data_lazy_load: false`, `__init__` loads the **whole map** into
torch tensors and only then cuts it down. At 2191 x 568 x 1220 that is 18.2 GB
for the weather and 6.1 GB for the target. Train and valid each keep a copy, so
the current single-cycle run already uses ~49 GB of the 56 GB available.

**Change.** After A1 the files arrive already cut down, so most of this goes
away. What is left:

- `__getitem__` uses `[..., idx_cell]` instead of `[..., idx_lat, idx_lon]`
- the `itertools.product` and `unnest` code that builds `spacetime_index`
  (`wflow_sbm.py:76-83`) goes away
- keep the downsampler. It draws `train_rows_target` rows from the pool, and
  only those are loaded

**The `test` dataset is built every run.** `RNNDatasetGetterAndPreprocessor`
creates it whenever `test_temporal_range is not None`
(`itwinai/data.py:67-71`), and the training config sets it to 2022. So this is
real, not hypothetical — and today it is a big part of the memory problem,
because `period == "test"` takes **every cell of the full map**
(`np.argwhere(np.ones(shape))`, `wflow_sbm.py:52-57`).

After A1 it reads the pool like everything else, which is the right outcome:

- `test` is a **time** holdout (2022 against 2017-2019). Checking it on pool
  cells is still a fair holdout. Only the area changes.
- the "no masking" branch does nothing now, because the pool is already masked.
  Keep the check or delete it, but do not let it try to rebuild a full map that
  the archive no longer holds.
- memory for `test` drops from ~24 GB to the pool's share.

**`create_xarray_data` cannot read the `cell` axis.** `hython/utils.py:520`
hardcodes its dim order:

```python
for v in ["lat", "lon", "time", "variable"]:
    if output_shape.get(v):
```

`cell` is not in the list, so it is dropped from `reordered_out_shape` and the
reshape fails with `cannot reshape array of size 672000 into shape (40,1)`. It
fails loudly, which is the good case.

**The fix is one line** - add `"cell"` before `"time"`. The order matters and
is not arbitrary: `WflowSBM_HPC` builds `spacetime_index` as
`itertools.product(cells, times)`, so the flat prediction is cell-major.
Verified to round-trip exactly against the archive.

Two things to watch on that path:

- **pass `crs=None`.** A CRS on a scattered cell axis is meaningless, and
  `rio.write_crs` needs `rioxarray` imported anyway.
- **getting a map back needs a scatter step**, which is what `lat_i`/`lon_i`
  are for. `pool_archive.to_xarray` and `pool_archive.scatter_to_map` do both
  halves and are tested against the archive; move them into hython with H2.
  The result is 568 x 1220 with the 16,800 pool cells filled and the rest NaN.

**`Evaluator` is broken and is being deleted, not fixed.** `Evaluator.preprocess`
(`evaluator.py:184-202`) assumes a lat/lon grid in three places -
`list(ds_target.data_vars)` on what is a `torch.Tensor` by then, `~dataset.mask`
which is now `(cell,)`, and `len(ds_target.lat)` which is 16,800 per run rather
than 568. It does not survive the pool archive, but it was already incompatible
with `WflowSBM_HPC` before A1 for the tensor reason. Nothing constructs it -
no call site, no `_target_`, no import - so the `evaluator:` block in the
training config is dead config. Do not spend H2 effort on it.

**Calibration is unaffected.** `WflowSBMCal` reads the full-map
`predictor_emo1_alps.zarr`, not the pool archive, so `inference.py:94` still
writes a dense `inference_parameter.nc` for wflow. Only the training and
evaluation path meets the `cell` axis.

**Full-map checking moves outside.** Use the cycle-0 and last-cycle files kept
by A1. If you would rather not, set `test_temporal_range: null` during the loop
and skip the dataset entirely.

## H3 — same cells in every run, train and valid kept apart

**Files:** `hython/datasets/wflow_sbm.py:71-73`,
`hython/sampler/downsampler.py:63-69`

**Change.** Choose base cells, then take **every run** of each:

```python
places = train_rows_target // runs             # A2(b2)
chosen = train_cells[:places]                  # from the fixed cycle-0 split
rows = np.concatenate([chosen + m * pool_size for m in range(runs)])
```

**`RandomDownsampler` cannot do this.** It only takes fractions, and
`sampling_idx` does `np.random.choice(space, int(len(space)*frac_space))` over
the flat list (`downsampler.py:49-69`). Run against the stacked axis it picks
rows independently, so different runs would get different cells — which is the
exact problem H3 exists to stop. Changing `frac_space` per cycle does not help;
the draw is the wrong shape.

So this needs a new downsampler that knows `pool_size` and picks base cells
before expanding to runs, or the choice moved into `WflowSBM_HPC.__init__`
with the downsampler skipped. The first keeps the sampling logic in one place.

Two things have to hold at once.

**(i) Every run uses the same cells.** Not just the same number — the same
cells. The surrogate learns from same cell, same weather, different parameters.
If runs used different cells, each cell would have one parameter set again and
we are back to the problem in Background. Taking all runs of each chosen cell
also means every run contributes the same count.

**(ii) The train/valid split never changes.** `train_cells` and `valid_cells`
are separate lists of base cells, drawn once at cycle 0 from a fixed seed, and
saved next to the scaling numbers. Otherwise a cell could be in train at cycle
0 and in valid at cycle 3 — same weather, nearly the same parameters — and
validation would look best exactly when the surrogate starts getting worse.

**How a shrinking `places` fits with (ii).** The split never changes. Only how
many cells we take from the train side shrinks (A2(b2)). Keep `train_cells` in
a fixed shuffled order and take the first `places`, so each cycle's choice sits
inside the previous one. Nothing moves between train and valid, and no cell
joins training for the first time late in the run.

Valid should take its own fixed, smaller number rather than shrinking too, so
validation loss stays comparable across cycles.

**`frac_space` is replaced** by `train_rows_target` for train and valid. Remove
it unless something else reads it — a leftover `frac_space: 0.01` now meaning
1% of the pool is a 20x mistake waiting to happen.

### Pick new cells every epoch

**Today the cells are picked once.** `sampling_idx` runs inside `__init__`
(`wflow_sbm.py:73`), so the same cells are used for all 100 epochs.
`set_epoch` (`itwinai/trainer.py:225`) only reshuffles their order, and
`epoch_step` (`trainer/rnn.py:99`) just walks the list. `RandomDownsampler` is
random in *which* cells it takes, but it takes them once.

**Change: draw a fresh set each epoch, behind a switch.**

- new config flag, default on — `resample_cells_each_epoch`
- off: today's behaviour, one draw when the dataset is built
- on: the dataset holds every train row; a `Sampler` yields a fresh
  `places x runs` set of row indices per epoch
- seed the epoch's draw from (base seed, epoch number) so a run still repeats
- **validation is never resampled.** Keep it on its fixed cells, or the loss
  stops being comparable between epochs and between cycles

**Cost per epoch does not change** — still `train_rows_target` rows. What
changes is coverage: 612 cells per epoch over 100 epochs is ~61,000 draws
against a train pool of roughly 13,000 cells, so everything gets seen many
times.

**This is what makes the pool worth storing.** Without it the archive holds
16,800 cells and training only ever touches 612 of them at cycle 7. With it the
pool becomes the training set, used a slice at a time, and the shrinking
`places` in A2(b2) stops meaning "the model only sees 612 places" and starts
meaning "612 places per epoch, different ones each time".

**It fits the existing code.** `set_epoch` is already called every epoch and
already forwards to the loader's sampler, so a custom sampler drops in.

**Single GPU makes this easy.** On multiple ranks the epoch's draw would have to
be identical across ranks and then sharded between them; on one GPU neither
applies.

**The pool must be in memory for this to work** — you can no longer load only
the drawn rows. That is fine: the whole pool is ~2 GB (442 MB weather, 1.6 GB
target across 11 runs, 12 MB statics) against 62 GB.

**Expect a noisier training loss,** since each epoch sees different data.
Validation stays fixed, so early stopping and the learning-rate scheduler are
unaffected. Do not read the noise as instability.

## H4 — share the scaling numbers with calibration

**Files:** `hython/scaler.py:213-270`, `hython/datasets/wflow_sbm.py:101-113`

**Five inputs are scaled in both configs:** `wflow_uparea`, `wflow_landuse`,
`wflow_dem`, `Slope`, `WaterFrac`. They use `MinMax01` as
`scaler.static_inputs` in training and as `scaler.head_model_inputs` in
calibration. Same five, same scaler, two separate sets of numbers.

Today both look at the full map, so they agree. After A1 training would look at
5% of it, and the smallest and largest values are exactly what a 5% sample
misses. Measured on `emo1_static_calib.zarr`, three seeds:

| variable | true max | 5% pool max | ratio |
|---|---|---|---|
| `wflow_uparea` | 6.741e4 | 6.73e4 | 0.998 |
| `wflow_landuse` | 521 | 521 | 1.000 |
| `wflow_dem` | 4545 | 4009 / 4149 / 4122 | 0.901 |
| `Slope` | 144.6 | 123.7 / 118.8 / 117.1 | 0.829 |
| `WaterFrac` | 0.8888 | 0.7445 / 0.5977 / 0.5974 | **0.727** |

`uparea` and `landuse` survive, because many cells sit at the top. The other
three do not, and `WaterFrac` is both the worst and changes with the seed.

A cell with `WaterFrac = 0.85` becomes 0.96 under the true numbers but ~1.42
under pool numbers. At calibration `WflowSBMCal` hands the surrogate values
scaled the first way, and the surrogate never saw anything above 1.0.

**Change.** Work these five out on the **full map**, once, at cycle 0, and use
them everywhere. Cheap — the statics are 39 MB with no time axis — and A1
already keeps the full-map static file.

**How, because there is no mechanism today.** `generate_run_folder` is
`{work_dir}/{experiment_name}_{experiment_run}/` (`hython/utils.py:29-34`), and
the two paths use `train_multicycle` and `cal_multicycle`, so they write to
different folders. Loading the surrogate does not help: `CudaLSTM.model_uri`
points at the training config, and `load_model` follows it only as far as
`model.load_state_dict(...)` (`hython/models/__init__.py:37-44`) — **weights
move, numbers do not**. In `Hybrid.forward` the five arrive inside
`x_head_static`, already scaled by whatever `WflowSBMCal` worked out for
itself. So write the full-map numbers once to a path both configs name, and set
`scaling_use_cached: true` on both.

**How bad is it?** Probably not very. Only three of five are affected, and the
cells pushed out of range are the rare ones, not the bulk. The reason to fix it
is that it is **silent and changes with the seed** (`WaterFrac`'s pool max
moved between 0.60 and 0.74), so results would not reproduce and nothing would
ever raise an error. The fix is a few lines.

**Also make the cached path strict.** `load_or_compute` quietly recomputes when
the file is missing:

```python
if self.use_cached:
    try:
        self.load(type)
    except FileNotFoundError:
        LOGGER.info("Statistics not found ..., computing statistics..")
        self.compute(data, type, axes, **kwargs)
```

That would recompute on the pool and bring the problem back without saying so.
Make it raise instead.

**And rename the axes.** `load_or_compute` is called with
`axes=("lat","lon","time")` for the target and `("lat","lon")` for statics.
Those axes no longer exist in the training archive: use `("cell","time")` and
`("cell",)`. This is for the weather inputs, whose numbers stay pool-based and
are not shared with calibration. The five statics come from the frozen full-map
file.

Stacking avoids the harder version of this. With a separate `cycle` dimension,
reducing over `lat/lon/time` would have left a `cycle` axis on the numbers —
one set per cycle, and a shape error as soon as a run was added. On a stacked
axis the reduction covers every run automatically.

**`BoundedScaler` is not affected,** which is why the calibrated parameters are
safe. `KsatVer, c, f, RootingDepth, Sl` and the `aux_param` group use fixed
ranges in both configs. No data-derived number touches them, which is also why
perturbing in scaled space is consistent from end to end.

At calibration the parameters do not go through `BoundedScaler` at all.

**Corrected 2026-09-20.** An earlier draft said `Hybrid.rescale_input` is
`F.sigmoid(param)` and that this maps the `TransferNN` output into (0, 1). The
method exists, but `scale_head_input_parameter: false`
(`config_calibration_loop.yaml:17`) means it never runs, and `TransferNN`'s
output activation defaults to `"linear"`.

**The bound is a soft one, in the loss, and that is deliberate.**
`config_calibration_loop.yaml:45-65` applies `RangeBoundReg` with `factor:
1000` and bounds `[0, 1]` on `output: param`. It is a hinge —
`relu(x - ub) + relu(lb - x)` — zero inside the range and linear outside. A
sigmoid would saturate and kill the gradient near the bounds; a penalty keeps
it alive and lets the optimiser sit against the boundary.

So the space is the same one `BoundedScaler` uses in training, and no
data-derived number is involved either way. What changes is that the bound is
approximate: measured on real cycle-0 output, a few cells per parameter land
just outside, worst case 7.9% of the range, varying which parameter from run to
run. That is the equilibrium between data fit and penalty, not a defect.

`write_staticmaps` clamps to `CAL_PARAMS` before writing, so wflow never
receives an invalid value such as a negative conductivity. The clamp does not
touch the optimisation — only what is written out.

## H5 — check the files line up

**File:** `hython/datasets/wflow_sbm.py:172-190`

Files are matched by position, opened separately, and never aligned. If they
ever differ in length or order, parameters get paired with the wrong soil
moisture **silently** — no error, just a surrogate that is inexplicably bad.

A1 makes this easier to check, because all three files carry the same `lat` and
`lon` on `cell`. In `__init__`:

```python
n = self.xs.sizes["cell"]
assert self.y.sizes["cell"] == n
assert n % self.pool_size == 0
np.testing.assert_array_equal(
    self.xs.lat.values, np.tile(self.xd.lat.values, n // self.pool_size)
)
```

**`n % pool_size == 0` is weaker than it looks, and a dry run proved it.** A
16,800-row run followed by two 300-row runs is 17,400 rows, which divides by
300 cleanly and passes. The check that actually localises the fault is the
`run` coordinate:

```python
np.testing.assert_array_equal(
    ds.run.values, np.repeat(np.arange(runs, dtype="int32"), pool_size)
)
```

`pool_archive` now runs that first, and refuses at write time to append a run
whose length is not `pool_size` or whose pool differs from the archive's.

The exact lines will need adjusting once the files exist. The point is to turn
the worst failure into a loud one.

## H6 — give `RandomDownsampler` its own random generator

**File:** `hython/sampler/downsampler.py:56-58`

`__init__` calls the **global** `np.random.seed(self.seed)`. Train and valid use
the same `${sampling_seed}`, so which cells each one gets depends on the order
the two objects are built relative to when `sampling_idx` runs. It also changes
global numpy state for everything else in the process.

Use `self.rng = np.random.default_rng(seed)` and draw from `self.rng`. Give
train and valid different seeds.

## H7 — early stopping

**File:** `hython/itwinai/trainer.py:251-325`

**Not there today, has to be written.** But the epoch loop already has
everything it needs:

```python
best_loss = float("inf")
for epoch in tqdm(range(self.epochs)):
    ...
    avg_val_loss = torch.mean(torch.stack(worker_val_losses))...   # 275
    self.hython_trainer.lr_scheduler.step(avg_val_loss)            # 277
    if avg_val_loss < best_loss:                                   # 316
        best_loss = avg_val_loss
        best_model = self.model.state_dict()
self.model.load_state_dict(best_model)                             # 325
```

Validation loss is already averaged across workers, `best_loss` is already
tracked, and the best weights are already restored at the end. What is missing
is a patience counter and a `break` — about ten lines. Stopping early is safe,
because the model returned is the best one seen, not the last.

**Why it matters.** It replaces the guessed epoch counts in A2(c) with a rule.
Each cycle then runs as long as it needs, without picking a number that will
drift as the archive grows and the noise shrinks.

Two notes. The counter must use the gathered `avg_val_loss`, not a per-worker
one, or workers will stop at different epochs. And `patience: 10` in the
configs belongs to the learning-rate scheduler — early stopping needs its own,
larger, so a scheduled rate drop gets a chance to work first.

## H8 — validation must use the same start days every epoch

**File:** `hython/sampler/__init__.py` (the three `*TemporalDynamicDownsampler`
classes), wired in `hython/itwinai/trainer.py:435-460`.

**Status: planned, not started.** Found reading the 2026-09-21 smoke logs. We
act on it when the user decides.

**What happens today.** When `dynamic_downsampler` is set in the config (the
training config sets `frac_time: 0.3`), the trainer wraps *both* loaders:

    train  ->  RandomTemporalDynamicDownsampler
    valid  ->  SequentialTemporalDynamicDownsampler

Both call `random.sample` in `__iter__`, so both draw a **new** 30% of start
days every epoch. The "sequential" one only sorts its draw. So H3 keeps the
validation *cells* fixed, but the validation *days* still change per epoch.

Three problems follow:

1. **Validation loss is noisy.** It moves with the days drawn, not only with the
   model. In the smoke, val RMSE jumps between 0.07 and 0.11 from one epoch to
   the next. Early stopping (H7) and the learning-rate scheduler both read this
   number, and the multicycle loop compares it between cycles.
2. **The draw cannot be repeated.** `seed` goes to `np.random.seed`, but
   `random.sample` uses Python's `random` module, which that seed does not
   touch. (Not checked: whether itwinai seeds Python's `random` elsewhere.)
3. **Global state.** `np.random.seed` in `__init__` resets numpy's global
   generator for the whole process - the same fault as H6.

**The yaml cannot fix it.** `frac_time` sets the size of the subset, one value
for both loaders. Nothing makes the subset fixed. `dynamic_downsampler: null`
gives validation every start day, but training too - about 3.3x longer epochs
and no per-epoch variety of days in training.

**Planned change** (in the sampler classes, not a Pool-only switch in the
trainer):

- **Validation:** draw the start days **once**, in `__init__`, from the
  sampler's own `np.random.default_rng(seed)`. Use the same days every epoch.
- **Training:** keep a new draw every epoch, but from its own seeded generator,
  so a run repeats exactly.
- **New optional key** `dynamic_downsampler.frac_time_valid`. Defaults to
  `frac_time`, so existing configs keep today's validation size. `1.0` or
  `null` means every start day.
- Remove `np.random.seed(...)` from the constructors and the stray
  `print(self.total_subset_size)`.
- `DistributedTemporalDynamicDownsampler`: same seeding fix, and make
  `shuffle=False` (validation) draw once.

**Agreed 2026-09-21** (the plan above, and the recommended answers to the two
questions it raised - correct here if that is wrong):

- *Validation fraction.* Validation gets its own fixed subset of days, drawn
  once. Default: every start day, as long as one validation pass stays around a
  minute; otherwise the smallest fraction that does. Pick the number when
  implementing, from the measured pass time.
- *Distributed variant.* Fix the seeding and the draw-once for validation now.
  The missing split by rank is **left for later** - we run on one GPU - but
  gets a comment in the code and a line under "Still not checked".

**Validation needs its own budget (agreed 2026-09-21).** Today the validation
size follows the training settings twice over: `valid_downsampler.rows_target`
is `${train_rows_target}`, and the day fraction is the shared `frac_time`. With
the larger row budget of H9 that ties validation cost to training choices it
has nothing to do with. Validation gets its own row target and its own day
fraction, both fixed, chosen so one pass stays at a few hundred thousand
windows.

**Who else is affected.** `hython` is installed editable in the `emulator` env.
`hython-itwinai-plugin` (its trainer imports `SamplerBuilder` and asks for
`temporal-downsampling-sequential`) and the `notebooks/config/*.yaml` configs
pick the change up. Intended: their validation becomes fixed too. But their
random draws change, so old runs will not repeat exactly. The calibration
config has `dynamic_downsampler: null` and is not affected.

**Tests to add:**

- Validation days are the same across epochs.
- Training days change between epochs, and are the same for a given seed.
- A fraction of `1.0` gives every start day.
- The sample count equals cells x days in `spacetime_index`.

## H9 — more cells for the surrogate

**Files:** `config/config_training_calibration_loop.yaml:80` (`train_rows_target`)
and `:105` (`dynamic_downsampler.frac_time`); `hython/datasets/wflow_sbm.py`
(`WflowSBM_Pool.build_sample_index`).

**Status: values decided 2026-09-21, not applied yet.**

**Why.** The 2026-09-21 smoke surrogate was weak (validation NSE ~0.25) and
calibration made wflow worse than not calibrating. A test on the finished smoke
archive (`smoke_runs/rows_test/` on CEPH: 6 runs, 2-year window, trained from
scratch, scored on one fixed held-out set of 1000 validation cells x 6 runs x
every start day) showed the number of cells was the limit:

| variant | cells per run | pooled NSE | RMSE | sensitivity r | epochs (best) | s per epoch |
|---|---|---|---|---|---|---|
| 600 rows, `frac_time` 0.3 | 100 | 0.31 | 0.075 | 0.54 | 12 (2) | 9 |
| 6000 rows, 0.3 | 1000 | 0.63 | 0.055 | 0.73 | 30 (26) | 32 |
| 12000 rows, 0.1 | 2000 | 0.62 | 0.056 | 0.71 | 30 (30) | 22 |
| 24000 rows, 0.05 | 4000 | 0.61 | 0.056 | 0.72 | 30 (28) | 23 |

From 100 to 1000 cells per run, skill doubles. Beyond that it holds, at a
lower cost per epoch when start days are traded for places: windows one day
apart are nearly the same sequence. The smoke's own cycle 4 model,
warm-started over five cycles at 600 rows, scored the same as the 600-row one.
The three large variants all reached the 30-epoch ceiling while still
improving. Sensitivity slope was ~0.5 at best: the surrogate recovers only
about half of theta's effect on vwc.

**Decided.**

- `train_rows_target: 22000` (was 6730). **Rows stay fixed across cycles**
  (A2 b2), so cells per run still shrink as the archive grows: 5500 at 4 runs,
  2000 at 11 - never below the ~1000 where skill stopped improving.
- `frac_time: 0.1` (was 0.3): ~97 start days per cell per epoch on the
  production training window (2017-2019, 975 start days).

**Cost, estimated.** ~22000 x 975 x 0.1 = 2.1M training windows per epoch.
At the ~55 us per window measured for the 12000-row variant, that is **~2 min
per epoch**, plus validation, the same in every cycle. So 30 epochs is ~1 h
and 50 epochs ~1.7 h. Linear scaling from the 2-year test to the 3-year
window is an assumption; the first real cycle 0 measures it.

**Epoch ceiling (decided 2026-09-21).** Longer surrogate training is
accepted, because early stopping (H7) ends it. The training config already has
`epochs: 100` as a ceiling and `early_stopping_patience: 20`; both stay. At ~2
min per epoch the ceiling is ~3.3 h, reached only if validation keeps improving.
Early stopping needs a steady validation loss to be trusted - one more reason
H8 comes first.

**Needed first: build the sample index with NumPy.** `build_sample_index`
makes a Python list with one tuple per (row, start day) and rebuilds it every
epoch (H3 resampling). Measured:

| rows x start days | time | peak memory |
|---|---|---|
| 24000 x 317 (the test) | 2.2 s | 0.8 GB |
| 52000 x 975 | 15 s | 5.2 GB |

At 22000 x 975 (21M pairs) it is ~6 s and ~2 GB each epoch, on a machine
that also runs 24 loader workers. `np.repeat`/`np.tile` give the same
cell-major index in a fraction of a second, without Python objects.
`tests/test_pool_epoch.py` already checks the index; add a check that the
NumPy build equals the old one.

**Validation** gets its own budget - see the H8 note. With
`rows_target: ${train_rows_target}` and the shared `frac_time`, validation
would otherwise be 22000 rows x ~25 days per epoch.

---

## Order to do it in

1. **H6** — on its own, no dependencies, makes everything after it repeatable.
2. **A1** — orchestrator only. Gives a one-run archive in the new layout to
   develop against.
3. **H1 + H2 + H3 + H4 axes** — one change to `WflowSBM_HPC.__init__` plus the
   scaler axes. Do them together; each is incomplete alone.
4. **H4 full-map numbers** — needed before the first calibration, not before
   the first training, so it can follow step 3.
5. **H4 strict mode, H5** — small safety, any time after.
6. **H7** — separate from all of it, can be written in parallel. It is what
   turns A2(c) from a guess into a rule.

A2 is config and scheduling in the orchestrator. It can land at any point, and
is worth doing before the first real run because it controls two thirds of the
time.

## What to check

- **A1:** `python pool_archive.py --append-test --verify`. Checks rows, that
  the pool redraws from the stored seed, that `lat`/`lon`/`lat_i`/`lon_i` agree
  with the map in both stores, that `run`/`cycle`/`member` match the manifest,
  and that the forcing repeats identically. Disk use is ~256 MB per run.
- **H1:** build a two-run archive where run 1 is run 0 with `KsatVer` multiplied
  by a known number. A surrogate trained on both must predict different soil
  moisture for the two rows of the same cell. Today's code cannot.
- **H2:** peak memory down from ~49 GB to the pool's share, and the `test`
  dataset no longer loading the full map. You **cannot** compare loss curves
  with the old code — A1 changes which cells exist. Instead pick a fixed set of
  cells present in both layouts and check the model output matches.
- **H3:** check the train and valid cell lists never overlap and never change
  between cycle 0 and cycle *n*.
- **H4:** check the numbers file is identical after cycle 0 and cycle *n*, and
  that the five statics hold the full-map maxima above (`WaterFrac` 0.8888,
  `Slope` 144.6, `wflow_dem` 4545), not pool maxima.
- **H5:** add to one file only, check the dataset raises.
- **H7:** on a run known to stop early, check it does, and that the restored
  weights are the best epoch and not the last.
- **A2(f):** time one wflow run at `julia_threads: 12` against one at 24 before
  assuming two at once is faster.

## Still not checked

- `DistributedTemporalDynamicDownsampler.__iter__` never splits samples by
  rank: on several GPUs every GPU would get every sample. Left for later (H8).

- **Training time is assumed to grow in step with rows.** The direction is
  certain. The hours are arithmetic, not measurement.
- **Is 612 places per epoch at cycle 7 enough?** Unknown, and much less
  pressing with per-epoch resampling on.
- **Does `dynamic_downsampler` (`frac_time: 0.3`) apply here?** It would not
  change the scaling argument, but it shifts the absolute hours.
- **H5's exact assertion** depends on how the files end up looking.

## References

- Tsai, W.-P. et al. (2021). From calibration to parameter learning.
  *Nat. Commun.* Code: Zenodo `10.5281/zenodo.5227738` — **no GitHub**. Unpacked
  at `/mnt/CEPH_PROJECTS/A_DROP/pilot5/dev/iacopo/tsai`.
- Ahmad, S. K. et al. (2025). Section 2.4.1. Perturbs the learned parameters
  every iteration; stops at 1% improvement.
- `mhpi/generic_deltaModel` (clone at
  `/home/iferrario/dev/hybrid_models/generic_deltaModel`) does **not** cite Tsai
  and has no surrogate. It writes the physical model in PyTorch directly, which
  is Tsai's other approach, the one that needs no archive. Not a reference here.
