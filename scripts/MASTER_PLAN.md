# Master plan: dPL runs for the parameter-learning article

Last updated: 2026-10-18. Tick a box when a step is done and add the date.
Detailed engineering notes are in `HYTHON_MULTICYCLE_PLAN.md`.

## Where things are

| What | Path |
|---|---|
| Python env (`emulator`) | `/mnt/CEPH_PROJECTS/A_DROP/pilot5/dev/conda/emulator/bin/python` |
| Julia + Wflow 0.8.1 | `/mnt/CEPH_PROJECTS/A_DROP/pilot5/dev/julia` (the scripts set `PATH` and `JULIA_DEPOT_PATH`) |
| c1ap campaign (theta RT0, cycles 0-1) | `A_DROP/pilot5/article/dpl_apriori/` (read only) |
| RT0 scaling decision | `A_DROP/pilot5/article/rt0_scaling_decision/{minmax,variance}/` |
| Final run | `A_DROP/pilot5/article/dpl_final/` |
| Transfer-error notebook | `work/iacopo/article/param_estimation/DPL_TRANSFER_ERROR.ipynb` (campaign and cycle set at the top) |
| Transfer-error outputs (surrogate predictions, per-cell KGEs) | `A_DROP/pilot5/article/dpl_final_analysis/transfer_error/cycle_<N>/` |
| RT0 files | `InterTwin/hydrologic_data/SSM-RT0-SIG0-R-CRRL/processed/alps_rt0old_2017-2022_{theta,minmax,variance}.nc` |

## Fixed decisions

- Periods: train 2017-2019, validation 2020, test 2021-2022.
- Cycle 0 archive: 4 perturbed runs + the a priori run (no new wflow runs; reused from the archive).
- KGE weights (r, alpha, beta) = (1, 0.25, 0.75). Article reason: RT0 is noisy (less weight on alpha), and the RT0-wflow bias should matter less (beta < 1).
- Final run: 8 cycles (0-7), stopping rule off; the convergence rule picks the cycle to analyse afterwards.
- Note for the article: Sentinel-1B was lost in Dec 2021, so 2022 has about half the RT0 observations (median 28 per cell vs ~62; 67% of land cells with >= 20 obs vs 80%).

## 0. Before any run

- [ ] Get a larger VM. This one has 31 GB RAM; c1ap calibration peaked at ~50 GB.
- [ ] On the new VM: check RAM, cores, GPU. If the core count changes, check `JULIA_THREADS = 8` in the two scripts.

## 1. RT0 scaling decision (next)

Which RT0 version to calibrate on. theta = c1ap (already done). Run the other two, one at a time.
Script: `run_rt0_scaling_decision.py`. ~16 h per version.

- [x] Script written (2026-10-18)
- [ ] `--rt0 minmax --seed`, then `--check`
- [ ] Run minmax (cycles 0-1)
- [ ] `--rt0 variance --seed`, then `--check`
- [ ] Run variance (cycles 0-1)
- [ ] Compare theta / minmax / variance, per cycle: median per-cell
      dKGE = KGE(wflow_cal, RT0_v) - KGE(wflow_apriori, RT0_v) on validation and test, with r, alpha, beta;
      share of cells with RootingDepth < 10 mm.
- [ ] **Decide the RT0 version** (user).

## 2. Final dPL run with the winner

Script: `run_final_campaign.py`. Continues the winner's cycles 0-1 up to cycle 7. ~60-70 h.

- [x] Script written; `dpl_final/` seeded from c1ap (theta) (2026-10-18)
- [ ] If the winner is NOT theta: delete `dpl_final/`, point `SRC` in `run_final_campaign.py` at the winner's folder, set `rdc.OBS` to its RT0, seed again.
- [ ] `--check`
- [ ] Run cycles 2-7
- [ ] Table of scores per cycle; apply the convergence rule; choose the cycle to analyse (user).

## 2b. Seed sensitivity of the transfer function (after step 2)

With the surrogate of the chosen cycle (`CudaLSTM_<N>.pt`), run the calibration 10 times with
different seeds, to see how much the seed changes the learned parameters (the neural transfer function).

- [ ] Choose the cycle N (from step 2).
- [x] Starting point: random initial weights per seed (user, 2026-10-18; first chose cycle N-1,
      changed because a shared start hides the effect of the initialisation).
      Check before writing the script: on 2026-09-22 an untrained network gave ~0 in scaled space for
      every parameter with 3 seeds, i.e. every parameter starts near its lower bound. Different
      weights may still mean nearly the same starting parameter maps.
- [x] Score: surrogate KGE against RT0 per seed, no wflow runs (user, 2026-10-18; no time for 10 wflow runs).
      Reading: similar surrogate skill with divergent parameter maps = equifinality;
      different skill = poor convergence.
      Limitation: this is equifinality as seen by the surrogate. Interpret each parameter with its
      per-parameter fidelity (step 3): divergence in a parameter the surrogate barely responds to
      (c, Sl so far) is surrogate insensitivity.
- [x] Script `run_transfer_seed_sensitivity.py --cycle N` written (2026-10-18), not run.
      Outputs: `A_DROP/pilot5/article/transfer_seed_sensitivity/cycle_<N>/seed_<s>/`, scores in `summary.json`.
      Seeds 0-9; the seed sets the initial weights and the batch order. RT0 is read from the cycle-N calibration config.
- [ ] `--cycle N --check`
- [ ] Run the 10 calibrations (~5 h each, ~50 h). Not at the same time as another campaign.
- [ ] Compare across seeds: parameter maps (spread per cell and per parameter) and surrogate KGE vs RT0
      (same cells, same period, same weights as the calibration).

## 3. Analyses on the final run (after step 2)

### Transfer error (new notebook, nothing taken from the old C6 notebooks) — design agreed 2026-10-18

Built and tested on `dpl_final` cycle 1 (= c1ap copy) now; `CAMPAIGN` and `CYCLE` set at the top.
Notebook `article/param_estimation/DPL_TRANSFER_ERROR.ipynb`; outputs in
`A_DROP/pilot5/article/dpl_final_analysis/transfer_error/cycle_<N>/` (see "Where things are").

- Target, per cell: G_sur = KGE(sur(θ*),RT0) − KGE(sur(θ0),RT0); G_wflow = same for wflow;
  ΔKGE_transf = G_sur − G_wflow (signed; > 0 = surrogate oversells the gain).
- KGE weighted (1, 0.25, 0.75) as main, standard (1, 1, 1) as check. Period 2021-2022, 2022 as check.
  dPL dynamic RT0 mask, >= 20 obs per cell, same cells for all four KGEs.
  Surrogate predictions on the full map: reuse the prediction code of `c6_transfer_inputs.py`.
- [x] 1. Scatter G_sur vs G_wflow with the 1:1 line; map-level medians of G_sur and G_wflow.
      Run on cycle 1 (2026-10-18): median G_sur 0.061, G_wflow 0.028, ΔKGE_transf 0.030, 72% > 0;
      r(G_sur, G_wflow) 0.59. Same picture with standard KGE and with 2022 only.
- [ ] 2. Noise band per cell (bootstrap over RT0 days) -> share of cells < 0 / ≈ 0 / > 0, and a map.
      Written 2026-10-18, not run. Proposed settings (to confirm): 1000 resamples, blocks of 10 days,
      90% interval decides the class.
- [ ] 3. Regression of signed ΔKGE_transf: GAM + linear model, no interactions; importance per predictor
      group from spatial block cross-validation (no p-values).
      Predictors (all computed; subset chosen later), all at θ0 unless stated:
      surrogate vs wflow (bias, |log α|, 1 − r; one term expected), wflow vs RT0 (bias, |log α|, 1 − r),
      N_RT0, rough, flat (kept for now; drop later if redundant with N_RT0), PE (computed; ρ = 0.06 in c1ap),
      elevation, DI.
      DI = dissimilarity index (Meyer & Pebesma 2021): distance to the nearest training point / mean distance
      between training points; 5 calibrated parameters scaled 0-1 by their bounds, unweighted;
      training points = all archive runs x pool cells; computed at θ* and at θ0.
      Written 2026-10-18, not run. Proposed choices (to confirm): rough and PE on successive observations
      (not consecutive days); flat = sd of RT0; bias signed; su/mt on the KGE days; lakes and missing cells
      excluded; blocks of 0.5°, 5 folds; importance as "R² lost without the group" and "R² of the group alone".
      First run (cycle 1, unclipped): out-of-fold R² linear 0.026, GAM 0.078; elevation strongest (GAM only,
      non-linear); DI explains nothing. Predictors now clipped at 1/99 percentiles (user, 2026-10-18).
- Response fidelity (slope, r) is NOT a predictor: it is the step between the causes and the transfer
  error (mediator); used in the discussion.
- Dropped for now: R² ceiling, KGE component split, normalised gains, location-scale model, interactions.

### Other analyses

- [ ] Add a `final` campaign to `c6_transfer_inputs.py` (paths under `dpl_final/`).
- [ ] Per-parameter fidelity and KGE for every cycle (all inputs are kept per cycle).
- [ ] Update the C6.1 / C6.2 notebooks (new copies, own figure suffix).
- [ ] Decide whether the test score in the article is 2022 or 2021-2022 (both can be computed from the kept outputs).

## 4. Article text

- [ ] Paragraph on how the wflow outputs were subsampled for surrogate training (draft written 2026-10-18, check the numbers against the final run: 5 runs at cycle 0 -> 4,400 cells x 5 runs).
- [ ] Sentence on the KGE weights.
- [ ] Sentence on the reduced 2022 RT0 coverage (Sentinel-1B).

## Open items, not blocking

- [ ] Rebuild the c1 files in `emo1/` that a test run overwrote on 2026-10-05 (only `c6_transfer_inputs.py --campaign c1` needs them).
- [ ] Test guard so pytest can never write to production folders (proposed, not approved).
- [ ] C6.2 one-parameter wflow runs (`c62_param_runs_c1ap.py`): redo for the final run instead of c1ap?
- [x] `persistent_workers` timing test (2026-10-18, data loading only, c1ap cycle-1 calibration datasets, 24 workers, 16-core VM):
      saves ~1 s (half-year window) to ~2.3 s (1-year window) per loader per epoch, i.e. ~2-5 s per epoch,
      ~2-3% of a ~5 min calibration epoch. Peak memory +0 to +4 GB. Not worth changing the trainer for speed (user's call).
