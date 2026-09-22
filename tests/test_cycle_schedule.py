"""Per-cycle run schedule and wflow retry (A2 a).

Cycle 0 is the only time the surrogate sees parameters far from the current
guess - its members are spread across the physical range by a Latin hypercube.
Later cycles carry no level offset, so one run supplies the per-cell noise just
as well, and the archive stops growing four times faster than it needs to.

That makes a failed wflow run expensive: with one run per cycle, a failure
means the cycle adds nothing at all. Hence the retry.
"""

import subprocess as sp
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
run_dpl_cycle = pytest.importorskip("run_dpl_cycle")

CycleConfig = run_dpl_cycle.CycleConfig


def test_cycle_zero_runs_several_members():
    assert CycleConfig().members_for(0) == 4


def test_later_cycles_run_one():
    cfg = CycleConfig()
    for cycle in range(1, cfg.n_cycles):
        assert cfg.members_for(cycle) == 1


def test_total_runs_over_eight_cycles():
    """4 + 1 per cycle would be 32 archive runs; this schedule gives 11."""
    cfg = CycleConfig()
    assert sum(cfg.members_for(c) for c in range(cfg.n_cycles)) == 11


def test_jitter_does_not_anneal_to_nothing():
    """After cycle 0 the per-cell noise is the only local signal there is, so
    `jitter_end` must not fall back to the old 0.03."""
    cfg = CycleConfig()
    assert 0.05 <= cfg.jitter_end <= 0.08
    assert cfg.jitter_end < cfg.jitter_start


def test_members_are_built_at_the_scheduled_count(monkeypatch):
    """`build_members` must follow the schedule, not `n_members` directly."""
    import numpy as np
    import xarray as xr

    cfg = CycleConfig()
    names = list(run_dpl_cycle.CAL_PARAMS)
    theta = xr.Dataset({n: ("cell", np.full(5, 0.5)) for n in names})
    monkeypatch.setattr(run_dpl_cycle, "perturb", lambda d, lo, hi, j, o, r: d)

    rng = np.random.default_rng(0)
    assert len(run_dpl_cycle.build_members(0, cfg, theta, rng)) == 4
    assert len(run_dpl_cycle.build_members(3, cfg, theta, rng)) == 1


def test_compression_is_lowered_but_not_disabled():
    """A2(f). Level 5 costs 20% of the simulation time for 2% of the disk.
    Level 0 is not the alternative - it is 2.5x the size."""
    assert CycleConfig().output_compression == 1


# ==== the retry


class FakeRun:
    """Fails `fail_times` times, then succeeds."""

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def __call__(self, cmd, *a, **kw):
        self.calls += 1
        rc = 1 if self.calls <= self.fail_times else 0
        return sp.CompletedProcess(cmd, rc)


@pytest.fixture
def wflow(tmp_path, monkeypatch):
    monkeypatch.setattr(run_dpl_cycle, "WD_WFLOW", tmp_path)
    (tmp_path / "wflow_sbm_workflow.toml").write_text(
        'starttime = "2016-12-31T00:00:00"\n[input]\npath_static = "x.nc"\n'
        '[output]\npath = "y.nc"\n[csv]\npath = "y.csv"\n'
    )
    return tmp_path


def test_the_shared_template_is_never_modified(wflow, monkeypatch):
    """Every future loop run reads it. A leftover setting there would apply to
    real runs silently."""
    template = wflow / "wflow_sbm_workflow.toml"
    before = template.read_text()
    monkeypatch.setattr(run_dpl_cycle.sp, "run", FakeRun(0))
    run_dpl_cycle.run_wflow(wflow / "s.nc", wflow / "o.nc", CycleConfig())
    assert template.read_text() == before


def test_compression_is_set_on_the_per_run_toml(wflow, monkeypatch):
    import tomllib
    monkeypatch.setattr(run_dpl_cycle.sp, "run", FakeRun(0))
    run_dpl_cycle.run_wflow(wflow / "s.nc", wflow / "o.nc", CycleConfig())
    written = tomllib.loads((wflow / "wflow_sbm_o.toml").read_text())
    assert written["output"]["compressionlevel"] == 1


def test_a_run_that_works_is_not_repeated(wflow, monkeypatch):
    fake = FakeRun(0)
    monkeypatch.setattr(run_dpl_cycle.sp, "run", fake)
    run_dpl_cycle.run_wflow(wflow / "s.nc", wflow / "o.nc", CycleConfig())
    assert fake.calls == 1


def test_a_flaky_run_is_retried_and_succeeds(wflow, monkeypatch):
    fake = FakeRun(1)
    monkeypatch.setattr(run_dpl_cycle.sp, "run", fake)
    run_dpl_cycle.run_wflow(wflow / "s.nc", wflow / "o.nc", CycleConfig())
    assert fake.calls == 2


def test_a_run_that_never_works_still_raises(wflow, monkeypatch):
    """Retrying must not turn a real failure into a silent one - the cycle
    would then ingest a missing or stale output."""
    fake = FakeRun(99)
    monkeypatch.setattr(run_dpl_cycle.sp, "run", fake)
    with pytest.raises(sp.CalledProcessError):
        run_dpl_cycle.run_wflow(wflow / "s.nc", wflow / "o.nc", CycleConfig())
    assert fake.calls == CycleConfig().wflow_retries + 1


# ==== pruning finished cycles


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A run_default holding outputs for cycles 0-3."""
    monkeypatch.setattr(run_dpl_cycle, "WD_WFLOW", tmp_path)
    d = tmp_path / "run_default"
    d.mkdir()
    cfg = CycleConfig()
    for cycle in range(4):
        for m in range(cfg.members_for(cycle)):
            (d / f"output_cycle{cycle}_m{m}.nc").write_bytes(b"x" * 100)
            (d / f"output_cycle{cycle}_m{m}.csv").write_text("q")
        (d / f"output_cycle{cycle}_cal.nc").write_bytes(b"x" * 100)
        (d / f"output_cycle{cycle}_cal.csv").write_text("q")
    return d


def names(d):
    return {p.name for p in d.iterdir()}


def test_cycle_zero_is_never_pruned(run_dir):
    """Kept for figures and for checking the surrogate against the full map."""
    run_dpl_cycle.prune_cycle_outputs(0, CycleConfig())
    assert "output_cycle0_m0.nc" in names(run_dir)
    assert "output_cycle0_cal.nc" in names(run_dir)


def test_a_finished_cycle_loses_its_netcdfs(run_dir):
    run_dpl_cycle.prune_cycle_outputs(2, CycleConfig())
    assert "output_cycle2_m0.nc" not in names(run_dir)
    assert "output_cycle2_cal.nc" not in names(run_dir)


def test_the_csv_survives(run_dir):
    """Small, and it holds the gauge discharge."""
    run_dpl_cycle.prune_cycle_outputs(2, CycleConfig())
    assert "output_cycle2_m0.csv" in names(run_dir)
    assert "output_cycle2_cal.csv" in names(run_dir)


def test_other_cycles_are_untouched(run_dir):
    before = {n for n in names(run_dir) if "cycle2" not in n}
    run_dpl_cycle.prune_cycle_outputs(2, CycleConfig())
    assert {n for n in names(run_dir) if "cycle2" not in n} == before


def test_pruning_can_be_switched_off(run_dir):
    import dataclasses
    cfg = dataclasses.replace(CycleConfig(), prune_outputs=False)
    run_dpl_cycle.prune_cycle_outputs(2, cfg)
    assert "output_cycle2_m0.nc" in names(run_dir)


def test_pruning_a_cycle_that_left_nothing_is_harmless(run_dir):
    """`skip_wflow` runs leave no outputs at all."""
    run_dpl_cycle.prune_cycle_outputs(3, CycleConfig())
    run_dpl_cycle.prune_cycle_outputs(3, CycleConfig())  # twice, still fine


# ==== sequential or parallel wflow (A2 f)


import dataclasses


def cfg_with(**kw):
    return dataclasses.replace(CycleConfig(), **kw)


def test_threads_are_split_across_concurrent_runs():
    """One run plateaus at ~8 threads, so `julia_threads` is a total budget."""
    assert cfg_with(wflow_parallel=1).threads_per_run() == 24
    assert cfg_with(wflow_parallel=3).threads_per_run() == 8
    assert cfg_with(wflow_parallel=4).threads_per_run() == 6


def test_a_run_always_gets_at_least_one_thread():
    assert cfg_with(wflow_parallel=99).threads_per_run() == 1


def test_sequential_is_the_default():
    assert CycleConfig().wflow_parallel == 1


def test_waves_are_one_member_each_when_sequential():
    members = ["a", "b", "c", "d"]
    waves = list(run_dpl_cycle.member_waves(members, cfg_with(wflow_parallel=1)))
    assert [len(w) for w in waves] == [1, 1, 1, 1]


def test_waves_group_members_when_parallel():
    members = ["a", "b", "c", "d"]
    waves = list(run_dpl_cycle.member_waves(members, cfg_with(wflow_parallel=3)))
    assert [len(w) for w in waves] == [3, 1]


def test_every_member_appears_exactly_once_and_in_order():
    """Archive run indices must not depend on the wave width."""
    members = list("abcdefg")
    for width in (1, 2, 3, 4, 99):
        waves = list(run_dpl_cycle.member_waves(members, cfg_with(wflow_parallel=width)))
        flat = [item for wave in waves for item in wave]
        assert flat == list(enumerate(members)), width


def test_concurrent_runs_all_execute(wflow, monkeypatch):
    seen = []
    monkeypatch.setattr(run_dpl_cycle, "run_wflow",
                        lambda s, o, c: seen.append(o.name))
    jobs = [(wflow / f"s{i}.nc", wflow / f"o{i}.nc") for i in range(4)]
    run_dpl_cycle.run_wflow_concurrently(jobs, cfg_with(wflow_parallel=4))
    assert sorted(seen) == [f"o{i}.nc" for i in range(4)]


def test_one_job_does_not_start_a_pool(wflow, monkeypatch):
    seen = []
    monkeypatch.setattr(run_dpl_cycle, "run_wflow",
                        lambda s, o, c: seen.append(o.name))
    run_dpl_cycle.run_wflow_concurrently([(wflow / "s.nc", wflow / "o.nc")],
                                         cfg_with(wflow_parallel=4))
    assert seen == ["o.nc"]


def test_a_failure_in_one_job_still_raises(wflow, monkeypatch):
    """And the other jobs are waited for first - a julia process left running
    against a half-written output is worse than the delay."""
    done = []

    def fake(s, o, c):
        if o.name == "o2.nc":
            raise sp.CalledProcessError(1, "julia")
        done.append(o.name)

    monkeypatch.setattr(run_dpl_cycle, "run_wflow", fake)
    jobs = [(wflow / f"s{i}.nc", wflow / f"o{i}.nc") for i in range(4)]
    with pytest.raises(sp.CalledProcessError):
        run_dpl_cycle.run_wflow_concurrently(jobs, cfg_with(wflow_parallel=4))
    assert sorted(done) == ["o0.nc", "o1.nc", "o3.nc"]


# ==== the clean scoring run is optional (it costs a full wflow simulation)


def test_scoring_every_second_cycle_is_the_default():
    """Half the clean runs, and `rel_tol` still works on alternate cycles."""
    cfg = CycleConfig()
    assert cfg.score_every == 2
    assert [c for c in range(cfg.n_cycles) if cfg.scores_cycle(c)] == [0, 2, 4, 6]


def test_scoring_can_be_switched_off_entirely():
    cfg = cfg_with(score_every=0)
    assert not any(cfg.scores_cycle(c) for c in range(cfg.n_cycles))


def test_scoring_can_be_thinned():
    cfg = cfg_with(score_every=3)
    assert [c for c in range(9) if cfg.scores_cycle(c)] == [0, 3, 6]


def test_a_negative_value_is_treated_as_off():
    assert not cfg_with(score_every=-1).scores_cycle(0)


def test_wflow_runs_saved_over_a_campaign():
    """The point of the switch: one wflow run per skipped cycle."""
    cfg = CycleConfig()
    default = sum(cfg.scores_cycle(c) for c in range(cfg.n_cycles))
    every = sum(cfg_with(score_every=1).scores_cycle(c) for c in range(cfg.n_cycles))
    never = sum(cfg_with(score_every=0).scores_cycle(c) for c in range(cfg.n_cycles))
    assert (default, every, never) == (4, 8, 0)


def test_convergence_cannot_fire_without_scores():
    """`converged` needs two entries; with scoring off there are none, so the
    loop runs the full n_cycles rather than stopping early."""
    assert run_dpl_cycle.converged([], 0.01) is False
    assert run_dpl_cycle.converged([{"kge_valid": 0.2}], 0.01) is False


def test_convergence_still_works_on_thinned_scores():
    """With score_every > 1 the rule compares the cycles that were scored."""
    h = [{"kge_valid": 0.230, "rmse": 0.10}, {"kge_valid": 0.2301, "rmse": 0.12}]
    assert run_dpl_cycle.converged(h, 0.01) is True


def test_first_and_last_only():
    """`score_every: 0` alone scores nothing - `score_first_last` is what gives
    you the two endpoints."""
    cfg = cfg_with(score_every=0, score_first_last=True)
    scored = [c for c in range(cfg.n_cycles) if cfg.scores_cycle(c)]
    assert scored == [0, cfg.n_cycles - 1]


def test_first_last_is_off_by_default():
    assert CycleConfig().score_first_last is False


def test_first_last_adds_to_a_thinned_schedule():
    """It is a floor, not a replacement: k-th cycles are still scored."""
    cfg = cfg_with(score_every=3, score_first_last=True, n_cycles=8)
    scored = [c for c in range(8) if cfg.scores_cycle(c)]
    assert scored == [0, 3, 6, 7]


def test_two_runs_instead_of_eight():
    cfg = cfg_with(score_every=0, score_first_last=True)
    assert sum(cfg.scores_cycle(c) for c in range(cfg.n_cycles)) == 2


def test_convergence_cannot_stop_the_loop_early_in_this_mode():
    """Only the last cycle produces a second score, so `converged()` is first
    evaluated when there is nothing left to skip."""
    cfg = cfg_with(score_every=0, score_first_last=True)
    scored = [c for c in range(cfg.n_cycles) if cfg.scores_cycle(c)]
    assert scored[-1] == cfg.n_cycles - 1 and len(scored) == 2
