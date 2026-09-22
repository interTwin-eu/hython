"""The loop's end: the last cycle always gets a clean wflow run and a score,
and member outputs are deleted after ingestion unless kept. `main()` runs
with wflow, training, calibration and ingestion replaced by stand-ins."""
import sys
from pathlib import Path

import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
rdc = pytest.importorskip("run_dpl_cycle")


@pytest.fixture
def loop(tmp_path, monkeypatch):
    wflow = tmp_path / "wflow"
    (wflow / "run_default").mkdir(parents=True)
    xr.Dataset({"x": ("a", [1.0])}).to_netcdf(wflow / "staticmaps.nc")
    monkeypatch.setattr(rdc, "WD_WFLOW", wflow)
    monkeypatch.setattr(rdc, "WD_CYCLE", tmp_path / "cycles")

    log = {"wflow": [], "ingested": [], "converge_after": None}

    def touch(path):
        Path(path).write_bytes(b"x")

    monkeypatch.setattr(rdc, "build_members",
                        lambda n, cfg, theta, rng: [object()] * cfg.members_for(n))
    monkeypatch.setattr(rdc, "write_staticmaps", lambda theta, dest: touch(dest))
    monkeypatch.setattr(rdc, "run_wflow_concurrently",
                        lambda jobs, cfg: [touch(o) for _, o in jobs])

    def ingest(static_nc, out_nc, cycle, member, state):
        assert out_nc.exists(), "ingested before the output existed"
        log["ingested"].append((cycle, member))

    monkeypatch.setattr(rdc, "ingest_member", ingest)
    monkeypatch.setattr(rdc.pool_archive, "verify", lambda: None)
    monkeypatch.setattr(rdc, "train_surrogate", lambda n, cfg: None)

    def calibrate(n, cfg):
        dest = rdc.WD_CYCLE / f"cycle_{n}" / "theta_cal.nc"
        touch(dest)
        return dest, {"kgecell": 0.3}

    monkeypatch.setattr(rdc, "calibrate", calibrate)
    monkeypatch.setattr(rdc, "read_theta", lambda p: None)

    def run_wflow(static, out, cfg):
        log["wflow"].append(Path(out).name)
        touch(out)

    monkeypatch.setattr(rdc, "run_wflow", run_wflow)
    monkeypatch.setattr(rdc, "score", lambda out, cfg: {"kge_valid": 0.2})
    monkeypatch.setattr(rdc, "score_baseline", lambda cfg: {"kge_valid": 0.1})
    monkeypatch.setattr(rdc, "relative_to_baseline", lambda new, base: {})
    monkeypatch.setattr(rdc, "has_converged",
                        lambda state, cfg: log["converge_after"] is not None
                        and state.cycle - 1 >= log["converge_after"])
    return log


def _cfg(**kw):
    base = dict(n_cycles=4, n_members=2, n_members_later=1, score_every=2)
    base.update(kw)
    return rdc.CycleConfig(**base)


def _state():
    return rdc.CycleState.load(rdc.WD_CYCLE / "state.json")


def test_an_unscored_last_cycle_is_scored_at_the_end(loop):
    """score_every 2 over 4 cycles scores 0 and 2; cycle 3 is the result."""
    rdc.main(_cfg())
    st = _state()
    assert st.scored_cycles == [0, 2, 3]
    assert len(st.history) == 3
    assert loop["wflow"][-1] == "output_cycle3_cal.nc"
    assert st.finished


def test_a_scored_last_cycle_is_not_run_twice(loop):
    rdc.main(_cfg(n_cycles=3))
    assert _state().scored_cycles == [0, 2]
    assert loop["wflow"].count("output_cycle2_cal.nc") == 1


def test_an_early_stop_at_an_unscored_cycle_is_scored(loop):
    loop["converge_after"] = 1
    rdc.main(_cfg())
    st = _state()
    assert st.cycle == 2
    assert st.scored_cycles == [0, 1]
    assert st.finished


def test_a_rerun_after_the_end_starts_no_new_cycle(loop):
    rdc.main(_cfg())
    runs = list(loop["wflow"])
    rdc.main(_cfg())
    assert loop["wflow"] == runs
    assert _state().scored_cycles == [0, 2, 3]


def test_member_outputs_are_deleted_after_ingestion(loop):
    rdc.main(_cfg(n_cycles=2))
    run_dir = rdc.WD_WFLOW / "run_default"
    assert loop["ingested"] == [(0, 0), (0, 1), (1, 0)]
    assert not list(run_dir.glob("output_cycle*_m*.nc"))
    # the clean runs are kept
    assert (run_dir / "output_cycle0_cal.nc").exists()
    assert (run_dir / "output_cycle1_cal.nc").exists()


def test_member_outputs_can_be_kept(loop):
    rdc.main(_cfg(n_cycles=1, keep_member_outputs=True))
    run_dir = rdc.WD_WFLOW / "run_default"
    assert sorted(p.name for p in run_dir.glob("output_cycle0_m*.nc")) == [
        "output_cycle0_m0.nc", "output_cycle0_m1.nc"]
