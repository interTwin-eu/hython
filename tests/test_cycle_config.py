"""Per-cycle config generation (H3, A2).

`write_cycle_config` materialises one config per cycle. The properties that
matter for H3 are cross-cycle: the train/valid partition must be identical in
every cycle's config, while `runs` must track the growing archive. Asserting
that on the *generated files* is what the downsampler unit tests cannot do -
they show `base_cells()` ignores `runs`, not that the orchestrator keeps
feeding the same `split_seed`.

Hermetic: writes into tmp_path, reads the real template, runs no pipeline.
"""

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

run_dpl_cycle = pytest.importorskip("run_dpl_cycle")

TEMPLATE = "config_training_calibration_loop"
SPLIT_KEYS = ("pool_size", "split", "valid_frac", "split_seed")


@pytest.fixture
def cycle_configs(tmp_path, monkeypatch):
    """The config the orchestrator would write for each of eight cycles.

    `runs` follows the archive: `n_members` runs at cycle 0, one per cycle
    after (A2 a).
    """
    monkeypatch.setattr(run_dpl_cycle, "WD_CYCLE", tmp_path)
    out = {}
    for cycle in range(8):
        runs = run_dpl_cycle.CycleConfig().n_members + cycle
        d, name = run_dpl_cycle.write_cycle_config(
            TEMPLATE,
            cycle,
            {"train_downsampler.runs": runs, "valid_downsampler.runs": runs},
        )
        out[cycle] = OmegaConf.load(d / f"{name}.yaml")
    return out


def test_split_keys_never_move_between_cycles(cycle_configs):
    """The H3 check the plan defers to a real run: a cell trained on at cycle 0
    must not be validated on at cycle n."""
    for block in ("train_downsampler", "valid_downsampler"):
        first = {k: cycle_configs[0][block][k] for k in SPLIT_KEYS}
        for cycle, cfg in cycle_configs.items():
            got = {k: cfg[block][k] for k in SPLIT_KEYS}
            assert got == first, f"{block} moved at cycle {cycle}: {got} != {first}"


def test_train_and_valid_name_opposite_splits(cycle_configs):
    for cfg in cycle_configs.values():
        assert cfg.train_downsampler.split == "train"
        assert cfg.valid_downsampler.split == "valid"


def test_runs_tracks_the_archive(cycle_configs):
    for cycle, cfg in cycle_configs.items():
        expected = run_dpl_cycle.CycleConfig().n_members + cycle
        assert cfg.train_downsampler.runs == expected
        assert cfg.valid_downsampler.runs == expected


def test_rows_per_epoch_stay_near_the_target(cycle_configs):
    """A2(b2): `places` shrinks as runs grow so the row budget holds."""
    from hython.sampler.downsampler import PoolDownsampler

    target = cycle_configs[0].train_rows_target
    for cycle, cfg in cycle_configs.items():
        d = PoolDownsampler(**{k: v for k, v in cfg.train_downsampler.items()
                               if k != "_target_"})
        rows = d._places(len(d.base_cells())) * d.runs
        assert rows <= target
        assert rows > target - d.runs, f"cycle {cycle}: {rows} rows vs {target}"


def test_template_is_not_mutated(cycle_configs):
    """A dead run must not leave the shared template in a cycle-n state."""
    template = OmegaConf.load(run_dpl_cycle.WD_CONFIG / f"{TEMPLATE}.yaml")
    assert template.train_downsampler.runs == 1


# ==== calibration must load the surrogate training just wrote


class _Stop(Exception):
    pass


def test_calibration_loads_the_surrogate_training_wrote(tmp_path, monkeypatch):
    """The calibration template names the surrogate through the training
    template's path, which carries the production `work_dir`. The 2026-09-21
    smoke redirected `work_dir` and so calibrated against a stale production
    surrogate. `calibrate` must point at `surrogate_weights()` explicitly."""
    monkeypatch.setattr(run_dpl_cycle, "WD_RUN", tmp_path / "runs")
    monkeypatch.setattr(run_dpl_cycle, "WD_CYCLE", tmp_path / "cycles")
    weights = run_dpl_cycle.surrogate_weights()
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"")

    def stop(out_dir, name):
        raise _Stop

    monkeypatch.setattr(run_dpl_cycle, "exec_pipeline", stop)
    with pytest.raises(_Stop):
        run_dpl_cycle.calibrate(0, run_dpl_cycle.CycleConfig())

    cfg = OmegaConf.load(tmp_path / "cycles" / "cycle_0" / "config_calibration_loop_c0.yaml")
    assert cfg.model_logger.CudaLSTM.model_uri == str(weights)


def test_calibration_refuses_to_run_without_a_surrogate(tmp_path, monkeypatch):
    monkeypatch.setattr(run_dpl_cycle, "WD_RUN", tmp_path / "runs")
    monkeypatch.setattr(run_dpl_cycle, "WD_CYCLE", tmp_path / "cycles")
    with pytest.raises(FileNotFoundError):
        run_dpl_cycle.calibrate(0, run_dpl_cycle.CycleConfig())
