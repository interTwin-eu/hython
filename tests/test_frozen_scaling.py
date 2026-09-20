"""Shared full-map scaling numbers for the five statics (H4).

`wflow_uparea`, `wflow_landuse`, `wflow_dem`, `Slope` and `WaterFrac` are
MinMax01-scaled in both configs: inside `static_inputs` when training the
surrogate, and inside `head_model_inputs` when calibrating. Each side used to
work its own numbers out, and after A1 the training side sees only the 5% pool,
whose maxima fall well short of the map's. The same cell then means two
different numbers on the two sides, and nothing raises.

Synthetic and hermetic: the "pool" here is a narrow slice of the "full map", so
the gap these tests close is the same one the real pool has.
"""

import numpy as np
import pytest
import xarray as xr
import yaml
from omegaconf import OmegaConf

from hython.scaler import Scaler

FIVE = ["wflow_uparea", "wflow_landuse", "wflow_dem", "Slope", "WaterFrac"]
OTHER = ["KsatVer", "thetaS"]
N = 200


@pytest.fixture
def full_map():
    rng = np.random.default_rng(0)
    return xr.Dataset({v: ("cell", rng.uniform(0, 100, N)) for v in FIVE + OTHER})


@pytest.fixture
def pool(full_map):
    """A subset that misses the extremes, as a 5% draw does."""
    keep = np.argsort(full_map["WaterFrac"].values)[20:-20]
    return full_map.isel(cell=keep)


@pytest.fixture
def frozen_file(tmp_path, full_map):
    center = full_map[FIVE].min("cell")
    scale = full_map[FIVE].max("cell") - center
    path = tmp_path / "frozen.yaml"
    with open(path, "w") as f:
        yaml.dump({"center": center.to_dict(), "scale": scale.to_dict()}, f)
    return path


def make_cfg(tmp_path, frozen, group, variables):
    return OmegaConf.create({
        "experiment_name": "t", "experiment_run": "t", "work_dir": str(tmp_path),
        "scaling_frozen_stats": str(frozen) if frozen else None,
        group: variables,
        "scaler": {group: {"lazy": False, "variant": [
            {"_target_": "hython.scaler.MinMax01Scaler", "variable": variables}
        ]}},
    })


def scaled(data, cfg, group):
    from hydra.utils import instantiate
    cfg = instantiate(cfg)
    sc = Scaler(cfg)
    sc.load_or_compute(data, group, is_train=True, axes=("cell",))
    return sc.archive[group]


def test_pool_numbers_differ_from_the_map_without_freezing(tmp_path, full_map, pool):
    """The problem, stated. If this ever stops holding the rest is moot."""
    a = scaled(full_map, make_cfg(tmp_path, None, "static_inputs", FIVE), "static_inputs")
    b = scaled(pool, make_cfg(tmp_path, None, "static_inputs", FIVE), "static_inputs")
    assert not np.isclose(float(a["scale"]["WaterFrac"]), float(b["scale"]["WaterFrac"]))


def test_freezing_gives_the_pool_the_full_map_numbers(tmp_path, full_map, pool, frozen_file):
    want = scaled(full_map, make_cfg(tmp_path, None, "static_inputs", FIVE), "static_inputs")
    got = scaled(pool, make_cfg(tmp_path, frozen_file, "static_inputs", FIVE), "static_inputs")
    for v in FIVE:
        np.testing.assert_allclose(float(got["center"][v]), float(want["center"][v]))
        np.testing.assert_allclose(float(got["scale"][v]), float(want["scale"][v]))


def test_training_and_calibration_agree(tmp_path, full_map, pool, frozen_file):
    """The invariant H4 exists for: the five must mean the same thing on both
    sides, although they sit in different groups, in different files, computed
    from different data."""
    train = scaled(pool, make_cfg(tmp_path, frozen_file, "static_inputs", FIVE + OTHER),
                   "static_inputs")
    cal = scaled(full_map, make_cfg(tmp_path, frozen_file, "head_model_inputs", FIVE),
                 "head_model_inputs")
    for v in FIVE:
        np.testing.assert_allclose(float(train["center"][v]), float(cal["center"][v]))
        np.testing.assert_allclose(float(train["scale"][v]), float(cal["scale"][v]))


def test_one_cell_maps_to_one_number_on_both_sides(tmp_path, full_map, pool, frozen_file):
    """The same physical value must scale to the same number either way."""
    train = scaled(pool, make_cfg(tmp_path, frozen_file, "static_inputs", FIVE),
                   "static_inputs")
    cal = scaled(full_map, make_cfg(tmp_path, frozen_file, "head_model_inputs", FIVE),
                 "head_model_inputs")
    value = 0.85 * float(full_map["WaterFrac"].max())
    t = (value - float(train["center"]["WaterFrac"])) / float(train["scale"]["WaterFrac"])
    c = (value - float(cal["center"]["WaterFrac"])) / float(cal["scale"]["WaterFrac"])
    np.testing.assert_allclose(t, c)
    assert 0.0 <= t <= 1.0


def test_variables_not_in_the_frozen_file_keep_their_own_numbers(
    tmp_path, full_map, pool, frozen_file
):
    """Only the five are shared. The weather and the other statics stay
    pool-derived, which is what they should be."""
    own = scaled(pool, make_cfg(tmp_path, None, "static_inputs", FIVE + OTHER),
                 "static_inputs")
    got = scaled(pool, make_cfg(tmp_path, frozen_file, "static_inputs", FIVE + OTHER),
                 "static_inputs")
    for v in OTHER:
        np.testing.assert_allclose(float(got["scale"][v]), float(own["scale"][v]))


def test_a_missing_frozen_file_raises(tmp_path, pool):
    cfg = make_cfg(tmp_path, tmp_path / "nope.yaml", "static_inputs", FIVE)
    with pytest.raises(FileNotFoundError, match="scaling_frozen_stats"):
        scaled(pool, cfg, "static_inputs")


def test_cached_mode_raises_instead_of_recomputing_on_the_pool(tmp_path, pool):
    """The silent fallback H4 removes: recomputing on the pool would bring the
    whole problem back with nothing in the log."""
    from hydra.utils import instantiate
    cfg = instantiate(make_cfg(tmp_path, None, "static_inputs", FIVE))
    sc = Scaler(cfg, use_cached=True)
    with pytest.raises(FileNotFoundError, match="No cached statistics"):
        sc.load_or_compute(pool, "static_inputs", is_train=True, axes=("cell",))
