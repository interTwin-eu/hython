"""ParameterInference turns the parameter network's scaled output back into
physical values with the calibration config's own bounds. It used to load a
hard-coded article config and its cached scaler statistics for this."""
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from omegaconf import OmegaConf

inference = pytest.importorskip("hython.itwinai.inference")
from hython.scaler import BoundedScaler, get_scaling_parameter  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "scripts" / "config" / "config_calibration_loop.yaml"


def _scaled(names, seed=0):
    rng = np.random.default_rng(seed)
    return xr.Dataset({p: (("lat", "lon"), rng.uniform(-0.05, 1.05, (4, 5))) for p in names})


def test_matches_bounded_scaler_inverse():
    """The same numbers the old path produced: BoundedScaler's inverse."""
    cfg = OmegaConf.load(CONFIG)
    names = list(cfg.head_model_inputs.cal_param)
    ranges = {p: list(cfg.scaling_static_range[p]) for p in names}
    ds = _scaled(names)

    got = inference.unscale_parameters(ds, ranges)

    center, scale = get_scaling_parameter(ranges, output_type="xarray")
    want = BoundedScaler(ranges).transform_inverse(ds, center, scale)
    for p in names:
        np.testing.assert_allclose(got[p].values, want[p].values, rtol=1e-12)


def test_bounds_map_to_the_ends():
    ranges = {"KsatVer": [1, 8000]}
    ds = xr.Dataset({"KsatVer": ("x", [0.0, 1.0])})
    assert list(inference.unscale_parameters(ds, ranges)["KsatVer"].values) == [1, 8000]


def test_trainer_block_passes_the_bounds():
    """The trainer's config block - what ParameterInference receives - must
    carry the same bounds and parameter list as the top of the config."""
    cfg = OmegaConf.load(CONFIG)
    block = next(s for s in cfg.training_pipeline.steps if "config" in s)["config"]
    assert OmegaConf.to_container(block.scaling_static_range, resolve=True) == \
        OmegaConf.to_container(cfg.scaling_static_range, resolve=True)
    assert list(block.head_model_inputs.cal_param) == list(cfg.head_model_inputs.cal_param)


def test_no_hard_coded_config_left():
    src = Path(inference.__file__).read_text()
    assert "article" not in src and "OmegaConf.load" not in src
