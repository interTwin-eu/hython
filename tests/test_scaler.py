import numpy as np
import pytest


from hython.normalizer import Scaler



def test_minmax_lstm():
    scaler = Scaler('./config.yaml')

    assert scaler.cfg.scaling_variant

@pytest.mark.parametrize(
    "use_cached",
    [
        True, False
    ],
)
def test_load_or_compute(use_cached):
    scaler = Scaler('./config.yaml', use_cached=use_cached)

    data = np.random.randint(0, 100, (10,100, 3))

    scaler.load_or_compute(data, "dynamic_input")
    
    assert scaler.archive.get("dynamic_input") is not None
    if use_cached:
        assert isinstance(scaler.archive.get("dynamic_input")["center"], np.ndarray)
    else:
        scaler.write("dynamic_input")

        assert isinstance(scaler.archive.get("dynamic_input")["center"], np.ndarray)

@pytest.mark.parametrize(
    "method",
    [
        "standard", "minmax"
    ],
)
def test_transform(method):
    scaler = Scaler('./config.yaml', use_cached=False)

    scaler.cfg.scaling_variant = method

    data = np.random.randint(0, 100, (10,100, 3))

    scaler.load_or_compute(data, "dynamic_input")
    
    data_norm = scaler.transform(data, "dynamic_input")
    if method == "minmax":
        assert np.allclose(np.array([0,0,0]), data_norm.min((0,1)) )
        assert np.allclose(np.array([1,1,1]), data_norm.max((0,1)) )
    elif method == "standard":
        assert np.allclose(np.array([0,0,0]), data_norm.mean((0,1)) )
        assert np.allclose(np.array([1,1,1]), data_norm.std((0,1)) )

