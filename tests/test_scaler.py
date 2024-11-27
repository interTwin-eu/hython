import numpy as np
import pytest
import xarray as xr

from hython.normalizer import Scaler



def test_minmax_lstm():
    scaler = Scaler('./config.yaml')

    assert scaler.cfg.scaling_variant

@pytest.mark.parametrize(
    "use_cached, data_type",
    [
        (False,"xarray"),
        (True,"xarray"),
        (False,"numpy"),
        (True,"numpy")
    ],
)
def test_load_or_compute_caching(use_cached, data_type):
    scaler = Scaler('./config.yaml', use_cached=use_cached)

    if data_type == "xarray":
        
        axis = ("cell", "time")
        data = np.random.randint(0, 100, (10,100, 3))
        data = xr.Dataset({
            "vwc":(["cell", "time", "feat"], data)
        })
        
    else:
        axis = (0,1)
        data = np.random.randint(0, 100, (10,100, 3))


    scaler.load_or_compute(data, "dynamic_input", axes=axis)
    
    assert scaler.archive.get("dynamic_input") is not None

    if use_cached:
        if data_type == "numpy":
            assert isinstance(scaler.archive.get("dynamic_input")["center"], np.ndarray)
        else:
            assert isinstance(scaler.archive.get("dynamic_input")["center"], xr.Dataset)
    else:
        scaler.write("dynamic_input")

        if data_type == "numpy":
            assert isinstance(scaler.archive.get("dynamic_input")["center"], np.ndarray)
        else:
            assert isinstance(scaler.archive.get("dynamic_input")["center"], xr.Dataset)

@pytest.mark.parametrize(
    "method, data_type",
    [
        ("standard", "xarray"),
        ("standard", "numpy"),
        ("minmax", "xarray"),
        ("minmax", "numpy"),
    ],
)
def test_transform(method, data_type):
    scaler = Scaler('./config.yaml', use_cached=False)

    scaler.cfg.scaling_variant = method

    if data_type == "numpy":
        axis = (0,1)
        data = np.random.randint(0, 100, (10,100, 3))
    else:
        axis = ("cell", "time")
        data = np.random.randint(0, 100, (10,100, 3))
        data = xr.Dataset({
            "vwc":(["cell", "time", "feat"], data)
        })
        
    scaler.load_or_compute(data, "dynamic_input", axes=axis)
    
    data_norm = scaler.transform(data, "dynamic_input")

    if method == "minmax":
        if data_type == "xarray":
            data1 = xr.Dataset({"vwc":(["feat"], np.array([0,0,0]))})
            data2 = xr.Dataset({
                "vwc":(["feat"], np.array([1,1,1]))
                })

            xr.testing.assert_allclose(data1, data_norm.min(dim=axis))
            xr.testing.assert_allclose(data2, data_norm.max(dim=axis))
        else:
            
            assert np.allclose(np.array([0,0,0]), data_norm.min(axis))
            assert np.allclose(np.array([1,1,1]), data_norm.max(axis))    
    elif method == "standard":
        if data_type == "xarray":
            data1 = xr.Dataset({
                "vwc":(["feat"], np.array([0,0,0]))
                })
            data2 = xr.Dataset({
                "vwc":(["feat"], np.array([1,1,1]))
                })
            xr.testing.assert_allclose(data1, data_norm.mean(dim=axis))
            xr.testing.assert_allclose(data2, data_norm.std(dim=axis)) 
        else:

            assert np.allclose(np.array([0,0,0]), data_norm.mean(axis))
            assert np.allclose(np.array([1,1,1]), data_norm.std(axis)) 

@pytest.mark.parametrize(
    "method, data_type",
    [
        ("standard", "xarray"),
        ("standard", "numpy"),
        ("minmax", "xarray"),
        ("minmax", "numpy"),
    ],
)
def test_inverse_transform(method, data_type):
    scaler = Scaler('./config.yaml', use_cached=False)

    scaler.cfg.scaling_variant = method

    if data_type == "numpy":
        axis = (0,1)
        data = np.random.randint(0, 100, (10,100, 3))
    else:
        axis = ("cell", "time")
        data = np.random.randint(0, 100, (10,100, 3))
        data = xr.Dataset({
            "vwc":(["cell", "time", "feat"], data)
        })
        
    scaler.load_or_compute(data, "dynamic_input", axes=axis)
    
    data_norm = scaler.transform(data, "dynamic_input")

    data_reconstructed = scaler.transform_inverse(data_norm, "dynamic_input")

    if method == "minmax":
        if data_type == "xarray":
            xr.testing.assert_allclose(data, data_reconstructed)
        else:      
            assert np.allclose(data, data_reconstructed)
  
    elif method == "standard":
        if data_type == "xarray":
            xr.testing.assert_allclose(data, data_reconstructed)
        else:
            assert np.allclose(data, data_reconstructed)


@pytest.mark.parametrize(
    "method, data_type",
    [
        ("standard", "xarray"),
        ("standard", "numpy"),
        ("minmax", "xarray"),
        ("minmax", "numpy"),
    ],
)
def test_transform_missing(method, data_type):
    """Xarray by default skip nans"""
    scaler = Scaler('./config.yaml', use_cached=False)

    scaler.cfg.scaling_variant = method

    if data_type == "numpy":
        axis = (0,1)
        data = np.random.randint(0, 100, (10,100, 3)).astype(float)
        data[0, 20, 0] = np.nan
    else:
        axis = ("cell", "time")
        data = np.random.randint(0, 100, (10,100, 3)).astype(float)
        data[0, 20, 0] = np.nan
        data = xr.Dataset({
            "vwc":(["cell", "time", "feat"], data)
        })
        
    scaler.load_or_compute(data, "dynamic_input", axes=axis)
    
    data_norm = scaler.transform(data, "dynamic_input")

    if method == "minmax":
        if data_type == "xarray":
            data1 = xr.Dataset({"vwc":(["feat"], np.array([0,0,0]))})
            data2 = xr.Dataset({
                "vwc":(["feat"], np.array([1,1,1]))
                })

            xr.testing.assert_allclose(data1, data_norm.min(dim=axis))
            xr.testing.assert_allclose(data2, data_norm.max(dim=axis))
        else:
            
            assert np.allclose(np.array([0,0,0]), data_norm.min(axis))
            assert np.allclose(np.array([1,1,1]), data_norm.max(axis))    
    elif method == "standard":
        if data_type == "xarray":
            data1 = xr.Dataset({
                "vwc":(["feat"], np.array([0,0,0]))
                })
            data2 = xr.Dataset({
                "vwc":(["feat"], np.array([1,1,1]))
                })
            xr.testing.assert_allclose(data1, data_norm.mean(dim=axis))
            xr.testing.assert_allclose(data2, data_norm.std(dim=axis)) 
        else:

            assert np.allclose(np.array([0,0,0]), data_norm.mean(axis))
            assert np.allclose(np.array([1,1,1]), data_norm.std(axis)) 
