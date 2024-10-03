import numpy as np
import pytest

from hython.normalizer import Normalizer

shape = (10, 6, 3)  # gridcells, time, features
shape_static = (10, 3)  # gridcells, features

data_dynamic = np.random.random(shape)

data_static = np.random.random(shape_static)


@pytest.mark.parametrize(
    "method, type, axis_order, axis",
    [
        ("standardize", "spacetime", "NTC", (0, 1)),
        ("standardize", "space", "NTC", 0),
        ("standardize", "time", "NTC", 1),
        ("minmax", "spacetime", "NTC", (0, 1)),
        ("minmax", "space", "NTC", 0),
        ("minmax", "time", "NTC", 1),
    ],
)
def test_normalizer_dynamic(method, type, axis_order, axis):
    norm = Normalizer(method=method, type=type, axis_order=axis_order)
    normalized = norm.normalize(data_dynamic)

    if method == "standardize":
        m1 = np.nanmean(data_dynamic, axis=axis)
        m2 = np.nanstd(data_dynamic, axis=axis)
        expected = (data_dynamic - np.expand_dims(m1, axis=axis)) / np.expand_dims(
            m2, axis=axis
        )

    if method == "minmax":
        m1 = np.nanmin(data_dynamic, axis=axis)
        m2 = np.nanmax(data_dynamic, axis=axis)
        expected = (data_dynamic - np.expand_dims(m1, axis=axis)) / (
            np.expand_dims(m2, axis=axis) - np.expand_dims(m1, axis=axis)
        )

    assert np.allclose(normalized, expected)
