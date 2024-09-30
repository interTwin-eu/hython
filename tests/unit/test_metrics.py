import pytest
import numpy as np

from hython.hython.metrics.standard import MSEMetric, RMSEMetric, compute_mse

TARGETS = ["vwc", "actevap"]


def test_1d():
    b = a = np.random.randn(100)
    ret = compute_mse(a, b)
    assert ret == 0


def test_2d():
    b = a = np.random.randn(100, 2)
    ret = compute_mse(a, b)
    assert np.all([ret[t] == 0 for t in range(a.shape[1])])


def test_mse_class():
    b = a = np.random.randn(100, 2)
    ret = MSEMetric()(a, b, TARGETS)

    assert np.all([ret[t] == 0 for t in TARGETS])