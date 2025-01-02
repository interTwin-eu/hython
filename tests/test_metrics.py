import pytest 
from omegaconf import OmegaConf
from hydra.utils import instantiate
import numpy as np
import os

from hython.metrics.standard import MSEMetric, RMSEMetric, compute_mse

TARGETS = ["vwc", "actevap"]

y1 = np.random.randn(100,3)
y2 = np.random.randn(100,3)

def test_some_metrics():
    cfg = instantiate(OmegaConf.load(f"{os.path.dirname(__file__)}/metrics.yaml"))

    ret = cfg.metric_fn(y1,y2, ["vwc", "act", "blu"])

    assert ret




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


def test_mse_class_valid_mask():
    b = a = np.random.randn(100, 2)
    mask = np.random.randint(0, 2, (100, 2)).astype(bool)
    ret = MSEMetric()(a, b, TARGETS, valid_mask=mask)

    assert np.all([ret[t] == 0 for t in TARGETS])