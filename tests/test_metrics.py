import pytest 
from omegaconf import OmegaConf
from hydra.utils import instantiate
import numpy as np
import os

y1 = np.random.randn(100,3)
y2 = np.random.randn(100,3)

def test_metrics():
    cfg = instantiate(OmegaConf.load(f"{os.path.dirname(__file__)}/metrics.yaml"))

    ret = cfg.metric_fn(y1,y2, ["vwc", "act", "blu"])

    assert ret


