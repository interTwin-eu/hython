from typing import Dict, List
import torch
import numpy as np
from torchmetrics import Metric as TorchMetric

def metric_decorator(y_true, y_pred, target_names, valid_mask=None, sample_weight=None):
    def target(wrapped):
        def wrapper():
            metrics = {}
            for idx, target in enumerate(target_names):
                iypred = y_pred[:, idx]
                iytrue = y_true[:, idx]
                if valid_mask is not None:
                    imask = valid_mask[:, idx]
                    iypred = iypred[imask]
                    iytrue = iytrue[imask]
                if issubclass(wrapped.__class__, TorchMetric):
                    metrics[target] = wrapped(iytrue, iypred).item()
                else:
                    metrics[target] = wrapped(iytrue, iypred, sample_weight=sample_weight)
            return metrics

        return wrapper

    return target



class CustomMetric:
    """
    """

    def __init__(self):
        pass

    def __call__(self) -> Dict:
        pass


class MetricCollection(CustomMetric):
    """Compute a collection of metrics

    Parameters
    ----------
    metrics: list of instantiated metrics. ex: metrics=[MSEMetric(), KGEMetric()]

    Returns
    -------
    Nested dictionary containing metrics and target variables, ex: {"MSEMetric": {"target_variable_1": value, "target_variable_2": value, ...}, "KGEMetric": { ... }, ...}

    """

    def __init__(self, metrics: List[CustomMetric | TorchMetric] = []):
        self.metrics = metrics

    def __call__(self, y_true, y_pred, target_names: list[str], valid_mask=None):
        ret = {}
        for metric in self.metrics:
            if issubclass(metric.__class__, TorchMetric):
                # torchmetrics work only with torch tensors
                if isinstance(y_true, np.ndarray):
                    y_true = torch.from_numpy(y_true)
                    y_pred = torch.from_numpy(y_pred)
                ret[metric.__class__.__name__] = metric_decorator(y_true, y_pred, target_names = target_names, valid_mask=valid_mask)(metric)()
            else:
                # custom metrics work only with numpy arrays
                ret[metric.__class__.__name__] = metric(y_true, y_pred, target_names, valid_mask)
        ret2 = {}
        for itarget in target_names:
            metrics = {}
            for imetric in ret:
                metrics[imetric] = ret[imetric][itarget]
            ret2[itarget] = metrics

        return ret2
