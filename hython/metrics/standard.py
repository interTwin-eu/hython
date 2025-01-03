import numpy as np
import xarray as xr
from hython.utils import keep_valid
from typing import Dict, List


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
                metrics[target] = wrapped(iytrue, iypred, sample_weight=sample_weight)
            return metrics

        return wrapper

    return target


def get_metrics():
    return


class Metric:
    """
    Hython is currently supporting sequence-to-one training (predicting the last time step of the sequence). Therefore it assumes that
    the shape of y_true and y_pred is (N, C).

    In the future it will also support sequence-to-sequence training for forecasting applications.

    TODO: In forecasting, the shape of y_true and y_pred is going to be (N,T,C), where T is the n of future time steps.

    """

    def __init__(self):
        pass

    def __call__(self) -> Dict:
        pass


class MetricCollection(Metric):
    """Compute a collection of metrics

    Parameters
    ----------
    metrics: list of instantiated metrics. ex: metrics=[MSEMetric(), KGEMetric()]

    Returns
    -------
    Nested dictionary containing metrics and target variables, ex: {"MSEMetric": {"target_variable_1": value, "target_variable_2": value, ...}, "KGEMetric": { ... }, ...}

    """

    def __init__(self, metrics: List[Metric] = []):
        self.metrics = metrics

    def __call__(self, y_true, y_pred, target_names: list[str], valid_mask=None):
        ret = {}
        for metric in self.metrics:
            ret[metric.__class__.__name__] = metric(y_true, y_pred, target_names)

        ret2 = {}
        for km in ret:
            for kt in ret[km]:
                ret2[kt] = km
        return ret2


class MSEMetric(Metric):
    """
    Mean Squared Error (MSE)

    Parameters
    ----------
    y_pred (numpy.array): The true values. [sample, feature]
    y_true (numpy.array): The predicted values. [sample, feature]
    target_names: List of targets that contribute in the loss computation.
    valid_mask:

    Returns
    -------
    Dictionary of MSE metric for each target. {'target': mse_metric}

    """

    def __call__(self, y_true, y_pred, target_names: list[str], valid_mask=None):
        return metric_decorator(y_true, y_pred, target_names, valid_mask=valid_mask)(
            compute_mse
        )()


class RMSEMetric(Metric):
    def __call__(self, y_true, y_pred, target_names: list[str], valid_mask=None):
        return metric_decorator(y_true, y_pred, target_names, valid_mask=valid_mask)(
            compute_rmse
        )()


class KGEMetric(Metric):
    def __call__(self, y_true, y_pred, target_names: list[str], valid_mask=None):
        return metric_decorator(y_true, y_pred, target_names, valid_mask=valid_mask)(
            compute_kge
        )()


class PearsonMetric(Metric):
    def __call__(self, y_true, y_pred, target_names: list[str], valid_mask=None):
        return metric_decorator(y_true, y_pred, target_names, valid_mask=valid_mask)(
            compute_pearson
        )()


class PBIASMetric(Metric):
    def __call__(self, y_true, y_pred, target_names: list[str], valid_mask=None):
        return metric_decorator(y_true, y_pred, target_names, valid_mask=valid_mask)(
            compute_pbias
        )()


class NSEMetric(Metric):
    def __call__(self, y_true, y_pred, target_names: list[str], valid_mask=None):
        return metric_decorator(y_true, y_pred, target_names, valid_mask=valid_mask)(
            compute_nse
        )()


# == METRICS
# The metrics below should work for both numpy or xarray inputs. The usage of xarray inputs is supported as it is handy for lazy computations
# e.g. compute_mse(y_true.chunk(lat=100,lon=100), y_pred.chunk(lat=100,lon=100)).compute()


# DISCHARGE


def compute_fdc_fms():
    """ """
    pass


def compute_fdc_fhv():
    """ """
    pass


def compute_fdc_flv():
    """ """
    pass


# SOIL MOISTURE


def compute_hr():
    """Hit Rate, proportion of time soil is correctly simulated as wet.
    Wet threshold is when x >= 0.8 percentile
    Dry threshold is when x <= 0.2 percentile
    """
    pass


def compute_far():
    """False Alarm Rate"""
    pass


def compute_csi():
    """Critical success index"""
    pass


# GENERAL


def compute_nse(
    y_true: xr.DataArray,
    y_pred,
    dim="time",
    axis=0,
    skipna=False,
    sample_weight=None,
    valid_mask=None,
):
    den = ((y_true - y_pred.mean()) ** 2).sum()
    num = ((y_pred - y_true) ** 2).sum()

    value = 1 - num / den

    return value


def compute_variance(ds, dim="time", axis=0, std=False):
    if isinstance(ds, xr.DataArray):
        return ds.std(dim=dim) if std else ds.var(dim=dim)
    else:
        return np.nanstd(ds, axis=axis) if std else np.nanvar(ds, axis=axis)


def compute_gamma(y_true: xr.DataArray, y_pred, axis=0):
    if isinstance(y_true, xr.DataArray):
        pass
    else:
        y_true, y_pred = keep_valid(y_true, y_pred)
        m1, m2 = np.mean(y_true, axis=axis), np.mean(y_pred, axis=axis)
    return (np.nanstd(y_pred, axis=axis) / m2) / (np.nanstd(y_true, axis=axis) / m1)


def compute_pbias(
    y_true: xr.DataArray,
    y_pred,
    dim="time",
    axis=0,
    skipna=False,
    sample_weight=None,
    valid_mask=None,
):
    if isinstance(y_true, xr.DataArray) or isinstance(y_pred, xr.DataArray):
        return 100 * (
            (y_pred - y_true).mean(dim=dim, skipna=skipna)
            / np.abs(y_true).mean(dim=dim, skipna=skipna)
        )
    else:
        y_true, y_pred = keep_valid(y_true, y_pred)

        return 100 * (
            np.mean(y_pred - y_true, axis=axis) / np.mean(np.abs(y_true), axis=axis)
        )


def compute_bias(y_true: xr.DataArray, y_pred, dim="time", axis=0, skipna=False):
    if isinstance(y_true, xr.DataArray) or isinstance(y_pred, xr.DataArray):
        return (y_pred - y_true).mean(dim=dim, skipna=skipna)
    else:
        y_true, y_pred = keep_valid(y_true, y_pred)
        return np.mean(y_pred - y_true, axis=axis)


def compute_rmse(y_true, y_pred, dim="time", axis=0, skipna=False):
    if isinstance(y_true, xr.DataArray) or isinstance(y_pred, xr.DataArray):
        return np.sqrt(((y_pred - y_true) ** 2).mean(dim=dim, skipna=skipna))
    else:
        y_true, y_pred = keep_valid(y_true, y_pred)
        return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=axis))


def compute_mse(
    y_true,
    y_pred,
    axis=0,
    dim="time",
    sample_weight=None,
    skipna=False,
):
    if isinstance(y_true, xr.DataArray) or isinstance(y_pred, xr.DataArray):
        return ((y_pred - y_true) ** 2).mean(dim=dim, skipna=skipna)
    else:
        y_true, y_pred = keep_valid(y_true, y_pred)
        return np.average((y_pred - y_true) ** 2, axis=axis, weights=sample_weight)


def compute_pearson(y_true, y_pred, axis=0, dim="time", sample_weight=None):
    if isinstance(y_true, xr.DataArray) or isinstance(y_pred, xr.DataArray):
        return xr.corr(y_true, y_pred, dim=dim, weights=sample_weight)
    else:
        y_true, y_pred = keep_valid(y_true, y_pred)
        y_true_m = y_true.mean(axis=axis)
        y_pred_m = y_pred.mean(axis=axis)
        num = np.sum((y_true - y_true_m) * (y_pred - y_pred_m), axis=axis)
        den = np.sqrt(
            np.sum((y_true - y_true_m) ** 2, axis=0)
            * np.sum((y_pred - y_pred_m) ** 2, axis=0)
        )
        return num / den


def compute_kge(y_true, y_pred, sample_weight=None):
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return np.array([np.nan, np.nan, np.nan, np.nan])

    # r = np.corrcoef(observed, simulated)[1, 0]
    # alpha = np.std(simulated, ddof=1) /np.std(observed, ddof=1)
    # beta = np.mean(simulated) / np.mean(observed)
    # kge = 1 - np.sqrt(np.power(r-1, 2) + np.power(alpha-1, 2) + np.power(beta-1, 2))
    y_true, y_pred = keep_valid(y_true, y_pred)

    m_ytrue, m_ypred = np.mean(y_true, axis=0), np.mean(y_pred, axis=0)
    num_r = np.sum((y_true - m_ytrue) * (y_pred - m_ypred), axis=0)
    den_r = np.sqrt(np.sum((y_true - m_ytrue) ** 2, axis=0)) * np.sqrt(
        np.sum((y_pred - m_ypred) ** 2, axis=0)
    )
    r = num_r / den_r
    beta = m_ypred / m_ytrue
    gamma = (np.std(y_pred, axis=0) / m_ypred) / (np.std(y_true, axis=0) / m_ytrue)
    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2)

    return np.array([kge, r, gamma, beta])


def compute_kge_parallel(y_true, y_pred):
    kge = xr.apply_ufunc(
        compute_kge,
        y_true,
        y_pred,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["kge"]],
        output_dtypes=[float],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"kge": 4}},
    )

    kge = kge.assign_coords({"kge": ["kge", "r", "alpha", "beta"]})
    return kge


def kge_metric(y_true, y_pred, target_names):
    """
    The Kling Gupta efficiency metric

    Parameters:
    y_pred (numpy.array): The true values.
    y_true (numpy.array): The predicted values.
    targes: List of targets that contribute in the loss computation.

    Shape
    y_true: numpy.array of shape (N, T).
    y_pred: numpy.array of shape (N, T).

    Returns:
    Dictionary of kge metric for each target. {'target': kge_value}
    """

    metrics = {}

    for idx, target in enumerate(target_names):
        observed = y_true[:, idx]
        simulated = y_pred[:, idx]
        r = np.corrcoef(observed, simulated)[1, 0]
        alpha = np.std(simulated, ddof=1) / np.std(observed, ddof=1)
        beta = np.mean(simulated) / np.mean(observed)
        kge = 1 - np.sqrt(
            np.power(r - 1, 2) + np.power(alpha - 1, 2) + np.power(beta - 1, 2)
        )
        metrics[target] = kge

    return metrics
