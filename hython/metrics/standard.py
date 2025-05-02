import numpy as np
import torch
from torch import vmap
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
            ret[metric.__class__.__name__] = metric(y_true, y_pred, target_names, valid_mask)

        ret2 = {}
        for itarget in target_names:
            metrics = {}
            for imetric in ret:
                metrics[imetric] = ret[imetric][itarget]
            ret2[itarget] = metrics

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
            compute_kge2
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
    y_true,
    y_pred,
    dim="time",
    axis=0,
    skipna=False,
    sample_weight=None,
    valid_mask=None,
):
    
    y_true, y_pred = keep_valid(y_true, y_pred)

    den = ((y_true - y_pred.mean()) ** 2).sum()
    num = ((y_pred - y_true) ** 2).sum()

    value = 1 - num / den

    return value

def compute_nse2(
    y_true: xr.DataArray,
    y_pred,
    dim="time",
    axis=0,
    skipna=False,
    sample_weight=None,
    valid_mask=None,
):
    y_true, y_pred = keep_valid(y_true, y_pred)
    den = np.sum(((y_true - np.mean(y_pred)) ** 2))
    num = np.sum(((y_pred - y_true) ** 2))

    value = 1 - num / den

    return value

def compute_nse_parallel(y_true, y_pred, return_all=False):
    if return_all:
        nse = xr.apply_ufunc(
            compute_nse2,
            y_true,
            y_pred,
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[["kge"]],
            output_dtypes=[float],
            vectorize=True,
            dask="parallelized",
            dask_gufunc_kwargs={"output_sizes": {"kge": 4}},
        )
        #kge = kge.assign_coords({"kge": ["kge", "r", "alpha", "beta"]})
    else:
        nse = xr.apply_ufunc(
            compute_nse,
            y_true,
            y_pred,
            input_core_dims=[["time"], ["time"]],
            #output_core_dims=[["kge"]],
            output_dtypes=[float],
            vectorize=True,
            dask="parallelized",
            #dask_gufunc_kwargs={"output_sizes": {"kge": 1}},
        )
        
    return nse

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


def compute_rmse(
    y_true,
    y_pred,
    dim="time",
    axis=0,
    skipna=False,
    sample_weight=None,
    valid_mask=None,
):
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


def compute_pearson(
    y_true, y_pred, axis=0, dim="time", sample_weight=None, valid_mask=None
):
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


def compute_kge(y_true, y_pred, sample_weight=None, valid_mask=None, return_all=False):
    if return_all:
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            return np.array([np.nan, np.nan, np.nan, np.nan])
    else:
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            return np.array([np.nan])    

    true, pred = keep_valid(y_true, y_pred)

    # m_ytrue, m_ypred = np.mean(y_true, axis=0), np.mean(y_pred, axis=0)
    # num_r = np.sum((y_true - m_ytrue) * (y_pred - m_ypred), axis=0)
    # den_r = np.sqrt(np.sum((y_true - m_ytrue) ** 2, axis=0)) * np.sqrt(
    #     np.sum((y_pred - m_ypred) ** 2, axis=0)
    # )
    # r = num_r / den_r
    # beta = m_ypred / m_ytrue
    # gamma = (np.std(y_pred, axis=0) / m_ypred) / (np.std(y_true, axis=0) / m_ytrue)
    # kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2)

    r = np.corrcoef(true, pred)[1, 0]
    alpha = np.std(pred, ddof=1) / np.std(true, ddof=1)
    beta = np.mean(pred) / np.mean(true)
    #gamma = (np.std(y_pred, axis=0) / m_ypred) / (np.std(y_true, axis=0) / m_ytrue)
    kge = 1 - np.sqrt(
        np.power(r - 1, 2) + np.power(alpha - 1, 2) + np.power(beta - 1, 2)
    )
    if return_all:
        ret = np.array([kge, r, alpha, beta])
    else:
        ret = kge
    return ret

def compute_kge2(true, pred, sample_weight=None, valid_mask=None, return_all=False):
    true, pred = keep_valid(true, pred)
    r = np.corrcoef(true, pred)[1, 0]
    alpha = np.std(pred, ddof=1) / np.std(true, ddof=1)
    beta = np.mean(pred) / np.mean(true)
    kge = 1 - np.sqrt(
        np.power(r - 1, 2) + np.power(alpha - 1, 2) + np.power(beta - 1, 2)
    )
    return kge

def compute_kge_parallel(y_true, y_pred, return_all=False):
    if return_all:
        kge = xr.apply_ufunc(
            compute_kge2,
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
    else:
        kge = xr.apply_ufunc(
            compute_kge2,
            y_true,
            y_pred,
            input_core_dims=[["time"], ["time"]],
            #output_core_dims=[["kge"]],
            output_dtypes=[float],
            vectorize=True,
            dask="parallelized",
            #dask_gufunc_kwargs={"output_sizes": {"kge": 1}},
        )
        
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


# def spaeff_metric_torch(sim, obs, return_all = False):
#     """
#     Compute the SPAEF metric using PyTorch.

#     Parameters:
#     - sim: torch.Tensor, simulated  (1D or flattened 2D)
#     - obs: torch.Tensor, observed (1D or flattened 2D)

#     Returns:
#     - spaef: float, SPAEF score
#     - alpha: float, correlation coefficient
#     - beta: float, coefficient of variation ratio
#     - gamma: float, histogram intersection
#     """
#     # Remove NaNs
#     mask = ~torch.isnan(sim) & ~torch.isnan(obs)
#     sim, obs = sim[mask], obs[mask]

#     # Compute correlation coefficient (alpha)
#     alpha = torch.corrcoef(torch.stack((sim, obs)))[0, 1]

#     # Compute coefficient of variation ratio (beta)
#     beta = (torch.std(sim) / torch.mean(sim)) / (torch.std(obs) / torch.mean(obs))

#     # Compute histogram intersection (gamma)
#     bins = int(torch.sqrt(torch.tensor(len(obs), dtype=torch.float32)))
#     hist_sim = torch.histc(sim, bins=bins, min=sim.min().item(), max=sim.max().item())
#     hist_obs = torch.histc(obs, bins=bins, min=obs.min().item(), max=obs.max().item())

#     gamma = torch.sum(torch.min(hist_sim, hist_obs)) / torch.sum(hist_obs)

#     # Compute SPAEF
#     spaef = 1 - torch.sqrt((alpha - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
#     if return_all:
#         return spaef, alpha, beta, gamma
#     else:
#         return spaef


def compute_spaef(observed: torch.Tensor, simulated: torch.Tensor) -> torch.Tensor:
    """
    Compute the Spatial Efficiency (SPAEF) between an observed and simulated feature set.
    :param observed: 1D torch tensor (true feature vector)
    :param simulated: 1D torch tensor (simulated feature vector)
    :return: SPAEF score (tensor)
    """
    r = torch.corrcoef(torch.stack([observed, simulated]))[0, 1]
    alpha = torch.std(simulated) / torch.std(observed)
    
    hist_obs = torch.histc(observed, bins=100, min=observed.min(), max=observed.max())
    hist_sim = torch.histc(simulated, bins=100, min=simulated.min(), max=simulated.max())
    hist_obs /= hist_obs.sum()
    hist_sim /= hist_sim.sum()
    beta = torch.min(hist_obs, hist_sim).sum()
    
    return 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

def aggregate_spaef_over_time(obs_series: torch.Tensor, sim_series: torch.Tensor, method: str = 'mean') -> torch.Tensor:
    """
    Compute the aggregated SPAEF score over a time series of feature vectors.
    :param obs_series: 3D torch tensor (B, T, F) - observed feature vectors over time
    :param sim_series: 3D torch tensor (B, T, F) - simulated feature vectors over time
    :param method: Aggregation method ('mean', 'median', or 'percentile_X')
    :return: Aggregated SPAEF score (tensor)
    """
    if obs_series.shape != sim_series.shape:
        raise ValueError("Observed and simulated series must have the same shape")
    
    spaef_fn = vmap(vmap(compute_spaef, in_dims=(1, 1)), in_dims=(0, 0))  # Apply across time and batch
    spaef_values = spaef_fn(obs_series, sim_series)
    
    if method == 'mean':
        return torch.mean(spaef_values, dim=1)  # Aggregate over time for each batch
    elif method == 'median':
        return torch.median(spaef_values, dim=1).values
    elif method.startswith('percentile_'):
        percentile = float(method.split('_')[1])
        return torch.quantile(spaef_values, percentile / 100, dim=1)
    else:
        raise ValueError("Invalid aggregation method. Choose 'mean', 'median', or 'percentile_X'")