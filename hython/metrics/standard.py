import numpy as np
import xarray as xr

__all__ = ["Metric", "MSEMetric", "RMSEMetric"]


def metric_decorator(y_true, y_pred, target_names, sample_weight=None):
    def target(wrapped):
        def wrapper():
            metrics = {}
            for idx, target in enumerate(target_names):
                metrics[target] = wrapped(y_true[:, idx], y_pred[:, idx], sample_weight)
            return metrics

        return wrapper

    return target


class Metric:
    """
    Hython is currently supporting sequence-to-one training (predicting the last time step of the sequence). Therefore it assumes that
    the shape of y_true and y_pred is (N, C).

    In the future it will also support sequence-to-sequence training for forecasting applications.

    TODO: In forecasting, the shape of y_true and y_pred is going to be (N,T,C), where T is the n of future time steps.

    """

    def __init__(self):
        pass


class MSEMetric(Metric):
    """
    Mean Squared Error (MSE)

    Parameters
    ----------
    y_pred (numpy.array): The true values.
    y_true (numpy.array): The predicted values.
    target_names: List of targets that contribute in the loss computation.

    Returns
    -------
    Dictionary of MSE metric for each target. {'target': mse_metric}

    """

    def __call__(self, y_pred, y_true, target_names: list[str]):
        return metric_decorator(y_pred, y_true, target_names)(compute_mse)()


class RMSEMetric(Metric):
    def __call__(self, y_pred, y_true, target_names: list[str]):
        return metric_decorator(y_pred, y_true, target_names)(compute_rmse)()


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


def compute_variance(ds, dim="time", axis=0, std=False):
    if isinstance(ds, xr.DataArray):
        return ds.std(dim=dim) if std else ds.var(dim=dim)
    else:
        return np.std(ds, axis=axis) if std else np.var(ds, axis=axis)


def compute_gamma(y_true: xr.DataArray, y_pred, axis=0):
    m1, m2 = np.mean(y_true, axis=axis), np.mean(y_pred, axis=axis)
    return (np.std(y_pred, axis=axis) / m2) / (np.std(y_true, axis=axis) / m1)


def compute_pbias(y_true: xr.DataArray, y_pred, dim="time", axis=0):
    if isinstance(y_true, xr.DataArray) or isinstance(y_pred, xr.DataArray):
        return 100 * (
            (y_pred - y_true).mean(dim=dim, skipna=False)
            / np.abs(y_true).mean(dim=dim, skipna=False)
        )
    else:
        return 100 * (
            np.mean(y_pred - y_true, axis=axis) / np.mean(np.abs(y_true), axis=axis)
        )


def compute_bias(y_true: xr.DataArray, y_pred, dim="time", axis=0):
    if isinstance(y_true, xr.DataArray) or isinstance(y_pred, xr.DataArray):
        return (y_pred - y_true).mean(dim=dim, skipna=False)
    else:
        return np.mean(y_pred - y_true, axis=axis)


def compute_rmse(y_true, y_pred, dim="time", axis=0):
    if isinstance(y_true, xr.DataArray) or isinstance(y_pred, xr.DataArray):
        return np.sqrt(((y_pred - y_true) ** 2).mean(dim=dim, skipna=False))
    else:
        return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=axis))


def compute_mse(y_true, y_pred, axis=0, dim="time", sample_weight=None):
    if isinstance(y_true, xr.DataArray) or isinstance(y_pred, xr.DataArray):
        return ((y_pred - y_true) ** 2).mean(dim=dim, skipna=False)
    else:
        return np.average((y_pred - y_true) ** 2, axis=axis, weights=sample_weight)


def compute_pearson(y_true, y_pred, axis=0, dim="time", sample_weight=None):
    if isinstance(y_true, xr.DataArray) or isinstance(y_pred, xr.DataArray):
        return xr.corr(y_true, y_pred, dim=dim, weights=sample_weight)
    else:
        raise NotImplementedError


def compute_kge(y_true, y_pred):
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return np.array([np.nan, np.nan, np.nan, np.nan])

    # r = np.corrcoef(observed, simulated)[1, 0]
    # alpha = np.std(simulated, ddof=1) /np.std(observed, ddof=1)
    # beta = np.mean(simulated) / np.mean(observed)
    # kge = 1 - np.sqrt(np.power(r-1, 2) + np.power(alpha-1, 2) + np.power(beta-1, 2))

    m1, m2 = np.mean(y_true, axis=0), np.mean(y_pred, axis=0)
    num_r = np.sum((y_true - m1) * (y_pred - m2), axis=0)
    den_r = np.sqrt(np.sum((y_true - m1) ** 2, axis=0)) * np.sqrt(
        np.sum((y_pred - m2) ** 2, axis=0)
    )
    r = num_r / den_r
    beta = m2 / m1
    gamma = (np.std(y_pred, axis=0) / m2) / (np.std(y_true, axis=0) / m1)
    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2)

    return np.array([kge, r, gamma, beta])


def compute_kge_parallel(y_target, y_pred):
    kge = xr.apply_ufunc(
        compute_kge,
        y_target,
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
