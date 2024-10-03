import numpy as np
import xarray as xr
import pandas as pd
import math

__all__ = ["MSEMetric", "RMSEMetric"]


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

def compute_fdc_fms(observed_flow: pd.DataFrame, observed_col: str, 
                 simulated_flow: pd.DataFrame, simulated_col: str) -> float:
    """
    Compute the bias between observed and simulated discharge values
    at specified exceedance probabilities (0.2 and 0.7).

    Parameters:
    observed_flow (pd.DataFrame): DataFrame containing observed flow data.
    observed_col (str): Column name for observed discharge values.
    simulated_flow (pd.DataFrame): DataFrame containing simulated flow data.
    simulated_col (str): Column name for simulated discharge values.

    Returns:
    float: Bias percentage.
    """

    # Check if both DataFrames have the same number of records
    if len(observed_flow) != len(simulated_flow):
        raise ValueError("The observed and simulated DataFrames must have the same number of records.")
    
    # Step 1: Sort simulated flow data
    data_simulated_sorted = simulated_flow.sort_values(by=simulated_col, ascending=False).reset_index(drop=True)
    
    # Step 2: Calculate exceedance probabilities for simulated data
    n_simulated = len(data_simulated_sorted)
    data_simulated_sorted['exceedance_probability'] = (data_simulated_sorted.index + 1) / (n_simulated + 1)
    
    # Step 3: Extract discharge values for exceedance probabilities 0.2 and 0.7 (simulated)
    QSM1 = data_simulated_sorted.loc[data_simulated_sorted['exceedance_probability'] >= 0.2, simulated_col].iloc[0]
    QSM2 = data_simulated_sorted.loc[data_simulated_sorted['exceedance_probability'] >= 0.7, simulated_col].iloc[0]

    # Step 4: Sort observed flow data
    data_observed_sorted = observed_flow.sort_values(by=observed_col, ascending=False).reset_index(drop=True)
    
    # Step 5: Calculate exceedance probabilities for observed data
    n_observed = len(data_observed_sorted)
    data_observed_sorted['exceedance_probability'] = (data_observed_sorted.index + 1) / (n_observed + 1)
    
    # Step 6: Extract discharge values for exceedance probabilities 0.2 and 0.7 (observed)
    QOM1 = data_observed_sorted.loc[data_observed_sorted['exceedance_probability'] >= 0.2, observed_col].iloc[0]
    QOM2 = data_observed_sorted.loc[data_observed_sorted['exceedance_probability'] >= 0.7, observed_col].iloc[0]

    # Step 7: Calculate bias
    biasFMS = (((math.log(QSM1) - math.log(QSM2)) - (math.log(QOM1) - math.log(QOM2))) /
                (math.log(QOM1) - math.log(QOM2))) * 100
    
    print(f'BiasFMS : {biasFMS}')

    return biasFMS

def compute_fdc_fhv(observed_flow: pd.DataFrame, observed_col: str, 
                     simulated_flow: pd.DataFrame, simulated_col: str) -> float:
    """
    Compute the Bias FHV (Flow Volume Bias) between observed and simulated discharge values
    at an exceedance probability of 0.02.

    Parameters:
    observed_flow (pd.DataFrame): DataFrame containing observed flow data.
    observed_col (str): Column name for observed discharge values.
    simulated_flow (pd.DataFrame): DataFrame containing simulated flow data.
    simulated_col (str): Column name for simulated discharge values.

    Returns:
    float: Bias FHV percentage.
    """
    
    # Check if both DataFrames have the same number of records
    if len(observed_flow) != len(simulated_flow):
        raise ValueError("The observed and simulated DataFrames must have the same number of records.")
    
    # Sort simulated flow data
    data_simulated_sorted = simulated_flow.sort_values(by=simulated_col, ascending=False).reset_index(drop=True)
    
    # Calculate exceedance probabilities for simulated data
    n_simulated = len(data_simulated_sorted)
    data_simulated_sorted['exceedance_probability'] = (data_simulated_sorted.index + 1) / (n_simulated + 1)

    # Sort observed flow data
    data_observed_sorted = observed_flow.sort_values(by=observed_col, ascending=False).reset_index(drop=True)
    
    # Calculate exceedance probabilities for observed data
    n_observed = len(data_observed_sorted)
    data_observed_sorted['exceedance_probability'] = (data_observed_sorted.index + 1) / (n_observed + 1)

    # Calculate FHV for exceedance probability <= 0.02
    fhv_qo = data_observed_sorted.loc[data_observed_sorted['exceedance_probability'] <= 0.02, observed_col]
    fhv_qs = data_simulated_sorted.loc[data_simulated_sorted['exceedance_probability'] <= 0.02, simulated_col]
    
    # Ensure both fhv_qo and fhv_qs are not empty
    if fhv_qo.empty or fhv_qs.empty:
        raise ValueError("No data available for exceedance probability <= 0.02.")

    # Calculate FHV
    fhv = fhv_qs.values - fhv_qo.values
    FHV_numerator = fhv.sum()
    FHV_denominator = fhv_qo.sum()
    
    # Calculate bias for FHV
    biasFHV = (FHV_numerator / FHV_denominator) * 100

    print(f'BiasFHV : {biasFHV}')
    
    return float(biasFHV)


def compute_fdc_flv(observed_flow: pd.DataFrame, observed_col: str, 
                     simulated_flow: pd.DataFrame, simulated_col: str) -> float:
    """
    Compute the Bias FLV (Flow Volume Bias) between observed and simulated discharge values
    at an exceedance probability of 0.7.

    Parameters:
    observed_flow (pd.DataFrame): DataFrame containing observed flow data.
    observed_col (str): Column name for observed discharge values.
    simulated_flow (pd.DataFrame): DataFrame containing simulated flow data.
    simulated_col (str): Column name for simulated discharge values.

    Returns:
    float: Bias FLV percentage.
    """

    # Check if both DataFrames have the same number of records
    if len(observed_flow) != len(simulated_flow):
        raise ValueError("The observed and simulated DataFrames must have the same number of records.")
    
    # Sort simulated flow data
    data_simulated_sorted = simulated_flow.sort_values(by=simulated_col, ascending=False).reset_index(drop=True)
    
    # Calculate exceedance probabilities for simulated data
    n_simulated = len(data_simulated_sorted)
    data_simulated_sorted['exceedance_probability'] = (data_simulated_sorted.index + 1) / (n_simulated + 1)

    # Sort observed flow data
    data_observed_sorted = observed_flow.sort_values(by=observed_col, ascending=False).reset_index(drop=True)
    
    # Calculate exceedance probabilities for observed data
    n_observed = len(data_observed_sorted)
    data_observed_sorted['exceedance_probability'] = (data_observed_sorted.index + 1) / (n_observed + 1)

    # Calculate FLV for exceedance probability >= 0.7
    flv_qo = data_observed_sorted.loc[data_observed_sorted['exceedance_probability'] >= 0.7, observed_col]
    flv_qs = data_simulated_sorted.loc[data_simulated_sorted['exceedance_probability'] >= 0.7, simulated_col]

    # Ensure both flv_qo and flv_qs are not empty
    if flv_qo.empty or flv_qs.empty:
        raise ValueError("No data available for exceedance probability >= 0.7.")

    # Calculate FLV numerators
    FLV_numerator1 = (np.log(flv_qs) - np.log(flv_qs.min())).sum()
    FLV_numerator2 = (np.log(flv_qo) - np.log(flv_qo.min())).sum()

    # Calculate bias for FLV
    biasFLV = (-100 * (FLV_numerator1 - FLV_numerator2)) / FLV_numerator2

    print(f'BiasFLV : {biasFLV}')
    
    return float(biasFLV)


# SOIL MOISTURE


def compute_hr(observed: xr.DataArray, simulated: xr.DataArray, wet_threshold_percentile: float = 0.8, dry_threshold_percentile: float = 0.2) -> dict:
    """
    Hit Rate: Proportion of time soil is correctly simulated as wet and dry.
    
    Wet threshold is when x >= 80th percentile.
    Dry threshold is when x <= 20th percentile.
    
    Parameters:
    observed (xr.DataArray): Observed soil moisture data (lat, lon, time).
    simulated (xr.DataArray): Simulated soil moisture data (lat, lon, time).
    
    Returns:
    tuple: Wet threshold hit rate (%), Dry threshold hit rate (%).
    """
    
    # Compute the 80th and 20th percentiles for observed and simulated data along the time dimension
    observed_wet_quan = observed.quantile(wet_threshold_percentile, dim='time')
    simulated_wet_quan = simulated.quantile(wet_threshold_percentile, dim='time')
    
    observed_dry_quan = observed.quantile(dry_threshold_percentile, dim='time')
    simulated_dry_quan = simulated.quantile(dry_threshold_percentile, dim='time')

    # Create masks for "wet" periods (80th percentile) and "dry" periods (20th percentile)
    observed_wet = observed >= observed_wet_quan
    simulated_wet = simulated >= simulated_wet_quan
    
    observed_dry = observed <= observed_dry_quan
    simulated_dry = simulated <= simulated_dry_quan

    # Calculate the hit rate for "wet" periods
    wet_hits = (observed_wet & simulated_wet).sum(dim='time')
    total_wet_periods = observed_wet.sum(dim='time')

    total_wet_hits = wet_hits.sum().values  # Convert to numpy array
    total_wet_periods_sum = total_wet_periods.sum().values  # Convert to numpy array

    # Avoid division by zero in case there are no "wet" periods in observed data
    if total_wet_periods_sum == 0:
        wet_hit_rate = 0.0
    else:
        wet_hit_rate = (total_wet_hits / total_wet_periods_sum) * 100

    # Calculate the hit rate for "dry" periods
    dry_hits = (observed_dry & simulated_dry).sum(dim='time')
    total_dry_periods = observed_dry.sum(dim='time')

    total_dry_hits = dry_hits.sum().values  # Convert to numpy array
    total_dry_periods_sum = total_dry_periods.sum().values  # Convert to numpy array

    # Avoid division by zero in case there are no "dry" periods in observed data
    if total_dry_periods_sum == 0:
        dry_hit_rate = 0.0
    else:
        dry_hit_rate = (total_dry_hits / total_dry_periods_sum) * 100

    hit_rate = {
    f'wet_threshold_{wet_threshold_percentile}_hit_rate': float(wet_hit_rate),
    f'dry_threshold_{dry_threshold_percentile}_hit_rate': float(dry_hit_rate)
    }

    print(hit_rate)
    
    return hit_rate

def compute_far(observed: xr.DataArray, simulated: xr.DataArray, wet_threshold_percentile: float = 0.8, dry_threshold_percentile: float = 0.2) -> dict:
    """
    Compute False Alarm Rate (FAR) for wet and dry predictions.

    Parameters:
    observed (xr.DataArray): Observed soil moisture data (lat, lon, time).
    simulated (xr.DataArray): Simulated soil moisture data (lat, lon, time).

    Returns:
    tuple: FAR for wet predictions (%), FAR for dry predictions (%).
    """
    
    # Compute the 80th and 20th percentiles for observed and simulated data along the time dimension
    observed_wet_quan = observed.quantile(wet_threshold_percentile, dim='time')
    simulated_wet_quan = simulated.quantile(wet_threshold_percentile, dim='time')

    observed_dry_quan = observed.quantile(dry_threshold_percentile, dim='time')
    simulated_dry_quan = simulated.quantile(dry_threshold_percentile, dim='time')

    # Create masks for "wet" and "dry" periods based on the percentiles
    observed_wet = observed >= observed_wet_quan
    simulated_wet = simulated >= simulated_wet_quan

    observed_dry = observed <= observed_dry_quan
    simulated_dry = simulated <= simulated_dry_quan

    # Calculate hits and false alarms for "wet" periods
    wet_hits = (observed_wet & simulated_wet).sum(dim='time')
    wet_false_alarms = (simulated_wet & ~observed_wet).sum(dim='time')

    # Sum hits and false alarms for "wet" across all spatial dimensions (lat, lon)
    total_wet_hits = wet_hits.sum().values  # Convert to numpy array
    total_wet_false_alarms = wet_false_alarms.sum().values  # Convert to numpy array

    # Calculate False Alarm Rate for wet conditions
    if (total_wet_hits + total_wet_false_alarms) == 0:
        wet_far = 0.0
    else:
        wet_far = (total_wet_false_alarms / (total_wet_false_alarms + total_wet_hits)) * 100  # As a percentage

    # Calculate hits and false alarms for "dry" periods
    dry_hits = (observed_dry & simulated_dry).sum(dim='time')
    dry_false_alarms = (simulated_dry & ~observed_dry).sum(dim='time')

    # Sum hits and false alarms for "dry" across all spatial dimensions (lat, lon)
    total_dry_hits = dry_hits.sum().values  # Convert to numpy array
    total_dry_false_alarms = dry_false_alarms.sum().values  # Convert to numpy array

    # Calculate False Alarm Rate for dry conditions
    if (total_dry_hits + total_dry_false_alarms) == 0:
        dry_far = 0.0
    else:
        dry_far = (total_dry_false_alarms / (total_dry_false_alarms + total_dry_hits)) * 100  # As a percentage

    far = {
    f'wet_threshold_{wet_threshold_percentile}_far': float(wet_far),
    f'dry_threshold_{dry_threshold_percentile}_far': float(dry_far)
    }

    print(far)
    
    return far


def compute_csi(observed: xr.DataArray, simulated: xr.DataArray, wet_threshold_percentile: float = 0.8, dry_threshold_percentile: float = 0.2) -> dict:
    """
    Compute the Critical Success Index (CSI) for wet and dry predictions.

    Parameters:
    observed (xr.DataArray): Observed soil moisture data (lat, lon, time).
    simulated (xr.DataArray): Simulated soil moisture data (lat, lon, time).

    Returns:
    tuple: CSI for wet predictions (%), CSI for dry predictions (%).
    """
    
    # Compute the 80th and 20th percentiles for observed and simulated data along the time dimension
    observed_wet_quan = observed.quantile(wet_threshold_percentile, dim='time')
    simulated_wet_quan = simulated.quantile(wet_threshold_percentile, dim='time')

    observed_dry_quan = observed.quantile(dry_threshold_percentile, dim='time')
    simulated_dry_quan = simulated.quantile(dry_threshold_percentile, dim='time')

    # Create masks for "wet" and "dry" periods based on the percentiles
    observed_wet = observed >= observed_wet_quan
    simulated_wet = simulated >= simulated_wet_quan

    observed_dry = observed <= observed_dry_quan
    simulated_dry = simulated <= simulated_dry_quan

    # Calculate hits, false alarms, and misses for "wet" periods
    wet_hits = (observed_wet & simulated_wet).sum(dim='time')
    wet_false_alarms = (simulated_wet & ~observed_wet).sum(dim='time')
    wet_misses = (~simulated_wet & observed_wet).sum(dim='time')

    # Sum hits, false alarms, and misses for "wet" across all spatial dimensions (lat, lon)
    total_wet_hits = wet_hits.sum().values  # Convert to numpy array
    total_wet_false_alarms = wet_false_alarms.sum().values  # Convert to numpy array
    total_wet_misses = wet_misses.sum().values  # Convert to numpy array

    # Calculate Critical Success Index for wet conditions
    csi_wet = (total_wet_hits / (total_wet_hits + total_wet_false_alarms + total_wet_misses)) * 100 if (total_wet_hits + total_wet_false_alarms + total_wet_misses) > 0 else 0.0

    # Calculate hits, false alarms, and misses for "dry" periods
    dry_hits = (observed_dry & simulated_dry).sum(dim='time')
    dry_false_alarms = (simulated_dry & ~observed_dry).sum(dim='time')
    dry_misses = (~simulated_dry & observed_dry).sum(dim='time')

    # Sum hits, false alarms, and misses for "dry" across all spatial dimensions (lat, lon)
    total_dry_hits = dry_hits.sum().values  # Convert to numpy array
    total_dry_false_alarms = dry_false_alarms.sum().values  # Convert to numpy array
    total_dry_misses = dry_misses.sum().values  # Convert to numpy array

    # Calculate Critical Success Index for dry conditions
    csi_dry = (total_dry_hits / (total_dry_hits + total_dry_false_alarms + total_dry_misses)) * 100 if (total_dry_hits + total_dry_false_alarms + total_dry_misses) > 0 else 0.0

    csi = {
    f'wet_threshold_{wet_threshold_percentile}_csi': float(csi_wet),
    f'dry_threshold_{dry_threshold_percentile}_csi': float(csi_dry)
    }

    print(csi)
    
    return csi


# GENERAL

def compute_variance(ds,dim="time", axis=0, std=False):
    if isinstance(ds, xr.DataArray):
        return ds.std(dim=dim) if std else ds.var(dim=dim)
    else:
        return np.std(ds, axis=axis) if std else np.var(ds, axis=axis) 
    
def compute_gamma(y_true: xr.DataArray, y_pred, axis=0):
    m1, m2 = np.mean(y_true, axis=axis), np.mean(y_pred, axis=axis)
    return (np.std(y_pred, axis=axis) / m2) / (np.std(y_true, axis=axis) / m1)
    
def compute_pbias(y_true: xr.DataArray, y_pred, dim="time", axis=0):
    if isinstance(y_true, xr.DataArray) or isinstance(y_pred, xr.DataArray):
         return 100 * ( (y_pred - y_true).mean(dim=dim, skipna=False) / np.abs(y_true).mean(dim=dim, skipna=False))
    else:
        return 100 * ( np.mean(y_pred - y_true, axis=axis) / np.mean(np.abs(y_true), axis=axis) )

def compute_bias(y_true: xr.DataArray, y_pred, dim="time", axis=0):
    if isinstance(y_true, xr.DataArray) or isinstance(y_pred, xr.DataArray):
         return  (y_pred - y_true).mean(dim=dim, skipna=False)
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

