import numpy as np


class Metric:
    def __init__(self):
        pass


class MSEMetric(Metric):
    def __call__(self, y_pred, y_true, target_names: list[str]):
        return mse_metric(y_pred, y_true, target_names)


def mse_metric(y_pred, y_true, target_names, sample_weight=None):
    """
    Mean Squared Error (MSE)

    Parameters
    ----------
    y_pred (numpy.array): The true values.
    y_true (numpy.array): The predicted values.
    target_names: List of targets that contribute in the loss computation.

    Shape
    y_true: numpy.array of shape (N, T).
    y_pred: numpy.array of shape (N, T).
    (256,3) means 256 samples with 3 targets.

    Returns
    -------
    Dictionary of MSE metric for each target. {'target': mse_metric}
    """
    metrics = {}
    for idx, target in enumerate(target_names):
        metric_epoch = np.average((y_true[:, idx]- y_pred[:, idx]) ** 2, axis=0, weights=sample_weight)
        metrics[target] = metric_epoch

    return metrics


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


# DISCHARGE 

def fdc_fms():
    """
    """
    pass 

def fdc_fhv():
    """
    """
    pass 

def fdc_flv():
    """
    """
    pass


# SOIL MOISTURE


def hr():
    """Hit Rate, proportion of time soil is correctly simulated as wet.
        Wet threshold is when x >= 0.8 percentile
        Dry threshold is when x <= 0.2 percentile
    """
    pass 

def far():
    """False Alarm Rate"""
    pass 

def csi():
    """Critical success index"""
    pass
