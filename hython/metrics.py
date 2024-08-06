from sklearn.metrics import mean_squared_error
import numpy as np


class Metric:
    pass


class MSEMetric(Metric):
    def __init__(self, target_names: list[str]):
        self.target_names = target_names

    def __call__(self, y_pred, y_true, target_names, valid_mask = None):
        return mse_metric(y_pred, y_true, target_names, valid_mask)


def mse_metric(y_pred, y_true, target_names, valid_mask = None):
    """
    Root Mean Squared Error (RMSE) metric for regression task.

    Parameters:
    y_pred (numpy.array): The true values.
    y_true (numpy.array): The predicted values.
    targes: List of targets that contribute in the loss computation.

    Shape
    y_true: numpy.array of shape (N, T).
    y_pred: numpy.array of shape (N, T).
    (256,3) means 256 samples with 3 targets.

    Returns:
    Dictionary of MSE metric for each target. {'target': mse_metric}
    """
    metrics = {}
    for idx, target in enumerate(target_names):
        iypred = y_pred[:, idx]
        iytrue = y_true[:, idx]
        if valid_mask is not None:
            imask = valid_mask[:, idx]
            iypred = iypred[imask]
            iytrue = iytrue[imask]
        metric_epoch = mean_squared_error(iytrue, iypred)
        metrics[target] = metric_epoch

    return metrics


def kge_metric(y_true, y_pred, target_names):
    """
    The Kling Gupta efficiency metric used in the hydrology sciences

    Parameters:
    y_pred (numpy.array): The true values.
    y_true (numpy.array): The predicted values.
    targes: List of targets that contribute in the loss computation.

    Shape
    y_true: numpy.array of shape (N, T).
    y_pred: numpy.array of shape (N, T).
    (256,3) means 256 samples with 3 targets.

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
