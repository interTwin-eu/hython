from sklearn.metrics import mean_squared_error


def mse_metric(y_pred, y_true, target_names):
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
        metric_epoch = mean_squared_error(y_pred[:,idx], y_true[:,idx], squared=True) #y_true[:,idx], y_pred[:,idx]
        metrics[target] = metric_epoch
    
    return metrics