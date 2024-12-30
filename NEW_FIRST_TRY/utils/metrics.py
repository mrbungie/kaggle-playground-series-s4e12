import numpy as np
from sklearn.metrics import root_mean_squared_log_error

def source_range_root_mean_squared_log_error(y_true, y_pred, *args, **kwargs):
    y_true_orig_range = np.power(np.e, y_true) - 1
    y_pred_orig_range = np.power(np.e, y_pred) - 1

    return root_mean_squared_log_error(y_true_orig_range, y_pred_orig_range, *args, **kwargs)   

def rmsle_objective(y_pred, dataset):
    y_true = dataset.get_label()
    
    y_true = np.power(np.e, y_true) - 1
    y_pred = np.power(np.e, y_pred) - 1
    
    # Ensure no negative predictions
    y_pred = np.maximum(y_pred, 1e-15)
    log_diff = np.log(y_pred + 1) - np.log(y_true + 1)
    grad = 2 * log_diff / (y_pred + 1)
    hess = 2 / (y_pred + 1)**2
    
    return grad, hess

def rmsle_metric(y_pred, y_true):    
    y_true = np.power(np.e, y_true) - 1
    y_pred = np.power(np.e, y_pred) - 1

    rmsle = root_mean_squared_log_error(y_true, y_pred)
    
    return 'rmsle', rmsle, False  # False because lower is better