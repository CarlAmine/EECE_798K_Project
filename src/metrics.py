"""
Metrics for evaluating model predictions.
"""

import numpy as np
from scipy.integrate import odeint


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))


def relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative L2 error: ||y_true - y_pred|| / ||y_true||."""
    diff_norm = np.linalg.norm(y_true - y_pred)
    true_norm = np.linalg.norm(y_true)
    return diff_norm / (true_norm + 1e-16)


def rollout(dx_dt_fn, x0: np.ndarray, t: np.ndarray,
           library_fn=None) -> np.ndarray:
    """
    Simulate ODE forward from initial condition.
    
    Parameters
    ----------
    dx_dt_fn : callable
        Function that takes (x, t) and returns dx/dt
        OR a function that takes x and library_fn and returns dx/dt
    x0 : np.ndarray
        Initial state vector
    t : np.ndarray
        Time points
    library_fn : callable, optional
        Function to evaluate library (if dx_dt_fn needs it)
        
    Returns
    -------
    np.ndarray
        Simulated states (len(t), len(x0))
    """
    def ode_rhs(x, t_local):
        if library_fn is not None:
            return dx_dt_fn(x.reshape(1, -1), library_fn).flatten()
        else:
            return dx_dt_fn(x, t_local)
    
    x_rollout = odeint(ode_rhs, x0, t)
    return x_rollout


def rollout_error(x_true: np.ndarray, x_pred: np.ndarray,
                 metric: str = 'relative') -> float:
    """
    Compute rollout prediction error.
    
    Parameters
    ----------
    x_true : np.ndarray
        True states (n_timepoints, n_vars)
    x_pred : np.ndarray
        Predicted states (n_timepoints, n_vars)
    metric : str
        'mse', 'rmse', 'mae', or 'relative'
        
    Returns
    -------
    float
        Computed error
    """
    if metric == 'mse':
        return mse(x_true, x_pred)
    elif metric == 'rmse':
        return rmse(x_true, x_pred)
    elif metric == 'mae':
        return mae(x_true, x_pred)
    elif metric == 'relative':
        return relative_error(x_true, x_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def trajectory_correlation(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    """
    Compute correlation between true and predicted trajectories.
    
    Parameters
    ----------
    x_true : np.ndarray
        True states (n_timepoints, n_vars)
    x_pred : np.ndarray
        Predicted states (n_timepoints, n_vars)
        
    Returns
    -------
    np.ndarray
        Correlation per variable
    """
    x_true = np.atleast_2d(x_true)
    x_pred = np.atleast_2d(x_pred)
    
    if x_true.shape[0] == 1:
        x_true = x_true.T
    if x_pred.shape[0] == 1:
        x_pred = x_pred.T
    
    n_vars = x_true.shape[1]
    correlations = []
    
    for i in range(n_vars):
        y_true = x_true[:, i]
        y_pred = x_pred[:, i]
        
        # Normalize
        y_true_norm = (y_true - np.mean(y_true)) / (np.std(y_true) + 1e-16)
        y_pred_norm = (y_pred - np.mean(y_pred)) / (np.std(y_pred) + 1e-16)
        
        corr = np.mean(y_true_norm * y_pred_norm)
        correlations.append(corr)
    
    return np.array(correlations)


def prediction_horizon(x_true: np.ndarray, x_pred: np.ndarray,
                      threshold: float = 0.5) -> int:
    """
    Find the prediction horizon at which error exceeds threshold.
    
    Parameters
    ----------
    x_true : np.ndarray
        True states
    x_pred : np.ndarray
        Predicted states
    threshold : float
        Error threshold (as fraction of true signal range)
        
    Returns
    -------
    int
        Time horizon where relative error exceeds threshold
    """
    x_true = np.atleast_2d(x_true)
    x_pred = np.atleast_2d(x_pred)
    
    if x_true.shape[0] == 1:
        x_true = x_true.T
    if x_pred.shape[0] == 1:
        x_pred = x_pred.T
    
    for t in range(len(x_true)):
        rel_err = relative_error(x_true[:t+1], x_pred[:t+1])
        if rel_err > threshold:
            return t
    
    return len(x_true) - 1


def compute_metrics_rollout(x_true: np.ndarray, x_pred: np.ndarray) -> dict:
    """
    Compute comprehensive metrics for rollout.
    
    Parameters
    ----------
    x_true : np.ndarray
        True states
    x_pred : np.ndarray
        Predicted states
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    metrics = {
        'mse': mse(x_true, x_pred),
        'rmse': rmse(x_true, x_pred),
        'mae': mae(x_true, x_pred),
        'relative_error': relative_error(x_true, x_pred),
        'correlation': trajectory_correlation(x_true, x_pred),
        'prediction_horizon_50pct': prediction_horizon(x_true, x_pred, 0.5),
    }
    
    return metrics
