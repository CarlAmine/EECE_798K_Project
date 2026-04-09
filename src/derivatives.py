"""
Derivative estimation methods: finite differences, central differences, Savitzky-Golay.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline, interp1d


def finite_difference_forward(y: np.ndarray, t: np.ndarray = None) -> np.ndarray:
    """
    Estimate derivative using forward finite differences.
    
    Parameters
    ----------
    y : np.ndarray
        Signal values
    t : np.ndarray, optional
        Time points. If None, assumes unit time steps.
        
    Returns
    -------
    np.ndarray
        Estimated derivatives (same length as y, last point is repeated)
    """
    y = np.asarray(y)
    if t is None:
        t = np.arange(len(y))
    
    dt = np.diff(t)
    dy = np.diff(y)
    dydt = dy / dt
    
    # Pad to match original length (use last derivative for last point)
    dydt = np.append(dydt, dydt[-1])
    
    return dydt


def finite_difference_backward(y: np.ndarray, t: np.ndarray = None) -> np.ndarray:
    """
    Estimate derivative using backward finite differences.
    
    Parameters
    ----------
    y : np.ndarray
        Signal values
    t : np.ndarray, optional
        Time points. If None, assumes unit time steps.
        
    Returns
    -------
    np.ndarray
        Estimated derivatives (same length as y, first point is repeated)
    """
    y = np.asarray(y)
    if t is None:
        t = np.arange(len(y))
    
    dt = np.diff(t)
    dy = np.diff(y)
    dydt = dy / dt
    
    # Pad to match original length (use first derivative for first point)
    dydt = np.insert(dydt, 0, dydt[0])
    
    return dydt


def finite_difference_central(y: np.ndarray, t: np.ndarray = None) -> np.ndarray:
    """
    Estimate derivative using central finite differences.
    
    Parameters
    ----------
    y : np.ndarray
        Signal values
    t : np.ndarray, optional
        Time points. If None, assumes unit time steps.
        
    Returns
    -------
    np.ndarray
        Estimated derivatives (same length as y)
    """
    y = np.asarray(y)
    if t is None:
        t = np.arange(len(y))
    
    dydt = np.zeros_like(y, dtype=float)
    
    # Central difference for interior points
    for i in range(1, len(y) - 1):
        dydt[i] = (y[i + 1] - y[i - 1]) / (t[i + 1] - t[i - 1])
    
    # Forward/backward for endpoints
    dydt[0] = (y[1] - y[0]) / (t[1] - t[0])
    dydt[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
    
    return dydt


def compute_derivative(x: np.ndarray, dt: float = 1.0, method: str = 'central') -> np.ndarray:
    """
    Compute the derivative of a 1D signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    dt : float
        Time step size.
    method : str
        Derivative method ('forward', 'backward', 'central').

    Returns
    -------
    np.ndarray
        Estimated derivative signal.
    """
    x = np.asarray(x, dtype=float)
    if method == 'forward':
        return finite_difference_forward(x, np.arange(0, len(x) * dt, dt))
    elif method == 'backward':
        return finite_difference_backward(x, np.arange(0, len(x) * dt, dt))
    elif method == 'central':
        return finite_difference_central(x, np.arange(0, len(x) * dt, dt))
    else:
        raise ValueError(f"Unknown derivative method: {method}")


def derivative_savgol(y: np.ndarray, t: np.ndarray = None, 
                     window: int = 5, polyorder: int = 2) -> np.ndarray:
    """
    Estimate derivative using Savitzky-Golay filter differentiation.
    
    Parameters
    ----------
    y : np.ndarray
        Signal values
    t : np.ndarray, optional
        Time points. If None, assumes unit time steps.
    window : int
        Window length (must be odd)
    polyorder : int
        Polynomial order
        
    Returns
    -------
    np.ndarray
        Estimated derivatives
    """
    y = np.asarray(y)
    if t is None:
        t = np.arange(len(y))
    
    # Ensure window is odd and reasonable
    if window % 2 == 0:
        window += 1
    if window > len(y):
        window = min(len(y), max(3, 2 * len(y) // 3))
        if window % 2 == 0:
            window += 1
    
    # Savgol differentiation
    dydt = savgol_filter(y, window, polyorder, deriv=1)
    
    # Scale by time spacing (assumes regular spacing)
    dt = np.mean(np.diff(t))
    dydt = dydt / dt
    
    return dydt


def derivative_spline(y: np.ndarray, t: np.ndarray = None) -> np.ndarray:
    """
    Estimate derivative by fitting a cubic spline and taking its derivative.
    
    Parameters
    ----------
    y : np.ndarray
        Signal values
    t : np.ndarray, optional
        Time points. If None, assumes unit time steps.
        
    Returns
    -------
    np.ndarray
        Estimated derivatives (evaluated at time points)
    """
    y = np.asarray(y)
    if t is None:
        t = np.arange(len(y), dtype=float)
    
    t = np.asarray(t, dtype=float)
    
    # Fit cubic spline
    cs = CubicSpline(t, y)
    
    # Evaluate derivative at the same points
    dydt = cs(t, 1)
    
    return dydt


def estimate_derivatives_df(df: pd.DataFrame,
                           time_col: str = 'time',
                           var_cols: list = None,
                           method: str = 'central',
                           window: int = 5,
                           polyorder: int = 2,
                           add_to_df: bool = True) -> tuple:
    """
    Estimate derivatives for multiple columns in a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    time_col : str
        Name of time column
    var_cols : list, optional
        Columns to differentiate. If None, all except time_col.
    method : str
        Method: 'forward', 'backward', 'central', 'savgol', 'spline'
    window : int
        Window size for 'savgol'
    polyorder : int
        Polynomial order for 'savgol'
    add_to_df : bool
        If True, add derivatives to dataframe as 'd<col>' columns
        
    Returns
    -------
    tuple
        (df_derivatives or dict_derivatives, df_with_added or None)
    """
    if var_cols is None:
        var_cols = [c for c in df.columns if c != time_col]
    
    t = df[time_col].values
    derivatives = {}
    
    method = method.lower()
    
    for col in var_cols:
        if col not in df.columns:
            continue
        
        y = df[col].values
        
        if method == 'forward':
            dydt = finite_difference_forward(y, t)
        elif method == 'backward':
            dydt = finite_difference_backward(y, t)
        elif method == 'central':
            dydt = finite_difference_central(y, t)
        elif method == 'savgol':
            dydt = derivative_savgol(y, t, window, polyorder)
        elif method == 'spline':
            dydt = derivative_spline(y, t)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        derivatives[col] = dydt
    
    if add_to_df:
        df_out = df.copy()
        for col, dydt in derivatives.items():
            df_out[f'd{col}dt'] = dydt
        return df_out, derivatives
    else:
        return derivatives, None


def estimate_velocity_from_accel(accel: np.ndarray, t: np.ndarray,
                                 initial_vel: float = 0.0) -> np.ndarray:
    """
    Integrate acceleration to estimate velocity.
    
    Parameters
    ----------
    accel : np.ndarray
        Acceleration signal
    t : np.ndarray
        Time points
    initial_vel : float
        Initial velocity (default 0)
        
    Returns
    -------
    np.ndarray
        Estimated velocity
    """
    dt = np.diff(t)
    vel = np.zeros_like(accel)
    vel[0] = initial_vel
    
    for i in range(len(accel) - 1):
        vel[i + 1] = vel[i] + accel[i] * dt[i]
    
    return vel
