"""
Data preprocessing utilities: smoothing, normalization, detrending.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, detrend
from scipy.ndimage import uniform_filter1d

from .derivatives import derivative_savgol, compute_derivative


def remove_invalid_rows(df: pd.DataFrame, 
                       invalid_value: float = np.nan) -> pd.DataFrame:
    """
    Remove rows with invalid values (NaN, inf, etc.).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    invalid_value : float
        Value to consider invalid (default NaN)
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with invalid rows removed
    """
    df_clean = df.dropna()
    df_clean = df_clean[~np.isinf(df_clean).any(axis=1)]
    return df_clean.reset_index(drop=True)


def smooth_savgol(signal: np.ndarray, window: int = 5, 
                  polyorder: int = 2) -> np.ndarray:
    """
    Smooth signal using Savitzky-Golay filter.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D)
    window : int
        Window length (must be odd)
    polyorder : int
        Polynomial order
        
    Returns
    -------
    np.ndarray
        Smoothed signal
    """
    if window % 2 == 0:
        window += 1
    if window > len(signal):
        window = min(len(signal), max(3, 2 * len(signal) // 3))
        if window % 2 == 0:
            window += 1
    
    return savgol_filter(signal, window, polyorder)


def smooth_uniform(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Smooth signal using uniform (moving average) filter.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D)
    window : int
        Window size
        
    Returns
    -------
    np.ndarray
        Smoothed signal
    """
    return uniform_filter1d(signal, size=window, mode='nearest')


def normalize_minmax(data: np.ndarray, axis: int = 0) -> tuple:
    """
    Normalize data to [0, 1] range using min-max scaling.
    
    Parameters
    ----------
    data : np.ndarray
        Input data (can be 1D or 2D)
    axis : int
        Axis along which to compute min/max
        
    Returns
    -------
    tuple
        (normalized_data, min_vals, max_vals)
    """
    data = np.asarray(data)
    data_min = np.min(data, axis=axis, keepdims=True)
    data_max = np.max(data, axis=axis, keepdims=True)
    
    # Avoid division by zero
    range_vals = data_max - data_min
    range_vals[range_vals == 0] = 1.0
    
    normalized = (data - data_min) / range_vals
    return normalized, data_min, data_max


def normalize_zscore(data: np.ndarray, axis: int = 0) -> tuple:
    """
    Normalize data using z-score (standardization).
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    axis : int
        Axis along which to compute mean/std
        
    Returns
    -------
    tuple
        (normalized_data, means, stds)
    """
    data = np.asarray(data)
    means = np.mean(data, axis=axis, keepdims=True)
    stds = np.std(data, axis=axis, keepdims=True)
    
    # Avoid division by zero
    stds[stds == 0] = 1.0
    
    normalized = (data - means) / stds
    return normalized, means, stds


def denormalize_minmax(data: np.ndarray, data_min: np.ndarray, 
                      data_max: np.ndarray) -> np.ndarray:
    """
    Reverse min-max normalization.
    
    Parameters
    ----------
    data : np.ndarray
        Normalized data
    data_min : np.ndarray
        Original minimum values
    data_max : np.ndarray
        Original maximum values
        
    Returns
    -------
    np.ndarray
        Denormalized data
    """
    range_vals = data_max - data_min
    range_vals[range_vals == 0] = 1.0
    return data * range_vals + data_min


def denormalize_zscore(data: np.ndarray, means: np.ndarray, 
                      stds: np.ndarray) -> np.ndarray:
    """
    Reverse z-score normalization.
    
    Parameters
    ----------
    data : np.ndarray
        Normalized data
    means : np.ndarray
        Original means
    stds : np.ndarray
        Original standard deviations
        
    Returns
    -------
    np.ndarray
        Denormalized data
    """
    return data * stds + means


def preprocess_dataframe(df: pd.DataFrame, 
                        time_col: str = 'time',
                        var_cols: list = None,
                        smooth: bool = False,
                        smooth_window: int = 5,
                        smooth_method: str = 'savgol',
                        normalize: bool = False,
                        norm_method: str = 'minmax',
                        remove_outliers: bool = True) -> tuple:
    """
    Apply preprocessing pipeline to a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    time_col : str
        Name of time column
    var_cols : list, optional
        Columns to process. If None, all except time_col
    smooth : bool
        Whether to smooth signals
    smooth_window : int
        Smoothing window size
    smooth_method : str
        'savgol' or 'uniform'
    normalize : bool
        Whether to normalize signals
    norm_method : str
        'minmax' or 'zscore'
    remove_outliers : bool
        Whether to remove rows with NaN/inf
        
    Returns
    -------
    tuple
        (processed_df, scalers) where scalers is a dict with
        normalization parameters if normalize=True
    """
    df = df.copy()
    scalers = {}
    
    if remove_outliers:
        df = remove_invalid_rows(df)
    
    if var_cols is None:
        var_cols = [c for c in df.columns if c != time_col]
    
    if smooth:
        for col in var_cols:
            if col in df.columns:
                method = smooth_method.lower()
                if method == 'savgol':
                    df[col] = smooth_savgol(df[col].values, 
                                           window=smooth_window)
                elif method == 'uniform':
                    df[col] = smooth_uniform(df[col].values, 
                                           window=smooth_window)
    
    if normalize:
        data_to_norm = df[var_cols].values
        method = norm_method.lower()
        
        if method == 'minmax':
            normalized, mins, maxs = normalize_minmax(data_to_norm)
            scalers['method'] = 'minmax'
            scalers['mins'] = mins
            scalers['maxs'] = maxs
        elif method == 'zscore':
            normalized, means, stds = normalize_zscore(data_to_norm)
            scalers['method'] = 'zscore'
            scalers['means'] = means
            scalers['stds'] = stds
        
        df[var_cols] = normalized
    
    return df, scalers


def clean_lanl_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw LANL dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input LANL dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with required columns and invalid rows removed.
    """
    validate_cols = {'acoustic_data', 'time_to_failure'}
    missing = validate_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing LANL required columns: {missing}")

    df_clean = df[['acoustic_data', 'time_to_failure']].copy()
    df_clean = df_clean.dropna().reset_index(drop=True)
    df_clean = df_clean[~np.isinf(df_clean['acoustic_data'])]
    df_clean = df_clean[~np.isinf(df_clean['time_to_failure'])]
    df_clean['time'] = np.arange(len(df_clean), dtype=float)
    return df_clean[['time', 'acoustic_data', 'time_to_failure']]


def smooth_signal(x: np.ndarray, window: int = 101, polyorder: int = 3) -> np.ndarray:
    """
    Smooth a 1D signal using Savitzky-Golay filtering.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    window : int
        Window length.
    polyorder : int
        Polynomial order.

    Returns
    -------
    np.ndarray
        Smoothed signal.
    """
    return smooth_savgol(x, window=window, polyorder=polyorder)


def normalize_signal(x: np.ndarray, method: str = 'minmax') -> tuple:
    """
    Normalize a 1D signal using min-max or z-score scaling.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    method : str
        'minmax' or 'zscore'.

    Returns
    -------
    tuple
        (normalized_signal, params)
    """
    method = method.lower()
    if method == 'minmax':
        x_norm, x_min, x_max = normalize_minmax(x)
        return x_norm.ravel(), {'method': 'minmax', 'min': x_min.ravel(), 'max': x_max.ravel()}
    elif method == 'zscore':
        x_norm, mean, std = normalize_zscore(x)
        return x_norm.ravel(), {'method': 'zscore', 'mean': mean.ravel(), 'std': std.ravel()}
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def build_proxy_state(df: pd.DataFrame,
                      acoustic_col: str = 'acoustic_data',
                      tau_col: str = 'tau_proxy',
                      v_col: str = 'V_proxy',
                      smooth_window: int = 101,
                      smooth_polyorder: int = 3,
                      derivative_method: str = 'central') -> pd.DataFrame:
    """
    Build the observed proxy state for the LANL dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input LANL dataframe with 'acoustic_data'.
    acoustic_col : str
        Column name for acoustic signal.
    tau_col : str
        Name of the output tau proxy column.
    v_col : str
        Name of the output velocity proxy column.
    smooth_window : int
        Smoothing window length.
    smooth_polyorder : int
        Savitzky-Golay polynomial order.
    derivative_method : str
        Method for derivative computation.

    Returns
    -------
    pd.DataFrame
        DataFrame with tau_proxy and V_proxy columns added.
    """
    if acoustic_col not in df.columns:
        raise ValueError(f"Missing required column: {acoustic_col}")

    df = df.copy()
    tau_signal = smooth_signal(df[acoustic_col].values.astype(float),
                               window=smooth_window,
                               polyorder=smooth_polyorder)
    df[tau_col] = tau_signal

    v_proxy = compute_derivative(tau_signal, dt=1.0, method=derivative_method)
    df[v_col] = v_proxy

    return df


def add_lanl_proxies(df: pd.DataFrame,
                     time_col: str = 'time',
                     acoustic_col: str = 'acoustic_data',
                     tau_col: str = 'tau_proxy',
                     v_col: str = 'V_proxy',
                     smooth: bool = True,
                     smooth_window: int = 101,
                     smooth_polyorder: int = 3,
                     derivative_method: str = 'savgol') -> pd.DataFrame:
    """
    Add LANL observed-state proxies for SINDy-style modeling.

    The LANL dataset does not contain a direct shear stress measurement.
    We treat acoustic_data as a proxy for tau and derive a velocity proxy
    from its smoothed derivative.

    Parameters
    ----------
    df : pd.DataFrame
        Input LANL dataframe.
    time_col : str
        Name of the time column.
    acoustic_col : str
        Name of the acoustic data column.
    tau_col : str
        Name of the output tau proxy column.
    v_col : str
        Name of the output velocity proxy column.
    smooth : bool
        Whether to smooth the acoustic signal before differentiation.
    smooth_window : int
        Window length for Savitzky-Golay smoothing.
    smooth_polyorder : int
        Polynomial order for Savitzky-Golay smoothing.
    derivative_method : str
        Differentiation method, currently only 'savgol' is supported.

    Returns
    -------
    pd.DataFrame
        DataFrame with added proxy columns.
    """
    if acoustic_col not in df.columns:
        raise ValueError(f"Missing acoustic column: {acoustic_col}")

    df = df.copy()
    df[tau_col] = df[acoustic_col].astype(float)

    signal = df[acoustic_col].values.astype(float)
    if smooth:
        signal = smooth_savgol(signal, window=smooth_window, polyorder=smooth_polyorder)
        df[f'{acoustic_col}_smoothed'] = signal

    if derivative_method == 'savgol':
        t = df[time_col].values if time_col in df.columns else None
        v_proxy = derivative_savgol(signal, t=t, window=smooth_window, polyorder=smooth_polyorder)
    else:
        raise ValueError(f"Unsupported derivative_method: {derivative_method}")

    df[v_col] = v_proxy
    return df
