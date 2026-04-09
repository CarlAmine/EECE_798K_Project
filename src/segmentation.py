"""
Cycle segmentation: identify and extract stick-slip cycles from time-series data.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def find_peaks_simple(signal: np.ndarray, threshold: float = None,
                     prominence: float = None) -> np.ndarray:
    """
    Find peaks in a signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    threshold : float, optional
        Minimum height for peaks
    prominence : float, optional
        Minimum prominence
        
    Returns
    -------
    np.ndarray
        Indices of peaks
    """
    if threshold is None:
        threshold = np.mean(signal) + 0.5 * np.std(signal)
    
    peaks, _ = find_peaks(signal, height=threshold, prominence=prominence)
    
    return peaks


def find_local_extrema(signal: np.ndarray) -> tuple:
    """
    Find local maxima and minima.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
        
    Returns
    -------
    tuple
        (maxima_indices, minima_indices)
    """
    maxima, _ = find_peaks(signal)
    minima, _ = find_peaks(-signal)
    
    return maxima, minima


def segment_by_peaks(data: np.ndarray, peak_indices: np.ndarray) -> list:
    """
    Segment data between peaks.
    
    Parameters
    ----------
    data : np.ndarray
        Input data (or indices)
    peak_indices : np.ndarray
        Indices of peaks
        
    Returns
    -------
    list
        List of (start_idx, end_idx) tuples for segments between consecutive peaks
    """
    segments = []
    for i in range(len(peak_indices) - 1):
        start = peak_indices[i]
        end = peak_indices[i + 1]
        segments.append((start, end))
    
    return segments


def segment_cycles_autocorrelation(signal: np.ndarray, 
                                   max_lag: int = None) -> list:
    """
    Estimate cycle period using autocorrelation and segment accordingly.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    max_lag : int, optional
        Maximum lag to check for periodicity
        
    Returns
    -------
    list
        List of (start_idx, end_idx) tuples for identified cycles
    """
    if max_lag is None:
        max_lag = len(signal) // 4
    
    # Compute autocorrelation
    signal = signal - np.mean(signal)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr / autocorr[len(autocorr) // 2]
    
    # Find peaks in autocorrelation (starting from lag 1)
    center = len(autocorr) // 2
    acf_right = autocorr[center + 1:center + max_lag]
    
    peaks, _ = find_peaks(acf_right)
    
    if len(peaks) == 0:
        return [(0, len(signal))]
    
    # Use first peak as period estimate
    period = peaks[0] + 1
    
    # Segment based on period
    segments = []
    for i in range(0, len(signal) - period, period):
        segments.append((i, i + period))
    
    # Add final partial segment if it exists
    if segments[-1][1] < len(signal):
        segments.append((segments[-1][1], len(signal)))
    
    return segments


def segment_by_threshold_crossings(signal: np.ndarray, 
                                   threshold: float = None,
                                   rising: bool = True) -> list:
    """
    Segment data based on threshold crossings.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    threshold : float, optional
        Threshold value. If None, uses mean.
    rising : bool
        If True, detect rising crossings; if False, falling
        
    Returns
    -------
    list
        List of (start_idx, end_idx) tuples
    """
    if threshold is None:
        threshold = np.mean(signal)
    
    # Detect crossings
    crossings = np.where(np.diff(signal > threshold).astype(int) != 0)[0]
    
    if len(crossings) < 2:
        return [(0, len(signal))]
    
    segments = []
    for i in range(0, len(crossings) - 1, 2 if not rising else 2):
        start = crossings[i]
        end = crossings[i + 1] if i + 1 < len(crossings) else len(signal)
        if end > start + 1:
            segments.append((start, end))
    
    return segments


def extract_segment(df: pd.DataFrame, start_idx: int, 
                   end_idx: int) -> pd.DataFrame:
    """
    Extract a time segment from dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    start_idx : int
        Start index (inclusive)
    end_idx : int
        End index (exclusive)
        
    Returns
    -------
    pd.DataFrame
        Segment (reset index to start from 0)
    """
    segment = df.iloc[start_idx:end_idx].copy()
    return segment.reset_index(drop=True)


def extract_segments_df(df: pd.DataFrame, segments: list) -> list:
    """
    Extract multiple segments from dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    segments : list
        List of (start_idx, end_idx) tuples
        
    Returns
    -------
    list
        List of extracted segment dataframes
    """
    extracted = []
    for start, end in segments:
        seg = extract_segment(df, start, end)
        extracted.append(seg)
    
    return extracted


def segment_by_failure_resets(df: pd.DataFrame, ttf_col: str = 'time_to_failure',
                              min_cycle_length: int = 1000) -> list:
    """
    Segment LANL cycles based on resets in the time_to_failure signal.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with LANL failure signal.
    ttf_col : str
        Column name for the time_to_failure signal.
    min_cycle_length : int
        Minimum number of samples in a valid cycle.

    Returns
    -------
    list
        List of (start_idx, end_idx) tuples for each cycle.
    """
    if ttf_col not in df.columns:
        raise ValueError(f"Missing required column: {ttf_col}")

    tt = df[ttf_col].values
    resets = np.where(np.diff(tt) > 0)[0] + 1
    if len(resets) == 0:
        return [(0, len(df))] if len(df) >= min_cycle_length else []

    starts = np.concatenate(([0], resets))
    ends = np.concatenate((resets, [len(df)]))

    cycles = []
    for start, end in zip(starts, ends):
        if end - start >= min_cycle_length:
            cycles.append((int(start), int(end)))
    return cycles


def simple_stick_slip_segmentation(df: pd.DataFrame, 
                                   accel_col: str = 'atotal',
                                   time_col: str = 'time',
                                   method: str = 'peaks',
                                   threshold_frac: float = 0.5,
                                   min_segment_length: int = 10) -> tuple:
    """
    Simple stick-slip cycle segmentation.
    
    Assumes cycles are characterized by repeated peaks in acceleration.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with acceleration column
    accel_col : str
        Column name for acceleration (e.g., 'atotal')
    time_col : str
        Column name for time
    method : str
        'peaks' or 'autocorr' or 'threshold'
    threshold_frac : float
        Fraction of std above mean for threshold (used in 'peaks' method)
    min_segment_length : int
        Minimum length for a valid segment
        
    Returns
    -------
    tuple
        (segments_list, peaks_info_dict)
    """
    signal = df[accel_col].values
    t = df[time_col].values if time_col in df.columns else np.arange(len(df))
    
    if method == 'peaks':
        threshold = np.mean(signal) + threshold_frac * np.std(signal)
        peaks = find_peaks_simple(signal, threshold=threshold)
        segments = segment_by_peaks(np.arange(len(signal)), peaks)
    elif method == 'autocorr':
        segments = segment_cycles_autocorrelation(signal)
    elif method == 'threshold':
        threshold = np.mean(signal) + threshold_frac * np.std(signal)
        segments = segment_by_threshold_crossings(signal, threshold=threshold)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Filter by minimum length
    segments = [(s, e) for s, e in segments if (e - s) >= min_segment_length]
    
    info = {
        'method': method,
        'n_segments': len(segments),
        'segment_lengths': [e - s for s, e in segments],
        'mean_segment_length': np.mean([e - s for s, e in segments]) if segments else 0,
        'min_segment_length': min([e - s for s, e in segments]) if segments else 0,
        'max_segment_length': max([e - s for s, e in segments]) if segments else 0,
    }
    
    return segments, info


def segment_lanl_cycles(df: pd.DataFrame,
                        failure_col: str = 'time_to_failure',
                        min_cycle_length: int = 1000) -> list:
    """
    Identify LANL laboratory earthquake cycles from the time_to_failure signal.

    The LANL training data is a continuous series where time_to_failure
    decreases until a laboratory earthquake occurs and then resets to a larger value.
    This function segments cycles at the points where the failure timer resets.

    Parameters
    ----------
    df : pd.DataFrame
        LANL dataframe containing a failure column.
    failure_col : str
        Column name for the time to failure signal.
    min_cycle_length : int
        Minimum number of samples for a valid cycle.

    Returns
    -------
    list
        List of (start_idx, end_idx) cycle tuples.
    """
    if failure_col not in df.columns:
        raise ValueError(f"Missing failure column: {failure_col}")

    tt = df[failure_col].values
    reset_indices = np.where(np.diff(tt) > 0)[0] + 1
    if len(reset_indices) == 0:
        return [(0, len(df))] if len(df) >= min_cycle_length else []

    starts = np.concatenate(([0], reset_indices))
    ends = np.concatenate((reset_indices, [len(df)]))

    segments = []
    for start, end in zip(starts, ends):
        if end - start >= min_cycle_length:
            segments.append((int(start), int(end)))

    return segments


def extract_lanl_cycle_segments(df: pd.DataFrame,
                                failure_col: str = 'time_to_failure',
                                min_cycle_length: int = 1000) -> list:
    """
    Extract individual LANL cycles as dataframe segments.

    Parameters
    ----------
    df : pd.DataFrame
        LANL training data.
    failure_col : str
        Column name for the time to failure signal.
    min_cycle_length : int
        Minimum length for a valid cycle.

    Returns
    -------
    list
        List of pandas DataFrames for each cycle.
    """
    segments = segment_lanl_cycles(df, failure_col=failure_col,
                                   min_cycle_length=min_cycle_length)
    return extract_segments_df(df, segments)
