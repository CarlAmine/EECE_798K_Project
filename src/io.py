"""
Data I/O utilities for loading mechanical datasets.
Supports CSV, MAT, and other common formats.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    **kwargs
        Additional arguments passed to pd.read_csv
        
    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    return pd.read_csv(filepath, **kwargs)


def load_mat(filepath: str) -> dict:
    """
    Load a MATLAB .mat file.
    
    Parameters
    ----------
    filepath : str
        Path to .mat file
        
    Returns
    -------
    dict
        Dictionary-like structure with arrays
    """
    try:
        import scipy.io as sio
        return sio.loadmat(filepath)
    except ImportError:
        raise ImportError("scipy required for MAT file loading")


def load_mechanical_data(filepath: str, file_type: str = None) -> pd.DataFrame:
    """
    Load mechanical data from various formats.
    Automatically detects format if not specified.
    
    Parameters
    ----------
    filepath : str
        Path to data file
    file_type : str, optional
        File type ('csv', 'mat', etc.). If None, inferred from extension.
        
    Returns
    -------
    pd.DataFrame
        Loaded data
        
    Raises
    ------
    ValueError
        If file type is not supported or file not found
    FileNotFoundError
        If file does not exist
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if file_type is None:
        file_type = filepath.suffix.lstrip('.').lower()
    
    if file_type == 'csv':
        return load_csv(str(filepath))
    elif file_type in ['mat', 'matlab']:
        data = load_mat(str(filepath))
        # Try to convert to DataFrame if it's a single matrix
        # This is format-dependent; adjust as needed
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def validate_columns(df: pd.DataFrame, required_cols: list = None, 
                     expected_cols: list = None) -> bool:
    """
    Validate that DataFrame has required columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to validate
    required_cols : list, optional
        Columns that must be present; raises ValueError if missing
    expected_cols : list, optional
        Columns that should be present; returns False if missing
        
    Returns
    -------
    bool
        True if all expected columns are present (or if expected_cols is None)
        
    Raises
    ------
    ValueError
        If any required columns are missing
    """
    cols = set(df.columns)
    
    if required_cols:
        missing = set(required_cols) - cols
        if missing:
            raise ValueError(f"Missing required columns: {missing}. "
                           f"Available: {cols}")
    
    if expected_cols:
        return set(expected_cols).issubset(cols)
    
    return True


def load_lanl_train(csv_path: str = 'data/train.csv', nrows: int | None = None) -> pd.DataFrame:
    """
    Load the LANL earthquake prediction training data.

    Parameters
    ----------
    csv_path : str
        Path to the LANL train.csv file.
    nrows : int | None
        Number of rows to read from the file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing columns ['time', 'acoustic_data', 'time_to_failure'].
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            "Expected LANL dataset at data/train.csv. "
            f"Please place the Kaggle LANL train.csv file there or pass a valid csv_path."
        )

    df = pd.read_csv(csv_path, nrows=nrows)
    validate_columns(df, required_cols=['acoustic_data', 'time_to_failure'])

    df = df.copy()
    df['time'] = np.arange(len(df), dtype=float)
    df = df[['time', 'acoustic_data', 'time_to_failure']]
    return df


def describe_dataset(df: pd.DataFrame) -> dict:
    """
    Generate a summary of dataset properties.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to describe
        
    Returns
    -------
    dict
        Summary with keys: n_samples, n_vars, columns, dtypes, 
        missing_values, time_col, physical_vars
    """
    summary = {
        'n_samples': len(df),
        'n_vars': len(df.columns),
        'columns': list(df.columns),
        'dtypes': dict(df.dtypes),
        'missing_values': dict(df.isnull().sum()),
    }
    
    # Try to identify time column
    time_cols = [c for c in df.columns if c.lower() in ['time', 't']]
    summary['time_col'] = time_cols[0] if time_cols else None
    
    # Identify physical variables (non-time columns)
    if summary['time_col']:
        summary['physical_vars'] = [c for c in df.columns 
                                    if c != summary['time_col']]
    else:
        summary['physical_vars'] = df.columns.tolist()
    
    return summary
