"""
Módulo para ingeniería de características.
"""

import pandas as pd
import numpy as np


def create_lagged_features(df: pd.DataFrame, column: str, lags: list) -> pd.DataFrame:
    """
    Crea características con valores rezagados.
    
    Args:
        df: DataFrame
        column: Columna para crear lags
        lags: Lista de números de lag
    
    Returns:
        DataFrame con nuevas columnas de lag
    """
    df_lagged = df.copy()
    for lag in lags:
        df_lagged[f'{column}_lag_{lag}'] = df[column].shift(lag)
    return df_lagged.dropna()


def create_rolling_features(df: pd.DataFrame, column: str, windows: list) -> pd.DataFrame:
    """
    Crea características de media móvil.
    
    Args:
        df: DataFrame
        column: Columna para crear rolling features
        windows: Lista de tamaños de ventana
    
    Returns:
        DataFrame con nuevas columnas de rolling
    """
    df_rolling = df.copy()
    for window in windows:
        df_rolling[f'{column}_rolling_mean_{window}'] = df[column].rolling(window).mean()
        df_rolling[f'{column}_rolling_std_{window}'] = df[column].rolling(window).std()
    return df_rolling.dropna()


def create_temporal_features(df: pd.DataFrame, date_column: str = None) -> pd.DataFrame:
    """
    Crea características temporales (hora, día, mes, año, etc.).
    
    Args:
        df: DataFrame con columna de fecha
        date_column: Nombre de la columna de fecha (si es None, usa índice)
    
    Returns:
        DataFrame con características temporales
    """
    df_temporal = df.copy()
    
    if date_column is not None:
        df_temporal[date_column] = pd.to_datetime(df_temporal[date_column])
        date_series = df_temporal[date_column]
    else:
        date_series = pd.to_datetime(df_temporal.index)
    
    df_temporal['year'] = date_series.dt.year
    df_temporal['month'] = date_series.dt.month
    df_temporal['day'] = date_series.dt.day
    df_temporal['dayofweek'] = date_series.dt.dayofweek
    df_temporal['quarter'] = date_series.dt.quarter
    df_temporal['dayofyear'] = date_series.dt.dayofyear
    
    return df_temporal
