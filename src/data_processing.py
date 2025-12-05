"""
Módulo para procesamiento de datos.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_data(filepath: str) -> pd.DataFrame:
    """
    Carga datos desde un archivo CSV.
    
    Args:
        filepath: Ruta del archivo CSV
    
    Returns:
        DataFrame cargado
    """
    return pd.read_csv(filepath)


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Maneja valores faltantes en el dataframe.
    
    Args:
        df: DataFrame con posibles valores faltantes
        strategy: Estrategia ('mean', 'median', 'drop', 'forward_fill')
    
    Returns:
        DataFrame sin valores faltantes
    """
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'drop':
        return df.dropna()
    elif strategy == 'forward_fill':
        return df.fillna(method='ffill')
    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")


def normalize_data(df: pd.DataFrame, columns: list = None) -> Tuple[pd.DataFrame, dict]:
    """
    Normaliza los datos utilizando z-score.
    
    Args:
        df: DataFrame a normalizar
        columns: Columnas a normalizar (todas si es None)
    
    Returns:
        Tupla con DataFrame normalizado y parámetros de normalización
    """
    from sklearn.preprocessing import StandardScaler
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    scaler = StandardScaler()
    df_normalized = df.copy()
    df_normalized[columns] = scaler.fit_transform(df[columns])
    
    norm_params = {
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist(),
        'columns': columns
    }
    
    return df_normalized, norm_params
