"""
Módulo para definición de modelos.
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb


def create_linear_model():
    """Crea un modelo de regresión lineal."""
    return LinearRegression()


def create_random_forest(n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
    """
    Crea un modelo de Random Forest.
    
    Args:
        n_estimators: Número de árboles
        max_depth: Profundidad máxima de árboles
        random_state: Semilla aleatoria
    
    Returns:
        Modelo de Random Forest
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )


def create_gradient_boosting(n_estimators: int = 100, learning_rate: float = 0.1, random_state: int = 42):
    """
    Crea un modelo de Gradient Boosting.
    
    Args:
        n_estimators: Número de estimadores
        learning_rate: Tasa de aprendizaje
        random_state: Semilla aleatoria
    
    Returns:
        Modelo de Gradient Boosting
    """
    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )


def create_xgboost(n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 6, random_state: int = 42):
    """
    Crea un modelo de XGBoost.
    
    Args:
        n_estimators: Número de estimadores
        learning_rate: Tasa de aprendizaje
        max_depth: Profundidad máxima
        random_state: Semilla aleatoria
    
    Returns:
        Modelo de XGBoost
    """
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
