"""
Módulo de utilidades generales.
"""

import logging


def setup_logger(name: str, level=logging.INFO):
    """
    Configura un logger con nombre específico.
    
    Args:
        name: Nombre del logger
        level: Nivel de logging (default: INFO)
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Handler para consola
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
