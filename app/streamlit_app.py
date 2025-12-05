"""
Aplicación principal con Streamlit para visualización y predicción.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Añadir el directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logger
from src.data_processing import load_data

logger = setup_logger(__name__)


def main():
    """Función principal de la aplicación Streamlit."""
    
    # Configuración de la página
    st.set_page_config(
        page_title="ML Forecasting App",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Título
    st.title("🎯 Machine Learning Forecasting App")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📋 Menú")
    menu = st.sidebar.radio(
        "Selecciona una sección:",
        ["Home", "Cargar Datos", "Análisis Exploratorio", "Entrenamiento", "Predicción"]
    )
    
    # Home
    if menu == "Home":
        st.header("Bienvenido al Proyecto de Forecasting")
        st.markdown("""
        Esta aplicación permite:
        - 📥 Cargar y explorar datos
        - 📊 Realizar análisis exploratorio de datos (EDA)
        - 🤖 Entrenar modelos de machine learning
        - 🔮 Hacer predicciones
        
        ### Características
        - Múltiples modelos disponibles (Regresión Lineal, Random Forest, XGBoost)
        - Normalización de datos
        - Ingeniería de características
        - Evaluación automática de modelos
        """)
    
    # Cargar Datos
    elif menu == "Cargar Datos":
        st.header("📥 Cargar Datos")
        
        uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("Archivo cargado correctamente!")
            
            st.subheader("Vista Previa de los Datos")
            st.dataframe(df.head(10))
            
            st.subheader("Información del Dataset")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Número de Filas", df.shape[0])
            with col2:
                st.metric("Número de Columnas", df.shape[1])
            with col3:
                st.metric("Valores Faltantes", df.isnull().sum().sum())
            
            # Guardar en sesión
            st.session_state.df = df
            st.info("Datos guardados en sesión. Continúa con el siguiente paso.")
    
    # Análisis Exploratorio
    elif menu == "Análisis Exploratorio":
        st.header("📊 Análisis Exploratorio de Datos")
        
        if 'df' not in st.session_state:
            st.warning("Por favor, carga los datos primero.")
        else:
            df = st.session_state.df
            
            st.subheader("Estadísticas Descriptivas")
            st.dataframe(df.describe())
            
            st.subheader("Correlación entre Variables")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                st.dataframe(corr_matrix)
    
    # Entrenamiento
    elif menu == "Entrenamiento":
        st.header("🤖 Entrenamiento de Modelos")
        st.info("Funcionalidad en desarrollo...")
    
    # Predicción
    elif menu == "Predicción":
        st.header("🔮 Predicción")
        st.info("Funcionalidad en desarrollo...")


if __name__ == "__main__":
    main()
