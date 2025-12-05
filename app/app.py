"""
Aplicación Streamlit para simulación de predicciones de ventas de noviembre 2025.
Implementa predicciones recursivas día por día actualizando lags.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import joblib
import warnings

warnings.filterwarnings('ignore')

# Obtener la ruta base del proyecto (parent del directorio app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_final.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "inferencia_df_transformado.csv"

# Verificar que las rutas existen
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"El modelo no existe en: {MODEL_PATH}")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Los datos no existen en: {DATA_PATH}")

# Configuración de la página
st.set_page_config(
    page_title="Sales Forecasting November 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
    <style>
    .main-header {
        color: #667eea;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .black-friday {
        background-color: #ffe6e6;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== FUNCIONES AUXILIARES ====================

@st.cache_resource
def load_model():
    """Carga el modelo entrenado."""
    try:
        model = joblib.load(str(MODEL_PATH))
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        return None

@st.cache_data
def load_inference_data():
    """Carga los datos de inferencia de noviembre 2025."""
    try:
        df = pd.read_csv(str(DATA_PATH))
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")
        return None

def get_unique_products(df):
    """Extrae los productos únicos del dataframe."""
    return sorted(df['nombre'].unique().tolist())

def prepare_product_data(df, product_name):
    """Prepara los datos para un producto específico."""
    product_df = df[df['nombre'] == product_name].copy()
    product_df = product_df.sort_values('fecha').reset_index(drop=True)
    return product_df

def update_lags(row, previous_prediction, lag_values):
    """Actualiza los valores de lag para la predicción recursiva."""
    row['unidades_vendidas_lag7'] = lag_values['lag6']
    row['unidades_vendidas_lag6'] = lag_values['lag5']
    row['unidades_vendidas_lag5'] = lag_values['lag4']
    row['unidades_vendidas_lag4'] = lag_values['lag3']
    row['unidades_vendidas_lag3'] = lag_values['lag2']
    row['unidades_vendidas_lag2'] = lag_values['lag1']
    row['unidades_vendidas_lag1'] = previous_prediction
    return row

def update_moving_average(recent_predictions, window=7):
    """Calcula la media móvil de las últimas predicciones."""
    if len(recent_predictions) < window:
        return np.mean(recent_predictions) if recent_predictions else 0
    return np.mean(recent_predictions[-window:])

def make_recursive_predictions(model, df, discount_adjustment, competition_scenario):
    """
    Realiza predicciones recursivas día por día actualizando los lags.
    
    Args:
        model: Modelo entrenado
        df: DataFrame preparado para un producto (noviembre)
        discount_adjustment: Ajuste de descuento en porcentaje (-50 a +50)
        competition_scenario: 'actual' (0%), 'lower' (-5%), 'higher' (+5%)
    
    Returns:
        DataFrame con predicciones
    """
    df = df.copy()
    
    # Ajustar precios según los controles
    df['precio_venta'] = df['precio_base'] * (1 + discount_adjustment / 100)
    
    # Ajustar precio de competencia según el escenario
    precio_competencia_base = df['precio_competencia'].copy()
    if competition_scenario == 'lower':
        df['precio_competencia'] = precio_competencia_base * 0.95
    elif competition_scenario == 'higher':
        df['precio_competencia'] = precio_competencia_base * 1.05
    
    # Recalcular variables de precio
    df['descuento_porcentaje'] = ((df['precio_venta'] - df['precio_base']) / df['precio_base']) * 100
    df['ratio_precio'] = df['precio_venta'] / df['precio_competencia']
    
    # Obtener feature names del modelo
    feature_names = model.feature_names_in_
    
    predictions = []
    recent_predictions = []
    
    for idx, row in df.iterrows():
        # Para el primer día, usar los lags del archivo
        if idx > 0:
            # Preparar lags para los días siguientes
            lag_values = {
                'lag1': df.loc[idx-1, 'unidades_vendidas_lag1'],
                'lag2': df.loc[idx-1, 'unidades_vendidas_lag2'],
                'lag3': df.loc[idx-1, 'unidades_vendidas_lag3'],
                'lag4': df.loc[idx-1, 'unidades_vendidas_lag4'],
                'lag5': df.loc[idx-1, 'unidades_vendidas_lag5'],
                'lag6': df.loc[idx-1, 'unidades_vendidas_lag6'],
            }
            df.loc[idx] = update_lags(row, predictions[-1], lag_values)
            
            # Actualizar media móvil
            df.loc[idx, 'unidades_vendidas_media_movil_7d'] = update_moving_average(
                recent_predictions, window=7
            )
        
        # Preparar features para la predicción
        X = df.loc[[idx], feature_names].copy()
        
        # Hacer predicción
        pred = model.predict(X)[0]
        pred = max(0, pred)  # No permitir predicciones negativas
        predictions.append(pred)
        recent_predictions.append(pred)
    
    df['prediccion_unidades'] = predictions
    df['ingresos_proyectados'] = df['prediccion_unidades'] * df['precio_venta']
    
    return df

def format_currency(value):
    """Formatea un valor como moneda en euros."""
    return f"€{value:,.2f}"

def format_units(value):
    """Formatea unidades sin decimales."""
    return f"{int(round(value)):,}"

# ==================== APLICACIÓN PRINCIPAL ====================

def main():
    # Cargar modelo y datos
    model = load_model()
    df = load_inference_data()
    
    if model is None or df is None:
        st.error("No se pudieron cargar los componentes necesarios.")
        return
    
    # Sidebar - Controles de Simulación
    with st.sidebar:
        st.markdown("### 🎛️ Controles de Simulación")
        st.divider()
        
        # Selector de producto
        products = get_unique_products(df)
        selected_product = st.selectbox(
            "📦 Selecciona Producto",
            products,
            help="Elige el producto para simular"
        )
        
        st.divider()
        
        # Slider de descuento
        discount = st.slider(
            "💰 Ajuste de Descuento",
            min_value=-50,
            max_value=50,
            value=0,
            step=5,
            help="Ajusta el descuento sobre el precio base"
        )
        
        discount_label = "Sin cambios" if discount == 0 else f"{discount:+d}%"
        st.caption(f"**Descuento:** {discount_label}")
        
        st.divider()
        
        # Escenario de competencia
        st.markdown("**🏆 Escenario de Competencia**")
        competition = st.radio(
            "Elige escenario:",
            ["Actual (0%)", "Competencia -5%", "Competencia +5%"],
            help="Cómo varía el precio de la competencia"
        )
        
        competition_map = {
            "Actual (0%)": "actual",
            "Competencia -5%": "lower",
            "Competencia +5%": "higher"
        }
        competition_scenario = competition_map[competition]
        
        st.divider()
        
        # Botón de simulación
        simulate_button = st.button(
            "🚀 Simular Ventas",
            use_container_width=True,
            type="primary"
        )
    
    # Zona principal - Dashboard
    if simulate_button:
        with st.spinner("⏳ Realizando predicciones recursivas..."):
            # Preparar datos del producto
            product_df = prepare_product_data(df, selected_product)
            
            # Hacer predicciones recursivas
            results_df = make_recursive_predictions(
                model,
                product_df,
                discount,
                competition_scenario
            )
        
        # Header
        st.markdown(f"<div class='main-header'>📈 Simulación de Ventas - Noviembre 2025</div>", 
                   unsafe_allow_html=True)
        st.markdown(f"### 📦 Producto: **{selected_product}**")
        st.divider()
        
        # KPIs destacados
        total_units = results_df['prediccion_unidades'].sum()
        total_revenue = results_df['ingresos_proyectados'].sum()
        avg_price = results_df['precio_venta'].mean()
        avg_discount = results_df['descuento_porcentaje'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Unidades Proyectadas",
                format_units(total_units),
                delta="30 días",
                delta_color="off"
            )
        
        with col2:
            st.metric(
                "💵 Ingresos Proyectados",
                format_currency(total_revenue),
                delta="Noviembre",
                delta_color="off"
            )
        
        with col3:
            st.metric(
                "🏷️ Precio Promedio",
                format_currency(avg_price),
                delta="Por unidad",
                delta_color="off"
            )
        
        with col4:
            discount_display = f"{avg_discount:+.1f}%"
            st.metric(
                "💲 Descuento Promedio",
                discount_display,
                delta="Vs. precio base",
                delta_color="off"
            )
        
        st.divider()
        
        # Gráfico de predicción diaria
        st.markdown("### 📈 Predicción Diaria de Ventas")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Línea principal de predicción
        ax.plot(
            results_df['dia_mes'].values,
            results_df['prediccion_unidades'].values,
            linewidth=2.5,
            color='#667eea',
            marker='o',
            markersize=6,
            label='Predicción'
        )
        
        # Marcar Black Friday
        black_friday_idx = results_df[results_df['dia_mes'] == 28]
        if not black_friday_idx.empty:
            bf_day = black_friday_idx.iloc[0]['dia_mes']
            bf_units = black_friday_idx.iloc[0]['prediccion_unidades']
            
            # Línea vertical de Black Friday
            ax.axvline(x=bf_day, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7)
            
            # Punto destacado
            ax.scatter([bf_day], [bf_units], color='#e74c3c', s=300, zorder=5, 
                      edgecolors='darkred', linewidth=2)
            
            # Anotación
            ax.annotate(
                '🔥 BLACK FRIDAY',
                xy=(bf_day, bf_units),
                xytext=(bf_day - 3, bf_units + max(results_df['prediccion_unidades']) * 0.1),
                fontsize=10,
                fontweight='bold',
                color='#e74c3c',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffe6e6', edgecolor='#e74c3c'),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2)
            )
        
        # Estilo del gráfico
        sns.set_style("whitegrid")
        ax.set_xlabel('Día de Noviembre', fontsize=12, fontweight='bold')
        ax.set_ylabel('Unidades Vendidas', fontsize=12, fontweight='bold')
        ax.set_title('Predicción de Ventas Diarias - Noviembre 2025', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 31, 2))
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.divider()
        
        # Tabla detallada
        st.markdown("### 📋 Detalles Diarios de Noviembre")
        
        # Preparar tabla para mostrar
        display_df = results_df[[
            'fecha', 'nombre_dia_semana', 'precio_venta', 'precio_competencia',
            'descuento_porcentaje', 'prediccion_unidades', 'ingresos_proyectados'
        ]].copy()
        
        display_df.columns = [
            'Fecha', 'Día Semana', 'P. Venta', 'P. Competencia',
            'Descuento', 'Unidades', 'Ingresos'
        ]
        
        display_df['Fecha'] = display_df['Fecha'].dt.strftime('%d/%m/%Y')
        display_df['P. Venta'] = display_df['P. Venta'].apply(format_currency)
        display_df['P. Competencia'] = display_df['P. Competencia'].apply(format_currency)
        display_df['Descuento'] = display_df['Descuento'].apply(lambda x: f"{x:+.1f}%")
        display_df['Unidades'] = display_df['Unidades'].apply(format_units)
        display_df['Ingresos'] = display_df['Ingresos'].apply(format_currency)
        
        # Usar HTML para colorear Black Friday
        def highlight_black_friday(row):
            if row['Fecha'].startswith('28'):
                return ['background-color: #ffe6e6'] * len(row)
            return [''] * len(row)
        
        styled_df = display_df.style.apply(highlight_black_friday, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Comparativa de escenarios
        st.markdown("### 🎯 Comparativa de Escenarios de Competencia")
        
        col1, col2, col3 = st.columns(3)
        
        # Escenario Actual
        results_actual = make_recursive_predictions(model, product_df, discount, 'actual')
        units_actual = results_actual['prediccion_unidades'].sum()
        revenue_actual = results_actual['ingresos_proyectados'].sum()
        
        with col1:
            st.markdown("**📊 Escenario Actual**")
            st.metric("Unidades", format_units(units_actual), delta=None)
            st.metric("Ingresos", format_currency(revenue_actual), delta=None)
        
        # Escenario Competencia -5%
        results_lower = make_recursive_predictions(model, product_df, discount, 'lower')
        units_lower = results_lower['prediccion_unidades'].sum()
        revenue_lower = results_lower['ingresos_proyectados'].sum()
        units_delta_lower = units_lower - units_actual
        revenue_delta_lower = revenue_lower - revenue_actual
        
        with col2:
            st.markdown("**📈 Competencia -5%**")
            st.metric("Unidades", format_units(units_lower), 
                     delta=f"{units_delta_lower:+.0f}" if units_delta_lower != 0 else None)
            st.metric("Ingresos", format_currency(revenue_lower),
                     delta=format_currency(revenue_delta_lower) if revenue_delta_lower != 0 else None)
        
        # Escenario Competencia +5%
        results_higher = make_recursive_predictions(model, product_df, discount, 'higher')
        units_higher = results_higher['prediccion_unidades'].sum()
        revenue_higher = results_higher['ingresos_proyectados'].sum()
        units_delta_higher = units_higher - units_actual
        revenue_delta_higher = revenue_higher - revenue_actual
        
        with col3:
            st.markdown("**📉 Competencia +5%**")
            st.metric("Unidades", format_units(units_higher),
                     delta=f"{units_delta_higher:+.0f}" if units_delta_higher != 0 else None)
            st.metric("Ingresos", format_currency(revenue_higher),
                     delta=format_currency(revenue_delta_higher) if revenue_delta_higher != 0 else None)
        
        st.divider()
        
        # Información adicional
        st.markdown("### ℹ️ Información de la Simulación")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"✅ **Descuento aplicado:** {discount:+d}% sobre precio base")
        
        with col2:
            st.info(f"✅ **Escenario competencia:** {competition}")

if __name__ == "__main__":
    main()