# 📊 Sales Forecasting - Machine Learning Project

Sistema de predicción de ventas utilizando Machine Learning para forecasting de productos deportivos. El proyecto incluye análisis de datos históricos, ingeniería de características, entrenamiento de modelos y una aplicación interactiva con Streamlit para simulaciones de escenarios.

## 🎯 Características Principales

- **Predicción recursiva día por día** con actualización automática de lags
- **Análisis de escenarios** de competencia y descuentos
- **Dashboard interactivo** con Streamlit
- **Detección de eventos especiales** (Black Friday, festivos españoles)
- **Visualizaciones avanzadas** de ventas proyectadas
- **Comparativa de múltiples escenarios** de pricing

## 📁 Estructura del Proyecto

```
Forecasting/
├── data/
│   ├── raw/                        # Datos crudos originales
│   │   ├── entrenamiento/          # Datos históricos para entrenar
│   │   │   ├── ventas.csv          # Histórico de ventas
│   │   │   └── competencia.csv     # Precios de competencia
│   │   └── inferencia/             # Datos para predicción
│   │       └── ventas_2025_inferencia.csv
│   └── processed/                  # Datos procesados
│       ├── df.csv                  # Dataset entrenamiento procesado
│       └── inferencia_df_transformado.csv  # Dataset inferencia procesado
├── notebooks/                      # Jupyter notebooks
│   ├── entrenamiento.ipynb         # Pipeline de entrenamiento
│   └── forecasting.ipynb           # Pipeline de inferencia
├── src/                            # Código fuente reutilizable
│   ├── __init__.py
│   ├── data_processing.py          # Procesamiento de datos
│   ├── features.py                 # Ingeniería de características
│   ├── models.py                   # Definición y entrenamiento
│   └── utils.py                    # Utilidades generales
├── models/                         # Modelos entrenados
│   └── modelo_final.joblib         # Modelo XGBoost entrenado
├── app/                            # Aplicación Streamlit
│   ├── __init__.py
│   ├── app.py                      # App principal de forecasting
│   └── streamlit_app.py            # App alternativa (en desarrollo)
├── docs/                           # Documentación
├── tests/                          # Tests unitarios
├── requirements.txt                # Dependencias Python
├── .gitignore                      # Archivos excluidos de Git
└── README.md                       # Este archivo
```

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.10 o superior
- pip (gestor de paquetes de Python)
- Git

### Pasos de Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone <repository-url>
   cd Forecasting
   ```

2. **Crear entorno virtual:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verificar instalación:**
   ```bash
   python -c "import streamlit; import pandas; import sklearn; print('✅ Todo instalado correctamente')"
   ```

## 💻 Uso

### Aplicación de Forecasting (Streamlit)

Para ejecutar la aplicación principal de predicción de ventas:

```bash
streamlit run app/app.py
```

La aplicación se abrirá en tu navegador en `http://localhost:8501`

#### Funcionalidades de la App:

1. **Selección de Producto**: Elige entre 25 productos deportivos
2. **Ajuste de Descuento**: Slider de -50% a +50% sobre precio base
3. **Escenarios de Competencia**: 
   - Actual (0%)
   - Competencia baja el precio (-5%)
   - Competencia sube el precio (+5%)
4. **Visualizaciones**:
   - Gráfico de predicción diaria
   - Destaque especial de Black Friday
   - Tabla detallada por día
   - Comparativa de escenarios
5. **Métricas**: Unidades proyectadas, ingresos, precio promedio, descuento

### Notebooks de Análisis

Para ejecutar los notebooks de desarrollo:

```bash
# Iniciar Jupyter Lab
jupyter lab

# O Jupyter Notebook
jupyter notebook
```

**Notebooks disponibles:**
- `entrenamiento.ipynb`: Pipeline completo de entrenamiento del modelo
- `forecasting.ipynb`: Proceso de inferencia y generación de predicciones

### Ejecutar Tests

```bash
pytest tests/
```

## 🔬 Metodología y Pipeline

### 1. Procesamiento de Datos

- **Carga de datos históricos**: Ventas y precios de competencia
- **Feature Engineering**:
  - Variables temporales: año, mes, día, semana, trimestre
  - Días especiales: festivos españoles, Black Friday, Cyber Monday
  - Lags: últimos 7 días de ventas
  - Media móvil de 7 días
  - Ratio de precios vs competencia
  - One-hot encoding de productos y categorías

### 2. Entrenamiento del Modelo

- **Algoritmo**: XGBoost (Gradient Boosting)
- **Features**: +90 variables incluyendo lags, temporales, y one-hot encoding
- **Validación**: Train-test split temporal
- **Métricas**: MAE, RMSE, R²
- **Guardado**: modelo_final.joblib

### 3. Predicción Recursiva

El sistema implementa predicción **día por día** para noviembre 2025:
- Predice día 1 usando lags históricos
- Actualiza lags con la predicción del día 1
- Predice día 2 con lags actualizados
- Repite el proceso para los 30 días
- Actualiza media móvil de 7 días en cada paso

### 4. Simulación de Escenarios

- **Variables de control**:
  - Descuento: -50% a +50%
  - Precio competencia: -5%, 0%, +5%
- **Recálculo automático** de features dependientes
- **Comparativa visual** entre escenarios

## 📊 Datos del Proyecto

### Dataset de Entrenamiento

- **Período**: Datos históricos de ventas
- **Productos**: 25 productos deportivos
- **Categorías**: Running, Fitness, Outdoor, Wellness
- **Variables**: Ventas, precios, competencia, festivos, promociones

### Dataset de Inferencia

- **Período**: Noviembre 2025 (30 días)
- **Estructura**: Pre-procesado con features temporales y lags iniciales
- **Uso**: Predicción recursiva día por día

## 🛠️ Tecnologías Utilizadas

### Core
- **Python 3.10+**
- **pandas 2.1+**: Manipulación de datos
- **numpy 1.26+**: Operaciones numéricas
- **scikit-learn 1.3+**: Preprocesamiento y métricas

### Machine Learning
- **XGBoost 2.0+**: Modelo principal de predicción
- **joblib**: Serialización de modelos

### Visualización & App
- **Streamlit 1.29+**: Aplicación web interactiva
- **matplotlib 3.8+**: Gráficos estáticos
- **seaborn 0.13+**: Visualizaciones estadísticas

### Desarrollo
- **Jupyter Lab 4.0+**: Notebooks interactivos
- **pytest**: Testing
- **holidays**: Detección de festivos españoles

## 📈 Características del Modelo

- ✅ Predicción recursiva con actualización de lags
- ✅ Manejo de eventos especiales (Black Friday)
- ✅ Soporte para múltiples productos y categorías
- ✅ Simulación de escenarios de pricing
- ✅ Validación con datos históricos
- ✅ Interfaz amigable para business users

- ✅ Interfaz amigable para business users

## 🎨 Capturas de Pantalla

### Dashboard Principal
La aplicación muestra:
- KPIs destacados: unidades proyectadas, ingresos, precio promedio
- Gráfico interactivo de predicción diaria
- Destaque visual de Black Friday
- Tabla detallada con información día por día
- Comparativa de escenarios de competencia

## 📝 Notas Importantes

### Archivos Grandes No Incluidos en Git

Debido a su tamaño, los siguientes archivos **NO** están incluidos en el repositorio:
- `models/modelo_final.joblib` (~50-100 MB)
- `data/raw/entrenamiento/*.csv`
- `data/processed/*.csv`

**Para trabajar con el proyecto completo**, necesitarás:
1. Los datos de entrenamiento originales
2. Ejecutar el notebook `entrenamiento.ipynb` para generar el modelo
3. Ejecutar el notebook `forecasting.ipynb` para procesar datos de inferencia

Alternativamente, puedes usar **Git LFS** para archivos grandes.

### Variables del Dataset

El modelo utiliza las siguientes variables (ver `copilot-instructions.md` para lista completa):
- Features temporales: fecha, año, mes, día_semana, trimestre, etc.
- Lags: últimos 7 días de ventas
- Features de pricing: precio_base, precio_venta, descuento_porcentaje
- Competencia: precio_competencia, ratio_precio
- One-hot encoding: nombre, categoría, subcategoría
- Eventos: es_festivo, es_black_friday, es_navidad, etc.

## 🤝 Contribuir

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/NuevaFuncionalidad`)
3. Commit tus cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/NuevaFuncionalidad`)
5. Abrir un Pull Request

### Guías de Estilo

- Código en español para comentarios y variables
- Usar type hints cuando sea posible
- Documentar funciones con docstrings
- Seguir PEP 8 para formato de código

## 🐛 Reportar Problemas

Si encuentras algún bug o tienes sugerencias, por favor abre un **Issue** en GitHub con:
- Descripción del problema
- Pasos para reproducirlo
- Comportamiento esperado vs real
- Screenshots (si aplica)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👨‍💻 Autor

**Alexis Acevedo**

