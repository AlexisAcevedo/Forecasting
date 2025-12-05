# Guía de Contribución

¡Gracias por tu interés en contribuir al proyecto! 🎉

## 🚀 Cómo Contribuir

### 1. Fork y Clone

```bash
# Fork el proyecto en GitHub, luego:
git clone https://github.com/TU_USUARIO/Forecasting.git
cd Forecasting
```

### 2. Crear una Rama

```bash
git checkout -b feature/mi-nueva-funcionalidad
```

Tipos de ramas:
- `feature/` - Nueva funcionalidad
- `bugfix/` - Corrección de bugs
- `hotfix/` - Corrección urgente
- `docs/` - Documentación

### 3. Configurar Entorno

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 4. Hacer Cambios

- Sigue las guías de estilo (ver abajo)
- Escribe código limpio y documentado
- Agrega tests si es necesario
- Actualiza la documentación

### 5. Commit

```bash
git add .
git commit -m "feat: agregar nueva funcionalidad X"
```

Tipos de commits (Conventional Commits):
- `feat:` - Nueva funcionalidad
- `fix:` - Corrección de bug
- `docs:` - Documentación
- `style:` - Formato de código
- `refactor:` - Refactorización
- `test:` - Tests
- `chore:` - Tareas de mantenimiento

### 6. Push y Pull Request

```bash
git push origin feature/mi-nueva-funcionalidad
```

Luego abre un Pull Request en GitHub con:
- Descripción clara de los cambios
- Referencias a issues relacionados
- Screenshots si aplica

## 📋 Guías de Estilo

### Python

- **PEP 8**: Seguir estándar de Python
- **Comentarios**: En español, claros y concisos
- **Docstrings**: Para todas las funciones y clases
- **Type Hints**: Usar cuando sea posible
- **Imports**: Ordenados (stdlib → third-party → local)

Ejemplo:

```python
def calcular_prediccion(
    modelo: XGBRegressor,
    datos: pd.DataFrame,
    dias: int = 30
) -> pd.DataFrame:
    """
    Calcula predicciones recursivas día por día.
    
    Args:
        modelo: Modelo XGBoost entrenado
        datos: DataFrame con features preparados
        dias: Número de días a predecir
        
    Returns:
        DataFrame con predicciones diarias
    """
    # Implementación...
    pass
```

### Git

- Commits pequeños y atómicos
- Mensajes descriptivos en español
- Referencias a issues cuando aplique

### Tests

- Tests unitarios para funciones críticas
- Nombrar tests descriptivamente
- Usar pytest fixtures

## 🐛 Reportar Bugs

Usa el template de **Bug Report** en Issues e incluye:
- Descripción clara del problema
- Pasos para reproducir
- Comportamiento esperado vs actual
- Entorno (OS, Python version, etc.)

## 💡 Sugerir Features

Usa el template de **Feature Request** en Issues e incluye:
- Descripción del feature
- Motivación y casos de uso
- Propuesta de implementación

## ❓ Preguntas

Si tienes preguntas, abre un **Issue** con la etiqueta `question`.

## 📝 Proceso de Review

1. Tu PR será revisado por los maintainers
2. Se pueden solicitar cambios
3. Una vez aprobado, se hará merge
4. Se cerrará el issue relacionado

## 🙏 Código de Conducta

- Sé respetuoso y profesional
- Acepta feedback constructivo
- Ayuda a otros contribuidores
- Mantén un ambiente colaborativo

---

¡Gracias por contribuir! 🚀
