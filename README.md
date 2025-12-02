# Sistema Inteligente de Recomendación de Precios

Este proyecto consiste en un sistema inteligente que recomienda precios óptimos para productos, estimando la demanda esperada y los ingresos proyectados en función de distintos escenarios de precios. Utiliza un modelo de **XGBoost** entrenado sobre un conjunto de datos de precios de productos, y la justificación de las recomendaciones se realiza mediante **SHAP** para la interpretación de características.

## Características

- **Predicción de Precio Óptimo**: Basado en un conjunto de datos de productos, el sistema predice el precio óptimo que maximiza los ingresos esperados.
- **Estimación de Demanda**: La demanda esperada y los ingresos se calculan para cada precio simulado.
- **Elasticidad del Precio**: El sistema calcula la elasticidad del producto para determinar cómo cambia la demanda ante variaciones de precio.
- **Explicación SHAP**: Utiliza **SHAP** para proporcionar una explicación detallada de cómo cada característica afecta la recomendación de precio.

## Tecnologías Utilizadas

- **Python**: Lenguaje de programación principal.
- **XGBoost**: Modelo de **regresión** utilizado para la predicción de la demanda y los ingresos.
- **SHAP**: Herramienta para la interpretación de modelos, proporcionando explicaciones locales.
- **Streamlit**: Interfaz de usuario web para la recomendación interactiva de precios.
- **CustomTkinter**: Interfaz gráfica de usuario (GUI) para la versión de escritorio.
- **pandas**: Librería para el manejo de datos y procesamiento de DataFrames.
- **joblib**: Para guardar y cargar el modelo entrenado.
- **scikit-learn**: Para el preprocesamiento de datos y validación del modelo.

## Instalación

Para configurar el proyecto en tu entorno local, sigue estos pasos:

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/tu_usuario/sistema-inteligente-precios.git
   cd sistema-inteligente-precios
   
2. **Crear un entorno virtual (opcional pero recomendado)**:
    ```bash
   python -m venv venv
   source venv/bin/activate  # En Linux/macOS
   venv\Scripts\activate     # En Windows
    
3. **Instalar las dependencias**:
Asegúrate de tener pip actualizado y luego instala todas las dependencias necesarias:
```bash
pip install -r requirements.txt

Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE
 para más detalles.
