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

##Uso
Versión de Streamlit (Web):

Para ejecutar la interfaz web con Streamlit, corre el siguiente comando:

streamlit run app_streamlit.py


La aplicación se abrirá en tu navegador y podrás cargar un archivo CSV, seleccionar una categoría de producto, ingresar un precio y recibir la recomendación de precio con una justificación detallada.

Versión de Tkinter (Escritorio):

Para ejecutar la versión de escritorio con Tkinter, corre el siguiente comando:

python app_tkinter.py


La interfaz gráfica de usuario de Tkinter se abrirá, donde podrás cargar el archivo CSV, ingresar un precio y obtener la recomendación de precio, junto con la justificación.

Estructura del Proyecto
sistema-inteligente-precios/
│
├── app_streamlit.py         # Interfaz web con Streamlit
├── app_tkinter.py          # Interfaz de escritorio con Tkinter
├── modelo_precios.py       # Lógica del modelo de recomendación y predicción
├── retail_price.csv        # Conjunto de datos de precios de productos
├── xgb_model.joblib        # Modelo XGBoost entrenado
├── requirements.txt        # Dependencias necesarias para el proyecto
└── README.md               # Documentación del proyecto

Dependencias

El proyecto requiere las siguientes librerías:

pandas==1.5.3

numpy==1.24.3

joblib==1.5.2

xgboost==3.1.2

shap==0.50.0

category_encoders==2.9.0

customtkinter==4.0.0

scikit-learn==1.1.2

streamlit==1.19.0

matplotlib==3.10.7

plotly==6.5.0

seaborn==0.13.2

Para instalar todas las dependencias, ejecuta el siguiente comando en tu entorno de desarrollo:

pip install -r requirements.txt
