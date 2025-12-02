# modelo_precios.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap
import category_encoders as ce

# =========================
# RUTAS Y CONSTANTES
# =========================

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "retail_price.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "xgb_model.joblib"

TARGET = "qty"

# =========================
# CARGA DE DATOS Y PREPROC
# =========================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"No se encontró el dataset en {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(f"El dataset debe contener la columna de target '{TARGET}'.")

if "product_category_name" not in df.columns:
    raise ValueError("El dataset debe contener la columna 'product_category_name'.")

# Lista de categorías para UI (Streamlit & otros)
CATEGORIES = sorted(df["product_category_name"].dropna().unique().tolist())

# Copia de trabajo para preprocesar
df_proc = df.copy()

# Columnas numéricas y categóricas
NUM_COLS = df_proc.select_dtypes(include=[np.number]).columns.tolist()
CAT_COLS = df_proc.select_dtypes(include=["object", "category"]).columns.tolist()

# Relleno de nulos sencillo en df_proc (como en el notebook)
for c in NUM_COLS:
    if df_proc[c].isnull().any():
        df_proc[c].fillna(df_proc[c].median(), inplace=True)

for c in CAT_COLS:
    if df_proc[c].isnull().any():
        df_proc[c].fillna("__missing__", inplace=True)

# Feature engineering básico (igual lógica que el notebook)
if "unit_price" in df_proc.columns and "qty" in df_proc.columns:
    df_proc["revenue"] = df_proc["unit_price"] * df_proc["qty"]

if "product_weight_g" in df_proc.columns:
    df_proc["weight_kg"] = df_proc["product_weight_g"] / 1000.0

# Eliminar filas sin target
df_proc = df_proc[~df_proc[TARGET].isnull()].reset_index(drop=True)

# Columnas que NO usaremos como features (ajustable si tu notebook excluye otras)
EXCLUDE = ["month_year"] if "month_year" in df_proc.columns else []
FEATURE_CANDIDATES = [c for c in df_proc.columns if c not in EXCLUDE + [TARGET]]

# Columnas categóricas que sí son features
CATEGORICAL_COLS = [c for c in CAT_COLS if c in FEATURE_CANDIDATES]

# Medianas numéricas, para imputar en filas nuevas
NUMERIC_MEDIANS = df_proc[NUM_COLS].median()

# =========================
# TARGET ENCODER
# =========================

if len(CATEGORICAL_COLS) > 0:
    te = ce.TargetEncoder(cols=CATEGORICAL_COLS, smoothing=0.3)
    te.fit(df_proc[CATEGORICAL_COLS], df_proc[TARGET])
else:
    te = None

# =========================
# CARGA DEL MODELO
# =========================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el modelo entrenado en {MODEL_PATH}. "
        f"Asegúrate de haber guardado 'rf_model.joblib' en la carpeta 'models'."
    )

MODEL = joblib.load(MODEL_PATH)

# Lista final de features según el modelo entrenado
if hasattr(MODEL, "feature_names_in_"):
    FEATURES = list(MODEL.feature_names_in_)
else:
    FEATURES = FEATURE_CANDIDATES

# =========================
# EXPLAINER SHAP (opcional)
# =========================

try:
    EXPLAINER = shap.Explainer(MODEL)
except Exception:
    EXPLAINER = None


# ==============================================
# FUNCIÓN AUXILIAR: PREPARAR FILA PARA EL MODELO
# ==============================================

def _prepare_row_for_model(row_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe un DataFrame de 1 fila con columnas "crudas" del dataset original
    y devuelve un DataFrame X listo para ser pasado a MODEL.predict().
    """
    row = row_raw.copy()

    # Feature engineering igual que en el preprocesamiento
    if "revenue" in FEATURES and {"unit_price", "qty"}.issubset(row.columns):
        row["revenue"] = row["unit_price"] * row["qty"]

    if "weight_kg" in FEATURES and "product_weight_g" in row.columns:
        row["weight_kg"] = row["product_weight_g"] / 1000.0

    # Relleno de nulos para las columnas numéricas
    for c in NUM_COLS:
        if c in row.columns:
            row[c] = row[c].fillna(NUMERIC_MEDIANS.get(c, 0.0))

    # Relleno para categóricas
    for c in CATEGORICAL_COLS:
        if c in row.columns:
            row[c] = row[c].fillna("__missing__")

    # TargetEncoder sobre categóricas
    row_enc = row.copy()
    if te is not None and len(CATEGORICAL_COLS) > 0:
        # TargetEncoder espera DF, no serie
        row_enc[CATEGORICAL_COLS] = te.transform(row_enc[CATEGORICAL_COLS])

    # Asegurar que las columnas de FEATURES existan y en el orden correcto
    X = pd.DataFrame(columns=FEATURES)
    for f in FEATURES:
        if f in row_enc.columns:
            X[f] = row_enc[f].values
        else:
            # Si el modelo espera una columna que no está en la fila
            X[f] = [0.0]

    return X


# ==============================================
# FUNCIÓN: SIMULACIÓN DE PRECIOS
# ==============================================

def simulate_price_recommendation(
    model,
    row_original: pd.DataFrame,
    price_col: str = "unit_price",
    price_grid=None,
):
    """
    Simula una grilla de precios alrededor del precio ingresado
    y devuelve:
      - sim_df: DataFrame con columnas [price, predicted_qty, predicted_revenue]
      - best_price: precio que maximiza el revenue
    """
    base_price = float(row_original[price_col].values[0])

    if base_price <= 0:
        base_price = 1.0

    if price_grid is None:
        low = base_price * 0.8
        high = base_price * 1.2
        price_grid = np.linspace(low, high, 21)

    rows = []
    for p in price_grid:
        r_raw = row_original.copy()
        r_raw[price_col] = p

        Xp = _prepare_row_for_model(r_raw)
        qty_pred = float(model.predict(Xp)[0])
        revenue_pred = qty_pred * p

        rows.append(
            {
                "price": float(p),
                "predicted_qty": qty_pred,
                "predicted_revenue": revenue_pred,
            }
        )

    sim_df = pd.DataFrame(rows)
    best_row = sim_df.loc[sim_df["predicted_revenue"].idxmax()]
    return sim_df, float(best_row["price"]) # type: ignore


# ==========================
# RECOMMENDACIÓN DE PRECIO
# ==========================
FEATURES = FEATURE_CANDIDATES

def recommend_price_and_explain(
    row_original, 
    model=MODEL, 
    encoder=te, 
    features=FEATURES, 
    price_col='unit_price'
):
    """
    Recibe un DataFrame de 1 fila con las columnas originales
    y devuelve recomendación de precio + explicación.
    Funciona aunque falten columnas.
    """

    # Copia segura
    row = row_original.copy()

    # --- RECREAR FEATURES ENGINEERED EXACTAMENTE COMO EL PREPROCESSING ---
    # Revenue
    if 'unit_price' in row.columns and 'qty' in row.columns and 'revenue' in features:
        row['revenue'] = row['unit_price'] * row['qty']

    # Peso en kg
    if 'product_weight_g' in row.columns and 'weight_kg' in features:
        row['weight_kg'] = row['product_weight_g'] / 1000.0

    # --- RELLENAR COLUMNAS QUE FALTEN PARA EL MODELO ---
    for c in features:
        if c not in row.columns:
            row[c] = np.nan

    # --- RELLENO DE NULOS (MISMA LÓGICA DEL PREPROC) ---
    for c in NUM_COLS:
        if c in row.columns:
            row[c] = row[c].fillna(df[c].median())

    for c in CATEGORICAL_COLS:
        if c in row.columns:
            row[c] = row[c].fillna('__missing__')

    # --- ENCODING CATEGÓRICO ---
    if len(CATEGORICAL_COLS) > 0:
        try:
            row[CATEGORICAL_COLS] = encoder.transform(row[CATEGORICAL_COLS]) # type: ignore
        except:
            pass

    # --- PREDICCIÓN A PRECIO ACTUAL ---
    Xrow = row[features]
    current_qty = float(model.predict(Xrow)[0])
    current_price = float(row[price_col].values[0])
    current_rev = current_qty * current_price

    # --- SIMULACIÓN DE PRECIOS ---
    sim_df, best_price = simulate_price_recommendation(
        model, row,
        price_col=price_col,
    )

    best_row = sim_df.loc[sim_df['price'] == best_price].iloc[0]

    # --- SHAP LOCAL EXPLANATION ---
    try:
        x_enc = Xrow.fillna(0)
        if EXPLAINER is not None:
            shap_local = EXPLAINER(x_enc)
            shap_vals = shap_local.values[0]
            idx_top = np.argsort(np.abs(shap_vals))[::-1][:5]
            top_features = [(Xrow.columns[i], float(shap_vals[i])) for i in idx_top]
        else:
            top_features = []
    except:
        top_features = []

    # --- ELASTICIDAD --- (respuesta qty a +/-10% cambio en precio)
    price_up = current_price * 1.1
    price_down = current_price * 0.9

    test_up = row.copy()
    test_down = row.copy()
    test_up[price_col] = price_up
    test_down[price_col] = price_down

    if len(CATEGORICAL_COLS) > 0:
        try:
            if encoder is not None:
                test_up[CATEGORICAL_COLS] = encoder.transform(test_up[CATEGORICAL_COLS])
                test_down[CATEGORICAL_COLS] = encoder.transform(test_down[CATEGORICAL_COLS])
        except:
            pass

    qty_up = float(model.predict(test_up[features])[0])
    qty_down = float(model.predict(test_down[features])[0])

    pct_qty = (qty_down - qty_up) / ((qty_down + qty_up) / 2 + 1e-9)
    pct_price = (price_down - price_up) / ((price_down + price_up) / 2 + 1e-9)
    elasticity = pct_qty / (pct_price + 1e-9)

    # --- SALIDA FINAL ---
    return {
        'current_qty': current_qty,
        'current_revenue': current_rev,
        'recommended_price': best_price,
        'recommended_pred_qty': best_row['predicted_qty'],
        'recommended_pred_revenue': best_row['predicted_revenue'],
        'top_shap_features': top_features,
        'elasticity_approx': elasticity,
        'simulation_table': sim_df
    }

# =========================
# CARGA DE DATOS Y PREPROC
# =========================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"No se encontró el dataset en {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(f"El dataset debe contener la columna de target '{TARGET}'.")

if "product_category_name" not in df.columns:
    raise ValueError("El dataset debe contener la columna 'product_category_name'.")

CATEGORIES = sorted(df["product_category_name"].dropna().unique().tolist())



# ======================================================
# JUSTIFICACIÓN COMPLETA
# ======================================================

def build_justification(result, category: str) -> str:
    price = result["recommended_price"]
    qty = result["recommended_pred_qty"]
    revenue = result["recommended_pred_revenue"]
    elasticity = result["elasticity_approx"]
    shap = result["top_shap_features"]

    if elasticity < -1:
        e_text = "muy sensible al precio"
    elif elasticity < -0.5:
        e_text = "moderadamente sensible"
    elif elasticity < -0.1:
        e_text = "poco sensible"
    else:
        e_text = "prácticamente inelástica"

    # -----------------------------------------------------------
    # INTERPRETACIÓN PERSONALIZADA DE LA DEMANDA ESTIMADA
    # -----------------------------------------------------------
    if qty < 5:
        dem_msg = (
            "La demanda estimada es muy baja. Esto suele indicar que el producto es especialmente sensible al precio "
            "o que el mercado para esta categoría tiene poco movimiento en este periodo."
        )
    elif qty < 20:
        dem_msg = (
            "La demanda esperada es moderada. Representa un comportamiento estable donde el precio recomendado mantiene "
            "un equilibrio razonable entre ventas y ganancia."
        )
    else:
        dem_msg = (
            "La demanda proyectada es alta, lo cual es un excelente indicador. El precio recomendado está en un punto "
            "donde el volumen de ventas puede ser fuerte sin sacrificar rentabilidad."
        )

    # -----------------------------------------------------------
    # INTERPRETACIÓN PERSONALIZADA DE LA ELASTICIDAD
    # -----------------------------------------------------------
    if elasticity < -1.2:
        ela_msg = (
            "Este producto es altamente elástico. Pequeños cambios en el precio generan variaciones fuertes en la demanda. "
            "Los consumidores reaccionan rápidamente ante cualquier ajuste."
        )
    elif elasticity < -0.6:
        ela_msg = (
            "El producto muestra elasticidad considerable. El precio influye de forma notable en la decisión de compra, "
            "pero sin llegar a ser extremo."
        )
    elif elasticity < -0.2:
        ela_msg = (
            "El producto tiene elasticidad moderada. El precio afecta a la demanda, aunque los cambios no son tan drásticos."
        )
    else:
        ela_msg = (
            "El producto es prácticamente inelástico. Los consumidores mantienen su comportamiento aunque el precio varíe."
        )

    # ===========================================================
    # INTERPRETACIÓN AVANZADA Y PERSONALIZADA DE SHAP
    # ===========================================================
    if not shap:
        shap_interpretation = (
            "\n\nNo se pudo generar una interpretación detallada de los factores SHAP para este producto. "
            "Esto no afecta la recomendación de precio, pero limita el nivel de explicación disponible.\n"
        )
    else:
        shap_interpretation = (
            "\n\n         "
            "Interpretación avanzada de los Factores SHAP\n\n"
        )

        # Detectamos máximo absoluto para identificar el más fuerte
        max_abs = max(abs(v) for _, v in shap)

        for feature, value in shap:
            abs_v = abs(value)

            # --------------- NIVEL DE IMPACTO ---------------
            if abs_v == max_abs:
                fuerza = "Este es el impacto más fuerte de todos"
            elif abs_v > 7:
                fuerza = "Su efecto es bastante grande"
            elif abs_v > 3:
                fuerza = "Tiene un impacto moderado pero importante"
            elif abs_v > 1:
                fuerza = "Su influencia es perceptible y relevante"
            elif abs_v > 0.3:
                fuerza = "Su efecto es ligero"
            else:
                fuerza = "Su efecto es muy pequeño"

            # --------------- TIPO DE IMPACTO (positivo/negativo) ---------------
            if value < -7:
                tono = "muy negativo"
                explicacion = (
                    "Históricamente, cuando esta variable aumenta, la demanda suele caer fuertemente. "
                    "El modelo interpreta que este valor va en contra del precio que funciona bien."
                )
                implicacion = (
                    "Empuja la predicción hacia reducir la demanda y sugiere que el precio ingresado "
                    "está por encima de lo óptimo."
                )

            elif value < -3:
                tono = "negativo"
                explicacion = (
                    "Este valor indica que la variable tiende a disminuir la demanda en este contexto. "
                    "No es un impacto extremo, pero sí consistente con patrones históricos."
                )
                implicacion = (
                    "El modelo compensa este efecto ajustando el precio recomendado un poco a la baja."
                )

            elif value < -0.8:
                tono = "ligeramente negativo"
                explicacion = (
                    "Sugiere que la situación actual de esta variable no es favorable para sostener "
                    "precios más altos, pero tampoco es un efecto severo."
                )
                implicacion = (
                    "La recomendación final atenúa el precio sugerido para equilibrar este pequeño freno en la demanda."
                )

            elif value < -0.2:
                tono = "muy suave negativo"
                explicacion = (
                    "Indica un pequeño sesgo hacia la reducción de demanda, pero el efecto es débil."
                )
                implicacion = (
                    "Su participación en el precio recomendado es mínima."
                )

            elif value < 0.2:
                tono = "casi neutro"
                explicacion = (
                    "Este factor prácticamente no cambia la demanda. "
                    "En los datos históricos su influencia es marginal."
                )
                implicacion = (
                    "No afecta de manera significativa la decisión del precio recomendado."
                )

            elif value < 1.2:
                tono = "positivo"
                explicacion = (
                    "Este valor indica que la variable ayuda ligeramente a aumentar la demanda. "
                    "Es un empuje favorable pero no dominante."
                )
                implicacion = (
                    "Contribuye a que el modelo no recomiende bajar demasiado el precio."
                )

            elif value < 4:
                tono = "bastante positivo"
                explicacion = (
                    "Refleja que esta variable fortalece la demanda de forma clara. "
                    "Históricamente, cuando este valor crece, el mercado responde bien."
                )
                implicacion = (
                    "Esto actúa como contrapeso ante otros factores negativos."
                )

            else:
                tono = "muy positivo"
                explicacion = (
                    "Este es un impulso grande hacia arriba en la demanda. "
                    "En los datos, esta variable está fuertemente asociada con mejores ventas."
                )
                implicacion = (
                    "Ayuda notablemente a sostener el precio sugerido o incluso subirlo ligeramente."
                )

            # --------------- MENSAJE FINAL ---------------
            shap_interpretation += (
                f"\n• {feature}: {value:.4f} ({tono})\n"
                f"\n{fuerza}.\n\n"
                f"- Interpretación: {explicacion}\n\n"
                f"- ¿Qué implica?: {implicacion}\n\n\n"
            )

    justification = f"""
Esta recomendación se construye a partir del análisis integral del producto dentro
de su categoría "{category}", considerando tanto su comportamiento histórico como la
respuesta esperada del mercado frente a variaciones de precio.

                                                                       
        Precio recomendado


El sistema sugiere fijar un precio de ${price:.2f}. Este valor equilibra el nivel de demanda esperada con la 
maximización del ingreso proyectado. 

El precio no se elige solo porque sea “alto” o “bajo”, sino porque representa el punto donde la relación 
entre volumen vendido y ganancia estimada es más favorable para este producto.

                                                                   
        Demanda esperada y su impacto


• Demanda estimada: {qty:.2f} unidades.

{dem_msg} Se proyecta un ingreso aproximado de ${revenue:.2f}. 

                                                               
        Elasticidad del producto

• Elasticidad del producto: {elasticity:.4f}

{ela_msg}

Este dato es importante porque explica qué tan “delicado” es el mercado con respecto a los cambios de precio.
{shap_interpretation}
""".strip()

    return justification.strip()
    