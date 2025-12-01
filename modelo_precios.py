# modelo_precios.py
# Lógica del Sistema Inteligente de Recomendación de Precios
# - Carga retail_price.csv
# - Reconstruye el TargetEncoder para categorías
# - Carga el modelo entrenado (rf_model.joblib)
# - Prepara fila, simula escenarios y devuelve recomendación + explicación

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
MODEL_PATH = MODEL_DIR / "rf_model.joblib"

TARGET = "qty"

# =========================
# CARGA DE DATOS Y PREPROC
# =========================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"No se encontró el dataset en {DATA_PATH}")

DATA_PATH = BASE_DIR / "retail_price.csv"

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


# ==============================================
# FUNCIÓN PRINCIPAL: RECOMENDACIÓN + EXPLICACIÓN
# ==============================================

def recommend_price_and_explain(
    row_original: pd.DataFrame,
    model=MODEL,
    price_col: str = "unit_price",
):
    """
    Recibe un DataFrame de 1 fila con las columnas originales de retail_price.csv
    y devuelve:
      - precio recomendado
      - demanda e ingreso esperados
      - elasticidad aproximada
      - top factores shap
      - tabla de simulación
    """

    # Predicción a precio actual
    Xrow = _prepare_row_for_model(row_original)
    current_qty = float(model.predict(Xrow)[0])
    current_price = float(row_original[price_col].values[0])
    current_rev = current_qty * current_price

    # Simulación de precios
    sim_df, best_price = simulate_price_recommendation(
        model=model,
        row_original=row_original,
        price_col=price_col,
    )
    best_row = sim_df.loc[sim_df["price"] == best_price].iloc[0]

    # SHAP local
    top_features = []
    if EXPLAINER is not None:
        try:
            shap_local = EXPLAINER(Xrow)
            shap_vals = shap_local.values[0]
            idx_top = np.argsort(np.abs(shap_vals))[::-1][:5]
            top_features = [(Xrow.columns[i], float(shap_vals[i])) for i in idx_top]
        except Exception:
            top_features = []

    # Elasticidad aproximada (cambio ±10% en el precio)
    price_up = current_price * 1.10
    price_down = current_price * 0.90

    row_up = row_original.copy()
    row_up[price_col] = price_up
    X_up = _prepare_row_for_model(row_up)
    qty_up = float(model.predict(X_up)[0])

    row_down = row_original.copy()
    row_down[price_col] = price_down
    X_down = _prepare_row_for_model(row_down)
    qty_down = float(model.predict(X_down)[0])

    pct_qty = (qty_down - qty_up) / ((qty_down + qty_up) / 2 + 1e-9)
    pct_price = (price_down - price_up) / ((price_down + price_up) / 2 + 1e-9)
    elasticity = pct_qty / (pct_price + 1e-9)

    return {
        "current_qty": current_qty,
        "current_revenue": current_rev,
        "recommended_price": best_price,
        "recommended_pred_qty": float(best_row["predicted_qty"]),
        "recommended_pred_revenue": float(best_row["predicted_revenue"]),
        "top_shap_features": top_features,
        "elasticity_approx": float(elasticity),
        "simulation_table": sim_df,
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


# ==============================================
# FUNCIÓN: JUSTIFICACIÓN EN TEXTO
# ==============================================

def build_justification(result: dict, category: str) -> str:
    price = result["recommended_price"]
    qty = result["recommended_pred_qty"]
    revenue = result["recommended_pred_revenue"]
    elasticity = result["elasticity_approx"]
    shap_list = result.get("top_shap_features", [])

    # Interpretación simple de elasticidad
    if elasticity < -1:
        e_text = "muy sensible al precio"
    elif elasticity < -0.5:
        e_text = "moderadamente sensible al precio"
    elif elasticity < -0.1:
        e_text = "poco sensible al precio"
    else:
        e_text = "prácticamente inelástica"

    # Interpretación de demanda
    if qty < 5:
        dem_msg = (
            "La demanda estimada es baja, lo que sugiere un mercado reducido o alta "
            "sensibilidad al precio en esta categoría."
        )
    elif qty < 20:
        dem_msg = (
            "La demanda esperada es moderada y refleja un equilibrio razonable entre "
            "volumen de ventas e ingreso esperado."
        )
    else:
        dem_msg = (
            "La demanda proyectada es alta, indicando que el precio recomendado coloca "
            "al producto en una zona atractiva para el mercado."
        )

    shap_lines = []
    for feature, value in shap_list:
        tono = "positivo" if value >= 0 else "negativo"
        shap_lines.append(f"• {feature}: {value:.4f} ({tono})")

    shap_text = "\n".join(shap_lines) if shap_lines else "No se pudieron calcular factores SHAP."

    texto = f"""
Esta recomendación se basa en el comportamiento histórico de la categoría **{category}** 
y en la forma en que el modelo ha aprendido la relación entre precio y demanda.

**1. Precio recomendado**

El sistema sugiere fijar un precio de **${price:.2f}**. Este valor busca maximizar el ingreso esperado
frente a otros niveles de precio simulados para este mismo producto.

**2. Demanda esperada e ingresos**

La demanda estimada a este nivel de precio es de aproximadamente **{qty:.2f} unidades**.
{dem_msg}
Con este nivel de demanda se proyecta un ingreso de **${revenue:,.2f}**.

**3. Elasticidad aproximada del producto**

La elasticidad calculada es **{elasticity:.4f}**, por lo que el producto es **{e_text}**.  
En términos prácticos, esto indica qué tan fuerte responde la demanda ante variaciones en el precio.

**4. Principales factores explicativos (tipo SHAP)**

{shap_text}
"""
    return texto.strip()
