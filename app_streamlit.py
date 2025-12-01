# app_streamlit.py
# Dashboard empresarial con Streamlit usando modelo_precios.py

import streamlit as st
import pandas as pd
import plotly.express as px

from modelo_precios import (
    df,
    CATEGORIES,
    recommend_price_and_explain,
    build_justification,
)

st.set_page_config(
    page_title="Sistema Inteligente de Precios",
    layout="wide",
    page_icon="📊",
)

st.title("📊 Sistema Inteligente de Recomendación de Precios")
st.markdown(
    """
Este panel permite analizar el efecto del precio sobre la demanda, 
estimar ingresos esperados y ver la explicación del modelo de forma visual.
"""
)

# ------------------ SIDEBAR: CONFIGURACIÓN ------------------ #

st.sidebar.header("Configuración del escenario")

selected_category = st.sidebar.selectbox(
    "Categoría de producto",
    options=CATEGORIES,
    index=0 if CATEGORIES else 0,
)

input_price = st.sidebar.number_input(
    "Precio unitario ingresado (USD)",
    min_value=0.0,
    value=39.99,
    step=0.5,
    format="%.2f",
)

run_button = st.sidebar.button("🚀 Generar recomendación")

st.sidebar.markdown(
    """
- Prueba distintos precios para ver cómo cambia:
  - la demanda estimada  
  - los ingresos proyectados  
  - la elasticidad aproximada
"""
)


def ejecutar_recomendacion_streamlit(category: str, price_input: float):
    df_cat = df[df["product_category_name"] == category]
    if df_cat.empty:
        st.error(f"No se encontraron filas para la categoría '{category}'.")
        return None

    row = df_cat.iloc[[0]].copy()
    row["unit_price"] = price_input

    result = recommend_price_and_explain(row)
    return result


# ======================================================
#                     MAIN LAYOUT
# ======================================================

if run_button:
    result = ejecutar_recomendacion_streamlit(selected_category, input_price) # type: ignore

    if result is None:
        st.stop()

    # ------- KPIs PRINCIPALES -------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="💰 Precio recomendado",
            value=f"${result['recommended_price']:.2f}",
        )

    with col2:
        st.metric(
            label="📦 Demanda esperada (unidades)",
            value=f"{result['recommended_pred_qty']:.2f}",
        )

    with col3:
        st.metric(
            label="📈 Ingreso estimado",
            value=f"${result['recommended_pred_revenue']:.2f}",
        )

    st.markdown("---")
    st.subheader("Resumen ejecutivo del escenario")

    col4, col5 = st.columns(2)

    with col4:
        st.markdown("**Detalle de la recomendación**")
        st.write(
            f"- Categoría: `{selected_category}`\n"
            f"- Precio ingresado: **${input_price:.2f}**\n"
            f"- Precio recomendado por el modelo: **${result['recommended_price']:.2f}**"
        )

    with col5:
        st.markdown("**Elasticidad aproximada**")
        st.write(f"Elasticidad ≈ **{result['elasticity_approx']:.4f}**")

        ela = result["elasticity_approx"]
        if ela < -1:
            desc = "Alta sensibilidad al precio (producto elástico)."
        elif ela < -0.5:
            desc = "Sensibilidad moderada al precio."
        elif ela < -0.1:
            desc = "Baja sensibilidad al precio."
        else:
            desc = "Demanda prácticamente inelástica."
        st.info(desc)

    # ------- EXPLICACIÓN DEL MODELO -------
    st.markdown("---")
    st.subheader("🧠 Explicación del modelo")

    tab1, tab2 = st.tabs(["Factores tipo SHAP", "Justificación completa"])

    with tab1:
        shap_list = result.get("top_shap_features", [])
        if not shap_list:
            st.write("No se pudieron recuperar factores SHAP.")
        else:
            st.write("Principales factores que influyen en la predicción:")
            shap_df = pd.DataFrame(shap_list, columns=["Feature", "SHAP value"])
            st.dataframe(shap_df, width="stretch")


    with tab2:
        just_text = build_justification(result, selected_category) # type: ignore
        st.write(just_text)

    # ------- ESCENARIOS SIMULADOS -------
    st.markdown("---")
    st.subheader("📈 Escenarios simulados de precio vs ingresos")

    sim_df = result.get("simulation_table", pd.DataFrame()).copy()
    if sim_df.empty:
        st.write("No hay tabla de simulación disponible.")
    else:
        if {"price", "predicted_qty", "predicted_revenue"}.issubset(sim_df.columns):
            plot_df = sim_df[["price", "predicted_revenue"]].rename(
                columns={
                    "price": "Precio (USD)",
                    "predicted_revenue": "Ingresos predichos (USD)",
                }
            )

            fig = px.line(
                plot_df,
                x="Precio (USD)",
                y="Ingresos predichos (USD)",
                markers=True,
                title="Curva de ingresos esperados según el modelo",
            )

            # Punto del precio recomendado
            idx_closest = (plot_df["Precio (USD)"] - result["recommended_price"]).abs().idxmin()
            y_reco = plot_df.loc[idx_closest, "Ingresos predichos (USD)"]

            fig.add_scatter(
                x=[result["recommended_price"]],
                y=[y_reco],
                mode="markers",
                marker=dict(size=10),
                name="Precio recomendado",
            )

            st.plotly_chart(fig, width="stretch")

            st.markdown("#### Tabla de escenarios simulados")
            table_df = sim_df.rename(
                columns={
                    "price": "Precio (USD)",
                    "predicted_qty": "Demanda predicha",
                    "predicted_revenue": "Ingresos predichos (USD)",
                }
            )
            st.dataframe(table_df, width="stretch")
        else:
            st.write("La estructura de `simulation_table` no es la esperada.")
            st.plotly_chart(sim_df, width="stretch")

else:
    st.info(
        "Configura una categoría y un precio en el panel lateral y luego haz clic en **'Generar recomendación'**."
    )
