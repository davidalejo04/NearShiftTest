import pandas as pd
import streamlit as st
import plotly.express as px

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Dashboard de Logística y Ventas (Olist)",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Ejecutivo: Ventas, Retención y Envíos (Olist)")
st.markdown("---")

# 1. Carga de datos desde GitHub (Raw links para los múltiples archivos parquet)
@st.cache_data
def load_data():
    base_url = "https://raw.githubusercontent.com/davidalejo04/NearShiftTest/main/data"
    
    # Lectura de los archivos principales necesarios para el modelo relacional
    orders = pd.read_parquet(f"{base_url}/olist_orders_dataset.parquet")
    order_items = pd.read_parquet(f"{base_url}/olist_order_items_dataset.parquet")
    customers = pd.read_parquet(f"{base_url}/olist_customers_dataset.parquet")
    products = pd.read_parquet(f"{base_url}/olist_products_dataset.parquet")
    
    # Cruce (Merge) de las tablas para unificar el dataset de análisis
    df = order_items.merge(orders, on="order_id", how="inner")
    df = df.merge(customers, on="customer_id", how="inner")
    df = df.merge(products, on="product_id", how="inner")
    
    return df

try:
    with st.spinner("Cargando y uniendo los conjuntos de datos desde GitHub..."):
        df = load_data()
except Exception as e:
    st.error(f"No se pudieron cargar los archivos parquet desde GitHub. Verifica las rutas o la conexión. Error: {e}")
    df = pd.DataFrame()

if not df.empty:
    # Conversión de columnas de fecha/timestamp
    date_columns = [col for col in df.columns if 'date' in col or 'timestamp' in col]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # ----------------------------------------------------
    # FILTROS LATERALES
    # ----------------------------------------------------
    st.sidebar.header("Filtros Globales")
    selected_state = st.sidebar.multiselect(
        "Seleccionar Estado(s) del Cliente", 
        options=sorted(df['customer_state'].unique()) if 'customer_state' in df.columns else []
    )
    
    if selected_state:
        df_filtered = df[df['customer_state'].isin(selected_state)]
    else:
        df_filtered = df.copy()

    # ----------------------------------------------------
    # CÁLCULOS DE VARIABLES GENERALES (Fila Superior)
    # ----------------------------------------------------
    if 'price' in df_filtered.columns and 'order_purchase_timestamp' in df_filtered.columns:
        df_filtered['order_month'] = df_filtered['order_purchase_timestamp'].dt.to_period('M').astype(str)
        total_revenue = df_filtered['price'].sum()
    else:
        total_revenue = 0

    avg_purchase_price = df_filtered['price'].mean() if 'price' in df_filtered.columns else 0

    if {'order_delivered_customer_date', 'order_estimated_delivery_date'}.issubset(df_filtered.columns):
        valid_delivery = df_filtered.dropna(subset=['order_delivered_customer_date', 'order_estimated_delivery_date'])
        if len(valid_delivery) > 0:
            on_time_shipping = (valid_delivery['order_delivered_customer_date'] <= valid_delivery['order_estimated_delivery_date']).mean() * 100
        else:
            on_time_shipping = 0.0
    else:
        on_time_shipping = 0.0

    if {'customer_unique_id', 'order_purchase_timestamp'}.issubset(df_filtered.columns):
        customer_orders = df_filtered.groupby('customer_unique_id')['order_purchase_timestamp'].agg(['min', 'max']).reset_index()
        customer_orders['days_diff'] = (customer_orders['max'] - customer_orders['min']).dt.days
        
        total_customers = len(customer_orders)
        if total_customers > 0:
            repeat_3m = (customer_orders['days_diff'] >= 90).mean() * 100
            repeat_7m = (customer_orders['days_diff'] >= 210).mean() * 100
            repeat_2y = (customer_orders['days_diff'] >= 730).mean() * 100
        else:
            repeat_3m = repeat_7m = repeat_2y = 0.0
    else:
        repeat_3m = repeat_7m = repeat_2y = 0.0

    # ----------------------------------------------------
    # RENDERIZADO DE LA FILA SUPERIOR (Métricas Clave)
    # ----------------------------------------------------
    st.subheader("📌 Indicadores Clave Generales")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Revenue Total", f"${total_revenue:,.2f}")
    col2.metric("Precio Promedio Compra", f"${avg_purchase_price:,.2f}")
    col3.metric("Cumplimiento Envíos", f"{on_time_shipping:.1f}%")
    col4.metric("Clientes Retención (3 Meses)", f"{repeat_3m:.1f}%")

    col5, col6, col7 = st.columns(3)
    col5.metric("Retención (7 Meses)", f"{repeat_7m:.1f}%")
    col6.metric("Retención (2 Años)", f"{repeat_2y:.1f}%")

    if 'product_id' in df_filtered.columns:
        top_products = df_filtered['product_id'].value_counts().head(5).reset_index()
        top_products.columns = ['Product ID', 'Cantidad Comprada']
        col7.write("🏆 **Top 5 Productos más Comprados**")
        col7.dataframe(top_products, hide_index=True)

    st.markdown("---")

    # ----------------------------------------------------
    # MAPA CLOROPLETH SEGÚN VARIABLE SELECCIONADA
    # ----------------------------------------------------
    st.subheader("🗺️ Mapa Geográfico por Estado (Brasil - Olist)")
    
    metric_choice = st.selectbox(
        "Seleccione la variable a plasmar en el mapa:",
        options=[
            "Revenue Total por Estado", 
            "Precio Promedio de Compra por Estado", 
            "% Cumplimiento de Envíos por Estado",
            "Tasa de Recompra (3 Meses) por Estado"
        ]
    )

    if 'customer_state' in df_filtered.columns:
        if metric_choice == "Revenue Total por Estado":
            map_df = df_filtered.groupby('customer_state')['price'].sum().reset_index()
            map_df.columns = ['state', 'value']
            color_label = "Revenue ($)"
        elif metric_choice == "Precio Promedio de Compra por Estado":
            map_df = df_filtered.groupby('customer_state')['price'].mean().reset_index()
            map_df.columns = ['state', 'value']
            color_label = "Precio Promedio ($)"
        elif metric_choice == "% Cumplimiento de Envíos por Estado":
            df_filtered['on_time'] = df_filtered['order_delivered_customer_date'] <= df_filtered['order_estimated_delivery_date']
            map_df = df_filtered.groupby('customer_state')['on_time'].mean().reset_index()
            map_df['value'] = map_df['on_time'] * 100
            map_df = map_df[['customer_state', 'value']]
            map_df.columns = ['state', 'value']
            color_label = "% Cumplimiento"
        else:
            cust_st = df_filtered.groupby(['customer_unique_id', 'customer_state'])['order_purchase_timestamp'].agg(['min', 'max']).reset_index()
            cust_st['is_repeat_3m'] = (cust_st['max'] - cust_st['min']).dt.days >= 90
            map_df = cust_st.groupby('customer_state')['is_repeat_3m'].mean().reset_index()
            map_df['value'] = map_df['is_repeat_3m'] * 100
            map_df = map_df[['customer_state', 'value']]
            map_df.columns = ['state', 'value']
            color_label = "% Recompra (3M)"

        # Gráfico Choropleth adaptado para estados de Brasil (usando scope de Sudamérica)
        fig = px.choropleth(
            map_df,
            locations='state',
            locationmode="USA-states", # Nota: Para estados brasileños en Plotly se suele mapear mediante GeoJSON o abreviaturas estándar en scope latinoamericano.
            color='value',
            scope="south america", 
            color_continuous_scale="Blues",
            labels={'value': color_label}
        )
        
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No se encontró la columna 'customer_state' en el conjunto de datos.")

else:
    st.info("No se pudieron procesar los datos. Por favor, asegúrate de que los archivos estén disponibles en la rama principal de GitHub.")
