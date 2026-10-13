import json
import os
import urllib.request
 
import pandas as pd
import plotly.express as px
import streamlit as st
 
# ----------------------------------------------------------------------------
# Configuración de la página
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard de Logística y Ventas (Olist)",
    page_icon="📊",
    layout="wide",
)
 
st.title("📊 Dashboard Ejecutivo: Ventas, Retención y Envíos (Olist)")
st.markdown("---")
 
GITHUB_BASE_URL = "https://raw.githubusercontent.com/davidalejo04/NearShiftTest/main/data"
GEOJSON_URL = (
    "https://raw.githubusercontent.com/codeforamerica/click_that_hood/"
    "master/public/data/brazil-states.geojson"
)
# Si existe una carpeta local "data/" (o la variable DATA_DIR) se usa en lugar de GitHub
LOCAL_DIR = os.environ.get("DATA_DIR", "data")
 
# Estados de pedido que NO representan una venta real
EXCLUDED_STATUS = ["canceled", "unavailable"]
 
# Ventanas de recompra (en días) desde la primera compra del cliente
WINDOWS = {"3 meses": 90, "7 meses": 210, "2 años": 730}
 
 
# ----------------------------------------------------------------------------
# Carga y preparación de datos
# ----------------------------------------------------------------------------
def _read(name: str) -> pd.DataFrame:
    local_path = os.path.join(LOCAL_DIR, f"{name}.parquet")
    if os.path.exists(local_path):
        return pd.read_parquet(local_path)
    return pd.read_parquet(f"{GITHUB_BASE_URL}/{name}.parquet")
 
 
@st.cache_data(show_spinner=False)
def load_data():
    """Devuelve (orders, items).
 
    orders -> una fila por pedido (con cliente, estado y valor pagado).
    items  -> una fila por ítem de pedido (con categoría del producto).
 
    OJO: olist_order_items NO trae precio. El valor monetario del pedido
    sale de olist_order_payments.payment_value (1 fila por pedido), por eso
    el análisis de ventas se hace a nivel de PEDIDO para no duplicar ingresos.
    """
    orders = _read("olist_orders_dataset")
    customers = _read("olist_customers_dataset")
    payments = _read("olist_order_payments_dataset")
    items = _read("olist_order_items_dataset")
    products = _read("olist_products_dataset")
 
    # Fechas
    for col in [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
 
    # Pedidos válidos (sin cancelados / no disponibles)
    orders = orders[~orders["order_status"].isin(EXCLUDED_STATUS)]
 
    # Pagos: por seguridad se agrupa por pedido (evita duplicar ingresos)
    pay = payments.groupby("order_id", as_index=False)["payment_value"].sum()
 
    orders = (
        orders.merge(customers[["customer_id", "customer_unique_id", "customer_state"]],
                     on="customer_id", how="inner")
        .merge(pay, on="order_id", how="left")
    )
    orders["payment_value"] = orders["payment_value"].fillna(0.0)
    orders["order_month"] = orders["order_purchase_timestamp"].dt.to_period("M").astype(str)
 
    # Cumplimiento: se compara por FECHA (la fecha estimada viene sin hora;
    # comparar con el timestamp completo marcaría como tarde entregas del mismo día)
    delivered = orders["order_delivered_customer_date"]
    orders["is_delivered"] = delivered.notna()
    orders["on_time"] = (
        delivered.dt.normalize() <= orders["order_estimated_delivery_date"].dt.normalize()
    ).where(orders["is_delivered"])
 
    # Ítems con categoría de producto, solo de pedidos válidos
    items = items[items["order_id"].isin(orders["order_id"])]
    items = items.merge(products, on="product_id", how="left")
    items["product_category_name"] = items["product_category_name"].fillna("sin categoría")
    items = items.merge(orders[["order_id", "customer_state"]], on="order_id", how="left")
 
    return orders.reset_index(drop=True), items.reset_index(drop=True)
 
 
@st.cache_data(show_spinner=False)
def build_customer_table(orders: pd.DataFrame) -> pd.DataFrame:
    """Una fila por cliente único: primera compra, estado de la primera compra
    y días hasta su segunda compra (NaN si nunca recompró)."""
    o = orders.sort_values("order_purchase_timestamp")
    o = o.assign(rank=o.groupby("customer_unique_id").cumcount())
 
    first = o[o["rank"] == 0].set_index("customer_unique_id")
    second = o[o["rank"] == 1].set_index("customer_unique_id")
 
    cust = pd.DataFrame({
        "first_purchase": first["order_purchase_timestamp"],
        "state": first["customer_state"],
    })
    cust["second_purchase"] = second["order_purchase_timestamp"]
    cust["days_to_repeat"] = (cust["second_purchase"] - cust["first_purchase"]).dt.days
    return cust.reset_index()
 
 
@st.cache_data(show_spinner=False)
def load_geojson():
    with urllib.request.urlopen(GEOJSON_URL, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))
 
 
# ----------------------------------------------------------------------------
# Funciones de cálculo
# ----------------------------------------------------------------------------
def repeat_rate(cust: pd.DataFrame, days: int, max_date: pd.Timestamp):
    """% de clientes que volvieron a comprar dentro de `days` días de su primera
    compra. Se cuentan solo clientes cuya ventana ya pudo completarse (primera
    compra al menos `days` días antes del último dato); de lo contrario la tasa
    quedaría artificialmente baja.
    Si NINGÚN cliente tiene la ventana completa (p. ej. 2 años en un dataset que
    solo cubre ~2 años), se usa a todos los clientes como aproximación.
    Devuelve (tasa %, clientes en la base, ventana_completa)."""
    eligible = cust[cust["first_purchase"] <= max_date - pd.Timedelta(days=days)]
    complete = not eligible.empty
    base = eligible if complete else cust
    if base.empty:
        return 0.0, 0, complete
    repeated = base["days_to_repeat"] <= days  # NaN -> False
    return repeated.mean() * 100, len(base), complete
 
 
def on_time_rate(orders: pd.DataFrame) -> float:
    delivered = orders[orders["is_delivered"]]
    return delivered["on_time"].astype(float).mean() * 100 if len(delivered) else 0.0
 
 
def metric_by_state(orders: pd.DataFrame, cust: pd.DataFrame, metric: str, max_date) -> pd.DataFrame:
    if metric == "Revenue total":
        s = orders.groupby("customer_state")["payment_value"].sum()
    elif metric == "Precio promedio de compra":
        s = orders.groupby("customer_state")["payment_value"].mean()
    elif metric == "% Cumplimiento de envíos":
        d = orders[orders["is_delivered"]]
        s = d.groupby("customer_state")["on_time"].apply(lambda x: x.astype(float).mean() * 100)
    else:  # Recompra por ventana
        label = metric.replace("Tasa de recompra (", "").replace(")", "")
        days = WINDOWS[label]
        s = cust.groupby("state").apply(
            lambda g: repeat_rate(g, days, max_date)[0], include_groups=False
        )
    out = s.rename("value").reset_index()
    out.columns = ["state", "value"]
    return out
 
 
# ----------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------
try:
    with st.spinner("Cargando y uniendo los conjuntos de datos..."):
        orders_all, items_all = load_data()
        cust_all = build_customer_table(orders_all)
except Exception as e:
    st.error(f"No se pudieron cargar los archivos parquet. Verifica las rutas o la conexión. Error: {e}")
    st.stop()
 
max_date = orders_all["order_purchase_timestamp"].max()
 
# ----------------------------------------------------------------------------
# Filtros laterales
# ----------------------------------------------------------------------------
st.sidebar.header("Filtros Globales")
all_states = sorted(orders_all["customer_state"].dropna().unique())
selected_states = st.sidebar.multiselect("Estado(s) del cliente", options=all_states)
 
if selected_states:
    orders = orders_all[orders_all["customer_state"].isin(selected_states)]
    items = items_all[items_all["customer_state"].isin(selected_states)]
    cust = cust_all[cust_all["state"].isin(selected_states)]
else:
    orders, items, cust = orders_all, items_all, cust_all
 
if orders.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()
 
st.sidebar.caption(
    f"Pedidos analizados: {len(orders):,}\n\n"
    f"Se excluyen pedidos {', '.join(EXCLUDED_STATUS)}.\n\n"
    f"Datos hasta: {max_date:%Y-%m-%d}"
)
 
# ----------------------------------------------------------------------------
# Fila superior: indicadores generales
# ----------------------------------------------------------------------------
st.subheader("📌 Indicadores Clave Generales")
 
total_revenue = orders["payment_value"].sum()
avg_purchase = orders["payment_value"].mean()
on_time = on_time_rate(orders)
rates = {label: repeat_rate(cust, days, max_date) for label, days in WINDOWS.items()}
 
c1, c2, c3 = st.columns(3)
c1.metric("Revenue Total", f"${total_revenue:,.2f}")
c2.metric("Precio Promedio de Compra (por pedido)", f"${avg_purchase:,.2f}")
c3.metric("Cumplimiento de Envíos", f"{on_time:.1f}%")
 
c4, c5, c6, c7 = st.columns([1, 1, 1, 2])
for col, (label, (rate, n, complete)) in zip([c4, c5, c6], rates.items()):
    note = (f"Base: {n:,} clientes con ventana completa" if complete else
            f"Base: {n:,} clientes. Aproximación: el dataset no cubre {label} completos "
            "de seguimiento para ningún cliente.")
    col.metric(f"Recompra ({label})", f"{rate:.1f}%", help=note)
 
top_products = (
    items.groupby(["product_id", "product_category_name"])
    .size()
    .sort_values(ascending=False)
    .head(5)
    .reset_index(name="Cantidad Comprada")
    .rename(columns={"product_id": "Producto", "product_category_name": "Categoría"})
)
c7.write("🏆 **Top 5 Productos más Comprados**")
c7.dataframe(top_products, hide_index=True)
 
# Revenue por mes
st.markdown("#### 📅 Revenue Total por Mes")
monthly = orders.groupby("order_month", as_index=False)["payment_value"].sum()
fig_month = px.bar(
    monthly, x="order_month", y="payment_value",
    labels={"order_month": "Mes", "payment_value": "Revenue ($)"},
)
fig_month.update_layout(margin={"r": 0, "t": 10, "l": 0, "b": 0}, xaxis_tickangle=-45)
st.plotly_chart(fig_month)
 
st.markdown("---")
 
# ----------------------------------------------------------------------------
# Mapa coroplético de Brasil
# ----------------------------------------------------------------------------
st.subheader("🗺️ Mapa Geográfico por Estado (Brasil)")
 
metric_options = {
    "Revenue total": "Revenue ($)",
    "Precio promedio de compra": "Precio promedio ($)",
    "% Cumplimiento de envíos": "% Cumplimiento",
    "Tasa de recompra (3 meses)": "% Recompra 3M",
    "Tasa de recompra (7 meses)": "% Recompra 7M",
    "Tasa de recompra (2 años)": "% Recompra 2A",
}
metric_choice = st.selectbox("Seleccione la variable a mostrar en el mapa:", list(metric_options))
 
map_df = metric_by_state(orders, cust, metric_choice, max_date)
 
try:
    geojson = load_geojson()
    fig_map = px.choropleth(
        map_df,
        geojson=geojson,
        locations="state",
        featureidkey="properties.sigla",
        color="value",
        color_continuous_scale="Blues",
        labels={"value": metric_options[metric_choice], "state": "Estado"},
        hover_name="state",
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=600)
    st.plotly_chart(fig_map)
except Exception as e:
    st.error(f"No se pudo cargar el GeoJSON de Brasil: {e}")
 
with st.expander("Ver datos del mapa"):
    st.dataframe(map_df.sort_values("value", ascending=False), hide_index=True)
 
st.caption(
    "Notas: el revenue se calcula con el valor pagado por pedido (payment_value). "
    "La recompra mide el % de clientes únicos con una segunda compra dentro de 90 / 210 / 730 días "
    "desde su primera compra, considerando solo clientes cuya ventana ya pudo completarse."
)
