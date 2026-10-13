import json
import os
import re
import unicodedata
import urllib.request
 
import pandas as pd
import plotly.express as px
import streamlit as st

from groq import Groq
import tabulate
 
# ----------------------------------------------------------------------------
# Configuración de la página (sin panel lateral)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard de Logística y Ventas (Olist)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
GITHUB_BASE_URL = "https://raw.githubusercontent.com/davidalejo04/NearShiftTest/main/data"
GEOJSON_URL = (
    "https://raw.githubusercontent.com/codeforamerica/click_that_hood/"
    "master/public/data/brazil-states.geojson"
)
# Coordenadas de municipios (los parquet no traen latitud/longitud)
MUNICIPIOS_URL = "https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/municipios.csv"
ESTADOS_URL = "https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/estados.csv"
 
# Si existe una carpeta local "data/" (o la variable DATA_DIR) se usa en lugar de GitHub
LOCAL_DIR = os.environ.get("DATA_DIR", "data")
 
EXCLUDED_STATUS = ["canceled", "unavailable"]          # no son ventas reales
WINDOWS = {"3 meses": 90, "7 meses": 210, "2 años": 730}  # ventanas de recompra (días)
 
MAP_STYLE = "carto-positron"   # mapa base limpio, sin necesidad de token
COLOR_SCALE = "YlGnBu"

API_K = st.secrets["API_K"]
 
METRICS = {
    "Revenue total":              dict(label="Revenue ($)",          prefix="$", suffix="",  fmt=",.0f"),
    "Precio promedio de compra":  dict(label="Precio promedio ($)",  prefix="$", suffix="",  fmt=",.2f"),
    "% Cumplimiento de envíos":   dict(label="% Cumplimiento",       prefix="",  suffix="%", fmt=".1f"),
    "Tasa de recompra (3 meses)": dict(label="% Recompra 3M",        prefix="",  suffix="%", fmt=".1f", days=90),
    "Tasa de recompra (7 meses)": dict(label="% Recompra 7M",        prefix="",  suffix="%", fmt=".1f", days=210),
    "Tasa de recompra (2 años)":  dict(label="% Recompra 2A",        prefix="",  suffix="%", fmt=".1f", days=730),
}
 
# Columnas por nivel geográfico: (columna en orders/items, columna en tabla de clientes)
LEVELS = {"Estado": ("customer_state", "state"), "Ciudad": ("city_key", "city_key")}
 
 
# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def norm(s) -> str:
    """minúsculas, sin acentos ni signos: 'Mogi-Guaçu' -> 'mogi guacu'."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()
 
 
def pretty_key(k: str) -> str:
    if " - " in k:
        city, uf = k.rsplit(" - ", 1)
        return f"{city.title()} - {uf}"
    return k
 
 
def fmt_pct(x) -> str:
    return "N/D" if pd.isna(x) else f"{x:.1f}%"
 
 
def _read(name: str) -> pd.DataFrame:
    local_path = os.path.join(LOCAL_DIR, f"{name}.parquet")
    if os.path.exists(local_path):
        return pd.read_parquet(local_path)
    return pd.read_parquet(f"{GITHUB_BASE_URL}/{name}.parquet")
 
 
# ----------------------------------------------------------------------------
# Carga y preparación de datos
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_data():
    """Devuelve (orders, items, cust).
 
    orders -> una fila por pedido (cliente, estado, ciudad y valor pagado).
    items  -> una fila por ítem (con categoría del producto).
    cust   -> una fila por cliente único (primera compra y días hasta recomprar).
 
    OJO: olist_order_items NO trae precio; el valor del pedido sale de
    olist_order_payments.payment_value. Por eso se analiza a nivel de PEDIDO.
    """
    orders = _read("olist_orders_dataset")
    customers = _read("olist_customers_dataset")
    payments = _read("olist_order_payments_dataset")
    items = _read("olist_order_items_dataset")
    products = _read("olist_products_dataset")
 
    for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
 
    orders = orders[~orders["order_status"].isin(EXCLUDED_STATUS)]
    pay = payments.groupby("order_id", as_index=False)["payment_value"].sum()
 
    orders = (
        orders.merge(customers[["customer_id", "customer_unique_id", "customer_city", "customer_state"]],
                     on="customer_id", how="inner")
        .merge(pay, on="order_id", how="left")
    )
    orders["payment_value"] = orders["payment_value"].fillna(0.0)
    orders["order_month"] = orders["order_purchase_timestamp"].dt.to_period("M").astype(str)
    orders["city_key"] = orders["customer_city"] + " - " + orders["customer_state"]
 
    # Cumplimiento: se compara por FECHA (la estimada no trae hora)
    delivered = orders["order_delivered_customer_date"]
    orders["is_delivered"] = delivered.notna()
    cmp = delivered.dt.normalize() <= orders["order_estimated_delivery_date"].dt.normalize()
    orders["on_time"] = cmp.where(orders["is_delivered"]).astype(float)
 
    # Ítems con categoría de producto
    items = items[items["order_id"].isin(orders["order_id"])]
    items = items.merge(products, on="product_id", how="left")
    items["product_category_name"] = items["product_category_name"].fillna("sin categoría")
    items = items.merge(orders[["order_id", "customer_state", "city_key"]], on="order_id", how="left")
 
    # Tabla de clientes únicos
    o = orders.sort_values("order_purchase_timestamp")
    o = o.assign(rank=o.groupby("customer_unique_id").cumcount())
    first = o[o["rank"] == 0].set_index("customer_unique_id")
    second = o[o["rank"] == 1].set_index("customer_unique_id")
    cust = pd.DataFrame({
        "first_purchase": first["order_purchase_timestamp"],
        "state": first["customer_state"],
        "city_key": first["city_key"],
    })
    cust["days_to_repeat"] = (second["order_purchase_timestamp"] - cust["first_purchase"]).dt.days
    cust = cust.reset_index()
 
    return orders.reset_index(drop=True), items.reset_index(drop=True), cust
 
 
@st.cache_data(show_spinner=False)
def load_geojson():
    with urllib.request.urlopen(GEOJSON_URL, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))
 
 
@st.cache_data(show_spinner=False)
def load_cities_geo() -> pd.DataFrame:
    m = pd.read_csv(MUNICIPIOS_URL, encoding="utf-8-sig")
    e = pd.read_csv(ESTADOS_URL, encoding="utf-8-sig")
    m = m.merge(e[["codigo_uf", "uf"]], on="codigo_uf")
    m["mk"] = m["nome"].map(norm) + "|" + m["uf"]
    m = m.rename(columns={"latitude": "lat", "longitude": "lon", "nome": "label"})
    return m.drop_duplicates("mk")[["mk", "lat", "lon", "label"]]
 
 
@st.cache_data(show_spinner=False)
def geo_base(level: str, min_orders: int):
    """Unidades geográficas del mapa (siempre con TODOS los datos, para poder
    elegir cualquier punto). Devuelve (df, pedidos_sin_coordenadas, pedidos_total)."""
    orders, _, _ = load_data()
    if level == "Estado":
        b = (orders.groupby("customer_state").size().rename("n_orders").reset_index()
             .rename(columns={"customer_state": "key"}))
        b["label"] = b["key"]
        return b.sort_values("key").reset_index(drop=True), 0, int(b["n_orders"].sum())
 
    g = (orders.groupby(["city_key", "customer_city", "customer_state"]).size()
         .rename("n_orders").reset_index())
    g = g[g["n_orders"] >= min_orders]
    g["mk"] = g["customer_city"].map(norm) + "|" + g["customer_state"]
    g = g.merge(load_cities_geo(), on="mk", how="left")
    lost = int(g.loc[g["lat"].isna(), "n_orders"].sum())
    total = int(g["n_orders"].sum())
    g = g.dropna(subset=["lat"]).rename(columns={"city_key": "key"})
    g["label"] = g["label"] + " - " + g["customer_state"]
    return g[["key", "label", "lat", "lon", "n_orders"]].sort_values("key").reset_index(drop=True), lost, total
 
 
# ----------------------------------------------------------------------------
# Cálculos
# ----------------------------------------------------------------------------
def window_mask(cust, days, max_date, complete):
    """Clientes que cuentan para la tasa: los que ya pudieron cumplir la ventana.
    Si NADIE la cumple en todo el dataset (p. ej. 2 años), se usa a todos."""
    if complete:
        return cust["first_purchase"] <= max_date - pd.Timedelta(days=days)
    return pd.Series(True, index=cust.index)
 
 
def repeat_rate(cust, days, max_date, complete):
    base = cust[window_mask(cust, days, max_date, complete)]
    if base.empty:
        return float("nan"), 0
    return (base["days_to_repeat"] <= days).mean() * 100, len(base)
 
 
def on_time_rate(orders) -> float:
    d = orders[orders["is_delivered"]]
    return d["on_time"].mean() * 100 if len(d) else float("nan")
 
 
def group_values(orders, cust, metric, max_date, complete, ocol, ccol) -> pd.Series:
    if metric == "Revenue total":
        return orders.groupby(ocol)["payment_value"].sum()
    if metric == "Precio promedio de compra":
        return orders.groupby(ocol)["payment_value"].mean()
    if metric == "% Cumplimiento de envíos":
        return orders[orders["is_delivered"]].groupby(ocol)["on_time"].mean() * 100
    days = METRICS[metric]["days"]
    base = cust[window_mask(cust, days, max_date, complete[days])]
    return (base["days_to_repeat"] <= days).groupby(base[ccol]).mean() * 100
 
 
def read_selection(event) -> list:
    """Extrae las claves (estado o 'ciudad - UF') de la selección del mapa."""
    try:
        points = event["selection"]["points"]
    except Exception:
        return []
    keys = []
    for p in points:
        cd = p.get("customdata")
        if cd:
            keys.append(cd[0] if isinstance(cd, (list, tuple)) else cd)
        elif p.get("location"):
            keys.append(p["location"])
    return sorted(set(keys))
 
 
# ----------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------
st.title("📊 Dashboard Ejecutivo: Ventas, Retención y Envíos (Olist)")
st.markdown("---")
 
try:
    with st.spinner("Cargando y uniendo los conjuntos de datos..."):
        orders_all, items_all, cust_all = load_data()
except Exception as e:
    st.error(f"No se pudieron cargar los archivos parquet. Verifica las rutas o la conexión. Error: {e}")
    st.stop()
 
max_date = orders_all["order_purchase_timestamp"].max()
complete = {d: bool((cust_all["first_purchase"] <= max_date - pd.Timedelta(days=d)).any())
            for d in WINDOWS.values()}
 
# ----------------------------------------------------------------------------
# Estado de la selección del mapa (se lee ANTES de dibujar los indicadores,
# para que el resto del dashboard se filtre en la misma ejecución)
# ----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("level", "Estado")
ss.setdefault("metric", list(METRICS)[0])
ss.setdefault("min_orders", 10)
ss.setdefault("sel", [])
ss.setdefault("ver", 0)
ss.setdefault("last_event", [])
ss.setdefault("last_sig", None)
 
 
def reset_selection():
    ss["sel"] = []
    ss["ver"] += 1
    ss["last_event"] = []
 
 
level = ss["level"]
metric = ss["metric"]
min_orders = ss["min_orders"] if level == "Ciudad" else 0
chart_key = f"map_{level}_{ss['ver']}"
event_keys = read_selection(ss.get(chart_key))
sig = (level, metric, min_orders)
if ss["last_sig"] != sig:
    # La figura cambió (variable / umbral): el evento recibido no es un clic del usuario
    ss["last_event"] = event_keys
    ss["last_sig"] = sig
elif event_keys != ss["last_event"]:
    ss["sel"] = event_keys
    ss["last_event"] = event_keys
 
ocol, ccol = LEVELS[level]
sel = ss["sel"]
if sel:
    orders = orders_all[orders_all[ocol].isin(sel)]
    items = items_all[items_all[ocol].isin(sel)]
    cust = cust_all[cust_all[ccol].isin(sel)]
else:
    orders, items, cust = orders_all, items_all, cust_all
 
if orders.empty:
    st.warning("No hay datos para la selección actual.")
    st.button("Limpiar selección", on_click=reset_selection, key="clear_empty")
    st.stop()
 
# ----------------------------------------------------------------------------
# Fila superior: indicadores generales
# ----------------------------------------------------------------------------
if sel:
    b1, b2 = st.columns([5, 1])
    shown = ", ".join(pretty_key(k) for k in sel[:8]) + (f" y {len(sel) - 8} más" if len(sel) > 8 else "")
    b1.info(f"🔎 Dashboard filtrado por {level.lower()}: **{shown}**")
    b2.button("Limpiar selección", on_click=reset_selection, key="clear_top", use_container_width=True)
 
st.subheader("📌 Indicadores Clave Generales")
 
c1, c2, c3 = st.columns(3)
c1.metric("Revenue Total", f"${orders['payment_value'].sum():,.2f}")
c2.metric("Precio Promedio de Compra (por pedido)", f"${orders['payment_value'].mean():,.2f}")
c3.metric("Cumplimiento de Envíos", fmt_pct(on_time_rate(orders)))
 
c4, c5, c6, c7 = st.columns([1, 1, 1, 2])
for col, (label, days) in zip([c4, c5, c6], WINDOWS.items()):
    rate, n = repeat_rate(cust, days, max_date, complete[days])
    note = (f"Base: {n:,} clientes con ventana completa." if complete[days] else
            f"Base: {n:,} clientes. Aproximación: ningún cliente del dataset tiene {label} completos de seguimiento.")
    col.metric(f"Recompra ({label})", fmt_pct(rate), help=note)
 
# El dataset no trae nombre de producto: se muestra la categoría como nombre
top = (items.groupby(["product_id", "product_category_name"]).size()
       .sort_values(ascending=False).head(5).reset_index(name="Cantidad Comprada"))
names = top["product_category_name"].str.replace("_", " ").str.title()
top["Producto"] = names + names.groupby(names).cumcount().map(lambda i: "" if i == 0 else f" ({i + 1})")
c7.write("🏆 **Top 5 Productos más Comprados**")
c7.dataframe(top[["Producto", "Cantidad Comprada"]], hide_index=True)
 
st.markdown("#### 📅 Revenue Total por Mes")
monthly = orders.groupby("order_month", as_index=False)["payment_value"].sum()
fig_month = px.bar(monthly, x="order_month", y="payment_value",
                   labels={"order_month": "Mes", "payment_value": "Revenue ($)"},
                   color_discrete_sequence=["#2a7f9e"])
fig_month.update_layout(margin={"r": 0, "t": 10, "l": 0, "b": 0}, xaxis_tickangle=-45)
st.plotly_chart(fig_month)
 
st.markdown("---")
 
# ----------------------------------------------------------------------------
# Mapa interactivo
# ----------------------------------------------------------------------------
st.subheader("🗺️ Mapa Interactivo")
 
m1, m2, m3 = st.columns([1.2, 2, 1.6])
m1.radio("Ver por", list(LEVELS), key="level", horizontal=True, on_change=reset_selection)
m2.selectbox("Variable a mostrar en el mapa", list(METRICS), key="metric")
if level == "Ciudad":
    m3.slider("Mínimo de pedidos por ciudad", 1, 200, key="min_orders", on_change=reset_selection,
              help="Oculta ciudades con pocos pedidos (sus tasas serían poco confiables).")
 
spec = METRICS[metric]
base, lost_orders, total_orders = (geo_base(level, min_orders) if level == "Ciudad" else geo_base(level, 0))
map_df = base.copy()
map_df["value"] = map_df["key"].map(group_values(orders_all, cust_all, metric, max_date, complete, ocol, ccol))
if level == "Ciudad":
    map_df = map_df.dropna(subset=["value"]).sort_values("n_orders", ascending=False).reset_index(drop=True)
map_df = map_df.reset_index(drop=True)
 
hover_extra = "<br>Pedidos: %{customdata[2]:,}<extra></extra>"
try:
    if level == "Estado":
        fig = px.choropleth_map(
            map_df, geojson=load_geojson(), locations="key", featureidkey="properties.sigla",
            color="value", custom_data=["key", "label", "n_orders"],
            color_continuous_scale=COLOR_SCALE, map_style=MAP_STYLE,
            center={"lat": -14.5, "lon": -52.0}, zoom=3.1, opacity=0.8,
            labels={"value": spec["label"]},
        )
        fig.update_traces(
            marker_line_width=0.8, marker_line_color="white",
            hovertemplate=f"<b>%{{customdata[1]}}</b><br>{spec['label']}: {spec['prefix']}%{{z:{spec['fmt']}}}{spec['suffix']}" + hover_extra,
        )
    else:
        fig = px.scatter_map(
            map_df, lat="lat", lon="lon", size="n_orders", size_max=26, color="value",
            custom_data=["key", "label", "n_orders"],
            color_continuous_scale=COLOR_SCALE, map_style=MAP_STYLE,
            center={"lat": -14.5, "lon": -52.0}, zoom=3.1, opacity=0.8,
            labels={"value": spec["label"]},
        )
        fig.update_traces(
            hovertemplate=f"<b>%{{customdata[1]}}</b><br>{spec['label']}: {spec['prefix']}%{{marker.color:{spec['fmt']}}}{spec['suffix']}" + hover_extra,
        )
 
    # Resaltar lo seleccionado
    idx = map_df.index[map_df["key"].isin(sel)].tolist()
    if idx:
        fig.update_traces(selectedpoints=idx, unselected=dict(marker=dict(opacity=0.3)),
                          selected=dict(marker=dict(opacity=1)))
 
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=640, clickmode="event+select", dragmode="pan",
        coloraxis_colorbar=dict(title=spec["label"], thickness=12, len=0.6),
    )
    st.plotly_chart(fig, key=chart_key, on_select="rerun", selection_mode="points",
                    config={"scrollZoom": True, "displaylogo": False})
    st.caption("👆 Haz clic en un " + ("estado" if level == "Estado" else "punto (ciudad)") +
               " para filtrar todo el dashboard · Shift + clic para elegir varios · clic en una zona vacía o "
               "«Limpiar selección» para quitar el filtro.")
except Exception as e:
    st.error(f"No se pudo dibujar el mapa: {e}")
 
with st.expander("Ver datos del mapa"):
    table = map_df[["label", "value", "n_orders"]].rename(
        columns={"label": level, "value": spec["label"], "n_orders": "Pedidos"})
    st.dataframe(table.sort_values(spec["label"], ascending=False), hide_index=True)
 
if sel:
    st.button("Limpiar selección", on_click=reset_selection, key="clear_map")
 
# ----------------------------------------------------------------------------
# Consideraciones (al final del dashboard)
# ----------------------------------------------------------------------------
st.markdown("---")

st.subheader("👾 PREGUNTALE A LA IA 👾")

# 2. Convertirlos a texto en formato Markdown
contexto_dataframes = f"""
Aquí tienes los 3 DataFrames para analizar:

### DataFrame 1: Productos
{orders.head(1000).to_markdown(index=False)}

"""


# 3. Enviar a la API de Groq
client = Groq(api_key=API_K)

prompt = f"""
{contexto_dataframes}

Pregunta: Analiza los  conjuntos de datos y dime qué producto con mayor cumplimiento de envios tiene mejor desempeño de ventas.
"""

chat_completion = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model="meta-llama/llama-prompt-guard-2-22m",  # O el modelo de Groq que prefieras
)

print(chat_completion.choices[0].message.content)


st.markdown("---")

st.subheader("📝 Consideraciones")
notes = [
    f"**Pedidos analizados:** {len(orders):,} de {len(orders_all):,} · datos hasta {max_date:%Y-%m-%d}. "
    f"Se excluyen pedidos {', '.join(EXCLUDED_STATUS)}.",
    "**Revenue:** suma del valor pagado por pedido (`payment_value`). "
    "El precio promedio es el valor promedio por pedido.",
    "**Cumplimiento de envíos:** % de pedidos entregados cuya fecha de entrega fue igual o anterior a la fecha estimada.",
    "**Recompra:** % de clientes únicos con una segunda compra dentro de 90 / 210 / 730 días desde su primera compra. "
    "Solo cuentan clientes cuya ventana ya pudo completarse."
    + ("" if all(complete.values()) else
       " Para los 2 años ningún cliente tiene seguimiento completo en este dataset, por lo que se usa a todos los clientes (aproximación)."),
    "**Top 5 de productos:** el dataset no incluye nombre del producto, por eso se muestra su categoría; "
    "si dos productos comparten categoría se numeran, p. ej. «Ferramentas Jardim (2)».",
    "**Mapa:** siempre muestra todos los datos para poder elegir otra zona; la selección filtra los indicadores, "
    "el top 5 y el revenue mensual. En el nivel «Ciudad» el tamaño del punto es el número de pedidos y el color la variable elegida.",
]
if level == "Ciudad" and total_orders:
    notes.append(f"**Ubicación de ciudades:** coordenadas de municipios de Brasil (IBGE); "
                 f"{lost_orders:,} pedidos ({lost_orders / total_orders:.1%}) pertenecen a ciudades sin coordenadas y no aparecen en el mapa.")
for n in notes:
    st.markdown(f"- {n}")
