# app.py
# Electricity Demand: Coordinador API (today) + ML Forecast (today+1+2)

import os
import io
import zipfile
from datetime import datetime, timedelta

import holidays
import joblib
import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st
import plotly.express as px
import xgboost as xgb

# ----- Streamlit cache polyfills for old versions -----
if not hasattr(st, "cache_resource"):
    def _cache_resource(**_):
        return st.cache(allow_output_mutation=True)
    st.cache_resource = _cache_resource  # type: ignore[attr-defined]

if not hasattr(st, "cache_data"):
    def _cache_data(**_):
        return st.cache()
    st.cache_data = _cache_data  # type: ignore[attr-defined]
# ------------------------------------------------------

# -----------------------------
# CONFIG
# -----------------------------
LAT, LON = -42.3565, -73.7192
CL_TZ = "America/Santiago"

DEFAULT_COORD_API_URL = "https://sipub.api.coordinador.cl:443/cmg-programado-pid/v4/findByDate"
DEFAULT_USER_KEY = "bae940cb90ef4111ffacb6baa9806fc0"  # consider moving to st.secrets
CMG_KEYS = ["Chiloe110", "Chonchi110", "Pid-Pid110"]

MODEL_ZIP = "xgbmodel_vgpt5.zip"
MODEL_FILENAME = "xgbmodel_vgpt5.ubj"

HOURLY_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "cloudcover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "rain",
    "snowfall",
]

# -----------------------------
# HELPERS
# -----------------------------
def now_cl():
    return datetime.now(pytz.timezone(CL_TZ))

def date_str(d: datetime):
    return d.strftime("%Y-%m-%d")

def today_date_str():
    return date_str(now_cl())

def tomorrow_date_str():
    return date_str(now_cl() + timedelta(days=1))

@st.cache_resource(show_spinner=False)
def load_model_from_zip(zip_path: str, inner_filename: str, api: str = "xgb_booster"):
    with zipfile.ZipFile(zip_path, 'r') as z:
        blob = z.read(inner_filename)

    if api == "xgb_booster":
        model = xgb.Booster()
        model.load_model(bytearray(blob))
        return model

    if api == "xgb_sklearn":
        model = xgb.XGBRegressor()
        model.load_model(bytearray(blob))
        return model

    if api == "joblib":
        return joblib.load(io.BytesIO(blob))

    raise ValueError(f"Unknown api='{api}'")

@st.cache_data(show_spinner=False, ttl=10 * 60)
def get_open_meteo_forecast(lat: float, lon: float, target_date: str) -> pd.DataFrame:
    hourly_vars = ",".join(HOURLY_VARS)
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly={hourly_vars}"
        f"&timezone=auto"
        f"&start_date={target_date}&end_date={target_date}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly).copy()
    df["datetime"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"cloudcover": "cloud_cover"})
    return df

@st.cache_data(show_spinner=False, ttl=10 * 60)
def get_open_meteo_range(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly weather from Open-Meteo for an inclusive date range [start_date, end_date].
    Returns a single continuous dataframe so lag features can bridge across days.
    """
    hourly_vars = ",".join(HOURLY_VARS)
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly={hourly_vars}"
        f"&timezone=auto"
        f"&start_date={start_date}&end_date={end_date}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly).copy()
    df["datetime"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"cloudcover": "cloud_cover"})
    return df.sort_values("datetime").reset_index(drop=True)


def create_lagged_features(df: pd.DataFrame, required_features) -> pd.DataFrame:
    df = df.copy()
    base_lags = set()
    for col in required_features:
        if "_lag" in col:
            var, lag = col.rsplit("_lag", 1)
            try:
                lag = int(lag)
            except:
                continue
            base_lags.add((var, lag))

    for var, lag in base_lags:
        if var not in df.columns:
            df[var] = np.nan
        if lag == 0:
            df[f"{var}_lag0"] = df[var]
        else:
            df[f"{var}_lag{lag}"] = df[var].shift(lag)

    return df.dropna().reset_index(drop=True)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    cl_holidays = holidays.CL()
    df["is_holiday"] = df["datetime"].dt.date.apply(lambda x: x in cl_holidays)
    return df

# === NEW: explicit CMG fetch for any date range ===
@st.cache_data(show_spinner=False, ttl=5 * 60)
def get_coordinator_cmg_by_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch CMG from Coordinador for [start_date, end_date)."""
    params = {
        "startDate": start_date,
        "endDate": end_date,  # exclusive end
        "page": 1,
        "limit": 20000,
        "user_key": DEFAULT_USER_KEY,
    }
    r = requests.get(DEFAULT_COORD_API_URL, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    if "data" not in payload:
        return pd.DataFrame()

    df = pd.DataFrame(payload["data"])
    if df.empty:
        return df

    # Filter to desired CMG keys if present
    if "llave_cmg" in df.columns:
        df = df.loc[df["llave_cmg"].isin(CMG_KEYS)].copy()

    # -------- Robust timestamp building --------
    ts = None
    if "fechaHora" in df.columns:
        ts = df["fechaHora"]
    elif "fecha_hora" in df.columns:
        ts = df["fecha_hora"]
    elif {"fecha", "hora"}.issubset(df.columns):
        # combine date + hour safely
        ts = (
            pd.to_datetime(df["fecha"], errors="coerce")
            + pd.to_timedelta(pd.to_numeric(df["hora"], errors="coerce"), unit="h")
        )
    elif "time" in df.columns:
        ts = df["time"]

    if ts is None:
        # no recognized timestamp fields; create an all-NaT Series
        df["timestamp"] = pd.Series(pd.NaT, index=df.index)
    else:
        # ensure Series, then coerce to tz-aware UTC and convert to Chile tz
        if not isinstance(ts, pd.Series):
            ts = pd.Series(ts, index=df.index)
        ts = pd.to_datetime(ts, errors="coerce", utc=True)  # naive treated as UTC
        # ts is now datetime64[ns, UTC], so .dt is safe
        df["timestamp"] = ts.dt.tz_convert(CL_TZ)
    # -------------------------------------------

    # Heuristic for CMG numeric column
    if "cmg_usd_mwh" in df.columns:
        df["cmg_usd_mwh"] = pd.to_numeric(df["cmg_usd_mwh"], errors="coerce")
    else:
        for candidate in ["cmg", "valor", "programado", "price", "value"]:
            if candidate in df.columns:
                df["cmg_usd_mwh"] = pd.to_numeric(df[candidate], errors="coerce")
                break
        else:
            df["cmg_usd_mwh"] = np.nan

    sort_cols = [c for c in ["llave_cmg", "timestamp"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    return df

# -----------------------------
# PLOTTING
# -----------------------------
def plot_cmg(df: pd.DataFrame):
    if df.empty or df["timestamp"].isna().all():
        st.info("No data (or unknown schema) from Coordinador API for the selected day.")
        return
    fig = px.line(
        df,
        x="timestamp",
        y="cmg_usd_mwh",
        color="llave_cmg" if "llave_cmg" in df.columns else None,
        markers=True,
        labels={"timestamp": "Hora (CLT)", "cmg_usd_mwh": "CMG programado (USD/MWh)"},
        title=None,
    )
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(range=[0, None])  # <<< start y at 0
    st.plotly_chart(fig, use_container_width=True)

def plot_ml_forecast(df_pred: pd.DataFrame):
    if df_pred.empty:
        st.info("No prediction points to show.")
        return
    # If multiple days, color by date for readability
    color_col = "target_date" if "target_date" in df_pred.columns else None
    fig = px.line(
        df_pred,
        x="datetime",
        y="predicted_demand",
        color=color_col,
        markers=True,
        labels={
            "datetime": "Hora (CLT)",
            "predicted_demand": "Demanda predicha (MW)",
            "target_date": "Día objetivo",
        },
        title=None,
    )
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(range=[0, None])  # <<< start y at 0
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Pronóstico de Precios", page_icon="⚡", layout="wide")

st.title("Pronóstico de Precios Pudidi⚡")
st.markdown(
    "Esta aplicación muestra **el costo marginal programado (CMg) más reciente** desde la API del Coordinador "
    "y una **predicción con machine learning para los próximos días** usando pronósticos meteorológicos y comportamientos históricos."
)

# === NUEVO: Selector de fecha en la barra lateral, por defecto al 'hoy' de Chile ===
with st.sidebar:
    st.header("Configuración")
    default_today = now_cl().date()
    selected_date = st.date_input(
        "Día de referencia (hora de Chile)",
        value=default_today,
        help="Se usa como ‘hoy’. El modelo ML corre para hoy + 2 días. El CMG se muestra para este día seleccionado.",
    )
    show_table_cmg = st.checkbox("Mostrar datos del Coordinador", value=False)

# Botón de refresco (solo para volver a ejecutar; Streamlit se recarga con cualquier interacción)
_ = st.button("🔄 Actualizar datos")

# Preparar strings de fecha para el horizonte ML (hoy, +1, +2)
today_ds = selected_date.strftime("%Y-%m-%d")
tomorrow_ds = (datetime.combine(selected_date, datetime.min.time()) + timedelta(days=1)).strftime("%Y-%m-%d")
day_after_ds = (datetime.combine(selected_date, datetime.min.time()) + timedelta(days=2)).strftime("%Y-%m-%d")
ml_days = [today_ds, tomorrow_ds, day_after_ds]

st.markdown(f"**‘Hoy’ seleccionado (Chile):** `{today_ds}`  |  **Horizonte ML:** `{ml_days}`")



# -----------------------------
# 2) PREDICCIÓN ML (hoy + 2 días) — rezagos continuos entre días
# -----------------------------
st.subheader("Predicción Machine Learning para SE Dalcahue 23kV")

with st.spinner("Cargando modelo y construyendo variables a partir de Open-Meteo..."):
    # Construir rango de fechas inclusivo para meteo (desde 'hoy' seleccionado hasta día+2)
    start_ds = today_ds
    end_ds = day_after_ds  # fin inclusivo para Open-Meteo

    # 1) Datos meteorológicos para todo el rango en un solo dataframe continuo
    meteo_df = get_open_meteo_range(LAT, LON, start_ds, end_ds)

    # 2) Cargar modelo
    if not os.path.exists(MODEL_ZIP):
        st.error(f"Archivo de modelo `{MODEL_ZIP}` no encontrado en el directorio de trabajo.")
        st.stop()
    model = load_model_from_zip(MODEL_ZIP, MODEL_FILENAME, api="xgb_sklearn")
    required_features = list(getattr(model, "feature_names_in_", []))
    if not required_features:
        st.error("El modelo no contiene `feature_names_in_`. Reentrene el modelo con scikit-learn 1.0+ para incluirlo.")
        st.stop()

    # 3) Construir features UNA SOLA VEZ para todo el rango (rezagos cruzan días)
    feats = meteo_df.copy()
    feats = add_time_features(feats)
    feats = create_lagged_features(feats, required_features)

    if feats.empty:
        st.info("No hay suficientes filas después de crear rezagos para ejecutar predicciones.")
        pred_df = pd.DataFrame(columns=["datetime", "predicted_demand", "target_date"])
    else:
        X = feats[required_features]
        feats["predicted_demand"] = model.predict(X)

        # Etiquetar cada fila con su día calendario para la leyenda
        feats["target_date"] = feats["datetime"].dt.date.astype(str)

        # Mantener solo los 3 días solicitados
        day_set = {today_ds, tomorrow_ds, day_after_ds}
        pred_df = feats.loc[feats["target_date"].isin(day_set), ["datetime", "predicted_demand", "target_date"]]

plot_ml_forecast(pred_df)

with st.expander("Ver tabla de predicción ML"):
    st.dataframe(pred_df.reset_index(drop=True))


# -----------------------------
# 1) API COORDINADOR (día seleccionado)
# -----------------------------
st.subheader("CMg programado (API del Coordinador)")
with st.spinner("Obteniendo datos de la API del Coordinador..."):
    cmg_df = get_coordinator_cmg_by_range(today_ds, (datetime.strptime(today_ds, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"))
plot_cmg(cmg_df)


if show_table_cmg and not cmg_df.empty:
    st.dataframe(cmg_df)
with st.sidebar:
    st.markdown("---")
    st.caption("📌 Autor: Alejandro Bañados")



