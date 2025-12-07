import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
import requests
import math

warnings.filterwarnings("ignore")

from krishi_saathi_llm import KrishiSaathiAdvisor

st.set_page_config(page_title="Oilseed Yield Advisor", page_icon="🌾", layout="wide")

# ---------------------------------------------------------
# Session State Setup
# ---------------------------------------------------------
if "model" not in st.session_state:
    st.session_state.model = None

if "advisor" not in st.session_state:
    try:
        st.session_state.advisor = KrishiSaathiAdvisor()
    except Exception as e:
        st.warning(f"LLM Advisor init failed: {e}")
        st.session_state.advisor = None

if "weather_features" not in st.session_state:
    st.session_state.weather_features = {}

# ---------------------------------------------------------
# Helper: Crop Benchmarks (rough)
# ---------------------------------------------------------
CROP_BENCHMARK_YIELD = {
    "sunflower": 1.5,
    "soybean": 1.8,
    "mustard": 1.4,
    "groundnut": 2.0,
    "sesame": 0.9,
    "castor": 1.7,
    "safflower": 0.8,
    "niger": 0.7,
}

# Approx MSP-like prices (₹ per ton)
CROP_MSP_RS_PER_TON = {
    "sunflower": 64100,
    "soybean": 30500,
    "mustard": 56500,
    "groundnut": 64500,
    "sesame": 78500,
    "castor": 55000,
    "safflower": 56500,
    "niger": 60000,
}

# Approximate costs for interventions
IRRIGATION_COST_PER_EVENT_RS = 1500
N_COST_PER_KG_RS = 35
P_COST_PER_KG_RS = 60
K_COST_PER_KG_RS = 40
NDVI_COST_PER_DELTA1_RS = 2000

# Weather Based Alerts Rules
CROP_ALERT_RULES = {
    "sunflower": {"max_temp_flowering": 34, "min_rain_flowering": 30, "max_humidity": 80},
    "soybean": {"max_temp_flowering": 32, "min_rain_flowering": 35, "max_humidity": 85},
    "mustard": {"max_temp_flowering": 30, "min_rain_flowering": 25, "max_humidity": 88},
    "groundnut": {"max_temp_flowering": 35, "min_rain_flowering": 40, "max_humidity": 82},
    "sesame": {"max_temp_flowering": 33, "min_rain_flowering": 25, "max_humidity": 80},
    "castor": {"max_temp_flowering": 34, "min_rain_flowering": 28, "max_humidity": 84},
    "safflower": {"max_temp_flowering": 32, "min_rain_flowering": 22, "max_humidity": 80},
    "niger": {"max_temp_flowering": 30, "min_rain_flowering": 20, "max_humidity": 78},
}

def generate_weather_alerts(crop, temp_flowering, rain_flowering, humidity):
    crop = crop.lower()
    rules = CROP_ALERT_RULES.get(crop, CROP_ALERT_RULES["soybean"])
    alerts = []

    if temp_flowering >= rules["max_temp_flowering"] + 3:
        alerts.append("🔥 Severe heat at flowering → high risk of flower drop.")
    elif temp_flowering >= rules["max_temp_flowering"]:
        alerts.append("🌡️ High temperature at flowering → moderate stress.")

    if rain_flowering <= 0.5 * rules["min_rain_flowering"]:
        alerts.append("💧 Very low rainfall → severe moisture stress.")
    elif rain_flowering <= rules["min_rain_flowering"]:
        alerts.append("💧 Low rainfall → moisture stress risk.")

    if humidity >= rules["max_humidity"] + 5:
        alerts.append("🦠 Very high humidity → high fungal/pest risk.")
    elif humidity >= rules["max_humidity"]:
        alerts.append("🦠 High humidity → risk of foliar diseases.")

    if not alerts:
        alerts.append("✅ No major weather red flags detected.")
    return alerts


@st.cache_data
def load_districts_csv():
    try:
        df = pd.read_csv("data/india_districts_2023.csv")
        df["display"] = df["district"] + ", " + df["state"]
        return df
    except Exception:
        return None

@st.cache_data
def load_pincode_csv():
    try:
        return pd.read_csv("data/india_pincodes.csv")
    except Exception:
        return None

def geocode_with_nominatim(query: str):
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={query}, India"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    data = resp.json()
    if not data:
        return None, None
    return float(data[0]["lat"]), float(data[0]["lon"])

def resolve_location(district_df, pincode_df, district_display, pincode):
    if pincode and pincode_df is not None:
        row = pincode_df.loc[pincode_df["pincode"].astype(str) == str(pincode)]
        if not row.empty:
            r = row.iloc[0]; return float(r["lat"]), float(r["lon"]), r["district"]
    if district_display and district_df is not None:
        row = district_df.loc[district_df["display"] == district_display]
        if not row.empty:
            r = row.iloc[0]; return float(r["lat"]), float(r["lon"]), r["district"]
    lat, lon = geocode_with_nominatim(district_display)
    return lat, lon, district_display


def fetch_weather_features(lat, lon, sowing_date, maturity_days):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,precipitation"
        "&timezone=auto&past_days=90&forecast_days=16"
    )
    data = requests.get(url).json()
    if "hourly" not in data: return {"error": "Weather unavailable"}

    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temp": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "rain": data["hourly"]["precipitation"],
    })

    flowering_date = sowing_date + pd.Timedelta(days=int(maturity_days * 0.35))
    gs_df = df[df["time"] >= sowing_date]
    fl_df = df[(df["time"] >= flowering_date - pd.Timedelta(days=7)) &
               (df["time"] <= flowering_date + pd.Timedelta(days=7))]

    return {
        "mean_temp_gs_C": round(gs_df["temp"].mean(), 2),
        "temp_flowering_C": round(fl_df["temp"].mean() if not fl_df.empty else gs_df["temp"].mean(), 2),
        "seasonal_rain_mm": round(gs_df["rain"].sum(), 2),
        "rain_flowering_mm": round(fl_df["rain"].sum(), 2) if not fl_df.empty else round(gs_df["rain"].sum(), 2),
        "humidity_mean_pct": round(gs_df["humidity"].mean(), 2),
        "flowering_day": str(flowering_date.date()),
    }

# ---------------------------------------------------------
# UI HEADER
# ---------------------------------------------------------
st.title("🌾 Oilseed Yield Prediction & Krishi Saathi 2.0 Advisory")

district_df = load_districts_csv()
pincode_df = load_pincode_csv()

# Upload model
st.sidebar.header("Upload CatBoost Model")
uploaded_model_file = st.sidebar.file_uploader("Upload .pkl/.joblib", type=["pkl", "joblib"])
if uploaded_model_file and st.session_state.model is None:
    st.session_state.model = joblib.load(uploaded_model_file)
    st.sidebar.success("Model loaded!")

# ---------------------------------------------------------
# FIELD INPUT FORM
# ---------------------------------------------------------
st.markdown("## Field Input Form")

with st.form("prediction_form"):

    with st.expander("Crop Details", expanded=True):
        crop = st.selectbox("Crop", ["sunflower", "soybean", "mustard", "groundnut", "sesame", "castor", "safflower", "niger"])
        maturity_days = st.number_input("Maturity Days", 60, 200, 120)

    with st.expander("Weather & Climate", expanded=True):
        search_text = st.text_input("Start typing district name")
        district_display = None
        if search_text and district_df is not None:
            filtered = [d for d in district_df["display"] if search_text.lower() in d.lower()]
            district_display = st.selectbox("Select District", filtered)
        else:
            district_display = st.text_input("District (manual if needed)", "")

        pincode = st.text_input("PIN Code (optional)")
        sowing_date = st.date_input("Sowing Date")
        autofill_weather = st.form_submit_button("🌤 Autofill Weather")

        sowing_ts = pd.to_datetime(sowing_date)
        if autofill_weather:
            lat, lon, loc = resolve_location(district_df, pincode_df, district_display, pincode)
            feats = fetch_weather_features(lat, lon, sowing_ts, maturity_days)
            if "error" not in feats:
                st.session_state.weather_features = feats
                st.success("Weather autofilled!")

        wf = st.session_state.weather_features
        mean_temp_gs_C = st.number_input("Mean Temp (°C)", 10.0, 45.0, wf.get("mean_temp_gs_C", 25.0))
        temp_flowering_C = st.number_input("Temp at Flowering (°C)", 10.0, 45.0, wf.get("temp_flowering_C", 26.0))
        seasonal_rain_mm = st.number_input("Seasonal Rain (mm)", 0.0, 2000.0, wf.get("seasonal_rain_mm", 600.0))
        rain_flowering_mm = st.number_input("Rainfall at Flowering (mm)", 0.0, 500.0, wf.get("rain_flowering_mm", 100.0))
        humidity_mean_pct = st.number_input("Humidity (%)", 0.0, 100.0, wf.get("humidity_mean_pct", 65.0))

    with st.expander("Soil & Fertility"):
        soil_pH = st.number_input("Soil pH", 3.5, 10.0, 6.5)
        clay_pct = st.number_input("Clay %", 0.0, 70.0, 25.0)
        soil_N_status_kg_ha = st.number_input("Soil N (kg/ha)", 0.0, 400.0, 200.0)
        soil_P_status_kg_ha = st.number_input("Soil P (kg/ha)", 0.0, 100.0, 50.0)
        soil_K_status_kg_ha = st.number_input("Soil K (kg/ha)", 0.0, 500.0, 150.0)
        fert_N_kg_ha = st.number_input("Fertilizer N (kg/ha)", 0.0, 250.0, 80.0)
        fert_P_kg_ha = st.number_input("Fertilizer P (kg/ha)", 0.0, 120.0, 40.0)
        fert_K_kg_ha = st.number_input("Fertilizer K (kg/ha)", 0.0, 250.0, 50.0)
        soil_moisture_pct = st.slider("Soil Moisture (%)", 0, 100, 40)

    with st.expander("Irrigation"):
        irrigation_events = st.number_input("Number of Irrigations", 0, 15, 2)

    with st.expander("NDVI & Crop Health"):
        ndvi_flowering = st.slider("NDVI Flowering", 0.0, 1.0, 0.60)
        ndvi_peak = st.slider("NDVI Peak", 0.0, 1.0, 0.80)
        ndvi_veg_slope = st.slider("NDVI Vegetative Slope", -2.0, 2.0, 0.3)

    farmer_query = st.text_area("Your Question", "How can I improve my yield?")
    advisory_language = st.selectbox("Language", ["auto", "English", "Hindi", "Odia"])
    predict_button = st.form_submit_button("Predict Yield 🌾")


# ---------------------------------------------------------
# Standalone Prediction Logic
# ---------------------------------------------------------
if predict_button:

    if st.session_state.model is None:
        st.error("Upload the model to continue.")
        st.stop()

    input_data = {
        "crop": crop,
        "maturity_days": maturity_days,
        "mean_temp_gs_C": mean_temp_gs_C,
        "temp_flowering_C": temp_flowering_C,
        "seasonal_rain_mm": seasonal_rain_mm,
        "rain_flowering_mm": rain_flowering_mm,
        "humidity_mean_pct": humidity_mean_pct,
        "soil_pH": soil_pH,
        "clay_pct": clay_pct,
        "soil_N_status_kg_ha": soil_N_status_kg_ha,
        "soil_P_status_kg_ha": soil_P_status_kg_ha,
        "soil_K_status_kg_ha": soil_K_status_kg_ha,
        "fert_N_kg_ha": fert_N_kg_ha,
        "fert_P_kg_ha": fert_P_kg_ha,
        "fert_K_kg_ha": fert_K_kg_ha,
        "irrigation_events": irrigation_events,
        "ndvi_flowering": ndvi_flowering,
        "ndvi_peak": ndvi_peak,
        "ndvi_veg_slope": ndvi_veg_slope,
        "soil_moisture_pct": soil_moisture_pct,
    }

    df = pd.DataFrame([input_data])
    df["crop"] = df["crop"].astype(str)
    try:
        df = df[st.session_state.model.feature_names_]
    except Exception:
        pass

    predicted_yield = float(st.session_state.model.predict(df)[0])
    st.session_state.predicted_yield = predicted_yield
    st.session_state.input_data = input_data
    st.metric("Predicted Yield (t/ha)", f"{predicted_yield:.2f}")
    st.success("ML prediction completed!")

# ---------------------------------------------------------
# Advisory Button - triggers only RAG when clicked
# ---------------------------------------------------------
if "predicted_yield" in st.session_state:

    if st.button("Get Expert Advisory 🧠🌱"):
        input_data = st.session_state.input_data
        predicted_yield = st.session_state.predicted_yield

        alerts = generate_weather_alerts(
            input_data["crop"],
            input_data["temp_flowering_C"],
            input_data["rain_flowering_mm"],
            input_data["humidity_mean_pct"]
        )

        yield_context = {
            "crop": input_data["crop"],
            "yield": predicted_yield,
            "unit": "t/ha",
            "features": input_data,
            "alerts": alerts
        }

        with st.spinner("Generating advisory..."):
            advisory = st.session_state.advisor.chat(
                session_id=f"session_{input_data['crop']}",
                farmer_query=farmer_query,
                yield_dict=yield_context,
                language=advisory_language,
            )
        st.subheader("LLM Advisory")
        st.write(advisory)
