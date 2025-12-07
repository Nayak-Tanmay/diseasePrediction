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
NDVI_COST_PER_DELTA1_RS = 2000  # cost per +1.0 NDVI (≈ 0.1 → 200 Rs)

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


def compute_sowing_doy(date_obj):
    return int(date_obj.timetuple().tm_yday)


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
            r = row.iloc[0]
            return float(r["lat"]), float(r["lon"]), f"{r['district']}, {r['state']}"

    if district_display and district_df is not None:
        row = district_df.loc[district_df["display"] == district_display]
        if not row.empty:
            r = row.iloc[0]
            return float(r["lat"]), float(r["lon"]), f"{r['district']}, {r['state']}"

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

    if "hourly" not in data:
        return {"error": "Weather unavailable"}

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

    mean_temp_gs = round(gs_df["temp"].mean(), 2)
    temp_flowering = round(fl_df["temp"].mean() if not fl_df.empty else mean_temp_gs, 2)
    seasonal_rain = round(gs_df["rain"].sum(), 2)
    rain_flowering = round(fl_df["rain"].sum(), 2) if not fl_df.empty else seasonal_rain
    humidity_mean = round(gs_df["humidity"].mean(), 2)

    return {
        "mean_temp_gs_C": mean_temp_gs,
        "temp_flowering_C": temp_flowering,
        "seasonal_rain_mm": seasonal_rain,
        "rain_flowering_mm": rain_flowering,
        "humidity_mean_pct": humidity_mean,
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
        base_yield = st.number_input("Base Yield Potential (t/ha)", 0.5, 6.0, 2.5)

    with st.expander("Weather & Climate", expanded=True):
        col1, col2 = st.columns([2, 1])
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
        soil_texture = st.selectbox("Soil Texture", ["sandy", "loamy", "clay", "silty", "peaty"])
        soil_depth_cm = st.number_input("Soil Depth (cm)", 10, 200, 100)
        soil_pH = st.number_input("Soil pH", 3.5, 10.0, 6.5)
        soil_oc_pct = st.number_input("Soil Organic Carbon (%)", 0.0, 5.0, 1.5)
        clay_pct = st.number_input("Clay %", 0.0, 70.0, 25.0)

        soil_N_status_kg_ha = st.number_input("Soil N (kg/ha)", 0.0, 400.0, 200.0)
        soil_P_status_kg_ha = st.number_input("Soil P (kg/ha)", 0.0, 100.0, 50.0)
        soil_K_status_kg_ha = st.number_input("Soil K (kg/ha)", 0.0, 500.0, 150.0)

        fert_N_kg_ha = st.number_input("Fertilizer N (kg/ha)", 0.0, 250.0, 80.0)
        fert_P_kg_ha = st.number_input("Fertilizer P (kg/ha)", 0.0, 120.0, 40.0)
        fert_K_kg_ha = st.number_input("Fertilizer K (kg/ha)", 0.0, 250.0, 50.0)

        soil_moisture_pct = st.slider("Soil Moisture (%)", 0, 100, 40)
        seed_moisture_pct = st.number_input("Seed Moisture at Harvest (%)", 0.0, 30.0, 10.0)

    with st.expander("Irrigation"):
        irrigation_events = st.number_input("Number of Irrigations", 0, 15, 2)

    with st.expander("NDVI & Crop Health"):
        ndvi_early = st.slider("NDVI Early", 0.0, 1.0, 0.55)
        ndvi_flowering = st.slider("NDVI Flowering", 0.0, 1.0, 0.60)
        ndvi_peak = st.slider("NDVI Peak", 0.0, 1.0, 0.80)
        ndvi_late = st.slider("NDVI Late", 0.0, 1.0, 0.45)
        ndvi_veg_slope = st.slider("NDVI Vegetative Slope", -2.0, 2.0, 0.3)

    st.markdown("### Advisory Settings")
    farmer_query = st.text_area("Your Question", "How can I improve my yield?")
    advisory_language = st.selectbox("Language", ["auto", "English", "Hindi", "Odia"])

    submit_button = st.form_submit_button("Predict & Advise 🚀")


# ---------------------------------------------------------
# Prediction Logic (Stores results to session_state)
# ---------------------------------------------------------
if submit_button:
    if st.session_state.model is None:
        st.error("Upload the model to continue.")
        st.stop()

    sowing_doy = compute_sowing_doy(sowing_date)

    input_data = {
        "crop": crop,
        "maturity_days": maturity_days,
        "base_yield_potential_t_ha": base_yield,
        "mean_temp_gs_C": mean_temp_gs_C,
        "temp_flowering_C": temp_flowering_C,
        "seasonal_rain_mm": seasonal_rain_mm,
        "rain_flowering_mm": rain_flowering_mm,
        "humidity_mean_pct": humidity_mean_pct,
        "soil_pH": soil_pH,
        "soil_oc_pct": soil_oc_pct,
        "soil_texture": soil_texture,
        "clay_pct": clay_pct,
        "soil_depth_cm": soil_depth_cm,
        "soil_N_status_kg_ha": soil_N_status_kg_ha,
        "soil_P_status_kg_ha": soil_P_status_kg_ha,
        "soil_K_status_kg_ha": soil_K_status_kg_ha,
        "fert_N_kg_ha": fert_N_kg_ha,
        "fert_P_kg_ha": fert_P_kg_ha,
        "fert_K_kg_ha": fert_K_kg_ha,
        "sowing_doy": sowing_doy,
        "irrigation_events": irrigation_events,
        "ndvi_early": ndvi_early,
        "ndvi_flowering": ndvi_flowering,
        "ndvi_peak": ndvi_peak,
        "ndvi_late": ndvi_late,
        "ndvi_veg_slope": ndvi_veg_slope,
        "seed_moisture_pct": seed_moisture_pct,
        "soil_moisture_pct": soil_moisture_pct,
        "district_display": district_display
    }

    df = pd.DataFrame([input_data])
    df[["crop", "soil_texture"]] = df[["crop", "soil_texture"]].astype(str)

    try:
        df = df[st.session_state.model.feature_names_]
    except Exception:
        pass

    predicted_yield = float(st.session_state.model.predict(df)[0])

    # Save to session state so simulation & advisory can run independently
    st.session_state.predicted_yield = predicted_yield
    st.session_state.input_data = input_data

    st.metric("Predicted Yield (t/ha)", f"{predicted_yield:.2f}")


# ---------------------------------------------------------
# 📈 ADVANCED YIELD SIMULATION (Feature-wise + Cost–Benefit)
# ---------------------------------------------------------
if "predicted_yield" in st.session_state:

    predicted_yield = st.session_state.predicted_yield
    base_data = st.session_state.input_data.copy()
    crop_name = str(base_data["crop"]).lower()
    msp = CROP_MSP_RS_PER_TON.get(crop_name, 50000)

    st.markdown("## 📈 Advanced Yield Simulation – Feature-wise Impact & Profitability")
    st.info("Adjust one feature at a time to see its impact on yield and approximate profit (per hectare).")

    def simulate_change(updated_data: dict) -> float:
        sim_df = pd.DataFrame([updated_data])
        sim_df[["crop", "soil_texture"]] = sim_df[["crop", "soil_texture"]].astype(str)
        try:
            sim_df = sim_df[st.session_state.model.feature_names_]
        except Exception:
            pass
        return float(st.session_state.model.predict(sim_df)[0])

    def compute_cost(feature_key: str, base_val, new_val) -> float:
        delta = float(new_val) - float(base_val)
        if abs(delta) < 1e-6:
            return 0.0
        cost = 0.0
        if feature_key == "irrigation_events":
            extra_events = max(0.0, delta)
            cost = extra_events * IRRIGATION_COST_PER_EVENT_RS
        elif feature_key in ("soil_N_status_kg_ha", "fert_N_kg_ha"):
            extra_n = max(0.0, delta)
            cost = extra_n * N_COST_PER_KG_RS
        elif feature_key in ("soil_P_status_kg_ha", "fert_P_kg_ha"):
            extra_p = max(0.0, delta)
            cost = extra_p * P_COST_PER_KG_RS
        elif feature_key in ("soil_K_status_kg_ha", "fert_K_kg_ha"):
            extra_k = max(0.0, delta)
            cost = extra_k * K_COST_PER_KG_RS
        elif feature_key == "soil_moisture_pct":
            extra_m = max(0.0, delta)
            eq_irrig = extra_m / 10.0  # approx: +10% moisture ≈ 1 irrigation
            cost = eq_irrig * IRRIGATION_COST_PER_EVENT_RS
        elif feature_key.startswith("ndvi"):
            extra_ndvi = max(0.0, delta)
            cost = extra_ndvi * NDVI_COST_PER_DELTA1_RS
        # other features assumed to have no direct incremental cost modeled here
        return cost

    # key, label, min, max, step, is_float
    feature_configs = [
        ("base_yield_potential_t_ha", "Base Yield Potential (t/ha)", 0.5, 6.0, 0.1, True),
        ("maturity_days", "Maturity Days", 60, 200, 5, False),
        ("mean_temp_gs_C", "Mean Temp (°C)", 5.0, 50.0, 0.5, True),
        ("temp_flowering_C", "Temp at Flowering (°C)", 5.0, 50.0, 0.5, True),
        ("seasonal_rain_mm", "Seasonal Rain (mm)", 0.0, 2500.0, 10.0, True),
        ("rain_flowering_mm", "Rainfall at Flowering (mm)", 0.0, 800.0, 10.0, True),
        ("humidity_mean_pct", "Mean Humidity (%)", 0.0, 100.0, 1.0, False),
        ("soil_pH", "Soil pH", 3.5, 9.5, 0.1, True),
        ("soil_oc_pct", "Soil Organic C (%)", 0.0, 5.0, 0.1, True),
        ("clay_pct", "Clay (%)", 0.0, 70.0, 1.0, False),
        ("soil_depth_cm", "Soil Depth (cm)", 10, 200, 5, False),
        ("soil_N_status_kg_ha", "Soil N (kg/ha)", 0, 400, 5, False),
        ("soil_P_status_kg_ha", "Soil P (kg/ha)", 0, 200, 5, False),
        ("soil_K_status_kg_ha", "Soil K (kg/ha)", 0, 500, 5, False),
        ("fert_N_kg_ha", "Fertilizer N (kg/ha)", 0, 250, 5, False),
        ("fert_P_kg_ha", "Fertilizer P (kg/ha)", 0, 150, 5, False),
        ("fert_K_kg_ha", "Fertilizer K (kg/ha)", 0, 300, 5, False),
        ("irrigation_events", "Irrigation Events (no.)", 0, 20, 1, False),
        ("soil_moisture_pct", "Soil Moisture (%)", 0, 100, 1, False),
        ("seed_moisture_pct", "Seed Moisture at Harvest (%)", 5.0, 25.0, 0.5, True),
        ("ndvi_early", "NDVI Early", 0.0, 1.0, 0.02, True),
        ("ndvi_flowering", "NDVI Flowering", 0.0, 1.0, 0.02, True),
        ("ndvi_peak", "NDVI Peak", 0.0, 1.0, 0.02, True),
        ("ndvi_late", "NDVI Late", 0.0, 1.0, 0.02, True),
        ("ndvi_veg_slope", "NDVI Vegetative Slope", -2.0, 2.0, 0.05, True),
        ("sowing_doy", "Sowing Day of Year", 1, 365, 5, False),
    ]

    impacts = []

    left, right = st.columns(2)
    for idx, (key, label, min_v, max_v, step, is_float) in enumerate(feature_configs):
        col = left if idx % 2 == 0 else right
        with col:
            base_val = base_data.get(key)
            if base_val is None:
                continue

            if is_float:
                base_val_f = float(base_val)
                val = st.slider(
                    label,
                    float(min_v),
                    float(max_v),
                    float(base_val_f),
                    step=float(step),
                    key=f"sim_{key}",
                )
            else:
                base_val_i = int(base_val)
                val = st.slider(
                    label,
                    int(min_v),
                    int(max_v),
                    int(base_val_i),
                    step=int(step),
                    key=f"sim_{key}",
                )

            updated = base_data.copy()
            updated[key] = val
            sim_y = simulate_change(updated)
            delta_y = sim_y - predicted_yield
            cost = compute_cost(key, base_val, val)
            revenue_gain = delta_y * msp
            net_gain = revenue_gain - cost

            st.write(f"Yield: **{sim_y:.2f} t/ha** (Δ {delta_y:+.2f})")
            if abs(revenue_gain) > 0.0 or abs(cost) > 0.0:
                st.caption(
                    f"Approx. revenue Δ ≈ ₹{revenue_gain:,.0f}, "
                    f"cost ≈ ₹{cost:,.0f}, net ≈ ₹{net_gain:,.0f}"
                )

            impacts.append({
                "feature": key,
                "label": label,
                "sim_yield": sim_y,
                "delta_yield": delta_y,
                "revenue_gain": revenue_gain,
                "cost": cost,
                "net_gain": net_gain,
            })

    # Store all impacts for LLM to use later
    st.session_state.simulation_impacts = impacts

    # Top 3 by net gain
    if impacts:
        sorted_impacts = sorted(impacts, key=lambda x: x["net_gain"], reverse=True)
        top3 = sorted_impacts[:3]

        st.markdown("### 🏆 Top 3 High-Impact Interventions (by Net Profit per ha)")
        for i, imp in enumerate(top3, start=1):
            st.write(
                f"**{i}. {imp['label']}** → ΔYield {imp['delta_yield']:+.2f} t/ha, "
                f"Revenue ≈ ₹{imp['revenue_gain']:,.0f}, "
                f"Cost ≈ ₹{imp['cost']:,.0f}, "
                f"Net ≈ ₹{imp['net_gain']:,.0f}"
            )

        # Bar chart of Net Gain for all features
        chart_df = pd.DataFrame(
            {
                "Feature": [imp["label"] for imp in impacts],
                "NetGain_Rs": [imp["net_gain"] for imp in impacts],
            }
        ).sort_values("NetGain_Rs", ascending=False)

        st.markdown("### 📊 Net Profit by Feature (Simulation)")
        st.bar_chart(chart_df.set_index("Feature")["NetGain_Rs"])


# ---------------------------------------------------------
# LLM Advisory AFTER prediction & simulation
# ---------------------------------------------------------
if "predicted_yield" in st.session_state and st.session_state.advisor:

    predicted_yield = st.session_state.predicted_yield
    input_data = st.session_state.input_data
    simulation_impacts = st.session_state.get("simulation_impacts", None)

    alerts = generate_weather_alerts(
        input_data["crop"],
        input_data["temp_flowering_C"],
        input_data["rain_flowering_mm"],
        input_data["humidity_mean_pct"],
    )
    st.subheader("Weather Alerts")
    for a in alerts:
        st.warning(a)

    yield_context = {
        "crop": input_data["crop"],
        "yield": predicted_yield,
        "unit": "t/ha",
        "features": input_data,
        "alerts": alerts,
        "simulation_impacts": simulation_impacts,  # for LLM to use in "best plan"
    }

    with st.spinner("Generating advisory..."):
        advisory = st.session_state.advisor.chat(
            session_id=f"session_{input_data['crop']}",
            farmer_query=farmer_query,
            yield_dict=yield_context,
            language=advisory_language,
        )
        st.write(advisory)

    st.markdown("## 🌍 Global Benchmarking Against Top-Yield Countries")

    st.info(
        "Compare your predicted yield & field features vs global best practices "
        "of top-producing countries for this crop."
    )

    benchmark_button = st.button("🌍 Get Global Benchmarking Advice")

    if benchmark_button:
        benchmark_query = "Use global yield benchmarking PDF to compare with top countries..."
        combined_query = benchmark_query + "\n\nFarmer question:\n" + (farmer_query or "")

        with st.spinner("Generating benchmarking advisory..."):
            global_advisory = st.session_state.advisor.chat(
                session_id=f"session_{input_data['crop']}_benchmark",
                farmer_query=combined_query,
                yield_dict=yield_context,
                language=advisory_language,
            )
            st.write(global_advisory)

elif "predicted_yield" in st.session_state and st.session_state.advisor is None:
    st.error("LLM advisor not initialized.")
