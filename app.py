import joblib
import numpy as np
import pandas as pd
import streamlit as st

import requests
import time
from datetime import datetime
from zoneinfo import ZoneInfo

def aqi_level(aqi):
    levels = [
        (50, "😊", "Good", "Air quality is fine for outdoor activity."),
        (100, "🙂", "Moderate", "Fine for most people. Sensitive people may want shorter outdoor workouts."),
        (150, "😷", "Unhealthy for sensitive groups", "Children, older adults and people with asthma should limit long time outside."),
        (200, "🤒", "Unhealthy", "Reduce outdoor exercise and consider a mask."),
        (300, "🚨", "Very unhealthy", "Stay indoors where possible and keep windows closed."),
    ]
    for limit, emoji, name, advice in levels:
        if aqi <= limit:
            return emoji, name, advice
    return "⚠️", "Hazardous", "Avoid going outside."

def get_astana_weather():
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": 51.1694,
        "longitude": 71.4491,
        "daily": "temperature_2m_min,temperature_2m_max,wind_speed_10m_max",
        "hourly": "surface_pressure",
        "timezone": "Asia/Almaty",
        "forecast_days": 1,
        "wind_speed_unit": "kmh",
    }

    for attempt in range(3):
        try:
            response = requests.get(
                url, params=params, timeout=15
            )
            response.raise_for_status()
            data = response.json()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2)

    today = datetime.now(ZoneInfo("Asia/Almaty")).date().isoformat()

    # Find today's index in the returned daily data
    dates = data["daily"]["time"]
    if today not in dates:
        raise ValueError("Today's weather data is unavailable.")

    i = dates.index(today)

    # Today's hourly pressure readings
    hourly = data["hourly"]
    pressure_values = [
        p for t, p in zip(hourly["time"], hourly["surface_pressure"])
        if t.startswith(today) and p is not None
    ]

    return {
        "date": today,
        "minimum_temperature_C":
            data["daily"]["temperature_2m_min"][i],
        "maximum_temperature_C":
            data["daily"]["temperature_2m_max"][i],
        "maximum_wind_speed_kmh":
            data["daily"]["wind_speed_10m_max"][i],
        "pressure_hPa":
            pressure_values[-1] if pressure_values else None,
    }

model = joblib.load("model.pkl")

st.title("Astana air quality estimator")
st.write(
    "Estimates daily AQI from weather variables. "
    "This is a demo, not an official forecast."
)

st.header("Astana weather")

if "weather" not in st.session_state:
    st.session_state.weather = None

if st.button("Get today's weather"):
    try:
        weather = get_astana_weather()
        st.session_state["weather"] = weather

        st.session_state["tmin"] = min(30.0, max(-40.0, float(weather["minimum_temperature_C"])))
        st.session_state["tmax"] = min(40.0, max(-35.0, float(weather["maximum_temperature_C"])))
        st.session_state["wspd"] = min(50.0, max(0.0, float(weather["maximum_wind_speed_kmh"])))

        pressure = weather["pressure_hPa"]

        if pressure is not None:
            st.session_state["pres"] = min(1060.0, max(980.0, float(pressure)))

    except (requests.RequestException, ValueError, KeyError) as e:
        st.error(f"Could not retrieve weather data: {e}")

weather = st.session_state["weather"]

if weather is not None:
    st.caption(f"Date: {weather['date']}")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Minimum temperature", f"{weather['minimum_temperature_C']} °C")

    with col2:
        st.metric("Maximum temperature", f"{weather['maximum_temperature_C']} °C")

    with col3:
        st.metric("Maximum wind speed", f'{weather["maximum_wind_speed_kmh"]} km/h')

    with col4:
        pressure = weather["pressure_hPa"]
        st.metric("Atmospheric pressure", f"{pressure} hPa" if pressure is not None else "Unavailable")


tmin = st.slider("Minimum temperature (°C)", -40.0, 30.0, step=0.1, key="tmin")
tmax = st.slider("Maximum temperature (°C)", -35.0, 40.0, step=0.1, key="tmax")
wspd = st.slider("Wind speed (km/h)", 0.0, 50.0, step=0.1, key="wspd")
pres = st.slider("Pressure (hPa)", 980.0, 1060.0, step=0.1, key="pres")

month = st.selectbox("Month", list(range(1, 13)))

if st.button("Estimate AQI"):
    if tmax < tmin:
        st.warning("Maximum temperature must not be below the minimum.")
    else:
        cols = ["tmin", "tmax", "wspd", "pres", "month_sin", "month_cos"]
        row = [[tmin, tmax, wspd, pres, np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]]
        aqi = model.predict(pd.DataFrame(row, columns=cols))[0]
        st.metric("Estimated AQI", f"{aqi:.0f}")
        emoji, name, advice = aqi_level(aqi)
        st.subheader(f"{emoji} {name}")
        st.write(advice)
        st.caption("Typical error: about 16 AQI points. Data: US Embassy monitor via aqicn.org (World Air Quality Index project), not validated.")