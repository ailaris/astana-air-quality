import time
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

TZ = ZoneInfo("Asia/Almaty")

TEXT = {
    "en": {
        "title": "Astana air quality estimator",
        "intro": "Estimates daily AQI from weather variables. This is a demo, not an official forecast.",
        "weather_header": "Astana weather",
        "get_weather": "Get today's weather",
        "date": "Date",
        "m_min": "Minimum temperature",
        "m_max": "Maximum temperature",
        "m_wind": "Average wind speed",
        "m_pres": "Sea-level pressure",
        "u_wind": "km/h",
        "u_pres": "hPa",
        "unavailable": "Unavailable",
        "min_temp": "Minimum temperature (°C)",
        "max_temp": "Maximum temperature (°C)",
        "wind": "Average wind speed (km/h)",
        "pressure": "Sea-level pressure (hPa)",
        "month": "Month",
        "estimate": "Estimate AQI",
        "est_aqi": "Estimated AQI",
        "warn_range": "Maximum temperature must not be below the minimum.",
        "error_weather": "Could not retrieve weather data",
        "caption": (
            "Typical error: about 16 AQI points. The model tends to underestimate the worst days. "
            "Data: US Embassy monitor via aqicn.org (World Air Quality Index project), not validated."
        ),
        "level_names": [
            "Good",
            "Moderate",
            "Unhealthy for sensitive groups",
            "Unhealthy",
            "Very unhealthy",
            "Hazardous",
        ],
        "level_advice": [
            "Air quality is fine for outdoor activity.",
            "Fine for most people. Sensitive people may want shorter outdoor workouts.",
            "Children, older adults and people with asthma should limit long time outside.",
            "Reduce outdoor exercise and consider a mask.",
            "Stay indoors where possible and keep windows closed.",
            "Avoid going outside.",
        ],
    },
    "ru": {
        "title": "Оценка качества воздуха в Астане",
        "intro": "Оценивает суточный индекс качества воздуха (AQI) по погоде. Это демо, а не официальный прогноз.",
        "weather_header": "Погода в Астане",
        "get_weather": "Получить погоду на сегодня",
        "date": "Дата",
        "m_min": "Минимальная температура",
        "m_max": "Максимальная температура",
        "m_wind": "Средняя скорость ветра",
        "m_pres": "Давление на уровне моря",
        "u_wind": "км/ч",
        "u_pres": "гПа",
        "unavailable": "Нет данных",
        "min_temp": "Минимальная температура (°C)",
        "max_temp": "Максимальная температура (°C)",
        "wind": "Средняя скорость ветра (км/ч)",
        "pressure": "Давление на уровне моря (гПа)",
        "month": "Месяц",
        "estimate": "Оценить AQI",
        "est_aqi": "Оценка AQI",
        "warn_range": "Максимальная температура не может быть ниже минимальной.",
        "error_weather": "Не удалось получить данные о погоде",
        "caption": (
            "Типичная ошибка: около 16 пунктов AQI. Модель склонна занижать оценку в самые плохие дни. "
            "Данные: монитор посольства США через aqicn.org (World Air Quality Index project), не проверены."
        ),
        "level_names": [
            "Хорошо",
            "Умеренно",
            "Вредно для чувствительных групп",
            "Вредно",
            "Очень вредно",
            "Опасно",
        ],
        "level_advice": [
            "Качество воздуха хорошее, можно спокойно гулять на улице.",
            "Приемлемо для большинства людей. Чувствительным людям лучше сократить долгие нагрузки на улице.",
            "Детям, пожилым людям и людям с астмой стоит ограничить длительное пребывание на улице.",
            "Сократите физическую активность на улице и подумайте о маске.",
            "По возможности оставайтесь дома и держите окна закрытыми.",
            "Избегайте выхода на улицу.",
        ],
    },
    "kk": {
        "title": "Астанадағы ауа сапасын бағалау",
        "intro": "Ауа райы бойынша ауа сапасының тәуліктік индексін (AQI) бағалайды. Бұл демонстрациялық нұсқа, ресми болжам емес.",
        "weather_header": "Астанадағы ауа райы",
        "get_weather": "Бүгінгі ауа райын алу",
        "date": "Күні",
        "m_min": "Ең төмен температура",
        "m_max": "Ең жоғары температура",
        "m_wind": "Желдің орташа жылдамдығы",
        "m_pres": "Теңіз деңгейіндегі қысым",
        "u_wind": "км/сағ",
        "u_pres": "гПа",
        "unavailable": "Дерек жоқ",
        "min_temp": "Ең төмен температура (°C)",
        "max_temp": "Ең жоғары температура (°C)",
        "wind": "Желдің орташа жылдамдығы (км/сағ)",
        "pressure": "Теңіз деңгейіндегі қысым (гПа)",
        "month": "Ай",
        "estimate": "AQI бағалау",
        "est_aqi": "Бағаланған AQI",
        "warn_range": "Ең жоғары температура ең төменнен төмен болмауы керек.",
        "error_weather": "Ауа райы деректерін алу мүмкін болмады",
        "caption": (
            "Әдеттегі қате: шамамен 16 AQI тармағы. Модель ең нашар күндерді төмен бағалауға бейім. "
            "Деректер: АҚШ елшілігінің мониторы, aqicn.org (World Air Quality Index project) арқылы, тексерілмеген."
        ),
        "level_names": [
            "Жақсы",
            "Орташа",
            "Сезімтал топтар үшін зиянды",
            "Зиянды",
            "Өте зиянды",
            "Қауіпті",
        ],
        "level_advice": [
            "Ауа сапасы жақсы, далада еркін серуендеуге болады.",
            "Көпшілікке жарайды. Сезімтал адамдар далада ұзақ жаттығудан аулақ болғаны жөн.",
            "Балалар, егде жастағы адамдар және астмасы бар адамдар далада ұзақ болмауы керек.",
            "Далада жаттығуды азайтыңыз, маска тағуды ойластырыңыз.",
            "Мүмкіндігінше үйде болыңыз, терезелерді жауып қойыңыз.",
            "Далаға шықпаңыз.",
        ],
    },
}

LANG_NAMES = {"English": "en", "Русский": "ru", "Қазақша": "kk"}

LEVELS = [(50, "😊"), (100, "🙂"), (150, "😷"), (200, "🤒"), (300, "🚨")]

FEATURE_COLUMNS = ["tmin", "tmax", "wspd", "pres", "month_sin", "month_cos"]


def aqi_level(aqi, T):
    for i, (limit, emoji) in enumerate(LEVELS):
        if aqi <= limit:
            return emoji, T["level_names"][i], T["level_advice"][i]
    return "⚠️", T["level_names"][5], T["level_advice"][5]


def clamp(value, low, high):
    return min(high, max(low, float(value)))


def get_astana_weather():
    """Today's weather for Astana. The model was trained on daily mean wind speed
    and mean sea-level pressure, so we request those daily values."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 51.1694,
        "longitude": 71.4491,
        "daily": "temperature_2m_min,temperature_2m_max,wind_speed_10m_mean,pressure_msl_mean",
        "timezone": "Asia/Almaty",
        "forecast_days": 1,
        "wind_speed_unit": "kmh",
    }

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2)

    today = datetime.now(TZ).date().isoformat()
    daily = data["daily"]
    if today not in daily["time"]:
        raise ValueError("Today's weather data is unavailable.")
    i = daily["time"].index(today)

    return {
        "date": today,
        "tmin": daily["temperature_2m_min"][i],
        "tmax": daily["temperature_2m_max"][i],
        "wspd": daily["wind_speed_10m_mean"][i],
        "pres": daily["pressure_msl_mean"][i],
    }


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


st.set_page_config(page_title="Astana air quality", page_icon="🌫️")

lang_name = st.radio("Language / Язык / Тіл", list(LANG_NAMES), horizontal=True)
T = TEXT[LANG_NAMES[lang_name]]

model = load_model()

DEFAULTS = {
    "tmin": -5.0,
    "tmax": 3.0,
    "wspd": 10.0,
    "pres": 1015.0,
    "month": datetime.now(TZ).month,
    "weather": None,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

st.title(T["title"])
st.write(T["intro"])

st.header(T["weather_header"])

if st.button(T["get_weather"]):
    try:
        weather = get_astana_weather()
        st.session_state["weather"] = weather
        st.session_state["tmin"] = clamp(weather["tmin"], -40.0, 30.0)
        st.session_state["tmax"] = clamp(weather["tmax"], -35.0, 40.0)
        st.session_state["wspd"] = clamp(weather["wspd"], 0.0, 50.0)
        if weather["pres"] is not None:
            st.session_state["pres"] = clamp(weather["pres"], 980.0, 1060.0)
        st.session_state["month"] = datetime.now(TZ).month
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        st.error(f"{T['error_weather']}: {e}")

weather = st.session_state["weather"]
if weather is not None:
    st.caption(f"{T['date']}: {weather['date']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(T["m_min"], f"{weather['tmin']} °C")
    c2.metric(T["m_max"], f"{weather['tmax']} °C")
    c3.metric(T["m_wind"], f"{weather['wspd']} {T['u_wind']}")
    shown_pressure = weather["pres"]
    c4.metric(
        T["m_pres"],
        f"{shown_pressure} {T['u_pres']}" if shown_pressure is not None else T["unavailable"],
    )

tmin = st.number_input(T["min_temp"], min_value=-40.0, max_value=30.0, step=0.1, format="%.1f", key="tmin")
tmax = st.number_input(T["max_temp"], min_value=-35.0, max_value=40.0, step=0.1, format="%.1f", key="tmax")
wspd = st.number_input(T["wind"], min_value=0.0, max_value=50.0, step=0.1, format="%.1f", key="wspd")
pres = st.number_input(T["pressure"], min_value=980.0, max_value=1060.0, step=0.1, format="%.1f", key="pres")
month = st.selectbox(T["month"], list(range(1, 13)), key="month")

if st.button(T["estimate"]):
    if tmax < tmin:
        st.warning(T["warn_range"])
    else:
        row = [[tmin, tmax, wspd, pres, np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]]
        aqi = model.predict(pd.DataFrame(row, columns=FEATURE_COLUMNS))[0]
        st.metric(T["est_aqi"], f"{aqi:.0f}")
        emoji, name, advice = aqi_level(aqi, T)
        st.subheader(f"{emoji} {name}")
        st.write(advice)
        st.caption(T["caption"])