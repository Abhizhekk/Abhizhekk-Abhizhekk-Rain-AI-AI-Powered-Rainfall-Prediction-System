import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
from geopy.geocoders import Nominatim
import plotly.express as px
import plotly.graph_objects as go

# ================================================
# PAGE CONFIG
# ================================================
st.set_page_config(
    page_title="Rain AI - Smart Weather Prediction",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================================
# MODERN GLASSMORPHISM UI STYLES
# ================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Glass card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }
    
    /* Hero section */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(255,255,255,0.3);
    }
    
    .hero-subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    /* Main prediction card */
    .prediction-hero {
        background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(30px);
        border-radius: 32px;
        padding: 3rem;
        text-align: center;
        border: 2px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin: 2rem 0;
    }
    
    .city-name {
        font-size: 2rem;
        color: white;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: capitalize;
    }
    
    .temperature {
        font-size: 5rem;
        font-weight: 700;
        color: white;
        line-height: 1;
        margin: 1rem 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    .rain-probability {
        font-size: 1.5rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
        margin-top: 1.5rem;
        padding: 1rem 2rem;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 50px;
        display: inline-block;
        backdrop-filter: blur(10px);
    }
    
    .probability-high {
        background: rgba(239, 68, 68, 0.2);
        border: 2px solid rgba(239, 68, 68, 0.5);
    }
    
    .probability-medium {
        background: rgba(251, 191, 36, 0.2);
        border: 2px solid rgba(251, 191, 36, 0.5);
    }
    
    .probability-low {
        background: rgba(34, 197, 94, 0.2);
        border: 2px solid rgba(34, 197, 94, 0.5);
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.25);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: translateY(-3px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        color: white;
        font-size: 1.1rem;
        padding: 1rem 1.5rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(255, 255, 255, 0.6);
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
        background: rgba(255, 255, 255, 0.25);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.6);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 1rem 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Chart styling */
    .chart-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-top: 2rem;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: white transparent transparent transparent;
    }
    
    /* Success/Error messages */
    .stAlert {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        color: white;
    }
    
    /* Weather icon animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .weather-icon {
        animation: float 3s ease-in-out infinite;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ================================================
# LOAD MODEL & SCALER
# ================================================
@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model("rainfall_model_v2_balanced.h5")
        scaler = joblib.load("scaler_v2_balanced.save")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, scaler = load_assets()

# ================================================
# WEATHER DATA FETCHING
# ================================================
def fetch_history(lat, lon):
    """Fetch 14-day weather history and current conditions"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,shortwave_radiation",
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
        "past_days": 14,
        "timezone": "auto"
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        
        # Aggregate to daily data
        hist = df.resample("D", on="time").agg({
            "temperature_2m": "mean",
            "relative_humidity_2m": "mean",
            "wind_speed_10m": "max",
            "surface_pressure": "mean",
            "shortwave_radiation": "sum"
        }).tail(14)
        
        # Scale values
        hist["surface_pressure"] /= 10
        hist["shortwave_radiation"] /= 1000
        
        return hist, data["current"]
    except Exception as e:
        st.error(f"Failed to fetch weather data: {e}")
        st.stop()

# ================================================
# HEADER
# ================================================
st.markdown('<h1 class="hero-title"><span class="weather-icon">🌧️</span> Rain AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-Powered Rainfall Prediction with 14-Day Weather Analysis</p>', unsafe_allow_html=True)

# ================================================
# INPUT SECTION
# ================================================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    with st.form("predict_form", clear_on_submit=False):
        city = st.text_input(
            "",
            value="Mumbai",
            placeholder="🌍 Enter city name...",
            label_visibility="collapsed"
        )
        run = st.form_submit_button(
            "🔮 Predict Rainfall",
            use_container_width=True
        )

# ================================================
# PREDICTION LOGIC
# ================================================
if run and city:
    with st.spinner("🔍 Analyzing weather patterns..."):
        # Geocode city
        try:
            geo = Nominatim(user_agent="rain_ai_v2").geocode(city)
            if not geo:
                st.error("❌ City not found. Please check the spelling.")
                st.stop()
            
            lat, lon = geo.latitude, geo.longitude
        except Exception as e:
            st.error(f"❌ Geocoding error: {e}")
            st.stop()
        
        # Fetch weather data
        hist, current = fetch_history(lat, lon)
        
        # Prepare data for model
        model_df = hist.copy()
        model_df.columns = ["temp", "humidity", "wind", "pressure", "solar"]
        
        # Make prediction
        X = scaler.transform(model_df).reshape(1, 14, 5)
        prob = float(model.predict(X, verbose=0)[0][0]) * 100
        
        # Determine probability class
        if prob >= 70:
            prob_class = "probability-high"
            rain_emoji = "🌧️"
            rain_status = "High Chance"
        elif prob >= 40:
            prob_class = "probability-medium"
            rain_emoji = "⛅"
            rain_status = "Moderate Chance"
        else:
            prob_class = "probability-low"
            rain_emoji = "☀️"
            rain_status = "Low Chance"
    
    # ================================================
    # MAIN PREDICTION CARD
    # ================================================
    st.markdown(f"""
    <div class="prediction-hero">
        <div class="city-name">📍 {city.title()}</div>
        <div class="temperature">{current['temperature_2m']:.1f}°C</div>
        <div class="rain-probability {prob_class}">
            {rain_emoji} Rain Probability: <strong>{prob:.1f}%</strong>
            <br><small style="font-size: 0.9rem; opacity: 0.8;">{rain_status} of Rain</small>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ================================================
    # WEATHER METRICS
    # ================================================
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_data = [
        ("💧", "Humidity", f"{current['relative_humidity_2m']:.0f}%"),
        ("💨", "Wind Speed", f"{current['wind_speed_10m']:.1f} km/h"),
        ("🌡️", "Pressure", f"{current['surface_pressure']:.0f} mb"),
        ("🌤️", "Condition", rain_status)
    ]
    
    for col, (icon, label, value) in zip([col1, col2, col3, col4], metrics_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ================================================
    # TEMPERATURE TREND CHART
    # ================================================
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 14-Day Temperature Trend</div>', unsafe_allow_html=True)
    
    trend = hist.copy()
    trend = trend.reset_index()
    trend.columns = ["Date", "Temperature", "Humidity", "Wind", "Pressure", "Solar"]
    
    fig_temp = go.Figure()
    
    fig_temp.add_trace(go.Scatter(
        x=trend["Date"],
        y=trend["Temperature"],
        mode='lines+markers',
        name='Temperature',
        line=dict(color='rgba(255, 255, 255, 0.8)', width=3, shape='spline'),
        marker=dict(size=8, color='white', line=dict(color='rgba(102, 126, 234, 0.8)', width=2)),
        fill='tozeroy',
        fillcolor='rgba(255, 255, 255, 0.1)'
    ))
    
    fig_temp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            title='Date'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            title='Temperature (°C)'
        ),
        hovermode='x unified',
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # ================================================
    # HUMIDITY & PRESSURE CHART
    # ================================================
    st.markdown('<div class="section-title">🌊 Humidity & Pressure Analysis</div>', unsafe_allow_html=True)
    
    fig_multi = go.Figure()
    
    fig_multi.add_trace(go.Scatter(
        x=trend["Date"],
        y=trend["Humidity"],
        name='Humidity (%)',
        line=dict(color='rgba(147, 197, 253, 0.8)', width=2),
        marker=dict(size=6)
    ))
    
    fig_multi.add_trace(go.Scatter(
        x=trend["Date"],
        y=trend["Pressure"],
        name='Pressure (mb/10)',
        line=dict(color='rgba(252, 211, 77, 0.8)', width=2),
        marker=dict(size=6),
        yaxis='y2'
    ))
    
    fig_multi.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            title='Date'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            title='Humidity (%)'
        ),
        yaxis2=dict(
            showgrid=False,
            overlaying='y',
            side='right',
            title='Pressure (mb/10)'
        ),
        hovermode='x unified',
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_multi, use_container_width=True)
    
    # ================================================
    # FOOTER INFO
    # ================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <small style="color: rgba(255,255,255,0.7);">
            🤖 Powered by AI • 📍 Location: {lat:.4f}, {lon:.4f} • 🕒 Last Updated: Now
        </small>
    </div>
    """, unsafe_allow_html=True)

elif run and not city:
    st.warning("⚠️ Please enter a city name")

# ================================================
# INITIAL STATE MESSAGE
# ================================================
if not run:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 3rem;">
        <h3 style="color: white; margin-bottom: 1rem;">🌤️ Welcome to Rain AI</h3>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem;">
            Enter any city name above to get AI-powered rainfall predictions<br>
            based on 14 days of weather pattern analysis.
        </p>
        <br>
        <p style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">
            ✨ Powered by TensorFlow & Real-time Weather Data
        </p>
    </div>
    """, unsafe_allow_html=True)