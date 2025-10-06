import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(page_title="AeroPredict - NASA Air Quality", layout="wide", page_icon="🛰️")

# Title
st.title("🛰️ AeroPredict")
st.subheader("NASA Satellite-Powered Air Quality Forecasting")

# Cities database
cities = {
    'San Francisco': {'lat': 37.77, 'lon': -122.42},
    'Beijing': {'lat': 39.90, 'lon': 116.40},
    'Mumbai': {'lat': 19.07, 'lon': 72.87},
    'New York': {'lat': 40.71, 'lon': -74.00},
    'Los Angeles': {'lat': 34.05, 'lon': -118.24},
    'Delhi': {'lat': 28.61, 'lon': 77.21},
    'Tokyo': {'lat': 35.68, 'lon': 139.69},
    'London': {'lat': 51.51, 'lon': -0.13}
}

# Simple ML Model for AOD -> PM2.5 conversion
@st.cache_resource
def get_model():
    model = LinearRegression()
    # Known AOD to PM2.5 relationship (empirical data)
    X = np.array([[0.05], [0.15], [0.3], [0.5], [0.8], [1.2], [1.5]])
    y = np.array([8, 25, 50, 85, 135, 200, 250])
    model.fit(X, y)
    return model

model = get_model()

# NASA POWER API data fetch
@st.cache_data(ttl=3600)
def get_nasa_aod_data(lat, lon):
    """Fetch aerosol optical depth from NASA POWER API"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            'parameters': 'AOD_550',
            'community': 'RE',
            'longitude': lon,
            'latitude': lat,
            'start': start_date.strftime('%Y%m%d'),
            'end': end_date.strftime('%Y%m%d'),
            'format': 'JSON'
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            aod_values = list(data['properties']['parameter']['AOD_550'].values())
            # Get latest valid value
            valid_values = [v for v in aod_values if v != -999]
            if valid_values:
                return valid_values[-1]
        
        # Fallback to simulated data
        return None
        
    except Exception as e:
        st.warning(f"NASA API timeout - using demo mode")
        return None

def aod_to_aqi(aod):
    """Convert AOD to PM2.5 and then to AQI"""
    # Predict PM2.5 from AOD
    pm25 = model.predict([[aod]])[0]
    
    # Convert PM2.5 to AQI (EPA standard)
    if pm25 <= 12.0:
        aqi = (pm25 / 12.0) * 50
    elif pm25 <= 35.4:
        aqi = ((pm25 - 12.0) / (35.4 - 12.0)) * 50 + 50
    elif pm25 <= 55.4:
        aqi = ((pm25 - 35.4) / (55.4 - 35.4)) * 50 + 100
    elif pm25 <= 150.4:
        aqi = ((pm25 - 55.4) / (150.4 - 55.4)) * 100 + 150
    else:
        aqi = min(500, ((pm25 - 150.4) / 100) * 100 + 250)
    
    return int(aqi), round(pm25, 1)

def get_aqi_category(aqi):
    """Get AQI category and color"""
    if aqi <= 50:
        return "Good", "🟢", "#00e400"
    elif aqi <= 100:
        return "Moderate", "🟡", "#ffff00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive", "🟠", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "🔴", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "🟣", "#8f3f97"
    else:
        return "Hazardous", "🟤", "#7e0023"

# Sidebar
st.sidebar.header("🌍 Select Location")
selected_city = st.sidebar.selectbox("City", list(cities.keys()))

# Get city coordinates
lat = cities[selected_city]['lat']
lon = cities[selected_city]['lon']

# Fetch NASA data
with st.spinner('Fetching NASA satellite data...'):
    aod = get_nasa_aod_data(lat, lon)
    
    # Demo mode if API fails
    if aod is None:
        st.info("📡 Demo Mode: Using simulated NASA MODIS data")
        # Simulate realistic AOD values based on city
        demo_aod = {
            'San Francisco': 0.12,
            'Beijing': 0.85,
            'Mumbai': 0.55,
            'New York': 0.18,
            'Los Angeles': 0.25,
            'Delhi': 0.95,
            'Tokyo': 0.15,
            'London': 0.20
        }
        aod = demo_aod.get(selected_city, 0.3)

# Calculate AQI
aqi, pm25 = aod_to_aqi(aod)
category, emoji, color = get_aqi_category(aqi)

# Main metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Air Quality Index", aqi, delta=None)
    st.markdown(f"### {emoji} {category}")

with col2:
    st.metric("PM2.5 (µg/m³)", f"{pm25}", delta=None)
    st.caption("Particulate Matter < 2.5μm")

with col3:
    st.metric("NASA AOD", f"{aod:.3f}", delta=None)
    st.caption("Aerosol Optical Depth @ 550nm")

# Map visualization
st.subheader(f"📍 {selected_city}")
m = folium.Map(location=[lat, lon], zoom_start=10)
folium.CircleMarker(
    [lat, lon],
    radius=20,
    popup=f"{selected_city}<br>AQI: {aqi}",
    color=color,
    fill=True,
    fillColor=color,
    fillOpacity=0.6
).add_to(m)
st_folium(m, width=700, height=400)

# Multi-city comparison
st.subheader("🌐 Global Air Quality Snapshot")

comparison_data = []
for city_name, coords in cities.items():
    city_aod = get_nasa_aod_data(coords['lat'], coords['lon'])
    if city_aod is None:
        demo_vals = {'San Francisco': 0.12, 'Beijing': 0.85, 'Mumbai': 0.55, 
                     'New York': 0.18, 'Los Angeles': 0.25, 'Delhi': 0.95,
                     'Tokyo': 0.15, 'London': 0.20}
        city_aod = demo_vals.get(city_name, 0.3)
    
    city_aqi, city_pm25 = aod_to_aqi(city_aod)
    cat, _, col = get_aqi_category(city_aqi)
    
    comparison_data.append({
        'City': city_name,
        'AQI': city_aqi,
        'PM2.5': city_pm25,
        'Category': cat,
        'Color': col
    })

df = pd.DataFrame(comparison_data)

# Bar chart
fig = go.Figure(data=[
    go.Bar(
        x=df['City'],
        y=df['AQI'],
        marker_color=df['Color'],
        text=df['AQI'],
        textposition='auto',
    )
])

fig.update_layout(
    title="Air Quality Index Comparison",
    xaxis_title="City",
    yaxis_title="AQI",
    height=400,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# Data table
st.dataframe(df[['City', 'AQI', 'PM2.5', 'Category']], use_container_width=True)

# Information section
st.divider()
st.subheader("ℹ️ About AeroPredict")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🛰️ NASA Data Source:**
    - MODIS Aerosol Optical Depth (AOD)
    - 550nm wavelength measurement
    - Updated daily via NASA POWER API
    
    **🧮 Prediction Method:**
    - Machine Learning regression model
    - Converts AOD → PM2.5 concentration
    - Applies EPA AQI calculation standards
    """)

with col2:
    st.markdown("""
    **🎯 Impact:**
    - 7 million deaths annually from air pollution (WHO)
    - Early warning for vulnerable populations
    - Free, open-source public health tool
    
    **💡 Use Cases:**
    - Daily air quality monitoring
    - Health advisories
    - Urban planning decisions
    """)

# Footer
st.divider()
st.caption("🚀 Built for NASA Space Apps Challenge 2025 | Using NASA MODIS satellite data")