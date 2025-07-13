# Updated 4:22pm, made formatting changes
"""
Mount Rainier Weather Prediction Web Application

This is the main web interface for the Mount Rainier Weather Prediction Tool.
It provides climbers with an easy-to-use dashboard to check weather forecasts
and safety assessments for climbing Mount Rainier.

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path
import re
import subprocess

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

# Now import the modules
from config.config import *
from utils import (
    calculate_wind_chill, format_forecast_date, generate_forecast_hours,
    round_to_significant_figures
)
from model_training import WeatherModelTrainer
from risk_assessment import RiskAssessor
from feature_engineering import FeatureEngineer

# Constants
FORECAST_HOURS = 72

# Page configuration
st.set_page_config(
    page_title="Mount Rainier Summit Forecast",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look with navigation
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: 0.03em;
    }
    .main-subtitle {
        font-size: 1.3rem;
        color: #444;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
        letter-spacing: 0.01em;
    }
    .metric-card-row {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2.5rem;
        justify-content: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 60%, #e3f2fd 100%);
        box-shadow: 0 2px 8px rgba(30, 64, 175, 0.07);
        border-radius: 1rem;
        padding: 1.5rem 2.5rem;
        min-width: 180px;
        text-align: center;
        border: 1px solid #e3e3e3;
    }
    .metric-label {
        font-size: 1.1rem;
        color: #1976d2;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #222;
    }
    .stPlotlyChart {
        margin-bottom: 2.5rem;
    }
    .stExpander {
        margin-top: 2rem;
    }
    .about-section, .tech-card, .feature-card {
        background: linear-gradient(135deg, #222 0%, #1a1a1a 100%);
        color: #fff !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.12);
        border-left: 5px solid #1f77b4;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .tech-stack {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    .tech-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.07);
        border-left: 4px solid #1f77b4;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    .feature-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.07);
        border-top: 4px solid #43a047;
    }
    .disclaimer-section {
        background: linear-gradient(135deg, #4e342e 0%, #3e2723 100%);
        color: #fff !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.12);
        border-left: 5px solid #ffb300;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)



def show_about_page():
    """Display the About & How It Works page with images and better visual presentation"""
    st.markdown('<h1 class="main-header">ℹ️ About & How It Works</h1>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">A comprehensive weather prediction and safety assessment system for Mount Rainier</div>', unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("""
    <div class="about-section">
        <h2>🏔️ Project Overview</h2>
        <p>This is a comprehensive weather prediction and safety assessment system specifically designed for Mount Rainier summit climbing. 
        It combines multiple data sources with advanced machine learning to provide accurate, elevation-adjusted forecasts that help climbers 
        make informed safety decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Sources Section
    st.markdown('<h2>📊 Data Sources</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="tech-card">
            <h3>🌍 Open-Meteo API</h3>
            <p>Real-time weather data with elevation corrections for Mount Rainier's summit (14,411ft). 
            Provides hourly updates with high accuracy for temperature, wind, pressure, and precipitation.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tech-card">
            <h3>🛰️ ERA5 Satellite Data</h3>
            <p>Global reanalysis data from Copernicus Climate Data Store. 
            Combines satellite observations, weather station data, and computer models for comprehensive coverage.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tech-card">
            <h3>🏔️ Camp Muir Station</h3>
            <p>Ground truth data from 10,000ft elevation weather station on Mount Rainier. 
            Provides highly accurate local measurements for validation and calibration.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tech-card">
            <h3>🌤️ NWS Forecasts</h3>
            <p>National Weather Service predictions integrated for additional verification. 
            Ensures alignment with official weather forecasts and emergency alerts.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # --- Machine Learning Approach Section ---
    st.markdown("""
    <div class="about-section">
        <h2>Machine Learning Approach</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-cards-grid">
        <div class="feature-card" style="border-top: 4px solid #43a047;">
            <h3>🎯 XGBoost Models</h3>
            <p>Separate regression models for each weather variable (temperature, wind, pressure, precipitation). Trained on historical data with time series cross-validation for robust performance.</p>
        </div>
        <div class="feature-card" style="border-top: 4px solid #43a047;">
            <h3>🛠️ Feature Engineering</h3>
            <p>100+ derived features including time patterns, weather interactions, and rolling statistics. Captures complex relationships that simple weather models miss.</p>
        </div>
        <div class="feature-card" style="border-top: 4px solid #43a047;">
            <h3>🌐 Hybrid Forecasting</h3>
            <p>Combines ML predictions with trend analysis and multiple data sources. Provides more accurate forecasts than single-source approaches.</p>
        </div>
        <div class="feature-card" style="border-top: 4px solid #43a047;">
            <h3>⚠️ Risk Assessment</h3>
            <p>Multi-factor safety scoring system that evaluates wind, temperature, pressure, and precipitation. Provides clear risk levels and climbing recommendations.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Add or update CSS for 2x2 grid layout
    st.markdown("""
    <style>
    .feature-cards-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 2rem;
        margin: 2rem 0;
    }
    @media (max-width: 900px) {
        .feature-cards-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Technical Stack Section ---
    st.markdown("""
    <div class="about-section">
        <h2>Technical Stack</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="tech-card" style="border-left: 4px solid #43a047;">
        <h3>🐍 Backend</h3>
        <ul>
            <li>Python 3.8+</li>
            <li>Pandas & NumPy</li>
            <li>XGBoost</li>
            <li>Scikit-learn</li>
        </ul>
    </div>
    <div class="tech-card">
        <h3>🌐 Frontend</h3>
        <ul>
            <li>Streamlit</li>
            <li>Plotly</li>
            <li>HTML/CSS</li>
            <li>JavaScript</li>
        </ul>
    </div>
    <div class="tech-card">
        <h3>📊 Data Processing</h3>
        <ul>
            <li>Feature Engineering</li>
            <li>Time Series Analysis</li>
            <li>API Integration</li>
            <li>Data Validation</li>
        </ul>
    </div>
    <div class="tech-card">
        <h3>🏗️ Architecture</h3>
        <ul>
            <li>Modular Design</li>
            <li>Error Handling</li>
            <li>Caching</li>
            <li>Production Ready</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features
    st.markdown('<h2>🎯 Key Features</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>⏰ 72-Hour Forecasts</h3>
            <p>Hourly granularity predictions for temperature, wind, pressure, and precipitation. 
            Covers the critical planning window for summit attempts.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🏔️ Elevation-Adjusted</h3>
            <p>Predictions specifically calibrated for Mount Rainier's 14,411ft summit. 
            Accounts for altitude effects on temperature, pressure, and wind patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>⚠️ Real-Time Safety</h3>
            <p>Continuous risk assessment with traffic light system (Green/Yellow/Red). 
            Identifies dangerous conditions and optimal climbing windows.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Interactive Visualizations</h3>
            <p>Dynamic charts showing weather trends, risk timelines, and climbing windows. 
            Exportable data for offline planning and analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # --- Developer Skills Demonstrated Section ---
    st.markdown("""
    <div class="about-section">
        <h2>👨‍💻 Developer Skills Demonstrated</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="skills-grid">
        <div class="skill-card">
            <div class="skill-icon">🧠</div>
            <div class="skill-title">Machine Learning</div>
            <div class="skill-desc">Model development, training, validation, and deployment. Feature engineering, hyperparameter tuning, and performance evaluation.</div>
        </div>
        <div class="skill-card">
            <div class="skill-icon">🛠️</div>
            <div class="skill-title">Data Engineering</div>
            <div class="skill-desc">API integration, data cleaning, feature creation, and time series analysis. Handles multiple data sources with different formats and quality levels.</div>
        </div>
        <div class="skill-card">
            <div class="skill-icon">🎨</div>
            <div class="skill-title">User Experience</div>
            <div class="skill-desc">Intuitive interface design, responsive layouts, and accessibility. Focus on user needs and safety-critical information presentation.</div>
        </div>
        <div class="skill-card">
            <div class="skill-icon">🏭</div>
            <div class="skill-title">Production Quality</div>
            <div class="skill-desc">Error handling, logging, caching, and modular architecture. Code is maintainable, testable, and ready for production deployment.</div>
        </div>
    </div>
    <style>
    .skills-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
        gap: 2rem;
        margin: 2.5rem 0 2rem 0;
    }
    .skill-card {
        background: linear-gradient(135deg, #222 0%, #1a1a1a 100%);
        color: #fff;
        border-left: 5px solid #1f77b4;
        border-radius: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.12);
        padding: 2rem 1.5rem 1.5rem 1.5rem;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        min-height: 180px;
        transition: box-shadow 0.2s;
    }
    .skill-card:hover {
        box-shadow: 0 4px 24px rgba(31,119,180,0.18);
        border-left: 5px solid #43a047;
    }
    .skill-icon {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .skill-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        letter-spacing: 0.01em;
    }
    .skill-desc {
        font-size: 1rem;
        color: #e0e0e0;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Safety Disclaimer
    st.markdown("""
    <div class="disclaimer-section">
        <h3>⚠️ Important Safety Disclaimer</h3>
        <p><strong>This tool provides predictions only and should not be the sole factor in climbing decisions.</strong></p>
        <ul>
            <li>Always check official weather forecasts from the National Weather Service</li>
            <li>Consult with experienced mountain guides and rangers</li>
            <li>Assess current conditions on the mountain before attempting to climb</li>
            <li>Be prepared to turn back if conditions deteriorate</li>
            <li>Have proper equipment, training, and emergency plans</li>
        </ul>
        <p><strong>Remember:</strong> The mountain will always be there. Make sure you are too.</p>
    </div>
    """, unsafe_allow_html=True)

def show_forecast_page():
    """Display the main forecast page"""
    # Main content for forecast page
    st.markdown('<h1 class="main-header">🏔️ Mount Rainier Summit Forecast</h1>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">A machine learning-powered weather and safety dashboard for climbers</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Forecast Settings")
    
    # Date selection
    forecast_date = st.sidebar.date_input(
        "Select Forecast Date",
        value=datetime.now().date(),
        min_value=datetime.now().date(),
        max_value=(datetime.now() + timedelta(days=7)).date()
    )
    
    # Time selection
    forecast_time = st.sidebar.time_input(
        "Select Forecast Time",
        value=datetime.now().time()
    )
    
    # Combine date and time
    forecast_datetime = datetime.combine(forecast_date, forecast_time)
    
    # Model selection - Open-Meteo as default
    model_option = st.sidebar.selectbox(
        "Data Source",
        ["Open-Meteo (Fresh Data)", "Machine Learning Model Forecast (ML + Trends)", "Sample Data (Demo)", "Trained Models"],
        index=0,  # Make Open-Meteo the default
        help="Choose between Open-Meteo for fresh data, ML forecast, sample data, or trained models"
    )
    
    # Main content
    if model_option == "Open-Meteo (Fresh Data)":
        st.info("🔮 Using Open-Meteo for fresh weather data.")
        refresh_openmeteo_data()
        forecast_data = load_weather_data()
        # Apply small correction using trained model
        try:
            trainer = load_models()
            feature_engineer = FeatureEngineer()
            # Generate model prediction for the same timestamps
            model_pred = generate_model_forecast(trainer, feature_engineer, forecast_data.index[0], hours=len(forecast_data))
            # Align indices
            model_pred = model_pred.reindex(forecast_data.index, method='nearest')
            # Apply a small blend: 90% Open-Meteo, 10% model delta
            for var in ['temperature_F', 'wind_speed_mph', 'air_pressure_hPa', 'precip_hourly']:
                if var in forecast_data.columns and var in model_pred.columns:
                    delta = model_pred[var] - forecast_data[var]
                    forecast_data[var] = forecast_data[var] + 0.1 * delta
        except Exception as e:
            st.warning(f"⚠️ Could not apply model correction: {e}")
        # Create risk assessment
        risk_assessor = RiskAssessor()
        risk_assessment = risk_assessor.assess_climbing_safety(forecast_data)
        
    elif model_option == "Machine Learning Model Forecast (ML + Trends)":
        st.info("🔮 Using Machine Learning Model Forecast: ML baseline + trend analysis for improved accuracy.")
        
        try:
            # Generate hybrid forecast
            forecast_data = generate_hybrid_forecast(forecast_datetime.replace(minute=0, second=0, microsecond=0), hours=72)
            
            if forecast_data is not None:
                # Create risk assessment
                risk_assessor = RiskAssessor()
                risk_assessment = risk_assessor.assess_climbing_safety(forecast_data)
            else:
                st.error("❌ Failed to generate ML forecast")
                st.info("Falling back to Open-Meteo data.")
                refresh_openmeteo_data()
                forecast_data = load_weather_data()
                risk_assessor = RiskAssessor()
                risk_assessment = risk_assessor.assess_climbing_safety(forecast_data)
                
        except Exception as e:
            st.error(f"❌ Error in ML forecasting: {e}")
            st.info("Falling back to Open-Meteo data.")
            refresh_openmeteo_data()
            forecast_data = load_weather_data()
            risk_assessor = RiskAssessor()
            risk_assessment = risk_assessor.assess_climbing_safety(forecast_data)
        
    elif model_option == "Sample Data (Demo)":
        st.info("📊 Using sample data for demonstration. For real predictions, use Open-Meteo or train models with actual data.")
        
        # Load sample data
        forecast_data = load_sample_data()
        
        # Create risk assessment
        risk_assessor = RiskAssessor()
        risk_assessment = risk_assessor.assess_climbing_safety(forecast_data)
        
    else:  # Trained Models
        try:
            trainer = load_models()
            st.success("✅ Models loaded successfully!")
            feature_engineer = FeatureEngineer()
            forecast_data = generate_model_forecast(trainer, feature_engineer, forecast_datetime.replace(minute=0, second=0, microsecond=0), hours=72)
            risk_assessor = RiskAssessor()
            risk_assessment = risk_assessor.assess_climbing_safety(forecast_data)
        except Exception as e:
            st.error(f"❌ Error loading models or generating forecast: {e}")
            st.info("Falling back to Open-Meteo data.")
            refresh_openmeteo_data()
            forecast_data = load_weather_data()
            risk_assessor = RiskAssessor()
            risk_assessment = risk_assessor.assess_climbing_safety(forecast_data)
    
    # Display forecast information
    st.markdown('<div class="section-title">📅 Forecast for {}</div>'.format(forecast_datetime.strftime('%B %d, %Y at %I:%M %p')), unsafe_allow_html=True)
    
    # Show current conditions in a compact metrics row (reverted to previous style)
    now = pd.Timestamp.now()
    if forecast_data.index.tz is not None:
        now = now.tz_localize(forecast_data.index.tz)
    else:
        now = now.tz_localize(None)
    nearest_idx = forecast_data.index.get_indexer([now], method='nearest')[0]
    current = forecast_data.iloc[nearest_idx]
    cols = st.columns(4)
    
    # Map column names to handle different naming conventions
    temp_col = 'temperature_F' if 'temperature_F' in current.index else 'temperature'
    wind_col = 'wind_speed_mph' if 'wind_speed_mph' in current.index else 'wind_speed'
    pressure_col = 'air_pressure_hPa' if 'air_pressure_hPa' in current.index else 'pressure'
    precip_col = 'precip_hourly' if 'precip_hourly' in current.index else 'precipitation'
    
    # Display metrics with proper error handling
    try:
        temp_value = current[temp_col] if temp_col in current.index else "N/A"
        cols[0].metric("Temperature", f"{temp_value:.1f}°F" if temp_value != "N/A" else "N/A")
    except Exception as e:
        cols[0].metric("Temperature", "N/A")
    
    try:
        wind_value = current[wind_col] if wind_col in current.index else "N/A"
        cols[1].metric("Wind Speed", f"{wind_value:.1f} mph" if wind_value != "N/A" else "N/A")
    except Exception as e:
        cols[1].metric("Wind Speed", "N/A")
    
    try:
        pressure_value = current[pressure_col] if pressure_col in current.index else "N/A"
        cols[2].metric("Pressure", f"{pressure_value:.1f} hPa" if pressure_value != "N/A" else "N/A")
    except Exception as e:
        cols[2].metric("Pressure", "N/A")
    
    try:
        precip_value = current[precip_col] if precip_col in current.index else "N/A"
        cols[3].metric("Precipitation", f"{precip_value:.2f} in/hr" if precip_value != "N/A" else "N/A")
    except Exception as e:
        cols[3].metric("Precipitation", "N/A")
    
    # Four separate graphs for each variable with modern color palette
    def plot_single_variable(df, var, label, color):
        import plotly.graph_objects as go
        now = pd.Timestamp.now()
        if df.index.tz is not None:
            now = now.tz_localize(df.index.tz)
        else:
            now = now.tz_localize(None)
        start_time = now - pd.Timedelta(hours=36)
        end_time = now + pd.Timedelta(hours=48)
        window_df = df[(df.index >= start_time) & (df.index <= end_time)]
        history = window_df[window_df.index < now]
        forecast = window_df[window_df.index >= now]
        fig = go.Figure()
        if not history.empty:
            fig.add_trace(go.Scatter(x=history.index, y=history[var], mode='lines', name=f'{label} (history)', line=dict(color=color, dash='solid')))
        if not forecast.empty:
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast[var], mode='lines', name=f'{label} (forecast)', line=dict(color=color, dash='dot')))
        fig.update_layout(
            title=f'{label}: 36h History + 48h Forecast',
            xaxis_title='Time',
            yaxis_title=label,
            xaxis_range=[start_time, end_time],
            plot_bgcolor='#222',  # Dark background for high contrast
            paper_bgcolor='#222',
            font=dict(color='white'),
            margin=dict(l=40, r=40, t=60, b=40),
            title_font=dict(size=20, color='#1f77b4'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_xaxes(title_font=dict(color='white'), tickfont=dict(color='white'))
        fig.update_yaxes(title_font=dict(color='white'), tickfont=dict(color='white'))
        return fig
    st.plotly_chart(plot_single_variable(forecast_data, 'temperature_F', 'Temperature (°F)', '#e53935'), use_container_width=True)
    st.plotly_chart(plot_single_variable(forecast_data, 'wind_speed_mph', 'Wind Speed (mph)', '#1e88e5'), use_container_width=True)
    if 'air_pressure_hPa' in forecast_data.columns and not forecast_data['air_pressure_hPa'].isnull().all():
        st.plotly_chart(plot_single_variable(forecast_data, 'air_pressure_hPa', 'Pressure (hPa)', '#43a047'), use_container_width=True)
    else:
        st.warning('No air pressure data available. Run Open-Meteo ingestion for full variables.')
    if 'precip_hourly' in forecast_data.columns and not forecast_data['precip_hourly'].isnull().all():
        st.plotly_chart(plot_single_variable(forecast_data, 'precip_hourly', 'Precipitation (in/hr)', '#8e24aa'), use_container_width=True)
    else:
        st.warning('No precipitation data available. Run Open-Meteo ingestion for full variables.')
    
    # Section title for safety assessment
    st.markdown('<div class="section-title">⚠️ Safety Assessment</div>', unsafe_allow_html=True)
    display_risk_assessment(risk_assessment)
    
    # Section title for best climbing window
    st.markdown('<div class="section-title">🟩 Best & Worst Climbing Windows (Visual)</div>', unsafe_allow_html=True)
    
    try:
        overall_analysis = risk_assessment.get('overall_analysis', {})
        best_window_str = overall_analysis.get('best_climbing_window', 'Not available')
        worst_window_str = overall_analysis.get('worst_conditions', 'Not available')
        
        if best_window_str != 'Not available' and worst_window_str != 'Not available':
            window_bar_fig = plot_best_climbing_window_bar(best_window_str, worst_window_str, forecast_data, window_hours=10)
            st.plotly_chart(window_bar_fig, use_container_width=True)
        else:
            st.info("Climbing window data not available.")
    except Exception as e:
        st.error(f"Error displaying climbing windows: {e}")
        st.info("Climbing window visualization not available.")
    
    # Section title for risk plot
    st.markdown('<div class="section-title">📈 Risk Assessment Over Time</div>', unsafe_allow_html=True)
    
    try:
        risk_fig = create_risk_plot(risk_assessment)
        if risk_fig:
            st.plotly_chart(risk_fig, use_container_width=True)
        else:
            st.info("Risk plot data not available.")
    except Exception as e:
        st.error(f"Error creating risk plot: {e}")
        st.info("Risk visualization not available.")
    
    # Detailed hourly breakdown
    with st.expander("📋 Hourly Risk Breakdown", expanded=False):
        try:
            risk_data = risk_assessment.get('risk_data', pd.DataFrame())
            if not risk_data.empty:
                st.write("**First 24 Hours:**")
                display_columns = ['temperature_F', 'wind_speed_mph', 'air_pressure_hPa', 'precip_hourly', 'risk_score', 'risk_level']
                # Filter to only show columns that exist
                available_columns = [col for col in display_columns if col in risk_data.columns]
                if available_columns:
                    st.dataframe(risk_data[available_columns].head(24), use_container_width=True)
                    if st.checkbox("Show full 72-hour breakdown"):
                        st.write("**Full 72-Hour Forecast:**")
                        st.dataframe(risk_data[available_columns], use_container_width=True)
                else:
                    st.info("No risk data columns available for display.")
            else:
                st.info("No risk data available for hourly breakdown.")
        except Exception as e:
            st.error(f"Error displaying hourly breakdown: {e}")
            st.info("Hourly breakdown not available.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>⚠️ Safety Disclaimer:</strong> This tool provides predictions only and should not be the sole factor in climbing decisions. 
        Always check official weather forecasts, consult with experienced guides, and assess current conditions on the mountain.</p>
        <p>Mount Rainier Summit Forecast Tool | Built with ❤️ for climber safety</p>
    </div>
    """, unsafe_allow_html=True)

def refresh_openmeteo_data():
    """Automatically refresh Open-Meteo data on app load"""
    try:
        # Run the Open-Meteo ingestion script
        result = subprocess.run(
            [sys.executable, "src/openmeteo_ingest.py"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        if result.returncode == 0:
            st.success("✅ Fresh weather data loaded from Open-Meteo")
        else:
            st.warning("⚠️ Could not refresh weather data, using cached data")
    except Exception as e:
        st.warning(f"⚠️ Error refreshing data: {e}")

@st.cache_resource
def load_models():
    """Load trained models (cached)"""
    trainer = WeatherModelTrainer()
    trainer.load_models()
    return trainer

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    # Generate sample data for the next 72 hours
    start_date = datetime.now()
    timestamps = generate_forecast_hours(start_date, FORECAST_HOURS)
    
    # Create realistic sample data
    np.random.seed(42)
    
    # Temperature: typically 20-60°F at summit
    temperatures = np.random.normal(40, 15, FORECAST_HOURS)
    temperatures = np.clip(temperatures, 10, 70)
    
    # Wind speed: typically 5-30 mph
    wind_speeds = np.random.exponential(8, FORECAST_HOURS)
    wind_speeds = np.clip(wind_speeds, 0, 50)
    
    # Pressure: typically 20-22 inches Hg at elevation
    pressures = np.random.normal(21, 0.5, FORECAST_HOURS)
    
    # Precipitation: mostly 0, occasional light precipitation
    precipitation = np.random.exponential(0.1, FORECAST_HOURS)
    precipitation = np.where(np.random.random(FORECAST_HOURS) > 0.8, precipitation, 0)
    
    # Calculate wind chill
    wind_chill = [calculate_wind_chill(temp, wind) for temp, wind in zip(temperatures, wind_speeds)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature_F': temperatures,
        'wind_speed_mph': wind_speeds,
        'air_pressure_hPa': pressures,
        'precip_hourly': precipitation,
        'wind_chill': wind_chill
    })
    df.set_index('timestamp', inplace=True)
    
    return df

def create_weather_plots(forecast_data):
    """Create weather forecast plots"""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature (°F)', 'Wind Speed (mph)', 
                       'Pressure (inHg)', 'Precipitation (mm/hr)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Map column names to match the actual data format
    temp_col = 'temperature_F' if 'temperature_F' in forecast_data.columns else 'temperature'
    wind_col = 'wind_speed_mph' if 'wind_speed_mph' in forecast_data.columns else 'wind_speed'
    pressure_col = 'air_pressure_hPa' if 'air_pressure_hPa' in forecast_data.columns else 'pressure'
    precip_col = 'precip_hourly' if 'precip_hourly' in forecast_data.columns else 'precipitation'
    
    # Temperature plot
    fig.add_trace(
        go.Scatter(x=forecast_data.index, y=forecast_data[temp_col],
                  mode='lines+markers', name='Temperature', line=dict(color='red')),
        row=1, col=1
    )
    
    # Wind speed plot
    fig.add_trace(
        go.Scatter(x=forecast_data.index, y=forecast_data[wind_col],
                  mode='lines+markers', name='Wind Speed', line=dict(color='blue')),
        row=1, col=2
    )
    
    # Pressure plot
    fig.add_trace(
        go.Scatter(x=forecast_data.index, y=forecast_data[pressure_col],
                  mode='lines+markers', name='Pressure', line=dict(color='green')),
        row=2, col=1
    )
    
    # Precipitation plot
    fig.add_trace(
        go.Scatter(x=forecast_data.index, y=forecast_data[precip_col],
                  mode='lines+markers', name='Precipitation', line=dict(color='purple')),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="72-Hour Weather Forecast for Mount Rainier Summit",
        title_x=0.5
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_yaxes(title_text="Temperature (°F)", row=1, col=1)
    fig.update_yaxes(title_text="Wind Speed (mph)", row=1, col=2)
    fig.update_yaxes(title_text="Pressure (inHg)", row=2, col=1)
    fig.update_yaxes(title_text="Precipitation (mm/hr)", row=2, col=2)
    
    return fig

def create_weather_plots_with_history(forecast_data, history_data, now):
    """
    Create weather forecast plots with 36 hours of history (solid) and 48 hours of forecast (dotted)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature (°F)', 'Wind Speed (mph)', 
                       'Pressure (inHg)', 'Precipitation (mm/hr)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    # Map column names
    temp_col = 'temperature_F' if 'temperature_F' in forecast_data.columns else 'temperature'
    wind_col = 'wind_speed_mph' if 'wind_speed_mph' in forecast_data.columns else 'wind_speed'
    pressure_col = 'air_pressure_hPa' if 'air_pressure_hPa' in forecast_data.columns else 'pressure'
    precip_col = 'precip_hourly' if 'precip_hourly' in forecast_data.columns else 'precipitation'
    # Historical data: last 36 hours up to now
    hist = history_data[history_data.index < now].tail(36)
    # Forecast data: 48 hours from now
    fut = forecast_data[forecast_data.index >= now].head(48)
    # Plot for each variable
    # Temperature
    fig.add_trace(
        go.Scatter(x=hist.index, y=hist[temp_col], mode='lines', name='Temp (hist)', line=dict(color='red', dash='solid')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=fut.index, y=fut[temp_col], mode='lines', name='Temp (forecast)', line=dict(color='red', dash='dot')),
        row=1, col=1
    )
    # Wind
    fig.add_trace(
        go.Scatter(x=hist.index, y=hist[wind_col], mode='lines', name='Wind (hist)', line=dict(color='blue', dash='solid')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=fut.index, y=fut[wind_col], mode='lines', name='Wind (forecast)', line=dict(color='blue', dash='dot')),
        row=1, col=2
    )
    # Pressure
    fig.add_trace(
        go.Scatter(x=hist.index, y=hist[pressure_col], mode='lines', name='Pressure (hist)', line=dict(color='green', dash='solid')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=fut.index, y=fut[pressure_col], mode='lines', name='Pressure (forecast)', line=dict(color='green', dash='dot')),
        row=2, col=1
    )
    # Precipitation
    fig.add_trace(
        go.Scatter(x=hist.index, y=hist[precip_col], mode='lines', name='Precip (hist)', line=dict(color='purple', dash='solid')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=fut.index, y=fut[precip_col], mode='lines', name='Precip (forecast)', line=dict(color='purple', dash='dot')),
        row=2, col=2
    )
    # Layout
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="36h History + 48h Forecast for Mount Rainier Summit",
        title_x=0.5
    )
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_yaxes(title_text="Temperature (°F)", row=1, col=1)
    fig.update_yaxes(title_text="Wind Speed (mph)", row=1, col=2)
    fig.update_yaxes(title_text="Pressure (inHg)", row=2, col=1)
    fig.update_yaxes(title_text="Precipitation (mm/hr)", row=2, col=2)
    return fig

def ordinal(n):
    # Helper to get 1st, 2nd, 3rd, 4th, etc.
    return "%d%s" % (n, "tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

def format_friendly_datetime(dt):
    # Format: Monday, July 1st 7:56am
    day_name = dt.strftime('%A')
    month_name = dt.strftime('%B')
    day = ordinal(dt.day)
    hour = dt.strftime('%I').lstrip('0')
    minute = dt.strftime('%M')
    ampm = dt.strftime('%p').lower()
    return f"{day_name}, {month_name} {day} {hour}:{minute}{ampm}"

def format_window_str(window_str):
    # Parse and format both datetimes in the window string
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) to (\d{4}-\d{2}-\d{2} \d{2}:\d{2})", window_str)
    if not match:
        return window_str
    start_str, end_str = match.groups()
    start_dt = pd.to_datetime(start_str)
    end_dt = pd.to_datetime(end_str)
    return f"{format_friendly_datetime(start_dt)} to {format_friendly_datetime(end_dt)}"

def create_risk_plot(risk_assessment):
    """Create risk assessment plot"""
    risk_data = risk_assessment['risk_data']
    if risk_data.empty:
        return None
    # Extract data
    timestamps = risk_data.index
    risk_scores = risk_data['risk_score'].values
    risk_levels = risk_data['risk_level'].values
    # Softer color palette
    color_map = {'low': '#7fc97f', 'moderate': '#fdc086', 'high': '#f0027f', 'extreme': '#bf5b17'}
    risk_colors = [color_map.get(level, '#bdbdbd') for level in risk_levels]
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=risk_scores,
        mode='lines+markers',
        line=dict(color='white', width=2),
        marker=dict(color=risk_colors, size=10),
        name='Risk Score'
    ))
    # Add threshold lines
    fig.add_hline(y=3.0, line_dash="dash", line_color="#7fc97f", annotation_text="Low Risk Threshold")
    fig.add_hline(y=5.5, line_dash="dash", line_color="#fdc086", annotation_text="Moderate Risk Threshold")
    fig.add_hline(y=8.0, line_dash="dash", line_color="#f0027f", annotation_text="High Risk Threshold")
    fig.update_layout(
        title="Risk Assessment Over Time",
        xaxis_title="Time",
        yaxis_title="Risk Score",
        height=400,
        yaxis=dict(range=[0, 10]),
        plot_bgcolor='#222',
        paper_bgcolor='#222',
        font=dict(color='white')
    )
    fig.update_xaxes(title_text="Time")
    return fig

def display_risk_assessment(risk_assessment):
    try:
        overall_analysis = risk_assessment.get('overall_analysis', {})
        risk_level = overall_analysis.get('overall_risk_level', overall_analysis.get('overall_risk', 'unknown'))
        risk_score = overall_analysis.get('overall_risk_score', None)
        best_window = overall_analysis.get('best_climbing_window', 'Not available')
        worst_window = overall_analysis.get('worst_conditions', 'Not available')
        
        # Custom color for low risk (teal/blue-green)
        low_color = '#d0f5f2'
        low_text = '#006d5b'
        moderate_color = '#fff3cd'
        moderate_text = '#856404'
        high_color = '#f8d7da'
        high_text = '#721c24'
        
        if risk_level == 'low':
            box_color = low_color
            text_color = low_text
        elif risk_level == 'moderate':
            box_color = moderate_color
            text_color = moderate_text
        else:
            box_color = high_color
            text_color = high_text
        
        st.markdown(f"""
        <div style='background-color: {box_color}; color: {text_color}; padding: 2rem; border-radius: 1rem; border-left: 8px solid {text_color}; margin-bottom: 2rem;'>
            <span style='font-size:2rem; font-weight:bold;'>
                {'✅' if risk_level == 'low' else '⚠️' if risk_level == 'moderate' else '❌'} <b>Overall Risk Level: {risk_level.upper()}</b>
            </span><br>
            <b>Risk Score:</b> {f'{risk_score:.1f}' if risk_score is not None else 'N/A'}/10<br>
            <b>Best Climbing Window:</b> {best_window}<br>
            <b>Worst Conditions:</b> {worst_window}<br>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying risk assessment: {e}")
        st.info("Risk assessment data is incomplete or unavailable.")

def parse_best_window_indices(best_window_str, forecast_data):
    """Parse the best climbing window string and return start and end hour indices (0-72)"""
    # Example: '2025-07-03 07:44 to 2025-07-03 17:44'
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) to (\d{4}-\d{2}-\d{2} \d{2}:\d{2})", best_window_str)
    if not match:
        return 0, 0
    start_str, end_str = match.groups()
    
    # Parse datetime strings and ensure they have the same timezone as forecast_data
    start_dt = pd.to_datetime(start_str)
    end_dt = pd.to_datetime(end_str)
    
    # Get the timezone from forecast_data index
    timestamps = forecast_data.index
    if timestamps.tz is not None:
        # If forecast_data has timezone, localize the parsed datetimes to the same timezone
        start_dt = start_dt.tz_localize(timestamps.tz)
        end_dt = end_dt.tz_localize(timestamps.tz)
    else:
        # If forecast_data has no timezone, make sure parsed datetimes are also timezone-naive
        start_dt = start_dt.tz_localize(None)
        end_dt = end_dt.tz_localize(None)
    
    # Find the closest indices in the forecast data
    start_idx = (timestamps >= start_dt).argmax()
    end_idx = (timestamps >= end_dt).argmax()
    return start_idx, end_idx

def plot_best_climbing_window_bar(best_window_str, worst_window_str, forecast_data, window_hours=10):
    """Create a horizontal bar showing the best climbing window (green) and worst climbing window (red) on a 0-72 hour scale"""
    start_idx, end_idx = parse_best_window_indices(best_window_str, forecast_data)
    worst_start_idx, worst_end_idx = parse_best_window_indices(worst_window_str, forecast_data)
    # If the window is less than window_hours, extend it
    if end_idx - start_idx < window_hours:
        end_idx = min(start_idx + window_hours, len(forecast_data)-1)
    if worst_end_idx - worst_start_idx < window_hours:
        worst_end_idx = min(worst_start_idx + window_hours, len(forecast_data)-1)
    x = list(range(len(forecast_data)))
    y = [1]*len(forecast_data)
    colors = ['#e0e0e0']*len(forecast_data)
    for i in range(start_idx, end_idx):
        colors[i] = '#4CAF50'  # Highlight best window in green
    for i in range(worst_start_idx, worst_end_idx):
        colors[i] = '#e53935'  # Highlight worst window in red (overrides green if overlap)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=y,
        marker_color=colors,
        orientation='v',
        hoverinfo='skip',
        showlegend=False
    ))
    fig.update_layout(
        height=100,
        width=800,
        title="Best (Green) & Worst (Red) 10-Hour Climbing Windows",
        xaxis=dict(
            title="Forecast Hour (0-72)",
            tickmode='linear',
            tick0=0,
            dtick=6,
            range=[0, 72]
        ),
        yaxis=dict(
            visible=False
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def generate_hybrid_forecast(start_datetime, hours=72):
    """
    Generate a 72-hour hybrid forecast using ML baseline + trend analysis.
    Returns a DataFrame with predicted weather variables.
    """
    import pandas as pd
    import numpy as np
    from datetime import timedelta
    import pickle
    from pathlib import Path
    
    # Import hybrid forecaster
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from hybrid_forecast import HybridForecaster
    
    try:
        # Create hybrid forecaster
        forecaster = HybridForecaster()
        
        # Load recent historical data for trend analysis
        historical_data = load_recent_historical_data()
        
        # Generate hybrid forecast
        forecast_df = forecaster.generate_hybrid_forecast(
            start_time=start_datetime, 
            hours=hours, 
            historical_data=historical_data
        )
        
        if forecast_df is not None:
            st.success("✅ Hybrid forecast generated successfully!")
            st.info(f"🔮 Using ML baseline + trend analysis approach")
            return forecast_df
        else:
            st.error("❌ Failed to generate hybrid forecast")
            return None
            
    except Exception as e:
        st.error(f"❌ Error in hybrid forecasting: {e}")
        return None

def load_recent_historical_data():
    """
    Load recent historical data for trend analysis
    """
    # Try multiple possible historical data files
    historical_files = [
        Path("data/processed/cleaned_weather_apr_jul_inch.csv"),
        Path("data/processed/engineered_features.csv"),
        Path("data/processed/merged_data.csv")
    ]
    
    for file_path in historical_files:
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Get the last 72 hours of data
                cutoff_date = datetime.now() - timedelta(hours=72)
                recent_data = df[df.index >= cutoff_date].copy()
                
                if len(recent_data) > 0:
                    st.info(f"📊 Using {len(recent_data)} hours of historical data for trend analysis")
                    return recent_data
                else:
                    st.warning(f"⚠️ No recent data found in {file_path}")
                    
            except Exception as e:
                st.error(f"❌ Error loading {file_path}: {e}")
                continue
    
    st.warning("⚠️ No historical data found for trend analysis")
    return None

def generate_model_forecast(trainer, feature_engineer, start_datetime, hours=72):
    """
    Generate a 72-hour forecast using trained future-only models.
    Returns a DataFrame with predicted weather variables.
    """
    import pandas as pd
    import numpy as np
    from datetime import timedelta
    import pickle
    from pathlib import Path
    
    # Generate future hourly timestamps
    timestamps = [start_datetime + timedelta(hours=i) for i in range(hours)]
    df = pd.DataFrame(index=pd.DatetimeIndex(timestamps, name="timestamp"))
    
    # Engineer features for these timestamps
    df_features = feature_engineer.create_time_features(df)
    # Select only future-available features
    future_features = [
        'hour_of_day', 'day_of_week', 'day_of_month', 'month', 'year',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'season_num', 'is_weekend', 'is_daylight'
    ]
    X_future = df_features[future_features]
    # Load future-only models
    model_dir = Path("data/models")
    model_files = {
        'temperature_F': model_dir / "temperature_F_future_model.pkl",
        'wind_speed_mph': model_dir / "wind_speed_mph_future_model.pkl",
        'air_pressure_hPa': model_dir / "air_pressure_hPa_future_model.pkl",
        'precip_hourly': model_dir / "precip_hourly_future_model.pkl"
    }
    preds = {}
    for var, model_path in model_files.items():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        # Align columns to model's expected order
        X = X_future.reindex(columns=model.feature_names_in_, fill_value=0)
        preds[var] = model.predict(X)
    # Build forecast DataFrame
    forecast_df = pd.DataFrame(preds, index=df_features.index)
    # Calculate wind chill
    from utils import calculate_wind_chill
    forecast_df['wind_chill'] = [calculate_wind_chill(t, w) for t, w in zip(forecast_df['temperature_F'], forecast_df['wind_speed_mph'])]
    return forecast_df

def load_weather_data():
    """
    Load weather data from Open-Meteo, standardize columns for downstream use.
    Returns a DataFrame with columns: temperature_F, wind_speed_mph, air_pressure_hPa, precip_hourly.
    """
    # Try to load the Open-Meteo data
    merged_path = Path("data/processed/merged_data.csv")
    if merged_path.exists():
        try:
            df = pd.read_csv(merged_path, index_col=0, parse_dates=True)
            # Ensure the correct column names are present
            if 'temperature_F' in df.columns and 'wind_speed_mph' in df.columns:
                st.success("✅ Loaded fresh weather data from Open-Meteo")
                return df
        except Exception as e:
            st.warning(f"⚠️ Error loading Open-Meteo data: {e}")
    
    # Fallback to sample data
    st.warning("⚠️ Using sample data - Open-Meteo data not available")
    return load_sample_data()

def plot_history_and_forecast(df):
    """
    Plot 36h history (solid) and 48h forecast (dotted) for each variable.
    """
    import plotly.graph_objects as go
    now = pd.Timestamp.now(tz=df.index.tz) if df.index.tz else pd.Timestamp.now()
    # Split history and forecast
    history = df[df.index < now].last('36H')
    forecast = df[df.index >= now].first('48H')
    fig = go.Figure()
    # Temperature
    fig.add_trace(go.Scatter(x=history.index, y=history['temperature_F'], mode='lines', name='Temp (history)', line=dict(color='red', dash='solid')))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast['temperature_F'], mode='lines', name='Temp (forecast)', line=dict(color='red', dash='dot')))
    # Wind
    fig.add_trace(go.Scatter(x=history.index, y=history['wind_speed_mph'], mode='lines', name='Wind (history)', line=dict(color='blue', dash='solid')))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast['wind_speed_mph'], mode='lines', name='Wind (forecast)', line=dict(color='blue', dash='dot')))
    # Pressure (if available)
    if 'air_pressure_hPa' in df.columns:
        fig.add_trace(go.Scatter(x=history.index, y=history['air_pressure_hPa'], mode='lines', name='Pressure (history)', line=dict(color='green', dash='solid')))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast['air_pressure_hPa'], mode='lines', name='Pressure (forecast)', line=dict(color='green', dash='dot')))
    # Precip (if available)
    if 'precip_hourly' in df.columns:
        fig.add_trace(go.Scatter(x=history.index, y=history['precip_hourly'], mode='lines', name='Precip (history)', line=dict(color='purple', dash='solid')))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast['precip_hourly'], mode='lines', name='Precip (forecast)', line=dict(color='purple', dash='dot')))
    fig.update_layout(title='Mount Rainier Weather: 36h History + 48h Forecast', xaxis_title='Time', yaxis_title='Value', legend_title='Variable')
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    # Create tabs for navigation
    tab1, tab2 = st.tabs(["🏔️ Forecast", "ℹ️ About & How It Works"])
    
    with tab1:
        show_forecast_page()
    
    with tab2:
        show_about_page()

if __name__ == "__main__":
    main() 