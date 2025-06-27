"""
Mount Rainier Weather Prediction Web Application

This is the main web interface for the Mount Rainier Weather Prediction Tool.
It provides climbers with an easy-to-use dashboard to check weather forecasts
and safety assessments for climbing Mount Rainier.

Author: Weather Prediction Team
Purpose: Web interface for weather forecasting and safety assessment
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

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

# Now import our modules
from config.config import *
from utils import (
    calculate_wind_chill, format_forecast_date, generate_forecast_hours,
    round_to_significant_figures
)
from model_training import WeatherModelTrainer
from risk_assessment import RiskAssessment
from feature_engineering import FeatureEngineering

# Page configuration
st.set_page_config(
    page_title="Mount Rainier Summit Forecast",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .risk-moderate {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

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
        'temperature': temperatures,
        'wind_speed': wind_speeds,
        'pressure': pressures,
        'precipitation': precipitation,
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
    
    # Temperature plot
    fig.add_trace(
        go.Scatter(x=forecast_data.index, y=forecast_data['temperature'],
                  mode='lines+markers', name='Temperature', line=dict(color='red')),
        row=1, col=1
    )
    
    # Wind speed plot
    fig.add_trace(
        go.Scatter(x=forecast_data.index, y=forecast_data['wind_speed'],
                  mode='lines+markers', name='Wind Speed', line=dict(color='blue')),
        row=1, col=2
    )
    
    # Pressure plot
    fig.add_trace(
        go.Scatter(x=forecast_data.index, y=forecast_data['pressure'],
                  mode='lines+markers', name='Pressure', line=dict(color='green')),
        row=2, col=1
    )
    
    # Precipitation plot
    fig.add_trace(
        go.Scatter(x=forecast_data.index, y=forecast_data['precipitation'],
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

def create_risk_plot(risk_assessment):
    """Create risk assessment plot"""
    hourly_risks = risk_assessment['hourly_risks']
    
    if not hourly_risks:
        return None
    
    # Extract data
    timestamps = [hr['timestamp'] for hr in hourly_risks]
    risk_scores = [hr['risk_score'] for hr in hourly_risks]
    risk_levels = [hr['risk_level'] for hr in hourly_risks]
    
    # Color mapping
    colors = {'low': 'green', 'moderate': 'orange', 'high': 'red'}
    risk_colors = [colors.get(level, 'gray') for level in risk_levels]
    
    # Create plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=risk_scores,
        mode='lines+markers',
        line=dict(color='black', width=2),
        marker=dict(color=risk_colors, size=8),
        name='Risk Score'
    ))
    
    # Add threshold lines
    fig.add_hline(y=1, line_dash="dash", line_color="green", 
                  annotation_text="Low Risk Threshold")
    fig.add_hline(y=3, line_dash="dash", line_color="orange", 
                  annotation_text="Moderate Risk Threshold")
    fig.add_hline(y=4, line_dash="dash", line_color="red", 
                  annotation_text="High Risk Threshold")
    
    fig.update_layout(
        title="Risk Assessment Over Time",
        xaxis_title="Time",
        yaxis_title="Risk Score",
        height=400,
        yaxis=dict(range=[0, 6])
    )
    
    return fig

def display_risk_assessment(risk_assessment):
    """Display risk assessment results"""
    risk_level = risk_assessment['overall_risk']
    risk_score = risk_assessment['overall_score']
    
    # Risk level styling
    if risk_level == 'low':
        risk_class = 'risk-low'
        risk_icon = '✅'
    elif risk_level == 'moderate':
        risk_class = 'risk-moderate'
        risk_icon = '⚠️'
    else:
        risk_class = 'risk-high'
        risk_icon = '🚨'
    
    # Display risk summary
    st.markdown(f"""
    <div class="{risk_class}">
        <h3>{risk_icon} Overall Risk Level: {risk_level.upper()}</h3>
        <p><strong>Risk Score:</strong> {risk_score:.1f}/10</p>
        <p><strong>Assessment:</strong> {risk_assessment['risk_justification']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display day-by-day breakdown
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Day 1 Risk", f"{risk_assessment['day1_avg_risk']:.1f}/10")
    
    with col2:
        st.metric("Day 2 Risk", f"{risk_assessment['day2_avg_risk']:.1f}/10")
    
    with col3:
        st.metric("Day 3 Risk", f"{risk_assessment['day3_avg_risk']:.1f}/10")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏔️ Mount Rainier Summit Forecast</h1>', 
                unsafe_allow_html=True)
    
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
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Data Source",
        ["Sample Data (Demo)", "Trained Models"],
        help="Choose between sample data for demonstration or trained models"
    )
    
    # Main content
    if model_option == "Sample Data (Demo)":
        st.info("📊 Using sample data for demonstration. For real predictions, train models with actual data.")
        
        # Load sample data
        forecast_data = load_sample_data()
        
        # Create risk assessment
        risk_assessor = RiskAssessment()
        risk_assessment = risk_assessor.assess_forecast_risk(forecast_data)
        
    else:
        # Load trained models
        try:
            trainer = load_models()
            st.success("✅ Models loaded successfully!")
            
            # For now, use sample data (in real implementation, this would use actual models)
            forecast_data = load_sample_data()
            risk_assessor = RiskAssessment()
            risk_assessment = risk_assessor.assess_forecast_risk(forecast_data)
            
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            st.info("Falling back to sample data for demonstration.")
            forecast_data = load_sample_data()
            risk_assessor = RiskAssessment()
            risk_assessment = risk_assessor.assess_forecast_risk(forecast_data)
    
    # Display forecast information
    st.subheader(f"📅 Forecast for {forecast_datetime.strftime('%B %d, %Y at %I:%M %p')}")
    
    # Current conditions (first hour)
    current_conditions = forecast_data.iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Temperature", f"{current_conditions['temperature']:.1f}°F")
    
    with col2:
        st.metric("Wind Speed", f"{current_conditions['wind_speed']:.1f} mph")
    
    with col3:
        st.metric("Pressure", f"{current_conditions['pressure']:.2f} inHg")
    
    with col4:
        st.metric("Precipitation", f"{current_conditions['precipitation']:.2f} mm/hr")
    
    # Weather plots
    st.subheader("📈 Weather Forecast")
    weather_fig = create_weather_plots(forecast_data)
    st.plotly_chart(weather_fig, use_container_width=True)
    
    # Risk assessment
    st.subheader("⚠️ Safety Assessment")
    display_risk_assessment(risk_assessment)
    
    # Risk plot
    risk_fig = create_risk_plot(risk_assessment)
    if risk_fig:
        st.plotly_chart(risk_fig, use_container_width=True)
    
    # Safety recommendations
    st.subheader("🛡️ Safety Recommendations")
    recommendations = risk_assessor.get_safety_recommendations(risk_assessment['overall_risk'])
    
    for rec in recommendations:
        st.write(rec)
    
    # Detailed hourly breakdown
    st.subheader("📋 Hourly Risk Breakdown")
    
    # Create hourly risk table
    hourly_table = risk_assessor.format_hourly_risk_table(risk_assessment['hourly_risks'])
    
    if not hourly_table.empty:
        # Show first 24 hours by default
        st.write("**First 24 Hours:**")
        st.dataframe(hourly_table.head(24), use_container_width=True)
        
        # Option to show all hours
        if st.checkbox("Show full 72-hour breakdown"):
            st.write("**Full 72-Hour Forecast:**")
            st.dataframe(hourly_table, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>⚠️ Safety Disclaimer:</strong> This tool provides predictions only and should not be the sole factor in climbing decisions. 
        Always check official weather forecasts, consult with experienced guides, and assess current conditions on the mountain.</p>
        <p>Mount Rainier Summit Forecast Tool | Built with ❤️ for climber safety</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 