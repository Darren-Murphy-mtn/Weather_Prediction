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
import re

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
from risk_assessment import RiskAssessor
from feature_engineering import FeatureEngineer

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
    """Display risk assessment results"""
    overall_analysis = risk_assessment['overall_analysis']
    risk_level = overall_analysis['overall_risk']
    risk_score = overall_analysis['risk_summary']['average_risk_score']
    # Risk level styling
    if risk_level == 'low':
        risk_class = 'risk-low'
        risk_icon = '✅'
    elif risk_level == 'moderate':
        risk_class = 'risk-moderate'
        risk_icon = '⚠️'
    elif risk_level == 'high':
        risk_class = 'risk-high'
        risk_icon = '🚨'
    else:
        risk_class = 'risk-high'
        risk_icon = '🛑'
    # Display risk summary
    st.markdown(f"""
    <div class="{risk_class}">
        <h3>{risk_icon} Overall Risk Level: {risk_level.upper()}</h3>
        <p><strong>Risk Score:</strong> {risk_score:.1f}/10</p>
        <p><strong>Best Climbing Window:</strong> {format_window_str(overall_analysis['best_climbing_window'])}</p>
        <p><strong>Worst Conditions:</strong> {format_window_str(overall_analysis['worst_conditions'])}</p>
    </div>
    """, unsafe_allow_html=True)
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    # Display day-by-day breakdown
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Day 1 Risk", f"{overall_analysis['day1_avg_risk']:.1f}/10")
    with col2:
        st.metric("Day 2 Risk", f"{overall_analysis['day2_avg_risk']:.1f}/10")
    with col3:
        st.metric("Day 3 Risk", f"{overall_analysis['day3_avg_risk']:.1f}/10")

def parse_best_window_indices(best_window_str, forecast_data):
    """Parse the best climbing window string and return start and end hour indices (0-72)"""
    # Example: '2025-07-03 07:44 to 2025-07-03 17:44'
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) to (\d{4}-\d{2}-\d{2} \d{2}:\d{2})", best_window_str)
    if not match:
        return 0, 0
    start_str, end_str = match.groups()
    start_dt = pd.to_datetime(start_str)
    end_dt = pd.to_datetime(end_str)
    # Find the closest indices in the forecast data
    timestamps = forecast_data.index
    start_idx = (timestamps >= start_dt).argmax()
    end_idx = (timestamps >= end_dt).argmax()
    return start_idx, end_idx

def plot_best_climbing_window_bar(best_window_str, forecast_data, window_hours=10):
    """Create a horizontal bar showing the best climbing window on a 0-72 hour scale"""
    start_idx, end_idx = parse_best_window_indices(best_window_str, forecast_data)
    # If the window is less than window_hours, extend it
    if end_idx - start_idx < window_hours:
        end_idx = min(start_idx + window_hours, len(forecast_data)-1)
    x = list(range(len(forecast_data)))
    y = [1]*len(forecast_data)
    colors = ['#e0e0e0']*len(forecast_data)
    for i in range(start_idx, end_idx):
        colors[i] = '#4CAF50'  # Highlight best window in green
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
        title="Best 10-Hour Climbing Window (Green Box)",
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
        risk_assessor = RiskAssessor()
        risk_assessment = risk_assessor.assess_climbing_safety(forecast_data)
        
    else:
        # Load trained models
        try:
            trainer = load_models()
            st.success("✅ Models loaded successfully!")
            
            # For now, use sample data (in real implementation, this would use actual models)
            forecast_data = load_sample_data()
            risk_assessor = RiskAssessor()
            risk_assessment = risk_assessor.assess_climbing_safety(forecast_data)
            
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            st.info("Falling back to sample data for demonstration.")
            forecast_data = load_sample_data()
            risk_assessor = RiskAssessor()
            risk_assessment = risk_assessor.assess_climbing_safety(forecast_data)
    
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
    
    # Add best climbing window bar plot
    st.subheader("🟩 Best Climbing Window (Visual)")
    best_window_str = risk_assessment['overall_analysis']['best_climbing_window']
    window_bar_fig = plot_best_climbing_window_bar(best_window_str, forecast_data, window_hours=10)
    st.plotly_chart(window_bar_fig, use_container_width=True)
    
    # Risk plot
    risk_fig = create_risk_plot(risk_assessment)
    if risk_fig:
        st.plotly_chart(risk_fig, use_container_width=True)
    
    # Safety recommendations
    st.subheader("🛡️ Safety Recommendations")
    recommendations = risk_assessment['safety_recommendations']
    
    for rec in recommendations:
        st.write(rec)
    
    # Detailed hourly breakdown
    st.subheader("📋 Hourly Risk Breakdown")
    
    # Show risk data table
    risk_data = risk_assessment['risk_data']
    
    if not risk_data.empty:
        # Show first 24 hours by default
        st.write("**First 24 Hours:**")
        display_columns = ['temperature', 'wind_speed', 'pressure', 'precipitation', 'risk_score', 'risk_level']
        st.dataframe(risk_data[display_columns].head(24), use_container_width=True)
        
        # Option to show all hours
        if st.checkbox("Show full 72-hour breakdown"):
            st.write("**Full 72-Hour Forecast:**")
            st.dataframe(risk_data[display_columns], use_container_width=True)
    
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