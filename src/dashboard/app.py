"""Streamlit dashboard for EcoPredict"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from pathlib import Path
import json

from ..prediction.predict import EcoPredictionEngine
from ..gis.heatmap import HeatmapGenerator
from ..utils.logger import get_logger
from ..utils.helpers import validate_coordinates, load_config
from .components import (
    create_risk_gauge, create_species_chart, 
    create_climate_trends, create_land_use_pie
)

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="EcoPredict Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .risk-high { border-left-color: #dc3545; }
    .risk-medium { border-left-color: #ffc107; }
    .risk-low { border-left-color: #28a745; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    # Generate sample data
    np.random.seed(42)
    n_points = 100
    
    # Maharashtra bounds
    lat_range = (15.6, 22.0)
    lon_range = (72.6, 80.9)
    
    data = {
        'latitude': np.random.uniform(lat_range[0], lat_range[1], n_points),
        'longitude': np.random.uniform(lon_range[0], lon_range[1], n_points),
        'risk_score': np.random.beta(2, 5, n_points),  # Skewed towards lower risk
        'species_count': np.random.poisson(15, n_points),
        'forest_cover': np.random.uniform(0, 1, n_points),
        'temperature': np.random.normal(25, 5, n_points),
        'precipitation': np.random.exponential(2, n_points)
    }
    
    df = pd.DataFrame(data)
    df['risk_category'] = pd.cut(df['risk_score'], 
                                bins=[0, 0.3, 0.6, 1.0], 
                                labels=['Low', 'Medium', 'High'])
    
    return df


@st.cache_resource
def initialize_prediction_engine():
    """Initialize prediction engine"""
    try:
        engine = EcoPredictionEngine()
        # Try to load a model if available
        model_path = Path("models/trained/best_model.joblib")
        if model_path.exists():
            engine.load_model(str(model_path))
        return engine
    except Exception as e:
        st.error(f"Failed to initialize prediction engine: {e}")
        return None


def create_dashboard():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🌍 EcoPredict Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview", "Risk Prediction", "Species Analysis", "Climate Trends", "Data Explorer"]
    )
    
    # Load data
    data = load_sample_data()
    
    if page == "Overview":
        show_overview(data)
    elif page == "Risk Prediction":
        show_risk_prediction(data)
    elif page == "Species Analysis":
        show_species_analysis(data)
    elif page == "Climate Trends":
        show_climate_trends(data)
    elif page == "Data Explorer":
        show_data_explorer(data)


def show_overview(data):
    """Show overview dashboard"""
    st.header("📊 Ecological Risk Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_risk = data['risk_score'].mean()
        st.metric(
            label="Average Risk Score",
            value=f"{avg_risk:.3f}",
            delta=f"{(avg_risk - 0.35):.3f}" if avg_risk > 0.35 else f"{(avg_risk - 0.35):.3f}"
        )
    
    with col2:
        high_risk_count = len(data[data['risk_category'] == 'High'])
        st.metric(
            label="High Risk Areas",
            value=high_risk_count,
            delta=f"{high_risk_count - 15}" if high_risk_count > 15 else f"{high_risk_count - 15}"
        )
    
    with col3:
        avg_species = data['species_count'].mean()
        st.metric(
            label="Avg Species Count",
            value=f"{avg_species:.1f}",
            delta=f"{(avg_species - 15):.1f}"
        )
    
    with col4:
        avg_forest = data['forest_cover'].mean()
        st.metric(
            label="Avg Forest Cover",
            value=f"{avg_forest:.1%}",
            delta=f"{(avg_forest - 0.5):.1%}"
        )
    
    # Risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(
            data, x='risk_score', nbins=20,
            color_discrete_sequence=['#2E8B57'],
            title="Distribution of Risk Scores"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Categories")
        risk_counts = data['risk_category'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'},
            title="Risk Category Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Interactive map
    st.subheader("🗺️ Risk Map")
    
    # Create folium map
    center_lat = data['latitude'].mean()
    center_lon = data['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
    
    # Add points to map
    for idx, row in data.iterrows():
        color = 'red' if row['risk_category'] == 'High' else 'orange' if row['risk_category'] == 'Medium' else 'green'
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            popup=f"Risk: {row['risk_score']:.3f}<br>Species: {row['species_count']}<br>Forest: {row['forest_cover']:.1%}",
            color=color,
            fill=True,
            fillOpacity=0.7
        ).add_to(m)
    
    # Display map
    map_data = st_folium(m, width=700, height=500)


def show_risk_prediction(data):
    """Show risk prediction interface"""
    st.header("🎯 Risk Prediction")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Location Input")
        latitude = st.number_input("Latitude", min_value=15.0, max_value=22.0, value=19.0, step=0.01)
        longitude = st.number_input("Longitude", min_value=72.0, max_value=81.0, value=75.0, step=0.01)
    
    with col2:
        st.subheader("Environmental Factors")
        forest_cover = st.slider("Forest Cover (%)", 0, 100, 50) / 100
        species_count = st.slider("Species Count", 0, 50, 15)
        temperature = st.slider("Temperature (°C)", 10, 40, 25)
        precipitation = st.slider("Precipitation (mm)", 0, 10, 2)
    
    # Predict button
    if st.button("Predict Risk", type="primary"):
        if validate_coordinates(latitude, longitude):
            # Mock prediction (replace with actual model prediction)
            risk_score = np.random.beta(2, 5)  # Mock risk score
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Risk gauge
                fig = create_risk_gauge(risk_score)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk category
                if risk_score < 0.3:
                    category = "Low"
                    color = "#28a745"
                elif risk_score < 0.6:
                    category = "Medium"
                    color = "#ffc107"
                else:
                    category = "High"
                    color = "#dc3545"
                
                st.markdown(f"""
                <div class="metric-card risk-{category.lower()}">
                    <h3>Risk Category</h3>
                    <h2 style="color: {color};">{category}</h2>
                    <p>Score: {risk_score:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # Contributing factors
                st.subheader("Contributing Factors")
                factors = {
                    'Forest Cover': forest_cover * 100,
                    'Species Diversity': species_count,
                    'Temperature': temperature,
                    'Precipitation': precipitation
                }
                
                for factor, value in factors.items():
                    st.write(f"**{factor}:** {value}")
            
            # Show location on map
            st.subheader("Prediction Location")
            m = folium.Map(location=[latitude, longitude], zoom_start=10)
            
            folium.Marker(
                location=[latitude, longitude],
                popup=f"Risk Score: {risk_score:.3f}",
                icon=folium.Icon(color='red' if risk_score > 0.6 else 'orange' if risk_score > 0.3 else 'green')
            ).add_to(m)
            
            st_folium(m, width=700, height=400)
        
        else:
            st.error("Invalid coordinates. Please check latitude and longitude values.")


def show_species_analysis(data):
    """Show species analysis dashboard"""
    st.header("🦋 Species Analysis")
    
    # Species metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_species = data['species_count'].sum()
        st.metric("Total Species Observations", total_species)
    
    with col2:
        avg_diversity = data['species_count'].mean()
        st.metric("Average Diversity", f"{avg_diversity:.1f}")
    
    with col3:
        max_diversity = data['species_count'].max()
        st.metric("Maximum Diversity", max_diversity)
    
    # Species distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_species_chart(data)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Correlation with risk
        fig = px.scatter(
            data, x='species_count', y='risk_score',
            color='risk_category',
            color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'},
            title="Species Count vs Risk Score"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Biodiversity hotspots
    st.subheader("Biodiversity Hotspots")
    hotspots = data.nlargest(10, 'species_count')[['latitude', 'longitude', 'species_count', 'risk_score']]
    st.dataframe(hotspots, use_container_width=True)


def show_climate_trends(data):
    """Show climate trends dashboard"""
    st.header("🌡️ Climate Trends")
    
    # Climate metrics
    col1, col2 = st.columns(2)
    
    with col1:
        avg_temp = data['temperature'].mean()
        st.metric("Average Temperature", f"{avg_temp:.1f}°C")
    
    with col2:
        avg_precip = data['precipitation'].mean()
        st.metric("Average Precipitation", f"{avg_precip:.1f}mm")
    
    # Climate charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_climate_trends(data)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Temperature vs Risk
        fig = px.scatter(
            data, x='temperature', y='risk_score',
            color='risk_category',
            color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'},
            title="Temperature vs Risk Score"
        )
        st.plotly_chart(fig, use_container_width=True)


def show_data_explorer(data):
    """Show data explorer"""
    st.header("🔍 Data Explorer")
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.selectbox("Risk Category", ["All"] + list(data['risk_category'].unique()))
    
    with col2:
        species_min = st.number_input("Min Species Count", 0, int(data['species_count'].max()), 0)
    
    with col3:
        forest_min = st.slider("Min Forest Cover", 0.0, 1.0, 0.0)
    
    # Apply filters
    filtered_data = data.copy()
    
    if risk_filter != "All":
        filtered_data = filtered_data[filtered_data['risk_category'] == risk_filter]
    
    filtered_data = filtered_data[filtered_data['species_count'] >= species_min]
    filtered_data = filtered_data[filtered_data['forest_cover'] >= forest_min]
    
    # Display filtered data
    st.subheader(f"Filtered Data ({len(filtered_data)} records)")
    st.dataframe(filtered_data, use_container_width=True)
    
    # Download button
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="ecopredict_data.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    create_dashboard()