#!/usr/bin/env python3
"""Simple dashboard test for EcoPredict"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="EcoPredict Test Dashboard",
    page_icon="🌍",
    layout="wide"
)

def generate_sample_data():
    """Generate sample ecological data"""
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'latitude': np.random.uniform(15.6, 22.0, n_samples),
        'longitude': np.random.uniform(72.6, 80.9, n_samples),
        'temperature': np.random.normal(25, 5, n_samples),
        'precipitation': np.random.exponential(2, n_samples),
        'forest_cover': np.random.uniform(0, 1, n_samples),
        'species_count': np.random.poisson(15, n_samples),
        'urban_area': np.random.uniform(0, 1, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate risk score
    risk_score = (
        0.3 * (1 - df['forest_cover']) +
        0.2 * df['urban_area'] +
        0.15 * np.abs(df['temperature'] - 25) / 10 +
        0.35 * np.random.uniform(0, 1, n_samples)
    )
    
    df['risk_score'] = np.clip(risk_score, 0, 1)
    df['risk_category'] = pd.cut(df['risk_score'], bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Medium', 'High'])
    
    return df

def main():
    """Main dashboard function"""
    
    # Header
    st.title("🌍 EcoPredict Test Dashboard")
    st.markdown("**Ecological Risk Prediction System - Demo**")
    
    # Generate data
    data = generate_sample_data()
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Risk filter
    risk_filter = st.sidebar.selectbox(
        "Filter by Risk Level",
        ["All", "Low", "Medium", "High"]
    )
    
    # Apply filter
    if risk_filter != "All":
        filtered_data = data[data['risk_category'] == risk_filter]
    else:
        filtered_data = data
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Locations", len(filtered_data))
    
    with col2:
        avg_risk = filtered_data['risk_score'].mean()
        st.metric("Average Risk", f"{avg_risk:.3f}")
    
    with col3:
        high_risk_count = len(filtered_data[filtered_data['risk_category'] == 'High'])
        st.metric("High Risk Areas", high_risk_count)
    
    with col4:
        avg_species = filtered_data['species_count'].mean()
        st.metric("Avg Species Count", f"{avg_species:.1f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Score Distribution")
        fig_hist = px.histogram(
            filtered_data, 
            x='risk_score', 
            nbins=20,
            color_discrete_sequence=['#2E8B57']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("Risk Categories")
        risk_counts = filtered_data['risk_category'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Map
    st.subheader("🗺️ Geographic Risk Distribution")
    
    fig_map = px.scatter_mapbox(
        filtered_data,
        lat='latitude',
        lon='longitude',
        color='risk_score',
        size='species_count',
        hover_data=['temperature', 'forest_cover', 'risk_category'],
        color_continuous_scale='RdYlGn_r',
        mapbox_style='open-street-map',
        height=500
    )
    
    fig_map.update_layout(
        mapbox=dict(
            center=dict(lat=filtered_data['latitude'].mean(), lon=filtered_data['longitude'].mean()),
            zoom=6
        )
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Prediction Interface
    st.subheader("🎯 Risk Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Location**")
        lat = st.number_input("Latitude", min_value=15.0, max_value=22.0, value=19.0)
        lon = st.number_input("Longitude", min_value=72.0, max_value=81.0, value=75.0)
    
    with col2:
        st.write("**Environmental Factors**")
        temp = st.slider("Temperature (°C)", 10, 40, 25)
        forest = st.slider("Forest Cover", 0.0, 1.0, 0.5)
        urban = st.slider("Urban Area", 0.0, 1.0, 0.3)
    
    if st.button("Predict Risk"):
        # Simple prediction calculation
        risk = (0.3 * (1 - forest) + 0.2 * urban + 0.15 * abs(temp - 25) / 10 + 0.35 * 0.5)
        risk = max(0.0, min(1.0, risk))
        
        if risk < 0.3:
            category = "Low"
            color = "green"
        elif risk < 0.6:
            category = "Medium"
            color = "orange"
        else:
            category = "High"
            color = "red"
        
        st.success(f"**Prediction Result:**")
        st.write(f"Risk Score: **{risk:.3f}**")
        st.write(f"Risk Category: **:{color}[{category}]**")
    
    # Data table
    st.subheader("📊 Data Explorer")
    st.dataframe(filtered_data.head(100), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**EcoPredict** - Ecological Risk Prediction System | Test Dashboard")

if __name__ == "__main__":
    main()