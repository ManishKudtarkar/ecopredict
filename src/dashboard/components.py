"""Dashboard components for EcoPredict"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any


def create_risk_gauge(risk_score: float, title: str = "Risk Score") -> go.Figure:
    """Create a risk gauge chart
    
    Args:
        risk_score: Risk score between 0 and 1
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 0.5},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.6], 'color': "yellow"},
                {'range': [0.6, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def create_species_chart(data: pd.DataFrame) -> go.Figure:
    """Create species diversity chart
    
    Args:
        data: DataFrame with species data
        
    Returns:
        Plotly figure
    """
    fig = px.histogram(
        data, 
        x='species_count',
        nbins=15,
        title="Species Count Distribution",
        color_discrete_sequence=['#2E8B57']
    )
    
    fig.update_layout(
        xaxis_title="Species Count",
        yaxis_title="Frequency",
        showlegend=False
    )
    
    return fig


def create_climate_trends(data: pd.DataFrame) -> go.Figure:
    """Create climate trends chart
    
    Args:
        data: DataFrame with climate data
        
    Returns:
        Plotly figure
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Temperature Distribution', 'Precipitation Distribution'),
        vertical_spacing=0.1
    )
    
    # Temperature histogram
    fig.add_trace(
        go.Histogram(
            x=data['temperature'],
            name='Temperature',
            marker_color='orange',
            nbinsx=20
        ),
        row=1, col=1
    )
    
    # Precipitation histogram
    fig.add_trace(
        go.Histogram(
            x=data['precipitation'],
            name='Precipitation',
            marker_color='blue',
            nbinsx=20
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        title_text="Climate Variable Distributions",
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Precipitation (mm)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    
    return fig


def create_land_use_pie(land_use_data: Dict[str, float]) -> go.Figure:
    """Create land use pie chart
    
    Args:
        land_use_data: Dictionary with land use percentages
        
    Returns:
        Plotly figure
    """
    labels = list(land_use_data.keys())
    values = list(land_use_data.values())
    
    colors = {
        'forest': '#228B22',
        'agricultural': '#DAA520',
        'urban': '#696969',
        'water': '#4169E1',
        'grassland': '#9ACD32',
        'barren': '#D2691E'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=[colors.get(label.lower(), '#808080') for label in labels],
        textinfo='label+percent',
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Land Use Distribution",
        height=400
    )
    
    return fig


def create_correlation_heatmap(data: pd.DataFrame, 
                              columns: List[str] = None) -> go.Figure:
    """Create correlation heatmap
    
    Args:
        data: DataFrame with numerical data
        columns: Columns to include in correlation
        
    Returns:
        Plotly figure
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = data[columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Variable Correlation Matrix",
        height=500,
        width=500
    )
    
    return fig


def create_time_series_chart(data: pd.DataFrame,
                           date_column: str,
                           value_columns: List[str],
                           title: str = "Time Series") -> go.Figure:
    """Create time series chart
    
    Args:
        data: DataFrame with time series data
        date_column: Name of date column
        value_columns: List of value columns to plot
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, column in enumerate(value_columns):
        fig.add_trace(go.Scatter(
            x=data[date_column],
            y=data[column],
            mode='lines+markers',
            name=column,
            line=dict(color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_scatter_matrix(data: pd.DataFrame,
                         columns: List[str],
                         color_column: str = None) -> go.Figure:
    """Create scatter plot matrix
    
    Args:
        data: DataFrame with data
        columns: Columns to include in matrix
        color_column: Column to use for coloring
        
    Returns:
        Plotly figure
    """
    fig = px.scatter_matrix(
        data,
        dimensions=columns,
        color=color_column,
        title="Scatter Plot Matrix"
    )
    
    fig.update_layout(height=600)
    return fig


def create_box_plot(data: pd.DataFrame,
                   x_column: str,
                   y_column: str,
                   title: str = "Box Plot") -> go.Figure:
    """Create box plot
    
    Args:
        data: DataFrame with data
        x_column: Column for x-axis (categorical)
        y_column: Column for y-axis (numerical)
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = px.box(
        data,
        x=x_column,
        y=y_column,
        title=title,
        color=x_column
    )
    
    fig.update_layout(height=400)
    return fig


def create_3d_scatter(data: pd.DataFrame,
                     x_column: str,
                     y_column: str,
                     z_column: str,
                     color_column: str = None,
                     title: str = "3D Scatter Plot") -> go.Figure:
    """Create 3D scatter plot
    
    Args:
        data: DataFrame with data
        x_column: Column for x-axis
        y_column: Column for y-axis
        z_column: Column for z-axis
        color_column: Column for coloring
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = px.scatter_3d(
        data,
        x=x_column,
        y=y_column,
        z=z_column,
        color=color_column,
        title=title,
        height=500
    )
    
    return fig


def create_density_map(data: pd.DataFrame,
                      lat_column: str = "latitude",
                      lon_column: str = "longitude",
                      title: str = "Density Map") -> go.Figure:
    """Create density map
    
    Args:
        data: DataFrame with coordinate data
        lat_column: Latitude column name
        lon_column: Longitude column name
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = px.density_mapbox(
        data,
        lat=lat_column,
        lon=lon_column,
        radius=10,
        center=dict(lat=data[lat_column].mean(), lon=data[lon_column].mean()),
        zoom=6,
        mapbox_style="open-street-map",
        title=title
    )
    
    fig.update_layout(height=500)
    return fig


def create_metric_cards(metrics: Dict[str, Any]) -> str:
    """Create HTML for metric cards
    
    Args:
        metrics: Dictionary of metric name -> value
        
    Returns:
        HTML string
    """
    cards_html = '<div style="display: flex; gap: 1rem; margin: 1rem 0;">'
    
    for name, value in metrics.items():
        cards_html += f'''
        <div class="metric-card" style="flex: 1; padding: 1rem; background: #f0f2f6; border-radius: 0.5rem;">
            <h4 style="margin: 0; color: #666;">{name}</h4>
            <h2 style="margin: 0.5rem 0 0 0; color: #2E8B57;">{value}</h2>
        </div>
        '''
    
    cards_html += '</div>'
    return cards_html