#!/usr/bin/env python3
"""Simple test to demonstrate EcoPredict functionality"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def generate_ecological_data(n_samples=1000):
    """Generate synthetic ecological data"""
    print("🌍 Generating synthetic ecological data...")
    
    np.random.seed(42)
    
    # Maharashtra bounds
    lat_range = (15.6, 22.0)
    lon_range = (72.6, 80.9)
    
    data = {
        # Location
        'latitude': np.random.uniform(lat_range[0], lat_range[1], n_samples),
        'longitude': np.random.uniform(lon_range[0], lon_range[1], n_samples),
        
        # Climate variables
        'temperature': np.random.normal(25, 5, n_samples),
        'precipitation': np.random.exponential(2, n_samples),
        'humidity': np.random.normal(60, 15, n_samples),
        
        # Land use
        'forest_cover': np.random.uniform(0, 1, n_samples),
        'agricultural_area': np.random.uniform(0, 1, n_samples),
        'urban_area': np.random.uniform(0, 1, n_samples),
        
        # Biodiversity
        'species_count': np.random.poisson(15, n_samples),
        'endemic_species': np.random.poisson(2, n_samples),
        'threatened_species': np.random.poisson(1, n_samples),
        
        # Other factors
        'elevation': np.random.normal(500, 300, n_samples),
        'population_density': np.random.exponential(100, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate ecological risk score based on multiple factors
    risk_score = (
        0.25 * (1 - df['forest_cover']) +  # Deforestation increases risk
        0.20 * df['urban_area'] +  # Urbanization increases risk
        0.15 * np.abs(df['temperature'] - 25) / 10 +  # Temperature extremes
        0.10 * (1 / (df['species_count'] + 1)) +  # Low biodiversity = high risk
        0.10 * df['population_density'] / 1000 +  # Human pressure
        0.10 * (df['threatened_species'] / (df['species_count'] + 1)) +  # Threat ratio
        0.10 * np.random.normal(0, 0.1, n_samples)  # Random variation
    )
    
    # Normalize to 0-1 range
    df['ecological_risk'] = np.clip(risk_score, 0, 1)
    
    # Create risk categories
    df['risk_category'] = pd.cut(
        df['ecological_risk'], 
        bins=[0, 0.3, 0.6, 1.0], 
        labels=['Low', 'Medium', 'High']
    )
    
    print(f"✅ Generated {len(df)} ecological data points")
    print(f"   Risk score range: {df['ecological_risk'].min():.3f} - {df['ecological_risk'].max():.3f}")
    
    return df

def explore_data(df):
    """Explore the ecological data"""
    print("\n📊 Data Exploration:")
    
    # Basic statistics
    print(f"   Dataset shape: {df.shape}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    # Risk distribution
    risk_counts = df['risk_category'].value_counts()
    print(f"\n   Risk Distribution:")
    for category, count in risk_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   - {category}: {count} ({percentage:.1f}%)")
    
    # Key correlations with risk
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['ecological_risk'].abs().sort_values(ascending=False)
    
    print(f"\n   Top factors correlated with ecological risk:")
    for factor, corr in correlations.head(6).items():
        if factor != 'ecological_risk':
            print(f"   - {factor}: {corr:.3f}")

def train_models(df):
    """Train machine learning models"""
    print("\n🤖 Training Machine Learning Models...")
    
    # Prepare features and target
    feature_cols = [
        'temperature', 'precipitation', 'humidity', 'forest_cover',
        'agricultural_area', 'urban_area', 'species_count', 'endemic_species',
        'threatened_species', 'elevation', 'population_density'
    ]
    
    X = df[feature_cols]
    y = df['ecological_risk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = {}
    results = {}
    
    # Random Forest Model
    print("   Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    rf_pred = rf_model.predict(X_test)
    rf_metrics = {
        'r2': r2_score(y_test, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'mae': mean_absolute_error(y_test, rf_pred)
    }
    
    models['Random Forest'] = rf_model
    results['Random Forest'] = rf_metrics
    
    print(f"   ✅ Random Forest - R²: {rf_metrics['r2']:.3f}, RMSE: {rf_metrics['rmse']:.3f}")
    
    # Linear Regression Model
    print("   Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    lr_pred = lr_model.predict(X_test)
    lr_metrics = {
        'r2': r2_score(y_test, lr_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'mae': mean_absolute_error(y_test, lr_pred)
    }
    
    models['Linear Regression'] = lr_model
    results['Linear Regression'] = lr_metrics
    
    print(f"   ✅ Linear Regression - R²: {lr_metrics['r2']:.3f}, RMSE: {lr_metrics['rmse']:.3f}")
    
    return models, results, X_test, y_test, feature_cols

def make_predictions(models, feature_cols):
    """Make sample predictions"""
    print("\n🎯 Making Sample Predictions...")
    
    # Sample locations in Maharashtra
    sample_locations = [
        {'name': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777},
        {'name': 'Pune', 'lat': 18.5204, 'lon': 73.8567},
        {'name': 'Nagpur', 'lat': 21.1458, 'lon': 79.0882},
        {'name': 'Nashik', 'lat': 19.9975, 'lon': 73.7898}
    ]
    
    best_model = models['Random Forest']  # Use Random Forest as it typically performs better
    
    for location in sample_locations:
        # Generate sample environmental features for this location
        features = {
            'temperature': np.random.normal(26, 3),
            'precipitation': np.random.exponential(2.5),
            'humidity': np.random.normal(65, 10),
            'forest_cover': np.random.uniform(0.1, 0.8),
            'agricultural_area': np.random.uniform(0.1, 0.6),
            'urban_area': np.random.uniform(0.1, 0.5),
            'species_count': np.random.poisson(12),
            'endemic_species': np.random.poisson(1),
            'threatened_species': np.random.poisson(1),
            'elevation': np.random.normal(400, 200),
            'population_density': np.random.exponential(150)
        }
        
        # Make prediction
        feature_vector = np.array([[features[col] for col in feature_cols]])
        risk_score = best_model.predict(feature_vector)[0]
        
        # Determine risk category
        if risk_score < 0.3:
            risk_category = "Low"
        elif risk_score < 0.6:
            risk_category = "Medium"
        else:
            risk_category = "High"
        
        print(f"   📍 {location['name']}: Risk Score = {risk_score:.3f} ({risk_category} Risk)")

def visualize_results(df, models, results):
    """Create visualizations"""
    print("\n📈 Creating Visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EcoPredict Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Risk Score Distribution
    axes[0, 0].hist(df['ecological_risk'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(df['ecological_risk'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["ecological_risk"].mean():.3f}')
    axes[0, 0].set_xlabel('Ecological Risk Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Ecological Risk Scores')
    axes[0, 0].legend()
    
    # 2. Risk Categories
    risk_counts = df['risk_category'].value_counts()
    colors = ['green', 'orange', 'red']
    axes[0, 1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                  colors=colors, startangle=90)
    axes[0, 1].set_title('Risk Category Distribution')
    
    # 3. Model Performance Comparison
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    rmse_scores = [results[name]['rmse'] for name in model_names]
    
    x_pos = np.arange(len(model_names))
    axes[1, 0].bar(x_pos, r2_scores, color=['lightblue', 'lightcoral'])
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('Model Performance (R² Score)')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(model_names)
    
    # Add value labels on bars
    for i, v in enumerate(r2_scores):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 4. Feature Importance (Random Forest)
    rf_model = models['Random Forest']
    feature_cols = [
        'temperature', 'precipitation', 'humidity', 'forest_cover',
        'agricultural_area', 'urban_area', 'species_count', 'endemic_species',
        'threatened_species', 'elevation', 'population_density'
    ]
    
    importance = rf_model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1][:8]  # Top 8 features
    
    axes[1, 1].bar(range(len(sorted_idx)), importance[sorted_idx], color='lightgreen')
    axes[1, 1].set_xlabel('Features')
    axes[1, 1].set_ylabel('Importance')
    axes[1, 1].set_title('Top Feature Importance (Random Forest)')
    axes[1, 1].set_xticks(range(len(sorted_idx)))
    axes[1, 1].set_xticklabels([feature_cols[i] for i in sorted_idx], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('ecopredict_results.png', dpi=300, bbox_inches='tight')
    print("   ✅ Visualization saved as 'ecopredict_results.png'")
    plt.show()

def main():
    """Main function to run EcoPredict demonstration"""
    print("=" * 60)
    print("🌍 EcoPredict - Ecological Risk Prediction System")
    print("=" * 60)
    
    try:
        # Step 1: Generate data
        df = generate_ecological_data(n_samples=2000)
        
        # Step 2: Explore data
        explore_data(df)
        
        # Step 3: Train models
        models, results, X_test, y_test, feature_cols = train_models(df)
        
        # Step 4: Make predictions
        make_predictions(models, feature_cols)
        
        # Step 5: Visualize results
        visualize_results(df, models, results)
        
        print("\n" + "=" * 60)
        print("✅ EcoPredict demonstration completed successfully!")
        print("\nKey Results:")
        print(f"   • Analyzed {len(df)} ecological data points")
        print(f"   • Trained {len(models)} machine learning models")
        print(f"   • Best model R² score: {max(results[name]['r2'] for name in results):.3f}")
        print(f"   • Generated predictions for major Maharashtra cities")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()