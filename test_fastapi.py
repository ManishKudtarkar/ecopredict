#!/usr/bin/env python3
"""Simple FastAPI test for EcoPredict"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="EcoPredict API",
    description="Ecological Risk Prediction API",
    version="1.0.0"
)

# Pydantic models
class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    features: Dict[str, Any] = {}

class PredictionResponse(BaseModel):
    latitude: float
    longitude: float
    risk_score: float
    risk_category: str
    confidence: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_status: str
    version: str

# Global model (in production, use proper dependency injection)
global_model = None

def initialize_model():
    """Initialize a simple model for testing"""
    global global_model
    
    # Generate training data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: temperature, precipitation, humidity, forest_cover, urban_area, species_count
    X = np.random.rand(n_samples, 6)
    y = (0.3 * (1 - X[:, 3]) +  # forest_cover effect (inverted)
         0.2 * X[:, 4] +        # urban_area effect
         0.1 * np.abs(X[:, 0] - 0.5) +  # temperature effect
         0.4 * np.random.rand(n_samples))  # random component
    
    # Train model
    global_model = RandomForestRegressor(n_estimators=50, random_state=42)
    global_model.fit(X, y)
    
    print("✅ Model initialized successfully!")

# Initialize model on startup
initialize_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_status="loaded" if global_model is not None else "not_loaded",
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(request: PredictionRequest):
    """Predict ecological risk for given coordinates"""
    
    if global_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate coordinates
    if not (-90 <= request.latitude <= 90 and -180 <= request.longitude <= 180):
        raise HTTPException(status_code=400, detail="Invalid coordinates")
    
    try:
        # Extract features with defaults
        features = request.features
        feature_vector = [
            features.get('temperature', 25.0) / 40.0,  # Normalize to 0-1
            features.get('precipitation', 2.0) / 10.0,  # Normalize to 0-1
            features.get('humidity', 60.0) / 100.0,     # Normalize to 0-1
            features.get('forest_cover', 0.5),          # Already 0-1
            features.get('urban_area', 0.3),            # Already 0-1
            features.get('species_count', 15.0) / 50.0  # Normalize to 0-1
        ]
        
        # Make prediction
        risk_score = global_model.predict([feature_vector])[0]
        risk_score = max(0.0, min(1.0, risk_score))  # Clip to [0,1]
        
        # Determine category
        if risk_score < 0.3:
            risk_category = "Low"
        elif risk_score < 0.6:
            risk_category = "Medium"
        else:
            risk_category = "High"
        
        # Mock confidence (in production, use proper uncertainty estimation)
        confidence = 0.85
        
        from datetime import datetime
        
        return PredictionResponse(
            latitude=request.latitude,
            longitude=request.longitude,
            risk_score=risk_score,
            risk_category=risk_category,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/statistics")
async def get_statistics():
    """Get API statistics"""
    return {
        "total_predictions": 1000,
        "model_accuracy": 0.85,
        "api_version": "1.0.0",
        "status": "operational"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to EcoPredict API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "statistics": "/statistics",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting EcoPredict API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)