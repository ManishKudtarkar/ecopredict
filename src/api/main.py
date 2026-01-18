"""
FastAPI main application for EcoPredict
Provides REST API endpoints for ecological risk prediction
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.helpers import load_config, validate_coordinates
from models.base_model import BaseModel
from prediction.predict import EcoPredictionEngine
from .schemas import (
    PredictionRequest, PredictionResponse, 
    RiskZoneRequest, RiskZoneResponse,
    HeatmapRequest, HeatmapResponse
)

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EcoPredict API",
    description="Ecological Risk Prediction API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models and config
prediction_engine = None
config = None


@app.on_event("startup")
async def startup_event():
    """Initialize models and configuration on startup"""
    global prediction_engine, config
    
    try:
        # Load configuration
        config = load_config("config/config.yaml")
        
        # Initialize prediction engine
        prediction_engine = EcoPredictionEngine()
        prediction_engine.load_models("models/trained")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "EcoPredict API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "risk-zones": "/risk-zones", 
            "heatmap": "/heatmap",
            "models": "/models",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    status = "healthy"
    details = {}
    
    try:
        # Check if prediction engine is loaded
        if prediction_engine is None:
            status = "unhealthy"
            details["prediction_engine"] = "not loaded"
        else:
            details["prediction_engine"] = "loaded"
            details["available_models"] = list(prediction_engine.models.keys())
        
        # Check configuration
        if config is None:
            status = "unhealthy"
            details["config"] = "not loaded"
        else:
            details["config"] = "loaded"
        
    except Exception as e:
        status = "unhealthy"
        details["error"] = str(e)
    
    return {
        "status": status,
        "details": details
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(request: PredictionRequest):
    """
    Predict ecological risk for given coordinates
    """
    
    if prediction_engine is None:
        raise HTTPException(status_code=503, detail="Prediction engine not available")
    
    try:
        # Validate coordinates
        if not validate_coordinates(request.latitude, request.longitude):
            raise HTTPException(status_code=400, detail="Invalid coordinates")
        
        # Make prediction
        result = prediction_engine.predict_single(
            latitude=request.latitude,
            longitude=request.longitude,
            model_name=request.model_name
        )
        
        return PredictionResponse(
            latitude=request.latitude,
            longitude=request.longitude,
            risk_score=result['risk_score'],
            risk_category=result['risk_category'],
            confidence=result.get('confidence', 0.0),
            model_used=result['model_name'],
            features_used=result.get('features', [])
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(coordinates: List[Dict[str, float]], model_name: Optional[str] = None):
    """
    Predict ecological risk for multiple coordinates
    """
    
    if prediction_engine is None:
        raise HTTPException(status_code=503, detail="Prediction engine not available")
    
    try:
        results = []
        
        for coord in coordinates:
            if 'latitude' not in coord or 'longitude' not in coord:
                raise HTTPException(status_code=400, detail="Each coordinate must have latitude and longitude")
            
            lat, lon = coord['latitude'], coord['longitude']
            
            if not validate_coordinates(lat, lon):
                results.append({
                    "latitude": lat,
                    "longitude": lon,
                    "error": "Invalid coordinates"
                })
                continue
            
            try:
                result = prediction_engine.predict_single(lat, lon, model_name)
                results.append({
                    "latitude": lat,
                    "longitude": lon,
                    "risk_score": result['risk_score'],
                    "risk_category": result['risk_category'],
                    "confidence": result.get('confidence', 0.0)
                })
            except Exception as e:
                results.append({
                    "latitude": lat,
                    "longitude": lon,
                    "error": str(e)
                })
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/risk-zones", response_model=RiskZoneResponse)
async def get_risk_zones(request: RiskZoneRequest):
    """
    Generate risk zones for a given area
    """
    
    if prediction_engine is None:
        raise HTTPException(status_code=503, detail="Prediction engine not available")
    
    try:
        # Validate bounds
        min_lat, min_lon = request.bounds[0], request.bounds[1]
        max_lat, max_lon = request.bounds[2], request.bounds[3]
        
        if not (validate_coordinates(min_lat, min_lon) and validate_coordinates(max_lat, max_lon)):
            raise HTTPException(status_code=400, detail="Invalid bounds")
        
        # Generate risk zones
        zones = prediction_engine.generate_risk_zones(
            bounds=request.bounds,
            resolution=request.resolution,
            model_name=request.model_name
        )
        
        return RiskZoneResponse(
            bounds=request.bounds,
            resolution=request.resolution,
            zones=zones,
            model_used=request.model_name or "default"
        )
        
    except Exception as e:
        logger.error(f"Risk zones error: {e}")
        raise HTTPException(status_code=500, detail=f"Risk zones generation failed: {str(e)}")


@app.post("/heatmap", response_model=HeatmapResponse)
async def generate_heatmap(request: HeatmapRequest):
    """
    Generate heatmap data for visualization
    """
    
    if prediction_engine is None:
        raise HTTPException(status_code=503, detail="Prediction engine not available")
    
    try:
        # Generate heatmap
        heatmap_data = prediction_engine.generate_heatmap(
            bounds=request.bounds,
            resolution=request.resolution,
            model_name=request.model_name
        )
        
        return HeatmapResponse(
            bounds=request.bounds,
            resolution=request.resolution,
            data=heatmap_data,
            model_used=request.model_name or "default"
        )
        
    except Exception as e:
        logger.error(f"Heatmap error: {e}")
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {str(e)}")


@app.get("/models")
async def get_available_models():
    """
    Get information about available models
    """
    
    if prediction_engine is None:
        raise HTTPException(status_code=503, detail="Prediction engine not available")
    
    try:
        models_info = {}
        
        for name, model in prediction_engine.models.items():
            models_info[name] = model.get_model_info()
        
        return {
            "available_models": models_info,
            "default_model": prediction_engine.default_model
        }
        
    except Exception as e:
        logger.error(f"Models info error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models info: {str(e)}")


@app.get("/models/{model_name}/importance")
async def get_feature_importance(model_name: str):
    """
    Get feature importance for a specific model
    """
    
    if prediction_engine is None:
        raise HTTPException(status_code=503, detail="Prediction engine not available")
    
    if model_name not in prediction_engine.models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    try:
        model = prediction_engine.models[model_name]
        importance = model.get_feature_importance()
        
        return {
            "model_name": model_name,
            "feature_importance": importance
        }
        
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")


@app.get("/config")
async def get_configuration():
    """
    Get current configuration (non-sensitive parts)
    """
    
    if config is None:
        raise HTTPException(status_code=503, detail="Configuration not available")
    
    # Return only non-sensitive configuration
    public_config = {
        "project": config.get("project", {}),
        "risk_thresholds": config.get("risk_thresholds", {}),
        "model_params": config.get("model_params", {})
    }
    
    return public_config


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )