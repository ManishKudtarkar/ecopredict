"""API routes for EcoPredict"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path

from .schemas import (
    PredictionRequest, PredictionResponse, 
    RiskZoneRequest, RiskZoneResponse,
    HeatmapRequest, HeatmapResponse,
    HealthResponse
)
from ..prediction.predict import EcoPredictionEngine
from ..gis.risk_zones import RiskZoneAnalyzer
from ..gis.heatmap import HeatmapGenerator
from ..utils.logger import get_logger
from ..utils.helpers import validate_coordinates

logger = get_logger(__name__)

# Initialize router
router = APIRouter()

# Global instances (in production, use dependency injection)
prediction_engine = None
risk_analyzer = None
heatmap_generator = None


def get_prediction_engine():
    """Get prediction engine instance"""
    global prediction_engine
    if prediction_engine is None:
        try:
            prediction_engine = EcoPredictionEngine()
            prediction_engine.load_model("models/trained/best_model.joblib")
        except Exception as e:
            logger.error(f"Failed to load prediction engine: {e}")
            raise HTTPException(status_code=500, detail="Prediction engine not available")
    return prediction_engine


def get_risk_analyzer():
    """Get risk zone analyzer instance"""
    global risk_analyzer
    if risk_analyzer is None:
        risk_analyzer = RiskZoneAnalyzer()
    return risk_analyzer


def get_heatmap_generator():
    """Get heatmap generator instance"""
    global heatmap_generator
    if heatmap_generator is None:
        heatmap_generator = HeatmapGenerator()
    return heatmap_generator


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check if models are loaded
        engine = get_prediction_engine()
        model_status = "loaded" if engine.model is not None else "not_loaded"
        
        return HealthResponse(
            status="healthy",
            model_status=model_status,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_status="error",
            version="1.0.0"
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict_risk(
    request: PredictionRequest,
    engine: EcoPredictionEngine = Depends(get_prediction_engine)
):
    """Predict ecological risk for given coordinates"""
    try:
        # Validate coordinates
        if not validate_coordinates(request.latitude, request.longitude):
            raise HTTPException(status_code=400, detail="Invalid coordinates")
        
        # Make prediction
        prediction_result = engine.predict_single(
            latitude=request.latitude,
            longitude=request.longitude,
            features=request.features
        )
        
        return PredictionResponse(
            latitude=request.latitude,
            longitude=request.longitude,
            risk_score=prediction_result['risk_score'],
            risk_category=prediction_result['risk_category'],
            confidence=prediction_result.get('confidence', 0.0),
            contributing_factors=prediction_result.get('contributing_factors', {}),
            timestamp=pd.Timestamp.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(
    coordinates: List[Dict[str, float]],
    features: Optional[Dict[str, Any]] = None,
    engine: EcoPredictionEngine = Depends(get_prediction_engine)
):
    """Batch prediction for multiple coordinates"""
    try:
        if len(coordinates) > 1000:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size too large (max 1000)")
        
        results = []
        
        for coord in coordinates:
            if not validate_coordinates(coord['latitude'], coord['longitude']):
                continue
            
            prediction_result = engine.predict_single(
                latitude=coord['latitude'],
                longitude=coord['longitude'],
                features=features
            )
            
            results.append(PredictionResponse(
                latitude=coord['latitude'],
                longitude=coord['longitude'],
                risk_score=prediction_result['risk_score'],
                risk_category=prediction_result['risk_category'],
                confidence=prediction_result.get('confidence', 0.0),
                contributing_factors=prediction_result.get('contributing_factors', {}),
                timestamp=pd.Timestamp.now().isoformat()
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.post("/risk-zones", response_model=RiskZoneResponse)
async def generate_risk_zones(
    request: RiskZoneRequest,
    analyzer: RiskZoneAnalyzer = Depends(get_risk_analyzer)
):
    """Generate risk zone boundaries"""
    try:
        # Validate bounds
        bounds = request.bounds
        if not (validate_coordinates(bounds[1], bounds[0]) and 
                validate_coordinates(bounds[3], bounds[2])):
            raise HTTPException(status_code=400, detail="Invalid bounds")
        
        # Generate risk zones
        risk_zones = analyzer.generate_risk_zones(
            bounds=bounds,
            resolution=request.resolution,
            threshold_low=request.threshold_low,
            threshold_high=request.threshold_high
        )
        
        return RiskZoneResponse(
            bounds=bounds,
            risk_zones=risk_zones,
            resolution=request.resolution,
            timestamp=pd.Timestamp.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Risk zone generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Risk zone generation failed: {str(e)}")


@router.post("/heatmap", response_model=HeatmapResponse)
async def generate_heatmap(
    request: HeatmapRequest,
    generator: HeatmapGenerator = Depends(get_heatmap_generator)
):
    """Generate risk heatmap"""
    try:
        # Validate bounds
        bounds = request.bounds
        if not (validate_coordinates(bounds[1], bounds[0]) and 
                validate_coordinates(bounds[3], bounds[2])):
            raise HTTPException(status_code=400, detail="Invalid bounds")
        
        # Generate heatmap
        heatmap_data = generator.generate_heatmap(
            bounds=bounds,
            resolution=request.resolution,
            output_format=request.output_format
        )
        
        return HeatmapResponse(
            bounds=bounds,
            heatmap_data=heatmap_data,
            resolution=request.resolution,
            output_format=request.output_format,
            timestamp=pd.Timestamp.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Heatmap generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {str(e)}")


@router.get("/models")
async def list_available_models():
    """List available trained models"""
    try:
        models_dir = Path("models/trained")
        if not models_dir.exists():
            return {"models": []}
        
        models = []
        for model_file in models_dir.glob("*.joblib"):
            models.append({
                "name": model_file.stem,
                "path": str(model_file),
                "size": model_file.stat().st_size,
                "modified": model_file.stat().st_mtime
            })
        
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.get("/statistics")
async def get_prediction_statistics():
    """Get prediction statistics and model performance"""
    try:
        # This would typically come from a database or cache
        # For now, return mock statistics
        stats = {
            "total_predictions": 10000,
            "predictions_today": 150,
            "average_risk_score": 0.35,
            "high_risk_areas": 25,
            "model_accuracy": 0.87,
            "last_model_update": "2026-01-15T10:30:00Z"
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


@router.get("/regions/{region_name}")
async def get_region_info(region_name: str):
    """Get information about a specific region"""
    try:
        # Mock region data - in production, this would come from a database
        regions = {
            "maharashtra": {
                "name": "Maharashtra",
                "bounds": [68.0, 15.6, 80.9, 22.0],
                "area_km2": 307713,
                "population": 112374333,
                "biodiversity_hotspots": 15,
                "protected_areas": 50,
                "average_risk_score": 0.42
            }
        }
        
        region_data = regions.get(region_name.lower())
        if not region_data:
            raise HTTPException(status_code=404, detail="Region not found")
        
        return region_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get region info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get region info")