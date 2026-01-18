"""
Pydantic schemas for EcoPredict API
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class RiskCategory(str, Enum):
    """Risk category enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredictionRequest(BaseModel):
    """Request schema for single prediction"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    model_name: Optional[str] = Field(None, description="Model to use for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "latitude": 19.0760,
                "longitude": 72.8777,
                "model_name": "random_forest"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for single prediction"""
    latitude: float
    longitude: float
    risk_score: float = Field(..., ge=0, le=1, description="Risk score between 0 and 1")
    risk_category: RiskCategory
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    model_used: str
    features_used: List[str] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "latitude": 19.0760,
                "longitude": 72.8777,
                "risk_score": 0.65,
                "risk_category": "medium",
                "confidence": 0.85,
                "model_used": "random_forest",
                "features_used": ["temperature", "precipitation", "forest_cover"]
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions"""
    coordinates: List[Dict[str, float]] = Field(..., description="List of coordinate dictionaries")
    model_name: Optional[str] = Field(None, description="Model to use for predictions")
    
    @validator('coordinates')
    def validate_coordinates(cls, v):
        for coord in v:
            if 'latitude' not in coord or 'longitude' not in coord:
                raise ValueError("Each coordinate must have 'latitude' and 'longitude'")
            if not (-90 <= coord['latitude'] <= 90):
                raise ValueError("Latitude must be between -90 and 90")
            if not (-180 <= coord['longitude'] <= 180):
                raise ValueError("Longitude must be between -180 and 180")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "coordinates": [
                    {"latitude": 19.0760, "longitude": 72.8777},
                    {"latitude": 18.5204, "longitude": 73.8567}
                ],
                "model_name": "random_forest"
            }
        }


class RiskZoneRequest(BaseModel):
    """Request schema for risk zone generation"""
    bounds: Tuple[float, float, float, float] = Field(
        ..., 
        description="Bounding box as (min_lat, min_lon, max_lat, max_lon)"
    )
    resolution: float = Field(0.1, gt=0, le=1, description="Grid resolution in degrees")
    model_name: Optional[str] = Field(None, description="Model to use for predictions")
    
    @validator('bounds')
    def validate_bounds(cls, v):
        if len(v) != 4:
            raise ValueError("Bounds must have exactly 4 values")
        min_lat, min_lon, max_lat, max_lon = v
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError("Latitudes must be between -90 and 90")
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError("Longitudes must be between -180 and 180")
        if min_lat >= max_lat:
            raise ValueError("min_lat must be less than max_lat")
        if min_lon >= max_lon:
            raise ValueError("min_lon must be less than max_lon")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "bounds": [18.0, 72.0, 20.0, 74.0],
                "resolution": 0.1,
                "model_name": "random_forest"
            }
        }


class RiskZone(BaseModel):
    """Individual risk zone"""
    geometry: Dict[str, Any] = Field(..., description="GeoJSON geometry")
    properties: Dict[str, Any] = Field(..., description="Zone properties including risk level")


class RiskZoneResponse(BaseModel):
    """Response schema for risk zones"""
    bounds: Tuple[float, float, float, float]
    resolution: float
    zones: List[RiskZone]
    model_used: str
    
    class Config:
        schema_extra = {
            "example": {
                "bounds": [18.0, 72.0, 20.0, 74.0],
                "resolution": 0.1,
                "zones": [
                    {
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[72.0, 18.0], [72.1, 18.0], [72.1, 18.1], [72.0, 18.1], [72.0, 18.0]]]
                        },
                        "properties": {
                            "risk_level": "low",
                            "risk_score": 0.25
                        }
                    }
                ],
                "model_used": "random_forest"
            }
        }


class HeatmapRequest(BaseModel):
    """Request schema for heatmap generation"""
    bounds: Tuple[float, float, float, float] = Field(
        ..., 
        description="Bounding box as (min_lat, min_lon, max_lat, max_lon)"
    )
    resolution: float = Field(0.05, gt=0, le=0.5, description="Grid resolution in degrees")
    model_name: Optional[str] = Field(None, description="Model to use for predictions")
    
    @validator('bounds')
    def validate_bounds(cls, v):
        if len(v) != 4:
            raise ValueError("Bounds must have exactly 4 values")
        min_lat, min_lon, max_lat, max_lon = v
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError("Latitudes must be between -90 and 90")
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError("Longitudes must be between -180 and 180")
        if min_lat >= max_lat:
            raise ValueError("min_lat must be less than max_lat")
        if min_lon >= max_lon:
            raise ValueError("min_lon must be less than max_lon")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "bounds": [18.0, 72.0, 20.0, 74.0],
                "resolution": 0.05,
                "model_name": "random_forest"
            }
        }


class HeatmapPoint(BaseModel):
    """Individual heatmap data point"""
    latitude: float
    longitude: float
    risk_score: float = Field(..., ge=0, le=1)
    risk_category: RiskCategory


class HeatmapResponse(BaseModel):
    """Response schema for heatmap data"""
    bounds: Tuple[float, float, float, float]
    resolution: float
    data: List[HeatmapPoint]
    model_used: str
    
    class Config:
        schema_extra = {
            "example": {
                "bounds": [18.0, 72.0, 20.0, 74.0],
                "resolution": 0.05,
                "data": [
                    {
                        "latitude": 18.0,
                        "longitude": 72.0,
                        "risk_score": 0.35,
                        "risk_category": "low"
                    }
                ],
                "model_used": "random_forest"
            }
        }


class ModelInfo(BaseModel):
    """Model information schema"""
    model_name: str
    model_type: str
    is_trained: bool
    num_features: int
    feature_names: List[str]
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]


class ModelsResponse(BaseModel):
    """Response schema for available models"""
    available_models: Dict[str, ModelInfo]
    default_model: str


class FeatureImportanceResponse(BaseModel):
    """Response schema for feature importance"""
    model_name: str
    feature_importance: Dict[str, float]


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    detail: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Validation Error",
                "detail": "Invalid coordinates provided"
            }
        }