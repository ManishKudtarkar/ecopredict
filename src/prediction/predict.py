"""
Prediction engine for EcoPredict
Handles model loading and prediction orchestration
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib

from ..utils.logger import get_logger
from ..utils.helpers import load_config, validate_coordinates, get_risk_category, create_grid
from ..models.base_model import BaseModel

logger = get_logger(__name__)


class EcoPredictionEngine:
    """Main prediction engine for ecological risk assessment"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path) if Path(config_path).exists() else {}
        self.models = {}
        self.default_model = None
        self.feature_columns = []
        self.risk_thresholds = self.config.get('risk_thresholds', {'low': 0.3, 'medium': 0.6})
    
    def load_models(self, models_dir: str) -> None:
        """
        Load trained models from directory
        
        Args:
            models_dir: Directory containing saved models
        """
        models_path = Path(models_dir)
        
        if not models_path.exists():
            logger.warning(f"Models directory {models_dir} does not exist")
            return
        
        # Find all model files
        model_files = list(models_path.glob("*.pkl"))
        
        if not model_files:
            logger.warning(f"No model files found in {models_dir}")
            return
        
        for model_file in model_files:
            try:
                # Load model
                model = BaseModel.load(str(model_file))
                model_name = model_file.stem.replace('_model', '')
                
                self.models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
                
                # Set first model as default
                if self.default_model is None:
                    self.default_model = model_name
                    self.feature_columns = model.feature_names
                
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")
        
        logger.info(f"Loaded {len(self.models)} models. Default: {self.default_model}")
    
    def predict_single(self, 
                      latitude: float, 
                      longitude: float,
                      model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict ecological risk for a single location
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            model_name: Name of model to use (uses default if None)
            
        Returns:
            Dictionary with prediction results
        """
        
        # Validate coordinates
        if not validate_coordinates(latitude, longitude):
            raise ValueError("Invalid coordinates")
        
        # Select model
        if model_name is None:
            model_name = self.default_model
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available")
        
        model = self.models[model_name]
        
        # Generate features for this location
        features = self._generate_features(latitude, longitude)
        
        # Make prediction
        risk_score = model.predict(features)[0]
        
        # Determine risk category
        risk_category = get_risk_category(risk_score, self.risk_thresholds)
        
        # Calculate confidence (simplified)
        confidence = self._calculate_confidence(features, model)
        
        return {
            'risk_score': float(risk_score),
            'risk_category': risk_category,
            'confidence': float(confidence),
            'model_name': model_name,
            'features': list(features.columns)
        }
    
    def predict_batch(self, 
                     coordinates: List[Tuple[float, float]],
                     model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Predict ecological risk for multiple locations
        
        Args:
            coordinates: List of (latitude, longitude) tuples
            model_name: Name of model to use
            
        Returns:
            List of prediction results
        """
        
        results = []
        
        for lat, lon in coordinates:
            try:
                result = self.predict_single(lat, lon, model_name)
                result.update({'latitude': lat, 'longitude': lon})
                results.append(result)
            except Exception as e:
                results.append({
                    'latitude': lat,
                    'longitude': lon,
                    'error': str(e)
                })
        
        return results
    
    def generate_risk_zones(self, 
                           bounds: Tuple[float, float, float, float],
                           resolution: float = 0.1,
                           model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate risk zones for a geographic area
        
        Args:
            bounds: (min_lat, min_lon, max_lat, max_lon)
            resolution: Grid resolution in degrees
            model_name: Name of model to use
            
        Returns:
            List of risk zones as GeoJSON-like features
        """
        
        # Create prediction grid
        grid_df = create_grid(bounds, resolution)
        
        # Make predictions for all grid points
        predictions = []
        
        for _, row in grid_df.iterrows():
            try:
                result = self.predict_single(row['latitude'], row['longitude'], model_name)
                predictions.append({
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'risk_score': result['risk_score'],
                    'risk_category': result['risk_category']
                })
            except Exception as e:
                logger.warning(f"Prediction failed for {row['latitude']}, {row['longitude']}: {e}")
        
        # Group predictions by risk category and create zones
        zones = self._create_risk_zones(predictions, resolution)
        
        return zones
    
    def generate_heatmap(self, 
                        bounds: Tuple[float, float, float, float],
                        resolution: float = 0.05,
                        model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate heatmap data for visualization
        
        Args:
            bounds: (min_lat, min_lon, max_lat, max_lon)
            resolution: Grid resolution in degrees
            model_name: Name of model to use
            
        Returns:
            List of heatmap data points
        """
        
        # Create prediction grid
        grid_df = create_grid(bounds, resolution)
        
        # Limit grid size for performance
        if len(grid_df) > 10000:
            logger.warning(f"Grid too large ({len(grid_df)} points), sampling 10000 points")
            grid_df = grid_df.sample(n=10000, random_state=42)
        
        # Make predictions
        heatmap_data = []
        
        for _, row in grid_df.iterrows():
            try:
                result = self.predict_single(row['latitude'], row['longitude'], model_name)
                heatmap_data.append({
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'risk_score': result['risk_score'],
                    'risk_category': result['risk_category']
                })
            except Exception as e:
                logger.warning(f"Prediction failed for {row['latitude']}, {row['longitude']}: {e}")
        
        return heatmap_data
    
    def _generate_features(self, latitude: float, longitude: float) -> pd.DataFrame:
        """
        Generate features for a given location
        This is a simplified version - in practice, you'd load actual environmental data
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            DataFrame with features
        """
        
        # Generate synthetic features based on location
        # In a real implementation, you would:
        # 1. Query climate databases
        # 2. Extract land use data from GIS layers
        # 3. Query species occurrence databases
        # 4. Calculate derived features
        
        # For demonstration, generate plausible synthetic features
        np.random.seed(int((latitude + longitude) * 1000) % 2**32)
        
        # Climate features (vary with latitude)
        temperature = 20 + (latitude - 15) * 1.5 + np.random.normal(0, 2)
        precipitation = max(0, 800 + (latitude - 19) * 100 + np.random.normal(0, 200))
        humidity = np.clip(60 + np.random.normal(0, 15), 0, 100)
        wind_speed = max(0, np.random.exponential(3))
        
        # Land use features (vary with distance from urban centers)
        # Assume Mumbai is at (19.0760, 72.8777)
        dist_to_mumbai = np.sqrt((latitude - 19.0760)**2 + (longitude - 72.8777)**2)
        
        if dist_to_mumbai < 0.5:  # Urban area
            forest_cover = np.random.uniform(0.05, 0.2)
            agricultural_area = np.random.uniform(0.1, 0.3)
            urban_area = np.random.uniform(0.4, 0.8)
            water_bodies = np.random.uniform(0.02, 0.1)
        elif dist_to_mumbai < 1.5:  # Suburban
            forest_cover = np.random.uniform(0.2, 0.5)
            agricultural_area = np.random.uniform(0.3, 0.6)
            urban_area = np.random.uniform(0.1, 0.3)
            water_bodies = np.random.uniform(0.05, 0.15)
        else:  # Rural
            forest_cover = np.random.uniform(0.4, 0.8)
            agricultural_area = np.random.uniform(0.1, 0.4)
            urban_area = np.random.uniform(0.01, 0.1)
            water_bodies = np.random.uniform(0.05, 0.2)
        
        # Normalize land use to sum to 1
        total_landuse = forest_cover + agricultural_area + urban_area + water_bodies
        if total_landuse > 1:
            forest_cover /= total_landuse
            agricultural_area /= total_landuse
            urban_area /= total_landuse
            water_bodies /= total_landuse
        
        # Species features (higher diversity in forested areas)
        species_count = max(0, int(forest_cover * 50 + np.random.poisson(10)))
        endemic_species = max(0, int(species_count * np.random.uniform(0.05, 0.15)))
        threatened_species = max(0, int(species_count * np.random.uniform(0.02, 0.08)))
        
        # Create feature dictionary
        features_dict = {
            'temperature': temperature,
            'precipitation': precipitation,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'forest_cover': forest_cover,
            'agricultural_area': agricultural_area,
            'urban_area': urban_area,
            'water_bodies': water_bodies,
            'species_count': species_count,
            'endemic_species': endemic_species,
            'threatened_species': threatened_species
        }
        
        # Add derived features that might have been created during training
        features_dict['distance_from_center'] = dist_to_mumbai
        features_dict['habitat_quality_index'] = (
            forest_cover * 0.6 + water_bodies * 0.3 - urban_area * 0.4
        )
        features_dict['fragmentation_index'] = urban_area / (forest_cover + 0.001)
        
        # Create DataFrame with only the features the model expects
        if self.feature_columns:
            # Use only features that the model was trained on
            model_features = {}
            for feature in self.feature_columns:
                if feature in features_dict:
                    model_features[feature] = features_dict[feature]
                else:
                    # Set missing features to 0 or reasonable defaults
                    model_features[feature] = 0.0
            
            features_df = pd.DataFrame([model_features])
        else:
            # Use all generated features
            features_df = pd.DataFrame([features_dict])
        
        return features_df
    
    def _calculate_confidence(self, features: pd.DataFrame, model: BaseModel) -> float:
        """
        Calculate prediction confidence (simplified implementation)
        
        Args:
            features: Input features
            model: Trained model
            
        Returns:
            Confidence score between 0 and 1
        """
        
        # This is a simplified confidence calculation
        # In practice, you might use:
        # - Ensemble variance
        # - Distance to training data
        # - Model-specific uncertainty measures
        
        try:
            # For tree-based models, use feature importance alignment
            if hasattr(model.model, 'feature_importances_'):
                importances = model.model.feature_importances_
                # Higher confidence when important features have typical values
                confidence = 0.7 + np.random.uniform(0, 0.3)  # Simplified
            else:
                confidence = 0.8  # Default confidence
            
            return min(1.0, max(0.0, confidence))
            
        except Exception:
            return 0.5  # Default confidence if calculation fails
    
    def _create_risk_zones(self, 
                          predictions: List[Dict[str, Any]], 
                          resolution: float) -> List[Dict[str, Any]]:
        """
        Create risk zones from prediction grid
        
        Args:
            predictions: List of prediction results
            resolution: Grid resolution
            
        Returns:
            List of risk zones as GeoJSON-like features
        """
        
        zones = []
        
        # Group predictions by risk category
        risk_groups = {'low': [], 'medium': [], 'high': []}
        
        for pred in predictions:
            risk_category = pred['risk_category']
            if risk_category in risk_groups:
                risk_groups[risk_category].append(pred)
        
        # Create zones for each risk category
        for risk_level, points in risk_groups.items():
            if not points:
                continue
            
            # For simplicity, create rectangular zones
            # In practice, you'd use more sophisticated spatial clustering
            for point in points:
                lat, lon = point['latitude'], point['longitude']
                
                # Create a small rectangle around each point
                zone = {
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [[
                            [lon - resolution/2, lat - resolution/2],
                            [lon + resolution/2, lat - resolution/2],
                            [lon + resolution/2, lat + resolution/2],
                            [lon - resolution/2, lat + resolution/2],
                            [lon - resolution/2, lat - resolution/2]
                        ]]
                    },
                    'properties': {
                        'risk_level': risk_level,
                        'risk_score': point['risk_score'],
                        'center_lat': lat,
                        'center_lon': lon
                    }
                }
                
                zones.append(zone)
        
        logger.info(f"Created {len(zones)} risk zones")
        return zones
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        
        info = {
            'loaded_models': list(self.models.keys()),
            'default_model': self.default_model,
            'risk_thresholds': self.risk_thresholds
        }
        
        if self.models:
            info['model_details'] = {}
            for name, model in self.models.items():
                info['model_details'][name] = model.get_model_info()
        
        return info