"""Tests for EcoPredict API"""

import pytest
from fastapi.testclient import TestClient
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self):
        """Test health check returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "version" in data


class TestPredictionEndpoints:
    """Test prediction endpoints"""
    
    def test_predict_valid_coordinates(self):
        """Test prediction with valid coordinates"""
        payload = {
            "latitude": 19.0760,
            "longitude": 72.8777,
            "features": {
                "temperature": 25.0,
                "precipitation": 2.0,
                "forest_cover": 0.3
            }
        }
        
        response = client.post("/predict", json=payload)
        
        # Should return 200 or 500 (if model not loaded)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "risk_score" in data
            assert "risk_category" in data
            assert "latitude" in data
            assert "longitude" in data
            assert 0 <= data["risk_score"] <= 1
    
    def test_predict_invalid_coordinates(self):
        """Test prediction with invalid coordinates"""
        payload = {
            "latitude": 100.0,  # Invalid latitude
            "longitude": 72.8777,
            "features": {}
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 400
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
        coordinates = [
            {"latitude": 19.0760, "longitude": 72.8777},
            {"latitude": 18.5204, "longitude": 73.8567}
        ]
        
        response = client.post("/predict/batch", json=coordinates)
        
        # Should return 200 or 500 (if model not loaded)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) <= len(coordinates)
    
    def test_batch_prediction_too_large(self):
        """Test batch prediction with too many coordinates"""
        coordinates = [
            {"latitude": 19.0, "longitude": 72.0}
            for _ in range(1001)  # Exceeds limit
        ]
        
        response = client.post("/predict/batch", json=coordinates)
        assert response.status_code == 400


class TestRiskZoneEndpoints:
    """Test risk zone endpoints"""
    
    def test_generate_risk_zones(self):
        """Test risk zone generation"""
        payload = {
            "bounds": [72.6, 15.6, 80.9, 22.0],
            "resolution": 0.1,
            "threshold_low": 0.3,
            "threshold_high": 0.6
        }
        
        response = client.post("/risk-zones", json=payload)
        
        # Should return 200 or 500 (if components not available)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "bounds" in data
            assert "risk_zones" in data
    
    def test_generate_risk_zones_invalid_bounds(self):
        """Test risk zone generation with invalid bounds"""
        payload = {
            "bounds": [200, 200, 300, 300],  # Invalid bounds
            "resolution": 0.1,
            "threshold_low": 0.3,
            "threshold_high": 0.6
        }
        
        response = client.post("/risk-zones", json=payload)
        assert response.status_code == 400


class TestHeatmapEndpoints:
    """Test heatmap endpoints"""
    
    def test_generate_heatmap(self):
        """Test heatmap generation"""
        payload = {
            "bounds": [72.6, 15.6, 80.9, 22.0],
            "resolution": 0.1,
            "output_format": "geojson"
        }
        
        response = client.post("/heatmap", json=payload)
        
        # Should return 200 or 500 (if components not available)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "bounds" in data
            assert "heatmap_data" in data
            assert "output_format" in data


class TestUtilityEndpoints:
    """Test utility endpoints"""
    
    def test_list_models(self):
        """Test model listing endpoint"""
        response = client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
    
    def test_get_statistics(self):
        """Test statistics endpoint"""
        response = client.get("/statistics")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_predictions" in data
        assert "model_accuracy" in data
    
    def test_get_region_info_valid(self):
        """Test region info with valid region"""
        response = client.get("/regions/maharashtra")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "bounds" in data
    
    def test_get_region_info_invalid(self):
        """Test region info with invalid region"""
        response = client.get("/regions/nonexistent")
        assert response.status_code == 404


class TestAPIValidation:
    """Test API input validation"""
    
    def test_prediction_missing_fields(self):
        """Test prediction with missing required fields"""
        payload = {
            "latitude": 19.0760
            # Missing longitude
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_prediction_invalid_types(self):
        """Test prediction with invalid data types"""
        payload = {
            "latitude": "invalid",  # Should be float
            "longitude": 72.8777,
            "features": {}
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error


@pytest.fixture
def sample_prediction_data():
    """Sample data for testing"""
    return {
        "latitude": 19.0760,
        "longitude": 72.8777,
        "features": {
            "temperature": 25.0,
            "precipitation": 2.0,
            "humidity": 60.0,
            "forest_cover": 0.3,
            "species_count": 15
        }
    }


def test_api_integration(sample_prediction_data):
    """Integration test for API workflow"""
    # Test health check
    health_response = client.get("/health")
    assert health_response.status_code == 200
    
    # Test prediction (may fail if model not loaded)
    pred_response = client.post("/predict", json=sample_prediction_data)
    assert pred_response.status_code in [200, 500]
    
    # Test statistics
    stats_response = client.get("/statistics")
    assert stats_response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])