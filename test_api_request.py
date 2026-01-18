#!/usr/bin/env python3
"""Test API requests"""

import requests
import json

def test_api():
    """Test the EcoPredict API"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing EcoPredict API...")
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test prediction endpoint
    print("\n2. Testing prediction endpoint...")
    try:
        prediction_data = {
            "latitude": 19.0760,
            "longitude": 72.8777,
            "features": {
                "temperature": 25.0,
                "precipitation": 2.0,
                "humidity": 65.0,
                "forest_cover": 0.4,
                "urban_area": 0.3,
                "species_count": 15
            }
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=prediction_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Risk Score: {result['risk_score']:.3f}")
            print(f"   Risk Category: {result['risk_category']}")
            print(f"   Confidence: {result['confidence']:.2f}")
        else:
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test statistics endpoint
    print("\n3. Testing statistics endpoint...")
    try:
        response = requests.get(f"{base_url}/statistics")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n✅ API testing completed!")

if __name__ == "__main__":
    test_api()