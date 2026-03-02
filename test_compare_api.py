#!/usr/bin/env python3
"""
Test script for 3-model comparison API endpoint
"""
import os
import sys
import json
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rainfall_project.settings')
sys.path.insert(0, 'd:\\Du Bao Luong Mua')
django.setup()

from django.test import Client
from django.test.client import RequestFactory

def test_compare_api():
    """Test the /api/predict-compare/ endpoint"""
    client = Client()
    
    test_cases = [
        {"year": 2024, "month": 5, "day": 15},
        {"year": 2023, "month": 1, "day": 1},
        {"year": 2020, "month": 8, "day": 30},
    ]
    
    print("=" * 60)
    print("Testing /api/predict-compare/ Endpoint")
    print("=" * 60)
    
    for i, test_data in enumerate(test_cases, 1):
        print(f"\n[Test {i}] Testing with data: {test_data}")
        print("-" * 60)
        
        try:
            response = client.post(
                '/api/predict-compare/',
                data=json.dumps(test_data),
                content_type='application/json'
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Response received successfully")
                print(f"  - Date: {data.get('date', 'N/A')}")
                print(f"  - Number of models: {len(data.get('models', []))}")
                
                if data.get('models'):
                    for model in data['models']:
                        print(f"\n  {model['model']}:")
                        print(f"    - Has Rain: {model['has_rain']}")
                        print(f"    - Rain Probability: {model['rain_probability']:.4f}")
                        print(f"    - Predicted Rainfall: {model['predicted_rainfall']:.2f} mm")
                        print(f"    - MAE: {model['mae']:.4f}")
                        print(f"    - RMSE: {model['rmse']:.4f}")
                        print(f"    - R² Score: {model['r2_score']:.4f}")
                
                if data.get('consensus'):
                    print(f"\n  Consensus:")
                    print(f"    - Has Rain: {data['consensus']['has_rain']}")
                    print(f"    - Avg Rain Probability: {data['consensus']['avg_rain_probability']:.4f}")
                    print(f"    - Avg Rainfall: {data['consensus']['avg_rainfall']:.2f} mm")
                    print(f"    - Agreement Count: {data['consensus']['agreement_count']}/3")
            else:
                print(f"✗ Error: {response.content.decode()}")
                
        except Exception as e:
            print(f"✗ Exception: {e}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == '__main__':
    test_compare_api()
