"""
Test script for Django web interface
"""

import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://127.0.0.1:8000"

def test_daily_predict():
    """Test daily prediction endpoint"""
    print("Testing daily prediction...")

    try:
        response = requests.post(f"{BASE_URL}/daily-predict/")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_actual_input():
    """Test actual input endpoint"""
    print("\nTesting actual input...")

    # Test data
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    test_data = {
        'actual_date': yesterday,
        'actual_rainfall': 2.5
    }

    try:
        response = requests.post(f"{BASE_URL}/actual-input/", data=test_data)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_get_pages():
    """Test GET requests to pages"""
    print("\nTesting GET pages...")

    pages = ['/daily-predict/', '/actual-input/']

    for page in pages:
        try:
            response = requests.get(f"{BASE_URL}{page}")
            print(f"{page}: {response.status_code}")
        except Exception as e:
            print(f"{page}: Error - {str(e)}")

if __name__ == "__main__":
    print("🌧️ Testing Django Web Interface")
    print("=" * 40)

    # Test GET pages
    test_get_pages()

    # Test POST endpoints
    predict_ok = test_daily_predict()
    actual_ok = test_actual_input()

    print("\n" + "=" * 40)
    if predict_ok and actual_ok:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")

    print("\n📋 Test Results:")
    print(f"Daily Predict: {'✅' if predict_ok else '❌'}")
    print(f"Actual Input: {'✅' if actual_ok else '❌'}")