#!/usr/bin/env python
"""Comprehensive test of the 3-model comparison feature"""
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_prediction_endpoint():
    """Test the /api/predict-compare/ endpoint"""
    print("\n" + "="*70)
    print("TESTING 3-MODEL COMPARISON API ENDPOINT")
    print("="*70)
    
    test_data = {
        "year": 2024,
        "month": 5,
        "day": 15
    }
    
    print(f"\nRequest:")
    print(f"  URL: POST {BASE_URL}/api/predict-compare/")
    print(f"  Data: {json.dumps(test_data, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict-compare/",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nResponse:")
        print(f"  Status Code: {response.status_code}")
        print(f"  Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Valid JSON response")
            
            if data.get('success'):
                print(f"\n✓ SUCCESS - Prediction completed")
                print(f"  Date: {data.get('date')}")
                print(f"\n  Models:")
                for model in data.get('models', []):
                    print(f"    - {model['model']}")
                    print(f"      Has Rain: {'☔ Có' if model['has_rain'] else '☀️ Không'}")
                    print(f"      Probability: {model['rain_probability']*100:.2f}%")
                    print(f"      Predicted Rainfall: {model['predicted_rainfall']:.2f} mm")
                
                consensus = data.get('consensus', {})
                print(f"\n  Consensus:")
                print(f"    Has Rain: {'☔ Có mưa' if consensus.get('has_rain') else '☀️ Không mưa'}")
                print(f"    Avg Probability: {consensus.get('avg_rain_probability', 0)*100:.2f}%")
                print(f"    Avg Rainfall: {consensus.get('avg_rainfall', 0):.2f} mm")
                print(f"    Agreement: {consensus.get('agreement_count', 0)}/3 models agree")
                return True
            else:
                print(f"✗ FAILED - {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"✗ FAILED - Status {response.status_code}")
            print(f"  Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

def test_html_page():
    """Test that the predict HTML page loads correctly"""
    print("\n" + "="*70)
    print("TESTING HTML PAGE ACCESS")
    print("="*70)
    
    print(f"\nRequest: GET {BASE_URL}/predict/")
    
    try:
        response = requests.get(f"{BASE_URL}/predict/")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            html = response.text
            
            checks = {
                "Form ID 'compare-form'": 'id="compare-form"' in html,
                "Year input 'compare-year'": 'id="compare-year"' in html,
                "Month select 'compare-month'": 'id="compare-month"' in html,
                "Day input 'compare-day'": 'id="compare-day"' in html,
                "Compare button": 'id="compare-btn"' in html or 'submit' in html,
                "Output div 'compare-output'": 'id="compare-output"' in html,
                "Models table 'compare-models-body'": 'id="compare-models-body"' in html,
                "Consensus section": 'id="consensus-rain"' in html,
                "JavaScript fetch handler": 'fetch' in html and '/api/predict-compare/' in html
            }
            
            print("\nPage Components:")
            all_pass = True
            for check, result in checks.items():
                status = "✓" if result else "✗"
                print(f"  {status} {check}")
                if not result:
                    all_pass = False
            
            if all_pass:
                print("\n✓ SUCCESS - All page components present")
                return True
            else:
                print("\n⚠ WARNING - Some components missing")
                return False
        else:
            print(f"✗ FAILED - Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

def test_multiple_dates():
    """Test with different dates"""
    print("\n" + "="*70)
    print("TESTING MULTIPLE DATES")
    print("="*70)
    
    test_dates = [
        {"year": 2023, "month": 6, "day": 1},
        {"year": 2023, "month": 8, "day": 15},
        {"year": 2024, "month": 1, "day": 10},
    ]
    
    all_pass = True
    for test_data in test_dates:
        try:
            response = requests.post(
                f"{BASE_URL}/api/predict-compare/",
                json=test_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    date_str = data.get('date')
                    print(f"✓ {date_str}: {data['consensus']['avg_rain_probability']*100:.1f}% rain probability")
                else:
                    print(f"✗ {test_data}: {data.get('error')}")
                    all_pass = False
            else:
                print(f"✗ {test_data}: Status {response.status_code}")
                all_pass = False
        except Exception as e:
            print(f"✗ {test_data}: {e}")
            all_pass = False
    
    if all_pass:
        print("\n✓ SUCCESS - All dates processed successfully")
        return True
    return False

if __name__ == "__main__":
    print("\n" + "█"*70)
    print("  RAINFALL PREDICTION - 3 MODEL COMPARISON TEST SUITE")
    print("█"*70)
    
    # Wait for server to be ready
    max_retries = 5
    for i in range(max_retries):
        try:
            requests.get(f"{BASE_URL}/predict/", timeout=2)
            break
        except:
            if i < max_retries - 1:
                print(f"Waiting for server... ({i+1}/{max_retries})")
                time.sleep(1)
    
    # Run tests
    # run range API tests in addition
    def test_range_api():
        print("\n" + "="*70)
        print("TESTING RANGE PREDICTION API")
        print("="*70)
        try:
            resp = requests.get(f"{BASE_URL}/api/predict-range/?year=2024&month=5&start_day=1&num_days=3", timeout=5)
            print(f"Status Code: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"Response count: {data.get('count')}")
                return True
            else:
                print(f"Failed: {resp.text[:200]}")
                return False
        except Exception as e:
            print(f"Error: {e}")
            return False

    results = {
        "HTML Page Access": test_html_page(),
        "API Endpoint": test_prediction_endpoint(),
        "Multiple Dates": test_multiple_dates(),
        "Range API": test_range_api(),
    }
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - System is working correctly!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed - Review output above")
