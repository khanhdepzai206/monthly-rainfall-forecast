import os
import sys
import django
import json

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rainfall_project.settings')
sys.path.insert(0, '.')
django.setup()

from django.test import Client

client = Client()

print("Testing /api/predict-compare/ endpoint...")
print("=" * 60)

test_data = {"year": 2024, "month": 5, "day": 15}

try:
    response = client.post(
        '/api/predict-compare/',
        data=json.dumps(test_data),
        content_type='application/json'
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.get('content-type', 'N/A')}")
    print(f"\nResponse Content:")
    print(response.content.decode()[:500])
    
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"\n✓ Valid JSON Response:")
            print(json.dumps(data, indent=2))
        except:
            print(f"\n✗ Response is not valid JSON")
    else:
        print(f"\n✗ Error response")
        
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()
