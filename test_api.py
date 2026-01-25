import requests
import json

# Test API với các mô hình mới
models = ['gradient_boosting_weather', 'random_forest_weather', 'sarimax']

for model in models:
    print(f"\n=== Testing {model} ===")
    data = {
        'year': 2024,
        'month': 7,
        'model_type': model
    }

    try:
        response = requests.post('http://127.0.0.1:8000/predict/', data=data)
        print('Status Code:', response.status_code)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"Prediction: {result['rainfall']} mm")
                if 'metrics' in result and result['metrics']['mae']:
                    print(f"MAE: {result['metrics']['mae']}, RMSE: {result['metrics']['rmse']}")
            else:
                print('Error:', result.get('error'))
        else:
            print('HTTP Error')
    except Exception as e:
        print('Error:', e)