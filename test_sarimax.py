import requests

# Test SARIMAX
data = {'year': 2024, 'month': 7, 'model_type': 'sarimax'}
response = requests.post('http://127.0.0.1:8000/predict/', data=data)
print('Status:', response.status_code)
if response.status_code == 200:
    result = response.json()
    if result.get('success'):
        print(f'SARIMAX Prediction: {result["rainfall"]} mm')
        if 'metrics' in result and result['metrics']['mae']:
            print(f'MAE: {result["metrics"]["mae"]}, RMSE: {result["metrics"]["rmse"]}')
    else:
        print('Error:', result.get('error'))