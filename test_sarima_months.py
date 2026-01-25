import requests

# Test SARIMA cho các tháng khác nhau
months = [1, 7, 12]
for month in months:
    data = {'year': 2024, 'month': month, 'model_type': 'sarimax'}
    response = requests.post('http://127.0.0.1:8000/predict/', data=data)
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            print(f'Tháng {month}/2024: {result["rainfall"]} mm')
        else:
            print(f'Tháng {month}: Error - {result.get("error")}')
    else:
        print(f'Tháng {month}: HTTP {response.status_code}')