import pickle

with open('DuBao/models/sarimax_model.pkl', 'rb') as f:
    data = pickle.load(f)
print('Type of data:', type(data))
if isinstance(data, dict):
    print('Keys:', data.keys())
    if 'model' in data:
        print('Type of model:', type(data['model']))
        print('Has get_forecast:', hasattr(data['model'], 'get_forecast'))