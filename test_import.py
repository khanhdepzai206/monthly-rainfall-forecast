import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rainfall_project.settings')
sys.path.insert(0, '.')
django.setup()

from predictor.api_views import predict_compare_models_api
print('✓ API view imported successfully')
print(f'Function: {predict_compare_models_api.__name__}')

# Check if models directory exists
from pathlib import Path
models_dir = Path('./DuBao/models')
print(f'\nModels directory: {models_dir.absolute()}')
print(f'Models directory exists: {models_dir.exists()}')

if models_dir.exists():
    model_files = list(models_dir.glob('*.pkl'))
    print(f'Model files found: {len(model_files)}')
    for f in model_files:
        print(f'  - {f.name}')

# Check data directory
data_dir = Path('./DuBao/data')
print(f'\nData directory: {data_dir.absolute()}')
print(f'Data directory exists: {data_dir.exists()}')

if data_dir.exists():
    data_files = list(data_dir.glob('*.csv'))
    print(f'Data files found: {len(data_files)}')
    for f in data_files:
        print(f'  - {f.name}')
