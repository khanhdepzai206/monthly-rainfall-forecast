import os
from pathlib import Path

print('='*60)
print('FEATURE IMPLEMENTATION VERIFICATION')
print('='*60)

# 1. Check API view exists
print('\n[1] API Endpoint')
api_file = Path('predictor/api_views.py')
if api_file.exists():
    with open(api_file, encoding='utf-8') as f:
        content = f.read()
        if 'predict_compare_models_api' in content:
            print('✓ API endpoint function exists')
        else:
            print('✗ API endpoint function NOT found')

# 2. Check URL routing
print('\n[2] URL Routing')
urls_file = Path('predictor/urls.py')
if urls_file.exists():
    with open(urls_file, encoding='utf-8') as f:
        content = f.read()
        if 'api/predict-compare' in content:
            print('✓ Route /api/predict-compare/ mapped')
        else:
            print('✗ Route NOT mapped')

# 3. Check template
print('\n[3] Frontend Template')
template_file = Path('templates/predict.html')
if template_file.exists():
    with open(template_file, encoding='utf-8') as f:
        content = f.read()
        checks = {
            'compare-form': 'Form ID found',
            'compare-models-body': 'Results table found',
            'displayComparisonResult': 'JavaScript handler found',
            'consensus-rain': 'Consensus section found'
        }
        for check, desc in checks.items():
            if check in content:
                print(f'✓ {desc}')
            else:
                print(f'✗ {desc} NOT found')

# 4. Check models
print('\n[4] Trained Models')
models_dir = Path('DuBao/models')
required_models = [
    'classifier_gradientboosting.pkl',
    'classifier_randomforest.pkl',
    'classifier_xgboost.pkl',
    'regressor_gradientboosting.pkl',
    'regressor_randomforest.pkl',
    'regressor_xgboost.pkl'
]
all_exist = True
for model in required_models:
    model_path = models_dir / model
    if model_path.exists():
        print(f'✓ {model}')
    else:
        print(f'✗ {model} NOT found')
        all_exist = False

# 5. Check data
print('\n[5] Feature Data')
data_file = Path('DuBao/data/daily_combined.csv')
if data_file.exists():
    print(f'✓ daily_combined.csv exists')
    size = data_file.stat().st_size / 1024 / 1024
    print(f'  Size: {size:.2f} MB')
else:
    print('✗ daily_combined.csv NOT found')

# 6. Check documentation
print('\n[6] Documentation Files')
docs = [
    'MODEL_COMPARISON_GUIDE.md',
    'IMPLEMENTATION_SUMMARY.md',
    'QUICK_START.md'
]
for doc in docs:
    doc_path = Path(doc)
    if doc_path.exists():
        size = doc_path.stat().st_size / 1024
        print(f'✓ {doc} ({size:.1f} KB)')
    else:
        print(f'✗ {doc} NOT found')

print('\n' + '='*60)
print('VERIFICATION COMPLETE')
print('='*60)
