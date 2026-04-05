#!/usr/bin/env python
import joblib
import os
import pandas as pd

# Load a model and check its feature names
models_dir = 'DuBao/models'
rf_model = joblib.load(os.path.join(models_dir, 'rf_daily_model.pkl'))

# For sklearn models, feature names are stored in the model
if hasattr(rf_model, 'feature_names_in_'):
    print('RF model feature names:')
    print(rf_model.feature_names_in_)
elif hasattr(rf_model, 'n_features_in_'):
    print(f'RF model has {rf_model.n_features_in_} features')
else:
    print('Cannot determine feature names for RF model')

# Check the data file
df = pd.read_csv('DuBao/data/daily_features.csv')
exclude = {'date', 'target', 'datetime', 'rainfall'}
feature_cols = [c for c in df.columns if c not in exclude]
print(f'\nData has {len(feature_cols)} features: {feature_cols[:10]}...')

# Check the inner regressor
print(f'\nModel type: {type(rf_model)}')
if hasattr(rf_model, 'regressor_'):
    inner_model = rf_model.regressor_
    print(f'Inner model type: {type(inner_model)}')
    if hasattr(inner_model, 'feature_names_in_'):
        print('Inner model feature names:')
        print(inner_model.feature_names_in_)
    elif hasattr(inner_model, 'n_features_in_'):
        print(f'Inner model has {inner_model.n_features_in_} features')
        
    # For pipeline, check the final estimator
    if hasattr(inner_model, 'steps'):
        final_step = inner_model.steps[-1][1]
        print(f'Final estimator type: {type(final_step)}')
        if hasattr(final_step, 'feature_names_in_'):
            print('Final estimator feature names:')
            print(final_step.feature_names_in_)