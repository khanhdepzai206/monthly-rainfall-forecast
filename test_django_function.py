#!/usr/bin/env python
import os
import sys

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rainfall_project.settings')
sys.path.insert(0, os.path.dirname(__file__))

import django
django.setup()

print("Testing in Django environment...")

try:
    # Replicate the Django view logic
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DuBao'))
    from src.run_pipeline import get_daily_predictions

    print("About to call get_daily_predictions...")
    pred_rf, pred_lr, pred_xgb = get_daily_predictions()
    print("Success: RF=" + str(pred_rf) + ", LR=" + str(pred_lr) + ", XGB=" + str(pred_xgb))

except Exception as e:
    print("Error: " + str(e))
    import traceback
    traceback.print_exc()