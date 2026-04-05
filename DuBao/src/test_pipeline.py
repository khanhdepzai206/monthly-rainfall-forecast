#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for the complete updated rainfall prediction pipeline.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_pipeline():
    print('=== Testing Complete Pipeline ===')

    try:
        print('1. Testing data preparation...')
        from prepare_daily_data import prepare_daily_data
        prepare_daily_data()
        print('   ✓ Data preparation completed')

        print('2. Testing model training...')
        from train_daily_models import train_daily_models
        results = train_daily_models()
        print(f'   ✓ Models trained: {list(results.keys())}')

        print('3. Testing auto-retrain...')
        from auto_retrain import check_and_retrain
        check_and_retrain()
        print('   ✓ Retrain check completed')

        print('4. Testing predictions...')
        from run_pipeline import get_daily_predictions
        preds = get_daily_predictions()
        print(f'   ✓ Predictions: {preds}')

        print('✅ Pipeline test completed successfully!')

    except Exception as e:
        print(f'❌ Error during testing: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()