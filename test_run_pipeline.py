#!/usr/bin/env python
from run_pipeline import get_daily_predictions
print("Testing get_daily_predictions from DuBao/run_pipeline.py...")
result = get_daily_predictions()
print("Success:", result)