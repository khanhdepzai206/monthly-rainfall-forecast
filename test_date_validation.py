#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test date validation"""
import requests
import json

print("=" * 60)
print("TEST DATE VALIDATION")
print("=" * 60)

print("\n✅ Test 1: Ngày hợp lệ (01/04/2026)")
r = requests.post('http://127.0.0.1:8000/predict/', data={'year': 2026, 'month': 4, 'day': 1})
data = r.json()
print(f"Status: {r.status_code}")
print(f"Success: {data.get('success')}")
if data.get('success'):
    print(f"Dự đoán: {data.get('date_label')}")

print("\n❌ Test 2: Ngày vượt quá (03/04/2026)")
r = requests.post('http://127.0.0.1:8000/predict/', data={'year': 2026, 'month': 4, 'day': 3})
data = r.json()
print(f"Status: {r.status_code}")
print(f"Success: {data.get('success')}")
print(f"Error: {data.get('error')}")

print("\n✅ Test 3: Ngày cũ (15/01/2024)")
r = requests.post('http://127.0.0.1:8000/predict/', data={'year': 2024, 'month': 1, 'day': 15})
data = r.json()
print(f"Status: {r.status_code}")
print(f"Success: {data.get('success')}")
if data.get('success'):
    print(f"Dự đoán: {data.get('date_label')}")

print("\n" + "=" * 60)
