#!/usr/bin/env python
"""Test the predict page HTML interface"""
import requests
from bs4 import BeautifulSoup

try:
    # Test accessing the predict page
    print("Testing predict page access...")
    response = requests.get("http://127.0.0.1:8000/predict/")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        # Check if the comparison form is present
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for the comparison section
        comparison_form = soup.find(lambda tag: tag.name == 'h3' and '⚔️' in tag.text)
        if comparison_form:
            print("✓ Comparison form section found in HTML")
            # Look for input fields
            year_input = soup.find('input', {'name': 'compare_year'})
            month_input = soup.find('input', {'name': 'compare_month'})
            day_input = soup.find('input', {'name': 'compare_day'})
            
            if year_input and month_input and day_input:
                print("✓ All input fields found (year, month, day)")
            else:
                print("✗ Missing input fields")
        else:
            print("✗ Comparison form section not found")
    else:
        print(f"✗ Failed to access predict page: {response.status_code}")
        
except Exception as e:
    print(f"✗ Error: {e}")
