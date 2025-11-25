#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify age category fixes
"""

import sys
import json
sys.path.insert(0, '/mnt/project')

from ctis_utils import is_pediatric_trial, is_adult_trial
from ctis_report_generator import decode_age_categories
from ctis_config import AGE_CATEGORY_MAP

print("Testing Age Category Fixes")
print("=" * 60)

# Test 1: decode_age_categories with integer codes (as stored in DB)
print("\nTest 1: Decode age categories with integer codes")
age_json_int = json.dumps([6, 7, 8])
print(f"Input: {age_json_int}")
decoded = decode_age_categories(age_json_int)
print(f"Output: {decoded}")
print(f"Expected: 18-64 years,65-84 years,85+ years")
print(f"Pass: {decoded == '18-64 years,65-84 years,85+ years'}")

# Test 2: decode_age_categories with string codes
print("\nTest 2: Decode age categories with string codes")
age_json_str = json.dumps(["6", "7", "8"])
print(f"Input: {age_json_str}")
decoded = decode_age_categories(age_json_str)
print(f"Output: {decoded}")
print(f"Expected: 18-64 years,65-84 years,85+ years")
print(f"Pass: {decoded == '18-64 years,65-84 years,85+ years'}")

# Test 3: is_adult_trial with integer codes
print("\nTest 3: is_adult_trial with integer codes")
age_codes_int = [6, 7, 8]
print(f"Input: {age_codes_int}")
result = is_adult_trial(age_codes_int)
print(f"Output: {result}")
print(f"Expected: True")
print(f"Pass: {result == True}")

# Test 4: is_adult_trial with string codes
print("\nTest 4: is_adult_trial with string codes")
age_codes_str = ["6", "7", "8"]
print(f"Input: {age_codes_str}")
result = is_adult_trial(age_codes_str)
print(f"Output: {result}")
print(f"Expected: True")
print(f"Pass: {result == True}")

# Test 5: is_pediatric_trial with integer codes (should be False)
print("\nTest 5: is_pediatric_trial with integer codes [6,7,8]")
age_codes_int = [6, 7, 8]
print(f"Input: {age_codes_int}")
result = is_pediatric_trial(age_codes_int)
print(f"Output: {result}")
print(f"Expected: False")
print(f"Pass: {result == False}")

# Test 6: is_pediatric_trial with pediatric codes
print("\nTest 6: is_pediatric_trial with pediatric integer codes")
age_codes_ped = [3, 4, 5]
print(f"Input: {age_codes_ped}")
result = is_pediatric_trial(age_codes_ped)
print(f"Output: {result}")
print(f"Expected: True")
print(f"Pass: {result == True}")

# Test 7: Mixed pediatric and adult
print("\nTest 7: is_adult_trial and is_pediatric_trial with mixed codes")
age_codes_mixed = [5, 6, 7]
print(f"Input: {age_codes_mixed}")
is_adult = is_adult_trial(age_codes_mixed)
is_ped = is_pediatric_trial(age_codes_mixed)
print(f"is_adult: {is_adult}, is_pediatric: {is_ped}")
print(f"Expected: is_adult=True, is_pediatric=True")
print(f"Pass: {is_adult == True and is_ped == True}")

print("\n" + "=" * 60)
print("All tests completed!")