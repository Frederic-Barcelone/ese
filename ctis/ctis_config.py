#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTIS Configuration Module
Contains all configuration constants, patterns, and mappings
ctis/ctis_congif.py
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Optional

# ===================== API Configuration =====================

DEFAULT_BASE = os.environ.get("CTIS_BASE", "https://euclinicaltrials.eu")
BASE = DEFAULT_BASE.rstrip("/")
SEARCH_URL = f"{BASE}/ctis-public-api/search"
DETAIL_URL = f"{BASE}/ctis-public-api/retrieve/{{ct}}"
PORTAL_URL = f"{BASE}/search-for-clinical-trials/?lang=en"

# ===================== Output Paths (defaults) =====================

OUT_DIR = Path("ctis-out")
NDJSON_PATH = OUT_DIR / "ctis_full.ndjson"
DB_PATH = OUT_DIR / "ctis.db"
CTNUMBERS_PATH = OUT_DIR / "ct_numbers.txt"
FAILED_PATH = OUT_DIR / "failed_ctnumbers.txt"

# ===================== Request Configuration =====================

PAGE_SIZE = 100
MAX_WORKERS = 3
MAX_RETRIES = 6
BASE_BACKOFF = 1.0
JITTER_RANGE = (0.15, 0.45)
FINAL_COOLDOWN = 1.0
REPORT_EVERY = 50
RATE_LIMIT_RPS = 2.0
REQUEST_TIMEOUT = 60.0

# ===================== Search Parameters =====================

STATUS_SEGMENTS = [1, 2, 3, 4, 5, 6, 7, 8]
YEAR_START = 2019

# ===================== HTTP Headers =====================

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)

BASE_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-GB,en;q=0.9",
    "Content-Type": "application/json",
    "Origin": BASE,
    "Referer": PORTAL_URL,
    "Connection": "keep-alive",
}

# ===================== Regex Patterns =====================

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{6,}\d)")

# ===================== JSON Data Loading =====================

def load_ctis_mappings():
    """
    Load CTIS mappings from JSON files.
    Looks for files in the project directory first, then falls back to legacy mappings.
    
    Returns:
        tuple: (key_mappings, list_values_all)
    """
    # Try to find JSON files in project directory
    project_dir = Path(__file__).parent
    key_mappings_path = project_dir / "CTIS_Key_Mappings.json"
    list_values_path = project_dir / "CTIS_List_Values_All.json"
    
    key_mappings = {}
    list_values_all = {}
    
    # Load key mappings
    if key_mappings_path.exists():
        try:
            with open(key_mappings_path, 'r', encoding='utf-8') as f:
                key_mappings = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {key_mappings_path}: {e}")
    
    # Load full list values
    if list_values_path.exists():
        try:
            with open(list_values_path, 'r', encoding='utf-8') as f:
                list_values_all = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {list_values_path}: {e}")
    
    return key_mappings, list_values_all


def extract_code_name_mapping(list_values_all, ref_name):
    """
    Extract code->name mapping from the full list values data.
    
    Args:
        list_values_all: Dictionary with all CTIS list values
        ref_name: Reference entity name (e.g., 'Subject Age Range')
    
    Returns:
        dict: Mapping of code to name
    """
    if ref_name not in list_values_all:
        return {}
    
    mapping = {}
    items = list_values_all[ref_name].get('data', [])
    columns = list_values_all[ref_name].get('columns', [])
    
    if len(columns) < 2:
        return {}
    
    code_col = columns[0]  # Usually 'CODE'
    name_col = columns[1]  # Usually 'NAME'
    
    for item in items:
        code = str(item.get(code_col, '')).strip()
        name = str(item.get(name_col, '')).strip()
        
        if code and name:
            mapping[code] = name
    
    return mapping


# Load mappings from JSON files
_KEY_MAPPINGS, _LIST_VALUES_ALL = load_ctis_mappings()

# ===================== Data Mappings =====================

# Age Category Mapping - IMPORTANT: Custom mapping (1-8) used by extractors
# Note: The CTIS list values use codes 0-7 for "Subject Age Range secondary identifier",
# but this codebase internally uses 1-8 (with code 6 representing "18-64 years").
# The mapping below aligns with the internal representation used throughout the codebase.
AGE_CATEGORY_MAP = {
    "1": "Preterm newborn",
    "2": "Newborns (0-27 days)",
    "3": "Infants and toddlers (28 days-23 months)",
    "4": "Children (2-11 years)",
    "5": "Adolescents (12-17 years)",
    "6": "18-64 years",  # Note: Not in CTIS secondary identifier, but used internally
    "7": "65-84 years",
    "8": "85+ years"
}

# Phase Mapping - Load from JSON with fallback
if _KEY_MAPPINGS and 'trial_phases' in _KEY_MAPPINGS:
    # Use the key mappings which have cleaner phase descriptions
    _PHASE_RAW = _KEY_MAPPINGS['trial_phases']
    # Simplify phase names for display
    PHASE_MAP = {}
    for code, name in _PHASE_RAW.items():
        if 'Phase I' in name and 'Phase II' not in name:
            PHASE_MAP[code] = 'Phase I'
        elif 'Phase II' in name and 'Phase III' not in name and 'Phase I' not in name:
            PHASE_MAP[code] = 'Phase II'
        elif 'Phase III' in name and 'Phase IV' not in name and 'Phase II' not in name:
            PHASE_MAP[code] = 'Phase III'
        elif 'Phase IV' in name:
            PHASE_MAP[code] = 'Phase IV'
        elif 'Phase I and Phase II' in name:
            PHASE_MAP[code] = 'Phase I/II'
        elif 'Phase II and Phase III' in name:
            PHASE_MAP[code] = 'Phase II/III'
        elif 'phase III and phase IV' in name:
            PHASE_MAP[code] = 'Phase III/IV'
        else:
            PHASE_MAP[code] = name
else:
    # Fallback to legacy mapping
    PHASE_MAP = {
        "1": "Phase I",
        "2": "Phase I",  # Bioequivalence
        "3": "Phase I",  # Other
        "4": "Phase II",
        "5": "Phase III",
        "6": "Phase IV",
        "7": "Phase I/II",
        "8": "Phase I/II",  # Bioequivalence integrated
        "9": "Phase I/II",  # Other integrated
        "10": "Phase II/III",
        "11": "Phase III/IV",
    }

# Trial Status Mapping - Load from JSON with fallback
if _KEY_MAPPINGS and 'trial_status' in _KEY_MAPPINGS:
    TRIAL_STATUS_MAP = _KEY_MAPPINGS['trial_status']
else:
    # Fallback to legacy mapping
    TRIAL_STATUS_MAP = {
        "1": "Pending",
        "2": "Cancelled",
        "3": "Under evaluation",
        "4": "Withdrawn",
        "5": "Lapsed",
        "6": "Not Valid",
        "7": "Authorised",
        "8": "Not authorised",
        "9": "Expired",
        "10": "Halted",
        "11": "Suspended",
        "12": "Ended",
        "13": "Revoked"
    }

# Trial Category Mapping - Load from JSON with fallback
if _KEY_MAPPINGS and 'trial_categories' in _KEY_MAPPINGS:
    TRIAL_CATEGORY_MAP = _KEY_MAPPINGS['trial_categories']
else:
    # Fallback to legacy mapping
    TRIAL_CATEGORY_MAP = {
        "1": "Category 1",
        "2": "Category 2",
        "3": "Category 3"
    }

# Therapeutic Areas Mapping - Load from JSON
if _KEY_MAPPINGS and 'therapeutic_areas' in _KEY_MAPPINGS:
    THERAPEUTIC_AREAS_MAP = _KEY_MAPPINGS['therapeutic_areas']
else:
    THERAPEUTIC_AREAS_MAP = {}

# EEA Countries Mapping - Load from JSON
if _KEY_MAPPINGS and 'eea_countries' in _KEY_MAPPINGS:
    EEA_COUNTRIES_MAP = _KEY_MAPPINGS['eea_countries']
else:
    EEA_COUNTRIES_MAP = {}

# Age Range (broader categories) - Load from JSON
if _LIST_VALUES_ALL and 'Subject Age Range' in _LIST_VALUES_ALL:
    AGE_RANGE_MAP = extract_code_name_mapping(_LIST_VALUES_ALL, 'Subject Age Range')
elif _KEY_MAPPINGS and 'age_ranges' in _KEY_MAPPINGS:
    AGE_RANGE_MAP = _KEY_MAPPINGS['age_ranges']
else:
    AGE_RANGE_MAP = {
        "1": "In utero",
        "2": "0-17 years",
        "3": "18-64 years",
        "4": "65+ years"
    }

# MSC Public Status Code Map - This is NOT in the CTIS list values, keep as is
MSC_PUBLIC_STATUS_CODE_MAP = {
    1: "Authorised, recruitment pending",
    2: "Authorised, recruiting", 
    3: "Authorised, no longer recruiting",
    4: "Temporarily halted",
    5: "Ongoing, recruitment ended",  # Most common ongoing status
    6: "Restarted",
    7: "Withdrawn",
    8: "Ended",  # Final status
    9: "Suspended",
    # Add more codes as discovered
}

# Product Role Mapping - Load from JSON if available
if _LIST_VALUES_ALL and 'Clinical Trial Product Role' in _LIST_VALUES_ALL:
    _PRODUCT_ROLE_RAW = extract_code_name_mapping(_LIST_VALUES_ALL, 'Clinical Trial Product Role')
    # Convert to lowercase to match legacy format
    PRODUCT_ROLE_MAP = {code: name.lower() for code, name in _PRODUCT_ROLE_RAW.items()}
else:
    # Fallback to legacy mapping
    PRODUCT_ROLE_MAP = {
        '1': 'test',
        '2': 'comparator',
        '3': 'placebo',
        '4': 'auxiliary'
    }

# Blinding Mapping - Load from JSON if available
if _LIST_VALUES_ALL and 'Blinding Method' in _LIST_VALUES_ALL:
    _BLINDING_RAW = extract_code_name_mapping(_LIST_VALUES_ALL, 'Blinding Method')
    # Map to simplified names matching legacy format
    BLINDING_MAP = {}
    for code, name in _BLINDING_RAW.items():
        name_lower = name.lower().strip()
        if name_lower == 'single':
            BLINDING_MAP[code] = 'single-blind'
        elif name_lower == 'double':
            BLINDING_MAP[code] = 'double-blind'
        else:
            # Empty name or other - assume 'open'
            BLINDING_MAP[code] = 'open'
    # Ensure code '3' exists and is set to 'open' (it has empty name in CTIS data)
    if '3' not in BLINDING_MAP:
        BLINDING_MAP['3'] = 'open'
else:
    # Fallback to legacy mapping
    BLINDING_MAP = {
        "1": "single-blind",
        "2": "double-blind",
        "3": "open"
    }

# ===================== Status Normalization =====================

STATUS_NORMALIZATION = {
    # Authorised variants
    "authorised, recruitment pending": "Authorised, recruitment pending",
    "authorised recruitment pending": "Authorised, recruitment pending",
    "authorised pending": "Authorised, recruitment pending",
    
    # Authorised recruiting
    "authorised, recruiting": "Authorised, recruiting",
    "authorised recruiting": "Authorised, recruiting",
    
    # Ongoing recruiting
    "ongoing, recruiting": "Ongoing, recruiting",
    "ongoing recruiting": "Ongoing, recruiting",
    
    # Ongoing recruitment ended
    "ongoing, recruitment ended": "Ongoing, recruitment ended",
    "ongoing recruitment ended": "Ongoing, recruitment ended",
    
    # Temporarily halted
    "temporarily halted": "Temporarily halted",
    "temporary halt": "Temporarily halted",
    "halted": "Temporarily halted",
    
    # Suspended
    "suspended": "Suspended",
    
    # Ended
    "ended": "Ended",
    "completed": "Ended",
    "terminated": "Ended",
    
    # Not authorised
    "not authorised": "Not Authorised",
    "not authorized": "Not Authorised",
    "rejected": "Not Authorised",
    
    # Revoked
    "revoked": "Revoked",
    
    # Withdrawn
    "withdrawn": "Withdrawn",
    
    # Expired
    "expired": "Expired"
}

# ===================== Country ISO Code Mapping =====================

_COUNTRY_ISO_MAP: dict = None

def load_country_iso_mapping() -> Dict[str, Dict[str, str]]:
    """
    Load country to ISO code mapping from CTIS_List_Values_All.json
    Returns dict mapping country name to {iso2: XX, iso3: XXX}
    """
    global _COUNTRY_ISO_MAP
    
    if _COUNTRY_ISO_MAP is not None:
        return _COUNTRY_ISO_MAP
    
    _COUNTRY_ISO_MAP = {}
    
    list_values_path = Path(__file__).parent / "CTIS_List_Values_All.json"
    if not list_values_path.exists():
        print(f"WARNING: {list_values_path} not found, ISO codes will not be available")
        return _COUNTRY_ISO_MAP
    
    try:
        with open(list_values_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Find the Country table - look for key with ISO codes
        country_data = None
        
        # First try the exact key name
        if "All countries (world)" in data:
            country_data = data["All countries (world)"].get("data", [])
        
        # If not found, search for any key with 'country' and ISO codes
        if not country_data:
            for key in data:
                if 'country' in key.lower() or 'countries' in key.lower():
                    if isinstance(data[key], dict) and 'data' in data[key]:
                        test_data = data[key]['data']
                        # Check if first item has ISO code fields
                        if test_data and isinstance(test_data[0], dict):
                            if 'ISO' in str(test_data[0].keys()).upper():
                                country_data = test_data
                                break
        
        if country_data:
            for item in country_data:
                if isinstance(item, dict):
                    country_name = item.get("Country or Area Name")
                    iso2 = item.get('ISO "ALPHA-2 Code') or item.get("ISO ALPHA-2 Code")
                    iso3 = item.get("ISO ALPHA-3 Code")
                    
                    if country_name:
                        _COUNTRY_ISO_MAP[country_name] = {
                            'iso2': iso2 or '',
                            'iso3': iso3 or ''
                        }
        else:
            print("WARNING: Could not find country data with ISO codes in JSON file")
            
    except Exception as e:
        print(f"WARNING: Error loading country ISO codes: {e}")
    
    return _COUNTRY_ISO_MAP


def get_country_iso_codes(country_name: str) -> Dict[str, str]:
    """
    Get ISO codes for a country name.
    Returns dict with 'iso2' and 'iso3' keys.
    Handles common country name variations.
    """
    if not country_name:
        return {'iso2': '', 'iso3': ''}
    
    # Country name aliases - map CTIS names to JSON names
    COUNTRY_ALIASES = {
        'Czechia': 'Czech Republic',
        'Türkiye': 'Turkey',
        'Turkiye': 'Turkey',
        'Korea, Republic of': 'South Korea',
        'Republic of Korea': 'South Korea',
        'United States': 'United States of America',
        'USA': 'United States of America',
        'UK': 'United Kingdom',
        'Great Britain': 'United Kingdom',
        'Macedonia': 'North Macedonia',
        'The former Yugoslav Republic of Macedonia': 'North Macedonia',
        'Russia': 'Russian Federation',
        'Iran': 'Iran, Islamic Republic of',
        'Vietnam': 'Viet Nam',
        'Moldova': 'Moldova, Republic of',
        'Syria': 'Syrian Arab Republic',
        'Venezuela': 'Venezuela, Bolivarian Republic of',
        'Bolivia': 'Bolivia, Plurinational State of',
        'Tanzania': 'Tanzania, United Republic of',
        'Laos': "Lao People's Democratic Republic",
        'North Korea': "Korea, Democratic People's Republic of",
        'South Korea': 'Korea, Republic of',
    }
    
    # Try the original name first
    mapping = load_country_iso_mapping()
    result = mapping.get(country_name)
    
    # If not found, try alias
    if not result and country_name in COUNTRY_ALIASES:
        alias = COUNTRY_ALIASES[country_name]
        result = mapping.get(alias)
    
    # Return result or empty
    return result if result else {'iso2': '', 'iso3': ''}