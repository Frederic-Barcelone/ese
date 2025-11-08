"""
Helper functions
FDA/fda_syncher/utils/helpers.py
"""

import os
import glob
from datetime import datetime

# Import from YOUR config file
from syncher_keys import FORCE_REDOWNLOAD
from syncher_therapeutic_areas import normalize_term


def check_existing_file(filepath):
    """Check if file exists and should be used"""
    if FORCE_REDOWNLOAD:
        return False
    return os.path.exists(filepath)


def get_today_file(directory, pattern):
    """Get today's file if it exists"""
    today = datetime.now().strftime('%Y%m%d')
    matching_files = glob.glob(f"{directory}/*{pattern}*{today}*.json")
    if matching_files and not FORCE_REDOWNLOAD:
        return matching_files[0]
    return None


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)


def extract_drug_names_from_labels(labels):
    """Extract unique drug names from label results"""
    drug_names = set()
    for label in labels:
        # Skip None or non-dict labels
        if not isinstance(label, dict):
            continue
        
        openfda = label.get('openfda', {})
        if not isinstance(openfda, dict):
            continue
        
        brand_names = openfda.get('brand_name', [])
        generic_names = openfda.get('generic_name', [])
        
        drug_names.update(brand_names)
        drug_names.update(generic_names)
    return list(drug_names)