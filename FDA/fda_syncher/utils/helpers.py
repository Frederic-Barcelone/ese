"""
Helper functions - UPDATED VERSION v2.1
FDA/fda_syncher/utils/helpers.py

NEW: Added filter_pharmaceutical_drugs() to remove cosmetics/OTC products
"""

import os
import re
import glob
from datetime import datetime

# Import from YOUR config file
from syncher_keys import FORCE_REDOWNLOAD


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


def filter_pharmaceutical_drugs(drug_names):
    """
    Filter out non-pharmaceutical products (cosmetics, OTC, etc.)
    
    This prevents wasting API calls on products like:
    - Sunscreens and SPF products
    - Moisturizers and lotions
    - Cosmetic creams
    - Shampoos and soaps
    - Beauty products
    
    Returns:
        List of likely pharmaceutical drug names
    """
    if not drug_names:
        return []
    
    # Patterns that indicate NON-pharmaceutical products
    skip_patterns = [
        # Sunscreens and SPF
        r'SPF\s*\d+',
        r'sunscreen',
        r'sun\s*block',
        r'broad\s*spectrum',
        
        # Cosmetics and beauty
        r'moisturizer',
        r'lotion\s*$',
        r'body\s*lotion',
        r'face\s*cream',
        r'eye\s*cream',
        r'night\s*cream',
        r'day\s*cream',
        r'anti[- ]?aging',
        r'wrinkle',
        r'foundation',
        r'concealer',
        r'mascara',
        r'lipstick',
        r'lip\s*gloss',
        r'blush',
        r'bronzer',
        r'primer',
        r'tinted',
        r'glow',
        r'radiance',
        r'luminous',
        r'matte\s+foundation',
        
        # Skin care (non-Rx)
        r'cleanser',
        r'face\s*wash',
        r'body\s*wash',
        r'exfoliat',
        r'scrub',
        r'peel',
        r'mask\s*$',
        r'face\s*mask',
        r'serum\s*$',
        r'beauty\s*serum',
        r'toner',
        r'essence',
        
        # Hair care
        r'shampoo',
        r'conditioner',
        r'hair\s*(spray|gel|mousse|oil)',
        
        # Body care
        r'soap',
        r'hand\s*(cream|lotion|wash|sanitizer)',
        r'body\s*butter',
        r'massage\s*oil',
        r'bath\s*(oil|salt|bomb)',
        
        # Deodorants
        r'deodorant',
        r'antiperspirant',
        
        # Oral care (non-Rx)
        r'toothpaste',
        r'mouthwash',
        r'tooth\s*whitening',
        
        # Other non-pharma indicators
        r'cosmetic',
        r'beauty',
        r'spa\s',
        r'salon',
        r'fragrance',
        r'perfume',
        r'cologne',
        r'nail\s*polish',
        r'cuticle',
        
        # Specific problematic patterns from the log
        r'kojic\s*acid.*lotion',
        r'breast\s*enhance',
        r'scar\s*sheet',
        r'bee\s*venom',
        r'tag\s*recede',
        r'skin\s*care.*moisturizer',
        r'hydrating.*cream.*spf',
        
        # Long ingredient lists (usually cosmetics)
        r'HELIANTHUS.*SEED.*OIL.*TOCOPHEROL',
        r'OLEA.*EUROPAEA.*FRUIT.*OIL.*BUTTER',
        r'SESAMUM.*INDICUM.*EXTRACT',
        r'BUTYROSPERMUM.*PARKII',
        r'ARGANIA.*SPINOSA',
        r'CAMELLIA.*JAPONICA',
        
        # OTC products that aren't useful for rare disease research
        r'isopropyl\s*alcohol\s*sanitizer',
        r'hand\s*sanitizer',
        r'antibacterial\s*soap',
        r'benzethonium\s*chloride\s*liquid',
    ]
    
    # Compile all patterns into one regex (case insensitive)
    skip_regex = re.compile('|'.join(skip_patterns), re.IGNORECASE)
    
    # Patterns that indicate LIKELY pharmaceutical products (whitelist)
    pharma_indicators = [
        r'tablet',
        r'capsule',
        r'injection',
        r'solution\s+for',
        r'suspension',
        r'mg(/|\s|$)',  # Dosage indicator
        r'mcg',
        r'\d+\s*mg',
        r'oral',
        r'intravenous',
        r'subcutaneous',
        r'intramuscular',
        r'ophthalmic',
        r'otic',
        r'nasal\s*spray',
        r'inhaler',
        r'nebulizer',
        r'suppository',
        r'patch',
        r'extended[- ]release',
        r'delayed[- ]release',
        r'immediate[- ]release',
    ]
    
    pharma_regex = re.compile('|'.join(pharma_indicators), re.IGNORECASE)
    
    filtered = []
    skipped_count = 0
    
    for drug_name in drug_names:
        if not drug_name or not isinstance(drug_name, str):
            continue
        
        # Skip if matches non-pharma patterns
        if skip_regex.search(drug_name):
            skipped_count += 1
            continue
        
        # Skip very long names (usually ingredient lists for cosmetics)
        if len(drug_name) > 150:
            # Unless it has pharma indicators
            if not pharma_regex.search(drug_name):
                skipped_count += 1
                continue
        
        # Skip names that are mostly uppercase with lots of commas (ingredient lists)
        if drug_name.isupper() and drug_name.count(',') > 3:
            if not pharma_regex.search(drug_name):
                skipped_count += 1
                continue
        
        filtered.append(drug_name)
    
    if skipped_count > 0:
        print(f"    🧹 Filtered out {skipped_count} non-pharmaceutical products")
    
    return filtered


def extract_drug_names_from_labels(labels):
    """
    Extract unique drug names from label results
    
    UPDATED: Now filters out non-pharmaceutical products
    """
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
    
    # Convert to list and filter
    drug_list = list(drug_names)
    filtered_list = filter_pharmaceutical_drugs(drug_list)
    
    return filtered_list


def extract_drug_names_from_labels_unfiltered(labels):
    """
    Extract unique drug names from label results WITHOUT filtering
    Use this if you need all names regardless of product type
    """
    drug_names = set()
    
    for label in labels:
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
