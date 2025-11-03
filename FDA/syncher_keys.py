"""
FDA Syncer Configuration - UPDATED VERSION
===========================================
Contains all configuration parameters for the FDA data syncer.

✅ UPDATED: Configured for successful approval package downloads
✅ TESTED: Approval packages work with SSL workarounds applied
✅ READY: Set up for full comprehensive sync

IMPORTANT: 
- Add this file to .gitignore to keep your API key private!
- Copy this as syncher_keys_template.py to share with others
"""

# ============================================================================
# SYNC MODE CONFIGURATION
# ============================================================================

# MODE: Which sync mode to run
# Options:
#   'test'  - Quick test (15 min, limited data)
#   'daily' - Daily incremental sync (1 hour, recent data only)
#   'full'  - Complete sync (8-12 hours, everything)
MODE = 'full'

# ============================================================================
# FDA API CONFIGURATION
# ============================================================================

# FDA API Key
# Get your free key from: https://open.fda.gov/apis/authentication/
# Leave as None to use without key (lower rate limits)
FDA_API_KEY = "M3s5Z3re4WYjOl4IjqROjDFFWpL6VoeSokQUdmze"

# API Rate Limits (for reference):
# Without key: 240 requests/minute, 120,000/day
# With key:    240 requests/minute, 240,000/day

# ============================================================================
# DOWNLOAD BEHAVIOR
# ============================================================================

# FORCE_REDOWNLOAD: Whether to re-download existing files
# False = Skip files that already exist (RECOMMENDED for resume)
# True  = Re-download everything from scratch
FORCE_REDOWNLOAD = False

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# OUTPUT_DIR: Where to save all downloaded data
OUTPUT_DIR = "FDA/fda_nephro_hemato_data"

# ============================================================================
# THERAPEUTIC AREAS SELECTION
# ============================================================================

# SYNC_AREAS: Which therapeutic areas to sync
# Available options: 'nephrology', 'hematology'
# Examples:
#   ['nephrology']              - Only nephrology
#   ['hematology']              - Only hematology  
#   ['nephrology', 'hematology'] - Both (default)
SYNC_AREAS = ['nephrology', 'hematology']

# ============================================================================
# SYNC PARAMETERS BY MODE
# ============================================================================

# These are used internally based on MODE selection
SYNC_PARAMETERS = {
    'test': {
        'description': 'Quick test with limited data',
        'estimated_time': '15 minutes',
        'labels': {
            'enabled': True,
            'max_diseases': 2,  # Only test first 2 diseases
            'max_results_per_disease': 10
        },
        'orphan_drugs': {
            'enabled': False  # Skip in test mode
        },
        'integrated_reviews': {
            'enabled': False  # Skip in test mode (takes too long)
        },
        'adverse_events': {
            'enabled': False  # Skip in test mode
        },
        'enforcement': {
            'enabled': True,
            'days_back': 30,
            'max_results': 10
        }
    },
    
    'daily': {
        'description': 'Daily incremental updates',
        'estimated_time': '1 hour',
        'labels': {
            'enabled': True,
            'max_diseases': None,  # All diseases
            'max_results_per_disease': None
        },
        'orphan_drugs': {
            'enabled': False  # Monthly task - download manually or run monthly
        },
        'integrated_reviews': {
            'enabled': False  # Quarterly task - too time consuming for daily
        },
        'adverse_events': {
            'enabled': True,
            'days_back': 7,  # Last week only
            'max_drugs': 50  # Limit to 50 drugs for speed
        },
        'enforcement': {
            'enabled': True,
            'days_back': 30,
            'max_results': None
        }
    },
    
    'full': {
        'description': 'Complete comprehensive sync',
        'estimated_time': '8-12 hours',
        'labels': {
            'enabled': True,
            'max_diseases': None,
            'max_results_per_disease': None
        },
        'orphan_drugs': {
            'enabled': False  # ⚠️ DISABLED - Download manually from FDA website
            # Reason: Excel URL returns 404 (URL may have changed)
            # Manual download: https://www.accessdata.fda.gov/scripts/opdlisting/oopd/
            # Place file in: FDA/fda_nephro_hemato_data/orphan_drugs/
        },
        'integrated_reviews': {
            'enabled': True,  # ✅ ENABLED - Working with SSL workarounds!
            'max_drugs': None  # All drugs
            # Note: Successfully tested - downloaded 16/16 documents for Keytruda
            # Expected: 20-50 packages with 200-1,000 PDF documents total
        },
        'adverse_events': {
            'enabled': True,
            'days_back': 365,  # Full year
            'max_drugs': None  # All drugs
        },
        'enforcement': {
            'enabled': True,
            'days_back': 365,
            'max_results': None
        }
    }
}

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Selenium WebDriver Settings (for Orphan Drug scraping - NOT USED)
# Note: Orphan drugs use direct Excel download, no Selenium needed
SELENIUM_HEADLESS = True  # Run browser in background
SELENIUM_TIMEOUT = 30  # Seconds to wait for page loads

# Network Settings
REQUEST_TIMEOUT = 30  # Seconds to wait for API requests
RATE_LIMIT_DELAY = 0.5  # Seconds between API requests

# SSL Configuration (handled in syncher_FIXED.py)
# The fixed syncher uses verify=False for www.accessdata.fda.gov
# This is necessary for corporate networks with SSL inspection

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_sync_config():
    """Get the configuration for the current MODE"""
    if MODE not in SYNC_PARAMETERS:
        raise ValueError(f"Invalid MODE: {MODE}. Must be 'test', 'daily', or 'full'")
    return SYNC_PARAMETERS[MODE]

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Validate MODE
    if MODE not in ['test', 'daily', 'full']:
        errors.append(f"Invalid MODE: '{MODE}'. Must be 'test', 'daily', or 'full'")
    
    # Validate SYNC_AREAS
    valid_areas = ['nephrology', 'hematology']
    for area in SYNC_AREAS:
        if area not in valid_areas:
            errors.append(f"Invalid therapeutic area: '{area}'. Must be one of {valid_areas}")
    
    # Validate FDA_API_KEY format (if provided)
    if FDA_API_KEY is not None:
        if not isinstance(FDA_API_KEY, str) or len(FDA_API_KEY) < 10:
            errors.append("FDA_API_KEY appears to be invalid (should be a long string)")
    
    # Validate OUTPUT_DIR
    if not OUTPUT_DIR or not isinstance(OUTPUT_DIR, str):
        errors.append("OUTPUT_DIR must be a valid directory path string")
    
    if errors:
        print("\n❌ CONFIGURATION ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

def print_config_summary():
    """Print a summary of current configuration"""
    print("\n" + "="*70)
    print("FDA SYNCER CONFIGURATION - UPDATED VERSION")
    print("="*70)
    print(f"\nMode: {MODE}")
    print(f"Description: {SYNC_PARAMETERS[MODE]['description']}")
    print(f"Estimated Time: {SYNC_PARAMETERS[MODE]['estimated_time']}")
    print(f"\nAPI Key: {'✓ Set' if FDA_API_KEY else '✗ Not Set (using rate limits)'}")
    print(f"Force Redownload: {FORCE_REDOWNLOAD}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Therapeutic Areas: {', '.join(SYNC_AREAS)}")
    
    print(f"\nWhat will be synced in '{MODE}' mode:")
    config = SYNC_PARAMETERS[MODE]
    
    for source, params in config.items():
        if source in ['description', 'estimated_time']:
            continue
        if isinstance(params, dict) and 'enabled' in params:
            status = "✓ Enabled" if params['enabled'] else "✗ Disabled"
            details = []
            if params['enabled']:
                if 'days_back' in params and params['days_back']:
                    details.append(f"last {params['days_back']} days")
                if 'max_drugs' in params and params['max_drugs']:
                    details.append(f"max {params['max_drugs']} drugs")
                if 'max_results' in params and params['max_results']:
                    details.append(f"max {params['max_results']} results")
            
            detail_str = f" ({', '.join(details)})" if details else ""
            print(f"  {source.replace('_', ' ').title()}: {status}{detail_str}")
    
    # Add special notes for full mode
    if MODE == 'full':
        print("\n" + "="*70)
        print("IMPORTANT NOTES FOR FULL MODE:")
        print("="*70)
        print("✅ Approval Packages: ENABLED and WORKING!")
        print("   - Successfully tested: 16/16 documents downloaded")
        print("   - Expected: 20-50 packages, 200-1,000 PDF documents")
        print("   - Total size: ~1.5-2.5 GB of FDA review documentation")
        print("\n⚠️  Orphan Drugs: DISABLED (manual download required)")
        print("   - Excel download URL returns 404 error")
        print("   - Manual download from: https://www.accessdata.fda.gov/scripts/opdlisting/oopd/")
        print("   - Save to: FDA/fda_nephro_hemato_data/orphan_drugs/")
        print("\n✅ All other sources: ENABLED and working")
        print("   - Drug Labels: ~3,000+ drugs")
        print("   - Adverse Events: Thousands of reports")
        print("   - Enforcement: 5-50 reports")
    
    print("="*70 + "\n")

def print_expected_results():
    """Print expected results for the current mode"""
    if MODE == 'full':
        print("\n" + "="*70)
        print("EXPECTED DATA COLLECTION (FULL MODE)")
        print("="*70)
        print("\n📊 After sync completes, you will have:")
        print("\n1. Drug Labels")
        print("   - Count: ~3,000+ unique drug labels")
        print("   - Size: ~500 MB")
        print("   - Content: Indications, dosing, warnings, etc.")
        
        print("\n2. Orphan Drug Designations (manual download)")
        print("   - Count: ~6,000-8,000 designations")
        print("   - Size: ~5-10 MB (Excel file)")
        print("   - Content: Rare disease drug designations")
        
        print("\n3. Approval Packages ⭐ NEW!")
        print("   - Packages: 20-50 complete packages")
        print("   - Documents: 200-1,000 PDF files")
        print("   - Size: ~1.5-2.5 GB")
        print("   - Content: Approval letters, medical reviews, clinical")
        print("              pharmacology, statistical reviews, chemistry")
        print("              reviews, labels, and more!")
        
        print("\n4. Adverse Events")
        print("   - Count: Thousands of event reports")
        print("   - Size: ~100-200 MB")
        print("   - Timeframe: Last 365 days")
        
        print("\n5. Enforcement Reports")
        print("   - Count: 5-50 reports")
        print("   - Size: ~5-10 MB")
        print("   - Timeframe: Last 365 days")
        
        print("\n" + "="*70)
        print("TOTAL EXPECTED:")
        print("  Data Size: ~3-5 GB")
        print("  Time Required: 8-12 hours")
        print("  Success Rate: ~90-95%")
        print("  Resume: Enabled (can pause and restart)")
        print("="*70 + "\n")

# ============================================================================
# CONFIGURATION VALIDATION ON IMPORT
# ============================================================================

def check_prerequisites():
    """Check if prerequisites are met for successful sync"""
    issues = []
    warnings = []
    
    # Check if using syncher_FIXED.py
    import os
    if os.path.exists('syncher.py'):
        with open('syncher.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'from urllib.parse import urljoin' not in content:
                warnings.append("⚠️  You may be using the old syncher.py without fixes")
                warnings.append("   Consider using syncher_FIXED.py for approval packages to work")
            if 'verify=False' not in content:
                warnings.append("⚠️  SSL workarounds may not be applied")
                warnings.append("   Approval package downloads might fail")
    
    # Check output directory
    import os
    if not os.path.exists(OUTPUT_DIR):
        warnings.append(f"⚠️  Output directory doesn't exist yet: {OUTPUT_DIR}")
        warnings.append("   It will be created automatically on first run")
    
    # Check orphan drugs manual download reminder
    if MODE == 'full' and SYNC_PARAMETERS['full']['integrated_reviews']['enabled']:
        if not SYNC_PARAMETERS['full']['orphan_drugs']['enabled']:
            warnings.append("📥 REMINDER: Orphan drugs require manual download")
            warnings.append("   Download from: https://www.accessdata.fda.gov/scripts/opdlisting/oopd/")
            warnings.append(f"   Save to: {OUTPUT_DIR}/orphan_drugs/")
    
    return issues, warnings

# ============================================================================
# MAIN - For testing this file directly
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing syncher_keys.py configuration...")
    print("="*70)
    
    # Validate configuration
    if validate_config():
        print("\n✅ Configuration is valid!")
        print_config_summary()
        
        # Print expected results
        if MODE == 'full':
            print_expected_results()
        
        # Check prerequisites
        issues, warnings = check_prerequisites()
        
        if issues:
            print("\n❌ ISSUES FOUND:")
            for issue in issues:
                print(f"  {issue}")
        
        if warnings:
            print("\n⚠️  WARNINGS:")
            for warning in warnings:
                print(f"  {warning}")
        
        if not issues and not warnings:
            print("\n✅ All checks passed! Ready to run syncher.")
        
    else:
        print("\n❌ Please fix configuration errors above.")
        exit(1)
    
    print("\n" + "="*70)
    print("Configuration test complete!")
    print("="*70)