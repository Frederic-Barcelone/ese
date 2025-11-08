"""
Enhanced Diagnostic - Shows API Responses
==========================================
This will show us exactly what the FDA API is returning
"""

import sys
sys.path.insert(0, 'FDA')

import requests
from syncher_keys import FDA_API_KEY

print("="*70)
print("ENHANCED FDA API DIAGNOSTIC")
print("="*70)

# Test drugs
test_drugs = ['Keytruda', 'Opdivo', 'Dexamethasone', 'pembrolizumab', 'nivolumab']

endpoint = "https://api.fda.gov/drug/label.json"

for drug in test_drugs:
    print(f"\n{'='*70}")
    print(f"Testing: {drug}")
    print(f"{'='*70}")
    
    # Test 1: Brand name search
    print("\n[TEST 1] Searching by brand name...")
    search = f'openfda.brand_name:"{drug}"'
    params = {"search": search, "limit": 1}
    if FDA_API_KEY:
        params["api_key"] = FDA_API_KEY
    
    try:
        response = requests.get(endpoint, params=params, timeout=30, verify=False)
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                result = data['results'][0]
                openfda = result.get('openfda', {})
                
                brand_names = openfda.get('brand_name', [])
                generic_names = openfda.get('generic_name', [])
                app_numbers = openfda.get('application_number', [])
                
                print(f"  ✓ Found result!")
                print(f"    Brand names: {brand_names[:3]}")
                print(f"    Generic names: {generic_names[:3]}")
                print(f"    Application numbers: {app_numbers}")
                
                if app_numbers:
                    app_no = app_numbers[0].replace('NDA', '').replace('BLA', '').strip()
                    print(f"    → Cleaned app number: {app_no}")
                    
                    # Test if we can find the TOC
                    print(f"\n[TEST 2] Looking for TOC for {app_no}...")
                    base_url = "https://www.accessdata.fda.gov"
                    
                    found_toc = False
                    for year in range(2025, 2019, -1):
                        test_urls = [
                            f"{base_url}/drugsatfda_docs/nda/{year}/{app_no}TOC.cfm",
                            f"{base_url}/drugsatfda_docs/nda/{year}/{app_no}Orig1s000TOC.cfm",
                            f"{base_url}/drugsatfda_docs/bla/{year}/{app_no}Orig1s000TOC.cfm",
                        ]
                        
                        for url in test_urls:
                            try:
                                toc_response = requests.get(url, timeout=10, verify=False)
                                if toc_response.status_code == 200:
                                    print(f"    ✓ Found TOC: {url}")
                                    print(f"      Content length: {len(toc_response.content)} bytes")
                                    found_toc = True
                                    break
                            except:
                                pass
                        
                        if found_toc:
                            break
                    
                    if not found_toc:
                        print(f"    ✗ No TOC found for years 2020-2025")
                else:
                    print(f"    ✗ No application numbers in result")
            else:
                print(f"  ✗ No results found")
        else:
            print(f"  ✗ Error: {response.status_code}")
            print(f"    {response.text[:200]}")
    
    except Exception as e:
        print(f"  ✗ Exception: {e}")
    
    # Test 2: Generic name search (if different)
    if drug.lower() not in ['pembrolizumab', 'nivolumab']:
        print(f"\n[TEST 3] Searching by generic name...")
        generic_map = {
            'Keytruda': 'pembrolizumab',
            'Opdivo': 'nivolumab',
            'Dexamethasone': 'dexamethasone'
        }
        
        if drug in generic_map:
            generic = generic_map[drug]
            search = f'openfda.generic_name:"{generic}"'
            params = {"search": search, "limit": 1}
            if FDA_API_KEY:
                params["api_key"] = FDA_API_KEY
            
            try:
                response = requests.get(endpoint, params=params, timeout=30, verify=False)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('results'):
                        app_numbers = data['results'][0].get('openfda', {}).get('application_number', [])
                        print(f"  ✓ Found via generic name: {app_numbers}")
                    else:
                        print(f"  ✗ No results via generic name")
            except Exception as e:
                print(f"  ✗ Exception: {e}")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
print("\nSUMMARY:")
print("- If all drugs show 'Exception', you have a network/SSL issue")
print("- If drugs found but no TOC, the packages don't exist or are too old")
print("- If drugs not found, the search query needs adjustment")