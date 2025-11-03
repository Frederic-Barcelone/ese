import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

base_url = "https://www.accessdata.fda.gov"

# Test newer drugs that SHOULD have packages
test_drugs = {
    "Keytruda": "BLA125514",  # Approved 2014
    "Jardiance": "NDA204629",  # Approved 2014
    "Farxiga": "NDA202293",   # Approved 2014
}

for drug_name, app_num_full in test_drugs.items():
    # Extract number and type
    if app_num_full.startswith('BLA'):
        app_type = 'bla'
        app_no = app_num_full.replace('BLA', '')
    else:
        app_type = 'nda'
        app_no = app_num_full.replace('NDA', '')
    
    print(f"\n{'='*60}")
    print(f"Testing: {drug_name} ({app_num_full} → {app_type}/{app_no})")
    print(f"{'='*60}")
    
    found = False
    for year in range(2025, 2010, -1):
        toc_url = f"{base_url}/drugsatfda_docs/{app_type}/{year}/{app_no}Orig1s000TOC.cfm"
        
        try:
            response = requests.get(toc_url, timeout=10, verify=False)
            if response.status_code == 200 and len(response.content) > 1000:
                print(f"✓ FOUND in {year}!")
                print(f"  URL: {toc_url}")
                
                # Count PDFs
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                pdfs = soup.find_all('a', href=lambda x: x and x.endswith('.pdf'))
                print(f"  PDFs: {len(pdfs)} documents")
                
                found = True
                break
        except:
            pass
    
    if not found:
        print(f"✗ NOT FOUND (2010-2025)")