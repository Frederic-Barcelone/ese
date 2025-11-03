#!/usr/bin/env python3
"""
FDA Comprehensive Data Sync for Nephrology & Hematology
========================================================
With SMART RESUME - skips already downloaded files!

Configuration is in: syncher_keys.py
Therapeutic areas in: syncher_therapeutic_areas.py

Just run: python syncher.py

FIXED VERSION - Includes approval package URL fix and SSL workarounds
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import PyPDF2
import re
from pathlib import Path
import traceback
import glob
import urllib3
from urllib.parse import urljoin  # ADDED: For proper URL construction

# Suppress SSL warnings for corporate networks
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # ADDED

# ============================================================================
# IMPORT CONFIGURATION
# ============================================================================

# Import therapeutic areas (disease lists)
from syncher_therapeutic_areas import THERAPEUTIC_AREAS, get_all_therapeutic_areas

# Import configuration (API key, mode, settings)
from syncher_keys import (
    MODE, 
    FDA_API_KEY, 
    FORCE_REDOWNLOAD, 
    OUTPUT_DIR, 
    SYNC_AREAS,
    SYNC_PARAMETERS,
    REQUEST_TIMEOUT,
    RATE_LIMIT_DELAY,
    validate_config,
    print_config_summary,
    get_sync_config
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_existing_file(filepath, max_age_days=None):
    """Check if file exists and is recent enough"""
    if not os.path.exists(filepath):
        return False
    
    if FORCE_REDOWNLOAD:
        return False
    
    if max_age_days:
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath))
        if file_age.days > max_age_days:
            return False
    
    return True

def get_today_file(directory, pattern):
    """Get today's file if it exists"""
    today = datetime.now().strftime('%Y%m%d')
    matching_files = glob.glob(f"{directory}/*{today}*.json")
    if matching_files and not FORCE_REDOWNLOAD:
        return matching_files[0]
    return None

# ============================================================================
# COMPLETE APPROVAL PACKAGE SYNCER
# ============================================================================

class CompleteApprovalPackageSyncer:
    """Downloads ALL documents from FDA approval packages"""
    
    def __init__(self):
        self.base_url = "https://www.accessdata.fda.gov"
        self.search_url = f"{self.base_url}/scripts/cder/daf/index.cfm"
        
        # Document types to categorize
        self.doc_categories = {
            'approval_letter': [
                r'approval.*letter',
                r'approv.*ltr',
                r'^approval$'
            ],
            'label': [
                r'label',
                r'package.*insert',
                r'labeling',
                r'prescribing.*information'
            ],
            'integrated_review': [
                r'multi.*discipline.*review',
                r'integrated.*review',
                r'cross.*discipline.*team.*leader',
                r'summary.*review',
                r'combined.*review'
            ],
            'medical_review': [
                r'medical.*review',
                r'clinical.*review(?!.*memo)',
                r'^clinical$'
            ],
            'clinical_pharm_review': [
                r'clinical.*pharmacology.*review',
                r'clin.*pharm.*review',
                r'pharmacology.*review'
            ],
            'statistical_review': [
                r'statistical.*review',
                r'biometrics.*review',
                r'stats.*review'
            ],
            'pharmacology_review': [
                r'^pharmacology.*review',
                r'nonclinical.*review',
                r'pharm.*tox.*review'
            ],
            'chemistry_review': [
                r'chemistry.*review',
                r'CMC.*review',
                r'chemistry.*manufacturing'
            ],
            'microbiology_review': [
                r'microbiology.*review'
            ],
            'other_review': [
                r'review',
                r'assessment'
            ]
        }
    
    def find_application_number(self, drug_name):
        """Find NDA/BLA number using openFDA API - WORKING VERSION!"""
        
        # Use openFDA API instead of web scraping (more reliable)
        endpoint = "https://api.fda.gov/drug/label.json"
        
        # List of search attempts (from most specific to most general)
        search_attempts = []
        
        # Attempt 1: Original name
        search_attempts.append(drug_name)
        
        # Attempt 2: Remove dosage (50 mg, 2.5 mg, etc.)
        clean1 = re.sub(r'\d+(\.\d+)?\s*(mg|mcg|g|ml|%)', '', drug_name, flags=re.IGNORECASE)
        clean1 = ' '.join(clean1.split()).strip()
        if clean1 != drug_name and len(clean1) > 2:
            search_attempts.append(clean1)
        
        # Attempt 3: Remove salt forms
        clean2 = re.sub(r'\b(besylate|hydrochloride|potassium|sodium|maleate|tartrate|succinate|mesylate|cilexetil|medoxomil|carbonate)\b', 
                    '', clean1, flags=re.IGNORECASE)
        clean2 = ' '.join(clean2.split()).strip()
        if clean2 != clean1 and len(clean2) > 2:
            search_attempts.append(clean2)
        
        # Attempt 4: Take first drug if combination (X and Y -> X)
        if ' and ' in drug_name.lower():
            first_drug = drug_name.split(' and ')[0]
            first_drug = re.sub(r'\d+(\.\d+)?\s*(mg|mcg|g|ml|%)', '', first_drug, flags=re.IGNORECASE)
            first_drug = re.sub(r'\b(besylate|hydrochloride|potassium|sodium|maleate|tartrate|succinate|mesylate)\b', 
                            '', first_drug, flags=re.IGNORECASE)
            first_drug = ' '.join(first_drug.split()).strip()
            if len(first_drug) > 2:
                search_attempts.append(first_drug)
        
        # Try each search attempt with openFDA API
        for search_term in search_attempts:
            if not search_term or len(search_term) < 3:
                continue
            
            try:
                # Build search query for openFDA
                search_query = f'openfda.brand_name:"{search_term}" OR openfda.generic_name:"{search_term}"'
                
                params = {
                    "search": search_query,
                    "limit": 1
                }
                
                # Import API key from config
                try:
                    from syncher_keys import FDA_API_KEY
                    if FDA_API_KEY:
                        params["api_key"] = FDA_API_KEY
                except:
                    pass
                
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data.get('results'):
                    result = data['results'][0]
                    openfda = result.get('openfda', {})
                    
                    # Get application numbers
                    app_numbers = openfda.get('application_number', [])
                    if app_numbers:
                        # Clean the app number (remove NDA/BLA/ANDA prefix)
                        app_no = app_numbers[0].replace('NDA', '').replace('BLA', '').replace('ANDA', '').strip()
                        # Skip ANDA numbers (generics) - they start with 'A'
                        if not app_no.startswith('A'):
                            return app_no
                
            except:
                continue
        
        return None
    
    def get_approval_package_toc(self, app_number):
        """Get table of contents for approval package"""
        # Extended to 2025 to catch recent approvals
        for year in range(2025, 2010, -1):
            # Try NDA format with Orig1s000
            toc_url = f"{self.base_url}/drugsatfda_docs/nda/{year}/{app_number}Orig1s000TOC.cfm"
            try:
                # FIXED: Added verify=False for SSL workaround
                response = requests.get(toc_url, timeout=REQUEST_TIMEOUT, verify=False)
                if response.status_code == 200:
                    return toc_url, response.content, 'nda'
            except:
                pass
            
            # Try NDA format without Orig1s000
            toc_url = f"{self.base_url}/drugsatfda_docs/nda/{year}/{app_number}TOC.cfm"
            try:
                # FIXED: Added verify=False for SSL workaround
                response = requests.get(toc_url, timeout=REQUEST_TIMEOUT, verify=False)
                if response.status_code == 200:
                    return toc_url, response.content, 'nda'
            except:
                pass
            
            # Try BLA format
            toc_url = f"{self.base_url}/drugsatfda_docs/bla/{year}/{app_number}Orig1s000TOC.cfm"
            try:
                # FIXED: Added verify=False for SSL workaround
                response = requests.get(toc_url, timeout=REQUEST_TIMEOUT, verify=False)
                if response.status_code == 200:
                    return toc_url, response.content, 'bla'
            except:
                pass
        
        return None, None, None
    
    def categorize_document(self, doc_name):
        """Categorize document by name"""
        doc_name_lower = doc_name.lower()
        
        for category, patterns in self.doc_categories.items():
            for pattern in patterns:
                if re.search(pattern, doc_name_lower, re.IGNORECASE):
                    return category
        
        return 'other'
    
    def extract_all_documents(self, toc_content, toc_url):  # FIXED: Added toc_url parameter
        """Extract ALL PDF links from approval package"""
        soup = BeautifulSoup(toc_content, 'html.parser')
        
        documents = []
        pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$', re.IGNORECASE))
        
        for link in pdf_links:
            doc_name = link.text.strip()
            
            # FIXED: Use urljoin for proper URL construction
            doc_url = urljoin(toc_url, link['href'])
            
            category = self.categorize_document(doc_name)
            
            documents.append({
                'name': doc_name,
                'url': doc_url,
                'category': category,
                'filename': self.sanitize_filename(doc_name)
            })
        
        return documents
    
    def sanitize_filename(self, filename):
        """Create safe filename from document name"""
        # Remove special characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace spaces and multiple underscores
        filename = re.sub(r'\s+', '_', filename)
        filename = re.sub(r'_+', '_', filename)
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        # Ensure .pdf extension
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        return filename
    
    def download_complete_approval_package(self, drug_name, output_dir="./approval_packages"):
        """Download ALL documents from approval package"""
        
        # DON'T create directory yet - wait until we know we have data!
        
        # Find application number FIRST
        app_no = self.find_application_number(drug_name)
        if not app_no:
            return None  # Exit early - no directory created
        
        print(f"    Found: NDA/BLA {app_no}")
        
        # Get approval package TOC
        toc_url, toc_content, app_type = self.get_approval_package_toc(app_no)
        if not toc_content:
            print(f"    ✗ Could not find approval package")
            return None  # Exit early - no directory created
        
        print(f"    Found approval package")
        
        # FIXED: Pass toc_url to extract_all_documents
        documents = self.extract_all_documents(toc_content, toc_url)
        
        if not documents:
            print(f"    ✗ No documents found in approval package")
            return None  # Exit early - no directory created
        
        # NOW create the directory (we know we have data)
        drug_dir = os.path.join(output_dir, drug_name.replace(' ', '_'))
        os.makedirs(drug_dir, exist_ok=True)
        
        # Check if already downloaded
        existing_files = glob.glob(f"{drug_dir}/**/*.pdf", recursive=True)
        if existing_files and not FORCE_REDOWNLOAD:
            print(f"    ✓ Already downloaded: {len(existing_files)} documents")
            return {
                'drug': drug_name,
                'directory': drug_dir,
                'files': existing_files,
                'skipped': True,
                'count': len(existing_files)
            }
        
        print(f"    Found {len(documents)} documents")
        
        # Categorize documents
        by_category = {}
        for doc in documents:
            category = doc['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(doc)
        
        # Print summary
        print(f"    Document breakdown:")
        for category, docs in sorted(by_category.items()):
            print(f"      - {category.replace('_', ' ').title()}: {len(docs)}")
        
        # Download all documents
        print(f"    Downloading documents...")
        downloaded = []
        
        for i, doc in enumerate(documents, 1):
            try:
                # Create category subdirectory
                category_dir = os.path.join(drug_dir, doc['category'])
                os.makedirs(category_dir, exist_ok=True)
                
                # Download file
                if i % 5 == 0 or i == len(documents):
                    print(f"      Progress: {i}/{len(documents)}")
                
                # FIXED: Added verify=False for SSL workaround
                response = requests.get(doc['url'], timeout=60, verify=False)
                response.raise_for_status()
                
                # Save file
                filepath = os.path.join(category_dir, doc['filename'])
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                downloaded.append({
                    'name': doc['name'],
                    'category': doc['category'],
                    'filepath': filepath,
                    'size_mb': len(response.content) / (1024 * 1024)
                })
                
                time.sleep(1)  # Be nice to FDA servers
                
            except Exception as e:
                print(f"      ✗ Error downloading: {doc['name'][:40]}...")
                continue
        
        print(f"    ✓ Downloaded {len(downloaded)}/{len(documents)} documents")
        
        # Create index file
        self.create_package_index(drug_dir, drug_name, app_no, downloaded)
        
        return {
            'drug': drug_name,
            'app_no': app_no,
            'app_type': app_type,
            'directory': drug_dir,
            'toc_url': toc_url,
            'total_documents': len(documents),
            'downloaded': len(downloaded),
            'files': downloaded,
            'skipped': False
        }
    
    def create_package_index(self, drug_dir, drug_name, app_no, documents):
        """Create an index/manifest of all downloaded documents"""
        
        index_content = f"""# FDA Approval Package Index
## {drug_name} (NDA/BLA {app_no})

Downloaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Documents: {len(documents)}

## Documents by Category

"""
        
        # Group by category
        by_category = {}
        for doc in documents:
            category = doc['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(doc)
        
        # Write each category
        for category in sorted(by_category.keys()):
            index_content += f"\n### {category.replace('_', ' ').title()}\n\n"
            
            for doc in by_category[category]:
                rel_path = os.path.relpath(doc['filepath'], drug_dir)
                size_mb = doc['size_mb']
                index_content += f"- **{doc['name']}**\n"
                index_content += f"  - File: `{rel_path}`\n"
                index_content += f"  - Size: {size_mb:.2f} MB\n\n"
        
        # Save index
        index_file = os.path.join(drug_dir, 'INDEX.md')
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(index_content)

# ============================================================================
# MAIN SYNCER CLASS
# ============================================================================

class ComprehensiveFDASync:
    def __init__(self, api_key=None, output_dir="./fda_data"):
        self.api_key = api_key
        self.output_dir = output_dir
        self.base_url = "https://api.fda.gov"
        
        # Create output directories
        Path(f"{output_dir}/labels").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/orphan_drugs").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/approval_packages").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/adverse_events").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/enforcement").mkdir(parents=True, exist_ok=True)
        
    def build_search_query(self, keywords: List[str], field: str = "indications_and_usage") -> str:
        """Build comprehensive OR search query"""
        terms = [f'{field}:"{keyword}"' for keyword in keywords]
        return ' OR '.join(terms)
    
    # ========== SOURCE 1: openFDA Drug Labels ==========
    def sync_drug_labels_comprehensive(self, therapeutic_area='nephrology'):
        """Sync ALL drug labels for therapeutic area"""
        
        # Check if today's file exists
        today_file = get_today_file(f"{self.output_dir}/labels", therapeutic_area)
        if today_file:
            print(f"\n[1/5] Drug Labels for {therapeutic_area}...")
            print(f"  ✓ Using existing file from today: {os.path.basename(today_file)}")
            with open(today_file, 'r') as f:
                return json.load(f)
        
        diseases = THERAPEUTIC_AREAS[therapeutic_area]['rare_diseases']
        drug_classes = THERAPEUTIC_AREAS[therapeutic_area]['drug_classes']
        all_keywords = diseases + drug_classes
        
        print(f"\n[1/5] Syncing Drug Labels for {therapeutic_area}...")
        print(f"  Searching {len(all_keywords)} disease/class terms...")
        
        endpoint = f"{self.base_url}/drug/label.json"
        all_results = []
        seen_set_ids = set()
        
        search_fields = [
            'indications_and_usage',
            'description',
            'openfda.pharm_class_epc'
        ]
        
        for field in search_fields:
            search_query = self.build_search_query(all_keywords, field)
            
            skip = 0
            limit = 100
            
            while skip < 3000:  # Limit per field
                params = {
                    "search": search_query,
                    "limit": limit,
                    "skip": skip
                }
                
                if self.api_key:
                    params["api_key"] = self.api_key
                
                try:
                    response = requests.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
                    response.raise_for_status()
                    data = response.json()
                    
                    if not data.get('results'):
                        break
                    
                    for result in data['results']:
                        set_id = result.get('set_id')
                        if set_id and set_id not in seen_set_ids:
                            seen_set_ids.add(set_id)
                            all_results.append(result)
                    
                    if skip + limit >= data['meta']['results']['total']:
                        break
                    
                    skip += limit
                    time.sleep(RATE_LIMIT_DELAY)
                    
                except Exception as e:
                    print(f"    Error at skip {skip}: {e}")
                    break
        
        output_file = f"{self.output_dir}/labels/{therapeutic_area}_labels_{datetime.now().strftime('%Y%m%d')}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"  ✓ Synced {len(all_results)} unique drug labels")
        return all_results
    
    # ========== SOURCE 2: Orphan Drug Database (EXCEL DOWNLOAD - NO SELENIUM) ==========
    def sync_orphan_drugs_excel(self, therapeutic_area='nephrology'):
        """Download complete orphan drug Excel file - with SSL workaround"""
        
        # Check if today's file exists
        today = datetime.now().strftime('%Y%m%d')
        existing_files = glob.glob(f"{self.output_dir}/orphan_drugs/{therapeutic_area}_orphan_drugs_{today}.csv")
        if existing_files and not FORCE_REDOWNLOAD:
            print(f"\n[2/5] Orphan Drugs for {therapeutic_area}...")
            print(f"  ✓ Using existing file from today: {os.path.basename(existing_files[0])}")
            return pd.read_csv(existing_files[0])
        
        print(f"\n[2/5] Syncing Orphan Drug Database for {therapeutic_area}...")
        print(f"  Downloading complete FDA orphan drug database (Excel)...")
        
        # FDA's official Excel export URL
        excel_url = "https://www.accessdata.fda.gov/scripts/opdlisting/oopd/OOPD_Export.xlsx"
        
        try:
            # Download Excel file
            print(f"  Downloading from: {excel_url}")
            
            try:
                response = requests.get(excel_url, timeout=120, verify=True)
                response.raise_for_status()
            except requests.exceptions.SSLError:
                print(f"  ⚠️  SSL verification failed, trying without verification...")
                # Fallback: disable SSL verification (works on corporate networks)
                response = requests.get(excel_url, timeout=120, verify=False)
                response.raise_for_status()
            
            # Save temporary Excel file
            temp_file = f'{self.output_dir}/orphan_drugs/temp_orphan_drugs_{today}.xlsx'
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            file_size_mb = len(response.content) / (1024 * 1024)
            print(f"  ✓ Downloaded Excel file ({file_size_mb:.2f} MB)")
            
            # Read Excel into pandas
            print(f"  Reading Excel file...")
            df = pd.read_excel(temp_file)
            
            print(f"  ✓ Loaded {len(df)} total orphan drug designations")
            
            # Get diseases for this therapeutic area
            diseases = THERAPEUTIC_AREAS[therapeutic_area]['rare_diseases']
            
            # Build search pattern (case-insensitive)
            disease_pattern = '|'.join([re.escape(d) for d in diseases])
            
            # Find which columns exist
            possible_columns = [
                'Generic Name', 'Trade Name', 'Designation', 
                'Orphan Designation', 'Indication', 'CF Designation Date'
            ]
            existing_columns = [col for col in possible_columns if col in df.columns]
            
            if not existing_columns:
                print(f"  ⚠️  Warning: Could not find expected columns")
                print(f"  Available columns: {list(df.columns)}")
                existing_columns = list(df.columns)
            
            print(f"  Searching in {len(existing_columns)} columns...")
            
            # Create mask - search across all relevant columns
            mask = pd.Series([False] * len(df))
            for col in existing_columns:
                try:
                    mask |= df[col].astype(str).str.contains(disease_pattern, case=False, na=False, regex=True)
                except:
                    pass
            
            df_filtered = df[mask].copy()
            
            print(f"  ✓ Filtered to {len(df_filtered)} relevant orphan drugs for {therapeutic_area}")
            
            # Add metadata
            df_filtered['therapeutic_area'] = therapeutic_area
            df_filtered['sync_date'] = datetime.now().isoformat()
            df_filtered['search_diseases'] = ', '.join(diseases[:5]) + f' (+ {len(diseases)-5} more)' if len(diseases) > 5 else ', '.join(diseases)
            
            # Save filtered results
            output_file = f"{self.output_dir}/orphan_drugs/{therapeutic_area}_orphan_drugs_{today}.csv"
            df_filtered.to_csv(output_file, index=False)
            
            print(f"  ✓ Saved to: {output_file}")
            
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass
            
            # Show sample
            if len(df_filtered) > 0:
                print(f"\n  Sample results:")
                for i, row in df_filtered.head(3).iterrows():
                    generic_name = row.get('Generic Name', 'N/A')
                    trade_name = row.get('Trade Name', 'N/A')
                    designation = row.get('Designation', row.get('Orphan Designation', 'N/A'))
                    if isinstance(designation, str) and len(designation) > 60:
                        designation = designation[:60] + "..."
                    print(f"    - {generic_name} ({trade_name})")
            
            return df_filtered
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print(f"  ⚠️  Continuing without orphan drug data...")
            return pd.DataFrame()
    
    # ========== HELPER: Filter Drugs Likely to Have Approval Packages ==========
    def filter_drugs_with_packages(self, drug_names):
        """Filter to drugs likely to have approval packages"""
        
        # Patterns that indicate old generics unlikely to have packages
        skip_patterns = [
            r'^[A-Z][a-z]+$',  # Single word generic names (Tacrolimus, Nadolol)
            r'\b(hydrochloride|sodium|phosphate|sulfate|acetate|citrate)\b',  # Salt forms
        ]
        
        # Very common generic drugs unlikely to have recent approval packages
        common_generics = [
            'hydrochlorothiazide', 'spironolactone', 'furosemide', 'tacrolimus',
            'amlodipine', 'lisinopril', 'metoprolol', 'nadolol', 'calcipotriene',
            'prednisone', 'dexamethasone', 'methylprednisolone', 'hydrocortisone',
            'warfarin', 'heparin', 'aspirin', 'ibuprofen', 'acetaminophen',
            'amoxicillin', 'azithromycin', 'ciprofloxacin', 'doxycycline'
        ]
        
        filtered = []
        skipped = []
        
        for drug_name in drug_names:
            should_skip = False
            
            # Check skip patterns
            for pattern in skip_patterns:
                if re.search(pattern, drug_name, re.IGNORECASE):
                    should_skip = True
                    break
            
            # Check common generics
            if drug_name.lower() in common_generics:
                should_skip = True
            
            # Skip if all lowercase (generic indicator)
            if drug_name.islower():
                should_skip = True
            
            if should_skip:
                skipped.append(drug_name)
            else:
                filtered.append(drug_name)
        
        return filtered, skipped
    
    # ========== SOURCE 3: Complete Approval Packages ==========
    def sync_complete_approval_packages(self, therapeutic_area='nephrology', max_drugs=None):
        """Download COMPLETE approval packages (all documents)"""
        
        # Get list of drugs from labels
        labels_file = f"{self.output_dir}/labels/{therapeutic_area}_labels_{datetime.now().strftime('%Y%m%d')}.json"
        
        if not os.path.exists(labels_file):
            print("  ⚠ No labels file found. Run labels sync first!")
            return []
        
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        
        # Extract unique drug names
        drug_names = set()
        for label in labels:
            brand_names = label.get('openfda', {}).get('brand_name', [])
            drug_names.update(brand_names)
        
        drug_names = list(drug_names)
        
        # FILTER OUT GENERIC/OLD DRUGS
        print(f"\n[3/5] Syncing COMPLETE Approval Packages for {therapeutic_area}...")
        print(f"  Found {len(drug_names)} total drugs in labels...")
        print(f"  Filtering to drugs likely to have approval packages...")
        
        filtered_drugs, skipped_drugs = self.filter_drugs_with_packages(drug_names)
        
        print(f"  ✓ {len(filtered_drugs)} drugs to process")
        print(f"  ✓ {len(skipped_drugs)} generic/old drugs skipped")
        
        if len(skipped_drugs) <= 20:
            print(f"  Skipped: {', '.join(skipped_drugs[:20])}")
        
        drug_names = filtered_drugs
        
        if max_drugs:
            drug_names = drug_names[:max_drugs]
        
        print(f"  This will download ALL review documents (letters, reviews, labels, etc.)")
        
        # Check existing packages
        existing_packages = []
        for drug_name in drug_names:
            drug_dir = f"{self.output_dir}/approval_packages/{therapeutic_area}/{drug_name.replace(' ', '_')}"
            if os.path.exists(drug_dir):
                pdf_count = len(glob.glob(f"{drug_dir}/**/*.pdf", recursive=True))
                if pdf_count > 0:
                    existing_packages.append((drug_name, pdf_count))
        
        if existing_packages:
            print(f"  Already have {len(existing_packages)} complete packages:")
            for drug, count in existing_packages[:5]:
                print(f"    - {drug}: {count} documents")
            if len(existing_packages) > 5:
                print(f"    ... and {len(existing_packages) - 5} more")
        
        syncer = CompleteApprovalPackageSyncer()
        successful_downloads = []
        skipped_count = 0
        failed_count = 0
        
        for i, drug_name in enumerate(drug_names, 1):
            print(f"\n  [{i}/{len(drug_names)}] Processing: {drug_name}")
            
            try:
                package_info = syncer.download_complete_approval_package(
                    drug_name, 
                    output_dir=f"{self.output_dir}/approval_packages/{therapeutic_area}"
                )
                
                if package_info:
                    if package_info.get('skipped'):
                        skipped_count += 1
                    else:
                        successful_downloads.append(package_info)
                        print(f"    ✓ Package complete: {package_info['downloaded']} documents")
                else:
                    failed_count += 1
                    # Only print warning for first 10 failures
                    if failed_count <= 10:
                        print(f"    ⚠ No approval package found")
                
                time.sleep(2)
                
            except Exception as e:
                failed_count += 1
                if failed_count <= 10:
                    print(f"    ✗ Error: {e}")
                continue
        
        # Summary
        total_docs = sum(p['downloaded'] for p in successful_downloads)
        
        print(f"\n  ========================================")
        print(f"  APPROVAL PACKAGES SUMMARY")
        print(f"  ========================================")
        print(f"  ✓ Successfully downloaded: {len(successful_downloads)} packages")
        print(f"  ✓ Total documents: {total_docs}")
        print(f"  ✓ Skipped (already have): {skipped_count} packages")
        print(f"  ⚠ Not found: {failed_count} packages")
        
        return successful_downloads
    
    # ========== SOURCE 4: Adverse Events ==========
    def sync_adverse_events_comprehensive(self, therapeutic_area='nephrology', days_back=365, max_drugs=None):
        """Sync adverse events"""
        
        # Check if recent file exists (within 1 day)
        today = datetime.now().strftime('%Y%m%d')
        existing_files = glob.glob(f"{self.output_dir}/adverse_events/{therapeutic_area}_adverse_events_*.json")
        
        for file in existing_files:
            file_date = re.search(r'(\d{8})', os.path.basename(file))
            if file_date and file_date.group(1) == today and not FORCE_REDOWNLOAD:
                print(f"\n[4/5] Adverse Events for {therapeutic_area}...")
                print(f"  ✓ Using existing file from today: {os.path.basename(file)}")
                with open(file, 'r') as f:
                    return json.load(f)
        
        # Get drug names from labels
        labels_file = f"{self.output_dir}/labels/{therapeutic_area}_labels_{datetime.now().strftime('%Y%m%d')}.json"
        
        if not os.path.exists(labels_file):
            print("  ⚠ No labels file found. Run labels sync first!")
            return []
        
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        
        # Extract drug names
        drug_names = set()
        for label in labels:
            brand_names = label.get('openfda', {}).get('brand_name', [])
            generic_names = label.get('openfda', {}).get('generic_name', [])
            drug_names.update(brand_names + generic_names)
        
        drug_names = list(drug_names)
        
        if max_drugs:
            drug_names = drug_names[:max_drugs]
        
        print(f"\n[4/5] Syncing Adverse Events for {therapeutic_area}...")
        print(f"  Querying {len(drug_names)} drugs (last {days_back} days)...")
        
        endpoint = f"{self.base_url}/drug/event.json"
        
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        date_range = f"[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
        
        all_events = []
        
        for i, drug_name in enumerate(drug_names, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(drug_names)} drugs...")
            
            skip = 0
            limit = 100
            
            while skip < 1000:  # Max 1000 events per drug
                params = {
                    "search": f'patient.drug.medicinalproduct:"{drug_name}" AND receivedate:{date_range}',
                    "limit": limit,
                    "skip": skip
                }
                
                if self.api_key:
                    params["api_key"] = self.api_key
                
                try:
                    response = requests.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
                    response.raise_for_status()
                    data = response.json()
                    
                    if not data.get('results'):
                        break
                    
                    for event in data['results']:
                        event['query_drug'] = drug_name
                        event['therapeutic_area'] = therapeutic_area
                    
                    all_events.extend(data['results'])
                    
                    skip += limit
                    time.sleep(RATE_LIMIT_DELAY)
                    
                except Exception as e:
                    break
        
        # Save
        output_file = f"{self.output_dir}/adverse_events/{therapeutic_area}_adverse_events_{datetime.now().strftime('%Y%m%d')}.json"
        with open(output_file, 'w') as f:
            json.dump(all_events, f, indent=2)
        
        print(f"  ✓ Synced {len(all_events)} adverse events")
        return all_events
    
    # ========== SOURCE 5: Enforcement Reports ==========
    def sync_enforcement_reports(self, therapeutic_area='nephrology', days_back=365):
        """Sync enforcement reports"""
        
        # Check if today's file exists
        today = datetime.now().strftime('%Y%m%d')
        existing_files = glob.glob(f"{self.output_dir}/enforcement/{therapeutic_area}_enforcement_*.json")
        
        for file in existing_files:
            file_date = re.search(r'(\d{8})', os.path.basename(file))
            if file_date and file_date.group(1) == today and not FORCE_REDOWNLOAD:
                print(f"\n[5/5] Enforcement Reports for {therapeutic_area}...")
                print(f"  ✓ Using existing file from today: {os.path.basename(file)}")
                with open(file, 'r') as f:
                    return json.load(f)
        
        print(f"\n[5/5] Syncing Enforcement Reports for {therapeutic_area}...")
        
        endpoint = f"{self.base_url}/drug/enforcement.json"
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        date_range = f"[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
        
        params = {
            "search": f"report_date:{date_range}",
            "limit": 1000
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = requests.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
            data = response.json()
            results = data.get('results', [])
        except Exception as e:
            print(f"  Error: {e}")
            results = []
        
        # Save
        output_file = f"{self.output_dir}/enforcement/{therapeutic_area}_enforcement_{datetime.now().strftime('%Y%m%d')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  ✓ Synced {len(results)} enforcement reports")
        return results
    
    # ========== MASTER SYNC FUNCTIONS FOR EACH MODE ==========
    def sync_full(self, therapeutic_areas):
        """FULL MODE: Complete comprehensive sync"""
        print(f"\n{'='*70}")
        print(f"MODE: FULL SYNC")
        print(f"This will sync ALL FDA data sources (may take 8-12 hours)")
        print(f"Resume enabled: Skipping existing files")
        print(f"{'='*70}\n")
        
        results = {}
        
        for area in therapeutic_areas:
            print(f"\n{'='*70}")
            print(f"THERAPEUTIC AREA: {area.upper()}")
            print(f"{'='*70}")
            
            results[area] = {}
            
            # 1. Drug Labels
            results[area]['labels'] = self.sync_drug_labels_comprehensive(area)
            
            # 2. Orphan Drugs (Excel download - NO SELENIUM!)
            results[area]['orphan_drugs'] = self.sync_orphan_drugs_excel(area)
            
            # 3. COMPLETE Approval Packages (ALL DOCUMENTS!)
            results[area]['approval_packages'] = self.sync_complete_approval_packages(area, max_drugs=None)
            
            # 4. Adverse Events (all drugs, 1 year)
            results[area]['adverse_events'] = self.sync_adverse_events_comprehensive(area, days_back=365, max_drugs=None)
            
            # 5. Enforcement Reports
            results[area]['enforcement'] = self.sync_enforcement_reports(area, days_back=365)
        
        self.generate_summary_report(results)
        return results
    
    def sync_daily(self, therapeutic_areas):
        """DAILY MODE: Quick incremental updates"""
        print(f"\n{'='*70}")
        print(f"MODE: DAILY SYNC")
        print(f"This will do quick incremental updates (~1 hour)")
        print(f"Resume enabled: Skipping existing files")
        print(f"{'='*70}\n")
        
        results = {}
        
        for area in therapeutic_areas:
            print(f"\n{'='*70}")
            print(f"THERAPEUTIC AREA: {area.upper()}")
            print(f"{'='*70}")
            
            results[area] = {}
            
            # 1. Drug Labels (full sync - relatively quick)
            results[area]['labels'] = self.sync_drug_labels_comprehensive(area)
            
            # 2. Skip Orphan Drugs (monthly task)
            results[area]['orphan_drugs'] = "Skipped (monthly task)"
            print(f"\n[2/5] Skipping Orphan Drugs (monthly task)")
            
            # 3. Skip Approval Packages (quarterly task)
            results[area]['approval_packages'] = "Skipped (quarterly task)"
            print(f"\n[3/5] Skipping Approval Packages (quarterly task)")
            
            # 4. Adverse Events (last 7 days only, max 50 drugs)
            results[area]['adverse_events'] = self.sync_adverse_events_comprehensive(area, days_back=7, max_drugs=50)
            
            # 5. Enforcement Reports (last 30 days)
            results[area]['enforcement'] = self.sync_enforcement_reports(area, days_back=30)
        
        self.generate_summary_report(results)
        return results
    
    def sync_test(self, therapeutic_areas):
        """TEST MODE: Quick test with limited data"""
        print(f"\n{'='*70}")
        print(f"MODE: TEST SYNC")
        print(f"This will sync a small sample for testing (~15 minutes)")
        print(f"Resume enabled: Skipping existing files")
        print(f"{'='*70}\n")
        
        results = {}
        
        # Only test first therapeutic area
        area = therapeutic_areas[0]
        
        print(f"\n{'='*70}")
        print(f"THERAPEUTIC AREA: {area.upper()} (test only)")
        print(f"{'='*70}")
        
        results[area] = {}
        
        # 1. Drug Labels (limited search)
        print(f"\n[1/5] Testing Drug Labels...")
        # Test with just 2 diseases
        test_diseases = THERAPEUTIC_AREAS[area]['rare_diseases'][:2]
        print(f"  Testing with: {test_diseases}")
        
        endpoint = f"{self.base_url}/drug/label.json"
        search_query = ' OR '.join([f'indications_and_usage:"{d}"' for d in test_diseases])
        
        params = {"search": search_query, "limit": 10}
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = requests.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
            data = response.json()
            labels = data.get('results', [])
            print(f"  ✓ Found {len(labels)} labels")
            results[area]['labels'] = labels
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[area]['labels'] = []
        
        # 2. Test Orphan Drugs (Excel download)
        print(f"\n[2/5] Testing Orphan Drugs (Excel download)...")
        try:
            orphan_df = self.sync_orphan_drugs_excel(area)
            results[area]['orphan_drugs'] = f"Found {len(orphan_df)} orphan drugs"
            print(f"  ✓ Orphan drugs: {len(orphan_df)}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[area]['orphan_drugs'] = "Failed"
        
        # 3. Skip Approval Packages in test mode
        results[area]['approval_packages'] = "Test mode - skipped"
        print(f"\n[3/5] Skipping Approval Packages (test mode)")
        
        # 4. Skip Adverse Events in test mode
        results[area]['adverse_events'] = "Skipped (test mode)"
        print(f"\n[4/5] Skipping Adverse Events (test mode)")
        
        # 5. Test Enforcement Reports (last 30 days, limit 10)
        print(f"\n[5/5] Testing Enforcement Reports...")
        endpoint = f"{self.base_url}/drug/enforcement.json"
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        date_range = f"[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
        
        params = {"search": f"report_date:{date_range}", "limit": 10}
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = requests.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
            data = response.json()
            enforcement = data.get('results', [])
            print(f"  ✓ Found {len(enforcement)} enforcement reports")
            results[area]['enforcement'] = enforcement
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[area]['enforcement'] = []
        
        print(f"\n{'='*70}")
        print(f"TEST COMPLETE")
        print(f"✓ Labels: {len(results[area]['labels'])}")
        print(f"✓ Orphan Drugs: {results[area]['orphan_drugs']}")
        print(f"✓ Enforcement: {len(results[area]['enforcement'])}")
        print(f"{'='*70}\n")
        
        return results
    
    def generate_summary_report(self, results):
        """Generate markdown summary of sync"""
        
        report = f"""# FDA Data Sync Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Mode: {MODE}
Force Redownload: {FORCE_REDOWNLOAD}

"""
        
        for area, data in results.items():
            report += f"\n## {area.upper()}\n\n"
            
            if isinstance(data.get('labels'), list):
                report += f"- **Drug Labels**: {len(data['labels'])} drugs\n"
            else:
                report += f"- **Drug Labels**: {data.get('labels', 'N/A')}\n"
            
            if isinstance(data.get('orphan_drugs'), pd.DataFrame):
                report += f"- **Orphan Drugs**: {len(data['orphan_drugs'])} designations\n"
            else:
                report += f"- **Orphan Drugs**: {data.get('orphan_drugs', 'N/A')}\n"
            
            if isinstance(data.get('approval_packages'), list):
                report += f"- **Approval Packages**: {len(data['approval_packages'])} downloaded\n"
                total_docs = sum(p.get('downloaded', 0) for p in data['approval_packages'])
                report += f"  - Total Documents: {total_docs}\n"
            else:
                report += f"- **Approval Packages**: {data.get('approval_packages', 'N/A')}\n"
            
            if isinstance(data.get('adverse_events'), list):
                report += f"- **Adverse Events**: {len(data['adverse_events'])} events\n"
            else:
                report += f"- **Adverse Events**: {data.get('adverse_events', 'N/A')}\n"
            
            if isinstance(data.get('enforcement'), list):
                report += f"- **Enforcement Reports**: {len(data['enforcement'])} reports\n"
            else:
                report += f"- **Enforcement Reports**: {data.get('enforcement', 'N/A')}\n"
        
        # Save report
        report_file = f"{self.output_dir}/sync_summary_{MODE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n{report}")
        print(f"Summary saved to: {report_file}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Validate configuration first
    print(f"\n{'='*70}")
    print(f"FDA COMPREHENSIVE DATA SYNC - FIXED VERSION")
    print(f"Nephrology & Hematology")
    print(f"{'='*70}")
    
    if not validate_config():
        print("\n❌ Configuration errors detected. Please check syncher_keys.py")
        return
    
    # Print configuration summary
    print_config_summary()
    
    # Confirm if running full mode
    if MODE == 'full':
        print("\n⚠️  WARNING: FULL mode will take 8-12 hours and download ~7-11 GB")
        try:
            response = input("Continue? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Sync cancelled.")
                return
        except:
            print("Running in non-interactive mode. Proceeding with full sync...")
    
    # Initialize syncer
    syncer = ComprehensiveFDASync(
        api_key=FDA_API_KEY,
        output_dir=OUTPUT_DIR
    )
    
    start_time = datetime.now()
    
    try:
        if MODE == 'full':
            results = syncer.sync_full(SYNC_AREAS)
        elif MODE == 'daily':
            results = syncer.sync_daily(SYNC_AREAS)
        elif MODE == 'test':
            results = syncer.sync_test(SYNC_AREAS)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'='*70}")
        print(f"SYNC COMPLETE!")
        print(f"Duration: {duration}")
        print(f"Data saved to: {OUTPUT_DIR}")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print(f"\n\nSync interrupted by user. Partial data saved to: {OUTPUT_DIR}")
    except Exception as e:
        print(f"\n\nERROR during sync:")
        print(f"{e}")
        traceback.print_exc()

# Run the script
if __name__ == "__main__":
    main()