"""
Approval Packages Downloader - FINAL FIXED VERSION
===================================================
FIXED: Expanded year range to 2010-2025 (was 2020-2025)

Replace your FDA/fda_syncer/downloaders/approval_packages.py with this file
"""

import os
import re
import glob
import time
import requests
from datetime import datetime
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# Import from YOUR config files
from syncher_keys import FDA_API_KEY, OUTPUT_DIR, FORCE_REDOWNLOAD, get_sync_config

from ..utils.http_client import SimpleHTTPClient
from ..utils.helpers import ensure_dir


class ApprovalPackagesDownloader:
    """Downloads complete FDA approval packages"""
    
    def __init__(self):
        self.api_key = FDA_API_KEY
        self.output_dir = OUTPUT_DIR
        self.base_url = "https://www.accessdata.fda.gov"
        self.http_client = SimpleHTTPClient(max_retries=2)
        
        # Get config from syncher_keys.py
        self.config = get_sync_config()
        
        # Document categories
        self.doc_categories = {
            'approval_letter': [r'approval.*letter', r'approv.*ltr'],
            'label': [r'label', r'package.*insert'],
            'medical_review': [r'medical.*review', r'clinical.*review'],
            'clinical_pharm_review': [r'clinical.*pharmacology'],
            'statistical_review': [r'statistical.*review'],
            'chemistry_review': [r'chemistry.*review'],
            'other_review': [r'review']
        }
    
    def download(self, therapeutic_area, drug_names):
        """Download approval packages for therapeutic area"""
        
        # Check if disabled in config
        if not self.config['integrated_reviews']['enabled']:
            print(f"\n[APPROVAL PACKAGES] DISABLED in config")
            return []
        
        print(f"\n[APPROVAL PACKAGES] Downloading for {therapeutic_area}...")
        
        # Limit to 50 drugs max for optimal performance
        max_drugs = self.config['integrated_reviews'].get('max_drugs')
        if max_drugs is None or max_drugs > 50:
            max_drugs = 50
            print(f"  🎯 Limiting to {max_drugs} drugs for optimal performance")
        
        drug_names = drug_names[:max_drugs]
        
        area_output_dir = f"{self.output_dir}/approval_packages/{therapeutic_area}"
        ensure_dir(area_output_dir)
        
        successful = []
        failed = 0
        skipped = 0
        start_time = time.time()
        
        print(f"\n  Starting downloads...")
        
        for i, drug_name in enumerate(drug_names, 1):
            # Progress every 5 drugs
            if i % 5 == 0 or i == 1 or i == len(drug_names):
                elapsed_mins = (time.time() - start_time) / 60
                rate = i / elapsed_mins if elapsed_mins > 0 else 0
                eta_mins = ((len(drug_names) - i) / rate) if rate > 0 else 0
                
                print(f"  [{i}/{len(drug_names)}] | ✓{len(successful)} ⊗{skipped} ✗{failed} | "
                      f"Elapsed: {elapsed_mins:.1f}m | ETA: {eta_mins:.1f}m")
            
            result = self._download_package(drug_name, area_output_dir)
            
            if result is None:
                failed += 1
            elif result.get('skipped'):
                skipped += 1
            else:
                successful.append(result)
                if len(successful) <= 5:
                    print(f"    ✓ Found package for {drug_name}")
            
            time.sleep(0.1)
        
        total_mins = (time.time() - start_time) / 60
        
        print(f"\n  ✅ COMPLETE!")
        print(f"     Downloaded: {len(successful)} packages")
        print(f"     Skipped (cached): {skipped}")
        print(f"     Not found: {failed}")
        print(f"     Total time: {total_mins:.1f} minutes")
        
        if len(successful) > 0:
            avg_docs = sum(r.get('downloaded', 0) for r in successful) / len(successful)
            print(f"     Avg docs/package: {avg_docs:.1f}")
        
        return successful
    
    def _download_package(self, drug_name, output_dir):
        """Download package for one drug"""
        
        try:
            # Find application number
            app_no = self._find_app_number(drug_name)
            if not app_no:
                return None
            
            # Get TOC
            toc_url, toc_content = self._get_toc(app_no)
            if not toc_content:
                return None
            
            # Extract documents
            documents = self._extract_documents(toc_content, toc_url)
            if not documents:
                return None
            
            # Create drug directory
            drug_dir = os.path.join(output_dir, drug_name.replace(' ', '_'))
            
            # Check if already downloaded
            if not FORCE_REDOWNLOAD and os.path.exists(drug_dir):
                existing = len(glob.glob(f"{drug_dir}/**/*.pdf", recursive=True))
                if existing > 0:
                    return {'drug': drug_name, 'skipped': True, 'count': existing}
            
            ensure_dir(drug_dir)
            
            # Download all documents
            downloaded = self._download_all(documents, drug_dir)
            
            # Create index
            self._create_index(drug_dir, drug_name, app_no, downloaded)
            
            return {
                'drug': drug_name,
                'app_no': app_no,
                'downloaded': len(downloaded),
                'skipped': False
            }
        except Exception as e:
            return None
    
    def _find_app_number(self, drug_name):
        """Find NDA/BLA using openFDA API"""
        
        endpoint = "https://api.fda.gov/drug/label.json"
        
        # Clean drug name
        clean = re.sub(r'\d+(\.\d+)?\s*(mg|mcg|g|ml)', '', drug_name, flags=re.IGNORECASE)
        clean = ' '.join(clean.split()).strip()
        
        try:
            search = f'openfda.brand_name:"{clean}" OR openfda.generic_name:"{clean}"'
            params = {"search": search, "limit": 1}
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            response = self.http_client.get(endpoint, params=params)
            data = response.json()
            
            if data.get('results'):
                app_numbers = data['results'][0].get('openfda', {}).get('application_number', [])
                if app_numbers:
                    app_no = app_numbers[0].replace('NDA', '').replace('BLA', '').strip()
                    # Skip ANDAs (generics) - they don't have approval packages
                    if not app_no.startswith('A'):
                        return app_no
        except:
            pass
        
        return None
    
    def _get_toc(self, app_number):
        """Get table of contents
        
        FIXED: Check 2010-2025 (was 2020-2025) because many packages are older
        """
        
        # Check last 16 years - most packages are in this range
        for year in range(2025, 2009, -1):
            urls = [
                f"{self.base_url}/drugsatfda_docs/nda/{year}/{app_number}TOC.cfm",
                f"{self.base_url}/drugsatfda_docs/nda/{year}/{app_number}Orig1s000TOC.cfm",
                f"{self.base_url}/drugsatfda_docs/bla/{year}/{app_number}Orig1s000TOC.cfm",
            ]
            
            for url in urls:
                try:
                    response = self.http_client.get(url)
                    if response.status_code == 200:
                        return url, response.content
                except:
                    # Silent failure on 404s
                    continue
        
        return None, None
    
    def _extract_documents(self, toc_content, toc_url):
        """Extract PDF links from TOC"""
        
        try:
            soup = BeautifulSoup(toc_content, 'html.parser')
            documents = []
            
            pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$', re.IGNORECASE))
            
            for link in pdf_links:
                doc_name = link.text.strip()
                doc_url = urljoin(toc_url, link['href'])
                
                documents.append({
                    'name': doc_name,
                    'url': doc_url,
                    'category': self._categorize(doc_name),
                    'filename': self._sanitize(doc_name)
                })
            
            return documents
        except:
            return []
    
    def _categorize(self, doc_name):
        """Categorize document"""
        doc_lower = doc_name.lower()
        for category, patterns in self.doc_categories.items():
            for pattern in patterns:
                if re.search(pattern, doc_lower):
                    return category
        return 'other'
    
    def _sanitize(self, filename):
        """Create safe filename"""
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = re.sub(r'\s+', '_', filename)
        if len(filename) > 200:
            filename = filename[:200]
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        return filename
    
    def _download_all(self, documents, drug_dir):
        """Download all documents"""
        
        downloaded = []
        
        for doc in documents:
            try:
                category_dir = os.path.join(drug_dir, doc['category'])
                ensure_dir(category_dir)
                
                filepath = os.path.join(category_dir, doc['filename'])
                self.http_client.download_file(doc['url'], filepath)
                
                downloaded.append({
                    'name': doc['name'],
                    'filepath': filepath,
                    'size_mb': os.path.getsize(filepath) / (1024 * 1024)
                })
                
                time.sleep(0.2)
            except:
                continue
        
        return downloaded
    
    def _create_index(self, drug_dir, drug_name, app_no, documents):
        """Create INDEX.md"""
        
        try:
            content = f"# {drug_name} - NDA/BLA {app_no}\n\n"
            content += f"Downloaded: {datetime.now().strftime('%Y-%m-%d')}\n"
            content += f"Total Documents: {len(documents)}\n\n"
            
            for doc in documents:
                content += f"- {doc['name']} ({doc['size_mb']:.2f} MB)\n"
            
            with open(os.path.join(drug_dir, 'INDEX.md'), 'w') as f:
                f.write(content)
        except:
            pass