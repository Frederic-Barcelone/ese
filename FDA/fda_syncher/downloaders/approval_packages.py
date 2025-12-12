"""
Approval Packages Downloader - OPTIMIZED VERSION v2.1
======================================================
FIXED: Expanded year range to 2010-2025 (was 2020-2025)
NEW: Timeout protection to prevent stalls
NEW: Smarter TOC searching with early exit
NEW: Better progress tracking

Key improvements:
1. Timeout protection in _get_toc() - max 30 seconds per drug
2. Prioritizes recent years first (more likely to have packages)
3. Skips ANDAs (generics don't have approval packages)
4. Better error handling and logging
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
from ..utils.helpers import ensure_dir, filter_pharmaceutical_drugs


class ApprovalPackagesDownloader:
    """Downloads complete FDA approval packages with timeout protection"""
    
    def __init__(self):
        self.api_key = FDA_API_KEY
        self.output_dir = OUTPUT_DIR
        self.base_url = "https://www.accessdata.fda.gov"
        self.http_client = SimpleHTTPClient(max_retries=2)
        
        # Get config from syncher_keys.py
        self.config = get_sync_config()
        
        # Timeout settings
        self.max_toc_search_time = 20  # Max seconds to search for TOC per drug
        self.max_download_time = 120   # Max seconds to download all docs per drug
        
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
        
        # Stats tracking
        self.stats = {
            'found': 0,
            'not_found': 0,
            'skipped_cached': 0,
            'skipped_anda': 0,
            'timeout': 0,
            'error': 0
        }
    
    def download(self, therapeutic_area, drug_names):
        """Download approval packages for therapeutic area"""
        
        # Check if disabled in config
        if not self.config['integrated_reviews']['enabled']:
            print(f"\n[APPROVAL PACKAGES] DISABLED in config")
            return []
        
        print(f"\n[APPROVAL PACKAGES] Downloading for {therapeutic_area}...")
        
        # Filter drug names first
        original_count = len(drug_names)
        drug_names = filter_pharmaceutical_drugs(drug_names)
        print(f"  📋 Drug list: {original_count} → {len(drug_names)} (after filtering)")
        
        # Limit to max drugs for optimal performance
        max_drugs = self.config['integrated_reviews'].get('max_drugs')
        if max_drugs is None or max_drugs > 200:
            max_drugs = 200
            print(f"  🎯 Limiting to {max_drugs} drugs for optimal performance")
        
        drug_names = drug_names[:max_drugs]
        
        area_output_dir = f"{self.output_dir}/approval_packages/{therapeutic_area}"
        ensure_dir(area_output_dir)
        
        successful = []
        start_time = time.time()
        
        print(f"\n  Starting downloads...")
        
        for i, drug_name in enumerate(drug_names, 1):
            # Progress every 5 drugs
            if i % 5 == 0 or i == 1 or i == len(drug_names):
                elapsed_mins = (time.time() - start_time) / 60
                rate = i / elapsed_mins if elapsed_mins > 0 else 0
                eta_mins = ((len(drug_names) - i) / rate) if rate > 0 else 0
                
                print(f"  [{i}/{len(drug_names)}] | "
                      f"✓{self.stats['found']} "
                      f"⊗{self.stats['skipped_cached']} "
                      f"✗{self.stats['not_found']} "
                      f"⏱{self.stats['timeout']} | "
                      f"Elapsed: {elapsed_mins:.1f}m | ETA: {eta_mins:.1f}m")
            
            result = self._download_package(drug_name, area_output_dir)
            
            if result is None:
                pass  # Already tracked in stats
            elif result.get('skipped'):
                pass  # Already tracked
            else:
                successful.append(result)
                if len(successful) <= 10:
                    print(f"    ✓ Found package for {drug_name} ({result.get('downloaded', 0)} docs)")
            
            time.sleep(0.1)
        
        total_mins = (time.time() - start_time) / 60
        
        print(f"\n  ✅ COMPLETE!")
        print(f"     Found & Downloaded: {self.stats['found']} packages")
        print(f"     Skipped (cached): {self.stats['skipped_cached']}")
        print(f"     Skipped (ANDA/generic): {self.stats['skipped_anda']}")
        print(f"     Not found: {self.stats['not_found']}")
        print(f"     Timeouts: {self.stats['timeout']}")
        print(f"     Errors: {self.stats['error']}")
        print(f"     Total time: {total_mins:.1f} minutes")
        
        if len(successful) > 0:
            avg_docs = sum(r.get('downloaded', 0) for r in successful) / len(successful)
            print(f"     Avg docs/package: {avg_docs:.1f}")
        
        return successful
    
    def _download_package(self, drug_name, output_dir):
        """Download package for one drug with timeout protection"""
        
        drug_start_time = time.time()
        
        try:
            # Find application number
            app_no = self._find_app_number(drug_name)
            if not app_no:
                self.stats['not_found'] += 1
                return None
            
            # Check if ANDA (generic) - skip these
            if app_no.startswith('A') or 'ANDA' in app_no.upper():
                self.stats['skipped_anda'] += 1
                return None
            
            # Get TOC with timeout
            toc_url, toc_content = self._get_toc(app_no, drug_start_time)
            if not toc_content:
                self.stats['not_found'] += 1
                return None
            
            # Check total timeout
            if time.time() - drug_start_time > self.max_download_time:
                self.stats['timeout'] += 1
                return None
            
            # Extract documents
            documents = self._extract_documents(toc_content, toc_url)
            if not documents:
                self.stats['not_found'] += 1
                return None
            
            # Create drug directory
            drug_dir = os.path.join(output_dir, drug_name.replace(' ', '_').replace('/', '_'))
            
            # Check if already downloaded
            if not FORCE_REDOWNLOAD and os.path.exists(drug_dir):
                existing = len(glob.glob(f"{drug_dir}/**/*.pdf", recursive=True))
                if existing > 0:
                    self.stats['skipped_cached'] += 1
                    return {'drug': drug_name, 'skipped': True, 'count': existing}
            
            ensure_dir(drug_dir)
            
            # Download all documents with timeout check
            downloaded = self._download_all(documents, drug_dir, drug_start_time)
            
            # Create index
            self._create_index(drug_dir, drug_name, app_no, downloaded)
            
            self.stats['found'] += 1
            
            return {
                'drug': drug_name,
                'app_no': app_no,
                'downloaded': len(downloaded),
                'skipped': False
            }
        except Exception as e:
            self.stats['error'] += 1
            return None
    
    def _find_app_number(self, drug_name):
        """Find NDA/BLA using openFDA API"""
        
        endpoint = "https://api.fda.gov/drug/label.json"
        
        # Clean drug name - remove dosage info
        clean = re.sub(r'\d+(\.\d+)?\s*(mg|mcg|g|ml|%)', '', drug_name, flags=re.IGNORECASE)
        clean = ' '.join(clean.split()).strip()
        
        if not clean:
            return None
        
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
                    app_no = app_numbers[0]
                    
                    # Skip ANDAs (generics) - they don't have approval packages
                    if app_no.startswith('ANDA'):
                        return None
                    
                    # Clean up the application number
                    app_no = app_no.replace('NDA', '').replace('BLA', '').strip()
                    return app_no
        except:
            pass
        
        return None
    
    def _get_toc(self, app_number, start_time):
        """
        Get table of contents with timeout protection
        
        FIXED: 
        - Check 2010-2025 (16 years of data)
        - Timeout after max_toc_search_time seconds
        - Try most likely URL patterns first
        """
        
        # Check timeout
        if time.time() - start_time > self.max_toc_search_time:
            return None, None
        
        # Try most recent years first (more likely to succeed)
        current_year = datetime.now().year
        
        for year in range(current_year, 2009, -1):
            # Check timeout between years
            if time.time() - start_time > self.max_toc_search_time:
                return None, None
            
            # URL patterns in order of likelihood
            urls = [
                f"{self.base_url}/drugsatfda_docs/nda/{year}/{app_number}Orig1s000TOC.cfm",
                f"{self.base_url}/drugsatfda_docs/nda/{year}/{app_number}TOC.cfm",
                f"{self.base_url}/drugsatfda_docs/bla/{year}/{app_number}Orig1s000TOC.cfm",
            ]
            
            for url in urls:
                # Check timeout between URLs
                if time.time() - start_time > self.max_toc_search_time:
                    return None, None
                
                try:
                    response = self.http_client.get(url)
                    if response.status_code == 200:
                        # Verify it's actually a TOC page (has PDF links)
                        if b'.pdf' in response.content.lower():
                            return url, response.content
                except:
                    # Silent failure on 404s - expected
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
    
    def _download_all(self, documents, drug_dir, start_time):
        """Download all documents with timeout protection"""
        
        downloaded = []
        
        for doc in documents:
            # Check timeout
            if time.time() - start_time > self.max_download_time:
                print(f"      ⏱️ Timeout - downloaded {len(downloaded)}/{len(documents)} docs")
                break
            
            try:
                category_dir = os.path.join(drug_dir, doc['category'])
                ensure_dir(category_dir)
                
                filepath = os.path.join(category_dir, doc['filename'])
                
                # Skip if already exists
                if os.path.exists(filepath) and not FORCE_REDOWNLOAD:
                    downloaded.append({
                        'name': doc['name'],
                        'filepath': filepath,
                        'size_mb': os.path.getsize(filepath) / (1024 * 1024)
                    })
                    continue
                
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
            
            # Group by category
            by_category = {}
            for doc in documents:
                cat = doc.get('filepath', '').split('/')[-2] if '/' in doc.get('filepath', '') else 'other'
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(doc)
            
            for category, docs in by_category.items():
                content += f"\n## {category.replace('_', ' ').title()}\n"
                for doc in docs:
                    content += f"- {doc['name']} ({doc['size_mb']:.2f} MB)\n"
            
            with open(os.path.join(drug_dir, 'INDEX.md'), 'w') as f:
                f.write(content)
        except:
            pass
