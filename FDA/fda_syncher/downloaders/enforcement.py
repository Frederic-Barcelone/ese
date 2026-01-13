"""
Enforcement Reports Downloader - VERSION v2.1
Uses syncher_keys.py and syncher_therapeutic_areas.py
FDA/fda_syncher/downloaders/enforcement.py

UPDATED: Better logging and error handling
"""

import json
import os
from datetime import datetime, timedelta

# Import from YOUR config files
from syncher_keys import FDA_API_KEY, OUTPUT_DIR, get_sync_config

from ..utils.http_client import SimpleHTTPClient
from ..utils.helpers import get_today_file, ensure_dir


class EnforcementDownloader:
    """Downloads enforcement reports from openFDA"""
    
    def __init__(self):
        self.api_key = FDA_API_KEY
        self.output_dir = OUTPUT_DIR
        self.base_url = "https://api.fda.gov"
        self.http_client = SimpleHTTPClient()
        
        # Get config from syncher_keys.py
        self.config = get_sync_config()
        
        ensure_dir(f"{self.output_dir}/enforcement")
    
    def download(self, therapeutic_area):
        """Download enforcement reports for therapeutic area"""
        
        # Check if disabled in config
        if not self.config['enforcement']['enabled']:
            print("\n[ENFORCEMENT] DISABLED in config")
            return []
        
        # Check if today's file exists
        today_file = get_today_file(f"{self.output_dir}/enforcement", therapeutic_area)
        if today_file:
            print(f"\n[ENFORCEMENT] {therapeutic_area}")
            print(f"  [OK] Using existing file: {os.path.basename(today_file)}")
            with open(today_file, 'r') as f:
                return json.load(f)
        
        print(f"\n[ENFORCEMENT] Downloading for {therapeutic_area}...")
        
        # Get config settings
        days_back = self.config['enforcement']['days_back']
        max_results = self.config['enforcement'].get('max_results')
        
        print(f"  Querying last {days_back} days...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        date_range = f"[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
        
        endpoint = f"{self.base_url}/drug/enforcement.json"
        
        params = {
            "search": f"report_date:{date_range}",
            "limit": max_results if max_results else 1000
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = self.http_client.get(endpoint, params=params)
            data = response.json()
            results = data.get('results', [])
            
            # Add metadata
            for result in results:
                result['therapeutic_area'] = therapeutic_area
                result['query_date'] = datetime.now().isoformat()
            
        except Exception as e:
            print(f"  [X] Error: {e}")
            results = []
        
        # Save results
        output_file = f"{self.output_dir}/enforcement/{therapeutic_area}_enforcement_{datetime.now().strftime('%Y%m%d')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  [OK] Downloaded {len(results)} enforcement reports")
        
        # Print some stats
        if results:
            # Count by classification
            classifications = {}
            for r in results:
                cls = r.get('classification', 'Unknown')
                classifications[cls] = classifications.get(cls, 0) + 1
            
            print("  📊 By classification:")
            for cls, count in sorted(classifications.items()):
                print(f"     - {cls}: {count}")
        
        return results
