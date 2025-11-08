"""
Adverse Events Downloader - IMPROVED VERSION with Incremental Saving
=====================================================================
FIXED: Now saves progress incrementally to prevent data loss on interruption

Key improvements:
1. Saves after each batch of drugs (every 10 drugs)
2. Appends to existing file instead of overwriting
3. Can resume from partial progress
4. Progress tracking file to know where we left off
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from syncher_keys import FDA_API_KEY, OUTPUT_DIR, get_sync_config
from ..utils.http_client import SimpleHTTPClient
from ..utils.helpers import ensure_dir


class AdverseEventsDownloader:
    """Downloads adverse events from openFDA with incremental saving"""
    
    def __init__(self):
        self.api_key = FDA_API_KEY
        self.output_dir = OUTPUT_DIR
        self.base_url = "https://api.fda.gov"
        self.http_client = SimpleHTTPClient()
        self.config = get_sync_config()
        
        ensure_dir(f"{self.output_dir}/adverse_events")
        ensure_dir(f"{self.output_dir}/adverse_events/.progress")  # Progress tracking
    
    def download(self, therapeutic_area, drug_names):
        """Download adverse events for therapeutic area with incremental saves"""
        
        if not self.config['adverse_events']['enabled']:
            print(f"\n[ADVERSE EVENTS] DISABLED in config")
            return []
        
        # Check configuration
        days_back = self.config['adverse_events']['days_back']
        max_drugs = self.config['adverse_events'].get('max_drugs')
        
        if max_drugs:
            drug_names = drug_names[:max_drugs]
            print(f"  Limiting to {max_drugs} drugs (config setting)")
        
        # Setup file paths
        today = datetime.now().strftime('%Y%m%d')
        output_file = f"{self.output_dir}/adverse_events/{therapeutic_area}_adverse_events_{today}.json"
        progress_file = f"{self.output_dir}/adverse_events/.progress/{therapeutic_area}_{today}.txt"
        
        # Check if we have a completed file for today
        if os.path.exists(output_file) and not os.path.exists(progress_file):
            print(f"\n[ADVERSE EVENTS] {therapeutic_area}")
            print(f"  [OK] Using existing completed file: {os.path.basename(output_file)}")
            with open(output_file, 'r') as f:
                return json.load(f)
        
        print(f"\n[ADVERSE EVENTS] Downloading for {therapeutic_area}...")
        print(f"  Querying {len(drug_names)} drugs (last {days_back} days)...")
        
        # Load existing progress
        processed_drugs, all_events = self._load_progress(output_file, progress_file)
        
        if processed_drugs:
            print(f"  [RESUME] Continuing from drug #{len(processed_drugs)} ({len(all_events)} events so far)")
            # Filter out already processed drugs
            drug_names = [d for d in drug_names if d not in processed_drugs]
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        date_range = f"[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
        
        endpoint = f"{self.base_url}/drug/event.json"
        batch_size = 10  # Save every 10 drugs
        
        for i, drug_name in enumerate(drug_names, 1):
            if i % 10 == 0:
                print(f"    Progress: {len(processed_drugs) + i}/{len(processed_drugs) + len(drug_names)} drugs...")
            
            # Download events for this drug
            drug_events = self._download_drug_events(
                drug_name, 
                therapeutic_area,
                endpoint, 
                date_range
            )
            
            if drug_events:
                all_events.extend(drug_events)
            
            processed_drugs.add(drug_name)
            
            # Save incrementally every batch_size drugs
            if i % batch_size == 0 or i == len(drug_names):
                self._save_progress(
                    output_file, 
                    progress_file, 
                    processed_drugs, 
                    all_events
                )
                print(f"    [SAVED] Progress checkpoint: {len(all_events)} events")
        
        # Final save and cleanup
        self._finalize(output_file, progress_file, all_events)
        
        print(f"  [OK] Downloaded {len(all_events)} adverse events")
        return all_events
    
    def _download_drug_events(self, drug_name, therapeutic_area, endpoint, date_range):
        """Download events for a single drug"""
        drug_events = []
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
                response = self.http_client.get(endpoint, params=params)
                data = response.json()
                
                if not data.get('results'):
                    break
                
                for event in data['results']:
                    event['query_drug'] = drug_name
                    event['therapeutic_area'] = therapeutic_area
                
                drug_events.extend(data['results'])
                skip += limit
                
            except Exception as e:
                # Log error but continue with other drugs
                print(f"      Error for {drug_name}: {str(e)[:50]}")
                break
        
        return drug_events
    
    def _load_progress(self, output_file, progress_file):
        """Load existing progress if available"""
        processed_drugs = set()
        all_events = []
        
        # Load existing data file
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    all_events = json.load(f)
            except:
                all_events = []
        
        # Load progress tracking file
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    processed_drugs = set(line.strip() for line in f if line.strip())
            except:
                processed_drugs = set()
        
        return processed_drugs, all_events
    
    def _save_progress(self, output_file, progress_file, processed_drugs, all_events):
        """Save current progress"""
        # Save data
        with open(output_file, 'w') as f:
            json.dump(all_events, f, indent=2)
        
        # Save progress tracking
        with open(progress_file, 'w') as f:
            for drug in sorted(processed_drugs):
                f.write(f"{drug}\n")
    
    def _finalize(self, output_file, progress_file, all_events):
        """Finalize download and remove progress file"""
        # Final save
        with open(output_file, 'w') as f:
            json.dump(all_events, f, indent=2)
        
        # Remove progress file to indicate completion
        if os.path.exists(progress_file):
            os.remove(progress_file)