"""
Adverse Events Downloader - OPTIMIZED VERSION v2.0
=====================================================================
FIXED: Now saves progress incrementally to prevent data loss on interruption
NEW: Circuit breaker to prevent API abuse and allow recovery

Key improvements:
1. Saves after each batch of drugs (every 10 drugs)
2. Appends to existing file instead of overwriting
3. Can resume from partial progress
4. Progress tracking file to know where we left off
5. Circuit breaker pauses on repeated failures
6. Better error classification and logging
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from syncher_keys import FDA_API_KEY, OUTPUT_DIR, get_sync_config
from ..utils.http_client import SimpleHTTPClient
from ..utils.helpers import ensure_dir


class AdverseEventsDownloader:
    """Downloads adverse events from openFDA with incremental saving and circuit breaker"""
    
    def __init__(self):
        self.api_key = FDA_API_KEY
        self.output_dir = OUTPUT_DIR
        self.base_url = "https://api.fda.gov"
        self.http_client = SimpleHTTPClient()
        self.config = get_sync_config()
        
        # Circuit breaker tracking
        self.consecutive_errors = 0
        self.total_errors = 0
        self.total_requests = 0
        
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
        
        print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        endpoint = f"{self.base_url}/drug/event.json"
        batch_size = 10  # Save every 10 drugs
        
        for i, drug_name in enumerate(drug_names, 1):
            if i % 10 == 0:
                # Show progress with error rate
                error_rate = (self.total_errors / self.total_requests * 100) if self.total_requests > 0 else 0
                print(f"    Progress: {len(processed_drugs) + i}/{len(processed_drugs) + len(drug_names)} drugs... (Error rate: {error_rate:.1f}%)")
            
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
        
        # Print final statistics
        success_rate = ((self.total_requests - self.total_errors) / self.total_requests * 100) if self.total_requests > 0 else 0
        print(f"  [OK] Downloaded {len(all_events)} adverse events")
        print(f"  Stats: {self.total_requests} requests, {self.total_errors} errors ({success_rate:.1f}% success rate)")
        
        return all_events
    
    def _check_circuit_breaker(self):
        """Check if circuit breaker should trigger"""
        if self.consecutive_errors >= 10:
            print(f"\n⚠️  CIRCUIT BREAKER TRIGGERED!")
            print(f"    Consecutive errors: {self.consecutive_errors}")
            print(f"    Pausing for 5 minutes to allow API recovery...")
            print(f"    Time: {datetime.now().strftime('%H:%M:%S')}")
            time.sleep(300)  # 5 minute pause
            self.consecutive_errors = 0
            print(f"    Resuming at {datetime.now().strftime('%H:%M:%S')}")
    
    def _download_drug_events(self, drug_name, therapeutic_area, endpoint, date_range):
        """Download events for a single drug with circuit breaker protection"""
        drug_events = []
        skip = 0
        limit = 100
        drug_error_count = 0
        
        while skip < 1000:  # Max 1000 events per drug
            params = {
                "search": f'patient.drug.medicinalproduct:"{drug_name}" AND receivedate:{date_range}',
                "limit": limit,
                "skip": skip
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            try:
                self.total_requests += 1
                response = self.http_client.get(endpoint, params=params)
                data = response.json()
                
                # Check for API errors
                if 'error' in data:
                    error_msg = data['error'].get('message', 'Unknown error')
                    error_code = data['error'].get('code', 'UNKNOWN')
                    
                    # Count as error
                    self.total_errors += 1
                    self.consecutive_errors += 1
                    drug_error_count += 1
                    
                    # Log error but only print for first occurrence
                    if skip == 0:
                        print(f"      {drug_name}: FDA API error [{error_code}]: {error_msg[:50]}")
                    
                    # Check circuit breaker
                    self._check_circuit_breaker()
                    break
                
                # Check for results
                if not data.get('results'):
                    # No more results (reached end) - this is success, not error
                    self.consecutive_errors = 0
                    break
                
                # Success - reset consecutive error counter
                self.consecutive_errors = 0
                
                # Add metadata to each event
                for event in data['results']:
                    event['query_drug'] = drug_name
                    event['therapeutic_area'] = therapeutic_area
                
                drug_events.extend(data['results'])
                
                # Show progress for drugs with many events
                if len(drug_events) >= 600 and len(drug_events) % 100 == 0:
                    print(f"      {drug_name}: {len(drug_events)} events so far...")
                
                skip += limit
                
            except Exception as e:
                # Count as error
                self.total_errors += 1
                self.consecutive_errors += 1
                drug_error_count += 1
                
                # Log error but continue with other drugs
                if skip == 0:
                    error_str = str(e)
                    # Truncate long error messages
                    if len(error_str) > 100:
                        error_str = error_str[:97] + "..."
                    print(f"      {drug_name}: Request error: {error_str}")
                
                # Check circuit breaker
                self._check_circuit_breaker()
                break
        
        # Log success for drugs with events (and no errors)
        if drug_events and drug_error_count == 0:
            # Only log if we got a reasonable number of events
            if len(drug_events) < 600:
                print(f"      ✓ {drug_name}: {len(drug_events)} events")
        
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