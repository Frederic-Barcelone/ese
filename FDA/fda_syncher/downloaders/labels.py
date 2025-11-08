"""
Drug Labels Downloader - FIXED VERSION with Keyword Batching
==============================================================
FIXED: Batches keywords to avoid "414 URI Too Long" error
FIXED: Now saves progress in batches to prevent data loss on interruption

Key improvements:
1. Batches keywords into groups of 15 to avoid URL length limits
2. Saves after every 100 results
3. Can resume from partial progress
4. Deduplication preserved across resume
5. Tracks completed keyword batches
"""

import json
import os
from datetime import datetime
from pathlib import Path

from syncher_keys import FDA_API_KEY, OUTPUT_DIR, get_sync_config
from syncher_therapeutic_areas import THERAPEUTIC_AREAS, get_expanded_keywords
from ..utils.http_client import SimpleHTTPClient
from ..utils.helpers import ensure_dir


class LabelsDownloader:
    """Downloads drug labels from openFDA with incremental saving and keyword batching"""
    
    def __init__(self):
        self.api_key = FDA_API_KEY
        self.output_dir = OUTPUT_DIR
        self.base_url = "https://api.fda.gov"
        self.http_client = SimpleHTTPClient()
        self.config = get_sync_config()
        
        # Keyword batch size to avoid URI too long errors
        # 15 keywords = ~2000 chars per field, safe for most APIs
        self.keyword_batch_size = 15
        
        ensure_dir(f"{self.output_dir}/labels")
        ensure_dir(f"{self.output_dir}/labels/.progress")
    
    def download(self, therapeutic_area):
        """Download drug labels for therapeutic area with incremental saves"""
        
        if not self.config['labels']['enabled']:
            print(f"\n[LABELS] DISABLED in config")
            return []
        
        # Setup file paths
        today = datetime.now().strftime('%Y%m%d')
        output_file = f"{self.output_dir}/labels/{therapeutic_area}_labels_{today}.json"
        progress_file = f"{self.output_dir}/labels/.progress/{therapeutic_area}_{today}.json"
        
        # Check if we have a completed file for today
        if os.path.exists(output_file) and not os.path.exists(progress_file):
            print(f"\n[LABELS] {therapeutic_area}")
            print(f"  [OK] Using existing completed file: {os.path.basename(output_file)}")
            with open(output_file, 'r') as f:
                return json.load(f)
        
        print(f"\n[LABELS] Downloading for {therapeutic_area}...")
        
        # Get keywords
        all_keywords = get_expanded_keywords(therapeutic_area)
        
        # Apply max_diseases limit if in test mode
        max_diseases = self.config['labels'].get('max_diseases')
        if max_diseases:
            diseases = THERAPEUTIC_AREAS[therapeutic_area]['rare_diseases']
            drug_classes = THERAPEUTIC_AREAS[therapeutic_area]['drug_classes']
            limited_canonical = (diseases + drug_classes)[:max_diseases]
            all_keywords = limited_canonical
            print(f"  Limiting to first {max_diseases} canonical terms (test mode)")
        
        # Batch keywords to avoid URI too long errors
        keyword_batches = self._batch_keywords(all_keywords)
        print(f"  Searching {len(all_keywords)} keywords in {len(keyword_batches)} batches...")
        
        # Load existing progress
        all_results, seen_set_ids, search_state = self._load_progress(output_file, progress_file)
        
        if all_results:
            print(f"  [RESUME] Continuing from {len(all_results)} existing results")
        
        endpoint = f"{self.base_url}/drug/label.json"
        search_fields = ['indications_and_usage', 'description', 'openfda.pharm_class_epc']
        
        result_counter = 0
        save_interval = 100  # Save every 100 new results
        
        for field_idx, field in enumerate(search_fields):
            # Skip fields we've already completed
            if search_state.get('completed_fields', []) and field in search_state['completed_fields']:
                print(f"  [SKIP] Field already completed: {field}")
                continue
            
            print(f"  Searching field: {field} ({field_idx + 1}/{len(search_fields)})")
            
            # Process each keyword batch
            for batch_idx, keyword_batch in enumerate(keyword_batches):
                # Skip batches we've already completed for this field
                batch_key = f"{field}_batch_{batch_idx}"
                if search_state.get('completed_batches', {}).get(batch_key):
                    continue
                
                if batch_idx % 5 == 0 and batch_idx > 0:
                    print(f"    Progress: batch {batch_idx}/{len(keyword_batches)} ({len(all_results)} labels so far)")
                
                # Search this batch of keywords
                batch_results = self._search_keyword_batch(
                    endpoint, 
                    field, 
                    keyword_batch,
                    seen_set_ids
                )
                
                # Add unique results
                if batch_results:
                    for result in batch_results:
                        set_id = result.get('set_id')
                        if set_id and set_id not in seen_set_ids:
                            seen_set_ids.add(set_id)
                            all_results.append(result)
                            result_counter += 1
                    
                    # Save progress every save_interval new results
                    if result_counter >= save_interval:
                        # Mark this batch as completed
                        if 'completed_batches' not in search_state:
                            search_state['completed_batches'] = {}
                        search_state['completed_batches'][batch_key] = True
                        
                        self._save_progress(
                            output_file,
                            progress_file,
                            all_results,
                            search_state
                        )
                        print(f"    [SAVED] Checkpoint: {len(all_results)} unique labels")
                        result_counter = 0
                
                # Mark batch as completed
                if 'completed_batches' not in search_state:
                    search_state['completed_batches'] = {}
                search_state['completed_batches'][batch_key] = True
            
            # Mark field as completed
            if 'completed_fields' not in search_state:
                search_state['completed_fields'] = []
            search_state['completed_fields'].append(field)
            
            # Save after completing each field
            self._save_progress(output_file, progress_file, all_results, search_state)
            print(f"  Completed field: {field} - Total: {len(all_results)} unique labels")
        
        # Final save and cleanup
        self._finalize(output_file, progress_file, all_results)
        
        print(f"  [OK] Downloaded {len(all_results)} unique drug labels")
        return all_results
    
    def _batch_keywords(self, keywords):
        """Split keywords into batches to avoid URI too long errors"""
        batches = []
        for i in range(0, len(keywords), self.keyword_batch_size):
            batch = keywords[i:i + self.keyword_batch_size]
            batches.append(batch)
        return batches
    
    def _search_keyword_batch(self, endpoint, field, keywords, seen_set_ids):
        """Search a single batch of keywords"""
        search_query = self._build_search_query(keywords, field)
        
        skip = 0
        limit = 100
        max_results = 1000  # Max per batch to avoid too much data
        
        batch_results = []
        
        while skip < max_results:
            params = {
                "search": search_query,
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
                
                # Collect results
                batch_results.extend(data['results'])
                
                # Check if we've reached the end
                if skip + limit >= data['meta']['results']['total']:
                    break
                
                skip += limit
                
            except Exception as e:
                # Log error but continue - some batches may fail
                if '404' not in str(e) and '400' not in str(e):
                    print(f"      Error in batch: {str(e)[:80]}")
                break
        
        return batch_results
    
    def _build_search_query(self, keywords, field):
        """Build OR search query for a batch of keywords"""
        terms = [f'{field}:"{keyword}"' for keyword in keywords]
        return ' OR '.join(terms)
    
    def _load_progress(self, output_file, progress_file):
        """Load existing progress if available"""
        all_results = []
        seen_set_ids = set()
        search_state = {}
        
        # Load existing data
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    all_results = json.load(f)
                    # Rebuild seen set IDs
                    seen_set_ids = {r.get('set_id') for r in all_results if r.get('set_id')}
            except:
                pass
        
        # Load progress state
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    search_state = json.load(f)
            except:
                pass
        
        return all_results, seen_set_ids, search_state
    
    def _save_progress(self, output_file, progress_file, all_results, search_state):
        """Save current progress"""
        # Save data
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save state
        with open(progress_file, 'w') as f:
            json.dump(search_state, f, indent=2)
    
    def _finalize(self, output_file, progress_file, all_results):
        """Finalize download and remove progress file"""
        # Final save
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Remove progress file to indicate completion
        if os.path.exists(progress_file):
            os.remove(progress_file)