"""
Main Sync Orchestrator - PARALLEL VERSION v2.1
Coordinates all downloaders using syncher_keys.py config

FEATURES: 
- Parallel processing with ThreadPoolExecutor for 2-4x speedup
- All downloaders properly imported and initialized
- Orphan drugs skipped (manual download required)
- Proper error handling
- Thread-safe operations
- Better progress reporting

UPDATED v2.1:
- Improved logging
- Better error recovery
- Stats summary at end
"""

import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Import from YOUR config files
from syncher_keys import (
    MODE, 
    SYNC_AREAS, 
    OUTPUT_DIR, 
    validate_config, 
    print_config_summary,
    print_expected_results
)

# Import all downloaders
from .downloaders.labels import LabelsDownloader
from .downloaders.approval_packages import ApprovalPackagesDownloader
from .downloaders.adverse_events import AdverseEventsDownloader
from .downloaders.enforcement import EnforcementDownloader

from .utils.helpers import extract_drug_names_from_labels


class FDASyncOrchestrator:
    """Orchestrates the complete FDA data sync with parallel processing"""
    
    def __init__(self, max_workers=4):
        """
        Initialize orchestrator with parallel processing support
        
        Args:
            max_workers: Number of concurrent workers (default: 4)
                        - 2 workers: Safe for most systems
                        - 4 workers: Good balance (recommended)
                        - 8 workers: Fast, requires good connection
        """
        self.max_workers = max_workers
        self.print_lock = Lock()  # Thread-safe printing
        self.start_time = None
        
        print(f"\n{'='*70}")
        print("FDA DATA SYNCER - PARALLEL MODE v2.1")
        print(f"Mode: {MODE}")
        print(f"Workers: {max_workers}")
        print(f"Areas: {', '.join(SYNC_AREAS)}")
        print(f"Output: {OUTPUT_DIR}")
        print(f"{'='*70}\n")
        
        # Initialize all downloaders
        self.labels_dl = LabelsDownloader()
        self.packages_dl = ApprovalPackagesDownloader()
        self.adverse_dl = AdverseEventsDownloader()
        self.enforcement_dl = EnforcementDownloader()
    
    def _print_safe(self, message):
        """Thread-safe printing"""
        with self.print_lock:
            print(message)
    
    def run(self):
        """Run complete sync with parallel processing"""
        
        self.start_time = datetime.now()
        results = {}
        
        for area in SYNC_AREAS:
            self._print_safe(f"\n{'='*70}")
            self._print_safe(f"THERAPEUTIC AREA: {area.upper()}")
            self._print_safe(f"{'='*70}")
            
            area_start = time.time()
            results[area] = {}
            
            # Step 1: Drug Labels (sequential - needed for subsequent steps)
            self._print_safe("\n[STEP 1/5] Drug Labels...")
            try:
                labels = self.labels_dl.download(area)
                results[area]['labels'] = len(labels) if isinstance(labels, list) else 0
                self._print_safe(f"  ✓ Completed: {results[area]['labels']} labels")
            except Exception as e:
                self._print_safe(f"  ✗ Labels error: {e}")
                results[area]['labels'] = 0
                labels = []
            
            # Step 2: Orphan Drugs - SKIPPED (manual download required)
            self._print_safe("\n[STEP 2/5] Orphan Drugs - SKIPPED (manual download)")
            self._print_safe("  Download from: https://www.accessdata.fda.gov/scripts/opdlisting/oopd/")
            self._print_safe(f"  Save to: {OUTPUT_DIR}/orphan_drugs/")
            results[area]['orphan_drugs'] = 'manual'
            
            # Extract drug names once (now with filtering!)
            drug_names = extract_drug_names_from_labels(labels)
            self._print_safe(f"\n  📋 Extracted {len(drug_names)} pharmaceutical drug names from labels")
            
            # Step 3-5: Run remaining downloads in parallel
            self._print_safe(f"\n[STEP 3-5] Running parallel downloads with {self.max_workers} workers...")
            parallel_results = self._run_parallel_downloads(area, drug_names)
            results[area].update(parallel_results)
            
            # Print area summary
            area_duration = time.time() - area_start
            self._print_safe(f"\n  📊 {area.upper()} completed in {area_duration/60:.1f} minutes")
        
        # Print summary
        duration = datetime.now() - self.start_time
        self._print_summary(results, duration)
        
        return results
    
    def _run_parallel_downloads(self, area, drug_names):
        """Run approval packages, adverse events, and enforcement in parallel"""
        
        tasks = []
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            
            # Approval Packages
            if self._should_run('integrated_reviews'):
                future_packages = executor.submit(
                    self._download_approval_packages, 
                    area, 
                    drug_names
                )
                tasks.append(('approval_packages', future_packages))
            else:
                results['approval_packages'] = 0
            
            # Adverse Events
            if self._should_run('adverse_events'):
                future_adverse = executor.submit(
                    self._download_adverse_events,
                    area,
                    drug_names
                )
                tasks.append(('adverse_events', future_adverse))
            else:
                results['adverse_events'] = 0
            
            # Enforcement Reports
            if self._should_run('enforcement'):
                future_enforcement = executor.submit(
                    self._download_enforcement,
                    area
                )
                tasks.append(('enforcement', future_enforcement))
            else:
                results['enforcement'] = 0
            
            # Wait for all tasks to complete
            for name, future in tasks:
                try:
                    result = future.result()
                    results[name] = result
                    self._print_safe(f"  ✓ {name.replace('_', ' ').title()}: {result}")
                except Exception as e:
                    self._print_safe(f"  ✗ {name.replace('_', ' ').title()}: Error - {e}")
                    results[name] = 0
        
        return results
    
    def _should_run(self, component):
        """Check if component should run based on config"""
        from syncher_keys import get_sync_config
        config = get_sync_config()
        return config.get(component, {}).get('enabled', False)
    
    def _download_approval_packages(self, area, drug_names):
        """Download approval packages (thread worker)"""
        try:
            self._print_safe("  [Worker] Starting approval packages download...")
            packages = self.packages_dl.download(area, drug_names)
            count = len(packages) if isinstance(packages, list) else 0
            return count
        except Exception as e:
            self._print_safe(f"  [Worker] Approval packages error: {e}")
            return 0
    
    def _download_adverse_events(self, area, drug_names):
        """Download adverse events (thread worker)"""
        try:
            self._print_safe("  [Worker] Starting adverse events download...")
            adverse = self.adverse_dl.download(area, drug_names)
            count = len(adverse) if isinstance(adverse, list) else 0
            return count
        except Exception as e:
            self._print_safe(f"  [Worker] Adverse events error: {e}")
            return 0
    
    def _download_enforcement(self, area):
        """Download enforcement reports (thread worker)"""
        try:
            self._print_safe("  [Worker] Starting enforcement download...")
            enforcement = self.enforcement_dl.download(area)
            count = len(enforcement) if isinstance(enforcement, list) else 0
            return count
        except Exception as e:
            self._print_safe(f"  [Worker] Enforcement error: {e}")
            return 0
    
    def _print_summary(self, results, duration):
        """Print final summary"""
        
        print(f"\n{'='*70}")
        print("✅ PARALLEL SYNC COMPLETE!")
        print(f"{'='*70}")
        print(f"Duration: {duration}")
        print(f"Workers Used: {self.max_workers}")
        print("\nResults by therapeutic area:")
        
        total_items = 0
        for area, data in results.items():
            print(f"\n{area.upper()}:")
            for source, count in data.items():
                if isinstance(count, int):
                    print(f"  {source.replace('_', ' ').title()}: {count:,}")
                    total_items += count
                else:
                    print(f"  {source.replace('_', ' ').title()}: {count}")
        
        print("\n📊 TOTALS:")
        print(f"  Total Items Downloaded: {total_items:,}")
        print(f"  Therapeutic Areas: {len(results)}")
        print(f"  Estimated Speedup: 2x-{self.max_workers}x faster than sequential")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("  - Run fda_data_quality_check.py to verify data integrity")
        print("  - Check FDA_DATA/ folder for all downloaded files")
        print("  - For orphan drugs, manually download from FDA website")
        
        print(f"\n{'='*70}\n")


def main():
    """Main entry point with configurable workers"""
    
    # Validate config
    if not validate_config():
        print("\n[X] Configuration errors detected")
        print("Please check syncher_keys.py\n")
        return
    
    # Print config summary
    print_config_summary()
    
    # Print expected results for full mode
    if MODE == 'full':
        print_expected_results()
    
    # Determine worker count based on mode
    if MODE == 'test':
        max_workers = 2  # Conservative for testing
    elif MODE == 'daily':
        max_workers = 4  # Balanced for daily updates
    else:  # full
        max_workers = 4  # Reduced from 6 to be more conservative
    
    print(f"\n{'='*70}")
    print("PARALLEL PROCESSING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Mode: {MODE}")
    print(f"Workers: {max_workers}")
    print(f"Expected Speedup: 2x-{max_workers}x")
    print(f"{'='*70}\n")
    
    # Confirm for full mode
    if MODE == 'full':
        try:
            response = input("\nContinue with FULL sync? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Sync cancelled.")
                return
        except (EOFError, KeyboardInterrupt):
            print("Running in non-interactive mode...")
    
    # Run sync with parallel processing
    orchestrator = FDASyncOrchestrator(max_workers=max_workers)
    orchestrator.run()


if __name__ == "__main__":
    main()
