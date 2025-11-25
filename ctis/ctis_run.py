#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTIS Extraction Runner
Simple script to run CTIS data extraction with all parameters configured in code.
No command-line arguments needed - just edit the configuration section below.
ctis/ctis_run.py
"""

import sys
import time
import sqlite3
from pathlib import Path

# Import from modular structure
try:
    from ctis_database import init_db
    from ctis_utils import setup_output_dir, log
    from ctis_discovery import iter_ct_numbers_segmented
    from ctis_processor import process_multiple_trials, process_single_trial
    from ctis_http import RateLimiter, create_session, warm_up
    from ctis_config import PORTAL_URL
    import ctis_http  # For setting GLOBAL_RATE_LIMITER
except ImportError as e:
    print(f"ERROR: Could not import extractor modules: {e}")
    print("Make sure all ctis_*.py files are in the same directory.")
    sys.exit(1)


# ============================================================================
# CONFIGURATION - EDIT THESE PARAMETERS
# ============================================================================

class CTISConfig:
    """Configuration for CTIS extraction"""
    
    # ========== EXTRACTION MODE ============================
    # Choose ONE mode by setting it to a value, others to None
    
    # Mode 1: Extract a single trial by CT number
    #SINGLE_TRIAL = "2024-514133-38-00"
    SINGLE_TRIAL = None
    
    # Mode 2: Extract a specific number of trials
    TRIAL_COUNT = 20000  # Extract first N trials (set to None to disable)
    # TRIAL_COUNT = None  # Extract first N trials (set to None to disable)
    
    # Mode 3: Extract ALL trials (can take hours/days!)
    EXTRACT_ALL = None  # Set to True to extract entire database
    
    # ========== RARE DISEASE FILTER ========================================
    # Set to True to ONLY extract trials marked as rare diseases
    # Set to False to extract all trials (default behavior)
    FILTER_RARE_DISEASE_ONLY = True  # Set to True to enable filter
    
    # ========== OUTPUT SETTINGS ================================================
    OUTPUT_DIR = Path("ctis-out")  # Where to save database and files
    RESET_DATABASE = False  # Set to True to delete existing data and start fresh
    
    # ========== PERFORMANCE SETTINGS =================================================
    MAX_WORKERS = 3  # Number of concurrent download threads (1-5 recommended)
    RATE_LIMIT_RPS = 2.0  # Max requests per second (2-3 recommended, don't go higher!)
    PAGE_SIZE = 100  # Results per page (50-100 recommended)
    REQUEST_TIMEOUT = 60.0  # Seconds to wait for each request
    
    # ========== RETRY SETTINGS ===========================================
    MAX_RETRIES = 6  # How many times to retry failed requests
    BASE_BACKOFF = 1.0  # Initial wait time for retries (doubles each time)
    
    # ========== UPDATE BEHAVIOR =================================================
    CHECK_FOR_UPDATES = True  # If False, re-downloads all trials even if already in DB
    
    # ========== API ENDPOINTS (normally don't change these) ==========
    BASE_URL = "https://euclinicaltrials.eu"
    SEARCH_URL = f"{BASE_URL}/ctis-public-api/search"
    DETAIL_URL = f"{BASE_URL}/ctis-public-api/retrieve/{{ct}}"
    PORTAL_URL = f"{BASE_URL}/search-for-clinical-trials/?lang=en"


# ============================================================================
# RUN SCRIPT - DON'T EDIT BELOW UNLESS YOU KNOW WHAT YOU'RE DOING
# ============================================================================

def validate_config():
    """Validate configuration settings"""
    mode_count = sum([
        CTISConfig.SINGLE_TRIAL is not None,
        CTISConfig.TRIAL_COUNT is not None,
        bool(CTISConfig.EXTRACT_ALL)  # Convert to bool to handle None/False
    ])
    
    if mode_count == 0:
        print("ERROR: No extraction mode selected!")
        print("Please set one of: SINGLE_TRIAL, TRIAL_COUNT, or EXTRACT_ALL")
        return False
    
    if mode_count > 1:
        print("ERROR: Multiple extraction modes selected!")
        print("Please set only ONE of: SINGLE_TRIAL, TRIAL_COUNT, or EXTRACT_ALL")
        return False
    
    if CTISConfig.MAX_WORKERS < 1 or CTISConfig.MAX_WORKERS > 10:
        print("WARNING: MAX_WORKERS should be between 1 and 10")
        print(f"Current value: {CTISConfig.MAX_WORKERS}")
    
    if CTISConfig.RATE_LIMIT_RPS > 5:
        print("WARNING: RATE_LIMIT_RPS > 5 may cause rate limiting by CTIS server")
        print(f"Current value: {CTISConfig.RATE_LIMIT_RPS}")
    
    return True


def print_config_summary():
    """Print a summary of the configuration"""
    print("=" * 80)
    print("CTIS EXTRACTION CONFIGURATION")
    print("=" * 80)
    
    # Determine mode
    if CTISConfig.SINGLE_TRIAL:
        mode = f"Single Trial: {CTISConfig.SINGLE_TRIAL}"
    elif CTISConfig.TRIAL_COUNT:
        mode = f"Extract {CTISConfig.TRIAL_COUNT} trials"
    elif CTISConfig.EXTRACT_ALL:
        mode = "Extract ALL trials (full database)"
    else:
        mode = "NONE (ERROR)"
    
    print(f"Mode:              {mode}")
    print(f"Rare Disease Only: {CTISConfig.FILTER_RARE_DISEASE_ONLY}")
    print(f"Output Directory:  {CTISConfig.OUTPUT_DIR}")
    print(f"Reset Database:    {CTISConfig.RESET_DATABASE}")
    print(f"Check Updates:     {CTISConfig.CHECK_FOR_UPDATES}")
    print()
    print(f"Max Workers:       {CTISConfig.MAX_WORKERS}")
    print(f"Rate Limit:        {CTISConfig.RATE_LIMIT_RPS} req/sec")
    print(f"Page Size:         {CTISConfig.PAGE_SIZE}")
    print(f"Request Timeout:   {CTISConfig.REQUEST_TIMEOUT}s")
    print(f"Max Retries:       {CTISConfig.MAX_RETRIES}")
    print("=" * 80)
    print()


def run_extraction():
    """Main extraction function"""
    
    # Validate configuration
    if not validate_config():
        sys.exit(1)
    
    # Print configuration
    print_config_summary()
    
    # Setup paths
    out_dir = CTISConfig.OUTPUT_DIR
    db_path = out_dir / "ctis.db"
    ndjson_path = out_dir / "ctis_full.ndjson"
    ct_numbers_path = out_dir / "ct_numbers.txt"
    failed_path = out_dir / "failed_ctnumbers.txt"
    
    # Setup output directory
    log("Setting up output directory...")
    setup_output_dir(out_dir, reset=CTISConfig.RESET_DATABASE)
    
    if CTISConfig.RESET_DATABASE:
        log("Resetting database and output files...")
        for p in (ndjson_path, ct_numbers_path, failed_path):
            if p.exists():
                p.unlink()
    
    # Initialize database
    log("Initializing database...")
    conn = init_db(db_path, reset=CTISConfig.RESET_DATABASE)
    
    # Setup HTTP session
    log("Setting up HTTP session...")
    session = create_session()
    
    # Warm up connection
    try:
        warm_up(session)
    except Exception:
        pass
    
    # Setup rate limiter
    rate_limiter = RateLimiter(CTISConfig.RATE_LIMIT_RPS)
    
    # Set global rate limiter
    ctis_http.GLOBAL_RATE_LIMITER = rate_limiter
    
    try:
        # Execute based on mode
        if CTISConfig.SINGLE_TRIAL:
            # Single trial mode
            log(f"Extracting single trial: {CTISConfig.SINGLE_TRIAL}")
            success = process_single_trial(
                CTISConfig.SINGLE_TRIAL,
                conn,
                session,
                out_dir,
                ndjson_path
            )
            
            if success:
                log("Single trial extraction completed successfully!")
            else:
                log("Single trial extraction failed!", "ERROR")
                sys.exit(1)
        
        elif CTISConfig.TRIAL_COUNT or CTISConfig.EXTRACT_ALL:
            # Multiple trials mode
            limit = None if CTISConfig.EXTRACT_ALL else CTISConfig.TRIAL_COUNT
            
            if limit:
                if CTISConfig.FILTER_RARE_DISEASE_ONLY:
                    log(f"Extracting {limit} RARE DISEASE trials...")
                else:
                    log(f"Extracting {limit} trials...")
            else:
                if CTISConfig.FILTER_RARE_DISEASE_ONLY:
                    log("Extracting ALL RARE DISEASE trials (full database)...")
                else:
                    log("Extracting ALL trials (full database)...")
            
            # Discovery phase
            log("Starting trial discovery...")
            all_trials, trials_to_update = iter_ct_numbers_segmented(
                session=session,
                limit=limit,
                check_updates=CTISConfig.CHECK_FOR_UPDATES,
                ct_numbers_path=ct_numbers_path,
                db_path=db_path,
                page_size=CTISConfig.PAGE_SIZE,
                filter_rare_disease=CTISConfig.FILTER_RARE_DISEASE_ONLY,
            )
            
            if not trials_to_update:
                log("No trials need updating - all current!")
            else:
                # Extraction phase
                log(f"Extracting {len(trials_to_update)} trials...")
                process_multiple_trials(
                    trials_to_update,
                    conn,
                    session,
                    db_path,
                    ndjson_path,
                    failed_path
                )
                
                log("Extraction completed!")
    
    except KeyboardInterrupt:
        log("\nInterrupted by user. Progress saved!", "WARN")
        log("Run again with RESET_DATABASE=False to resume.", "WARN")
        conn.commit()
        sys.exit(0)
    
    except Exception as e:
        log(f"Extraction failed with error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup
        try:
            conn.commit()
        except Exception:
            pass
        conn.close()
        session.close()
    
    # Print final summary
    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Database:  {db_path}")
    print(f"NDJSON:    {ndjson_path}")
    
    # Print statistics
    try:
        conn_stats = sqlite3.connect(db_path)
        cursor = conn_stats.cursor()
        
        trial_count = cursor.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
        site_count = cursor.execute("SELECT COUNT(*) FROM trial_sites").fetchone()[0]
        people_count = cursor.execute("SELECT COUNT(*) FROM trial_people").fetchone()[0]
        
        # Count rare disease trials
        rare_disease_count = cursor.execute(
            "SELECT COUNT(*) FROM trials WHERE isRareDisease = 1"
        ).fetchone()[0]
        
        print()
        print("Database Statistics:")
        print(f"  Trials:         {trial_count:,}")
        print(f"  Rare Disease:   {rare_disease_count:,}")
        print(f"  Sites:          {site_count:,}")
        print(f"  People:         {people_count:,}")
        
        conn_stats.close()
    except Exception:
        pass
    
    print("=" * 80)


def main():
    """Entry point"""
    print()
    print("=" * 62)
    print("  CTIS Clinical Trials Data Extractor")
    print("  EU Clinical Trials Information System")
    print("  RARE DISEASE FILTER ENABLED" if CTISConfig.FILTER_RARE_DISEASE_ONLY else "")
    print("=" * 62)
    print()
    
    run_extraction()


if __name__ == "__main__":
    main()