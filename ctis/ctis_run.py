#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTIS Extraction Runner
Runs CTIS data extraction using settings from ctis_config.py

>>> EDIT ctis_config.py TO CHANGE SETTINGS <<<

ctis/ctis_run.py
"""

import sys
import time
import sqlite3
from pathlib import Path

# Import configuration - ALL settings come from here
from ctis_config import (
    # Extraction mode
    SINGLE_TRIAL, TRIAL_COUNT, EXTRACT_ALL,
    FILTER_RARE_DISEASE_ONLY,
    # PDF settings
    DOWNLOAD_PDFS, DOWNLOAD_ONLY, DOWNLOAD_FILE_TYPES, ONLY_FOR_PUBLICATION,
    # Output settings
    OUT_DIR, RESET_DATABASE, CHECK_FOR_UPDATES,
    # Performance settings
    MAX_WORKERS, RATE_LIMIT_RPS, PAGE_SIZE, REQUEST_TIMEOUT, MAX_RETRIES,
    # Paths
    PORTAL_URL,
)

# Import from modular structure
try:
    from ctis_database import init_db
    from ctis_utils import setup_output_dir, log
    from ctis_discovery import iter_ct_numbers_segmented
    from ctis_processor import process_multiple_trials, process_single_trial
    from ctis_http import RateLimiter, create_session, warm_up
    import ctis_http
except ImportError as e:
    print(f"ERROR: Could not import extractor modules: {e}")
    print("Make sure all ctis_*.py files are in the same directory.")
    sys.exit(1)

# Try to import PDF downloader (includes process_trial_documents)
try:
    from ctis_pdf_downloader import (
        create_documents_table, 
        get_document_stats,
        process_trial_documents
    )
    HAS_PDF_DOWNLOADER = True
except ImportError as e:
    HAS_PDF_DOWNLOADER = False
    process_trial_documents = None


def validate_config():
    """Validate configuration settings"""
    mode_count = sum([
        SINGLE_TRIAL is not None,
        TRIAL_COUNT is not None,
        bool(EXTRACT_ALL)
    ])
    
    if mode_count == 0:
        print("ERROR: No extraction mode selected!")
        print("Edit ctis_config.py and set one of: SINGLE_TRIAL, TRIAL_COUNT, or EXTRACT_ALL")
        return False
    
    if mode_count > 1:
        print("ERROR: Multiple extraction modes selected!")
        print("Edit ctis_config.py and set only ONE of: SINGLE_TRIAL, TRIAL_COUNT, or EXTRACT_ALL")
        return False
    
    if MAX_WORKERS < 1 or MAX_WORKERS > 10:
        print(f"WARNING: MAX_WORKERS should be between 1 and 10 (current: {MAX_WORKERS})")
    
    if RATE_LIMIT_RPS > 5:
        print(f"WARNING: RATE_LIMIT_RPS > 5 may cause rate limiting (current: {RATE_LIMIT_RPS})")
    
    return True


def print_config_summary():
    """Print a summary of the configuration"""
    print("=" * 80)
    print("CTIS EXTRACTION CONFIGURATION (from ctis_config.py)")
    print("=" * 80)
    
    # Determine mode
    if SINGLE_TRIAL:
        mode = f"Single Trial: {SINGLE_TRIAL}"
    elif TRIAL_COUNT:
        mode = f"Extract {TRIAL_COUNT} trials"
    elif EXTRACT_ALL:
        mode = "Extract ALL trials (full database)"
    else:
        mode = "NONE (ERROR)"
    
    print(f"Mode:              {mode}")
    print(f"Rare Disease Only: {FILTER_RARE_DISEASE_ONLY}")
    print(f"Output Directory:  {OUT_DIR}")
    print(f"Reset Database:    {RESET_DATABASE}")
    print(f"Check Updates:     {CHECK_FOR_UPDATES}")
    print()
    
    # PDF Download settings
    print("--- PDF Download Settings ---")
    print(f"Download PDFs:     {DOWNLOAD_PDFS}")
    if DOWNLOAD_PDFS:
        print(f"Download Only:     {DOWNLOAD_ONLY}")
        print(f"File Types:        {DOWNLOAD_FILE_TYPES or 'All'}")
        print(f"For Publication:   {ONLY_FOR_PUBLICATION}")
    print()
    
    print(f"Max Workers:       {MAX_WORKERS}")
    print(f"Rate Limit:        {RATE_LIMIT_RPS} req/sec")
    print(f"Page Size:         {PAGE_SIZE}")
    print(f"Request Timeout:   {REQUEST_TIMEOUT}s")
    print(f"Max Retries:       {MAX_RETRIES}")
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
    out_dir = OUT_DIR
    db_path = out_dir / "ctis.db"
    ndjson_path = out_dir / "ctis_full.ndjson"
    ct_numbers_path = out_dir / "ct_numbers.txt"
    failed_path = out_dir / "failed_ctnumbers.txt"
    pdf_dir = out_dir / "pdf"
    
    # Setup output directory
    log("Setting up output directory...")
    setup_output_dir(out_dir, reset=RESET_DATABASE)
    
    # Create PDF directory if PDF download enabled
    if DOWNLOAD_PDFS:
        pdf_dir.mkdir(parents=True, exist_ok=True)
    
    if RESET_DATABASE:
        log("Resetting database and output files...")
        for p in (ndjson_path, ct_numbers_path, failed_path):
            if p.exists():
                p.unlink()
    
    # Initialize database
    log("Initializing database...")
    conn = init_db(db_path, reset=RESET_DATABASE)
    
    # Initialize documents table if PDF download enabled
    if DOWNLOAD_PDFS and HAS_PDF_DOWNLOADER:
        create_documents_table(conn)
    
    # Setup HTTP session
    log("Setting up HTTP session...")
    session = create_session()
    
    # Warm up connection
    try:
        warm_up(session)
    except Exception:
        pass
    
    # Setup rate limiter
    rate_limiter = RateLimiter(RATE_LIMIT_RPS)
    ctis_http.GLOBAL_RATE_LIMITER = rate_limiter
    
    # Track trials for document download
    processed_trials = []
    
    try:
        # Handle DOWNLOAD_ONLY mode
        if DOWNLOAD_ONLY and DOWNLOAD_PDFS:
            log("Download-only mode: Downloading PDFs for existing trials...")
            
            if not HAS_PDF_DOWNLOADER:
                log("PDF downloader module not available", "WARN")
                log("Make sure ctis_pdf_downloader.py is in the same directory", "WARN")
                log("And run: pip install playwright && playwright install chromium", "WARN")
            else:
                # Get all trial CT numbers from database
                cursor = conn.execute("SELECT ctNumber FROM trials")
                processed_trials = [row[0] for row in cursor.fetchall()]
                
                if not processed_trials:
                    log("No trials in database! Run extraction first.")
                    sys.exit(1)
                
                log(f"Found {len(processed_trials)} trials in database")
                
                # Process documents (with update checking)
                process_trial_documents(processed_trials, conn, session, out_dir,
                                       check_updates=CHECK_FOR_UPDATES)
        
        # Execute based on mode
        elif SINGLE_TRIAL:
            # Single trial mode
            log(f"Extracting single trial: {SINGLE_TRIAL}")
            success = process_single_trial(
                SINGLE_TRIAL,
                conn,
                session,
                out_dir,
                ndjson_path
            )
            
            if success:
                log("Single trial extraction completed successfully!")
                processed_trials = [SINGLE_TRIAL]
            else:
                log("Single trial extraction failed!", "ERROR")
                sys.exit(1)
        
        elif TRIAL_COUNT or EXTRACT_ALL:
            # Multiple trials mode
            limit = None if EXTRACT_ALL else TRIAL_COUNT
            
            if limit:
                if FILTER_RARE_DISEASE_ONLY:
                    log(f"Extracting {limit} RARE DISEASE trials...")
                else:
                    log(f"Extracting {limit} trials...")
            else:
                if FILTER_RARE_DISEASE_ONLY:
                    log("Extracting ALL RARE DISEASE trials (full database)...")
                else:
                    log("Extracting ALL trials (full database)...")
            
            # Discovery phase
            log("Starting trial discovery...")
            all_trials, trials_to_update = iter_ct_numbers_segmented(
                session=session,
                limit=limit,
                check_updates=CHECK_FOR_UPDATES,
                ct_numbers_path=ct_numbers_path,
                db_path=db_path,
                page_size=PAGE_SIZE,
                filter_rare_disease=FILTER_RARE_DISEASE_ONLY,
            )
            
            if not trials_to_update:
                log("No trials need updating - all current!")
                # Still get all trials for PDF download
                cursor = conn.execute("SELECT ctNumber FROM trials")
                processed_trials = [row[0] for row in cursor.fetchall()]
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
                processed_trials = trials_to_update
            
            # PDF Download phase (after trial extraction)
            if DOWNLOAD_PDFS and HAS_PDF_DOWNLOADER and processed_trials:
                log("\n" + "=" * 60)
                log("Starting PDF/Document Download Phase")
                log("=" * 60)
                process_trial_documents(processed_trials, conn, session, out_dir, 
                                       check_updates=CHECK_FOR_UPDATES)
    
    except KeyboardInterrupt:
        log("\nInterrupted by user. Progress saved!", "WARN")
        log("Run again to resume.", "WARN")
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
    if DOWNLOAD_PDFS:
        print(f"PDFs:      {pdf_dir}")
    
    # Print statistics
    try:
        conn_stats = sqlite3.connect(db_path)
        cursor = conn_stats.cursor()
        
        trial_count = cursor.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
        site_count = cursor.execute("SELECT COUNT(*) FROM trial_sites").fetchone()[0]
        people_count = cursor.execute("SELECT COUNT(*) FROM trial_people").fetchone()[0]
        rare_disease_count = cursor.execute(
            "SELECT COUNT(*) FROM trials WHERE isRareDisease = 1"
        ).fetchone()[0]
        
        print()
        print("Database Statistics:")
        print(f"  Trials:         {trial_count:,}")
        print(f"  Rare Disease:   {rare_disease_count:,}")
        print(f"  Sites:          {site_count:,}")
        print(f"  People:         {people_count:,}")
        
        # Document statistics
        if DOWNLOAD_PDFS and HAS_PDF_DOWNLOADER:
            try:
                doc_stats = get_document_stats(conn_stats)
                print()
                print("Document Statistics:")
                print(f"  Total Trials:    {doc_stats.get('total', 0):,}")
                print(f"  Downloaded:      {doc_stats.get('downloaded', 0):,}")
                print(f"  Pending:         {doc_stats.get('pending', 0):,}")
                print(f"  Failed:          {doc_stats.get('failed', 0):,}")
                total_mb = doc_stats.get('total_size_bytes', 0) / (1024 * 1024)
                print(f"  Total Size:      {total_mb:.2f} MB")
            except Exception:
                pass
        
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
    if FILTER_RARE_DISEASE_ONLY:
        print("  RARE DISEASE FILTER ENABLED")
    if DOWNLOAD_PDFS:
        print("  PDF DOWNLOAD ENABLED")
    print("=" * 62)
    print()
    
    # Check PDF downloader availability
    if DOWNLOAD_PDFS and not HAS_PDF_DOWNLOADER:
        print("WARNING: PDF downloading enabled but ctis_pdf_downloader.py not found!")
        print("Make sure ctis_pdf_downloader.py is in the same directory.")
        print("Also run: pip install playwright && playwright install chromium")
        print()
    
    run_extraction()


if __name__ == "__main__":
    main()