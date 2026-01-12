#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTIS PDF/Document Downloader Module
Downloads trial document packages using the CTIS public view page.

WORKING URL: https://euclinicaltrials.eu/ctis-public/view/{ct_number}

This module uses Playwright browser automation because direct API endpoints
return 403 Forbidden. The CTIS public view page has a "Download clinical trial"
button that downloads a ZIP containing all public documents.

SETUP:
    pip install playwright
    playwright install chromium

ctis/ctis_pdf_downloader.py
VERSION: 4.0.0
"""

import sys
import time
import sqlite3
import zipfile
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timezone

try:
    from ctis_utils import log
except ImportError:
    # Standalone mode
    def log(msg, level="INFO"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [{level}] {msg}")

# ===================== Configuration =====================

CTIS_BASE = "https://euclinicaltrials.eu"
TRIAL_VIEW_URL = f"{CTIS_BASE}/ctis-public/view/{{ct}}"

PDF_FOLDER = "pdf"
HEADLESS = True
DOWNLOAD_TIMEOUT_MS = 120000  # 2 minutes
PAGE_LOAD_TIMEOUT_MS = 60000  # 1 minute
DELAY_BETWEEN_DOWNLOADS = 2.0  # seconds
EXTRACT_PDFS = True  # Auto-extract PDFs from downloaded ZIPs
DELETE_ZIP_AFTER_EXTRACT = False  # Keep ZIP files after extraction


# ===================== Browser Management =====================

_browser_context = None
_playwright = None
_playwright_available = None


def _check_playwright():
    """Check if Playwright is available."""
    global _playwright_available
    
    if _playwright_available is not None:
        return _playwright_available
    
    try:
        from playwright.sync_api import sync_playwright
        _playwright_available = True
    except ImportError:
        _playwright_available = False
    
    return _playwright_available


def _init_browser():
    """Initialize Playwright browser (reused across downloads)."""
    global _browser_context, _playwright
    
    if _browser_context is not None:
        return _browser_context
    
    if not _check_playwright():
        log("Playwright not installed!", "ERROR")
        log("Run: pip install playwright && playwright install chromium", "ERROR")
        return None
    
    from playwright.sync_api import sync_playwright
    
    log("Initializing browser for document downloads...")
    _playwright = sync_playwright().start()
    
    browser = _playwright.chromium.launch(
        headless=HEADLESS,
        args=['--disable-blink-features=AutomationControlled']
    )
    
    _browser_context = browser.new_context(
        accept_downloads=True,
        viewport={'width': 1920, 'height': 1080},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    )
    
    log("Browser ready")
    return _browser_context


def _close_browser():
    """Close browser and cleanup."""
    global _browser_context, _playwright
    
    if _browser_context:
        try:
            _browser_context.close()
        except:
            pass
        _browser_context = None
    
    if _playwright:
        try:
            _playwright.stop()
        except:
            pass
        _playwright = None


# Register cleanup on exit
import atexit
atexit.register(_close_browser)


# ===================== ZIP Extraction =====================

def extract_pdfs_from_zip(zip_path: Path, output_dir: Path, ct_number: str) -> List[str]:
    """
    Extract PDF files from a downloaded ZIP archive.

    Args:
        zip_path: Path to the ZIP file
        output_dir: Directory to extract PDFs to
        ct_number: Trial CT number (used for prefixing files)

    Returns:
        List of extracted PDF file paths
    """
    extracted = []

    if not zip_path.exists():
        log(f"  ZIP file not found: {zip_path}", "WARN")
        return extracted

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                # Only extract PDF files
                if name.lower().endswith('.pdf'):
                    # Get just the filename (ignore folder structure in ZIP)
                    base_name = Path(name).name
                    # Prefix with CT number if not already prefixed
                    if not base_name.startswith(ct_number):
                        out_name = f"{ct_number}_{base_name}"
                    else:
                        out_name = base_name

                    out_path = output_dir / out_name

                    # Extract the file
                    with zf.open(name) as src:
                        content = src.read()
                        with open(out_path, 'wb') as dst:
                            dst.write(content)

                    extracted.append(str(out_path))
                    log(f"    Extracted: {out_name}")

        if extracted:
            log(f"  Extracted {len(extracted)} PDF(s) from ZIP")
        else:
            log(f"  No PDFs found in ZIP archive")

    except zipfile.BadZipFile:
        log(f"  Invalid ZIP file: {zip_path}", "WARN")
    except Exception as e:
        log(f"  Error extracting ZIP: {e}", "WARN")

    return extracted


# ===================== Database Functions =====================

def create_documents_table(conn: sqlite3.Connection):
    """Create table to track document/trial package downloads."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trial_downloads (
            ctNumber TEXT PRIMARY KEY,
            downloaded INTEGER DEFAULT 0,
            file_path TEXT,
            file_name TEXT,
            file_size INTEGER DEFAULT 0,
            downloaded_at TEXT,
            error TEXT
        )
    """)
    conn.commit()


def get_document_stats(conn: sqlite3.Connection) -> Dict:
    """Get download statistics."""
    stats = {
        "total": 0,
        "downloaded": 0,
        "pending": 0,
        "failed": 0,
        "total_size_bytes": 0,
    }
    
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM trial_downloads")
        stats["total"] = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM trial_downloads WHERE downloaded = 1")
        stats["downloaded"] = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM trial_downloads WHERE downloaded = 0 AND error IS NULL")
        stats["pending"] = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM trial_downloads WHERE downloaded = 0 AND error IS NOT NULL")
        stats["failed"] = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COALESCE(SUM(file_size), 0) FROM trial_downloads WHERE downloaded = 1")
        stats["total_size_bytes"] = cursor.fetchone()[0]
    except sqlite3.Error:
        pass
    
    return stats


def is_trial_downloaded(conn: sqlite3.Connection, ct_number: str) -> bool:
    """Check if a trial package has already been downloaded."""
    try:
        cursor = conn.execute(
            "SELECT downloaded FROM trial_downloads WHERE ctNumber = ?",
            (ct_number,)
        )
        row = cursor.fetchone()
        return row is not None and row[0] == 1
    except sqlite3.Error:
        return False


def needs_update(conn: sqlite3.Connection, ct_number: str) -> bool:
    """
    Check if a trial's documents need to be re-downloaded.
    
    Compares the trial's lastUpdated timestamp with when we downloaded.
    Returns True if:
    - Trial was never downloaded
    - Trial has been updated since last download
    """
    try:
        # Get trial's lastUpdated from trials table
        cursor = conn.execute(
            "SELECT lastUpdated FROM trials WHERE ctNumber = ?",
            (ct_number,)
        )
        trial_row = cursor.fetchone()
        if not trial_row or not trial_row[0]:
            return True  # No trial record, need to download
        
        trial_updated = trial_row[0]
        
        # Get our download timestamp
        cursor = conn.execute(
            "SELECT downloaded, downloaded_at FROM trial_downloads WHERE ctNumber = ?",
            (ct_number,)
        )
        download_row = cursor.fetchone()
        
        if not download_row or download_row[0] != 1:
            return True  # Never downloaded
        
        downloaded_at = download_row[1]
        if not downloaded_at:
            return True  # No timestamp recorded
        
        # Parse timestamps and compare
        try:
            # Try dateutil first (handles more formats)
            try:
                from dateutil import parser as date_parser
                trial_dt = date_parser.parse(trial_updated)
                download_dt = date_parser.parse(downloaded_at)
            except ImportError:
                # Fallback: simple ISO format parsing
                # Handle formats like "2024-11-15T10:30:00Z" or "2024-11-15T10:30:00+00:00"
                trial_updated_clean = trial_updated.replace('Z', '+00:00')
                downloaded_at_clean = downloaded_at.replace('Z', '+00:00')
                
                # Python 3.7+ fromisoformat
                trial_dt = datetime.fromisoformat(trial_updated_clean)
                download_dt = datetime.fromisoformat(downloaded_at_clean)
            
            # Make both timezone-aware for comparison
            if trial_dt.tzinfo is None:
                trial_dt = trial_dt.replace(tzinfo=timezone.utc)
            if download_dt.tzinfo is None:
                download_dt = download_dt.replace(tzinfo=timezone.utc)
            
            # If trial was updated AFTER we downloaded, need to re-download
            if trial_dt > download_dt:
                log(f"  Update detected: trial updated {trial_updated}, downloaded {downloaded_at}")
                return True
            
            return False  # Already up to date
            
        except Exception as e:
            # If we can't parse dates, assume we need to download
            log(f"  Warning: Could not parse dates for {ct_number}: {e}", "WARN")
            return True
            
    except sqlite3.Error:
        return True  # On error, assume we need to download


def update_download_status(conn: sqlite3.Connection, result: Dict):
    """Update download status in database."""
    try:
        conn.execute("""
            INSERT OR REPLACE INTO trial_downloads 
            (ctNumber, downloaded, file_path, file_name, file_size, downloaded_at, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result["ct_number"],
            1 if result["success"] else 0,
            result.get("file_path"),
            result.get("file_name"),
            result.get("file_size", 0),
            result.get("downloaded_at"),
            result.get("error"),
        ))
        conn.commit()
    except sqlite3.Error as e:
        log(f"Database error updating download status: {e}", "ERROR")


# ===================== Single Trial Download =====================

def download_trial(
    ct_number: str,
    output_dir: Path,
    conn: Optional[sqlite3.Connection] = None,
    skip_existing: bool = True,
    check_updates: bool = True
) -> Dict:
    """
    Download a trial's document package using browser automation.
    
    Downloads the ZIP file that CTIS generates when clicking "Download clinical trial".
    This ZIP contains HTML summary and all public PDF documents.
    
    Args:
        ct_number: Trial CT number (e.g., "2024-512203-40-00")
        output_dir: Base output directory (files go to output_dir/pdf/)
        conn: Database connection for tracking (optional)
        skip_existing: Skip if already downloaded
        check_updates: Re-download if trial was updated since last download
    
    Returns:
        Dict with download result:
        {
            "ct_number": str,
            "success": bool,
            "file_path": str or None,
            "file_name": str or None,
            "file_size": int,
            "error": str or None,
            "downloaded_at": str (ISO format)
        }
    """
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    
    result = {
        "ct_number": ct_number,
        "success": False,
        "file_path": None,
        "file_name": None,
        "file_size": 0,
        "error": None,
        "downloaded_at": None,
    }
    
    # Check if we need to download
    if skip_existing and conn:
        if check_updates:
            # Check if trial needs update (smarter check)
            if not needs_update(conn, ct_number):
                log(f"  Skipping {ct_number} (up to date)")
                result["success"] = True
                result["error"] = "skipped"
                return result
        else:
            # Simple check - just see if downloaded
            if is_trial_downloaded(conn, ct_number):
                log(f"  Skipping {ct_number} (already downloaded)")
                result["success"] = True
                result["error"] = "skipped"
                return result
    
    # Ensure output directory exists
    pdf_dir = output_dir / PDF_FOLDER
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Get browser
    context = _init_browser()
    if not context:
        result["error"] = "Browser not available (Playwright not installed)"
        if conn:
            update_download_status(conn, result)
        return result
    
    page = context.new_page()
    
    try:
        # Navigate to trial view page
        url = TRIAL_VIEW_URL.format(ct=ct_number)
        log(f"  Loading: {ct_number}")
        
        page.goto(url, wait_until="networkidle", timeout=PAGE_LOAD_TIMEOUT_MS)
        page.wait_for_timeout(2000)
        
        # Handle cookie consent if present
        try:
            cookie_btn = page.locator('button:has-text("Accept")')
            if cookie_btn.count() > 0 and cookie_btn.first.is_visible():
                cookie_btn.first.click()
                page.wait_for_timeout(1000)
        except:
            pass
        
        # Find download button
        download_selectors = [
            'button:has-text("Download clinical trial")',
            'a:has-text("Download clinical trial")',
            'button:has-text("Download")',
            'a:has-text("Download")',
        ]
        
        download_btn = None
        for selector in download_selectors:
            try:
                elements = page.locator(selector)
                if elements.count() > 0 and elements.first.is_visible():
                    download_btn = elements.first
                    break
            except:
                continue
        
        if not download_btn:
            result["error"] = "Download button not found on page"
            log(f"  ✗ {ct_number}: No download button found", "WARN")
            if conn:
                update_download_status(conn, result)
            return result
        
        # Click and wait for download
        with page.expect_download(timeout=DOWNLOAD_TIMEOUT_MS) as download_info:
            download_btn.click()
        
        download = download_info.value
        temp_path = download.path()
        
        if not temp_path or not Path(temp_path).exists():
            result["error"] = "No file received from download"
            if conn:
                update_download_status(conn, result)
            return result
        
        # Move to output directory
        file_name = download.suggested_filename or f"{ct_number}_trial.zip"
        output_path = pdf_dir / file_name
        
        # Remove existing file if present
        if output_path.exists():
            output_path.unlink()
        
        import shutil
        shutil.move(str(temp_path), str(output_path))
        
        # Success!
        result["success"] = True
        result["file_path"] = str(output_path)
        result["file_name"] = file_name
        result["file_size"] = output_path.stat().st_size
        result["downloaded_at"] = datetime.now(timezone.utc).isoformat()

        log(f"  ✓ {ct_number}: {file_name} ({result['file_size']:,} bytes)")

        # Auto-extract PDFs from ZIP
        if EXTRACT_PDFS and file_name.lower().endswith('.zip'):
            extracted = extract_pdfs_from_zip(output_path, pdf_dir, ct_number)
            result["extracted_pdfs"] = extracted

            # Optionally delete ZIP after extraction
            if DELETE_ZIP_AFTER_EXTRACT and extracted:
                try:
                    output_path.unlink()
                    log(f"  Deleted ZIP after extraction")
                except Exception as e:
                    log(f"  Could not delete ZIP: {e}", "WARN")
        
    except PlaywrightTimeout:
        result["error"] = "Timeout waiting for download"
        log(f"  ✗ {ct_number}: Download timeout", "WARN")
    except Exception as e:
        result["error"] = str(e)
        log(f"  ✗ {ct_number}: {e}", "WARN")
    finally:
        page.close()
    
    # Update database
    if conn:
        update_download_status(conn, result)
    
    return result


# ===================== Batch Download =====================

def download_trials_batch(
    ct_numbers: List[str],
    output_dir: Path,
    conn: Optional[sqlite3.Connection] = None,
    skip_existing: bool = True,
    check_updates: bool = True,
    delay: float = DELAY_BETWEEN_DOWNLOADS
) -> List[Dict]:
    """
    Download multiple trials with progress tracking.
    
    Args:
        ct_numbers: List of trial CT numbers
        output_dir: Base output directory
        conn: Database connection for tracking
        skip_existing: Skip already downloaded trials
        check_updates: Re-download if trial updated since last download
        delay: Seconds between downloads (rate limiting)
    
    Returns:
        List of download result dicts
    """
    if not _check_playwright():
        log("Cannot download documents: Playwright not installed", "ERROR")
        log("Install with: pip install playwright && playwright install chromium", "ERROR")
        return []
    
    results = []
    total = len(ct_numbers)
    
    log(f"\n=== Starting Document Download for {total} Trials ===")
    log(f"Output folder: {output_dir / PDF_FOLDER}")
    if check_updates:
        log("Update checking: ENABLED (will re-download updated trials)")
    
    success_count = 0
    skip_count = 0
    update_count = 0
    fail_count = 0
    
    try:
        for i, ct_number in enumerate(ct_numbers, 1):
            log(f"\nProcessing documents for {ct_number} ({i}/{total})")
            
            # Check if this is an update before downloading
            was_downloaded = conn and is_trial_downloaded(conn, ct_number)
            
            result = download_trial(ct_number, output_dir, conn, skip_existing, check_updates)
            results.append(result)
            
            if result["success"]:
                if result.get("error") == "skipped":
                    skip_count += 1
                elif was_downloaded:
                    update_count += 1  # Re-downloaded due to update
                else:
                    success_count += 1
            else:
                fail_count += 1
            
            # Delay between downloads (rate limiting)
            if i < total and result.get("error") != "skipped":
                time.sleep(delay)
        
    finally:
        _close_browser()
    
    # Summary
    log(f"\n{'='*60}")
    log(f"Download Complete:")
    log(f"  New downloads: {success_count}")
    log(f"  Updated:       {update_count}")
    log(f"  Skipped:       {skip_count}")
    log(f"  Failed:        {fail_count}")
    log(f"{'='*60}")
    
    return results


# ===================== Main Entry Point for ctis_run.py =====================

def process_trial_documents(
    ct_numbers: List[str],
    conn: sqlite3.Connection,
    session,  # Not used - kept for API compatibility with ctis_run.py
    output_dir: Path,
    check_updates: bool = True
) -> List[Dict]:
    """
    Process and download documents for trials.
    
    This is the main entry point called by ctis_run.py
    
    Args:
        ct_numbers: List of trial CT numbers to download
        conn: Database connection for tracking
        session: HTTP session (not used - we use browser automation instead)
        output_dir: Base output directory
        check_updates: Re-download if trial was updated since last download
    
    Returns:
        List of download results
    """
    if not ct_numbers:
        log("No trials to process for document download")
        return []
    
    # Ensure documents table exists
    create_documents_table(conn)
    
    # Download using browser automation
    results = download_trials_batch(
        ct_numbers=ct_numbers,
        output_dir=output_dir,
        conn=conn,
        skip_existing=True,
        check_updates=check_updates
    )
    
    return results


# ===================== Extract Existing ZIPs =====================

def extract_all_zips(zip_dir: Path, output_dir: Optional[Path] = None) -> Dict:
    """
    Extract PDFs from all ZIP files in a directory.

    Args:
        zip_dir: Directory containing ZIP files
        output_dir: Directory for extracted PDFs (defaults to same as zip_dir)

    Returns:
        Dict with extraction statistics
    """
    if output_dir is None:
        output_dir = zip_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_zips": 0,
        "successful": 0,
        "failed": 0,
        "total_pdfs": 0,
        "extracted_files": []
    }

    # Find all ZIP files
    zip_files = list(zip_dir.glob("*.zip"))
    stats["total_zips"] = len(zip_files)

    if not zip_files:
        log(f"No ZIP files found in {zip_dir}")
        return stats

    log(f"\n=== Extracting PDFs from {len(zip_files)} ZIP files ===")

    for zip_path in zip_files:
        # Try to extract CT number from filename (e.g., "2024-512203-40-00_trial.zip")
        ct_number = zip_path.stem.split("_")[0] if "_" in zip_path.stem else zip_path.stem

        log(f"\nProcessing: {zip_path.name}")
        extracted = extract_pdfs_from_zip(zip_path, output_dir, ct_number)

        if extracted:
            stats["successful"] += 1
            stats["total_pdfs"] += len(extracted)
            stats["extracted_files"].extend(extracted)
        else:
            stats["failed"] += 1

    log(f"\n{'='*60}")
    log(f"Extraction Complete:")
    log(f"  ZIP files processed: {stats['total_zips']}")
    log(f"  Successful:          {stats['successful']}")
    log(f"  Failed:              {stats['failed']}")
    log(f"  Total PDFs extracted: {stats['total_pdfs']}")
    log(f"{'='*60}")

    return stats


# ===================== Standalone Execution =====================

if __name__ == "__main__":
    # ========== CONFIGURATION (edit these) ==========

    # Directory containing downloaded ZIP files
    ZIP_DIR = Path("ctis-out/pdf")

    # Directory to extract PDFs to (same as ZIP_DIR if None)
    OUTPUT_DIR = None

    # Set to True to only extract existing ZIPs (no new downloads)
    EXTRACT_ONLY = True

    # ================================================

    if EXTRACT_ONLY:
        # Extract PDFs from existing ZIP files
        if not ZIP_DIR.exists():
            print(f"ERROR: Directory not found: {ZIP_DIR}")
            sys.exit(1)

        output = OUTPUT_DIR if OUTPUT_DIR else ZIP_DIR
        stats = extract_all_zips(ZIP_DIR, output)
        print(f"\nDone! Extracted {stats['total_pdfs']} PDFs from {stats['successful']} ZIP files.")
    else:
        # Download mode - requires CT numbers from database or file
        print("Download mode not configured. Set EXTRACT_ONLY = False and provide CT numbers.")
        sys.exit(1)