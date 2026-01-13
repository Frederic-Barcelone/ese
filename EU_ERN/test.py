#!/usr/bin/env python3
"""
Download PDFs from saved scraper queue
======================================
Downloads all PDFs found during scraping but not yet downloaded.
"""

import json
import requests
import os
import re
import time

# Configuration
RESULTS_FILE = "EU_ERN_DATA/ern_scrape_results.json"
OUTPUT_DIR = "EU_ERN_DATA/guidelines"
DELAY = 2.0  # Seconds between downloads

def sanitize_filename(url, prefix=""):
    """Convert URL to safe filename."""
    filename = url.split("/")[-1].split("?")[0]
    if not filename or not filename.endswith(".pdf"):
        # Generate from URL hash
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
        filename = f"document_{url_hash}.pdf"
    
    # Clean up
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    
    if prefix:
        filename = f"{prefix}_{filename}"
    
    return filename

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load results
    if not os.path.exists(RESULTS_FILE):
        print(f"[!] Results file not found: {RESULTS_FILE}")
        return
    
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Count totals
    total_pdfs = 0
    for network_id, network_data in data.get("networks", {}).items():
        pdfs = network_data.get("pdfs_found", [])
        total_pdfs += len(pdfs)
    
    print("=" * 60)
    print("PDF Downloader - From Saved Queue")
    print("=" * 60)
    print(f"Results file: {RESULTS_FILE}")
    print(f"Output dir:   {OUTPUT_DIR}")
    print(f"Total PDFs:   {total_pdfs}")
    print("=" * 60)
    
    # Download PDFs
    downloaded = 0
    skipped = 0
    failed = 0
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; ERN-Research-Bot/4.0)"
    })
    
    for network_id, network_data in data.get("networks", {}).items():
        pdfs = network_data.get("pdfs_found", [])
        
        if not pdfs:
            continue
        
        print(f"\n[{network_id}] {len(pdfs)} PDFs")
        
        for i, pdf in enumerate(pdfs, 1):
            url = pdf.get("url", "")
            if not url:
                continue
            
            filename = sanitize_filename(url, prefix=network_id)
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Skip if exists
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                skipped += 1
                continue
            
            print(f"  [{i}/{len(pdfs)}] {filename[:50]}...", end=" ", flush=True)
            
            try:
                response = session.get(url, timeout=60)
                response.raise_for_status()
                
                with open(filepath, "wb") as f:
                    f.write(response.content)
                
                size_kb = len(response.content) / 1024
                print(f"✓ ({size_kb:.1f} KB)")
                downloaded += 1
                
                time.sleep(DELAY)
                
            except KeyboardInterrupt:
                print("\n\n[!] Interrupted! Progress saved.")
                break
            except Exception as e:
                print(f"✗ ({str(e)[:30]})")
                failed += 1
        else:
            continue
        break  # Break outer loop on KeyboardInterrupt
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped:    {skipped} (already exist)")
    print(f"  Failed:     {failed}")
    print(f"  Output:     {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()