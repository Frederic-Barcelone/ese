#!/usr/bin/env python3
"""
Test script for CTIS PDF downloads - Version 3
Properly interacts with the CTIS Single Page Application.

The CTIS portal is a React SPA - we need to:
1. Go to search page
2. Search for the trial
3. Click on the trial result to open detail panel
4. Click "Download clinical trial" button

SETUP:
    pip install playwright requests
    playwright install chromium

USAGE:
    python test_pdf_download_v3.py
"""

import json
import sys
import time
from pathlib import Path

# ===================== Configuration =====================

TEST_TRIAL = "2024-512203-40-00"
OUTPUT_DIR = Path("ctis-out")
PDF_DIR = OUTPUT_DIR / "pdf"

CTIS_BASE = "https://euclinicaltrials.eu"
SEARCH_URL = f"{CTIS_BASE}/search-for-clinical-trials/?lang=en"

# Set to False to see the browser (helpful for debugging)
HEADLESS = True


# ===================== Main Download Function =====================

def download_trial_via_ui(ct_number: str, output_dir: Path) -> bool:
    """
    Download trial documents by interacting with the CTIS web UI.
    
    Steps:
    1. Go to search page
    2. Enter trial number in search box
    3. Click search
    4. Click on the trial result
    5. Wait for detail panel
    6. Click "Download clinical trial" button
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    except ImportError:
        print("ERROR: Playwright not installed!")
        print("Run: pip install playwright && playwright install chromium")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading trial: {ct_number}")
    print(f"Output: {output_dir}")
    print()
    
    with sync_playwright() as p:
        print("Starting browser...")
        browser = p.chromium.launch(headless=HEADLESS, slow_mo=100)
        
        context = browser.new_context(
            accept_downloads=True,
            viewport={'width': 1920, 'height': 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )
        
        page = context.new_page()
        
        try:
            # Step 1: Go to search page
            print("Step 1: Loading search page...")
            page.goto(SEARCH_URL, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(2000)
            
            # Handle cookie consent if present
            try:
                cookie_btn = page.locator('button:has-text("Accept")')
                if cookie_btn.count() > 0:
                    print("  Accepting cookies...")
                    cookie_btn.first.click()
                    page.wait_for_timeout(1000)
            except:
                pass
            
            # Take screenshot
            page.screenshot(path=str(output_dir / "step1_search_page.png"))
            print("  Screenshot: step1_search_page.png")
            
            # Step 2: Enter trial number in search
            print(f"Step 2: Searching for trial {ct_number}...")
            
            # Find search input - try multiple selectors
            search_selectors = [
                'input[placeholder*="Search"]',
                'input[type="search"]',
                'input[name="search"]',
                '#search-input',
                '.search-input',
                'input[aria-label*="search" i]',
                'input',  # fallback - first input
            ]
            
            search_input = None
            for selector in search_selectors:
                try:
                    elements = page.locator(selector)
                    if elements.count() > 0:
                        # Check if it's visible
                        first = elements.first
                        if first.is_visible():
                            search_input = first
                            print(f"  Found search input: {selector}")
                            break
                except:
                    continue
            
            if not search_input:
                print("  ERROR: Could not find search input!")
                # List all inputs for debugging
                print("  Available inputs:")
                inputs = page.locator("input").all()
                for i, inp in enumerate(inputs[:10]):
                    try:
                        placeholder = inp.get_attribute("placeholder") or ""
                        input_type = inp.get_attribute("type") or ""
                        print(f"    {i+1}. type={input_type}, placeholder={placeholder[:30]}")
                    except:
                        pass
                page.screenshot(path=str(output_dir / "error_no_search.png"))
                browser.close()
                return False
            
            # Type the trial number
            search_input.fill(ct_number)
            page.wait_for_timeout(500)
            
            # Press Enter or click search button
            search_input.press("Enter")
            page.wait_for_timeout(3000)
            
            page.screenshot(path=str(output_dir / "step2_search_results.png"))
            print("  Screenshot: step2_search_results.png")
            
            # Step 3: Click on the trial result
            print("Step 3: Clicking on trial result...")
            
            # Look for the trial in results
            trial_selectors = [
                f'text="{ct_number}"',
                f'[data-ct-number="{ct_number}"]',
                f'tr:has-text("{ct_number}")',
                f'div:has-text("{ct_number}")',
                '.trial-row',
                'table tbody tr',
            ]
            
            trial_clicked = False
            for selector in trial_selectors:
                try:
                    elements = page.locator(selector)
                    count = elements.count()
                    if count > 0:
                        print(f"  Found {count} element(s) with: {selector}")
                        elements.first.click()
                        trial_clicked = True
                        break
                except Exception as e:
                    continue
            
            if not trial_clicked:
                print("  Warning: Could not click trial result directly")
                print("  Trying to find any clickable trial row...")
                
                # Try clicking first row in any table
                try:
                    rows = page.locator("table tbody tr")
                    if rows.count() > 0:
                        rows.first.click()
                        trial_clicked = True
                        print("  Clicked first table row")
                except:
                    pass
            
            page.wait_for_timeout(3000)
            page.screenshot(path=str(output_dir / "step3_trial_detail.png"))
            print("  Screenshot: step3_trial_detail.png")
            
            # Step 4: Look for download button
            print("Step 4: Looking for download button...")
            
            download_selectors = [
                'button:has-text("Download clinical trial")',
                'a:has-text("Download clinical trial")',
                'button:has-text("Download")',
                '[aria-label*="Download"]',
                '.download-button',
                'button[title*="Download"]',
                'a[title*="Download"]',
            ]
            
            download_btn = None
            for selector in download_selectors:
                try:
                    elements = page.locator(selector)
                    if elements.count() > 0:
                        first = elements.first
                        if first.is_visible():
                            download_btn = first
                            print(f"  Found download button: {selector}")
                            break
                except:
                    continue
            
            if not download_btn:
                print("  Could not find download button!")
                print("  All visible buttons:")
                buttons = page.locator("button:visible").all()
                for i, btn in enumerate(buttons[:15]):
                    try:
                        text = btn.text_content().strip()[:50]
                        if text:
                            print(f"    {i+1}. {text}")
                    except:
                        pass
                
                print("\n  All visible links:")
                links = page.locator("a:visible").all()
                for i, link in enumerate(links[:15]):
                    try:
                        text = link.text_content().strip()[:50]
                        if text:
                            print(f"    {i+1}. {text}")
                    except:
                        pass
                
                page.screenshot(path=str(output_dir / "error_no_download_btn.png"))
                browser.close()
                return False
            
            # Step 5: Click download and wait for file
            print("Step 5: Clicking download button...")
            
            try:
                with page.expect_download(timeout=120000) as download_info:
                    download_btn.click()
                    print("  Waiting for download...")
                
                download = download_info.value
                suggested_name = download.suggested_filename or f"{ct_number}_trial.zip"
                temp_path = download.path()
                
                if temp_path and Path(temp_path).exists():
                    output_path = output_dir / suggested_name
                    
                    import shutil
                    shutil.move(str(temp_path), str(output_path))
                    
                    size = output_path.stat().st_size
                    print(f"\n✓ SUCCESS!")
                    print(f"  File: {output_path.name}")
                    print(f"  Size: {size:,} bytes")
                    
                    browser.close()
                    return True
                    
            except PlaywrightTimeout:
                print("  Download timeout!")
                page.screenshot(path=str(output_dir / "error_download_timeout.png"))
            except Exception as e:
                print(f"  Download error: {e}")
                page.screenshot(path=str(output_dir / "error_download.png"))
            
            browser.close()
            return False
            
        except Exception as e:
            print(f"\nERROR: {e}")
            try:
                page.screenshot(path=str(output_dir / "error_exception.png"))
            except:
                pass
            browser.close()
            return False


# ===================== Alternative: Use the trial detail page directly =====================

def download_trial_direct(ct_number: str, output_dir: Path) -> bool:
    """
    Try to access trial detail page directly and download.
    Some trials might have a direct URL format.
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    except ImportError:
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try different URL patterns
    url_patterns = [
        f"{CTIS_BASE}/search-for-clinical-trials/?lang=en#clinicalTrial/{ct_number}",
        f"{CTIS_BASE}/search-for-clinical-trials/?lang=en&number={ct_number}",
        f"{CTIS_BASE}/ctis-public/view/{ct_number}",
    ]
    
    print(f"\nTrying alternative URL patterns...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        
        for url in url_patterns:
            print(f"  Trying: {url[:70]}...")
            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
                page.wait_for_timeout(2000)
                
                # Check if we landed on a detail page (has download button)
                download_btn = page.locator('button:has-text("Download"), a:has-text("Download")')
                if download_btn.count() > 0:
                    print(f"  Found download button!")
                    page.screenshot(path=str(output_dir / "alt_found_download.png"))
                    
                    # Try to download
                    try:
                        with page.expect_download(timeout=60000) as download_info:
                            download_btn.first.click()
                        
                        download = download_info.value
                        temp_path = download.path()
                        if temp_path and Path(temp_path).exists():
                            output_path = output_dir / (download.suggested_filename or f"{ct_number}.zip")
                            import shutil
                            shutil.move(str(temp_path), str(output_path))
                            print(f"  Downloaded: {output_path.name}")
                            browser.close()
                            return True
                    except:
                        pass
                        
            except Exception as e:
                continue
        
        browser.close()
        return False


# ===================== Main =====================

def main():
    print("=" * 60)
    print("CTIS PDF Download Test v3")
    print("Interacts with CTIS SPA interface")
    print("=" * 60)
    
    # Check Playwright
    try:
        from playwright.sync_api import sync_playwright
        print("✓ Playwright is installed")
    except ImportError:
        print("✗ Playwright not installed!")
        print("  Run: pip install playwright && playwright install chromium")
        sys.exit(1)
    
    if not HEADLESS:
        print("⚠ Running in visible mode (HEADLESS=False)")
    
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {PDF_DIR}")
    print(f"Trial: {TEST_TRIAL}")
    print()
    
    # Try main method: interact with UI
    print("=" * 60)
    print("METHOD 1: Full UI interaction")
    print("=" * 60)
    
    success = download_trial_via_ui(TEST_TRIAL, PDF_DIR)
    
    if success:
        print("\n✓ Download successful!")
    else:
        print("\n✗ UI method failed")
        
        # Try alternative methods
        print("\n" + "=" * 60)
        print("METHOD 2: Alternative URL patterns")
        print("=" * 60)
        
        success = download_trial_direct(TEST_TRIAL, PDF_DIR)
        
        if success:
            print("\n✓ Alternative method worked!")
        else:
            print("\n✗ All methods failed")
            print("\nCheck the screenshots in ctis-out/pdf/ to see what happened.")
            print("You may need to manually inspect the CTIS website to understand the UI.")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Check: {PDF_DIR}")


if __name__ == "__main__":
    main()