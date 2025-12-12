#!/usr/bin/env python3
"""
ERN (European Reference Networks) Resource Scraper v3.2
========================================================
CONFIG-DRIVEN VERSION WITH SMART STATE MANAGEMENT

This version reads network configuration from ern_config.json.
Only networks with "scrape": true will be processed.

NEW IN v3.2:
- SMART STATE MANAGEMENT: Automatically detects when crawl settings change
- When you increase max_crawl_depth or max_pages_per_network, networks that
  previously hit those limits will be automatically re-scraped
- Per-network crawl statistics tracking (pages scraped, max depth, hit limits)
- No need to manually delete state file when changing settings!

To change which networks are scraped:
1. Edit ern_config.json
2. Set "scrape": true for networks you want to scrape
3. Set "scrape": false for networks you want to skip
4. Run this script

Current networks enabled for scraping (in default config):
1. ERN-EuroBloodNet - Rare Haematological Diseases
2. ERKNet - Rare Kidney Diseases  
3. ERN-RITA - Immunodeficiency, Autoinflammatory and Autoimmune Diseases
4. ERN-EURO-NMD - Neuromuscular Diseases
5. MetabERN - Hereditary Metabolic Disorders
6. ERN-GUARD-HEART - Rare and Complex Diseases of the Heart
7. ERN-BOND - Rare Bone Diseases

Features:
- CONFIG-DRIVEN: Edit ern_config.json to enable/disable networks
- SMART RESCRAPE: Detects settings changes and re-scrapes as needed
- Extracts clean text content from web pages
- Outputs .md files with YAML frontmatter metadata
- Removes navigation, scripts, styles, cookie banners
- Preserves meaningful content: paragraphs, headings, lists
- Includes PDF links in extracted content
- RAG-ready format with source URL and title
- CONFIGURABLE CRAWL DEPTH via config file
- Domain-specific rate limiting
- Retry logic with exponential backoff
- Improved error handling and recovery

Usage:
    python ern_scraper.py                    # Use default config file
    python ern_scraper.py -c my_config.json  # Use custom config file
    python ern_scraper.py --list             # List all networks and their status
    python ern_scraper.py --list-enabled     # List only enabled networks

Author: Generated for ERN research
Date: 2025
"""

import os
import re
import json
import time
import hashlib
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from datetime import datetime
import logging
from collections import defaultdict
import random

# ============================================================================
# DEFAULT CONFIGURATION (used if config file not found)
# ============================================================================

DEFAULT_CONFIG_FILE = "ern_config.json"

DEFAULT_DOMAIN_DELAYS = {
    "health.ec.europa.eu": 8.0,
    "ec.europa.eu": 8.0,
    "ern-net.eu": 2.0,
    "default": 1.5
}

DEFAULT_SCRAPER_SETTINGS = {
    "max_crawl_depth": 10,
    "max_pages_per_network": 100,
    "default_request_delay": 1.5,
    "request_timeout": 45,
    "max_retries": 3,
    "backoff_base": 5.0,
    "skip_existing": True,
    "verbose": True,
    "output_directory": "EU_ERN_DATA"
}


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(config_path=None):
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file. If None, uses DEFAULT_CONFIG_FILE.
        
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_FILE
    
    # Try to find config file in current directory or script directory
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            print(f"[WARNING] Config file not found: {config_path}")
            print("[WARNING] Please create ern_config.json or specify a config file with -c")
            return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"[OK] Loaded configuration from: {config_path}")
        return config
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in config file: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Could not load config file: {e}")
        return None


def get_networks_to_scrape(config):
    """
    Get list of networks that have scrape=true in config.
    
    Returns:
        dict: Dictionary of network_id -> network_info for enabled networks
    """
    if config is None:
        return {}
    
    networks = config.get("networks", {})
    enabled = {}
    
    for network_id, network_info in networks.items():
        if network_info.get("scrape", False):
            enabled[network_id] = network_info
    
    return enabled


def get_ec_handbooks(config):
    """
    Get EC methodological handbooks from config.
    
    Returns:
        dict: Dictionary of handbook_id -> handbook_info
    """
    if config is None:
        return {}
    
    handbooks_config = config.get("ec_methodological_handbooks", {})
    if not handbooks_config.get("download_handbooks", True):
        return {}
    
    return handbooks_config.get("handbooks", {})


def list_networks(config, enabled_only=False):
    """
    Print a formatted list of all networks with their scrape status.
    
    Args:
        config: Configuration dictionary
        enabled_only: If True, only show enabled networks
    """
    if config is None:
        print("[ERROR] No configuration loaded")
        return
    
    networks = config.get("networks", {})
    
    print("\n" + "=" * 80)
    if enabled_only:
        print("ENABLED ERN NETWORKS (will be scraped)")
    else:
        print("ALL ERN NETWORKS - SCRAPE STATUS")
    print("=" * 80 + "\n")
    
    enabled_count = 0
    disabled_count = 0
    
    for network_id, info in sorted(networks.items()):
        scrape = info.get("scrape", False)
        
        if enabled_only and not scrape:
            continue
        
        if scrape:
            enabled_count += 1
            status = "[X] ENABLED "
        else:
            disabled_count += 1
            status = "[ ] disabled"
        
        print(f"{status}  {network_id}")
        print(f"             {info.get('name', 'N/A')}")
        print(f"             Disease Area: {info.get('short_description', info.get('disease_area', 'N/A'))}")
        print(f"             Website: {info.get('website', 'N/A')}")
        print()
    
    print("-" * 80)
    print(f"Total: {len(networks)} networks | Enabled: {enabled_count} | Disabled: {disabled_count}")
    print("-" * 80)
    print("\nTo change which networks are scraped, edit ern_config.json")
    print("and set 'scrape': true or 'scrape': false for each network.\n")


# ============================================================================
# ERN SCRAPER CLASS
# ============================================================================

class ERNScraper:
    """Main scraper class for collecting ERN resources with improved rate limiting."""
    
    def __init__(self, config=None, output_dir=None, request_delay=None, 
                 timeout=None, skip_existing=None, verbose=None):
        """
        Initialize the ERN scraper.
        
        Args:
            config: Configuration dictionary (loaded from JSON)
            output_dir: Directory to save downloaded files (overrides config)
            request_delay: Default seconds to wait between requests (overrides config)
            timeout: Request timeout in seconds (overrides config)
            skip_existing: If True, skip files that already exist (overrides config)
            verbose: If True, print detailed progress (overrides config)
        """
        self.config = config or {}
        settings = self.config.get("scraper_settings", DEFAULT_SCRAPER_SETTINGS)
        
        # Apply settings with overrides
        self.output_dir = output_dir or settings.get("output_directory", "EU_ERN_DATA")
        self.default_delay = request_delay or settings.get("default_request_delay", 1.5)
        self.timeout = timeout or settings.get("request_timeout", 45)
        self.skip_existing = skip_existing if skip_existing is not None else settings.get("skip_existing", True)
        self.verbose = verbose if verbose is not None else settings.get("verbose", True)
        
        # Get rate limiting settings
        self.max_retries = settings.get("max_retries", 3)
        self.backoff_base = settings.get("backoff_base", 5.0)
        self.max_crawl_depth = settings.get("max_crawl_depth", 10)
        self.max_pages_per_network = settings.get("max_pages_per_network", 100)
        
        # Domain delays from config or defaults
        self.domain_delays = self.config.get("domain_delays", DEFAULT_DOMAIN_DELAYS)
        
        # Track requests per domain for rate limiting
        self.domain_last_request = defaultdict(float)
        self.domain_request_count = defaultdict(int)
        
        # Create output directories
        self.dirs = {
            'root': self.output_dir,
            'guidelines': os.path.join(self.output_dir, 'guidelines'),
            'methodologies': os.path.join(self.output_dir, 'methodologies'),
            'factsheets': os.path.join(self.output_dir, 'factsheets'),
            'pages': os.path.join(self.output_dir, 'rag_content')
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
        
        # Setup logging
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'ern_scraper.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup session with retry adapter
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ERN-Research-Bot/3.1 (Academic Research; +https://example.org/research)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # State file for checkpointing
        self.state_file = os.path.join(self.output_dir, 'scraper_state.json')
        self.state = self._load_state()
        
        # Check if settings have changed and invalidate networks if needed
        self._check_settings_changed()
        
        # Get enabled networks from config
        self.networks_to_scrape = get_networks_to_scrape(config)
        self.ec_handbooks = get_ec_handbooks(config)
        
        # Results tracking
        self.results = {
            "metadata": {
                "scrape_date": datetime.now().isoformat(),
                "scraper_version": "3.2",
                "config_file": DEFAULT_CONFIG_FILE,
                "total_networks_enabled": len(self.networks_to_scrape),
                "output_dir": self.output_dir,
                "settings": {
                    "max_crawl_depth": self.max_crawl_depth,
                    "max_pages_per_network": self.max_pages_per_network
                }
            },
            "networks": {},
            "methodologies": [],
            "downloaded_files": [],
            "skipped_files": [],
            "errors": []
        }
    
    def _get_domain(self, url):
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc
    
    def _extract_clean_content(self, html, url="", title=""):
        """
        Extract clean, RAG-ready content from HTML.
        Returns markdown-formatted text with metadata.
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Get page title if not provided
        if not title:
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)
        
        # Get meta description
        meta_desc = ""
        meta = soup.find('meta', attrs={'name': 'description'})
        if meta:
            meta_desc = meta.get('content', '')
        
        # Remove non-content elements
        for tag in soup(['script', 'style', 'noscript', 'svg', 'iframe', 'nav', 
                         'header', 'footer', 'aside', 'form']):
            tag.decompose()
        
        # Remove elements with noise classes/IDs
        noise_patterns = ['cookie', 'gdpr', 'consent', 'banner', 'popup', 'modal',
                         'menu', 'nav', 'sidebar', 'social', 'share', 'comment',
                         'advertisement', 'newsletter', 'subscribe', 'widget', 
                         'breadcrumb', 'footer', 'header']
        for pattern in noise_patterns:
            for el in soup.find_all(class_=re.compile(pattern, re.I)):
                el.decompose()
            for el in soup.find_all(id=re.compile(pattern, re.I)):
                el.decompose()
        
        content_parts = []
        
        # Extract paragraphs
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if text and len(text) > 30:
                content_parts.append(text)
        
        # Extract headings with context
        for h in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            text = h.get_text(strip=True)
            if text and len(text) > 5:
                level = int(h.name[1])
                content_parts.append(f"{'#' * level} {text}")
        
        # Extract list items (not from menus)
        for ul in soup.find_all(['ul', 'ol']):
            parent_class = ' '.join(ul.get('class', []))
            if any(x in parent_class.lower() for x in ['menu', 'nav']):
                continue
            for li in ul.find_all('li', recursive=False):
                text = li.get_text(strip=True)
                if len(text) > 40:
                    content_parts.append(f"- {text[:300]}")
        
        # Extract PDF links
        pdf_links = []
        for a in soup.find_all('a', href=True):
            href = a.get('href', '')
            text = a.get_text(strip=True)
            if '.pdf' in href.lower() and text:
                pdf_links.append(f"- [PDF] {text}: {href}")
        
        # Deduplicate content
        seen = set()
        unique_parts = []
        for part in content_parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)
        
        # Build markdown document
        md_parts = [
            "---",
            f"url: {url}",
            f"title: {title}",
            f"scraped_date: {datetime.now().isoformat()}",
            "---",
            "",
            f"# {title}" if title else "",
            "",
            f"> {meta_desc}" if meta_desc else "",
            "",
            '\n\n'.join(unique_parts)
        ]
        
        if pdf_links:
            md_parts.append("\n\n## Related Documents\n")
            md_parts.append('\n'.join(pdf_links[:30]))
        
        return '\n'.join(md_parts)
    
    def _get_delay_for_domain(self, domain):
        """Get the appropriate delay for a domain."""
        for pattern, delay in self.domain_delays.items():
            if pattern in domain:
                return delay
        return self.domain_delays.get("default", self.default_delay)
    
    def _wait_for_rate_limit(self, url):
        """Wait appropriate time based on domain rate limiting."""
        domain = self._get_domain(url)
        delay = self._get_delay_for_domain(domain)
        
        last_request = self.domain_last_request[domain]
        elapsed = time.time() - last_request
        
        if elapsed < delay:
            wait_time = delay - elapsed
            # Add small random jitter to avoid patterns
            wait_time += random.uniform(0.1, 0.5)
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for {domain}")
            time.sleep(wait_time)
        
        self.domain_last_request[domain] = time.time()
        self.domain_request_count[domain] += 1
    
    def _load_state(self):
        """Load previous scraper state if exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.logger.info(f"Loaded previous state: {len(state.get('downloaded_urls', []))} URLs already processed")
                return state
            except Exception as e:
                self.logger.warning(f"Could not load state file: {e}")
        return {
            "downloaded_urls": [],
            "failed_urls": [],
            "scraped_networks": [],
            "network_stats": {},
            "settings": {},
            "last_run": None
        }
    
    def _check_settings_changed(self):
        """
        Check if crawl settings have changed since last run.
        If settings have increased, invalidate affected networks so they get re-scraped.
        Also handles legacy state files that don't have network_stats.
        """
        previous_settings = self.state.get("settings", {})
        prev_depth = previous_settings.get("max_crawl_depth", 0)
        prev_pages = previous_settings.get("max_pages_per_network", 0)
        network_stats = self.state.get("network_stats", {})
        scraped_networks = self.state.get("scraped_networks", [])
        
        # Check if this is a legacy state file (has scraped networks but no stats)
        is_legacy_state = len(scraped_networks) > 0 and len(network_stats) == 0
        
        if is_legacy_state:
            self.logger.info("[LEGACY STATE] Detected old state file without network statistics")
            self.logger.info(f"[LEGACY STATE] Settings increased or unknown - will re-scrape all {len(scraped_networks)} networks")
            self.logger.info("[LEGACY STATE] Networks to re-scrape:")
            for network_id in scraped_networks:
                self.logger.info(f"  -> {network_id}")
            
            # Clear the scraped networks list to force re-scraping
            self.state["scraped_networks"] = []
            self.state["network_stats"] = {}
        
        else:
            # Normal settings change detection
            settings_increased = False
            
            if self.max_crawl_depth > prev_depth and prev_depth > 0:
                self.logger.info(f"[SETTINGS CHANGE] Crawl depth increased: {prev_depth} -> {self.max_crawl_depth}")
                settings_increased = True
            
            if self.max_pages_per_network > prev_pages and prev_pages > 0:
                self.logger.info(f"[SETTINGS CHANGE] Max pages increased: {prev_pages} -> {self.max_pages_per_network}")
                settings_increased = True
            
            if settings_increased:
                self.logger.info("Checking which networks need re-scraping due to increased limits...")
                
                # Check each "completed" network to see if it hit the old limits
                networks_to_rescrape = []
                
                for network_id in scraped_networks:
                    stats = network_stats.get(network_id, {})
                    pages_scraped = stats.get("pages_scraped", 0)
                    max_depth_reached = stats.get("max_depth_reached", 0)
                    hit_page_limit = stats.get("hit_page_limit", False)
                    hit_depth_limit = stats.get("hit_depth_limit", False)
                    
                    should_rescrape = False
                    reasons = []
                    
                    # If no stats exist for this network, re-scrape it
                    if not stats:
                        should_rescrape = True
                        reasons.append("no stats available (legacy)")
                    
                    # If it hit the page limit and we've increased the limit
                    if hit_page_limit and self.max_pages_per_network > prev_pages:
                        should_rescrape = True
                        reasons.append(f"hit page limit ({pages_scraped}/{prev_pages})")
                    
                    # If it reached max depth and we've increased depth
                    if hit_depth_limit and self.max_crawl_depth > prev_depth:
                        should_rescrape = True
                        reasons.append(f"hit depth limit ({max_depth_reached}/{prev_depth})")
                    
                    # If pages scraped was at or near the old limit (within 90%)
                    if prev_pages > 0 and pages_scraped >= prev_pages * 0.9 and self.max_pages_per_network > prev_pages:
                        should_rescrape = True
                        reasons.append(f"near page limit ({pages_scraped}/{prev_pages})")
                    
                    if should_rescrape:
                        networks_to_rescrape.append(network_id)
                        self.logger.info(f"  -> {network_id} will be RE-SCRAPED: {', '.join(reasons)}")
                    else:
                        self.logger.info(f"  -> {network_id} OK (didn't hit limits)")
                
                # Remove networks that need re-scraping from the "completed" list
                if networks_to_rescrape:
                    self.state["scraped_networks"] = [
                        n for n in scraped_networks 
                        if n not in networks_to_rescrape
                    ]
                    self.logger.info(f"Marked {len(networks_to_rescrape)} network(s) for re-scraping")
                else:
                    self.logger.info("No networks need re-scraping (none hit previous limits)")
        
        # Update stored settings for next run
        self.state["settings"] = {
            "max_crawl_depth": self.max_crawl_depth,
            "max_pages_per_network": self.max_pages_per_network
        }
        self._save_state()
    
    def _save_state(self):
        """Save current scraper state for checkpointing."""
        self.state["last_run"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _url_to_filename(self, url, prefix=""):
        """Convert URL to safe filename."""
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        
        if not filename or filename == "" or filename.endswith('_en'):
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            filename = f"document_{url_hash}"
        
        # Clean up filename
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        filename = re.sub(r'_+', '_', filename)  # Remove multiple underscores
        
        if prefix:
            filename = f"{prefix}_{filename}"
        
        if not os.path.splitext(filename)[1]:
            filename += ".pdf"
        
        return filename
    
    def _file_exists(self, filepath):
        """Check if file exists and has content."""
        return os.path.exists(filepath) and os.path.getsize(filepath) > 0
    
    def _safe_request(self, url, retry_count=0):
        """
        Make a safe HTTP request with retry logic and exponential backoff.
        
        Args:
            url: URL to request
            retry_count: Current retry attempt number
            
        Returns:
            Response object or None if failed
        """
        # Wait for rate limit before making request
        self._wait_for_rate_limit(url)
        
        try:
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            
            # Handle rate limiting (429)
            if response.status_code == 429:
                if retry_count < self.max_retries:
                    # Exponential backoff with jitter
                    wait_time = self.backoff_base * (2 ** retry_count) + random.uniform(1, 3)
                    self.logger.warning(f"Rate limited (429) for {url}. Waiting {wait_time:.1f}s before retry {retry_count + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                    return self._safe_request(url, retry_count + 1)
                else:
                    self.logger.error(f"Max retries exceeded for {url} after rate limiting")
                    self.results["errors"].append({
                        "url": url,
                        "error": "429 Too Many Requests - max retries exceeded",
                        "timestamp": datetime.now().isoformat()
                    })
                    return None
            
            # Handle server errors (5xx)
            if response.status_code >= 500:
                if retry_count < self.max_retries:
                    wait_time = self.backoff_base * (2 ** retry_count)
                    self.logger.warning(f"Server error ({response.status_code}) for {url}. Waiting {wait_time:.1f}s before retry")
                    time.sleep(wait_time)
                    return self._safe_request(url, retry_count + 1)
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout:
            if retry_count < self.max_retries:
                self.logger.warning(f"Timeout for {url}. Retrying ({retry_count + 1}/{self.max_retries})")
                time.sleep(self.backoff_base)
                return self._safe_request(url, retry_count + 1)
            self.logger.error(f"Request timeout for {url} after {self.max_retries} retries")
            self.results["errors"].append({
                "url": url,
                "error": "Timeout after max retries",
                "timestamp": datetime.now().isoformat()
            })
            return None
            
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            # Don't log 404s as errors for guideline paths (they're expected)
            if "404" in error_msg:
                self.logger.debug(f"Page not found: {url}")
            else:
                self.logger.error(f"Request failed for {url}: {e}")
            self.results["errors"].append({
                "url": url,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            return None
    
    def _download_file(self, url, filename, subdir="guidelines"):
        """
        Download a file to the output directory.
        
        Returns:
            tuple: (filepath, status) where status is 'downloaded', 'skipped', or 'failed'
        """
        filepath = os.path.join(self.dirs[subdir], filename)
        
        # Check if already downloaded
        if self.skip_existing:
            if self._file_exists(filepath):
                self.logger.debug(f"Skipping existing file: {filename}")
                self.results["skipped_files"].append({
                    "url": url,
                    "filepath": filepath,
                    "reason": "file_exists"
                })
                return filepath, "skipped"
            
            if url in self.state["downloaded_urls"]:
                self.logger.debug(f"Skipping previously processed URL: {url}")
                self.results["skipped_files"].append({
                    "url": url,
                    "filepath": filepath,
                    "reason": "url_in_state"
                })
                return filepath, "skipped"
            
            # Skip URLs that have failed before
            if url in self.state.get("failed_urls", []):
                self.logger.debug(f"Skipping previously failed URL: {url}")
                return None, "skipped"
        
        # Download file
        response = self._safe_request(url)
        if response:
            # Verify we got actual content
            content_type = response.headers.get('Content-Type', '')
            if len(response.content) < 100 and 'pdf' in filename.lower():
                self.logger.warning(f"Suspiciously small PDF ({len(response.content)} bytes): {url}")
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"Downloaded: {filename} ({len(response.content):,} bytes)")
            self.state["downloaded_urls"].append(url)
            self.results["downloaded_files"].append({
                "url": url,
                "filepath": filepath,
                "size": len(response.content),
                "content_type": content_type,
                "timestamp": datetime.now().isoformat()
            })
            self._save_state()
            
            return filepath, "downloaded"
        
        # Track failed URLs
        if url not in self.state.get("failed_urls", []):
            self.state.setdefault("failed_urls", []).append(url)
            self._save_state()
        
        return None, "failed"
    
    def _download_with_fallback(self, urls, filename, subdir="methodologies"):
        """
        Try to download from multiple URLs, using fallback if primary fails.
        
        Args:
            urls: List of URLs to try in order
            filename: Destination filename
            subdir: Subdirectory for the file
            
        Returns:
            tuple: (filepath, status, successful_url)
        """
        for url in urls:
            filepath, status = self._download_file(url, filename, subdir)
            if status in ["downloaded", "skipped"]:
                return filepath, status, url
        
        return None, "failed", None
    
    def _extract_links(self, soup, base_url, patterns=None):
        """Extract relevant links from a page."""
        links = []
        if patterns is None:
            patterns = [
                r'guideline', r'cpg', r'cdst', r'protocol', r'pathway',
                r'consensus', r'recommendation', r'\.pdf$', r'clinical',
                r'diagnostic', r'therapeutic', r'flowchart', r'algorithm',
                r'handbook', r'manual', r'standard', r'care[\-_]?path'
            ]
        
        for a in soup.find_all('a', href=True):
            href = a.get('href', '')
            text = a.get_text(strip=True).lower()
            
            # Skip non-navigable links
            if href.startswith(('mailto:', 'javascript:', '#', 'tel:', 'data:')):
                continue
            
            # Build full URL
            full_url = urljoin(base_url, href)
            
            # Check if matches any pattern
            for pattern in patterns:
                if re.search(pattern, href.lower()) or re.search(pattern, text):
                    links.append({
                        "url": full_url,
                        "text": a.get_text(strip=True),
                        "pattern_matched": pattern
                    })
                    break
        
        # Deduplicate
        seen = set()
        unique_links = []
        for link in links:
            if link["url"] not in seen:
                seen.add(link["url"])
                unique_links.append(link)
        
        return unique_links
    
    def _discover_additional_pages(self, soup, base_url, network_id):
        """
        Discover additional pages that might contain guidelines.
        Look for navigation links, sitemap references, etc.
        """
        additional_paths = set()
        
        # Look for nav links
        nav_patterns = [
            r'guideline', r'resource', r'publication', r'document',
            r'clinical', r'pathway', r'protocol', r'download',
            r'library', r'knowledge', r'education'
        ]
        
        for a in soup.find_all('a', href=True):
            href = a.get('href', '').lower()
            text = a.get_text(strip=True).lower()
            
            for pattern in nav_patterns:
                if re.search(pattern, href) or re.search(pattern, text):
                    parsed = urlparse(urljoin(base_url, a.get('href')))
                    # Only add paths from same domain
                    if self._get_domain(base_url) in parsed.netloc:
                        path = parsed.path
                        if path and path not in ['/', '']:
                            additional_paths.add(path)
                    break
        
        return list(additional_paths)[:5]  # Limit to 5 additional paths
    
    def scrape_network(self, network_id, network_info):
        """Scrape a single ERN network for guidelines and resources."""
        self.logger.info("=" * 60)
        self.logger.info(f"Scraping {network_id}: {network_info.get('name', 'Unknown')}")
        self.logger.info("=" * 60)
        
        # Check if already fully scraped
        if self.skip_existing and network_id in self.state.get("scraped_networks", []):
            self.logger.info(f"Skipping {network_id} - already scraped in previous run")
            return self.results["networks"].get(network_id, {})
        
        # Track crawl statistics
        crawl_stats = {
            "max_depth_reached": 0,
            "hit_page_limit": False,
            "hit_depth_limit": False
        }
        
        network_data = {
            "id": network_id,
            "info": {
                "name": network_info.get("name"),
                "website": network_info.get("website"),
                "disease_area": network_info.get("disease_area"),
                "description": network_info.get("full_description", network_info.get("description", ""))
            },
            "pages_scraped": [],
            "pages_failed": [],
            "guidelines_found": [],
            "pdfs_found": [],
            "pdfs_downloaded": [],
            "crawl_stats": crawl_stats
        }
        
        website = network_info.get("website")
        if not website:
            self.logger.error(f"No website configured for {network_id}")
            return network_data
        
        all_links = []
        discovered_paths = set()
        
        # Fetch main page
        self.logger.info(f"Fetching main page: {website}")
        response = self._safe_request(website)
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            links = self._extract_links(soup, website)
            all_links.extend(links)
            network_data["pages_scraped"].append(website)
            
            # Save page as clean markdown (RAG-ready)
            page_file = os.path.join(self.dirs['pages'], f"{network_id}_main.md")
            clean_content = self._extract_clean_content(
                response.text, 
                url=website,
                title=f"{network_id} - Main Page"
            )
            with open(page_file, 'w', encoding='utf-8') as f:
                f.write(clean_content)
            
            # Discover additional pages
            discovered = self._discover_additional_pages(soup, website, network_id)
            discovered_paths.update(discovered)
        else:
            network_data["pages_failed"].append(website)
        
        # Crawl pages up to max_crawl_depth levels
        all_paths = list(network_info.get("guidelines_paths", [])) + list(discovered_paths)
        seen_urls = set()
        
        # Queue: (url, path, depth)
        crawl_queue = [(urljoin(website, p), p, 1) for p in all_paths]
        
        while crawl_queue:
            # Stop if we've crawled enough pages
            if len(network_data["pages_scraped"]) >= self.max_pages_per_network:
                self.logger.info(f"Reached max pages limit ({self.max_pages_per_network}), stopping crawl")
                crawl_stats["hit_page_limit"] = True
                break
            
            current_url, current_path, depth = crawl_queue.pop(0)
            
            # Track max depth reached
            if depth > crawl_stats["max_depth_reached"]:
                crawl_stats["max_depth_reached"] = depth
            
            # Check if we've hit depth limit
            if depth > self.max_crawl_depth:
                crawl_stats["hit_depth_limit"] = True
                continue
            
            if current_url in seen_urls:
                continue
            seen_urls.add(current_url)
            
            self.logger.info(f"Fetching [L{depth}]: {current_url}")
            
            response = self._safe_request(current_url)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                page_links = self._extract_links(soup, current_url)
                all_links.extend(page_links)
                network_data["pages_scraped"].append(current_url)
                
                # Save page as clean markdown (RAG-ready)
                safe_path = re.sub(r'[^\w\-]', '_', current_path).strip('_')[:50] or 'page'
                page_file = os.path.join(self.dirs['pages'], f"{network_id}_{safe_path}_{depth}.md")
                clean_content = self._extract_clean_content(
                    response.text,
                    url=current_url,
                    title=f"{network_id} - {safe_path}"
                )
                with open(page_file, 'w', encoding='utf-8') as f:
                    f.write(clean_content)
                
                # Find sub-pages if we haven't reached max depth
                if depth < self.max_crawl_depth:
                    for a in soup.find_all('a', href=True):
                        href = a.get('href', '')
                        
                        # Skip non-page links
                        if href.startswith(('mailto:', 'javascript:', '#', 'tel:')):
                            continue
                        
                        # Build full URL
                        sub_url = urljoin(current_url, href)
                        sub_parsed = urlparse(sub_url)
                        
                        # Only follow internal links (same domain)
                        if website.replace('https://', '').replace('http://', '').rstrip('/') not in sub_parsed.netloc:
                            continue
                        
                        sub_path = sub_parsed.path
                        
                        # Skip already seen, PDFs (will be downloaded separately), external
                        if sub_url in seen_urls:
                            continue
                        if sub_path.lower().endswith('.pdf'):
                            continue
                        
                        # Only follow if it's a sub-path or related path
                        if sub_path and sub_url not in [q[0] for q in crawl_queue]:
                            crawl_queue.append((sub_url, sub_path, depth + 1))
            else:
                network_data["pages_failed"].append(current_url)
        
        self.logger.info(f"Crawled {len(network_data['pages_scraped'])} pages total (max depth reached: {crawl_stats['max_depth_reached']})")
        if crawl_stats["hit_page_limit"]:
            self.logger.info(f"  -> Hit page limit ({self.max_pages_per_network})")
        if crawl_stats["hit_depth_limit"]:
            self.logger.info(f"  -> Hit depth limit ({self.max_crawl_depth})")
        
        # Categorize found links
        seen_urls = set()
        for link in all_links:
            if link["url"] not in seen_urls:
                seen_urls.add(link["url"])
                if link["url"].lower().endswith('.pdf'):
                    network_data["pdfs_found"].append(link)
                else:
                    network_data["guidelines_found"].append(link)
        
        self.logger.info(f"Found {len(network_data['pdfs_found'])} PDF links, "
                        f"{len(network_data['guidelines_found'])} other guideline links")
        
        # Download PDFs
        for pdf_link in network_data["pdfs_found"]:
            filename = self._url_to_filename(pdf_link["url"], prefix=network_id)
            filepath, status = self._download_file(pdf_link["url"], filename, "guidelines")
            if status in ["downloaded", "skipped"]:
                network_data["pdfs_downloaded"].append({
                    "url": pdf_link["url"],
                    "filepath": filepath,
                    "status": status,
                    "link_text": pdf_link.get("text", "")
                })
        
        # Download factsheet
        factsheet = network_info.get("factsheet")
        if factsheet:
            filename = f"{network_id}_factsheet.pdf"
            self._download_file(factsheet, filename, "factsheets")
        
        # Mark network as scraped and save stats for future runs
        if network_id not in self.state.get("scraped_networks", []):
            self.state.setdefault("scraped_networks", []).append(network_id)
        
        # Save network stats so we can detect if settings increase later
        self.state.setdefault("network_stats", {})[network_id] = {
            "pages_scraped": len(network_data["pages_scraped"]),
            "max_depth_reached": crawl_stats["max_depth_reached"],
            "hit_page_limit": crawl_stats["hit_page_limit"],
            "hit_depth_limit": crawl_stats["hit_depth_limit"],
            "pdfs_found": len(network_data["pdfs_found"]),
            "last_scraped": datetime.now().isoformat(),
            "settings_used": {
                "max_crawl_depth": self.max_crawl_depth,
                "max_pages_per_network": self.max_pages_per_network
            }
        }
        
        self._save_state()
        
        return network_data
    
    def scrape_methodologies(self):
        """Download all EC methodological handbooks with fallback URLs."""
        if not self.ec_handbooks:
            self.logger.info("No EC handbooks configured or downloads disabled")
            return
        
        self.logger.info("=" * 60)
        self.logger.info("Downloading EC Methodological Handbooks")
        self.logger.info("=" * 60)
        
        for handbook_id, handbook_info in self.ec_handbooks.items():
            filename = f"{handbook_id}.pdf"
            urls = handbook_info.get("urls", [])
            
            if not urls:
                self.logger.warning(f"No URLs configured for {handbook_id}")
                continue
            
            filepath, status, successful_url = self._download_with_fallback(
                urls, 
                filename, 
                "methodologies"
            )
            
            self.results["methodologies"].append({
                "id": handbook_id,
                "title": handbook_info.get("title", "Unknown"),
                "description": handbook_info.get("description", ""),
                "urls_tried": urls,
                "successful_url": successful_url,
                "local_file": filepath,
                "status": status
            })
    
    def run(self):
        """Run the complete scraping process."""
        self.logger.info("=" * 60)
        self.logger.info("Starting ERN Resource Scraper v3.2 (Smart State Management)")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Networks to scrape: {len(self.networks_to_scrape)}")
        self.logger.info(f"Skip existing files: {self.skip_existing}")
        self.logger.info(f"Max crawl depth: {self.max_crawl_depth}")
        self.logger.info(f"Max pages per network: {self.max_pages_per_network}")
        self.logger.info("=" * 60)
        
        if not self.networks_to_scrape:
            self.logger.warning("No networks enabled for scraping!")
            self.logger.warning("Edit ern_config.json and set 'scrape': true for desired networks")
            return self.results
        
        # List enabled networks
        self.logger.info("Enabled networks:")
        for network_id in self.networks_to_scrape:
            self.logger.info(f"  - {network_id}")
        
        # Download methodologies first (they're the most important)
        self.scrape_methodologies()
        
        # Scrape all enabled networks
        for network_id, network_info in self.networks_to_scrape.items():
            try:
                network_data = self.scrape_network(network_id, network_info)
                self.results["networks"][network_id] = network_data
            except KeyboardInterrupt:
                self.logger.warning("Interrupted by user. Saving progress...")
                self._save_state()
                self._save_results()
                raise
            except Exception as e:
                self.logger.error(f"Error scraping {network_id}: {e}")
                self.results["errors"].append({
                    "network": network_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Save results
        self._save_results()
        self._create_summary()
        
        # Print final statistics
        self.logger.info("=" * 60)
        self.logger.info("Scraping complete!")
        self.logger.info(f"Networks scraped: {len(self.results['networks'])}")
        self.logger.info(f"Downloaded: {len(self.results['downloaded_files'])} files")
        self.logger.info(f"Skipped: {len(self.results['skipped_files'])} files")
        self.logger.info(f"Errors: {len(self.results['errors'])}")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info("=" * 60)
        
        # Print domain statistics
        self.logger.info("Domain request statistics:")
        for domain, count in sorted(self.domain_request_count.items(), key=lambda x: -x[1]):
            self.logger.info(f"  {domain}: {count} requests")
        
        return self.results
    
    def _save_results(self):
        """Save scraping results to JSON."""
        output_file = os.path.join(self.output_dir, "ern_scrape_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Results saved to {output_file}")
    
    def _create_summary(self):
        """Create a human-readable summary of findings."""
        summary_file = os.path.join(self.output_dir, "SUMMARY.md")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# ERN Resources Summary\n\n")
            f.write(f"**Scrape Date:** {self.results['metadata']['scrape_date']}\n")
            f.write(f"**Scraper Version:** {self.results['metadata'].get('scraper_version', '3.1')}\n")
            f.write(f"**Output Directory:** {self.results['metadata']['output_dir']}\n")
            f.write(f"**Networks Scraped:** {self.results['metadata'].get('total_networks_enabled', 'N/A')}\n\n")
            
            f.write("## Statistics\n\n")
            f.write(f"- **Files Downloaded:** {len(self.results['downloaded_files'])}\n")
            f.write(f"- **Files Skipped (cached):** {len(self.results['skipped_files'])}\n")
            f.write(f"- **Errors:** {len(self.results['errors'])}\n\n")
            
            if self.results["methodologies"]:
                f.write("## EC Methodological Handbooks\n\n")
                for m in self.results["methodologies"]:
                    status_icon = "[OK]" if m['status'] in ['downloaded', 'skipped'] else "[FAIL]"
                    f.write(f"- {status_icon} **{m['title']}** ({m['status']})\n")
                    f.write(f"  - {m['description']}\n")
            
            f.write("\n## Networks Summary\n\n")
            for network_id, data in self.results["networks"].items():
                if not data:
                    continue
                f.write(f"### {network_id}\n")
                info = data.get('info', {})
                f.write(f"- **Full Name:** {info.get('name', 'N/A')}\n")
                f.write(f"- **Disease Area:** {info.get('disease_area', 'N/A')}\n")
                f.write(f"- **Website:** {info.get('website', 'N/A')}\n")
                f.write(f"- **Pages Scraped:** {len(data.get('pages_scraped', []))}\n")
                f.write(f"- **Pages Failed:** {len(data.get('pages_failed', []))}\n")
                f.write(f"- **PDFs Found:** {len(data.get('pdfs_found', []))}\n")
                f.write(f"- **PDFs Downloaded:** {len(data.get('pdfs_downloaded', []))}\n\n")
            
            if self.results["errors"]:
                f.write("## Errors\n\n")
                for err in self.results["errors"][:20]:
                    url = err.get('url', err.get('network', 'Unknown'))
                    f.write(f"- {url}: {err['error'][:100]}\n")
                if len(self.results["errors"]) > 20:
                    f.write(f"\n... and {len(self.results['errors']) - 20} more errors\n")
        
        self.logger.info(f"Summary saved to {summary_file}")


# ============================================================================
# MAIN FUNCTION - ALL CONFIGURATION HERE
# ============================================================================

def main():
    """
    Main function - Configure all scraper parameters here.
    
    Edit the parameters below to customize the scraping behavior.
    Re-running the script will skip already downloaded files.
    """
    
    # ========================================================================
    # CONFIGURATION PARAMETERS - EDIT THESE VALUES
    # ========================================================================
    
    # Output directory for all downloaded files
    OUTPUT_DIR = "EU_ERN_DATA"
    
    # Path to config file (contains network definitions)
    CONFIG_FILE = "ern_config.json"
    
    # Skip files that already exist (enables re-running without re-downloading)
    # Set to True to resume interrupted scrapes
    # Set to False to force re-download everything
    SKIP_EXISTING = True
    
    # Verbose logging (shows more details)
    VERBOSE = True
    
    # What to do:
    # - "scrape"       : Run the full scraping process
    # - "list"         : Just print the list of all ERN networks with status
    # - "list_enabled" : Just print networks that will be scraped
    ACTION = "scrape"
    
    # ========================================================================
    # EXECUTION
    # ========================================================================
    
    # Load configuration
    config = load_config(CONFIG_FILE)
    
    if config is None:
        print("\n[ERROR] Could not load configuration.")
        print(f"[ERROR] Please ensure {CONFIG_FILE} exists in the same directory.")
        return
    
    # Handle different actions
    if ACTION == "list":
        list_networks(config, enabled_only=False)
        return
    
    if ACTION == "list_enabled":
        list_networks(config, enabled_only=True)
        return
    
    if ACTION == "scrape":
        # Get enabled networks
        enabled = get_networks_to_scrape(config)
        
        if not enabled:
            print("\n" + "=" * 60)
            print("[WARNING] No networks are enabled for scraping!")
            print("=" * 60)
            print("\nTo enable networks, edit ern_config.json and set")
            print('"scrape": true for the networks you want to scrape.')
            return
        
        # Get settings from config
        settings = config.get("scraper_settings", {})
        max_depth = settings.get("max_crawl_depth", 10)
        max_pages = settings.get("max_pages_per_network", 100)
        
        # Print startup info
        print(f"\n{'='*60}")
        print("ERN Resource Scraper v3.2 (Smart State Management)")
        print(f"{'='*60}")
        print(f"Config file: {CONFIG_FILE}")
        print(f"Output directory: {OUTPUT_DIR}/")
        print(f"Skip existing: {SKIP_EXISTING}")
        print(f"Max crawl depth: {max_depth}")
        print(f"Max pages/network: {max_pages}")
        print(f"Networks enabled: {len(enabled)}")
        for network_id in enabled:
            print(f"  - {network_id}")
        print(f"Output format: Clean Markdown (.md) for RAG")
        print(f"{'='*60}\n")
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Create and run scraper
        scraper = ERNScraper(
            config=config,
            output_dir=OUTPUT_DIR,
            skip_existing=SKIP_EXISTING,
            verbose=VERBOSE
        )
        
        try:
            results = scraper.run()
            
            print(f"\n{'='*60}")
            print("[OK] SCRAPING COMPLETE!")
            print(f"{'='*60}")
            print(f"   Output directory: {OUTPUT_DIR}/")
            print(f"   Networks scraped: {len(results['networks'])}")
            print(f"   Downloaded: {len(results['downloaded_files'])} files")
            print(f"   Skipped (cached): {len(results['skipped_files'])} files")
            print(f"   Errors: {len(results['errors'])}")
            print(f"\n   Re-run this script to resume or update.")
            print(f"   Set SKIP_EXISTING=False to force re-download.")
            print(f"{'='*60}")
            
        except KeyboardInterrupt:
            print(f"\n{'='*60}")
            print("[WARNING] SCRAPING INTERRUPTED")
            print(f"{'='*60}")
            print("   Progress has been saved. Re-run to continue.")
            print(f"{'='*60}")
    
    else:
        print(f"Unknown action: {ACTION}")
        print("Valid actions: 'scrape', 'list', 'list_enabled'")


if __name__ == "__main__":
    main()