#!/usr/bin/env python3
"""
ERN (European Reference Networks) Resource Scraper v4.0
========================================================
ULTIMATE EDITION - ENTERPRISE-GRADE WEB SCRAPING
```
██████╗ ██████╗ ███╗   ██╗    ███████╗ ██████╗██████╗  █████╗ ██████╗ ███████╗██████╗ 
██╔════╝██╔══██╗████╗  ██║    ██╔════╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗
█████╗  ██████╔╝██╔██╗ ██║    ███████╗██║     ██████╔╝███████║██████╔╝█████╗  ██████╔╝
██╔══╝  ██╔══██╗██║╚██╗██║    ╚════██║██║     ██╔══██╗██╔══██║██╔═══╝ ██╔══╝  ██╔══██╗
███████╗██║  ██║██║ ╚████║    ███████║╚██████╗██║  ██║██║  ██║██║     ███████╗██║  ██║
╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝    ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝
```
"""
#!/usr/bin/env python3
"""
ERN (European Reference Networks) Resource Scraper v4.2
========================================================
ON-THE-FLY PDF EDITION - ENTERPRISE-GRADE WEB SCRAPING
```
â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•— â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•— â–ˆâ–ˆâ–ˆâ•—   â–ˆâ–ˆâ•—    â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•— â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•— â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•— â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•— 
â–ˆâ–ˆâ•”â•â•â•â•â•â–ˆâ–ˆâ•”â•â•â–ˆâ–ˆâ•—â–ˆâ–ˆâ–ˆâ–ˆâ•—  â–ˆâ–ˆâ•‘    â–ˆâ–ˆâ•”â•â•â•â•â•â–ˆâ–ˆâ•”â•â•â•â•â•â–ˆâ–ˆâ•”â•â•â–ˆâ–ˆâ•—â–ˆâ–ˆâ•”â•â•â–ˆâ–ˆâ•—â–ˆâ–ˆâ•”â•â•â–ˆâ–ˆâ•—â–ˆâ–ˆâ•”â•â•â•â•â•â–ˆâ–ˆâ•”â•â•â–ˆâ–ˆâ•—
â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•”â•â–ˆâ–ˆâ•”â–ˆâ–ˆâ•— â–ˆâ–ˆâ•‘    â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—â–ˆâ–ˆâ•‘     â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•”â•â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•‘â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•”â•â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•”â•
â–ˆâ–ˆâ•”â•â•â•  â–ˆâ–ˆâ•”â•â•â–ˆâ–ˆâ•—â–ˆâ–ˆâ•‘â•šâ–ˆâ–ˆâ•—â–ˆâ–ˆâ•‘    â•šâ•â•â•â•â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•‘     â–ˆâ–ˆâ•”â•â•â–ˆâ–ˆâ•—â–ˆâ–ˆâ•”â•â•â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•”â•â•â•â• â–ˆâ–ˆâ•”â•â•â•  â–ˆâ–ˆâ•”â•â•â–ˆâ–ˆâ•—
â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—â–ˆâ–ˆâ•‘  â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•‘ â•šâ–ˆâ–ˆâ–ˆâ–ˆâ•‘    â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•‘â•šâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—â–ˆâ–ˆâ•‘  â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•‘  â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•‘     â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—â–ˆâ–ˆâ•‘  â–ˆâ–ˆâ•‘
â•šâ•â•â•â•â•â•â•â•šâ•â•  â•šâ•â•â•šâ•â•  â•šâ•â•â•â•    â•šâ•â•â•â•â•â•â• â•šâ•â•â•â•â•â•â•šâ•â•  â•šâ•â•â•šâ•â•  â•šâ•â•â•šâ•â•     â•šâ•â•â•â•â•â•â•â•šâ•â•  â•šâ•â•
```

Major Features:

 SMART URL FILTERING     - Skips images, admin pages, fragments, low-value content
 URL NORMALIZATION       - Strips fragments, normalizes paths for deduplication  
 CONTENT-TYPE DETECTION  - Skips binary files before downloading
 PARALLEL DOWNLOADS      - Concurrent PDF downloading with configurable threads
 PROGRESS TRACKING       - Real-time statistics with ETA
  SITEMAP DETECTION      - Automatically discovers and parses sitemap.xml
 ROBOTS.TXT RESPECT      - Optional robots.txt compliance
 SQLITE STATE MANAGEMENT - Robust checkpoint/resume with database backend
 SMART RE-SCRAPING       - Detects settings changes, re-scrapes affected networks
 CONTENT QUALITY SCORING - Prioritizes high-value medical content
  ERROR RECOVERY         - Graceful handling of all error conditions
 DETAILED REPORTING      - Comprehensive statistics and summaries
 CONTENT FINGERPRINTING  - Detects duplicate content across pages
 RATE LIMITING           - Domain-specific delays with exponential backoff

Usage:
    python ern_scraper.py                    # Run with default config
    python ern_scraper.py -c my_config.json  # Use custom config file
    python ern_scraper.py --list             # List all networks and status
    python ern_scraper.py --list-enabled     # List only enabled networks
    python ern_scraper.py --stats            # Show statistics from last run
    python ern_scraper.py --reset            # Reset state and start fresh
    python ern_scraper.py --export csv       # Export results to CSV

Author: ERN Research Pipeline
Date: 2025
Version: 4.0 Ultimate
"""

import os
import re
import json
import time
import hashlib
import requests
import signal
import sys
import sqlite3
import csv
import gzip
from io import BytesIO
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, urlencode
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import threading
from contextlib import contextmanager

# ============================================================================
# VERSION & CONSTANTS
# ============================================================================

VERSION = "4.2"
VERSION_NAME = "On-The-Fly PDF Edition"
DEFAULT_CONFIG_FILE = "ern_config.json"

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    @classmethod
    def disable(cls):
        """Disable colors for non-terminal output."""
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ''
        cls.WARNING = cls.FAIL = cls.ENDC = cls.BOLD = cls.DIM = ''

# Check if we're in a terminal
if not sys.stdout.isatty():
    Colors.disable()

# ============================================================================
# URL FILTERING PATTERNS
# ============================================================================

# File extensions to SKIP (binary/non-content files)
SKIP_EXTENSIONS = {
    # Images
    '.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.ico', '.bmp', '.tiff',
    # Media
    '.mp4', '.mp3', '.avi', '.mov', '.wmv', '.flv', '.wav', '.ogg',
    # Archives
    '.zip', '.rar', '.7z', '.tar', '.gz',
    # Documents (we download PDFs separately)
    '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    # Other binary
    '.exe', '.dmg', '.apk', '.iso',
    # Fonts
    '.woff', '.woff2', '.ttf', '.eot', '.otf',
    # Data
    '.json', '.xml', '.csv', '.sql',
}

# URL path patterns to SKIP (low-value pages)
SKIP_PATH_PATTERNS = [
    # WordPress admin/system
    r'/wp-admin/',
    r'/wp-login\.php',
    r'/wp-includes/',
    r'/wp-content/plugins/',
    r'/wp-content/themes/',
    r'/xmlrpc\.php',
    r'/wp-json/',
    
    # Author/archive pages (usually low value)
    r'/author/',
    r'/tag/',
    r'/category/',
    r'/archive/',
    
    # Pagination (we crawl main pages)
    r'/page/\d+/?$',
    r'\?page=\d+',
    r'\?paged=\d+',
    
    # WordPress attachments (images)
    r'/attachment/',
    
    # Search pages
    r'/search/',
    r'\?s=',
    
    # Login/register
    r'/login',
    r'/register',
    r'/sign-?in',
    r'/sign-?up',
    r'/account',
    r'/my-?account',
    
    # Cart/shop (if any)
    r'/cart',
    r'/checkout',
    r'/shop/',
    
    # Feed
    r'/feed/?$',
    r'/rss/?$',
    
    # Print versions
    r'/print/',
    r'\?print=',
    
    # Share/social
    r'/share/',
    r'\?share=',
    
    # Language switchers with just lang param
    r'^/?\?lang=',
    
    # Calendar/date archives
    r'/\d{4}/\d{2}/?$',  # Just year/month with no content
    
    # Common non-content paths
    r'/cdn-cgi/',
    r'/assets/',
    r'/static/',
    r'/_next/',
    r'/_nuxt/',
]

# Compiled patterns for efficiency
SKIP_PATH_REGEX = [re.compile(p, re.IGNORECASE) for p in SKIP_PATH_PATTERNS]

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

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
    "max_retries": 2,           # v4.2: Reduced from 3 for faster failure
    "backoff_base": 5.0,        # v4.2: Reduced from default
    "skip_existing": True,
    "verbose": True,
    "output_directory": "EU_ERN_DATA",
    "parallel_pdf_downloads": False,
    "max_parallel_downloads": 3,
    "respect_robots_txt": False,
    "check_content_type": True,
    "min_content_length": 100,
    "save_interval": 50,
    "use_sqlite_state": True,
    "detect_duplicates": True,
    "parse_sitemaps": True,
    "download_pdfs_on_the_fly": True,  # v4.2: Download PDFs immediately when found
}


# ============================================================================
# SQLITE STATE MANAGER
# ============================================================================

class SQLiteStateManager:
    """
    SQLite-based state management for robust checkpoint/resume.
    Much faster and more reliable than JSON for large crawls.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
    
    @property
    def conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self):
        """Initialize database schema."""
        with self.conn:
            self.conn.executescript('''
                CREATE TABLE IF NOT EXISTS urls (
                    url TEXT PRIMARY KEY,
                    normalized_url TEXT,
                    status TEXT DEFAULT 'pending',
                    network_id TEXT,
                    depth INTEGER DEFAULT 0,
                    content_hash TEXT,
                    file_path TEXT,
                    file_size INTEGER,
                    content_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS networks (
                    network_id TEXT PRIMARY KEY,
                    status TEXT DEFAULT 'pending',
                    pages_scraped INTEGER DEFAULT 0,
                    pages_failed INTEGER DEFAULT 0,
                    pages_skipped INTEGER DEFAULT 0,
                    pdfs_found INTEGER DEFAULT 0,
                    pdfs_downloaded INTEGER DEFAULT 0,
                    max_depth_reached INTEGER DEFAULT 0,
                    hit_page_limit INTEGER DEFAULT 0,
                    hit_depth_limit INTEGER DEFAULT 0,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    settings_json TEXT
                );
                
                CREATE TABLE IF NOT EXISTS content_hashes (
                    hash TEXT PRIMARY KEY,
                    first_url TEXT,
                    count INTEGER DEFAULT 1
                );
                
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_urls_status ON urls(status);
                CREATE INDEX IF NOT EXISTS idx_urls_network ON urls(network_id);
                CREATE INDEX IF NOT EXISTS idx_urls_normalized ON urls(normalized_url);
                
                CREATE TABLE IF NOT EXISTS pdf_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    normalized_url TEXT,
                    network_id TEXT,
                    link_text TEXT,
                    status TEXT DEFAULT 'pending',
                    file_path TEXT,
                    file_size INTEGER,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_pdf_queue_status ON pdf_queue(status);
                CREATE INDEX IF NOT EXISTS idx_pdf_queue_network ON pdf_queue(network_id);
            ''')
            
            # Set version
            self.set_metadata('version', VERSION)
            self.set_metadata('last_run', datetime.now().isoformat())
    
    def set_metadata(self, key: str, value: str):
        """Set a metadata value."""
        with self.conn:
            self.conn.execute('''
                INSERT OR REPLACE INTO metadata (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, value))
    
    def get_metadata(self, key: str, default: str = None) -> Optional[str]:
        """Get a metadata value."""
        row = self.conn.execute(
            'SELECT value FROM metadata WHERE key = ?', (key,)
        ).fetchone()
        return row['value'] if row else default
    
    def url_exists(self, url: str) -> bool:
        """Check if URL has been processed."""
        normalized = URLUtils.normalize_url(url)
        row = self.conn.execute(
            'SELECT 1 FROM urls WHERE normalized_url = ? AND status IN (?, ?)',
            (normalized, 'downloaded', 'skipped')
        ).fetchone()
        return row is not None
    
    def url_failed(self, url: str) -> bool:
        """Check if URL previously failed."""
        normalized = URLUtils.normalize_url(url)
        row = self.conn.execute(
            'SELECT 1 FROM urls WHERE normalized_url = ? AND status = ?',
            (normalized, 'failed')
        ).fetchone()
        return row is not None
    
    def add_url(self, url: str, network_id: str, status: str = 'pending',
                depth: int = 0, content_hash: str = None, file_path: str = None,
                file_size: int = None, content_type: str = None):
        """Add or update a URL record."""
        normalized = URLUtils.normalize_url(url)
        with self.conn:
            self.conn.execute('''
                INSERT OR REPLACE INTO urls 
                (url, normalized_url, status, network_id, depth, content_hash, 
                 file_path, file_size, content_type, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (url, normalized, status, network_id, depth, content_hash,
                  file_path, file_size, content_type))
    
    def get_network_status(self, network_id: str) -> Optional[Dict]:
        """Get network scrape status."""
        row = self.conn.execute(
            'SELECT * FROM networks WHERE network_id = ?', (network_id,)
        ).fetchone()
        return dict(row) if row else None
    
    def update_network(self, network_id: str, **kwargs):
        """Update network status."""
        # Build dynamic update
        fields = []
        values = []
        for key, value in kwargs.items():
            if key == 'settings':
                fields.append('settings_json = ?')
                values.append(json.dumps(value))
            else:
                fields.append(f'{key} = ?')
                values.append(value)
        
        values.append(network_id)
        
        with self.conn:
            # Ensure network exists
            self.conn.execute('''
                INSERT OR IGNORE INTO networks (network_id) VALUES (?)
            ''', (network_id,))
            
            if fields:
                self.conn.execute(f'''
                    UPDATE networks SET {', '.join(fields)} WHERE network_id = ?
                ''', values)
    
    def mark_network_complete(self, network_id: str):
        """Mark a network as fully scraped."""
        with self.conn:
            self.conn.execute('''
                UPDATE networks SET status = 'complete', completed_at = CURRENT_TIMESTAMP
                WHERE network_id = ?
            ''', (network_id,))
    
    def is_network_complete(self, network_id: str) -> bool:
        """Check if network is already complete."""
        row = self.conn.execute(
            'SELECT status FROM networks WHERE network_id = ?', (network_id,)
        ).fetchone()
        return row and row['status'] == 'complete'
    
    def check_content_duplicate(self, content_hash: str, url: str) -> Optional[str]:
        """
        Check if content hash already exists (duplicate detection).
        Returns the first URL with this content, or None if new.
        """
        row = self.conn.execute(
            'SELECT first_url FROM content_hashes WHERE hash = ?', (content_hash,)
        ).fetchone()
        
        if row:
            # Increment count
            with self.conn:
                self.conn.execute(
                    'UPDATE content_hashes SET count = count + 1 WHERE hash = ?',
                    (content_hash,)
                )
            return row['first_url']
        else:
            # Add new hash
            with self.conn:
                self.conn.execute(
                    'INSERT INTO content_hashes (hash, first_url) VALUES (?, ?)',
                    (content_hash, url)
                )
            return None
    
    def get_statistics(self) -> Dict:
        """Get overall statistics."""
        stats = {}
        
        # URL counts by status
        for row in self.conn.execute('''
            SELECT status, COUNT(*) as count FROM urls GROUP BY status
        '''):
            stats[f'urls_{row["status"]}'] = row['count']
        
        # Network counts
        for row in self.conn.execute('''
            SELECT status, COUNT(*) as count FROM networks GROUP BY status
        '''):
            stats[f'networks_{row["status"]}'] = row['count']
        
        # Total bytes
        row = self.conn.execute('''
            SELECT SUM(file_size) as total FROM urls WHERE file_size IS NOT NULL
        ''').fetchone()
        stats['total_bytes'] = row['total'] or 0
        
        # Duplicate count
        row = self.conn.execute('''
            SELECT SUM(count - 1) as duplicates FROM content_hashes WHERE count > 1
        ''').fetchone()
        stats['duplicate_pages'] = row['duplicates'] or 0
        
        return stats
    
    def export_to_json(self, filepath: str):
        """Export state to JSON for compatibility."""
        data = {
            'metadata': {},
            'urls': [],
            'networks': [],
            'statistics': self.get_statistics()
        }
        
        # Metadata
        for row in self.conn.execute('SELECT key, value FROM metadata'):
            data['metadata'][row['key']] = row['value']
        
        # URLs (limited for large databases)
        for row in self.conn.execute('SELECT * FROM urls LIMIT 10000'):
            data['urls'].append(dict(row))
        
        # Networks
        for row in self.conn.execute('SELECT * FROM networks'):
            data['networks'].append(dict(row))
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
    
    # =========== PDF QUEUE METHODS (v4.2) ===========
    
    def add_pdf_to_queue(self, url: str, network_id: str, link_text: str = "") -> bool:
        """Add PDF to download queue. Returns True if new, False if exists."""
        normalized = URLUtils.normalize_url(url)
        try:
            with self.conn:
                cursor = self.conn.execute("""
                    INSERT OR IGNORE INTO pdf_queue 
                    (url, normalized_url, network_id, link_text, status)
                    VALUES (?, ?, ?, ?, 'pending')
                """, (url, normalized, network_id, link_text))
            return cursor.rowcount > 0
        except Exception:
            return False
    
    def get_pending_pdfs(self, network_id: str = None, limit: int = 1000) -> List[Dict]:
        """Get pending PDFs from queue."""
        if network_id:
            rows = self.conn.execute("""
                SELECT * FROM pdf_queue 
                WHERE status = 'pending' AND network_id = ?
                ORDER BY created_at LIMIT ?
            """, (network_id, limit)).fetchall()
        else:
            rows = self.conn.execute("""
                SELECT * FROM pdf_queue 
                WHERE status = 'pending'
                ORDER BY created_at LIMIT ?
            """, (limit,)).fetchall()
        return [dict(row) for row in rows]
    
    def update_pdf_status(self, url: str, status: str, file_path: str = None, 
                          file_size: int = None, error_message: str = None):
        """Update PDF download status."""
        with self.conn:
            self.conn.execute("""
                UPDATE pdf_queue 
                SET status = ?, file_path = ?, file_size = ?, 
                    error_message = ?, updated_at = CURRENT_TIMESTAMP
                WHERE url = ?
            """, (status, file_path, file_size, error_message, url))
    
    def is_pdf_in_queue(self, url: str) -> bool:
        """Check if PDF is already in queue (any status)."""
        normalized = URLUtils.normalize_url(url)
        row = self.conn.execute("""
            SELECT 1 FROM pdf_queue WHERE normalized_url = ?
        """, (normalized,)).fetchone()
        return row is not None
    
    def is_pdf_downloaded(self, url: str) -> bool:
        """Check if PDF was already downloaded."""
        normalized = URLUtils.normalize_url(url)
        row = self.conn.execute("""
            SELECT 1 FROM pdf_queue 
            WHERE normalized_url = ? AND status = 'downloaded'
        """, (normalized,)).fetchone()
        return row is not None
    
    def get_pdf_queue_stats(self) -> Dict:
        """Get PDF queue statistics."""
        stats = {'pending': 0, 'downloaded': 0, 'failed': 0}
        for row in self.conn.execute("""
            SELECT status, COUNT(*) as count FROM pdf_queue GROUP BY status
        """):
            stats[row['status']] = row['count']
        return stats


# ============================================================================
# CONTENT FINGERPRINTING
# ============================================================================

class ContentFingerprinter:
    """
    Creates fingerprints of content for duplicate detection.
    Uses simhash-like approach for near-duplicate detection.
    """
    
    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute content hash for exact duplicate detection."""
        # Normalize whitespace
        normalized = ' '.join(text.split()).lower()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
    
    @staticmethod
    def extract_text_features(html: str) -> List[str]:
        """Extract text features from HTML for comparison."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove scripts/styles
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        
        # Extract n-grams (shingles)
        words = text.lower().split()
        if len(words) < 5:
            return []
        
        # 5-word shingles
        shingles = []
        for i in range(len(words) - 4):
            shingle = ' '.join(words[i:i+5])
            shingles.append(shingle)
        
        return shingles
    
    @staticmethod
    def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


# ============================================================================
# SITEMAP PARSER
# ============================================================================

class SitemapParser:
    """Parses XML sitemaps to discover URLs."""
    
    @staticmethod
    def parse(content: str, base_url: str) -> List[str]:
        """Parse sitemap XML and return list of URLs."""
        urls = []
        
        try:
            soup = BeautifulSoup(content, 'xml')
            
            # Standard sitemap
            for loc in soup.find_all('loc'):
                url = loc.get_text(strip=True)
                if url:
                    urls.append(url)
            
            # Sitemap index (links to other sitemaps)
            for sitemap in soup.find_all('sitemap'):
                loc = sitemap.find('loc')
                if loc:
                    urls.append(loc.get_text(strip=True))
        
        except Exception:
            # Try regex fallback
            loc_pattern = re.compile(r'<loc>([^<]+)</loc>')
            urls = loc_pattern.findall(content)
        
        return urls
    
    @staticmethod
    def is_sitemap_index(content: str) -> bool:
        """Check if content is a sitemap index."""
        return '<sitemapindex' in content.lower()


# ============================================================================
# URL DISCOVERY ENGINE
# ============================================================================

class URLDiscovery:
    """
    DISCOVERY-FIRST URL discovery for ERN websites.
    Runs systematic discovery BEFORE scraping to find actual site structure.
    
    Key Features:
    - Depth-2 exploration: follows nav links to discover more pages
    - Config paths as FALLBACK only: used only if discovery finds < 5 URLs
    - Suggested config output: logs discovered paths for config updates
    
    Discovery Strategy:
    1. Sitemap URLs (most comprehensive)
    2. Navigation URLs (site structure)
    3. Depth-2 exploration (linked pages)
    4. Config paths (ONLY if discovery fails)
    """
    
    # Minimum URLs to consider discovery successful
    MIN_DISCOVERED_URLS = 5
    
    # High-value path patterns for medical/guideline content
    HIGH_VALUE_PATTERNS = [
        r'guideline', r'protocol', r'pathway', r'recommendation',
        r'clinical', r'patient', r'disease', r'disorder',
        r'diagnosis', r'treatment', r'therapy', r'care',
        r'education', r'training', r'webinar', r'course',
        r'publication', r'research', r'registry', r'database',
        r'expert', r'professional', r'healthcare', r'hcp',
        r'resource', r'document', r'download', r'pdf',
        r'about', r'network', r'member', r'center', r'centre',
        # ERN-specific patterns
        r'working-group', r'work-package', r'epag', r'e-pag',
        r'governance', r'affiliated', r'partner', r'trial',
        r'cpms', r'meeting', r'event', r'dissemination',
        r'wg-leader', r'coordinator', r'board',
    ]
    
    # Paths to always skip
    SKIP_PATTERNS = [
        r'/wp-admin', r'/wp-includes', r'/wp-content/plugins',
        r'/login', r'/logout', r'/register', r'/sign-in',
        r'/cart', r'/checkout', r'/shop', r'/store',
        r'/search', r'/\?s=',
        r'/tag/', r'/author/', r'/category/',
        r'/page/\d+', r'/feed', r'/rss',
        r'/privacy', r'/cookie', r'/gdpr', r'/terms',
        r'/404', r'/error',
        r'\.(jpg|jpeg|png|gif|svg|ico|css|js|woff|pdf)$',
    ]
    
    def __init__(self, session: requests.Session, timeout: int = 15, logger=None):
        self.session = session
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
    
    def discover(self, website: str, config_paths: List[str] = None, depth: int = 2) -> Dict:
        """
        DISCOVERY-FIRST: Systematic URL discovery before scraping.
        Config paths are ONLY used if discovery finds < MIN_DISCOVERED_URLS.
        
        Args:
            website: Base URL to discover
            config_paths: Fallback paths from config (used only if discovery fails)
            depth: How many levels deep to explore (default 2)
        
        Returns:
            Dict with discovered URLs and metadata:
            {
                'sitemap_urls': [...],
                'nav_urls': [...],
                'page_urls': [...],
                'depth2_urls': [...],
                'config_urls': [...],  # Only populated if discovery fails
                'all_urls': [...],
                'discovery_success': bool,
                'suggested_paths': [...],  # For config updates
            }
        """
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info(f"[DISCOVERY] Systematic URL discovery: {website}")
        self.logger.info("=" * 60)
        
        base_domain = URLUtils.extract_domain(website)
        seen_urls = set()
        
        results = {
            'sitemap_urls': [],
            'nav_urls': [],
            'page_urls': [],
            'depth2_urls': [],
            'config_urls': [],
            'all_urls': [],
            'discovery_success': False,
            'suggested_paths': [],
        }
        
        # STEP 1: SITEMAP
        self.logger.info("")
        self.logger.info("[Step 1/4] Checking sitemaps...")
        sitemap_urls = self._fetch_sitemaps(website)
        results['sitemap_urls'] = sitemap_urls
        if sitemap_urls:
            self.logger.info(f"   [OK] Found {len(sitemap_urls)} URLs in sitemap")
            for url in sitemap_urls:
                seen_urls.add(URLUtils.normalize_url(url))
        else:
            self.logger.info("   [--] No sitemap found")
        
        # STEP 2: NAVIGATION
        self.logger.info("")
        self.logger.info("[Step 2/4] Analyzing navigation...")
        nav_urls, page_urls = self._analyze_main_page(website)
        results['nav_urls'] = nav_urls
        results['page_urls'] = page_urls
        
        if nav_urls:
            self.logger.info(f"   [OK] Found {len(nav_urls)} navigation links:")
            for url in nav_urls[:15]:
                path = urlparse(url).path
                self.logger.info(f"        - {path}")
                seen_urls.add(URLUtils.normalize_url(url))
            if len(nav_urls) > 15:
                self.logger.info(f"        ... and {len(nav_urls) - 15} more")
        
        if page_urls:
            self.logger.info(f"   [OK] Found {len(page_urls)} page links")
            for url in page_urls:
                seen_urls.add(URLUtils.normalize_url(url))
        
        # STEP 3: DEPTH-2 EXPLORATION
        if depth >= 2 and nav_urls:
            self.logger.info("")
            self.logger.info("[Step 3/4] Exploring depth 2 (following nav links)...")
            depth2_urls = self._explore_depth2(nav_urls[:15], base_domain, seen_urls)
            results['depth2_urls'] = depth2_urls
            if depth2_urls:
                self.logger.info(f"   [OK] Found {len(depth2_urls)} additional URLs at depth 2")
        else:
            self.logger.info("")
            self.logger.info("[Step 3/4] Skipping depth-2 (no nav links or depth<2)")
        
        # STEP 4: EVALUATE DISCOVERY SUCCESS
        total_discovered = (len(results['sitemap_urls']) + len(results['nav_urls']) + 
                          len(results['page_urls']) + len(results['depth2_urls']))
        results['discovery_success'] = total_discovered >= self.MIN_DISCOVERED_URLS
        
        self.logger.info("")
        self.logger.info("[Step 4/4] Evaluating discovery...")
        self.logger.info(f"   Sitemap:    {len(results['sitemap_urls']):3d} URLs")
        self.logger.info(f"   Navigation: {len(results['nav_urls']):3d} URLs")
        self.logger.info(f"   Page links: {len(results['page_urls']):3d} URLs")
        self.logger.info(f"   Depth-2:    {len(results['depth2_urls']):3d} URLs")
        self.logger.info(f"   -------------------------")
        self.logger.info(f"   TOTAL:      {total_discovered:3d} URLs")
        self.logger.info(f"   Threshold:  {self.MIN_DISCOVERED_URLS} URLs")
        
        # CONFIG FALLBACK (only if discovery failed)
        if results['discovery_success']:
            self.logger.info("")
            self.logger.info(f"   [OK] Discovery SUCCESS! Config paths will be IGNORED.")
        else:
            self.logger.info("")
            self.logger.info(f"   [!!] Discovery FAILED (< {self.MIN_DISCOVERED_URLS} URLs)")
            self.logger.info(f"   [!!] Using config paths as FALLBACK...")
            if config_paths:
                for path in config_paths:
                    url = urljoin(website, path)
                    normalized = URLUtils.normalize_url(url)
                    if normalized not in seen_urls:
                        results['config_urls'].append(url)
                        seen_urls.add(normalized)
                self.logger.info(f"   [OK] Added {len(results['config_urls'])} config paths")
        
        # COMBINE AND PRIORITIZE
        all_urls = self._prioritize_urls(results, base_domain)
        results['all_urls'] = all_urls
        
        # EXTRACT SUGGESTED PATHS FOR CONFIG
        results['suggested_paths'] = self._extract_suggested_paths(all_urls, website)
        
        # LOG SUGGESTED CONFIG
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("[CONFIG SUGGESTION] Update ern_config.json with these paths:")
        self.logger.info("=" * 60)
        self.logger.info('"guidelines_paths": [')
        for path in results['suggested_paths'][:25]:
            self.logger.info(f'    "{path}",')
        self.logger.info(']')
        self.logger.info("=" * 60)
        self.logger.info(f"[READY] {len(all_urls)} URLs queued for scraping")
        self.logger.info("=" * 60)
        
        return results
    
    def _explore_depth2(self, nav_urls: List[str], base_domain: str, seen_urls: set) -> List[str]:
        """Follow navigation links to discover more URLs (depth-2 exploration)."""
        depth2_urls = []
        
        for i, url in enumerate(nav_urls):
            try:
                self.logger.info(f"      [{i+1}/{len(nav_urls)}] {urlparse(url).path[:40]}...")
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract links from this page
                    for a in soup.find_all('a', href=True):
                        href = a.get('href', '')
                        full_url = self._normalize_link(href, url, base_domain)
                        if full_url:
                            normalized = URLUtils.normalize_url(full_url)
                            if normalized not in seen_urls:
                                seen_urls.add(normalized)
                                depth2_urls.append(full_url)
            except Exception as e:
                self.logger.debug(f"      Error exploring {url}: {e}")
        
        return depth2_urls
    
    def _extract_suggested_paths(self, urls: List[str], website: str) -> List[str]:
        """Extract clean paths from URLs for config suggestions."""
        paths = []
        seen = set()
        
        for url in urls:
            try:
                path = urlparse(url).path
                if path and path != '/' and path not in seen:
                    # Ensure trailing slash
                    if not path.endswith('/'):
                        path = path + '/'
                    seen.add(path)
                    paths.append(path)
            except:
                pass
        
        return sorted(paths)
    
    def _prioritize_urls(self, results: Dict, base_domain: str) -> List[str]:
        """Combine and prioritize URLs from all sources."""
        seen = set()
        prioritized = []
        
        def add_urls(urls: List[str], filter_high_value: bool = False):
            for url in urls:
                normalized = URLUtils.normalize_url(url)
                if normalized in seen:
                    continue
                if URLUtils.extract_domain(url) != base_domain:
                    continue
                if URLUtils.should_skip_url(url):
                    continue
                
                if filter_high_value:
                    path = urlparse(url).path.lower()
                    if not any(re.search(p, path, re.I) for p in self.HIGH_VALUE_PATTERNS):
                        continue
                
                seen.add(normalized)
                prioritized.append(url)
        
        # Priority order:
        # 1. Navigation (site structure)
        add_urls(results['nav_urls'])
        # 2. High-value sitemap URLs
        add_urls(results['sitemap_urls'], filter_high_value=True)
        # 3. Depth-2 discovered URLs
        add_urls(results['depth2_urls'])
        # 4. Remaining sitemap URLs
        add_urls(results['sitemap_urls'])
        # 5. Other page URLs
        add_urls(results['page_urls'])
        # 6. Config fallback (only if discovery failed)
        add_urls(results['config_urls'])
        
        return prioritized
    
    def _fetch_sitemaps(self, website: str) -> List[str]:
        """Fetch and parse all sitemaps."""
        urls = []
        sitemap_locations = [
            '/sitemap.xml',
            '/wp-sitemap.xml',
            '/sitemap_index.xml',
            '/sitemap/sitemap.xml',
            '/sitemaps/sitemap.xml',
        ]
        
        for location in sitemap_locations:
            sitemap_url = urljoin(website, location)
            try:
                response = self.session.get(sitemap_url, timeout=self.timeout)
                if response.status_code == 200 and '<' in response.text:
                    content = response.text
                    parsed = SitemapParser.parse(content, website)
                    
                    # If sitemap index, fetch sub-sitemaps
                    if SitemapParser.is_sitemap_index(content):
                        for sub_url in parsed[:10]:  # Limit sub-sitemaps
                            if sub_url.endswith('.xml') or '.xml' in sub_url:
                                try:
                                    sub_resp = self.session.get(sub_url, timeout=self.timeout)
                                    if sub_resp.status_code == 200:
                                        sub_urls = SitemapParser.parse(sub_resp.text, website)
                                        urls.extend(sub_urls)
                                except Exception:
                                    pass
                    else:
                        urls.extend(parsed)
                    
                    if urls:
                        break  # Found working sitemap
            except Exception:
                continue
        
        return urls
    
    def _analyze_main_page(self, website: str) -> Tuple[List[str], List[str]]:
        """Analyze main page for links."""
        page_urls = []
        nav_urls = []
        base_domain = URLUtils.extract_domain(website)
        
        try:
            response = self.session.get(website, timeout=self.timeout)
            if response.status_code != 200:
                return page_urls, nav_urls
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract navigation links (higher priority)
            for nav in soup.find_all(['nav', 'header']):
                for a in nav.find_all('a', href=True):
                    url = self._normalize_link(a.get('href'), website, base_domain)
                    if url and url not in nav_urls:
                        nav_urls.append(url)
            
            # Extract all other links
            for a in soup.find_all('a', href=True):
                url = self._normalize_link(a.get('href'), website, base_domain)
                if url and url not in page_urls and url not in nav_urls:
                    page_urls.append(url)
        
        except Exception as e:
            self.logger.debug(f"Error analyzing main page: {e}")
        
        return page_urls, nav_urls
    
    def _normalize_link(self, href: str, base_url: str, base_domain: str) -> Optional[str]:
        """Normalize a link and check if it should be included."""
        if not href:
            return None
        
        # Skip non-navigable
        if href.startswith(('#', 'mailto:', 'tel:', 'javascript:', 'data:')):
            return None
        
        # Build full URL
        full_url = urljoin(base_url, href)
        
        # Must be same domain
        if URLUtils.extract_domain(full_url) != base_domain:
            return None
        
        # Check skip patterns
        path = urlparse(full_url).path.lower()
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, path, re.IGNORECASE):
                return None
        
        # Normalize
        return URLUtils.normalize_url(full_url)


# ============================================================================
# ROBOTS.TXT PARSER
# ============================================================================

class RobotsTxtParser:
    """Simple robots.txt parser."""
    
    def __init__(self, content: str, user_agent: str = '*'):
        self.disallowed = []
        self.allowed = []
        self.crawl_delay = None
        self.sitemaps = []
        self._parse(content, user_agent)
    
    def _parse(self, content: str, target_agent: str):
        """Parse robots.txt content."""
        current_agent = None
        applies_to_us = False
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse directive
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == 'user-agent':
                    current_agent = value.lower()
                    applies_to_us = (current_agent == '*' or 
                                    target_agent.lower() in current_agent)
                
                elif applies_to_us:
                    if key == 'disallow' and value:
                        self.disallowed.append(value)
                    elif key == 'allow' and value:
                        self.allowed.append(value)
                    elif key == 'crawl-delay':
                        try:
                            self.crawl_delay = float(value)
                        except ValueError:
                            pass
                
                # Sitemaps apply to all
                if key == 'sitemap':
                    self.sitemaps.append(value)
    
    def is_allowed(self, path: str) -> bool:
        """Check if a path is allowed."""
        # Check allowed first (more specific)
        for pattern in self.allowed:
            if path.startswith(pattern):
                return True
        
        # Check disallowed
        for pattern in self.disallowed:
            if path.startswith(pattern):
                return False
        
        return True


# ============================================================================
# PROGRESS TRACKER
# ============================================================================

class ProgressTracker:
    """Tracks and displays progress with ETA calculation."""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def update(self, amount: int = 1):
        """Update progress."""
        with self._lock:
            self.current += amount
    
    def get_eta(self) -> str:
        """Calculate and return ETA string."""
        if self.current == 0:
            return "calculating..."
        
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed
        remaining = self.total - self.current
        
        if rate > 0:
            eta_seconds = remaining / rate
            eta = timedelta(seconds=int(eta_seconds))
            return str(eta)
        return "unknown"
    
    def get_progress_bar(self, width: int = 30) -> str:
        """Generate ASCII progress bar."""
        if self.total == 0:
            return "[" + "=" * width + "]"
        
        progress = self.current / self.total
        filled = int(width * progress)
        bar = "" * filled + "" * (width - filled)
        percentage = progress * 100
        
        return f"[{bar}] {percentage:.1f}%"
    
    def __str__(self) -> str:
        return f"{self.description}: {self.get_progress_bar()} ({self.current}/{self.total}) ETA: {self.get_eta()}"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CrawlStats:
    """Statistics for a single network crawl."""
    pages_scraped: int = 0
    pages_failed: int = 0
    pages_skipped: int = 0
    pdfs_found: int = 0
    pdfs_downloaded: int = 0
    max_depth_reached: int = 0
    hit_page_limit: bool = False
    hit_depth_limit: bool = False
    urls_filtered: int = 0
    images_skipped: int = 0
    admin_pages_skipped: int = 0
    fragments_normalized: int = 0
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0


@dataclass
class GlobalStats:
    """Global scraper statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    bytes_downloaded: int = 0
    pages_saved: int = 0
    pdfs_downloaded: int = 0
    rate_limits_hit: int = 0
    retries: int = 0


# ============================================================================
# URL UTILITIES
# ============================================================================

class URLUtils:
    """Utility class for URL manipulation and filtering."""
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """
        Normalize URL by:
        - Removing fragments (#...)
        - Removing trailing slashes (except root)
        - Lowercasing scheme and host
        - Sorting query parameters
        """
        try:
            parsed = urlparse(url)
            
            # Lowercase scheme and netloc
            scheme = parsed.scheme.lower()
            netloc = parsed.netloc.lower()
            
            # Remove fragment
            fragment = ''
            
            # Normalize path - remove trailing slash except for root
            path = parsed.path
            if path != '/' and path.endswith('/'):
                path = path.rstrip('/')
            
            # Sort query parameters for consistent comparison
            if parsed.query:
                query_params = parse_qs(parsed.query, keep_blank_values=True)
                sorted_query = urlencode(sorted(query_params.items()), doseq=True)
            else:
                sorted_query = ''
            
            # Reconstruct URL
            normalized = urlunparse((scheme, netloc, path, parsed.params, sorted_query, fragment))
            return normalized
            
        except Exception:
            return url
    
    @staticmethod
    def get_extension(url: str) -> str:
        """Extract file extension from URL."""
        try:
            parsed = urlparse(url)
            path = parsed.path.lower()
            # Handle URLs like /file.pdf?download=1
            if '.' in path:
                ext = '.' + path.rsplit('.', 1)[-1]
                # Remove any trailing query-like parts
                ext = ext.split('?')[0].split('#')[0]
                return ext
        except Exception:
            pass
        return ''
    
    @staticmethod
    def is_binary_extension(url: str) -> bool:
        """Check if URL points to a binary file."""
        ext = URLUtils.get_extension(url)
        return ext in SKIP_EXTENSIONS
    
    @staticmethod
    def is_image(url: str) -> bool:
        """Check if URL points to an image."""
        ext = URLUtils.get_extension(url)
        return ext in {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.ico', '.bmp', '.tiff'}
    
    @staticmethod
    def is_pdf(url: str) -> bool:
        """Check if URL points to a PDF."""
        return URLUtils.get_extension(url) == '.pdf'
    
    @staticmethod
    def should_skip_path(url: str) -> Tuple[bool, str]:
        """
        Check if URL path matches any skip patterns.
        Returns (should_skip, reason).
        """
        try:
            parsed = urlparse(url)
            path = parsed.path + ('?' + parsed.query if parsed.query else '')
            
            for i, pattern in enumerate(SKIP_PATH_REGEX):
                if pattern.search(path):
                    return True, SKIP_PATH_PATTERNS[i]
            
            return False, ""
        except Exception:
            return False, ""
    
    @staticmethod
    def is_same_domain(url: str, base_domain: str) -> bool:
        """Check if URL belongs to the same domain."""
        try:
            parsed = urlparse(url)
            url_domain = parsed.netloc.lower()
            base_domain = base_domain.lower()
            
            # Handle www prefix
            url_domain = url_domain.replace('www.', '')
            base_domain = base_domain.replace('www.', '')
            
            return url_domain == base_domain or url_domain.endswith('.' + base_domain)
        except Exception:
            return False
    
    @staticmethod
    def extract_domain(url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return ""
    
    @staticmethod
    def has_fragment(url: str) -> bool:
        """Check if URL has a fragment."""
        try:
            return bool(urlparse(url).fragment)
        except Exception:
            return False
    
    @staticmethod
    def should_skip_url(url: str) -> bool:
        """
        Check if URL should be skipped based on extension or path patterns.
        Returns True if URL should be skipped.
        """
        # Skip binary files (images, videos, etc.)
        if URLUtils.is_binary_extension(url):
            return True
        
        # Skip images specifically
        if URLUtils.is_image(url):
            return True
        
        # Skip paths matching skip patterns
        should_skip, _ = URLUtils.should_skip_path(url)
        if should_skip:
            return True
        
        return False


# ============================================================================
# CONTENT ANALYZER
# ============================================================================

class ContentAnalyzer:
    """Analyzes and scores content quality."""
    
    # Keywords that indicate high-value content
    HIGH_VALUE_KEYWORDS = [
        'guideline', 'protocol', 'pathway', 'consensus', 'recommendation',
        'diagnosis', 'treatment', 'therapy', 'clinical', 'patient',
        'disease', 'syndrome', 'disorder', 'management', 'care',
        'registry', 'network', 'expert', 'specialist', 'healthcare',
        'evidence', 'research', 'study', 'publication', 'journal',
        'education', 'training', 'webinar', 'course', 'workshop',
    ]
    
    # Keywords that indicate low-value content
    LOW_VALUE_KEYWORDS = [
        'cookie', 'privacy policy', 'terms of service', 'disclaimer',
        'login', 'register', 'subscribe', 'newsletter',
        'cart', 'checkout', 'payment', 'shipping',
        '404', 'not found', 'error', 'access denied',
    ]
    
    @classmethod
    def score_content(cls, text: str, url: str = "") -> int:
        """
        Score content quality from 0-100.
        Higher scores indicate more valuable content for RAG.
        """
        if not text:
            return 0
        
        text_lower = text.lower()
        score = 50  # Base score
        
        # Length bonus (up to +20)
        word_count = len(text.split())
        if word_count > 500:
            score += 20
        elif word_count > 200:
            score += 15
        elif word_count > 100:
            score += 10
        elif word_count < 50:
            score -= 20
        
        # High-value keyword bonus (up to +20)
        high_value_count = sum(1 for kw in cls.HIGH_VALUE_KEYWORDS if kw in text_lower)
        score += min(high_value_count * 2, 20)
        
        # Low-value keyword penalty (up to -30)
        low_value_count = sum(1 for kw in cls.LOW_VALUE_KEYWORDS if kw in text_lower)
        score -= min(low_value_count * 5, 30)
        
        # URL path bonus
        if url:
            url_lower = url.lower()
            if any(kw in url_lower for kw in ['guideline', 'protocol', 'pathway', 'clinical']):
                score += 10
        
        # Clamp to 0-100
        return max(0, min(100, score))
    
    @classmethod
    def is_content_page(cls, html: str) -> bool:
        """Check if page has substantial content (not just navigation/error)."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove scripts, styles, nav
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        
        text = soup.get_text(strip=True)
        
        # Too short
        if len(text) < 200:
            return False
        
        # Check for error page indicators
        title = soup.find('title')
        if title:
            title_text = title.get_text().lower()
            if any(err in title_text for err in ['404', 'not found', 'error', 'access denied']):
                return False
        
        return True


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(config_path: Optional[str] = None) -> Optional[Dict]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file. If None, uses DEFAULT_CONFIG_FILE.
        
    Returns:
        dict: Configuration dictionary or None if failed
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


def get_networks_to_scrape(config: Optional[Dict]) -> Dict:
    """Get list of networks that have scrape=true in config."""
    if config is None:
        return {}
    
    networks = config.get("networks", {})
    return {
        network_id: network_info 
        for network_id, network_info in networks.items() 
        if network_info.get("scrape", False)
    }


def get_ec_handbooks(config: Optional[Dict]) -> Dict:
    """Get EC methodological handbooks from config."""
    if config is None:
        return {}
    
    handbooks_config = config.get("ec_methodological_handbooks", {})
    if not handbooks_config.get("download_handbooks", True):
        return {}
    
    return handbooks_config.get("handbooks", {})


def list_networks(config: Optional[Dict], enabled_only: bool = False):
    """Print a formatted list of all networks with their scrape status."""
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
            status = " ENABLED "
        else:
            disabled_count += 1
            status = "  disabled"
        
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
    """
    Enhanced ERN scraper with smart filtering, SQLite state management,
    content fingerprinting, and enterprise-grade features.
    """
    
    def __init__(self, config: Optional[Dict] = None, output_dir: Optional[str] = None,
                 request_delay: Optional[float] = None, timeout: Optional[int] = None,
                 skip_existing: Optional[bool] = None, verbose: Optional[bool] = None):
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
        
        # Advanced settings
        self.max_retries = settings.get("max_retries", 3)
        self.backoff_base = settings.get("backoff_base", 5.0)
        self.max_crawl_depth = settings.get("max_crawl_depth", 10)
        self.max_pages_per_network = settings.get("max_pages_per_network", 100)
        self.parallel_downloads = settings.get("parallel_pdf_downloads", False)
        self.max_parallel = settings.get("max_parallel_downloads", 3)
        self.check_content_type = settings.get("check_content_type", True)
        self.min_content_length = settings.get("min_content_length", 100)
        self.save_interval = settings.get("save_interval", 50)
        self.use_sqlite = settings.get("use_sqlite_state", True)
        self.detect_duplicates = settings.get("detect_duplicates", True)
        self.parse_sitemaps = settings.get("parse_sitemaps", True)
        self.respect_robots = settings.get("respect_robots_txt", False)
        self.download_pdfs_on_the_fly = settings.get("download_pdfs_on_the_fly", True)  # v4.2
        
        # Domain delays from config or defaults
        self.domain_delays = self.config.get("domain_delays", DEFAULT_DOMAIN_DELAYS)
        
        # Track requests per domain for rate limiting
        self.domain_last_request: Dict[str, float] = defaultdict(float)
        self.domain_request_count: Dict[str, int] = defaultdict(int)
        
        # Robots.txt cache
        self.robots_cache: Dict[str, RobotsTxtParser] = {}
        
        # Global statistics
        self.global_stats = GlobalStats()
        
        # Thread lock for concurrent operations
        self._lock = threading.Lock()
        
        # Graceful shutdown flag
        self._shutdown = False
        
        # Create output directories
        self.dirs = {
            'root': self.output_dir,
            'guidelines': os.path.join(self.output_dir, 'guidelines'),
            'methodologies': os.path.join(self.output_dir, 'methodologies'),
            'factsheets': os.path.join(self.output_dir, 'factsheets'),
            'pages': os.path.join(self.output_dir, 'rag_content'),
            'logs': os.path.join(self.output_dir, 'logs'),
            'exports': os.path.join(self.output_dir, 'exports'),
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Setup session
        self._setup_session()
        
        # Setup state management
        if self.use_sqlite:
            db_path = os.path.join(self.output_dir, 'scraper_state.db')
            self.state_db = SQLiteStateManager(db_path)
            self.state = self._load_state_from_db()
            self.logger.info(f"[*] Using SQLite state management: {db_path}")
        else:
            self.state_db = None
            self.state_file = os.path.join(self.output_dir, 'scraper_state.json')
            self.state = self._load_state()
        
        # Check if settings have changed
        self._check_settings_changed()
        
        # Get enabled networks from config
        self.networks_to_scrape = get_networks_to_scrape(config)
        self.ec_handbooks = get_ec_handbooks(config)
        
        # Content fingerprinter
        self.fingerprinter = ContentFingerprinter()
        
        # URL Discovery engine
        self.url_discovery = URLDiscovery(
            session=self.session,
            timeout=self.timeout,
            logger=self.logger
        )
        
        # Results tracking
        self.results = {
            "metadata": {
                "scrape_date": datetime.now().isoformat(),
                "scraper_version": VERSION,
                "version_name": VERSION_NAME,
                "config_file": DEFAULT_CONFIG_FILE,
                "total_networks_enabled": len(self.networks_to_scrape),
                "output_dir": self.output_dir,
                "settings": {
                    "max_crawl_depth": self.max_crawl_depth,
                    "max_pages_per_network": self.max_pages_per_network,
                    "use_sqlite": self.use_sqlite,
                    "detect_duplicates": self.detect_duplicates,
                    "parse_sitemaps": self.parse_sitemaps,
                }
            },
            "networks": {},
            "methodologies": [],
            "downloaded_files": [],
            "skipped_files": [],
            "duplicate_pages": [],
            "errors": [],
            "global_stats": asdict(self.global_stats)
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Print banner
        self._print_banner()
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.logger.warning("\n[*]  Received interrupt signal. Saving progress and shutting down...")
        self._shutdown = True
    
    def _print_banner(self):
        """Print startup banner."""
        if self.verbose:
            banner = f"""
{Colors.CYAN}
  {Colors.BOLD}ERN Resource Scraper v{VERSION}{Colors.ENDC}{Colors.CYAN}                                  
  {Colors.DIM}{VERSION_NAME}{Colors.ENDC}{Colors.CYAN}                                             

  {Colors.GREEN}{Colors.CYAN} Smart URL Filtering    {Colors.GREEN}{Colors.CYAN} SQLite State Management      
  {Colors.GREEN}{Colors.CYAN} Content Fingerprinting {Colors.GREEN}{Colors.CYAN} Parallel PDF Downloads       
  {Colors.GREEN}{Colors.CYAN} Sitemap Discovery      {Colors.GREEN}{Colors.CYAN} RAG-Ready Output             
{Colors.ENDC}
"""
            print(banner)
    
    def _load_state_from_db(self) -> Dict:
        """Load state from SQLite database."""
        if not self.state_db:
            return self._load_state()
        
        # Build state dict from database
        state = {
            "downloaded_urls": [],
            "failed_urls": [],
            "scraped_networks": [],
            "network_stats": {},
            "settings": {},
            "last_run": self.state_db.get_metadata('last_run'),
            "version": self.state_db.get_metadata('version', VERSION)
        }
        
        # Get settings
        settings_json = self.state_db.get_metadata('settings')
        if settings_json:
            try:
                state["settings"] = json.loads(settings_json)
            except json.JSONDecodeError:
                pass
        
        # FIXED: Also try to load scraped_networks from metadata
        scraped_networks_json = self.state_db.get_metadata('scraped_networks')
        if scraped_networks_json:
            try:
                state["scraped_networks"] = json.loads(scraped_networks_json)
            except json.JSONDecodeError:
                pass
        
        # Get completed networks from table (if metadata doesn't have it)
        if not state["scraped_networks"]:
            for row in self.state_db.conn.execute(
                "SELECT network_id FROM networks WHERE status = 'complete'"
            ):
                state["scraped_networks"].append(row['network_id'])
        
        # Get network stats
        for row in self.state_db.conn.execute("SELECT * FROM networks"):
            state["network_stats"][row['network_id']] = {
                'pages_scraped': row['pages_scraped'],
                'max_depth_reached': row['max_depth_reached'],
                'hit_page_limit': bool(row['hit_page_limit']),
                'hit_depth_limit': bool(row['hit_depth_limit']),
                'pdfs_found': row['pdfs_found'],
            }
        
        # FIXED: If SQLite has no data, try loading from JSON backup
        json_state_file = os.path.join(self.output_dir, 'scraper_state.json')
        if not state["scraped_networks"] and os.path.exists(json_state_file):
            self.logger.info("[*] SQLite empty, loading from JSON backup...")
            try:
                with open(json_state_file, 'r', encoding='utf-8') as f:
                    json_state = json.load(f)
                state["scraped_networks"] = json_state.get("scraped_networks", [])
                state["network_stats"] = json_state.get("network_stats", {})
                state["settings"] = json_state.get("settings", {})
                self.logger.info(f"[*] Loaded from JSON: {len(state['scraped_networks'])} networks complete")
            except Exception as e:
                self.logger.warning(f"Could not load JSON backup: {e}")
        
        stats = self.state_db.get_statistics()
        self.logger.info(f"[*] Loaded state: {stats.get('urls_downloaded', 0)} URLs processed, "
                        f"{len(state['scraped_networks'])} networks complete")
        
        return state
    
    def _fetch_robots_txt(self, base_url: str) -> Optional[RobotsTxtParser]:
        """Fetch and parse robots.txt for a domain."""
        domain = URLUtils.extract_domain(base_url)
        
        if domain in self.robots_cache:
            return self.robots_cache[domain]
        
        robots_url = f"{urlparse(base_url).scheme}://{domain}/robots.txt"
        
        try:
            response = self.session.get(robots_url, timeout=10)
            if response.status_code == 200:
                parser = RobotsTxtParser(response.text, 'ERN-Research-Bot')
                self.robots_cache[domain] = parser
                
                if parser.crawl_delay:
                    self.logger.info(f"[*] robots.txt crawl-delay: {parser.crawl_delay}s")
                    # Update domain delay
                    self.domain_delays[domain] = max(
                        self.domain_delays.get(domain, self.default_delay),
                        parser.crawl_delay
                    )
                
                if parser.sitemaps:
                    self.logger.info(f"[*]  Found {len(parser.sitemaps)} sitemap(s) in robots.txt")
                
                return parser
        except Exception as e:
            self.logger.debug(f"Could not fetch robots.txt: {e}")
        
        self.robots_cache[domain] = None
        return None
    
    def _is_allowed_by_robots(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        if not self.respect_robots:
            return True
        
        domain = URLUtils.extract_domain(url)
        robots = self.robots_cache.get(domain)
        
        if robots is None:
            return True
        
        path = urlparse(url).path
        return robots.is_allowed(path)
    
    def _fetch_sitemap_urls(self, base_url: str) -> List[str]:
        """Fetch and parse sitemap for additional URLs."""
        if not self.parse_sitemaps:
            return []
        
        urls = []
        sitemap_urls_to_check = []
        
        # Check robots.txt for sitemaps
        robots = self._fetch_robots_txt(base_url)
        if robots and robots.sitemaps:
            sitemap_urls_to_check.extend(robots.sitemaps)
        
        # Also try common sitemap locations
        parsed = urlparse(base_url)
        common_sitemaps = [
            f"{parsed.scheme}://{parsed.netloc}/sitemap.xml",
            f"{parsed.scheme}://{parsed.netloc}/sitemap_index.xml",
            f"{parsed.scheme}://{parsed.netloc}/sitemap/sitemap.xml",
        ]
        
        for sitemap_url in common_sitemaps:
            if sitemap_url not in sitemap_urls_to_check:
                sitemap_urls_to_check.append(sitemap_url)
        
        # Fetch and parse sitemaps
        for sitemap_url in sitemap_urls_to_check[:5]:  # Limit to 5 sitemaps
            try:
                response = self.session.get(sitemap_url, timeout=15)
                if response.status_code == 200:
                    content = response.text
                    
                    # Handle gzipped sitemaps
                    if sitemap_url.endswith('.gz'):
                        content = gzip.decompress(response.content).decode('utf-8')
                    
                    parsed_urls = SitemapParser.parse(content, base_url)
                    
                    # If it's a sitemap index, recursively fetch
                    if SitemapParser.is_sitemap_index(content):
                        self.logger.info(f"[*]  Found sitemap index with {len(parsed_urls)} sub-sitemaps")
                        for sub_url in parsed_urls[:10]:  # Limit sub-sitemaps
                            if sub_url.endswith('.xml') or sub_url.endswith('.xml.gz'):
                                try:
                                    sub_response = self.session.get(sub_url, timeout=15)
                                    if sub_response.status_code == 200:
                                        sub_content = sub_response.text
                                        if sub_url.endswith('.gz'):
                                            sub_content = gzip.decompress(sub_response.content).decode('utf-8')
                                        urls.extend(SitemapParser.parse(sub_content, base_url))
                                except Exception:
                                    pass
                    else:
                        urls.extend(parsed_urls)
                        self.logger.info(f"[*]  Parsed sitemap: {len(parsed_urls)} URLs")
            except Exception as e:
                self.logger.debug(f"Could not fetch sitemap {sitemap_url}: {e}")
        
        # Filter URLs to same domain
        base_domain = URLUtils.extract_domain(base_url)
        urls = [u for u in urls if URLUtils.is_same_domain(u, base_domain)]
        
        return list(set(urls))  # Deduplicate
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler (detailed)
        log_file = os.path.join(self.dirs['logs'], f'ern_scraper_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[file_handler, console_handler]
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Also create a symlink to latest log
        latest_log = os.path.join(self.output_dir, 'ern_scraper.log')
        try:
            if os.path.exists(latest_log):
                os.remove(latest_log)
            os.symlink(log_file, latest_log)
        except (OSError, NotImplementedError):
            # Symlinks may not work on Windows
            pass
    
    def _setup_session(self):
        """Setup requests session with proper headers and adapters."""
        self.session = requests.Session()
        
        # Enhanced headers
        self.session.headers.update({
            'User-Agent': f'Mozilla/5.0 (compatible; ERN-Research-Bot/{VERSION}; +https://health.ec.europa.eu)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'DNT': '1',
        })
        
        # Connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=0  # We handle retries ourselves
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def _load_state(self) -> Dict:
        """Load previous scraper state if exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                self.logger.info(f"[*] Loaded previous state: {len(state.get('downloaded_urls', []))} URLs already processed")
                return state
            except Exception as e:
                self.logger.warning(f"Could not load state file: {e}")
        
        return {
            "downloaded_urls": [],
            "failed_urls": [],
            "scraped_networks": [],
            "network_stats": {},
            "settings": {},
            "last_run": None,
            "version": VERSION
        }
    
    def _save_state(self):
        """Save current scraper state for checkpointing."""
        if self.state_db:
            # Save metadata to SQLite
            self.state_db.set_metadata('last_run', datetime.now().isoformat())
            self.state_db.set_metadata('settings', json.dumps({
                "max_crawl_depth": self.max_crawl_depth,
                "max_pages_per_network": self.max_pages_per_network
            }))
            
            # FIXED: Also save network state to SQLite
            for network_id in self.state.get("scraped_networks", []):
                stats = self.state.get("network_stats", {}).get(network_id, {})
                self.state_db.update_network(
                    network_id,
                    status='complete',
                    pages_scraped=stats.get("pages_scraped", 0),
                    max_depth_reached=stats.get("max_depth_reached", 0),
                    hit_page_limit=1 if stats.get("hit_page_limit", False) else 0,
                    hit_depth_limit=1 if stats.get("hit_depth_limit", False) else 0,
                    pdfs_found=stats.get("pdfs_found", 0),
                    pdfs_downloaded=stats.get("pdfs_downloaded", 0),
                    settings=stats.get("settings_used", {})
                )
            
            # Save scraped networks list to metadata for quick lookup
            self.state_db.set_metadata('scraped_networks', json.dumps(
                self.state.get("scraped_networks", [])
            ))
        
        # ALWAYS save to JSON as backup (regardless of SQLite)
        self.state["last_run"] = datetime.now().isoformat()
        self.state["version"] = VERSION
        
        state_file = os.path.join(self.output_dir, 'scraper_state.json')
        with self._lock:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2)
    
    def _check_settings_changed(self):
        """Check if crawl settings have changed since last run."""
        previous_settings = self.state.get("settings", {})
        prev_depth = previous_settings.get("max_crawl_depth", 0)
        prev_pages = previous_settings.get("max_pages_per_network", 0)
        network_stats = self.state.get("network_stats", {})
        scraped_networks = self.state.get("scraped_networks", [])
        
        # Check for legacy state or version upgrade
        prev_version = self.state.get("version", "0.0")
        if prev_version < VERSION:
            self.logger.info(f"[*] Detected version upgrade: {prev_version} -> {VERSION}")
        
        # Check if this is a legacy state file
        is_legacy_state = len(scraped_networks) > 0 and len(network_stats) == 0
        
        if is_legacy_state:
            self.logger.info("[*] [LEGACY STATE] Detected old state file without network statistics")
            self.logger.info(f"[*] [LEGACY STATE] Will re-scrape all {len(scraped_networks)} networks")
            self.state["scraped_networks"] = []
            self.state["network_stats"] = {}
        else:
            # Normal settings change detection
            settings_increased = False
            
            if self.max_crawl_depth > prev_depth and prev_depth > 0:
                self.logger.info(f"[*]  [SETTINGS] Crawl depth increased: {prev_depth} -> {self.max_crawl_depth}")
                settings_increased = True
            
            if self.max_pages_per_network > prev_pages and prev_pages > 0:
                self.logger.info(f"[*]  [SETTINGS] Max pages increased: {prev_pages} -> {self.max_pages_per_network}")
                settings_increased = True
            
            if settings_increased:
                self.logger.info("[*] Checking which networks need re-scraping...")
                networks_to_rescrape = []
                
                for network_id in scraped_networks:
                    stats = network_stats.get(network_id, {})
                    
                    if not stats:
                        networks_to_rescrape.append(network_id)
                        continue
                    
                    hit_page_limit = stats.get("hit_page_limit", False)
                    hit_depth_limit = stats.get("hit_depth_limit", False)
                    pages_scraped = stats.get("pages_scraped", 0)
                    
                    should_rescrape = False
                    reasons = []
                    
                    if hit_page_limit and self.max_pages_per_network > prev_pages:
                        should_rescrape = True
                        reasons.append(f"hit page limit ({pages_scraped}/{prev_pages})")
                    
                    if hit_depth_limit and self.max_crawl_depth > prev_depth:
                        should_rescrape = True
                        reasons.append("hit depth limit")
                    
                    if prev_pages > 0 and pages_scraped >= prev_pages * 0.9:
                        should_rescrape = True
                        reasons.append(f"near page limit ({pages_scraped})")
                    
                    if should_rescrape:
                        networks_to_rescrape.append(network_id)
                        self.logger.info(f"  [*] {network_id} will be RE-SCRAPED: {', '.join(reasons)}")
                    else:
                        self.logger.info(f"  [*] {network_id} OK (didn't hit limits)")
                
                if networks_to_rescrape:
                    self.state["scraped_networks"] = [
                        n for n in scraped_networks if n not in networks_to_rescrape
                    ]
                    self.logger.info(f"[*] Marked {len(networks_to_rescrape)} network(s) for re-scraping")
        
        # Update stored settings
        self.state["settings"] = {
            "max_crawl_depth": self.max_crawl_depth,
            "max_pages_per_network": self.max_pages_per_network
        }
        self._save_state()
    
    def _get_delay_for_domain(self, domain: str) -> float:
        """Get the appropriate delay for a domain."""
        for pattern, delay in self.domain_delays.items():
            if pattern in domain:
                return delay
        return self.domain_delays.get("default", self.default_delay)
    
    def _wait_for_rate_limit(self, url: str):
        """Wait appropriate time based on domain rate limiting."""
        domain = URLUtils.extract_domain(url)
        delay = self._get_delay_for_domain(domain)
        
        with self._lock:
            last_request = self.domain_last_request[domain]
            elapsed = time.time() - last_request
            
            if elapsed < delay:
                wait_time = delay - elapsed + random.uniform(0.1, 0.5)
                self.logger.debug(f"[*] Rate limit: waiting {wait_time:.2f}s for {domain}")
                time.sleep(wait_time)
            
            self.domain_last_request[domain] = time.time()
            self.domain_request_count[domain] += 1
    
    def _should_skip_url(self, url: str, base_domain: str) -> Tuple[bool, str]:
        """
        Determine if a URL should be skipped.
        Returns (should_skip, reason).
        """
        # Check if same domain
        if not URLUtils.is_same_domain(url, base_domain):
            return True, "external domain"
        
        # Check for binary extensions (except PDF)
        if URLUtils.is_binary_extension(url) and not URLUtils.is_pdf(url):
            if URLUtils.is_image(url):
                return True, "image file"
            return True, "binary file"
        
        # Check path patterns
        should_skip, pattern = URLUtils.should_skip_path(url)
        if should_skip:
            return True, f"path pattern: {pattern}"
        
        return False, ""
    
    def _safe_request(self, url: str, retry_count: int = 0, 
                      check_content: bool = False) -> Optional[requests.Response]:
        """
        Make a safe HTTP request with retry logic.
        
        Args:
            url: URL to request
            retry_count: Current retry attempt
            check_content: If True, verify content-type is HTML before returning
            
        Returns:
            Response object or None if failed
        """
        if self._shutdown:
            return None
        
        # Wait for rate limit
        self._wait_for_rate_limit(url)
        
        self.global_stats.total_requests += 1
        
        try:
            # First, do a HEAD request to check content-type (optional)
            if self.check_content_type and check_content:
                try:
                    head_response = self.session.head(url, timeout=10, allow_redirects=True)
                    content_type = head_response.headers.get('Content-Type', '').lower()
                    
                    # Skip if not HTML
                    if content_type and not any(ct in content_type for ct in ['text/html', 'application/xhtml']):
                        self.logger.debug(f"[*]  Skipping non-HTML content: {content_type}")
                        return None
                except Exception:
                    pass  # Fall back to GET
            
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            
            # Handle rate limiting (429)
            if response.status_code == 429:
                self.global_stats.rate_limits_hit += 1
                if retry_count < self.max_retries:
                    wait_time = self.backoff_base * (2 ** retry_count) + random.uniform(1, 3)
                    self.logger.warning(f"[*]  Rate limited (429). Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    self.global_stats.retries += 1
                    return self._safe_request(url, retry_count + 1, check_content)
                else:
                    self.logger.error(f"[*] Max retries exceeded for {url}")
                    self.global_stats.failed_requests += 1
                    return None
            
            # Handle server errors (5xx)
            if response.status_code >= 500:
                if retry_count < self.max_retries:
                    wait_time = self.backoff_base * (2 ** retry_count)
                    self.logger.warning(f"[*]  Server error ({response.status_code}). Retrying...")
                    time.sleep(wait_time)
                    self.global_stats.retries += 1
                    return self._safe_request(url, retry_count + 1, check_content)
            
            response.raise_for_status()
            self.global_stats.successful_requests += 1
            self.global_stats.bytes_downloaded += len(response.content)
            return response
            
        except requests.exceptions.Timeout:
            if retry_count < self.max_retries:
                self.logger.warning(f"[*]  Timeout. Retrying ({retry_count + 1}/{self.max_retries})")
                time.sleep(self.backoff_base)
                self.global_stats.retries += 1
                return self._safe_request(url, retry_count + 1, check_content)
            self.logger.error(f"[*] Timeout after {self.max_retries} retries: {url}")
            self.global_stats.failed_requests += 1
            return None
            
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if "404" in error_msg:
                self.logger.debug(f"[*] Page not found: {url}")
            else:
                self.logger.error(f"[*] Request failed: {error_msg[:100]}")
            self.global_stats.failed_requests += 1
            return None
    
    def _extract_clean_content(self, html: str, url: str = "", title: str = "") -> str:
        """Extract clean, RAG-ready content from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Get page title
        if not title:
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)
                # Clean up common title patterns
                title = re.sub(r'\s*[|-]\s*.*$', '', title)
        
        # Get meta description
        meta_desc = ""
        meta = soup.find('meta', attrs={'name': 'description'})
        if meta:
            meta_desc = meta.get('content', '')
        
        # Get canonical URL
        canonical = soup.find('link', attrs={'rel': 'canonical'})
        canonical_url = canonical.get('href', url) if canonical else url
        
        # Remove non-content elements
        for tag in soup(['script', 'style', 'noscript', 'svg', 'iframe', 'nav', 
                         'header', 'footer', 'aside', 'form', 'button', 'input']):
            tag.decompose()
        
        # Remove noise elements by class/ID
        noise_patterns = [
            'cookie', 'gdpr', 'consent', 'banner', 'popup', 'modal',
            'menu', 'nav', 'sidebar', 'social', 'share', 'comment',
            'advertisement', 'newsletter', 'subscribe', 'widget', 
            'breadcrumb', 'footer', 'header', 'login', 'search',
            'pagination', 'related-posts', 'author-bio'
        ]
        for pattern in noise_patterns:
            for el in soup.find_all(class_=re.compile(pattern, re.I)):
                el.decompose()
            for el in soup.find_all(id=re.compile(pattern, re.I)):
                el.decompose()
        
        content_parts = []
        seen_text = set()
        
        # Extract main content
        main_content = soup.find('main') or soup.find('article') or soup.find(class_=re.compile(r'content|main', re.I)) or soup
        
        # Extract headings
        for h in main_content.find_all(['h1', 'h2', 'h3', 'h4']):
            text = h.get_text(strip=True)
            if text and len(text) > 3 and text not in seen_text:
                seen_text.add(text)
                level = int(h.name[1])
                content_parts.append(f"{'#' * level} {text}")
        
        # Extract paragraphs
        for p in main_content.find_all('p'):
            text = p.get_text(strip=True)
            if text and len(text) > 30 and text not in seen_text:
                seen_text.add(text)
                content_parts.append(text)
        
        # Extract list items
        for ul in main_content.find_all(['ul', 'ol']):
            parent_class = ' '.join(ul.get('class', []))
            if any(x in parent_class.lower() for x in ['menu', 'nav']):
                continue
            for li in ul.find_all('li', recursive=False):
                text = li.get_text(strip=True)
                if len(text) > 30 and text not in seen_text:
                    seen_text.add(text)
                    content_parts.append(f"- {text[:500]}")
        
        # Extract tables (simplified)
        for table in main_content.find_all('table'):
            rows = table.find_all('tr')
            if len(rows) > 1:
                content_parts.append("\n[Table content]")
                for row in rows[:10]:  # Limit rows
                    cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                    if cells:
                        content_parts.append(" | ".join(cells[:5]))
        
        # Extract PDF links
        pdf_links = []
        for a in soup.find_all('a', href=True):
            href = a.get('href', '')
            if URLUtils.is_pdf(href):
                text = a.get_text(strip=True) or 'Document'
                full_url = urljoin(url, href)
                pdf_links.append(f"- [PDF] {text}: {full_url}")
        
        # Deduplicate content
        unique_parts = []
        for part in content_parts:
            if part not in unique_parts:
                unique_parts.append(part)
        
        # Calculate quality score
        full_text = '\n'.join(unique_parts)
        quality_score = ContentAnalyzer.score_content(full_text, url)
        
        # Build markdown document
        md_parts = [
            "---",
            f"url: {canonical_url}",
            f"title: {title}",
            f"scraped_date: {datetime.now().isoformat()}",
            f"quality_score: {quality_score}",
            "---",
            "",
            f"# {title}" if title else "",
            "",
        ]
        
        if meta_desc:
            md_parts.append(f"> {meta_desc}\n")
        
        md_parts.append('\n\n'.join(unique_parts))
        
        if pdf_links:
            md_parts.append("\n\n## Related Documents\n")
            md_parts.append('\n'.join(pdf_links[:30]))
        
        return '\n'.join(md_parts)
    
    def _url_to_filename(self, url: str, prefix: str = "") -> str:
        """Convert URL to safe filename."""
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        
        if not filename or filename == "" or filename.endswith('_en'):
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            filename = f"document_{url_hash}"
        
        # Clean up filename
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        filename = re.sub(r'_+', '_', filename)
        
        if prefix:
            filename = f"{prefix}_{filename}"
        
        if not os.path.splitext(filename)[1]:
            filename += ".pdf"
        
        return filename
    
    def _file_exists(self, filepath: str) -> bool:
        """Check if file exists and has content."""
        return os.path.exists(filepath) and os.path.getsize(filepath) > 0
    
    def _download_file(self, url: str, filename: str, subdir: str = "guidelines") -> Tuple[Optional[str], str]:
        """Download a file to the output directory."""
        filepath = os.path.join(self.dirs[subdir], filename)
        
        # Check if already downloaded
        if self.skip_existing:
            if self._file_exists(filepath):
                self.logger.debug(f"[*]  Skipping existing: {filename}")
                return filepath, "skipped"
            
            if url in self.state["downloaded_urls"]:
                return filepath, "skipped"
            
            if url in self.state.get("failed_urls", []):
                return None, "skipped"
        
        # Download file
        response = self._safe_request(url)
        if response:
            content_type = response.headers.get('Content-Type', '')
            
            # Verify we got actual PDF content
            if 'pdf' in filename.lower() and len(response.content) < 100:
                self.logger.warning(f"[*]  Suspiciously small PDF ({len(response.content)} bytes)")
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"[*] Downloaded: {filename} ({len(response.content):,} bytes)")
            
            with self._lock:
                self.state["downloaded_urls"].append(url)
            
            self.global_stats.pdfs_downloaded += 1
            self._save_state()
            
            return filepath, "downloaded"
        
        # Track failed URLs
        with self._lock:
            if url not in self.state.get("failed_urls", []):
                self.state.setdefault("failed_urls", []).append(url)
        self._save_state()
        
        return None, "failed"
    
    def _download_with_fallback(self, urls: List[str], filename: str, 
                                subdir: str = "methodologies") -> Tuple[Optional[str], str, Optional[str]]:
        """Try to download from multiple URLs."""
        for url in urls:
            filepath, status = self._download_file(url, filename, subdir)
            if status in ["downloaded", "skipped"]:
                return filepath, status, url
        return None, "failed", None
    
    def _download_pdf_immediate(self, url: str, network_id: str, link_text: str = "") -> Tuple[Optional[str], str]:
        """
        v4.2: Download PDF immediately and update queue status.
        Returns (filepath, status) where status is 'downloaded', 'skipped', or 'failed'.
        """
        # Check if already in queue and downloaded
        if self.state_db and self.state_db.is_pdf_downloaded(url):
            return None, "skipped"
        
        # Add to queue (tracks all PDFs we've seen)
        if self.state_db:
            self.state_db.add_pdf_to_queue(url, network_id, link_text)
        
        # Generate filename
        filename = self._url_to_filename(url, prefix=network_id)
        filepath = os.path.join(self.dirs['guidelines'], filename)
        
        # Check if file already exists on disk
        if self.skip_existing and self._file_exists(filepath):
            if self.state_db:
                file_size = os.path.getsize(filepath)
                self.state_db.update_pdf_status(url, 'downloaded', filepath, file_size)
            self.logger.debug(f"[PDF] Skipping existing: {filename}")
            return filepath, "skipped"
        
        # Download the PDF
        self.logger.info(f"[PDF] Downloading: {filename[:50]}...")
        response = self._safe_request(url)
        
        if response:
            # Verify we got actual PDF content
            content_length = len(response.content)
            if content_length < 100:
                self.logger.warning(f"[PDF] Suspiciously small ({content_length} bytes): {filename}")
                if self.state_db:
                    self.state_db.update_pdf_status(url, 'failed', error_message="File too small")
                return None, "failed"
            
            # Save the file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"[PDF] OK: {filename} ({content_length:,} bytes)")
            
            # Update state
            if self.state_db:
                self.state_db.update_pdf_status(url, 'downloaded', filepath, content_length)
            
            with self._lock:
                self.state["downloaded_urls"].append(url)
            
            self.global_stats.pdfs_downloaded += 1
            return filepath, "downloaded"
        else:
            # Download failed
            if self.state_db:
                self.state_db.update_pdf_status(url, 'failed', error_message="Download failed")
            self.logger.warning(f"[PDF] FAILED: {filename}")
            return None, "failed"
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str, 
                       patterns: Optional[List[str]] = None) -> List[Dict]:
        """Extract relevant links from a page."""
        links = []
        
        if patterns is None:
            patterns = [
                r'guideline', r'cpg', r'cdst', r'protocol', r'pathway',
                r'consensus', r'recommendation', r'\.pdf$', r'clinical',
                r'diagnostic', r'therapeutic', r'flowchart', r'algorithm',
                r'handbook', r'manual', r'standard', r'care[\-_]?path',
                r'education', r'training', r'resource', r'publication'
            ]
        
        for a in soup.find_all('a', href=True):
            href = a.get('href', '')
            text = a.get_text(strip=True).lower()
            
            # Skip non-navigable
            if href.startswith(('mailto:', 'javascript:', 'tel:', 'data:')):
                continue
            
            # Skip fragments-only
            if href.startswith('#'):
                continue
            
            # Build full URL
            full_url = urljoin(base_url, href)
            
            # Check patterns
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
            normalized = URLUtils.normalize_url(link["url"])
            if normalized not in seen:
                seen.add(normalized)
                unique_links.append(link)
        
        return unique_links
    
    def scrape_network(self, network_id: str, network_info: Dict) -> Dict:
        """Scrape a single ERN network with auto URL discovery."""
        self.logger.info("=" * 60)
        self.logger.info(f"[*] Scraping {network_id}: {network_info.get('name', 'Unknown')}")
        self.logger.info("=" * 60)
        
        # Check if already scraped
        if self.skip_existing and network_id in self.state.get("scraped_networks", []):
            self.logger.info(f"[*]  Skipping {network_id} - already scraped")
            return self.results["networks"].get(network_id, {})
        
        start_time = datetime.now()
        crawl_stats = CrawlStats(start_time=start_time.isoformat())
        
        network_data = {
            "id": network_id,
            "info": {
                "name": network_info.get("name"),
                "website": network_info.get("website"),
                "disease_area": network_info.get("disease_area"),
                "description": network_info.get("full_description", "")
            },
            "pages_scraped": [],
            "pages_failed": [],
            "pages_skipped": [],
            "guidelines_found": [],
            "pdfs_found": [],
            "pdfs_downloaded": [],
            "crawl_stats": asdict(crawl_stats)
        }
        
        website = network_info.get("website")
        if not website:
            self.logger.error(f"[*] No website configured for {network_id}")
            return network_data
        
        base_domain = URLUtils.extract_domain(website)
        all_links = []
        
        # Seen URLs (normalized)
        seen_urls: Set[str] = set()
        
        # 
        # PHASE 1: DISCOVERY-FIRST URL DISCOVERY
        # 
        config_paths = network_info.get("guidelines_paths", [])
        
        # Run discovery-first with depth-2 exploration
        discovery_result = self.url_discovery.discover(
            website=website, 
            config_paths=config_paths,
            depth=2  # Explore 2 levels deep
        )
        
        # Store discovery info in network data
        network_data["discovery_info"] = {
            "success": discovery_result.get('discovery_success', False),
            "sitemap_urls": len(discovery_result.get('sitemap_urls', [])),
            "nav_urls": len(discovery_result.get('nav_urls', [])),
            "depth2_urls": len(discovery_result.get('depth2_urls', [])),
            "total_urls": len(discovery_result.get('all_urls', [])),
            "suggested_paths": discovery_result.get('suggested_paths', [])[:30],
        }
        
        discovered_urls = discovery_result.get('all_urls', [])
        
        # Build crawl queue from discovered URLs
        crawl_queue: deque = deque()
        for url in discovered_urls:
            normalized = URLUtils.normalize_url(url)
            if normalized not in seen_urls:
                path = urlparse(url).path
                crawl_queue.append((url, path, 1))
                seen_urls.add(normalized)
        
        self.logger.info(f"[*] {len(crawl_queue)} URLs in crawl queue")
        
        # 
        # PHASE 2: FETCH MAIN PAGE
        # 
        self.logger.info(f"[*] Fetching main page: {website}")
        response = self._safe_request(website, check_content=True)
        
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            links = self._extract_links(soup, website)
            all_links.extend(links)
            network_data["pages_scraped"].append(website)
            crawl_stats.pages_scraped += 1
            seen_urls.add(URLUtils.normalize_url(website))
            
            # Save main page
            page_file = os.path.join(self.dirs['pages'], f"{network_id}_main.md")
            clean_content = self._extract_clean_content(
                response.text, url=website, title=f"{network_id} - Main Page"
            )
            with open(page_file, 'w', encoding='utf-8') as f:
                f.write(clean_content)
            self.global_stats.pages_saved += 1
        else:
            network_data["pages_failed"].append(website)
            crawl_stats.pages_failed += 1
        
        # 
        # PHASE 3: CRAWL DISCOVERED PAGES
        # 
        # Main crawl loop
        pages_this_session = 0
        while crawl_queue and not self._shutdown:
            # Check page limit
            if crawl_stats.pages_scraped >= self.max_pages_per_network:
                self.logger.info(f"[*] Reached max pages limit ({self.max_pages_per_network})")
                crawl_stats.hit_page_limit = True
                break
            
            current_url, current_path, depth = crawl_queue.popleft()
            
            # Track max depth
            if depth > crawl_stats.max_depth_reached:
                crawl_stats.max_depth_reached = depth
            
            # Check depth limit
            if depth > self.max_crawl_depth:
                crawl_stats.hit_depth_limit = True
                continue
            
            # Normalize URL
            normalized_url = URLUtils.normalize_url(current_url)
            
            # Check if URL has fragment (log but continue with normalized)
            if URLUtils.has_fragment(current_url):
                crawl_stats.fragments_normalized += 1
                self.logger.debug(f"[*] Normalized URL with fragment: {current_url}")
            
            # Check if should skip
            should_skip, skip_reason = self._should_skip_url(current_url, base_domain)
            if should_skip:
                crawl_stats.urls_filtered += 1
                if 'image' in skip_reason:
                    crawl_stats.images_skipped += 1
                elif 'admin' in skip_reason or 'wp-' in skip_reason:
                    crawl_stats.admin_pages_skipped += 1
                network_data["pages_skipped"].append({"url": current_url, "reason": skip_reason})
                crawl_stats.pages_skipped += 1
                continue
            
            self.logger.info(f"[*] [L{depth}] {current_url[:80]}...")
            
            response = self._safe_request(current_url, check_content=True)
            
            if response:
                # Check if it's actually HTML
                content_type = response.headers.get('Content-Type', '')
                if not any(ct in content_type.lower() for ct in ['text/html', 'application/xhtml']):
                    self.logger.debug(f"[*]  Skipping non-HTML: {content_type}")
                    continue
                
                # Check if it's a content page
                if not ContentAnalyzer.is_content_page(response.text):
                    self.logger.debug("[*]  Skipping low-content page")
                    crawl_stats.pages_skipped += 1
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                page_links = self._extract_links(soup, current_url)
                all_links.extend(page_links)
                network_data["pages_scraped"].append(current_url)
                crawl_stats.pages_scraped += 1
                pages_this_session += 1
                
                # Save page
                safe_path = re.sub(r'[^\w\-]', '_', current_path).strip('_')[:50] or 'page'
                page_file = os.path.join(self.dirs['pages'], f"{network_id}_{safe_path}_{depth}.md")
                clean_content = self._extract_clean_content(
                    response.text, url=current_url, title=f"{network_id} - {safe_path}"
                )
                with open(page_file, 'w', encoding='utf-8') as f:
                    f.write(clean_content)
                self.global_stats.pages_saved += 1
                
                # Find sub-pages
                if depth < self.max_crawl_depth:
                    for a in soup.find_all('a', href=True):
                        href = a.get('href', '')
                        
                        # Skip non-page links
                        if href.startswith(('mailto:', 'javascript:', '#', 'tel:', 'data:')):
                            continue
                        
                        sub_url = urljoin(current_url, href)
                        sub_normalized = URLUtils.normalize_url(sub_url)
                        
                        if sub_normalized in seen_urls:
                            continue
                        
                        # Quick filter check
                        should_skip, _ = self._should_skip_url(sub_url, base_domain)
                        if should_skip:
                            continue
                        
                        # Skip PDFs in queue (we'll download them separately)
                        if URLUtils.is_pdf(sub_url):
                            continue
                        
                        seen_urls.add(sub_normalized)
                        sub_path = urlparse(sub_url).path
                        crawl_queue.append((sub_url, sub_path, depth + 1))
                
                # Periodic save
                if pages_this_session % self.save_interval == 0:
                    self._save_state()
                    self.logger.info(f"[*] Progress saved ({pages_this_session} pages this session)")
            else:
                network_data["pages_failed"].append(current_url)
                crawl_stats.pages_failed += 1
        
        # Log crawl summary
        self.logger.info(f"[*] Crawled {crawl_stats.pages_scraped} pages (max depth: {crawl_stats.max_depth_reached})")
        self.logger.info(f"   [*] Pages skipped: {crawl_stats.pages_skipped}")
        self.logger.info(f"   [*] Images skipped: {crawl_stats.images_skipped}")
        self.logger.info(f"   [*] Admin pages skipped: {crawl_stats.admin_pages_skipped}")
        self.logger.info(f"   [*] Fragments normalized: {crawl_stats.fragments_normalized}")
        
        if crawl_stats.hit_page_limit:
            self.logger.info(f"   [*]  Hit page limit ({self.max_pages_per_network})")
        if crawl_stats.hit_depth_limit:
            self.logger.info(f"   [*]  Hit depth limit ({self.max_crawl_depth})")
        
        # Categorize found links AND download PDFs on-the-fly (v4.2)
        pdf_urls_seen = set()
        for link in all_links:
            url = link["url"]
            if URLUtils.is_pdf(url):
                normalized = URLUtils.normalize_url(url)
                if normalized not in pdf_urls_seen:
                    pdf_urls_seen.add(normalized)
                    network_data["pdfs_found"].append(link)
                    crawl_stats.pdfs_found += 1
                    
                    # v4.2: Download PDF immediately if enabled
                    if self.download_pdfs_on_the_fly and not self._shutdown:
                        filepath, status = self._download_pdf_immediate(
                            url, network_id, link.get("text", "")
                        )
                        if status in ["downloaded", "skipped"]:
                            network_data["pdfs_downloaded"].append({
                                "url": url,
                                "filepath": filepath,
                                "status": status,
                                "link_text": link.get("text", "")
                            })
                            crawl_stats.pdfs_downloaded += 1
            else:
                network_data["guidelines_found"].append(link)
        
        self.logger.info(f"[*] Found {crawl_stats.pdfs_found} PDFs, {len(network_data['guidelines_found'])} other links")
        
        # Download PDFs - BATCH MODE (only if on-the-fly is disabled)
        if not self.download_pdfs_on_the_fly and network_data["pdfs_found"]:
            self.logger.info(f"[*] Downloading {len(network_data['pdfs_found'])} PDFs (batch mode)...")
            
            if self.parallel_downloads and len(network_data["pdfs_found"]) > 1:
                # Parallel download
                with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
                    futures = {}
                    for pdf_link in network_data["pdfs_found"]:
                        filename = self._url_to_filename(pdf_link["url"], prefix=network_id)
                        future = executor.submit(self._download_file, pdf_link["url"], filename, "guidelines")
                        futures[future] = pdf_link
                    
                    for future in as_completed(futures):
                        pdf_link = futures[future]
                        try:
                            filepath, status = future.result()
                            if status in ["downloaded", "skipped"]:
                                network_data["pdfs_downloaded"].append({
                                    "url": pdf_link["url"],
                                    "filepath": filepath,
                                    "status": status,
                                    "link_text": pdf_link.get("text", "")
                                })
                                crawl_stats.pdfs_downloaded += 1
                        except Exception as e:
                            self.logger.error(f"[*] PDF download error: {e}")
            else:
                # Sequential download
                for pdf_link in network_data["pdfs_found"]:
                    if self._shutdown:
                        break
                    filename = self._url_to_filename(pdf_link["url"], prefix=network_id)
                    filepath, status = self._download_file(pdf_link["url"], filename, "guidelines")
                    if status in ["downloaded", "skipped"]:
                        network_data["pdfs_downloaded"].append({
                            "url": pdf_link["url"],
                            "filepath": filepath,
                            "status": status,
                            "link_text": pdf_link.get("text", "")
                        })
                        crawl_stats.pdfs_downloaded += 1
        
        # Show on-the-fly download summary
        if self.download_pdfs_on_the_fly:
            self.logger.info(f"[*] PDFs downloaded on-the-fly: {crawl_stats.pdfs_downloaded}/{crawl_stats.pdfs_found}")
        
        # Download factsheet
        factsheet = network_info.get("factsheet")
        if factsheet and not self._shutdown:
            filename = f"{network_id}_factsheet.pdf"
            self._download_file(factsheet, filename, "factsheets")
        
        # Finalize stats
        end_time = datetime.now()
        crawl_stats.end_time = end_time.isoformat()
        crawl_stats.duration_seconds = (end_time - start_time).total_seconds()
        
        # Update network data
        network_data["crawl_stats"] = asdict(crawl_stats)
        
        # Mark as scraped
        if not self._shutdown:
            with self._lock:
                if network_id not in self.state.get("scraped_networks", []):
                    self.state.setdefault("scraped_networks", []).append(network_id)
                
                self.state.setdefault("network_stats", {})[network_id] = {
                    "pages_scraped": crawl_stats.pages_scraped,
                    "max_depth_reached": crawl_stats.max_depth_reached,
                    "hit_page_limit": crawl_stats.hit_page_limit,
                    "hit_depth_limit": crawl_stats.hit_depth_limit,
                    "pdfs_found": crawl_stats.pdfs_found,
                    "pdfs_downloaded": crawl_stats.pdfs_downloaded,
                    "last_scraped": end_time.isoformat(),
                    "duration_seconds": crawl_stats.duration_seconds,
                    "settings_used": {
                        "max_crawl_depth": self.max_crawl_depth,
                        "max_pages_per_network": self.max_pages_per_network
                    }
                }
            
            self._save_state()
        
        self.logger.info(f"[*] Completed {network_id} in {crawl_stats.duration_seconds:.1f}s")
        
        return network_data
    
    def scrape_methodologies(self):
        """Download EC methodological handbooks."""
        if not self.ec_handbooks:
            self.logger.info("[*] No EC handbooks configured")
            return
        
        self.logger.info("=" * 60)
        self.logger.info("[*] Downloading EC Methodological Handbooks")
        self.logger.info("=" * 60)
        
        for handbook_id, handbook_info in self.ec_handbooks.items():
            if self._shutdown:
                break
            
            filename = f"{handbook_id}.pdf"
            urls = handbook_info.get("urls", [])
            
            if not urls:
                self.logger.warning(f"[*]  No URLs for {handbook_id}")
                continue
            
            filepath, status, successful_url = self._download_with_fallback(urls, filename, "methodologies")
            
            self.results["methodologies"].append({
                "id": handbook_id,
                "title": handbook_info.get("title", "Unknown"),
                "description": handbook_info.get("description", ""),
                "urls_tried": urls,
                "successful_url": successful_url,
                "local_file": filepath,
                "status": status
            })
    
    def run(self) -> Dict:
        """Run the complete scraping process."""
        self.logger.info("=" * 60)
        self.logger.info(f"[*] ERN Resource Scraper v{VERSION}")
        self.logger.info("=" * 60)
        self.logger.info(f"[*] Output: {self.output_dir}")
        self.logger.info(f"[*] Networks: {len(self.networks_to_scrape)}")
        self.logger.info(f"[*] Max depth: {self.max_crawl_depth}")
        self.logger.info(f"[*] Max pages/network: {self.max_pages_per_network}")
        self.logger.info(f"[*]  Skip existing: {self.skip_existing}")
        self.logger.info("=" * 60)
        
        if not self.networks_to_scrape:
            self.logger.warning("[*]  No networks enabled!")
            return self.results
        
        # List enabled networks
        self.logger.info("[*] Enabled networks:")
        for network_id in self.networks_to_scrape:
            self.logger.info(f"   [*] {network_id}")
        
        # Download methodologies
        self.scrape_methodologies()
        
        # Scrape networks
        for network_id, network_info in self.networks_to_scrape.items():
            if self._shutdown:
                break
            
            try:
                network_data = self.scrape_network(network_id, network_info)
                self.results["networks"][network_id] = network_data
            except Exception as e:
                self.logger.error(f"[*] Error scraping {network_id}: {e}")
                self.results["errors"].append({
                    "network": network_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Update global stats
        self.results["global_stats"] = asdict(self.global_stats)
        
        # Save results
        self._save_results()
        self._create_summary()
        
        # Print final statistics
        self._print_final_stats()
        
        return self.results
    
    def _save_results(self):
        """Save results to JSON."""
        output_file = os.path.join(self.output_dir, "ern_scrape_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"[*] Results saved to {output_file}")
    
    def _create_summary(self):
        """Create human-readable summary."""
        summary_file = os.path.join(self.output_dir, "SUMMARY.md")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# ERN Resources Summary\n\n")
            f.write(f"**Scrape Date:** {self.results['metadata']['scrape_date']}\n")
            f.write(f"**Scraper Version:** {VERSION}\n")
            f.write(f"**Output Directory:** {self.output_dir}\n\n")
            
            # Global stats
            gs = self.global_stats
            f.write("## Global Statistics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total Requests | {gs.total_requests:,} |\n")
            f.write(f"| Successful | {gs.successful_requests:,} |\n")
            f.write(f"| Failed | {gs.failed_requests:,} |\n")
            f.write(f"| Data Downloaded | {gs.bytes_downloaded / 1024 / 1024:.2f} MB |\n")
            f.write(f"| Pages Saved | {gs.pages_saved:,} |\n")
            f.write(f"| PDFs Downloaded | {gs.pdfs_downloaded:,} |\n")
            f.write(f"| Rate Limits Hit | {gs.rate_limits_hit:,} |\n")
            f.write(f"| Retries | {gs.retries:,} |\n\n")
            
            # Methodologies
            if self.results["methodologies"]:
                f.write("## EC Methodological Handbooks\n\n")
                for m in self.results["methodologies"]:
                    icon = "" if m['status'] in ['downloaded', 'skipped'] else ""
                    f.write(f"- {icon} **{m['title']}** ({m['status']})\n")
            
            # Networks
            f.write("\n## Networks Summary\n\n")
            for network_id, data in self.results["networks"].items():
                if not data:
                    continue
                
                stats = data.get('crawl_stats', {})
                f.write(f"### {network_id}\n")
                f.write(f"- **Name:** {data.get('info', {}).get('name', 'N/A')}\n")
                f.write(f"- **Pages Scraped:** {stats.get('pages_scraped', 0)}\n")
                f.write(f"- **Pages Skipped:** {stats.get('pages_skipped', 0)}\n")
                f.write(f"- **PDFs Found:** {stats.get('pdfs_found', 0)}\n")
                f.write(f"- **PDFs Downloaded:** {stats.get('pdfs_downloaded', 0)}\n")
                f.write(f"- **Duration:** {stats.get('duration_seconds', 0):.1f}s\n\n")
            
            # Errors
            if self.results["errors"]:
                f.write("## Errors\n\n")
                for err in self.results["errors"][:20]:
                    f.write(f"- {err.get('network', err.get('url', 'Unknown'))}: {err['error'][:100]}\n")
        
        self.logger.info(f"[*] Summary saved to {summary_file}")
    
    def _print_final_stats(self):
        """Print final statistics."""
        gs = self.global_stats
        
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("[*] SCRAPING COMPLETE!")
        self.logger.info("=" * 60)
        self.logger.info(f"[*] Total requests: {gs.total_requests:,}")
        self.logger.info(f"   [*] Successful: {gs.successful_requests:,}")
        self.logger.info(f"   [*] Failed: {gs.failed_requests:,}")
        self.logger.info(f"   [*] Retries: {gs.retries:,}")
        self.logger.info(f"[*] Data downloaded: {gs.bytes_downloaded / 1024 / 1024:.2f} MB")
        self.logger.info(f"[*] Pages saved: {gs.pages_saved:,}")
        self.logger.info(f"[*] PDFs downloaded: {gs.pdfs_downloaded:,}")
        self.logger.info(f"[*] Output: {self.output_dir}/")
        self.logger.info("=" * 60)
        
        # Domain stats
        self.logger.info("[*] Domain request distribution:")
        for domain, count in sorted(self.domain_request_count.items(), key=lambda x: -x[1])[:10]:
            self.logger.info(f"   {domain}: {count:,}")


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_to_csv(results: Dict, output_dir: str):
    """Export results to CSV files."""
    exports_dir = os.path.join(output_dir, 'exports')
    os.makedirs(exports_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export pages
    pages_file = os.path.join(exports_dir, f'pages_{timestamp}.csv')
    with open(pages_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['network_id', 'url', 'title', 'quality_score', 'file_path'])
        
        for network_id, data in results.get('networks', {}).items():
            for page in data.get('pages_scraped', []):
                if isinstance(page, str):
                    writer.writerow([network_id, page, '', '', ''])
    
    # Export PDFs
    pdfs_file = os.path.join(exports_dir, f'pdfs_{timestamp}.csv')
    with open(pdfs_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['network_id', 'url', 'title', 'file_path', 'status'])
        
        for network_id, data in results.get('networks', {}).items():
            for pdf in data.get('pdfs_downloaded', []):
                writer.writerow([
                    network_id,
                    pdf.get('url', ''),
                    pdf.get('link_text', ''),
                    pdf.get('filepath', ''),
                    pdf.get('status', '')
                ])
    
    # Export statistics
    stats_file = os.path.join(exports_dir, f'statistics_{timestamp}.csv')
    with open(stats_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['network_id', 'pages_scraped', 'pages_failed', 'pdfs_found', 
                        'pdfs_downloaded', 'duration_seconds'])
        
        for network_id, data in results.get('networks', {}).items():
            stats = data.get('crawl_stats', {})
            writer.writerow([
                network_id,
                stats.get('pages_scraped', 0),
                stats.get('pages_failed', 0),
                stats.get('pdfs_found', 0),
                stats.get('pdfs_downloaded', 0),
                stats.get('duration_seconds', 0)
            ])
    
    print(f"[*] Exported to CSV:")
    print(f"   [*] {pages_file}")
    print(f"   [*] {pdfs_file}")
    print(f"   [*] {stats_file}")


def show_statistics(output_dir: str):
    """Show statistics from the last run."""
    # Try SQLite first
    db_path = os.path.join(output_dir, 'scraper_state.db')
    if os.path.exists(db_path):
        try:
            state_db = SQLiteStateManager(db_path)
            stats = state_db.get_statistics()
            
            print(f"\n{Colors.CYAN}{'=' * 60}{Colors.ENDC}")
            print(f"{Colors.BOLD}[*] Scraper Statistics (SQLite){Colors.ENDC}")
            print(f"{Colors.CYAN}{'=' * 60}{Colors.ENDC}")
            
            print(f"\n{Colors.GREEN}URLs:{Colors.ENDC}")
            print(f"   Downloaded:  {stats.get('urls_downloaded', 0):,}")
            print(f"   Skipped:     {stats.get('urls_skipped', 0):,}")
            print(f"   Failed:      {stats.get('urls_failed', 0):,}")
            print(f"   Pending:     {stats.get('urls_pending', 0):,}")
            
            print(f"\n{Colors.GREEN}Networks:{Colors.ENDC}")
            print(f"   Complete:    {stats.get('networks_complete', 0)}")
            print(f"   Pending:     {stats.get('networks_pending', 0)}")
            
            print(f"\n{Colors.GREEN}Content:{Colors.ENDC}")
            print(f"   Total data:  {stats.get('total_bytes', 0) / 1024 / 1024:.2f} MB")
            print(f"   Duplicates:  {stats.get('duplicate_pages', 0)}")
            
            print(f"\n{Colors.GREEN}Last run:{Colors.ENDC} {state_db.get_metadata('last_run', 'Never')}")
            print(f"{Colors.CYAN}{'=' * 60}{Colors.ENDC}\n")
            
            state_db.close()
            return
        except Exception as e:
            print(f"Error reading SQLite: {e}")
    
    # Fall back to JSON
    json_path = os.path.join(output_dir, 'ern_scrape_results.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            results = json.load(f)
        
        print(f"\n{Colors.CYAN}{'=' * 60}{Colors.ENDC}")
        print(f"{Colors.BOLD}[*] Scraper Statistics (JSON){Colors.ENDC}")
        print(f"{Colors.CYAN}{'=' * 60}{Colors.ENDC}")
        
        meta = results.get('metadata', {})
        print(f"\nScrape date: {meta.get('scrape_date', 'Unknown')}")
        print(f"Version: {meta.get('scraper_version', 'Unknown')}")
        print(f"Networks: {len(results.get('networks', {}))}")
        print(f"Downloads: {len(results.get('downloaded_files', []))}")
        print(f"Errors: {len(results.get('errors', []))}")
        
        print(f"{Colors.CYAN}{'=' * 60}{Colors.ENDC}\n")
    else:
        print(f"No statistics found in {output_dir}")


def reset_state(output_dir: str):
    """Reset scraper state."""
    print(f"\n{Colors.WARNING}[*]  This will reset all scraper state!{Colors.ENDC}")
    confirm = input("Are you sure? (yes/no): ")
    
    if confirm.lower() == 'yes':
        # Remove SQLite database
        db_path = os.path.join(output_dir, 'scraper_state.db')
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"   Removed: {db_path}")
        
        # Remove JSON state
        json_path = os.path.join(output_dir, 'scraper_state.json')
        if os.path.exists(json_path):
            os.remove(json_path)
            print(f"   Removed: {json_path}")
        
        print(f"\n{Colors.GREEN}[*] State reset complete{Colors.ENDC}\n")
    else:
        print("Cancelled.")


def download_pending_pdfs(output_dir: str, config: Dict = None):
    """
    v4.2: Download pending PDFs from the queue.
    Use this to resume interrupted PDF downloads.
    """
    db_path = os.path.join(output_dir, 'scraper_state.db')
    
    if not os.path.exists(db_path):
        print(f"[!] No database found at {db_path}")
        print("[!] Run scraper first to discover PDFs.")
        return
    
    state_db = SQLiteStateManager(db_path)
    
    # Get queue stats
    stats = state_db.get_pdf_queue_stats()
    pending = stats.get('pending', 0)
    downloaded = stats.get('downloaded', 0)
    failed = stats.get('failed', 0)
    
    print("\n" + "=" * 60)
    print("[*] PDF Queue Status")
    print("=" * 60)
    print(f"   Pending:    {pending}")
    print(f"   Downloaded: {downloaded}")
    print(f"   Failed:     {failed}")
    print("=" * 60)
    
    if pending == 0:
        print("\n[*] No pending PDFs to download.")
        state_db.close()
        return
    
    # Get pending PDFs
    pending_pdfs = state_db.get_pending_pdfs(limit=1000)
    
    print(f"\n[*] Downloading {len(pending_pdfs)} pending PDFs...")
    
    # Setup session
    session = requests.Session()
    session.headers.update({
        'User-Agent': f'Mozilla/5.0 (compatible; ERN-Research-Bot/{VERSION}; +https://health.ec.europa.eu)',
        'Accept': 'application/pdf,*/*',
    })
    
    # Create output directory
    guidelines_dir = os.path.join(output_dir, 'guidelines')
    os.makedirs(guidelines_dir, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, pdf in enumerate(pending_pdfs, 1):
        url = pdf['url']
        network_id = pdf['network_id']
        
        # Generate filename
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename or not filename.endswith('.pdf'):
            url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
            filename = f"document_{url_hash}.pdf"
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        if network_id:
            filename = f"{network_id}_{filename}"
        
        filepath = os.path.join(guidelines_dir, filename)
        
        # Check if already exists
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            state_db.update_pdf_status(url, 'downloaded', filepath, os.path.getsize(filepath))
            skip_count += 1
            print(f"   [{i}/{len(pending_pdfs)}] SKIP (exists): {filename[:50]}")
            continue
        
        print(f"   [{i}/{len(pending_pdfs)}] Downloading: {filename[:50]}...", end=" ", flush=True)
        
        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            
            content_length = len(response.content)
            if content_length < 100:
                print(f"FAIL (too small: {content_length} bytes)")
                state_db.update_pdf_status(url, 'failed', error_message="File too small")
                fail_count += 1
                continue
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"OK ({content_length:,} bytes)")
            state_db.update_pdf_status(url, 'downloaded', filepath, content_length)
            success_count += 1
            
            # Rate limiting
            time.sleep(2.0)
            
        except KeyboardInterrupt:
            print("\n\n[!] Interrupted. Progress saved.")
            break
        except Exception as e:
            print(f"FAIL ({str(e)[:30]})")
            state_db.update_pdf_status(url, 'failed', error_message=str(e)[:200])
            fail_count += 1
    
    # Final stats
    print("\n" + "=" * 60)
    print("[*] Download Complete")
    print("=" * 60)
    print(f"   Downloaded: {success_count}")
    print(f"   Skipped:    {skip_count}")
    print(f"   Failed:     {fail_count}")
    print(f"   Output:     {guidelines_dir}/")
    print("=" * 60 + "\n")
    
    state_db.close()


# ============================================================================
# MAIN CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# What action to perform:
#   "scrape"         - Run the scraper (with auto URL discovery)
#   "reset"          - Reset state and then scrape
#   "stats"          - Show statistics only
#   "list"           - List all networks
#   "download_queue" - Download pending PDFs from queue (v4.2)
ACTION = "scrape"

# Output directory
OUTPUT_DIR = "EU_ERN_DATA"

# Config file path
CONFIG_FILE = "ern_config.json"

# Skip already downloaded files (set False to force re-download)
SKIP_EXISTING = True

# Verbose logging
VERBOSE = True

# Scrape only specific network (set to None to scrape all enabled)
# Example: SINGLE_NETWORK = "ERKNet"
SINGLE_NETWORK = None

# Enable parallel PDF downloads
PARALLEL_DOWNLOADS = False

# Override crawl settings (set to None to use config file values)
MAX_CRAWL_DEPTH = None        # e.g., 20
MAX_PAGES_PER_NETWORK = None  # e.g., 500

# Auto URL discovery (recommended: True)
# When True: discovers URLs from sitemap + navigation before scraping
# When False: uses only config file paths
AUTO_DISCOVER_URLS = True

# v4.2: Download PDFs immediately when found (recommended: True)
# When True: PDFs are downloaded as soon as they're discovered during crawl
# When False: PDFs are downloaded in batch after crawl completes (legacy mode)
DOWNLOAD_PDFS_ON_THE_FLY = True


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point - all configuration is above."""
    
    # Handle reset action
    if ACTION == "reset":
        print(f"\n[*] Resetting scraper state...")
        db_path = os.path.join(OUTPUT_DIR, 'scraper_state.db')
        json_path = os.path.join(OUTPUT_DIR, 'scraper_state.json')
        
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"   Removed: {db_path}")
        if os.path.exists(json_path):
            os.remove(json_path)
            print(f"   Removed: {json_path}")
        
        print(f"   [*] State reset complete\n")
        # Continue to scrape after reset
    
    # Load configuration
    config = load_config(CONFIG_FILE)
    
    if config is None:
        print(f"\n[*] Could not load configuration from {CONFIG_FILE}")
        print("Please ensure the config file exists.")
        return
    
    # Handle list action
    if ACTION == "list":
        list_networks(config, enabled_only=False)
        return
    
    # Handle stats action
    if ACTION == "stats":
        db_path = os.path.join(OUTPUT_DIR, 'scraper_state.db')
        if os.path.exists(db_path):
            try:
                state_db = SQLiteStateManager(db_path)
                stats = state_db.get_statistics()
                pdf_stats = state_db.get_pdf_queue_stats()
                
                print(f"\n{'=' * 60}")
                print(f"[*] Scraper Statistics")
                print(f"{'=' * 60}")
                print(f"\nURLs:")
                print(f"   Downloaded:  {stats.get('urls_downloaded', 0):,}")
                print(f"   Skipped:     {stats.get('urls_skipped', 0):,}")
                print(f"   Failed:      {stats.get('urls_failed', 0):,}")
                print(f"\nNetworks:")
                print(f"   Complete:    {stats.get('networks_complete', 0)}")
                print(f"\nPDF Queue (v4.2):")
                print(f"   Pending:     {pdf_stats.get('pending', 0):,}")
                print(f"   Downloaded:  {pdf_stats.get('downloaded', 0):,}")
                print(f"   Failed:      {pdf_stats.get('failed', 0):,}")
                print(f"\nContent:")
                print(f"   Total data:  {stats.get('total_bytes', 0) / 1024 / 1024:.2f} MB")
                print(f"   Duplicates:  {stats.get('duplicate_pages', 0)}")
                print(f"\nLast run: {state_db.get_metadata('last_run', 'Never')}")
                print(f"{'=' * 60}\n")
                
                state_db.close()
            except Exception as e:
                print(f"Error reading stats: {e}")
        else:
            print(f"No statistics found. Run scraper first.")
        return
    
    # Handle download_queue action (v4.2)
    if ACTION == "download_queue":
        download_pending_pdfs(OUTPUT_DIR, config)
        return
    
    # Get enabled networks
    enabled = get_networks_to_scrape(config)
    
    # Filter to single network if specified
    if SINGLE_NETWORK:
        if SINGLE_NETWORK in enabled:
            enabled = {SINGLE_NETWORK: enabled[SINGLE_NETWORK]}
        else:
            print(f"\n[*] Network '{SINGLE_NETWORK}' not found or not enabled")
            return
    
    if not enabled:
        print(f"\n[*]  No networks are enabled for scraping!")
        print("Edit ern_config.json and set 'scrape': true for desired networks.")
        return
    
    # Apply setting overrides
    settings = config.get("scraper_settings", {})
    if MAX_CRAWL_DEPTH is not None:
        settings["max_crawl_depth"] = MAX_CRAWL_DEPTH
    if MAX_PAGES_PER_NETWORK is not None:
        settings["max_pages_per_network"] = MAX_PAGES_PER_NETWORK
    if PARALLEL_DOWNLOADS:
        settings["parallel_pdf_downloads"] = True
    
    # Set auto discovery
    settings["auto_discover_urls"] = AUTO_DISCOVER_URLS
    
    # v4.2: Set on-the-fly PDF downloads
    settings["download_pdfs_on_the_fly"] = DOWNLOAD_PDFS_ON_THE_FLY
    
    config["scraper_settings"] = settings
    
    # Create and run scraper
    scraper = ERNScraper(
        config=config,
        output_dir=OUTPUT_DIR,
        skip_existing=SKIP_EXISTING,
        verbose=VERBOSE
    )
    
    # Override networks if single specified
    if SINGLE_NETWORK:
        scraper.networks_to_scrape = enabled
    
    try:
        results = scraper.run()
        
        if scraper._shutdown:
            print(f"\n[*]  Scraping interrupted. Progress saved.")
            print(f"   Re-run to continue.")
        else:
            print(f"\n[*] Scraping complete!")
            print(f"   Output: {scraper.output_dir}/")
    
    except KeyboardInterrupt:
        print(f"\n[*]  Interrupted. Progress saved.")
    except Exception as e:
        print(f"\n[*] Error: {e}")
        if VERBOSE:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()