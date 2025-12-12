#!/usr/bin/env python3
"""
EU Pharmaceutical Guidelines Scraper v7
========================================
A robust, production-ready scraper for EU pharmaceutical regulations and guidelines.

Features:
- Real recursive crawling with depth control
- Smart resume: tracks state, skips completed work
- Exponential backoff with jitter for rate limiting
- Link discovery and deduplication
- Progress persistence (survives interruptions)
- Comprehensive error handling
- Fresh start or continue from where you left off
- RAG Content Extraction (orchestrates eu_pharma_rag_extractor.py)

Sources:
- European Commission Health Portal (EudraLex)
- European Medicines Agency (EMA)
- EUR-Lex (EU Law Database)

Output:
- /pdfs/           → Downloaded PDF guideline files
- /rag_content/    → Markdown files for RAG (from HTML pages)
- /metadata/       → Catalogs and state files

Configuration:
    Edit the CONFIG dictionary below to customize behavior.

Usage:
    python Eu_pharma_guidelines_scraper.py

Version: 7.0
"""

import os
import re
import json
import time
import random
import logging
import hashlib
import signal
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Set, List, Tuple, Any
from urllib.parse import urljoin, urlparse, unquote, parse_qs
from collections import deque
from enum import Enum

import requests
from bs4 import BeautifulSoup


# =============================================================================
# CONFIGURATION - EDIT THIS SECTION
# =============================================================================

CONFIG = {
    # -------------------------------------------------------------------------
    # Output Settings
    # -------------------------------------------------------------------------
    "output_dir": "EU_guidelines_library",
    "log_file": "scraper.log",
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    
    # -------------------------------------------------------------------------
    # What to Scrape (set to False to skip)
    # -------------------------------------------------------------------------
    "include_eudralex_vol10": True,      # EudraLex Volume 10 clinical trials
    "include_ec_orphan": True,            # EC Orphan Medicinal Products
    "include_known_pdfs": True,           # Known important PDFs (ICH, etc.)
    "include_eurlex": True,               # EUR-Lex legislation
    "include_ema_guidelines": True,       # EMA scientific guidelines
    "include_ema_orphan": True,           # EMA orphan designation pages
    "include_deep_crawl": True,           # Follow links to discover more docs
    
    # -------------------------------------------------------------------------
    # RAG Content Extraction (HTML pages → Markdown)
    # -------------------------------------------------------------------------
    "include_rag_extraction": True,       # Extract RAG content from HTML pages
    "rag_output_subdir": "rag_content",   # Subdirectory for RAG markdown files
    "rag_output_format": "markdown",      # markdown, json, or both
    
    # -------------------------------------------------------------------------
    # Resume & Skip Settings
    # -------------------------------------------------------------------------
    "skip_existing": True,                # Skip files already downloaded
    "verify_checksum": False,             # Re-download if checksum mismatch
    "force_fresh_start": False,           # Ignore previous state, start fresh
    "skip_failed_urls": True,             # Skip URLs that failed before
    "max_failures_per_url": 3,            # Max retries before marking URL as failed
    
    # -------------------------------------------------------------------------
    # Network Settings
    # -------------------------------------------------------------------------
    "request_timeout": 60,                # Total timeout per request (seconds)
    "connect_timeout": 15,                # Connection timeout (seconds)
    "delay_between_requests": 1.5,        # Base delay between requests
    "eurlex_delay": 5.0,                  # Delay for EUR-Lex (rate limited)
    "ema_delay": 2.0,                     # Delay for EMA requests
    
    # -------------------------------------------------------------------------
    # Retry Settings
    # -------------------------------------------------------------------------
    "max_retries": 3,                     # Max retry attempts per request
    "backoff_base": 5.0,                  # Base seconds for exponential backoff
    "backoff_max": 120.0,                 # Maximum backoff time (seconds)
    "backoff_jitter": True,               # Add randomness to backoff
    "skip_404_retry": True,               # Don't retry 404 errors
    
    # -------------------------------------------------------------------------
    # Crawl Settings
    # -------------------------------------------------------------------------
    "max_crawl_depth": 10,                 # How deep to follow links
    "max_pages_per_source": 500,           # Limit pages per source (None=unlimited)
    "max_documents_total": None,          # Limit total documents (None=unlimited)
    "allowed_domains": [                  # Only crawl these domains
        "health.ec.europa.eu",
        "ec.europa.eu",
        "ema.europa.eu",
        "www.ema.europa.eu",
        "eur-lex.europa.eu",
        "database.ich.org",
    ],
    "document_extensions": [".pdf", ".doc", ".docx", ".xls", ".xlsx"],
    "skip_url_patterns": [                # Skip URLs matching these patterns
        r"/search\?",
        r"/login",
        r"/user/",
        r"javascript:",
        r"mailto:",
        r"#$",
    ],
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class DownloadStatus(Enum):
    """Status of a document download."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class Document:
    """Represents a scraped document/guideline."""
    title: str
    url: str
    category: str
    subcategory: str = ""
    source: str = ""
    document_type: str = "PDF"
    description: str = ""
    date_published: str = ""
    date_scraped: str = ""
    local_path: str = ""
    file_size: int = 0
    checksum: str = ""
    status: str = "pending"
    failure_count: int = 0
    last_error: str = ""


@dataclass
class CrawlState:
    """Persistent state for resumable crawling."""
    visited_urls: Set[str] = field(default_factory=set)
    downloaded_urls: Set[str] = field(default_factory=set)
    failed_urls: Dict[str, int] = field(default_factory=dict)
    pending_urls: deque = field(default_factory=deque)
    documents: List[Document] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=lambda: {
        "downloaded": 0, "skipped": 0, "failed": 0, 
        "pages_crawled": 0, "total_bytes": 0
    })
    last_save: str = ""
    
    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "visited_urls": list(self.visited_urls),
            "downloaded_urls": list(self.downloaded_urls),
            "failed_urls": self.failed_urls,
            "pending_urls": list(self.pending_urls),
            "documents": [asdict(d) for d in self.documents],
            "stats": self.stats,
            "last_save": datetime.now().isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CrawlState":
        """Load from serialized dict."""
        state = cls()
        state.visited_urls = set(data.get("visited_urls", []))
        state.downloaded_urls = set(data.get("downloaded_urls", []))
        state.failed_urls = data.get("failed_urls", {})
        state.pending_urls = deque(data.get("pending_urls", []))
        state.stats = data.get("stats", state.stats)
        state.last_save = data.get("last_save", "")
        
        for doc_data in data.get("documents", []):
            try:
                doc = Document(**{k: v for k, v in doc_data.items() 
                                  if k in Document.__dataclass_fields__})
                state.documents.append(doc)
            except TypeError:
                pass
        
        return state


# =============================================================================
# MAIN SCRAPER CLASS
# =============================================================================

class EUGuidelinesScraper:
    """
    Enhanced scraper for EU pharmaceutical guidelines.
    Supports resumable crawling, smart retries, and deep link discovery.
    """
    
    # Source URLs
    SOURCES = {
        "eudralex_vol10": {
            "url": "https://health.ec.europa.eu/medicinal-products/eudralex/eudralex-volume-10_en",
            "category": "Clinical Trials",
            "subcategory": "EudraLex Volume 10",
        },
        "eudralex_vol10_clinical": {
            "url": "https://health.ec.europa.eu/medicinal-products/eudralex/eudralex-volume-10/clinical-trials-guidelines_en",
            "category": "Clinical Trials",
            "subcategory": "Clinical Trials Guidelines",
        },
        "ec_orphan": {
            "url": "https://health.ec.europa.eu/medicinal-products/orphan-medicinal-products_en",
            "category": "Orphan Medicinal Products",
            "subcategory": "EC Portal",
        },
        "ec_clinical_trials": {
            "url": "https://health.ec.europa.eu/medicinal-products/clinical-trials/clinical-trials-regulation-eu-no-5362014_en",
            "category": "Clinical Trials",
            "subcategory": "CTR 536/2014",
        },
        "ema_orphan": {
            "url": "https://www.ema.europa.eu/en/human-regulatory-overview/orphan-designation-overview",
            "category": "Orphan Medicinal Products",
            "subcategory": "EMA Overview",
        },
        "ema_orphan_applying": {
            "url": "https://www.ema.europa.eu/en/human-regulatory/research-development/orphan-designation/applying-orphan-designation",
            "category": "Orphan Medicinal Products",
            "subcategory": "Application Process",
        },
        "ema_clinical_trials": {
            "url": "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/clinical-trials-human-medicines",
            "category": "Clinical Trials",
            "subcategory": "EMA Overview",
        },
        "ema_scientific_guidelines": {
            "url": "https://www.ema.europa.eu/en/human-regulatory/research-development/scientific-guidelines/clinical-efficacy-safety/clinical-efficacy-safety-general",
            "category": "Scientific Guidelines",
            "subcategory": "Efficacy & Safety",
        },
        "ema_ich_guidelines": {
            "url": "https://www.ema.europa.eu/en/human-regulatory/research-development/scientific-guidelines/ich-guidelines",
            "category": "ICH Guidelines",
            "subcategory": "EMA ICH Portal",
        },
        "ema_gcp": {
            "url": "https://www.ema.europa.eu/en/human-regulatory/research-development/compliance/good-clinical-practice",
            "category": "Good Clinical Practice",
            "subcategory": "GCP Overview",
        },
    }
    
    # Known Important PDFs
    KNOWN_PDFS = {
        "guideline_small_populations": {
            "url": "https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-clinical-trials-small-populations_en.pdf",
            "title": "Guideline on Clinical Trials in Small Populations",
            "category": "Clinical Trials",
            "subcategory": "Small Populations",
        },
        "ich_e6_gcp": {
            "url": "https://www.ema.europa.eu/en/documents/scientific-guideline/ich-e-6-r2-guideline-good-clinical-practice-step-5_en.pdf",
            "title": "ICH E6(R2) Good Clinical Practice",
            "category": "Good Clinical Practice",
            "subcategory": "ICH E6",
        },
        "ich_e8_general": {
            "url": "https://www.ema.europa.eu/en/documents/regulatory-procedural-guideline/ich-guideline-e8-r1-general-considerations-clinical-studies_en.pdf",
            "title": "ICH E8(R1) General Considerations for Clinical Studies",
            "category": "ICH Guidelines",
            "subcategory": "ICH E8",
        },
        "ich_e9_statistics": {
            "url": "https://www.ema.europa.eu/en/documents/scientific-guideline/ich-e-9-statistical-principles-clinical-trials-step-5_en.pdf",
            "title": "ICH E9 Statistical Principles for Clinical Trials",
            "category": "ICH Guidelines",
            "subcategory": "ICH E9",
        },
        "ich_e9_r1_estimands": {
            "url": "https://www.ema.europa.eu/en/documents/scientific-guideline/ich-e9-r1-addendum-estimands-sensitivity-analysis-clinical-trials-guideline-statistical-principles_en.pdf",
            "title": "ICH E9(R1) Addendum on Estimands",
            "category": "ICH Guidelines",
            "subcategory": "ICH E9",
        },
        "ich_e10_control_group": {
            "url": "https://www.ema.europa.eu/en/documents/scientific-guideline/ich-e-10-choice-control-group-clinical-trials-step-5_en.pdf",
            "title": "ICH E10 Choice of Control Group in Clinical Trials",
            "category": "ICH Guidelines",
            "subcategory": "ICH E10",
        },
        "ich_e11_paediatric": {
            "url": "https://www.ema.europa.eu/en/documents/scientific-guideline/ich-e11r1-guideline-clinical-investigation-medicinal-products-pediatric-population-revision-1-addendum_en.pdf",
            "title": "ICH E11(R1) Clinical Investigation in Paediatric Population",
            "category": "ICH Guidelines",
            "subcategory": "ICH E11",
        },
        "ich_e17_multiregional": {
            "url": "https://www.ema.europa.eu/en/documents/scientific-guideline/ich-guideline-e17-general-principles-planning-and-design-multi-regional-clinical-trials-step-5-first-version_en.pdf",
            "title": "ICH E17 Multi-Regional Clinical Trials",
            "category": "ICH Guidelines",
            "subcategory": "ICH E17",
        },
        "ich_e2f_dsur": {
            "url": "https://database.ich.org/sites/default/files/E2F_Guideline.pdf",
            "title": "ICH E2F Development Safety Update Report (DSUR)",
            "category": "Clinical Trials",
            "subcategory": "Safety Reporting",
        },
        "first_in_human": {
            "url": "https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-strategies-identify-mitigate-risks-first-human-early-clinical-trials-investigational_en.pdf",
            "title": "Guideline on First-in-Human Clinical Trials",
            "category": "Clinical Trials",
            "subcategory": "First-in-Human",
        },
        "bioequivalence": {
            "url": "https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-investigation-bioequivalence-rev1_en.pdf",
            "title": "Guideline on the Investigation of Bioequivalence",
            "category": "Clinical Trials",
            "subcategory": "Bioequivalence",
        },
        "adaptive_designs": {
            "url": "https://www.ema.europa.eu/en/documents/scientific-guideline/reflection-paper-methodological-issues-confirmatory-clinical-trials-planned-adaptive-design_en.pdf",
            "title": "Reflection Paper on Adaptive Designs",
            "category": "Clinical Trials",
            "subcategory": "Adaptive Designs",
        },
        "missing_data": {
            "url": "https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-missing-data-confirmatory-clinical-trials_en.pdf",
            "title": "Guideline on Missing Data in Confirmatory Clinical Trials",
            "category": "Clinical Trials",
            "subcategory": "Statistics",
        },
    }
    
    # EUR-Lex Legislation
    EURLEX_LEGISLATION = [
        {
            "title": "Regulation (EU) No 536/2014 - Clinical Trials Regulation",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32014R0536",
            "category": "Legislation",
            "subcategory": "Clinical Trials",
        },
        {
            "title": "Regulation (EC) No 141/2000 - Orphan Medicinal Products",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32000R0141",
            "category": "Legislation",
            "subcategory": "Orphan Medicinal Products",
        },
        {
            "title": "Directive 2001/20/EC - Clinical Trials Directive (Legacy)",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32001L0020",
            "category": "Legislation",
            "subcategory": "Clinical Trials",
        },
        {
            "title": "Commission Directive 2005/28/EC - GCP Directive",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32005L0028",
            "category": "Legislation",
            "subcategory": "Good Clinical Practice",
        },
        {
            "title": "Commission Regulation (EC) No 847/2000 - Orphan Designation",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32000R0847",
            "category": "Legislation",
            "subcategory": "Orphan Medicinal Products",
        },
        {
            "title": "Regulation (EC) No 726/2004 - EMA Establishment",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32004R0726",
            "category": "Legislation",
            "subcategory": "General",
        },
        {
            "title": "Regulation (EC) No 1901/2006 - Paediatric Regulation",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32006R1901",
            "category": "Legislation",
            "subcategory": "Paediatric",
        },
        {
            "title": "Implementing Regulation (EU) 2017/556 - GCP Inspections",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32017R0556",
            "category": "Legislation",
            "subcategory": "Good Clinical Practice",
        },
        {
            "title": "Delegated Regulation (EU) 2017/1569 - GMP for IMPs",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32017R1569",
            "category": "Legislation",
            "subcategory": "Quality / GMP",
        },
    ]
    
    def __init__(self, config: dict):
        """Initialize the scraper with configuration."""
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.pdfs_dir = self.output_dir / "pdfs"
        self.metadata_dir = self.output_dir / "metadata"
        self.state_file = self.metadata_dir / "crawl_state.json"
        
        self._setup_directories()
        self._setup_logging()
        
        self.session = self._create_session()
        self.state = self._load_or_create_state()
        self.used_filenames: Dict[str, int] = {}
        
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info(f"Initialized. Output: {self.output_dir.absolute()}")
        self.logger.info(f"State: {len(self.state.downloaded_urls)} downloaded, "
                        f"{len(self.state.visited_urls)} visited")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.warning(f"Signal {signum} received. Saving and exiting...")
        self._shutdown_requested = True
    
    def _setup_directories(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        for cat in ["clinical_trials", "orphan_medicinal_products", "gcp",
                    "ich_guidelines", "legislation", "scientific_guidelines", "other"]:
            (self.pdfs_dir / cat).mkdir(exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging."""
        log_level = getattr(logging, self.config.get("log_level", "INFO").upper())
        
        self.logger = logging.getLogger("EUScraper")
        self.logger.setLevel(log_level)
        self.logger.handlers = []
        
        log_path = self.output_dir / self.config.get("log_file", "scraper.log")
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(ch)
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with browser headers."""
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        })
        return s
    
    def _create_eurlex_session(self) -> requests.Session:
        """Create fresh session for EUR-Lex."""
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf,text/html,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "DNT": "1",
            "Referer": "https://eur-lex.europa.eu/",
        })
        return s
    
    def _load_or_create_state(self) -> CrawlState:
        """Load existing state or create fresh."""
        if self.config.get("force_fresh_start", False):
            self.logger.info("Fresh start requested - ignoring previous state")
            return CrawlState()
        
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                state = CrawlState.from_dict(data)
                self.logger.info(f"Loaded state from {state.last_save}")
                return state
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Could not load state: {e}")
        
        return CrawlState()
    
    def _save_state(self):
        """Persist state to disk."""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, indent=2, ensure_ascii=False)
        except IOError as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        base = self.config.get("backoff_base", 5.0)
        maximum = self.config.get("backoff_max", 120.0)
        backoff = min(base * (2 ** attempt), maximum)
        if self.config.get("backoff_jitter", True):
            backoff *= (0.5 + random.random())
        return backoff
    
    def _get_delay(self, url: str) -> float:
        """Get delay based on domain."""
        domain = urlparse(url).netloc.lower()
        if "eur-lex" in domain:
            return self.config.get("eurlex_delay", 5.0)
        elif "ema.europa.eu" in domain:
            return self.config.get("ema_delay", 2.0)
        return self.config.get("delay_between_requests", 1.5)
    
    def _make_request(self, url: str, session: Optional[requests.Session] = None) -> Optional[requests.Response]:
        """Make HTTP request with retries."""
        if session is None:
            session = self.session
        
        max_retries = self.config.get("max_retries", 3)
        timeout = (self.config.get("connect_timeout", 15), self.config.get("request_timeout", 60))
        
        for attempt in range(max_retries):
            if self._shutdown_requested:
                return None
            
            try:
                response = session.get(url, timeout=timeout, allow_redirects=True)
                
                if response.status_code == 404:
                    if self.config.get("skip_404_retry", True):
                        self.logger.warning(f"404: {url[:70]}")
                        return None
                
                if response.status_code == 429:
                    wait = int(response.headers.get("Retry-After", 60))
                    self.logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                
                if response.status_code >= 500:
                    raise requests.exceptions.RequestException(f"Server error {response.status_code}")
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed ({attempt + 1}/{max_retries}): {url[:60]} - {e}")
                if attempt < max_retries - 1:
                    backoff = self._calculate_backoff(attempt)
                    self.logger.info(f"Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
        
        return None
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        parsed = urlparse(url)
        params_to_remove = ['utm_source', 'utm_medium', 'utm_campaign', 'ref', 
                           'source', 'fbclid', 'gclid', '_ga']
        
        if parsed.query:
            params = parse_qs(parsed.query, keep_blank_values=True)
            filtered_params = {k: v for k, v in params.items() 
                             if k.lower() not in params_to_remove}
            sorted_query = '&'.join(f"{k}={v[0]}" for k, v in sorted(filtered_params.items()))
        else:
            sorted_query = ''
        
        normalized = f"{parsed.scheme}://{parsed.netloc.lower()}{parsed.path}"
        if sorted_query:
            normalized += f"?{sorted_query}"
        
        return normalized
    
    def _is_url_processed(self, url: str) -> bool:
        """Check if URL (or normalized variant) has been processed."""
        normalized = self._normalize_url(url)
        return (url in self.state.downloaded_urls or 
                normalized in self.state.downloaded_urls or
                url in self.state.visited_urls or
                normalized in self.state.visited_urls)
    
    def _is_allowed_domain(self, url: str) -> bool:
        """Check if URL domain is allowed."""
        allowed = self.config.get("allowed_domains", [])
        if not allowed:
            return True
        domain = urlparse(url).netloc.lower()
        return any(d in domain for d in allowed)
    
    def _should_skip_url(self, url: str) -> bool:
        """Check if URL matches skip patterns."""
        for pattern in self.config.get("skip_url_patterns", []):
            if re.search(pattern, url):
                return True
        return False
    
    def _is_document_url(self, url: str) -> bool:
        """Check if URL is a document."""
        url_lower = url.lower()
        doc_extensions = [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"]
        for ext in doc_extensions:
            if ext in url_lower:
                return True
        if any(p in url_lower for p in ["/document/download/", "/documents/", 
                                         "/download/", "format=pdf", "format=doc"]):
            return True
        return False
    
    def _sanitize_filename(self, name: str) -> str:
        """Clean filename."""
        name = re.sub(r'[<>:"/\\|?*]', "_", name)
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"_+", "_", name)
        return name.strip("._")[:200]
    
    def _get_unique_path(self, base: str, folder: Path, ext: str) -> Path:
        """Get unique filepath."""
        key = f"{folder}/{base}{ext}"
        if key in self.used_filenames:
            self.used_filenames[key] += 1
            base = f"{base}_{self.used_filenames[key]}"
        else:
            self.used_filenames[key] = 0
        return folder / f"{base}{ext}"
    
    def _get_category_folder(self, category: str, subcategory: str = "") -> str:
        """Map category to folder."""
        if subcategory:
            sub = subcategory.lower()
            if "ich" in sub:
                return "ich_guidelines"
            if "gcp" in sub or "good clinical practice" in sub:
                return "gcp"
        
        mapping = {
            "Clinical Trials": "clinical_trials",
            "Orphan Medicinal Products": "orphan_medicinal_products",
            "Good Clinical Practice": "gcp",
            "ICH Guidelines": "ich_guidelines",
            "Legislation": "legislation",
            "Scientific Guidelines": "scientific_guidelines",
        }
        return mapping.get(category, "other")
    
    def _validate_document(self, content: bytes, url: str, min_size: int = 500) -> Tuple[bool, str, str]:
        """Validate document content and detect type."""
        if len(content) < min_size:
            return False, "", ""
        
        content_start = content[:500].lower()
        html_indicators = [b"<!doctype", b"<html", b"<head", b"<body", b"<meta", 
                          b"<title", b"<script", b"<link rel"]
        if any(ind in content_start for ind in html_indicators):
            return False, "", ""
        
        if content[:4] == b"%PDF" or b"%PDF" in content[:50]:
            return True, ".pdf", "PDF"
        
        if content[:2] == b"PK":
            url_lower = url.lower()
            if ".xlsx" in url_lower or "excel" in url_lower:
                return True, ".xlsx", "Excel"
            elif ".pptx" in url_lower:
                return True, ".pptx", "PowerPoint"
            else:
                return True, ".docx", "Word"
        
        if content[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            url_lower = url.lower()
            if ".xls" in url_lower:
                return True, ".xls", "Excel"
            else:
                return True, ".doc", "Word"
        
        if len(content) > 50000:
            if b"</" in content[:1000]:
                return False, "", ""
            url_lower = url.lower()
            if ".pdf" in url_lower:
                return True, ".pdf", "PDF"
            elif ".docx" in url_lower:
                return True, ".docx", "Word"
            elif ".doc" in url_lower:
                return True, ".doc", "Word"
            elif ".xlsx" in url_lower:
                return True, ".xlsx", "Excel"
            elif ".xls" in url_lower:
                return True, ".xls", "Excel"
            return True, ".pdf", "PDF"
        
        return False, "", ""
    
    def _extract_title(self, url: str) -> str:
        """Extract title from URL."""
        path = urlparse(url).path
        name = unquote(path.split("/")[-1])
        for ext in [".pdf", ".doc", ".docx"]:
            if name.lower().endswith(ext):
                name = name[:-len(ext)]
        name = name.replace("-", " ").replace("_", " ")
        return name.title().strip() or "Unknown Document"
    
    def _determine_subcategory(self, text: str, url: str = "") -> str:
        """Determine subcategory from text."""
        combined = (text + " " + url).lower()
        
        if "gcp" in combined or "good clinical practice" in combined:
            return "Good Clinical Practice"
        if "ich" in combined:
            return "ICH Guidelines"
        if "safety" in combined or "adverse" in combined:
            return "Safety Reporting"
        if "quality" in combined or "gmp" in combined:
            return "Quality / GMP"
        if "orphan" in combined:
            return "Orphan Designation"
        if "first-in-human" in combined:
            return "First-in-Human"
        return "General"
    
    def _should_skip_failed(self, url: str) -> bool:
        """Check if URL failed too many times."""
        if not self.config.get("skip_failed_urls", True):
            return False
        max_fail = self.config.get("max_failures_per_url", 3)
        return self.state.failed_urls.get(url, 0) >= max_fail
    
    def download_document(self, url: str, title: str, category: str,
                         subcategory: str = "", source: str = "",
                         session: Optional[requests.Session] = None) -> Optional[Document]:
        """Download and save a document."""
        
        normalized_url = self._normalize_url(url)
        
        if self._should_skip_failed(url) or self._should_skip_failed(normalized_url):
            return None
        
        safe_title = self._sanitize_filename(title)
        folder = self.pdfs_dir / self._get_category_folder(category, subcategory)
        initial_path = folder / f"{safe_title}.pdf"
        
        if normalized_url in self.state.downloaded_urls or url in self.state.downloaded_urls:
            if self.config.get("skip_existing", True):
                self.logger.info(f"SKIP (exists): {title[:50]}...")
                self.state.stats["skipped"] += 1
                return Document(title=title, url=url, category=category,
                               subcategory=subcategory, source=source,
                               local_path=str(initial_path), status="skipped")
        
        if initial_path.exists() and self.config.get("skip_existing", True):
            self.logger.info(f"SKIP (file exists): {title[:50]}...")
            self.state.stats["skipped"] += 1
            self.state.downloaded_urls.add(url)
            self.state.downloaded_urls.add(normalized_url)
            return Document(title=title, url=url, category=category,
                           subcategory=subcategory, source=source,
                           local_path=str(initial_path), status="skipped")
        
        self.logger.info(f"Downloading: {title[:55]}...")
        
        response = self._make_request(url, session=session)
        if not response:
            self.state.failed_urls[url] = self.state.failed_urls.get(url, 0) + 1
            self.state.stats["failed"] += 1
            self.logger.error(f"FAILED: {title[:50]}")
            return None
        
        content = response.content
        min_size = 5000 if "eur-lex" in url.lower() else 500
        
        is_valid, ext, doc_type = self._validate_document(content, url, min_size)
        
        if not is_valid:
            self.logger.error(f"INVALID DOCUMENT: {title[:50]} ({len(content)} bytes)")
            self.state.failed_urls[url] = self.state.failed_urls.get(url, 0) + 1
            self.state.stats["failed"] += 1
            return None
        
        local_path = self._get_unique_path(safe_title, folder, ext)
        
        try:
            with open(local_path, "wb") as f:
                f.write(content)
        except IOError as e:
            self.logger.error(f"Save failed: {e}")
            self.state.stats["failed"] += 1
            return None
        
        doc = Document(
            title=title, url=url, category=category, subcategory=subcategory,
            source=source, document_type=doc_type,
            date_scraped=datetime.now().isoformat(),
            local_path=str(local_path), file_size=len(content),
            checksum=hashlib.md5(content).hexdigest(), status="downloaded"
        )
        
        self.state.documents.append(doc)
        self.state.downloaded_urls.add(url)
        self.state.downloaded_urls.add(normalized_url)
        self.state.stats["downloaded"] += 1
        self.state.stats["total_bytes"] += len(content)
        
        if url in self.state.failed_urls:
            del self.state.failed_urls[url]
        if normalized_url in self.state.failed_urls:
            del self.state.failed_urls[normalized_url]
        
        self.logger.info(f"SAVED: {local_path.name} ({len(content):,} bytes)")
        return doc
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Tuple[str, str]]:
        """Extract links from page with improved title detection."""
        links = []
        seen_urls = set()
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            
            if not href or href.startswith(("javascript:", "mailto:", "#")):
                continue
            
            full_url = urljoin(base_url, href).split("#")[0]
            
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)
            
            if not full_url.startswith("http"):
                continue
            if not self._is_allowed_domain(full_url):
                continue
            if self._should_skip_url(full_url):
                continue
            
            if text.lower() in ["word", "pdf", "view", "download", "word version", 
                                "pdf version", "read", "read more", "link", ""]:
                better_title = self._get_better_title(a, full_url)
                if better_title:
                    text = better_title
                else:
                    text = self._extract_title(full_url)
            
            links.append((full_url, text or self._extract_title(full_url)))
        
        return links
    
    def _get_better_title(self, link_element, url: str) -> Optional[str]:
        """Try to extract a better title from surrounding context."""
        parent = link_element.parent
        for _ in range(3):
            if parent is None:
                break
            
            prev_sibling = link_element.find_previous_sibling()
            if prev_sibling and prev_sibling.get_text(strip=True):
                text = prev_sibling.get_text(strip=True)
                if len(text) > 5 and len(text) < 200:
                    return text
            
            parent_text = parent.find(string=True, recursive=False)
            if parent_text and len(parent_text.strip()) > 5:
                return parent_text.strip()[:200]
            
            heading = parent.find(['h1', 'h2', 'h3', 'h4', 'h5', 'strong', 'b'])
            if heading:
                heading_text = heading.get_text(strip=True)
                if len(heading_text) > 5:
                    return heading_text[:200]
            
            link_element = parent
            parent = parent.parent
        
        return None
    
    def _crawl_page(self, url: str, category: str, subcategory: str = "",
                   source: str = "", depth: int = 0) -> List[Tuple[str, str, int]]:
        """Crawl page and return discovered links."""
        max_depth = self.config.get("max_crawl_depth", 3)
        
        if depth > max_depth:
            return []
        
        normalized_url = self._normalize_url(url)
        if normalized_url in self.state.visited_urls or url in self.state.visited_urls:
            return []
        
        self.state.visited_urls.add(url)
        self.state.visited_urls.add(normalized_url)
        self.state.stats["pages_crawled"] += 1
        
        self.logger.info(f"Crawling (d={depth}): {url[:65]}...")
        
        response = self._make_request(url)
        if not response:
            return []
        
        time.sleep(self._get_delay(url))
        
        soup = BeautifulSoup(response.text, "html.parser")
        links = self._extract_links(soup, url)
        
        discovered = []
        for link_url, link_text in links:
            if self._is_url_processed(link_url):
                continue
            
            if self._is_document_url(link_url):
                self.download_document(
                    url=link_url,
                    title=link_text or self._extract_title(link_url),
                    category=category,
                    subcategory=self._determine_subcategory(link_text, link_url),
                    source=source,
                )
                time.sleep(self._get_delay(link_url))
            elif self.config.get("include_deep_crawl", True) and depth < max_depth:
                discovered.append((link_url, link_text, depth + 1))
        
        return discovered
    
    # === Source Scrapers ===
    
    def scrape_eudralex_volume10(self):
        """Scrape EudraLex Volume 10."""
        self.logger.info("=" * 60)
        self.logger.info("Scraping EudraLex Volume 10...")
        
        urls = [
            self.SOURCES["eudralex_vol10"]["url"],
            "https://health.ec.europa.eu/medicinal-products/eudralex/eudralex-volume-10/chapter-i-application-and-application-documents_en",
            "https://health.ec.europa.eu/medicinal-products/eudralex/eudralex-volume-10/chapter-ii-safety-reporting_en",
            "https://health.ec.europa.eu/medicinal-products/eudralex/eudralex-volume-10/chapter-iii-quality-investigational-medicinal-products_en",
            "https://health.ec.europa.eu/medicinal-products/eudralex/eudralex-volume-10/chapter-iv-inspections_en",
            "https://health.ec.europa.eu/medicinal-products/eudralex/eudralex-volume-10/chapter-v-additional-documents_en",
        ]
        
        for url in urls:
            if self._shutdown_requested:
                break
            discovered = self._crawl_page(url, "Clinical Trials", "EudraLex Volume 10", "EC Health", 0)
            for link_url, link_text, new_depth in discovered[:20]:
                if self._shutdown_requested:
                    break
                self._crawl_page(link_url, "Clinical Trials",
                               self._determine_subcategory(link_text, link_url), "EC Health", new_depth)
        self._save_state()
    
    def scrape_ec_orphan_page(self):
        """Scrape EC Orphan pages."""
        self.logger.info("=" * 60)
        self.logger.info("Scraping EC Orphan Medicinal Products...")
        
        urls = [
            self.SOURCES["ec_orphan"]["url"],
            "https://health.ec.europa.eu/medicinal-products/orphan-medicinal-products/orphan-incentives_en",
        ]
        
        for url in urls:
            if self._shutdown_requested:
                break
            discovered = self._crawl_page(url, "Orphan Medicinal Products", "EC Portal", "EC Health", 0)
            for link_url, link_text, new_depth in discovered[:15]:
                if self._shutdown_requested:
                    break
                self._crawl_page(link_url, "Orphan Medicinal Products",
                               self._determine_subcategory(link_text, link_url), "EC Health", new_depth)
        self._save_state()
    
    def scrape_ema_orphan_pages(self):
        """Scrape EMA Orphan pages."""
        self.logger.info("=" * 60)
        self.logger.info("Scraping EMA Orphan Designation...")
        
        urls = [
            self.SOURCES["ema_orphan"]["url"],
            self.SOURCES["ema_orphan_applying"]["url"],
        ]
        
        for url in urls:
            if self._shutdown_requested:
                continue
            self._crawl_page(url, "Orphan Medicinal Products", "EMA Guidance", "EMA", 0)
        self._save_state()
    
    def scrape_known_pdfs(self):
        """Download known important PDFs."""
        self.logger.info("=" * 60)
        self.logger.info("Downloading known important guidelines...")
        
        for key, info in self.KNOWN_PDFS.items():
            if self._shutdown_requested:
                break
            
            url = info["url"]
            if url in self.state.downloaded_urls and self.config.get("skip_existing", True):
                self.logger.info(f"SKIP (exists): {info['title'][:50]}...")
                self.state.stats["skipped"] += 1
                continue
            
            self.download_document(url, info["title"], info["category"],
                                  info.get("subcategory", ""), "Direct Link")
            time.sleep(self.config.get("delay_between_requests", 1.5))
        self._save_state()
    
    def scrape_eurlex_legislation(self):
        """Download EUR-Lex legislation."""
        self.logger.info("=" * 60)
        self.logger.info("Downloading EUR-Lex legislation...")
        
        eurlex_session = self._create_eurlex_session()
        
        self.logger.info("Waiting 10s before EUR-Lex...")
        time.sleep(10)
        
        for i, leg in enumerate(self.EURLEX_LEGISLATION):
            if self._shutdown_requested:
                break
            
            url = leg["url"]
            if url in self.state.downloaded_urls and self.config.get("skip_existing", True):
                self.logger.info(f"SKIP (exists): {leg['title'][:50]}...")
                self.state.stats["skipped"] += 1
                continue
            
            self.download_document(url, leg["title"], leg["category"],
                                  leg["subcategory"], "EUR-Lex", eurlex_session)
            
            if i < len(self.EURLEX_LEGISLATION) - 1:
                delay = self.config.get("eurlex_delay", 5.0)
                self.logger.info(f"Waiting {delay}s...")
                time.sleep(delay)
        self._save_state()
    
    def scrape_ema_scientific_guidelines(self):
        """Scrape EMA Scientific Guidelines."""
        self.logger.info("=" * 60)
        self.logger.info("Scraping EMA Scientific Guidelines...")
        
        urls = [
            self.SOURCES["ema_scientific_guidelines"]["url"],
            self.SOURCES.get("ema_ich_guidelines", {}).get("url"),
            self.SOURCES.get("ema_gcp", {}).get("url"),
        ]
        
        for url in urls:
            if not url or self._shutdown_requested:
                continue
            discovered = self._crawl_page(url, "Scientific Guidelines", "Efficacy & Safety", "EMA", 0)
            for link_url, link_text, new_depth in discovered[:10]:
                if self._shutdown_requested:
                    break
                self._crawl_page(link_url, "Scientific Guidelines",
                               self._determine_subcategory(link_text, link_url), "EMA", new_depth)
        self._save_state()
    
    def run_rag_extraction(self):
        """Orchestrate RAG content extraction from HTML pages."""
        self.logger.info("=" * 60)
        self.logger.info("Starting RAG Content Extraction...")
        
        try:
            # Import the RAG extractor module
            import eu_pharma_rag_extractor as rag_extractor
            
            # Configure RAG extraction to use same output directory
            rag_config = rag_extractor.get_default_config()
            rag_config["output_dir"] = str(self.output_dir / self.config.get("rag_output_subdir", "rag_content"))
            rag_config["output_format"] = self.config.get("rag_output_format", "markdown")
            rag_config["force_fresh_start"] = self.config.get("force_fresh_start", False)
            rag_config["skip_existing"] = self.config.get("skip_existing", True)
            
            # Sync source settings
            rag_config["include_eudralex_vol10"] = self.config.get("include_eudralex_vol10", True)
            rag_config["include_ec_orphan"] = self.config.get("include_ec_orphan", True)
            rag_config["include_ema_orphan"] = self.config.get("include_ema_orphan", True)
            rag_config["include_ema_guidelines"] = self.config.get("include_ema_guidelines", True)
            rag_config["include_eurlex_pages"] = self.config.get("include_eurlex", True)
            rag_config["include_deep_crawl"] = self.config.get("include_deep_crawl", True)
            
            # Run extraction
            extractor = rag_extractor.run_extraction(rag_config)
            
            self.logger.info(f"RAG extraction complete: {extractor.state.stats.get('extracted', 0)} pages extracted")
            
        except ImportError as e:
            self.logger.error(f"Could not import eu_pharma_rag_extractor: {e}")
            self.logger.error("Make sure eu_pharma_rag_extractor.py is in the same directory")
        except Exception as e:
            self.logger.error(f"RAG extraction failed: {e}")
    
    def save_metadata(self):
        """Save metadata and catalogs."""
        self.logger.info("Saving metadata...")
        
        all_docs = {doc.url: doc for doc in self.state.documents}
        
        # Full catalog
        with open(self.metadata_dir / "full_catalog.json", "w", encoding="utf-8") as f:
            json.dump([asdict(d) for d in all_docs.values()], f, indent=2, ensure_ascii=False)
        
        # By category
        by_cat: Dict[str, List] = {}
        for doc in all_docs.values():
            by_cat.setdefault(doc.category, []).append(doc)
        
        for cat, docs in by_cat.items():
            path = self.metadata_dir / f"{self._sanitize_filename(cat)}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump([asdict(d) for d in docs], f, indent=2, ensure_ascii=False)
        
        # Summary
        summary = {
            "scrape_date": datetime.now().isoformat(),
            "total_documents": len(all_docs),
            "categories": {c: len(d) for c, d in by_cat.items()},
            "stats": self.state.stats,
            "failed_urls": len(self.state.failed_urls),
        }
        with open(self.metadata_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        self._save_state()
        return all_docs
    
    def print_summary(self):
        """Print final summary."""
        s = self.state.stats
        print("\n" + "=" * 60)
        print("SCRAPE COMPLETE")
        print("=" * 60)
        print(f"  Downloaded:    {s.get('downloaded', 0):>5}")
        print(f"  Skipped:       {s.get('skipped', 0):>5}")
        print(f"  Failed:        {s.get('failed', 0):>5}")
        print(f"  Pages Crawled: {s.get('pages_crawled', 0):>5}")
        print(f"  Total Size:    {s.get('total_bytes', 0)/1024/1024:.2f} MB")
        print(f"  Output:        {self.output_dir.absolute()}")
        print("=" * 60 + "\n")
    
    def run(self):
        """Run the scraper."""
        self.logger.info("=" * 60)
        self.logger.info("EU Pharmaceutical Guidelines Scraper v7")
        self.logger.info(f"Output: {self.output_dir.absolute()}")
        self.logger.info(f"Skip existing: {self.config.get('skip_existing')}")
        self.logger.info(f"Deep crawl: {self.config.get('include_deep_crawl')}")
        self.logger.info(f"RAG extraction: {self.config.get('include_rag_extraction')}")
        self.logger.info("=" * 60)
        
        try:
            # === PHASE 1: Download PDFs ===
            if self.config.get("include_eudralex_vol10", True):
                self.scrape_eudralex_volume10()
                if self._shutdown_requested:
                    raise KeyboardInterrupt
            
            if self.config.get("include_ec_orphan", True):
                self.scrape_ec_orphan_page()
                if self._shutdown_requested:
                    raise KeyboardInterrupt
            
            if self.config.get("include_ema_orphan", True):
                self.scrape_ema_orphan_pages()
                if self._shutdown_requested:
                    raise KeyboardInterrupt
            
            if self.config.get("include_known_pdfs", True):
                self.scrape_known_pdfs()
                if self._shutdown_requested:
                    raise KeyboardInterrupt
            
            if self.config.get("include_eurlex", True):
                self.scrape_eurlex_legislation()
                if self._shutdown_requested:
                    raise KeyboardInterrupt
            
            if self.config.get("include_ema_guidelines", True):
                self.scrape_ema_scientific_guidelines()
                if self._shutdown_requested:
                    raise KeyboardInterrupt
            
            # === PHASE 2: RAG Content Extraction ===
            if self.config.get("include_rag_extraction", True):
                self.run_rag_extraction()
            
            self.save_metadata()
            self.print_summary()
            
        except KeyboardInterrupt:
            self.logger.info("\nInterrupted - saving progress...")
            self.save_metadata()
            self.print_summary()
            self.logger.info("Run again to resume from where you left off.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║       EU Pharmaceutical Guidelines Scraper v7                ║
║                                                              ║
║  Sources: EudraLex, EMA, EUR-Lex                             ║
║  Topics:  Clinical Trials, Orphan Drugs, GCP, ICH            ║
║                                                              ║
║  Output:                                                     ║
║    /pdfs/        → PDF guideline files                       ║
║    /rag_content/ → Markdown for RAG (from HTML pages)        ║
║                                                              ║
║  Edit CONFIG at top of file to customize                     ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    scraper = EUGuidelinesScraper(CONFIG)
    scraper.run()


if __name__ == "__main__":
    main()