#!/usr/bin/env python3
"""
EU Pharmaceutical Guidelines - RAG Content Extractor v4.0
==========================================================
IMPROVEMENTS OVER v3.0:
- FIXED: EMA pages now use Playwright for JavaScript rendering
- KEEPS: Simple HTTP requests for EC and EUR-Lex (they work fine)
- Added: Hybrid approach - Playwright only when needed

REQUIREMENTS:
    pip install playwright
    playwright install chromium

This script extracts clean, structured content from EU pharma regulatory pages for RAG systems.

Sources:
- EudraLex Volume 10 (Clinical Trials) - HTTP
- EC Orphan Medicinal Products Portal - HTTP  
- EMA Scientific Guidelines - PLAYWRIGHT
- EMA Orphan Designation - PLAYWRIGHT
- EMA Clinical Trials - PLAYWRIGHT
- EUR-Lex Legislation pages - HTTP

Usage:
    python eu_pharma_rag_extractor_v4.py

Version: 4.0
"""

import os
import re
import json
import time
import hashlib
import logging
import signal
import asyncio
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Set, Tuple, Any
from urllib.parse import urljoin, urlparse, unquote, parse_qs
from collections import deque

import warnings
import requests
from bs4 import BeautifulSoup, Comment

# Suppress XML parsed as HTML warning (EUR-Lex sometimes returns XML)
try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    pass  # Older BeautifulSoup versions don't have this warning

# Playwright - only imported when needed for EMA
try:
    from playwright.sync_api import sync_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Output Settings
    "output_dir": "rag_content",
    "log_file": "rag_extractor_v4.log",
    "log_level": "INFO",
    
    # What to Scrape
    "include_eudralex_vol10": True,
    "include_ec_orphan": True,
    "include_ema_guidelines": True,
    "include_ema_orphan": True,
    "include_ema_clinical_trials": True,
    "include_ec_clinical_trials": True,
    "include_eurlex_pages": True,
    "include_deep_crawl": True,
    
    # Content Settings
    "min_content_length": 200,
    "min_word_count": 50,
    "max_summary_length": 500,
    "min_paragraph_length": 20,
    
    # Language filtering
    "english_only": True,
    "dedupe_by_content": True,
    
    # Resume Settings
    "skip_existing": False,
    
    # Network Settings
    "request_timeout": 45,
    "delay_between_requests": 1.5,
    "ema_delay": 2.0,  # Can be shorter with Playwright
    "eurlex_delay": 3.0,
    "max_retries": 3,
    "backoff_base": 2.0,
    
    # Playwright Settings (for EMA)
    "playwright_timeout": 30000,  # 30 seconds
    "playwright_wait_for": "networkidle",  # Wait for network to be idle
    
    # Crawl Settings
    "max_depth": 2,
    "max_pages_per_source": 100,
}

# Language codes to filter
EU_LANGUAGE_CODES = ["bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "ga", 
                     "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", 
                     "sk", "sl", "sv"]


# =============================================================================
# SOURCE DEFINITIONS
# =============================================================================

SOURCES = {
    # EC Health Portal - works fine with HTTP
    "eudralex_vol10": {
        "urls": [
            "https://health.ec.europa.eu/medicinal-products/eudralex_en",
        ],
        "follow_links": True,
        "link_patterns": [r"/eudralex", r"eudralex-volume", r"/clinical-trials", r"/medicinal-products"],
        "use_playwright": False,
    },
    
    "ec_orphan": {
        "urls": [
            "https://health.ec.europa.eu/medicinal-products/orphan-medicinal-products_en",
        ],
        "follow_links": True,
        "link_patterns": [r"/orphan", r"/medicinal-products"],
        "use_playwright": False,
    },
    
    "ec_clinical_trials": {
        "urls": [
            "https://health.ec.europa.eu/medicinal-products/clinical-trials/clinical-trials-regulation-eu-no-5362014_en",
        ],
        "follow_links": True,
        "link_patterns": [r"/clinical-trials"],
        "use_playwright": False,
    },
    
    # EMA - REQUIRES PLAYWRIGHT
    "ema_orphan": {
        "urls": [
            "https://www.ema.europa.eu/en/human-regulatory-overview/orphan-designation-overview",
            "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/orphan-designation-research-development",
            "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/orphan-designation-research-development/applying-orphan-designation",
            "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/orphan-designation-research-development/orphan-incentives",
            "https://www.ema.europa.eu/en/human-regulatory-overview/marketing-authorisation/orphan-designation-marketing-authorisation",
            "https://www.ema.europa.eu/en/human-regulatory-overview/orphan-designation-overview/legal-framework-orphan-designation",
            "https://www.ema.europa.eu/en/committees/committee-orphan-medicinal-products-comp",
        ],
        "follow_links": True,
        "link_patterns": [r"/orphan", r"orphan-designation", r"/comp"],
        "use_playwright": True,  # EMA needs JavaScript
    },
    
    "ema_guidelines": {
        "urls": [
            "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/scientific-guidelines",
            "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/scientific-guidelines/clinical-efficacy-safety-guidelines",
            "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/scientific-guidelines/clinical-efficacy-safety-guidelines/general-considerations",
            "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/scientific-guidelines/clinical-efficacy-safety-guidelines/biostatistics",
            "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/scientific-guidelines/clinical-efficacy-safety-guidelines/design-conduct-reporting-clinical-trials",
            "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/compliance-research-development/good-clinical-practice",
        ],
        "follow_links": True,
        "link_patterns": [r"/scientific-guidelines", r"/clinical-efficacy", r"/good-clinical-practice", r"/biostatistics"],
        "use_playwright": True,
    },
    
    "ema_clinical_trials": {
        "urls": [
            "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/clinical-trials-human-medicines",
            "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/clinical-trials-human-medicines/clinical-trial-regulation",
            "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/clinical-trials-human-medicines/clinical-trials-information-system-ctis",
        ],
        "follow_links": True,
        "link_patterns": [r"/clinical-trials", r"/ctis"],
        "use_playwright": True,
    },
    
    # EUR-Lex - works fine with HTTP
    "eurlex_info": {
        "urls": [
            "https://eur-lex.europa.eu/eli/reg/2014/536/oj",  # CTR
            "https://eur-lex.europa.eu/eli/reg/2000/141/oj",  # Orphan Regulation
        ],
        "follow_links": True,
        "link_patterns": [r"/eli/", r"/legal-content/"],
        "use_playwright": False,
    },
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExtractedPage:
    url: str
    title: str
    content: str
    description: str = ""
    source: str = ""
    headings: List[str] = field(default_factory=list)
    word_count: int = 0
    extracted_at: str = ""
    content_hash: str = ""

@dataclass 
class ExtractorState:
    visited_urls: Set[str] = field(default_factory=set)
    extracted_pages: List[str] = field(default_factory=list)
    content_hashes: Set[str] = field(default_factory=set)
    last_run: str = ""
    stats: Dict[str, int] = field(default_factory=lambda: {
        "extracted": 0,
        "skipped": 0,
        "failed": 0,
        "duplicates": 0,
    })
    
    def to_dict(self):
        return {
            "visited_urls": list(self.visited_urls),
            "extracted_pages": self.extracted_pages,
            "content_hashes": list(self.content_hashes),
            "last_run": self.last_run,
            "stats": self.stats,
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            visited_urls=set(data.get("visited_urls", [])),
            extracted_pages=data.get("extracted_pages", []),
            content_hashes=set(data.get("content_hashes", [])),
            last_run=data.get("last_run", ""),
            stats=data.get("stats", {}),
        )


# =============================================================================
# MAIN EXTRACTOR CLASS
# =============================================================================

class EUPharmaRAGExtractor:
    """Hybrid extractor using Playwright for EMA, HTTP for others."""
    
    def __init__(self, config: dict = None):
        self.config = config or CONFIG.copy()
        
        # Output directories
        self.output_dir = Path(self.config.get("output_dir", "rag_content"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.output_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Setup
        self._setup_logging()
        self.session = self._create_session()
        self.state = self._load_state()
        
        # Playwright browser (lazy init)
        self._browser = None
        self._playwright = None
        
        self._shutdown = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        self.logger.info(f"Initialized. Output: {self.output_dir.absolute()}")
    
    def _handle_signal(self, signum, frame):
        self.logger.warning(f"Signal {signum} received, shutting down...")
        self._shutdown = True
        self._cleanup_playwright()
    
    def _setup_logging(self):
        """Configure logging."""
        log_level = getattr(logging, self.config.get("log_level", "INFO").upper())
        
        self.logger = logging.getLogger("RAGExtractorV4")
        self.logger.setLevel(log_level)
        self.logger.handlers = []
        
        # Console
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(ch)
        
        # File
        log_path = self.output_dir / self.config.get("log_file", "extractor.log")
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(fh)
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with proper headers."""
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        })
        return session
    
    # =========================================================================
    # PLAYWRIGHT MANAGEMENT
    # =========================================================================
    
    def _init_playwright(self):
        """Initialize Playwright browser (lazy)."""
        if not PLAYWRIGHT_AVAILABLE:
            self.logger.error("Playwright not installed! Run: pip install playwright && playwright install chromium")
            return False
        
        if self._browser is None:
            self.logger.info("Starting Playwright browser...")
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                ]
            )
            self.logger.info("Playwright browser ready")
        return True
    
    def _cleanup_playwright(self):
        """Clean up Playwright resources."""
        if self._browser:
            try:
                self._browser.close()
            except:
                pass
            self._browser = None
        if self._playwright:
            try:
                self._playwright.stop()
            except:
                pass
            self._playwright = None
    
    def _fetch_with_playwright(self, url: str) -> Optional[str]:
        """Fetch page using Playwright (for JavaScript-heavy sites like EMA)."""
        if not self._init_playwright():
            return None
        
        try:
            context = self._browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
            )
            page = context.new_page()
            
            # Navigate and wait for content
            timeout = self.config.get("playwright_timeout", 30000)
            page.goto(url, timeout=timeout, wait_until="domcontentloaded")
            
            # Wait for dynamic content to load
            try:
                # Wait for main content area to appear
                page.wait_for_selector("article, main, [role='main'], .content", timeout=10000)
            except:
                pass
            
            # Extra wait for JavaScript rendering
            page.wait_for_timeout(2000)  # 2 seconds for JS to render
            
            # Get the rendered HTML
            html = page.content()
            
            context.close()
            return html
            
        except Exception as e:
            self.logger.warning(f"Playwright error for {url[:60]}: {e}")
            return None
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def _load_state(self) -> ExtractorState:
        """Load previous state."""
        state_path = self.metadata_dir / "extractor_state_v4.json"
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    return ExtractorState.from_dict(json.load(f))
            except Exception as e:
                self.logger.warning(f"Could not load state: {e}")
        return ExtractorState()
    
    def _save_state(self):
        """Save current state."""
        state_path = self.metadata_dir / "extractor_state_v4.json"
        self.state.last_run = datetime.now().isoformat()
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, indent=2)
    
    # =========================================================================
    # URL HANDLING
    # =========================================================================
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison."""
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        return f"{parsed.scheme}://{parsed.netloc}{path}"
    
    def _is_non_english_url(self, url: str) -> bool:
        """Check if URL is non-English language variant."""
        if not self.config.get("english_only", True):
            return False
        
        url_lower = url.lower()
        for lang in EU_LANGUAGE_CODES:
            if f"/{lang}/" in url_lower and f"/en/" not in url_lower:
                return True
            if url_lower.endswith(f"_{lang}"):
                return True
            if f"_{lang}/" in url_lower or f"_{lang}?" in url_lower:
                return True
        return False
    
    def _get_delay(self, url: str, use_playwright: bool) -> float:
        """Get appropriate delay for URL."""
        domain = urlparse(url).netloc.lower()
        if "ema.europa.eu" in domain:
            return self.config.get("ema_delay", 2.0)
        if "eur-lex.europa.eu" in domain:
            return self.config.get("eurlex_delay", 3.0)
        return self.config.get("delay_between_requests", 1.5)
    
    # =========================================================================
    # HYBRID FETCHING
    # =========================================================================
    
    def _fetch_page(self, url: str, use_playwright: bool = False) -> Optional[str]:
        """Fetch page - using Playwright for EMA, HTTP for others."""
        
        # Determine if we need Playwright
        domain = urlparse(url).netloc.lower()
        needs_playwright = use_playwright or "ema.europa.eu" in domain
        
        max_retries = self.config.get("max_retries", 3)
        
        for attempt in range(max_retries):
            try:
                delay = self._get_delay(url, needs_playwright)
                time.sleep(delay)
                
                if needs_playwright:
                    # Use Playwright for EMA
                    html = self._fetch_with_playwright(url)
                    if html:
                        return html
                else:
                    # Use simple HTTP for EC/EUR-Lex
                    timeout = self.config.get("request_timeout", 45)
                    response = self.session.get(url, timeout=timeout, allow_redirects=True)
                    
                    if response.status_code == 200:
                        return response.text
                    elif response.status_code == 404:
                        self.logger.warning(f"404: {url[:70]}")
                        return None
                    elif response.status_code == 403:
                        self.logger.warning(f"403 Forbidden: {url[:70]}")
                        time.sleep(5)
                        continue
                    else:
                        self.logger.warning(f"HTTP {response.status_code}: {url[:70]}")
                        
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout (attempt {attempt+1}): {url[:60]}")
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request error (attempt {attempt+1}): {e}")
            except Exception as e:
                self.logger.warning(f"Fetch error (attempt {attempt+1}): {e}")
            
            # Backoff
            if attempt < max_retries - 1:
                wait = self.config.get("backoff_base", 2.0) ** attempt
                time.sleep(wait)
        
        return None
    
    # =========================================================================
    # CONTENT EXTRACTION
    # =========================================================================
    
    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Extract page title."""
        h1 = soup.find("h1")
        if h1:
            text = h1.get_text(strip=True)
            if text and len(text) > 3:
                return text[:200]
        
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()[:200]
        
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
            for sep in [" | ", " - ", " :: ", " – "]:
                if sep in title:
                    return title.split(sep)[0].strip()[:200]
            return title[:200]
        
        path = urlparse(url).path
        segments = [s for s in path.split("/") if s and s != "en"]
        if segments:
            return unquote(segments[-1].replace("-", " ").replace("_", " ")).title()
        
        return "Untitled Page"
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description."""
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            return meta["content"].strip()
        
        og = soup.find("meta", property="og:description")
        if og and og.get("content"):
            return og["content"].strip()
        
        return ""
    
    def _clean_soup(self, soup: BeautifulSoup) -> None:
        """Remove non-content elements."""
        for tag in soup.find_all(["script", "style", "noscript", "iframe"]):
            tag.decompose()
        
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()
        
        for selector in ["nav", "header", "footer", ".nav", ".menu", ".breadcrumb",
                         ".cookie", ".banner", ".sidebar", "aside", ".footer",
                         ".header", "#header", "#footer", "#nav", ".navigation",
                         ".ecl-site-header", ".ecl-site-footer", ".ecl-page-header"]:
            for tag in soup.select(selector):
                tag.decompose()
    
    def _find_main_content(self, soup: BeautifulSoup, url: str) -> Optional[BeautifulSoup]:
        """Find main content area - site-specific selectors."""
        domain = urlparse(url).netloc.lower()
        
        # =================================================================
        # EMA - Now with Playwright, we get the full rendered DOM
        # =================================================================
        if "ema.europa.eu" in domain:
            ema_selectors = [
                "article.node",
                "article",
                ".ecl-page-content",
                ".field--name-body",
                "[role='main']",
                "main",
                "#content",
                ".content",
            ]
            for selector in ema_selectors:
                main = soup.select_one(selector)
                if main and len(main.get_text(strip=True)) > 200:
                    return main
            
            # Fallback: find div with most paragraph content
            best_div = None
            best_score = 0
            for div in soup.find_all(["div", "section", "article"]):
                paragraphs = div.find_all("p")
                text_len = len(div.get_text(strip=True))
                score = len(paragraphs) * 100 + text_len
                if score > best_score:
                    best_score = score
                    best_div = div
            if best_div and best_score > 500:
                return best_div
        
        # =================================================================
        # EC Health Portal
        # =================================================================
        elif "health.ec.europa.eu" in domain:
            ec_selectors = ["main", "article", ".ecl-page-content", "#content", ".content"]
            for selector in ec_selectors:
                main = soup.select_one(selector)
                if main:
                    return main
        
        # =================================================================
        # EUR-Lex
        # =================================================================
        elif "eur-lex.europa.eu" in domain:
            eurlex_selectors = ["#document", ".eli-main-content", "#TexteOnly", ".content"]
            for selector in eurlex_selectors:
                main = soup.select_one(selector)
                if main:
                    return main
        
        # Generic fallback
        generic_selectors = ["main", "article", "#content", ".content", "[role='main']"]
        for selector in generic_selectors:
            main = soup.select_one(selector)
            if main:
                return main
        
        return soup.body if soup.body else soup
    
    def _extract_content(self, soup: BeautifulSoup, url: str) -> Tuple[str, List[str]]:
        """Extract main content as markdown."""
        self._clean_soup(soup)
        
        main = self._find_main_content(soup, url)
        if not main:
            return "", []
        
        lines = []
        headings = []
        seen_text = set()
        
        for element in main.descendants:
            if element.name is None:
                continue
            
            # Headings
            if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                text = element.get_text(strip=True)
                if text and text not in seen_text:
                    seen_text.add(text)
                    level = int(element.name[1])
                    lines.append("")
                    lines.append(f"{'#' * level} {text}")
                    lines.append("")
                    headings.append(text)
            
            # Paragraphs
            elif element.name == "p":
                text = element.get_text(strip=True)
                min_len = self.config.get("min_paragraph_length", 20)
                if text and len(text) >= min_len and text not in seen_text:
                    seen_text.add(text)
                    lines.append(text)
                    lines.append("")
            
            # Lists
            elif element.name == "li":
                if element.parent and element.parent.name in ["ul", "ol"]:
                    text = element.get_text(strip=True)
                    if text and text not in seen_text:
                        seen_text.add(text)
                        prefix = "-" if element.parent.name == "ul" else "1."
                        lines.append(f"{prefix} {text}")
        
        content = "\n".join(lines)
        content = re.sub(r"\n{3,}", "\n\n", content)
        
        return content.strip(), headings
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str, patterns: List[str]) -> List[str]:
        """Extract relevant links from page."""
        links = []
        base_domain = urlparse(base_url).netloc.lower()
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            
            # Skip anchors, javascript, etc.
            if href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
                continue
            
            # Resolve relative URLs
            full_url = urljoin(base_url, href)
            
            # Must be same domain
            url_domain = urlparse(full_url).netloc.lower()
            if url_domain != base_domain:
                continue
            
            # Skip non-English
            if self._is_non_english_url(full_url):
                continue
            
            # Skip PDFs and downloads
            if any(ext in full_url.lower() for ext in [".pdf", ".doc", ".xls", "/download/"]):
                continue
            
            # Check if URL matches any pattern
            if patterns:
                if not any(re.search(p, full_url, re.I) for p in patterns):
                    continue
            
            normalized = self._normalize_url(full_url)
            if normalized not in self.state.visited_urls:
                links.append(full_url)
        
        return links
    
    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================
    
    def _content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        # Use first 5000 chars for hashing
        return hashlib.md5(content[:5000].encode()).hexdigest()
    
    def _generate_filename(self, title: str, url: str) -> str:
        """Generate safe filename."""
        # Determine prefix
        domain = urlparse(url).netloc.lower()
        if "ema.europa.eu" in domain:
            prefix = "EMA"
        elif "health.ec.europa.eu" in domain:
            prefix = "EC"
        elif "eur-lex.europa.eu" in domain:
            prefix = "EURLEX"
        else:
            prefix = "EU"
        
        # Clean title
        safe_title = re.sub(r"[^\w\s-]", "", title)
        safe_title = re.sub(r"\s+", "_", safe_title)
        safe_title = safe_title[:80]
        
        # Add hash for uniqueness
        url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
        
        return f"{prefix}_{safe_title}_{url_hash}.md"
    
    def _save_page(self, page: ExtractedPage) -> bool:
        """Save extracted page to markdown file."""
        filename = self._generate_filename(page.title, page.url)
        filepath = self.output_dir / filename
        
        # Check for duplicate content
        if self.config.get("dedupe_by_content", True):
            if page.content_hash in self.state.content_hashes:
                self.logger.info(f"Duplicate content, skipping: {page.url[:60]}")
                self.state.stats["duplicates"] = self.state.stats.get("duplicates", 0) + 1
                return False
            self.state.content_hashes.add(page.content_hash)
        
        # Create YAML frontmatter
        frontmatter = f"""---
title: "{page.title}"
source: "{page.source}"
url: "{page.url}"
extracted_at: "{page.extracted_at}"
word_count: {page.word_count}
---

"""
        
        # Write file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(frontmatter)
            f.write(f"# {page.title}\n\n")
            if page.description:
                f.write(f"*{page.description}*\n\n")
            f.write(page.content)
        
        self.state.extracted_pages.append(str(filepath))
        self.state.stats["extracted"] = self.state.stats.get("extracted", 0) + 1
        self.logger.info(f"SAVED: {filename}")
        
        return True
    
    # =========================================================================
    # MAIN EXTRACTION LOOP
    # =========================================================================
    
    def _extract_single_url(self, url: str, source_name: str, use_playwright: bool = False) -> Optional[ExtractedPage]:
        """Extract content from a single URL."""
        # Fetch page
        html = self._fetch_page(url, use_playwright)
        if not html:
            self.state.stats["failed"] = self.state.stats.get("failed", 0) + 1
            return None
        
        # Parse HTML - use html.parser as primary (more forgiving with malformed content)
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:
            self.logger.error(f"Parse error for {url}: {e}")
            return None
        
        # Extract content
        title = self._extract_title(soup, url)
        description = self._extract_description(soup)
        content, headings = self._extract_content(soup, url)
        
        # Validate
        min_length = self.config.get("min_content_length", 200)
        min_words = self.config.get("min_word_count", 50)
        
        if len(content) < min_length:
            self.logger.warning(f"Content too short ({len(content)} chars): {url[:50]}")
            return None
        
        word_count = len(content.split())
        if word_count < min_words:
            self.logger.warning(f"Too few words ({word_count}): {url[:50]}")
            return None
        
        return ExtractedPage(
            url=url,
            title=title,
            content=content,
            description=description,
            source=source_name,
            headings=headings,
            word_count=word_count,
            extracted_at=datetime.now().isoformat(),
            content_hash=self._content_hash(content),
        )
    
    def _crawl_source(self, source_name: str, source_config: dict) -> int:
        """Crawl a single source."""
        self.logger.info("=" * 60)
        self.logger.info(f"Crawling source: {source_name}")
        
        urls = source_config.get("urls", [])
        follow_links = source_config.get("follow_links", True)
        patterns = source_config.get("link_patterns", [])
        use_playwright = source_config.get("use_playwright", False)
        max_pages = self.config.get("max_pages_per_source", 100)
        max_depth = self.config.get("max_depth", 2)
        
        if use_playwright:
            self.logger.info(f"Using Playwright for {source_name}")
        
        # Queue: (url, depth)
        queue = deque([(url, 0) for url in urls])
        source_visited = set()
        extracted_count = 0
        
        while queue and not self._shutdown and extracted_count < max_pages:
            url, depth = queue.popleft()
            
            # Skip if visited
            normalized = self._normalize_url(url)
            if normalized in self.state.visited_urls or normalized in source_visited:
                continue
            
            source_visited.add(normalized)
            self.state.visited_urls.add(normalized)
            
            # Skip non-English
            if self._is_non_english_url(url):
                continue
            
            self.logger.info(f"Extracting: {url[:80]}...")
            
            # Fetch and extract
            html = self._fetch_page(url, use_playwright)
            if not html:
                self.state.stats["failed"] = self.state.stats.get("failed", 0) + 1
                continue
            
            # Extract content
            page = self._extract_single_url(url, source_name, use_playwright)
            if page:
                if self._save_page(page):
                    extracted_count += 1
            
            # Follow links if within depth
            if follow_links and depth < max_depth and html:
                try:
                    # Use html.parser instead of lxml to avoid XML parsing errors
                    soup = BeautifulSoup(html, "html.parser")
                    new_links = self._extract_links(soup, url, patterns)
                    for link in new_links[:20]:  # Limit links per page
                        queue.append((link, depth + 1))
                except Exception as e:
                    self.logger.warning(f"Failed to parse links from {url}: {e}")
        
        self.logger.info(f"Source {source_name}: extracted {extracted_count} pages")
        return extracted_count
    
    def run(self):
        """Run the extraction."""
        self.logger.info("=" * 60)
        self.logger.info("EU Pharma RAG Content Extractor v4.0")
        self.logger.info(f"Output: {self.output_dir.absolute()}")
        self.logger.info(f"English only: {self.config.get('english_only', True)}")
        self.logger.info(f"Content dedup: {self.config.get('dedupe_by_content', True)}")
        self.logger.info(f"Playwright available: {PLAYWRIGHT_AVAILABLE}")
        self.logger.info("=" * 60)
        
        if not PLAYWRIGHT_AVAILABLE:
            self.logger.warning("Playwright not installed! EMA pages may not extract correctly.")
            self.logger.warning("Install with: pip install playwright && playwright install chromium")
        
        try:
            # Process each source
            for source_name, source_config in SOURCES.items():
                if self._shutdown:
                    break
                
                # Check if source is enabled
                config_key = f"include_{source_name}"
                if not self.config.get(config_key, True):
                    self.logger.info(f"Skipping {source_name} (disabled)")
                    continue
                
                self._crawl_source(source_name, source_config)
                self._save_state()
            
        finally:
            self._cleanup_playwright()
            self._save_state()
            
            # Summary
            self.logger.info("=" * 60)
            self.logger.info("EXTRACTION COMPLETE")
            self.logger.info(f"  Extracted: {self.state.stats.get('extracted', 0)}")
            self.logger.info(f"  Duplicates: {self.state.stats.get('duplicates', 0)}")
            self.logger.info(f"  Failed: {self.state.stats.get('failed', 0)}")
            self.logger.info(f"  Output: {self.output_dir.absolute()}")
            self.logger.info("=" * 60)


# =============================================================================
# COMPATIBILITY FUNCTIONS (for main scraper integration)
# =============================================================================

def get_default_config() -> dict:
    """Return default configuration - for compatibility with main scraper."""
    return CONFIG.copy()


def run_extraction(config: dict = None) -> EUPharmaRAGExtractor:
    """Run extraction and return extractor instance - for compatibility with main scraper."""
    if config is None:
        config = CONFIG.copy()
    
    extractor = EUPharmaRAGExtractor(config)
    extractor.run()
    return extractor


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║       EU Pharma RAG Content Extractor v4.0                   ║
║                                                              ║
║  NEW: Playwright for EMA (JavaScript rendering)              ║
║  KEEPS: Simple HTTP for EC/EUR-Lex (fast and reliable)       ║
║                                                              ║
║  REQUIREMENTS:                                               ║
║    pip install playwright                                    ║
║    playwright install chromium                               ║
║                                                              ║
║  Sources: EC Health, EMA (Playwright), EUR-Lex               ║
║  Output:  Markdown files for RAG systems                     ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    extractor = EUPharmaRAGExtractor(CONFIG)
    extractor.run()


if __name__ == "__main__":
    main()