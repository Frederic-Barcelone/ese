#!/usr/bin/env python3
"""
PDF Extractors Module - Enhanced Biomedical Text Processing
===========================================================
Enhanced PDF extraction with intelligent abbreviation protection,
paragraph reflow, and optimized text cleaning for biomedical NER.
"""

import re
import time
import os
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from collections import defaultdict
from functools import lru_cache

# ============================================================================
# Logging Configuration
# ============================================================================
try:
    from corpus_metadata.document_utils.metadata_logging_config import (
        get_logger, log_separator, log_metric
    )
    logger = get_logger('reader_pdf_extractors')
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('reader_pdf_extractors')
    
    def log_separator(logger, style='minor'):
        pass
    
    def log_metric(logger, name, value, unit=''):
        logger.debug(f"{name}: {value}{unit}")

# ============================================================================
# Optional Dependencies
# ============================================================================
try:
    import wordsegment
    wordsegment.load()
    WORDSEGMENT_AVAILABLE = True
    logger.debug("wordsegment loaded for intelligent word splitting")
except ImportError:
    WORDSEGMENT_AVAILABLE = False
    logger.debug("wordsegment not available")

try:
    import enchant
    SPELL_CHECKER = enchant.Dict("en_US")
    ENCHANT_AVAILABLE = True
    logger.debug("pyenchant loaded for spell checking")
except:
    SPELL_CHECKER = None
    ENCHANT_AVAILABLE = False
    logger.debug("pyenchant not available")

# ============================================================================
# Configuration Loading
# ============================================================================
# Removed config file loading - using built-in defaults directly
CONFIG = {}
ALLOWLIST_SHORT = {"FDA", "NIH", "EMA", "RCT", "ICU"}
ALIAS_MAP = {}

# ============================================================================
# Constants and Patterns
# ============================================================================

# Sentinel markers for abbreviation protection
SENT_L = "\uE000"
SENT_R = "\uE001"
SENTINEL_RE = re.compile(rf"{re.escape(SENT_L)}(\d+){re.escape(SENT_R)}")

# Hyphen variants
HYPH = r"[-\u2010\u2011\u2012\u2013\u2014\u2212]"

# Common words to exclude from abbreviation detection
COMMON_WORDS = {
    'clinical', 'pediatric', 'adult', 'mild', 'cut', 'specific',
    'limited', 'free', 'high', 'risk', 'long', 'term', 'end',
    'stage', 'treatment', 'patient', 'years', 'based', 'months',
    'rates', 'adults', 'renal', 'grade', 'point', 'guidance',
    'minimal', 'invasive', 'lumbar', 'decompression', 'clearance',
    'updates', 'primarily', 'extrapolating', 'recommendations'
}

# Build pattern for common words exclusion
COMMON_WORDS_UP = [w.upper() for w in COMMON_WORDS]
COMMON_WORDS_PAT = r"(?:%s)\b" % "|".join(
    map(re.escape, sorted(COMMON_WORDS_UP, key=len, reverse=True))
)

# Core abbreviation pattern
ABBR_CORE = rf"""
(?:
  (?!{COMMON_WORDS_PAT})
  [A-Z]{{2,}}(?:\d+[^\W_]*)?
| [A-Z]\d+[A-Za-z]*R?\d*
| [A-Z][A-Z0-9]*(?:{HYPH}|/)[A-Z][A-Z0-9]*
| SARS\s*{HYPH}?\s*CoV\s*{HYPH}?\s*2
| COVID\s*{HYPH}?\s*19
)
"""

# Abbreviation pattern with suffix support
ABBR_WITH_SUFFIX = re.compile(
    rf"""
    (?<![\w/])
    (?P<abbr>{ABBR_CORE})
    (?P<sfx>(?:{HYPH}(?:[a-z]{{2,}}|[A-Z][a-z]+|\d+))*)
    (?![\w-])
    """,
    re.VERBOSE
)

# Allowlist pattern
if ALLOWLIST_SHORT:
    ALLOW_RE = re.compile(
        r'(?<![\w/])(' + '|'.join(
            map(re.escape, sorted(ALLOWLIST_SHORT, reverse=True))
        ) + r')(?![\w-])'
    )
else:
    ALLOW_RE = None

# Section headings
SECTION_HEADINGS = {
    "ABSTRACT", "INTRODUCTION", "METHODS", "RESULTS", 
    "DISCUSSION", "CONCLUSIONS", "REFERENCES", "BACKGROUND"
}

# Pattern for sentence endings
SENT_PUNCT_RE = re.compile(r'[.!?:"")\]]\s*$')

# Patterns for units and compounds
UNIT_PAT = re.compile(
    r'(?i)\b\d+(?:\.\d+)?\s?'
    r'(?:mg|g|kg|μg|mcg|ml|l|mmol|mol|mM|μM|nM|pmol|IU|U|'
    r'year|yr|h|hr|min|s|day)(?:/[A-Za-z]+)?\b'
)

COMPOUND_KEEP = re.compile(
    r'\b(?:proof-of-concept|steroid-sparing|post-kidney|'
    r'double-blind|case-control|placebo-controlled)\b', 
    re.I
)

# Intra-abbreviation spaces
INTRA_ABBR_CLUSTER = re.compile(r'\b(?:[A-Za-z0-9]\s+){1,6}[A-Za-z0-9]\b')

# Verb pattern for filtering expansions
VERB_RE = re.compile(
    r'\b(accepts?|have|has|are|is|were|was|triggered|emerged|treated|'
    r'associated|induced|formed?|causes?|leads?|results?|shows?|'
    r'achieving|months|years|evidence|with)\b', 
    re.I
)

# ============================================================================
# Text Processing Functions
# ============================================================================

def pre_normalize(text: str) -> str:
    """Pre-normalize text to fix common PDF extraction issues"""
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    
    # Normalize dashes to ASCII hyphen
    text = re.sub(r'[\u2010-\u2015\u2212]', '-', text)
    
    # Micro sign to Greek mu
    text = re.sub(r'\u00B5', 'μ', text)
    
    # Handle line breaks in hyphenated words
    # Case 1: lowercase-newline-lowercase (true hyphenation)
    text = re.sub(r'(?<=[a-z])-\s*\n\s*(?=[a-z])', '', text)
    
    # Case 2: Medical abbreviations with hyphens
    text = re.sub(r'([A-Z]{2,})-\s*\n\s*([A-Z])', r'\1-\2', text)
    
    # Case 3: Mixed case medical terms
    text = re.sub(r'([A-Za-z]+)-\s*\n\s*([A-Z])', r'\1-\2', text)
    
    # Fix common OCR errors with clitics
    for word in ['but', 'and', 'or', 'of', 'to', 'the', 'for', 'in', 'on', 'with']:
        text = re.sub(rf'([a-z]+)-({word})\b', r'\1 \2', text, flags=re.IGNORECASE)
    
    # Canonicalize COVID variants
    text = re.sub(r'(?i)SARS\s*[–—-]?\s*CoV\s*[–—-]?\s*2', 'SARS-CoV-2', text)
    text = re.sub(r'(?i)COVID\s*[–—-]?\s*19', 'COVID-19', text)
    
    # Collapse intra-token spaced clusters
    text = INTRA_ABBR_CLUSTER.sub(lambda m: re.sub(r'\s+', '', m.group(0)), text)
    
    # Join split biomedical tokens
    biomedical_prefixes = ['CD', 'IL', 'TNF', 'TGF', 'VEGF', 'HLA', 'C', 'P']
    for prefix in biomedical_prefixes:
        text = re.sub(rf'\b({prefix})\s+(\d{{1,4}})\b', r'\1\2', text)
        text = re.sub(rf'\b({prefix})\s+(ANCA)\b', r'\1-\2', text)
    
    return text

def reflow_paragraphs(text: str) -> str:
    """Reflow text into coherent paragraphs"""
    lines = text.split("\n")
    buf = []
    paras = []
    
    def flush():
        if buf:
            paras.append(" ".join(buf).strip())
            buf.clear()
    
    for raw in lines:
        line = raw.strip()
        if not line:
            flush()
            continue
        
        if not buf:
            buf.append(line)
            continue
        
        prev = buf[-1]
        
        # If previous doesn't end with punctuation, join with space
        if not SENT_PUNCT_RE.search(prev):
            # Check for bullet points
            if re.match(r'^[-*•\u2022]\s+', line):
                flush()
                buf.append(line)
            else:
                buf[-1] = prev + " " + line
            continue
        
        # Check for headings
        if (re.match(r'^[A-Z][A-Za-z].{0,80}$', line) and 
            not line.endswith(".") and 
            len(line.split()) < 10):
            flush()
            buf.append(line)
        else:
            buf.append(line)
    
    flush()
    return "\n\n".join(paras)

def canonicalize_abbreviation(abbrev: str) -> str:
    """Canonicalize abbreviation"""
    s = abbrev.strip()
    
    if s.lower() in COMMON_WORDS:
        return None
    
    # Normalize hyphens
    s = re.sub(r'[\u2010-\u2015\u2212]', '-', s)
    s = re.sub(r'\s*/\s*', '/', s)
    
    # Uppercase
    u = s.upper()
    
    # Remove trailing 's' for plurals (except special cases)
    if u.endswith('S') and u not in {"SARS", "AIDS", "ARDS"} and len(u) >= 4:
        u = u[:-1]
    
    # Apply aliases
    u = ALIAS_MAP.get(u, u)
    
    return u

def is_valid_abbreviation(abbrev: str) -> bool:
    """Validate abbreviation"""
    if not abbrev or len(abbrev) < 2 or len(abbrev) > 25:
        return False
    
    if abbrev.lower() in COMMON_WORDS:
        return False
    
    if not any(c.isupper() for c in abbrev):
        return False
    
    if '-' in abbrev:
        parts = abbrev.split('-')
        for part in parts:
            if part and not (part[0].isupper() or part.isupper()):
                return False
    
    return True

def is_plausible_expansion(text: str) -> bool:
    """Validate expansion text"""
    if not text:
        return False
    
    tokens = re.findall(r"[^\W_]+", text)
    
    if not (2 <= len(tokens) <= 10):
        return False
    
    if VERB_RE.search(text):
        return False
    
    if text[0].islower():
        return False
    
    combined = ''.join(tokens)
    alpha_chars = sum(1 for c in combined if c.isalpha())
    if alpha_chars < 0.7 * len(combined):
        return False
    
    return True

def has_spacing_issues(text: str, sample_size: int = 500) -> bool:
    """Detect spacing issues in text"""
    sample = text[:sample_size] if text else ""
    
    concatenated_patterns = [
        r'[a-z]{3,}with[a-z]{3,}',
        r'[a-z]{3,}and[a-z]{3,}',
        r'[a-z]{3,}or[a-z]{3,}',
        r'[a-z]{3,}for[a-z]{3,}',
        r'[a-z]{3,}the[a-z]{3,}',
    ]
    
    camel_case_names = len(re.findall(r'[A-Z][a-z]+[A-Z][a-z]+', sample))
    long_lowercase = len(re.findall(r'[a-z]{14,}', sample))
    no_space_parens = len(re.findall(r'[a-zA-Z]\(', sample))
    concat_words = sum(
        len(re.findall(pattern, sample, re.IGNORECASE)) 
        for pattern in concatenated_patterns
    )
    
    # Need at least 2 indicators or strong evidence
    indicators = sum([
        camel_case_names > 0,
        long_lowercase > 0,
        no_space_parens > 1,
        concat_words > 0
    ])
    
    return indicators >= 2 or long_lowercase > 1

# ============================================================================
# PDF Text Cleaner
# ============================================================================

class PDFTextCleaner:
    """Advanced PDF text cleaning with abbreviation protection"""
    
    def __init__(self):
        self.use_word_segmentation = WORDSEGMENT_AVAILABLE
        self.use_spell_checking = ENCHANT_AVAILABLE
    
    def mask_allowlist_first(self, text: str) -> Tuple[str, list]:
        """Mask curated abbreviations first"""
        if not ALLOW_RE:
            return text, []
        
        kept = []
        def _sub(m):
            kept.append(m.group(1))
            return f"{SENT_L}{len(kept)-1}{SENT_R}"
        
        masked = ALLOW_RE.sub(_sub, text)
        return masked, kept
    
    def mask_abbreviations(self, text: str) -> Tuple[str, list]:
        """Mask biomedical abbreviations"""
        # Protect section headings
        protected_headings = []
        for heading in SECTION_HEADINGS:
            pattern = rf'\b{heading}\b'
            matches = list(re.finditer(pattern, text))
            for match in matches:
                protected_headings.append((match.start(), match.end()))
        
        kept = []
        def _sub(m):
            for start, end in protected_headings:
                if m.start() >= start and m.end() <= end:
                    return m.group(0)
            kept.append(m.group('abbr'))
            return f"{SENT_L}{len(kept)-1}{SENT_R}{m.group('sfx') or ''}"
        
        masked = ABBR_WITH_SUFFIX.sub(_sub, text)
        
        # Fix sentinel adjacency
        SENT = rf"{re.escape(SENT_L)}\d+{re.escape(SENT_R)}"
        masked = re.sub(rf'({SENT})(?=[a-z(])', r'\1 ', masked)
        masked = re.sub(rf'(?<=[a-z])({SENT})', r' \1', masked)
        
        return masked, kept
    
    def unmask(self, text: str, kept: list) -> str:
        """Restore masked abbreviations"""
        def _sub(m):
            idx = int(m.group(1))
            return kept[idx] if idx < len(kept) else m.group(0)
        return SENTINEL_RE.sub(_sub, text)
    
    def looks_plain_word(self, token: str) -> bool:
        """Check if token is safe for spacing fixes"""
        if not token or SENT_L in token or SENT_R in token:
            return False
        if any(ch.isdigit() for ch in token):
            return False
        if re.search(rf"{HYPH}|/", token):
            return False
        if UNIT_PAT.search(token) or COMPOUND_KEEP.search(token):
            return False
        return token.isalpha()
    
    @lru_cache(maxsize=128)
    def segment_word_cached(self, word: str) -> List[str]:
        """Cached word segmentation"""
        if not WORDSEGMENT_AVAILABLE:
            return [word]
        
        import wordsegment
        return wordsegment.segment(word)
    
    def segment_concatenated_words(self, text: str) -> str:
        """Intelligently split concatenated words"""
        if not self.use_word_segmentation:
            return text
        
        # Look for long lowercase sequences (raised threshold)
        concatenated_pattern = r'\b[a-z]{14,}\b'
        
        def split_if_needed(match):
            word = match.group(0)
            
            # Skip URLs, emails, technical terms
            if any(skip in word for skip in ['http', 'www', '@', '_']):
                return word
            
            # Skip if contains sentinels
            if SENT_L in word or SENT_R in word:
                return word
            
            # Check for bridge words nearby
            context_window = 20
            start = max(0, match.start() - context_window)
            end = min(len(text), match.end() + context_window)
            context = text[start:end].lower()
            
            bridge_words = ['with', 'and', 'for', 'the', 'that']
            has_bridge = any(f'{word}' in context for word in bridge_words)
            
            if not has_bridge:
                return word
            
            # Try segmentation
            segments = self.segment_word_cached(word)
            
            if len(segments) > 1:
                if self.use_spell_checking and SPELL_CHECKER:
                    valid_count = sum(
                        1 for seg in segments 
                        if len(seg) > 2 and SPELL_CHECKER.check(seg)
                    )
                    if valid_count >= len(segments) / 2:
                        return ' '.join(segments)
                else:
                    if all(len(seg) > 1 for seg in segments):
                        return ' '.join(segments)
            
            return word
        
        return re.sub(concatenated_pattern, split_if_needed, text)
    
    def fix_spacing_around_words(self, text: str) -> str:
        """Apply spacing fixes to safe tokens"""
        parts = re.split(r'(\s+)', text)
        out = []
        
        for p in parts:
            if not p or p.isspace():
                out.append(p)
                continue
            
            if not self.looks_plain_word(p):
                # Basic fixes for non-plain words
                p = re.sub(r'([A-Za-z])(\()', r'\1 \2', p)
                out.append(p)
                continue
            
            tok = p
            # CamelCase
            tok = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', tok)
            # ALLCAPS followed by lowercase
            tok = re.sub(r'([A-Z]{2,})([a-z])', r'\1 \2', tok)
            # Space before parenthesis
            tok = re.sub(r'([A-Za-z])(\()', r'\1 \2', tok)
            
            out.append(tok)
        
        return ''.join(out)
    
    def clean(self, text: str) -> str:
        """Main cleaning pipeline"""
        if not text:
            return ""
        
        start_time = time.time()
        original_length = len(text)
        
        # Basic cleaning
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Pre-normalize
        text = pre_normalize(text)
        
        # Reflow paragraphs
        text = reflow_paragraphs(text)
        
        # Clean up whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        
        # Check if spacing fixes needed
        if has_spacing_issues(text):
            pre_len = len(text)
            
            # Mask abbreviations
            text, kept_allowlist = self.mask_allowlist_first(text)
            masked, kept_abbr = self.mask_abbreviations(text)
            kept = kept_allowlist + kept_abbr
            
            # Calculate masking metrics for adaptive processing
            masked_bytes = sum(
                len(m) for m in 
                re.findall(rf"{re.escape(SENT_L)}\d+{re.escape(SENT_R)}", masked)
            )
            mask_ratio = masked_bytes / max(1, pre_len)
            
            # Apply fixes if not over-masked
            if mask_ratio < 0.08:
                fixed = self.fix_spacing_around_words(masked)
                fixed = self.segment_concatenated_words(fixed)
            else:
                fixed = masked
                logger.debug(f"Skipped segmentation (mask_ratio={mask_ratio:.3f})")
            
            # Unmask
            text = self.unmask(fixed, kept)
            
            logger.debug(f"Protected {len(kept)} abbreviations")
        
        # Final cleanup
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()
        
        # Log stats
        elapsed = time.time() - start_time
        reduction = ((original_length - len(text)) / original_length * 100) if original_length > 0 else 0
        logger.debug(f"Cleaning complete: {elapsed:.2f}s, {reduction:.1f}% reduction")
        
        return text

# ============================================================================
# Base Extractor
# ============================================================================

class PDFExtractor(ABC):
    """Abstract base class for PDF extractors"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.cleaner = PDFTextCleaner()
    
    @abstractmethod
    def extract(self, file_path: Path, max_pages: Optional[int] = None) -> Tuple[str, int, Dict[str, Any]]:
        """Extract text from PDF"""
        pass
    
    def _create_info(self, pages: int, chars: int, method: str, time_taken: float) -> Dict[str, Any]:
        """Create extraction info dict"""
        return {
            'extraction_method': method,
            'pages_extracted': pages,
            'characters_extracted': chars,
            'extraction_time': round(time_taken, 2),
            'average_chars_per_page': chars // pages if pages > 0 else 0
        }

# ============================================================================
# Pdfplumber Extractor
# ============================================================================

class PdfplumberExtractor(PDFExtractor):
    """Extract text using pdfplumber"""
    
    def __init__(self):
        super().__init__()
        try:
            import pdfplumber
            self.pdfplumber = pdfplumber
            self.available = True
        except ImportError:
            self.available = False
            logger.debug("pdfplumber not available")
    
    def extract(self, file_path: Path, max_pages: Optional[int] = None) -> Tuple[str, int, Dict[str, Any]]:
        """Extract with pdfplumber"""
        if not self.available:
            raise ImportError("pdfplumber not installed")
        
        start_time = time.time()
        all_text = []
        total_pages = 0
        
        with self.pdfplumber.open(file_path) as pdf:
            num_pages = len(pdf.pages)
            pages_to_extract = min(num_pages, max_pages) if max_pages else num_pages
            
            for i, page in enumerate(pdf.pages[:pages_to_extract]):
                total_pages += 1
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"Progress: {i + 1}/{pages_to_extract} pages")
                
                try:
                    text = self._extract_page_text(page)
                    if text:
                        all_text.append(text)
                except Exception as e:
                    logger.debug(f"Page {i + 1} error: {e}")
                    continue
        
        raw_text = '\n'.join(all_text)
        cleaned_text = self.cleaner.clean(raw_text)
        
        time_taken = time.time() - start_time
        info = self._create_info(total_pages, len(cleaned_text), 'pdfplumber', time_taken)
        
        return cleaned_text, total_pages, info
    
    def _extract_words_reflow(self, page) -> str:
        """Extract words and reflow into lines"""
        words = page.extract_words(use_text_flow=True, x_tolerance=1.5, y_tolerance=3)
        if not words:
            return ""
        
        # Group by line
        lines = defaultdict(list)
        for w in words:
            band = round(w.get("top", 0) / 3) * 3
            lines[band].append(w)
        
        out = []
        for band in sorted(lines.keys()):
            line_words = sorted(lines[band], key=lambda w: w.get("x0", 0))
            out.append(" ".join(w["text"] for w in line_words))
        
        return "\n".join(out)
    
    def _extract_page_text(self, page) -> str:
        """Try different extraction strategies"""
        strategies = [
            lambda p: p.extract_text(),
            lambda p: p.extract_text(layout=True),
            self._extract_words_reflow,
            lambda p: ' '.join([w['text'] for w in (p.extract_words() or [])])
        ]
        
        for strategy in strategies:
            try:
                text = strategy(page)
                if text and len(text.strip()) > 10:
                    return text
            except:
                continue
        
        return ''

# ============================================================================
# PyMuPDF Extractor
# ============================================================================

class PyMuPDFExtractor(PDFExtractor):
    """Extract text using PyMuPDF"""
    
    def __init__(self):
        super().__init__()
        try:
            import fitz
            self.fitz = fitz
            self.available = True
        except ImportError:
            self.available = False
            logger.debug("PyMuPDF not available")
    
    def extract(self, file_path: Path, max_pages: Optional[int] = None) -> Tuple[str, int, Dict[str, Any]]:
        """Extract with PyMuPDF"""
        if not self.available:
            raise ImportError("PyMuPDF not installed")
        
        start_time = time.time()
        all_text = []
        total_pages = 0
        
        doc = self.fitz.open(str(file_path))
        num_pages = len(doc)
        pages_to_extract = min(num_pages, max_pages) if max_pages else num_pages
        
        for i in range(pages_to_extract):
            total_pages += 1
            page = doc[i]
            
            text = self._extract_page_text(page)
            if text:
                all_text.append(text)
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Processed {i + 1}/{pages_to_extract} pages")
        
        doc.close()
        
        raw_text = '\n'.join(all_text)
        cleaned_text = self.cleaner.clean(raw_text)
        
        time_taken = time.time() - start_time
        info = self._create_info(total_pages, len(cleaned_text), 'PyMuPDF', time_taken)
        
        return cleaned_text, total_pages, info
    
    def _blocks_to_text(self, blocks) -> str:
        """Convert blocks to text with proper ordering"""
        # Filter valid blocks
        blocks = [b for b in blocks if len(b) > 4 and b[4].strip()]
        
        # Sort by Y coordinate (grouped) then X
        blocks.sort(key=lambda b: (round(b[1] / 10) * 10, b[0]))
        
        parts = []
        last_band = None
        
        for block in blocks:
            x0, y0, x1, y1, txt = block[:5]
            band = round(y0 / 10) * 10
            
            if last_band is not None and band != last_band:
                parts.append("\n")
            
            parts.append(txt.strip())
            last_band = band
        
        return "\n".join(parts)
    
    def _extract_page_text(self, page) -> str:
        """Extract text with different methods"""
        methods = [
            lambda p: p.get_text(),
            lambda p: self._blocks_to_text(p.get_text("blocks")),
            lambda p: p.get_text("text")
        ]
        
        for method in methods:
            try:
                result = method(page)
                if result and len(result.strip()) > 10:
                    return result
            except:
                continue
        
        return ''