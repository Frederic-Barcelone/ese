#!/usr/bin/env python3
"""
PDF Extractors Module - Enhanced with Adaptive Column Detection
===============================================================
Robust PDF extraction that automatically detects and handles:
- Single column layouts
- Two column layouts (academic papers)
- Mixed layouts (single-column abstract → two-column body)
- Tables spanning columns
- Headers/footers
- Figure captions

Key Features:
- Per-page column detection using word position analysis
- Reading order correction (left column before right)
- Table region detection and separate handling
- Header/footer exclusion zones
- Robust fallback strategies
"""

import re
import time
import os
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, NamedTuple
from collections import defaultdict
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum

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
# Configuration
# ============================================================================
CONFIG = {}
ALLOWLIST_SHORT = {"FDA", "NIH", "EMA", "RCT", "ICU"}
ALIAS_MAP = {}

# ============================================================================
# Column Detection Types
# ============================================================================

class LayoutType(Enum):
    """Page layout types"""
    SINGLE_COLUMN = "single"
    TWO_COLUMN = "two_column"
    THREE_COLUMN = "three_column"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class ColumnRegion:
    """Represents a detected column region"""
    x_start: float
    x_end: float
    y_start: float
    y_end: float
    confidence: float = 1.0


@dataclass
class PageLayout:
    """Detected layout for a single page"""
    layout_type: LayoutType
    columns: List[ColumnRegion]
    header_zone: float  # Y coordinate below which is header
    footer_zone: float  # Y coordinate above which is footer
    confidence: float
    

@dataclass 
class TextBlock:
    """A block of text with position"""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    block_type: str = "text"  # text, table, figure, header, footer


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
# Column Detection Algorithm
# ============================================================================

class ColumnDetector:
    """
    Adaptive column detection for PDF pages.
    
    Analyzes word positions to determine:
    - Number of columns (1, 2, or 3)
    - Column boundaries
    - Header/footer zones
    - Table regions
    """
    
    # Configuration - tuned for academic papers
    MIN_COLUMN_GAP = 8  # Minimum gap (points) between columns - lowered for tight layouts
    HEADER_ZONE_RATIO = 0.08  # Top 8% is potential header
    FOOTER_ZONE_RATIO = 0.08  # Bottom 8% is potential footer
    MIN_WORDS_FOR_DETECTION = 30  # Need enough words for reliable detection
    GAP_DETECTION_ZONE = (0.30, 0.70)  # Middle zone to look for column gaps (widened)
    NUM_HISTOGRAM_BINS = 60  # More bins for finer resolution
    
    def __init__(self):
        self.debug = False
    
    def detect_layout(self, words: List[Dict], page_width: float, page_height: float) -> PageLayout:
        """
        Detect page layout from word positions.
        
        Args:
            words: List of word dicts with 'x0', 'x1', 'top', 'bottom', 'text'
            page_width: Page width in points
            page_height: Page height in points
            
        Returns:
            PageLayout with detected structure
        """
        if not words or len(words) < self.MIN_WORDS_FOR_DETECTION:
            # Not enough words - assume single column
            return PageLayout(
                layout_type=LayoutType.SINGLE_COLUMN,
                columns=[ColumnRegion(0, page_width, 0, page_height)],
                header_zone=page_height * self.HEADER_ZONE_RATIO,
                footer_zone=page_height * (1 - self.FOOTER_ZONE_RATIO),
                confidence=0.5
            )
        
        # Calculate header/footer zones
        header_zone = page_height * self.HEADER_ZONE_RATIO
        footer_zone = page_height * (1 - self.FOOTER_ZONE_RATIO)
        
        # Filter words in body area (exclude header/footer)
        body_words = [
            w for w in words 
            if w.get('top', 0) > header_zone and w.get('top', 0) < footer_zone
        ]
        
        if len(body_words) < self.MIN_WORDS_FOR_DETECTION // 2:
            body_words = words  # Use all words if not enough in body
        
        # Analyze X-position distribution
        layout_type, columns, confidence = self._analyze_x_distribution(
            body_words, page_width, page_height
        )
        
        return PageLayout(
            layout_type=layout_type,
            columns=columns,
            header_zone=header_zone,
            footer_zone=footer_zone,
            confidence=confidence
        )
    
    def _analyze_x_distribution(
        self, 
        words: List[Dict], 
        page_width: float,
        page_height: float
    ) -> Tuple[LayoutType, List[ColumnRegion], float]:
        """
        Analyze horizontal distribution of words to detect columns.
        
        Uses histogram analysis to find gaps that indicate column boundaries.
        """
        if not words:
            return LayoutType.SINGLE_COLUMN, [ColumnRegion(0, page_width, 0, page_height)], 0.5
        
        # Get word start positions (x0)
        x_starts = [w.get('x0', 0) for w in words]
        x_ends = [w.get('x1', 0) for w in words]
        
        # Create histogram of X positions with finer resolution
        num_bins = self.NUM_HISTOGRAM_BINS
        bin_width = page_width / num_bins
        histogram = [0] * num_bins
        
        for x in x_starts:
            bin_idx = min(int(x / bin_width), num_bins - 1)
            histogram[bin_idx] += 1
        
        # Find gaps in the middle region (potential column separators)
        middle_start = int(num_bins * self.GAP_DETECTION_ZONE[0])
        middle_end = int(num_bins * self.GAP_DETECTION_ZONE[1])
        
        # Calculate average density (excluding empty bins)
        non_empty_bins = [h for h in histogram if h > 0]
        avg_density = sum(non_empty_bins) / len(non_empty_bins) if non_empty_bins else 1
        
        # Gap threshold - bins with much lower than average density
        # Use 15% of average for gap detection (allowing some cross-column content)
        gap_threshold = avg_density * 0.15
        
        # Also calculate a "significant drop" threshold for relative gaps
        # A gap is also detected if density drops to <30% of surrounding bins
        significant_drop_ratio = 0.30
        
        # Find significant gaps (consecutive low-density bins)
        gaps = []
        in_gap = False
        gap_start = None
        
        for i in range(middle_start, middle_end):
            # Check absolute threshold
            is_below_threshold = histogram[i] <= gap_threshold
            
            # Check relative drop (compared to neighbors)
            left_avg = sum(histogram[max(0, i-3):i]) / 3 if i > 0 else avg_density
            right_avg = sum(histogram[i+1:min(num_bins, i+4)]) / 3 if i < num_bins - 1 else avg_density
            neighbor_avg = (left_avg + right_avg) / 2
            is_significant_drop = histogram[i] < neighbor_avg * significant_drop_ratio if neighbor_avg > 0 else False
            
            if is_below_threshold or is_significant_drop:
                if not in_gap:
                    in_gap = True
                    gap_start = i
            else:
                if in_gap:
                    gap_end = i
                    gap_width = (gap_end - gap_start) * bin_width
                    # Check if gap contains any zero bins (stronger evidence)
                    has_zero_bin = any(histogram[j] == 0 for j in range(gap_start, gap_end))
                    # Check minimum count in gap region
                    min_in_gap = min(histogram[j] for j in range(gap_start, gap_end)) if gap_end > gap_start else 999
                    
                    if gap_width >= self.MIN_COLUMN_GAP or has_zero_bin or min_in_gap <= gap_threshold:
                        gap_center = ((gap_start + gap_end) / 2) * bin_width
                        gaps.append({
                            'center': gap_center,
                            'width': gap_width,
                            'start': gap_start * bin_width,
                            'end': gap_end * bin_width,
                            'is_empty': has_zero_bin,
                            'min_count': min_in_gap
                        })
                    in_gap = False
        
        # Handle gap at end of range
        if in_gap:
            gap_end = middle_end
            gap_width = (gap_end - gap_start) * bin_width
            has_zero_bin = any(histogram[j] == 0 for j in range(gap_start, gap_end))
            min_in_gap = min(histogram[j] for j in range(gap_start, gap_end)) if gap_end > gap_start else 999
            
            if gap_width >= self.MIN_COLUMN_GAP or has_zero_bin or min_in_gap <= gap_threshold:
                gap_center = ((gap_start + gap_end) / 2) * bin_width
                gaps.append({
                    'center': gap_center,
                    'width': gap_width,
                    'start': gap_start * bin_width,
                    'end': gap_end * bin_width,
                    'is_empty': has_zero_bin,
                    'min_count': min_in_gap
                })
        
        # Also look for isolated low-density bins (single-bin gaps)
        for i in range(middle_start, middle_end):
            # Check for significant dip compared to neighbors
            left_avg = sum(histogram[max(0, i-2):i]) / 2 if i > 1 else avg_density
            right_avg = sum(histogram[i+1:min(num_bins, i+3)]) / 2 if i < num_bins - 2 else avg_density
            
            if left_avg > 0 and right_avg > 0:
                neighbor_avg = (left_avg + right_avg) / 2
                # If this bin is less than 25% of neighbors and neighbors are substantial
                if (histogram[i] < neighbor_avg * 0.25 and 
                    neighbor_avg > avg_density * 0.5):
                    # Check if not already captured
                    already_captured = any(
                        g['start'] <= i * bin_width <= g['end'] for g in gaps
                    )
                    if not already_captured:
                        gaps.append({
                            'center': (i + 0.5) * bin_width,
                            'width': bin_width,
                            'start': i * bin_width,
                            'end': (i + 1) * bin_width,
                            'is_empty': histogram[i] == 0,
                            'min_count': histogram[i]
                        })
        
        # Sort gaps by position
        gaps.sort(key=lambda g: g['center'])
        
        # Merge nearby gaps (within 2 bins of each other)
        merged_gaps = []
        for gap in gaps:
            if merged_gaps and (gap['start'] - merged_gaps[-1]['end']) < bin_width * 2:
                # Merge with previous
                merged_gaps[-1]['end'] = gap['end']
                merged_gaps[-1]['width'] = merged_gaps[-1]['end'] - merged_gaps[-1]['start']
                merged_gaps[-1]['center'] = (merged_gaps[-1]['start'] + merged_gaps[-1]['end']) / 2
            else:
                merged_gaps.append(gap.copy())
        
        gaps = merged_gaps
        
        logger.debug(f"Column detection: found {len(gaps)} gaps in middle region")
        
        # Determine layout based on gaps
        if not gaps:
            # No significant gaps - single column
            return (
                LayoutType.SINGLE_COLUMN,
                [ColumnRegion(0, page_width, 0, page_height)],
                0.9
            )
        
        if len(gaps) >= 1:
            # Find the most significant gap (prefer empty gaps, then widest)
            best_gap = max(gaps, key=lambda g: (g.get('is_empty', False), g['width']))
            
            # Verify this is really a column gap by checking word distribution
            left_words = sum(1 for x in x_starts if x < best_gap['start'])
            right_words = sum(1 for x in x_starts if x > best_gap['end'])
            total_words = len(x_starts)
            
            # Both columns should have substantial content (at least 15% each)
            left_ratio = left_words / total_words if total_words > 0 else 0
            right_ratio = right_words / total_words if total_words > 0 else 0
            
            logger.debug(f"Gap at x={best_gap['center']:.1f}: left={left_ratio:.2%}, right={right_ratio:.2%}")
            
            if left_ratio > 0.15 and right_ratio > 0.15:
                # Confirmed two-column layout
                balance = min(left_ratio, right_ratio) / max(left_ratio, right_ratio)
                confidence = 0.7 + (balance * 0.3)  # Higher confidence if balanced
                
                return (
                    LayoutType.TWO_COLUMN,
                    [
                        ColumnRegion(0, best_gap['start'], 0, page_height, confidence),
                        ColumnRegion(best_gap['end'], page_width, 0, page_height, confidence)
                    ],
                    confidence
                )
            else:
                # Unbalanced - probably not true two-column
                return (
                    LayoutType.SINGLE_COLUMN,
                    [ColumnRegion(0, page_width, 0, page_height)],
                    0.7
                )
        
        # Default to single column
        return (
            LayoutType.SINGLE_COLUMN,
            [ColumnRegion(0, page_width, 0, page_height)],
            0.6
        )
    
    def detect_table_regions(
        self, 
        words: List[Dict], 
        page_width: float, 
        page_height: float
    ) -> List[ColumnRegion]:
        """
        Detect potential table regions based on word alignment patterns.
        
        Tables often have:
        - Regular horizontal spacing (columns)
        - Many short text fragments
        - Numeric content
        """
        if not words or len(words) < 10:
            return []
        
        tables = []
        
        # Group words by Y position (rows)
        rows = defaultdict(list)
        y_tolerance = 5  # Points
        
        for w in words:
            y = round(w.get('top', 0) / y_tolerance) * y_tolerance
            rows[y].append(w)
        
        # Look for sequences of rows with similar structure
        sorted_rows = sorted(rows.items())
        
        table_start = None
        table_rows = []
        
        for y, row_words in sorted_rows:
            # Check if this row looks like a table row
            if self._is_table_row(row_words, page_width):
                if table_start is None:
                    table_start = y
                table_rows.append((y, row_words))
            else:
                # End of potential table
                if len(table_rows) >= 3:  # At least 3 rows for a table
                    tables.append(self._create_table_region(table_rows, page_width))
                table_start = None
                table_rows = []
        
        # Handle table at end of page
        if len(table_rows) >= 3:
            tables.append(self._create_table_region(table_rows, page_width))
        
        return tables
    
    def _is_table_row(self, words: List[Dict], page_width: float) -> bool:
        """Check if a row of words looks like a table row."""
        if len(words) < 3:
            return False
        
        # Sort by X position
        sorted_words = sorted(words, key=lambda w: w.get('x0', 0))
        
        # Check for regular spacing
        gaps = []
        for i in range(len(sorted_words) - 1):
            gap = sorted_words[i+1].get('x0', 0) - sorted_words[i].get('x1', 0)
            gaps.append(gap)
        
        if not gaps:
            return False
        
        # Table rows often have multiple similar-sized gaps
        avg_gap = sum(gaps) / len(gaps)
        if avg_gap < 10:  # Too close together
            return False
        
        # Check if gaps are relatively consistent
        gap_variance = sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)
        
        # High variance suggests irregular spacing (not a table)
        return gap_variance < (avg_gap ** 2) * 2
    
    def _create_table_region(
        self, 
        table_rows: List[Tuple[float, List[Dict]]], 
        page_width: float
    ) -> ColumnRegion:
        """Create a ColumnRegion for a detected table."""
        all_words = [w for _, row in table_rows for w in row]
        
        x0 = min(w.get('x0', 0) for w in all_words)
        x1 = max(w.get('x1', page_width) for w in all_words)
        y0 = min(y for y, _ in table_rows)
        y1 = max(y for y, _ in table_rows) + 20  # Add some padding
        
        return ColumnRegion(x0, x1, y0, y1, confidence=0.8)


# ============================================================================
# Text Processing Functions
# ============================================================================

# ============================================================================
# Encoding Fix Functions
# ============================================================================

def fix_encoding(text: str) -> str:
    """
    Fix common encoding issues from PDF extraction.
    
    Handles:
    - UTF-8 bytes misinterpreted as Latin-1 (mojibake)
    - Common character substitutions
    - Ligature normalization
    """
    if not text:
        return text
    
    # Common mojibake patterns: UTF-8 interpreted as Latin-1
    # These occur when UTF-8 bytes are decoded as Latin-1/Windows-1252
    mojibake_map = {
        # French/Spanish accented characters
        'Ã©': 'é',  # é
        'Ã¨': 'è',  # è
        'Ãª': 'ê',  # ê
        'Ã«': 'ë',  # ë
        'Ã ': 'à',  # à
        'Ã¢': 'â',  # â
        'Ã¤': 'ä',  # ä
        'Ã®': 'î',  # î
        'Ã¯': 'ï',  # ï
        'Ã´': 'ô',  # ô
        'Ã¶': 'ö',  # ö
        'Ã¹': 'ù',  # ù
        'Ã»': 'û',  # û
        'Ã¼': 'ü',  # ü
        'Ã§': 'ç',  # ç
        'Ã±': 'ñ',  # ñ
        'Ã¡': 'á',  # á
        'Ã­': 'í',  # í
        'Ã³': 'ó',  # ó
        'Ãº': 'ú',  # ú
        
        # German characters
        'Ã„': 'Ä',  # Ä
        'Ã–': 'Ö',  # Ö
        'Ãœ': 'Ü',  # Ü
        'ÃŸ': 'ß',  # ß
        
        # Uppercase accents
        'Ã‰': 'É',  # É
        'Ã€': 'À',  # À
        'Ã': 'Í',   # Í (partial - need context)
        
        # Special characters
        'â€™': "'",  # Right single quote
        'â€˜': "'",  # Left single quote
        'â€œ': '"',  # Left double quote
        'â€': '"',   # Right double quote (partial)
        'â€"': '—',  # Em dash
        'â€"': '–',  # En dash
        'â€¢': '•',  # Bullet
        'â€¦': '…',  # Ellipsis
        'Â©': '©',   # Copyright
        'Â®': '®',   # Registered
        'Â°': '°',   # Degree
        'Âµ': 'µ',   # Micro
        'Â±': '±',   # Plus-minus
        'Â²': '²',   # Superscript 2
        'Â³': '³',   # Superscript 3
        'Â½': '½',   # One half
        'Â¼': '¼',   # One quarter
        'Â¾': '¾',   # Three quarters
        
        # Greek letters (common in scientific text)
        'Î±': 'α',  # alpha
        'Î²': 'β',  # beta
        'Î³': 'γ',  # gamma
        'Î´': 'δ',  # delta
        'Îµ': 'ε',  # epsilon
        'Î¼': 'μ',  # mu
        'Ï€': 'π',  # pi
        'Ï': 'σ',   # sigma (lowercase)
        'Î£': 'Σ',  # Sigma (uppercase)
        'Î©': 'Ω',  # Omega
        
        # Cleanup partial encodings
        'Ã\u0082': 'Â',
        'Ã\u0083': 'Ã',
    }
    
    # Apply mojibake fixes
    for wrong, right in mojibake_map.items():
        text = text.replace(wrong, right)
    
    # Try to fix remaining Ã-based mojibake using byte-level repair
    # Pattern: Ã followed by a character in the range 0x80-0xBF
    def fix_utf8_sequence(match):
        """Attempt to reconstruct UTF-8 from mojibake."""
        try:
            # Get the matched text
            s = match.group(0)
            # Try to encode as Latin-1 and decode as UTF-8
            return s.encode('latin-1').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            return match.group(0)
    
    # Match Ã followed by characters that could be part of UTF-8 sequence
    text = re.sub(r'Ã[\x80-\xBF]', fix_utf8_sequence, text)
    
    # Normalize common ligatures
    ligatures = {
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬀ': 'ff',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl',
        'Ĳ': 'IJ',
        'ĳ': 'ij',
        'Œ': 'OE',
        'œ': 'oe',
        'Æ': 'AE',
        'æ': 'ae',
    }
    
    for lig, expanded in ligatures.items():
        text = text.replace(lig, expanded)
    
    return text


# ============================================================================
# Footer and Figure Detection
# ============================================================================

# Patterns for footer detection
FOOTER_PATTERNS = [
    r'©\s*(?:The\s+)?Author\(?s?\)?',  # Copyright notice
    r'Creative\s+Commons',  # CC license
    r'Open\s+Access',
    r'https?://[^\s]+license',  # License URLs
    r'doi\.org/',  # DOI
    r'This\s+article\s+is\s+licensed',
    r'Page\s+\d+\s+of\s+\d+',  # Page numbers
    r'^\d+\s*$',  # Standalone page numbers
]

FOOTER_PATTERN = re.compile('|'.join(FOOTER_PATTERNS), re.IGNORECASE)

# Patterns for figure/table captions
FIGURE_PATTERNS = [
    r'^Fig(?:ure)?\.?\s*\d+',  # Fig. 1, Figure 1
    r'^Table\.?\s*\d+',  # Table 1
    r'^\(See\s+(?:figure|legend)\s+on',  # (See figure on next page)
    r'^See\s+legend\s+on',
]

FIGURE_PATTERN = re.compile('|'.join(FIGURE_PATTERNS), re.IGNORECASE)


def is_footer_text(text: str) -> bool:
    """Check if text appears to be footer/copyright content."""
    if not text or len(text) < 10:
        return False
    return bool(FOOTER_PATTERN.search(text))


def is_figure_caption(text: str) -> bool:
    """Check if text appears to be a figure or table caption."""
    if not text:
        return False
    return bool(FIGURE_PATTERN.match(text.strip()))


def clean_footer_text(text: str) -> str:
    """
    Remove or clean footer-like content that spans columns.
    
    Footer text often gets jumbled when it spans both columns.
    We detect and either remove or relocate it.
    """
    lines = text.split('\n')
    cleaned_lines = []
    footer_buffer = []
    
    for line in lines:
        stripped = line.strip()
        
        # Check if this line is footer content
        if is_footer_text(stripped):
            # Accumulate footer lines
            footer_buffer.append(stripped)
        else:
            # If we have buffered footer content, decide what to do
            if footer_buffer:
                # If footer is substantial, keep a cleaned version at the end
                # For now, we'll just skip garbled footers
                footer_buffer = []
            
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def extract_figure_captions(text: str) -> Tuple[str, List[str]]:
    """
    Extract figure captions from text.
    
    Returns:
        Tuple of (text_without_captions, list_of_captions)
    """
    lines = text.split('\n')
    main_lines = []
    captions = []
    
    in_caption = False
    current_caption = []
    
    for line in lines:
        stripped = line.strip()
        
        if is_figure_caption(stripped):
            # Start of a new caption
            if current_caption:
                captions.append(' '.join(current_caption))
            current_caption = [stripped]
            in_caption = True
        elif in_caption:
            # Check if this continues the caption or is new content
            # Captions usually end with a period and next line starts with capital
            if (current_caption and 
                current_caption[-1].rstrip().endswith('.') and
                stripped and stripped[0].isupper() and
                not stripped.startswith('(') and
                len(stripped) > 50):
                # Likely end of caption, new paragraph
                captions.append(' '.join(current_caption))
                current_caption = []
                in_caption = False
                main_lines.append(line)
            else:
                # Continue caption
                current_caption.append(stripped)
        else:
            main_lines.append(line)
    
    # Don't forget last caption
    if current_caption:
        captions.append(' '.join(current_caption))
    
    return '\n'.join(main_lines), captions


def clean_figure_caption(caption: str) -> str:
    """
    Clean a figure caption that may have column-mixing artifacts.
    
    Figure captions spanning columns often have interleaved text.
    This attempts to reconstruct a sensible caption.
    """
    if not caption:
        return ""
    
    # Remove obviously garbled patterns
    # Pattern: text abruptly switching topics mid-sentence
    
    # Remove trailing fragments that look like they're from another column
    # These often appear after the main caption content
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', caption)
    
    cleaned_sentences = []
    for sent in sentences:
        # Skip sentences that are clearly garbled
        # - Very short fragments
        # - Sentences starting with lowercase (unless continuing)
        # - Sentences with too many uppercase words (likely headers mixed in)
        
        if len(sent) < 10:
            continue
        
        # Count uppercase word ratio
        words = sent.split()
        if not words:
            continue
            
        uppercase_words = sum(1 for w in words if w and w[0].isupper())
        uppercase_ratio = uppercase_words / len(words)
        
        # If most words start uppercase and it's not the first sentence, likely garbled
        if uppercase_ratio > 0.7 and len(words) > 5 and cleaned_sentences:
            continue
        
        # Check for common caption content
        if any(marker in sent.lower() for marker in ['fig', 'table', 'panel', 'image', 'graph', 'data', 'show', 'illustrat']):
            cleaned_sentences.append(sent)
        elif not cleaned_sentences:
            # Keep first sentence regardless
            cleaned_sentences.append(sent)
        elif sent[0].islower() or sent.startswith('('):
            # Continuation
            cleaned_sentences.append(sent)
        else:
            # Check if it seems related to previous content
            if len(cleaned_sentences) < 5:  # Allow some continuation
                cleaned_sentences.append(sent)
    
    return ' '.join(cleaned_sentences)


def pre_normalize(text: str) -> str:
    """Pre-normalize text to fix common PDF extraction issues"""
    
    # First, fix encoding issues
    text = fix_encoding(text)
    
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
    text = re.sub(r'(?i)SARS\s*[-–—]?\s*CoV\s*[-–—]?\s*2', 'SARS-CoV-2', text)
    text = re.sub(r'(?i)COVID\s*[-–—]?\s*19', 'COVID-19', text)
    
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
        
        # Pre-normalize (includes encoding fix)
        text = pre_normalize(text)
        
        # Clean footer content that spans columns (often garbled)
        text = clean_footer_text(text)
        
        # Extract and relocate figure captions
        text, captions = extract_figure_captions(text)
        
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
        
        # Re-append figure captions at the end (cleaned)
        if captions:
            # Clean each caption
            cleaned_captions = []
            for caption in captions:
                # Remove garbled parts from captions
                caption = clean_figure_caption(caption)
                if caption and len(caption) > 20:  # Keep only substantial captions
                    cleaned_captions.append(caption)
            
            if cleaned_captions:
                text = text + "\n\n" + "\n\n".join(cleaned_captions)
        
        # Final cleanup
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines
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
        self.column_detector = ColumnDetector()
    
    @abstractmethod
    def extract(self, file_path: Path, max_pages: Optional[int] = None) -> Tuple[str, int, Dict[str, Any]]:
        """Extract text from PDF"""
        pass
    
    def _create_info(self, pages: int, chars: int, method: str, time_taken: float, 
                     layout_stats: Optional[Dict] = None) -> Dict[str, Any]:
        """Create extraction info dict"""
        info = {
            'extraction_method': method,
            'pages_extracted': pages,
            'characters_extracted': chars,
            'extraction_time': round(time_taken, 2),
            'average_chars_per_page': chars // pages if pages > 0 else 0
        }
        if layout_stats:
            info['layout_detection'] = layout_stats
        return info


# ============================================================================
# Pdfplumber Extractor with Column Detection
# ============================================================================

class PdfplumberExtractor(PDFExtractor):
    """Extract text using pdfplumber with adaptive column detection"""
    
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
        """Extract with pdfplumber using column-aware extraction"""
        if not self.available:
            raise ImportError("pdfplumber not installed")
        
        start_time = time.time()
        all_text = []
        total_pages = 0
        layout_stats = {
            'single_column_pages': 0,
            'two_column_pages': 0,
            'mixed_pages': 0,
            'detection_confidence_avg': 0.0
        }
        confidence_sum = 0.0
        
        with self.pdfplumber.open(file_path) as pdf:
            num_pages = len(pdf.pages)
            pages_to_extract = min(num_pages, max_pages) if max_pages else num_pages
            
            for i, page in enumerate(pdf.pages[:pages_to_extract]):
                total_pages += 1
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"Progress: {i + 1}/{pages_to_extract} pages")
                
                try:
                    # Extract with column detection
                    text, layout = self._extract_page_with_columns(page)
                    
                    if text:
                        all_text.append(text)
                    
                    # Update layout stats
                    if layout:
                        confidence_sum += layout.confidence
                        if layout.layout_type == LayoutType.SINGLE_COLUMN:
                            layout_stats['single_column_pages'] += 1
                        elif layout.layout_type == LayoutType.TWO_COLUMN:
                            layout_stats['two_column_pages'] += 1
                        else:
                            layout_stats['mixed_pages'] += 1
                            
                except Exception as e:
                    logger.debug(f"Page {i + 1} error: {e}")
                    # Fallback to simple extraction
                    try:
                        text = page.extract_text() or ""
                        if text:
                            all_text.append(text)
                    except:
                        pass
                    continue
        
        # Calculate average confidence
        if total_pages > 0:
            layout_stats['detection_confidence_avg'] = round(confidence_sum / total_pages, 2)
        
        raw_text = '\n\n'.join(all_text)  # Double newline between pages
        cleaned_text = self.cleaner.clean(raw_text)
        
        time_taken = time.time() - start_time
        info = self._create_info(total_pages, len(cleaned_text), 'pdfplumber_columns', 
                                 time_taken, layout_stats)
        
        logger.debug(f"Layout detection: {layout_stats}")
        
        return cleaned_text, total_pages, info
    
    def _extract_page_with_columns(self, page) -> Tuple[str, Optional[PageLayout]]:
        """
        Extract text from a page with column detection.
        
        1. Get all words with positions
        2. Detect column layout
        3. Extract text in reading order
        """
        # Get words with positions
        try:
            words = page.extract_words(
                x_tolerance=3,
                y_tolerance=3,
                keep_blank_chars=False,
                use_text_flow=False  # We'll handle flow ourselves
            )
        except Exception as e:
            logger.debug(f"Word extraction failed: {e}")
            words = []
        
        if not words:
            # Fallback to simple extraction
            text = page.extract_text() or ""
            return text, None
        
        # Get page dimensions
        page_width = page.width
        page_height = page.height
        
        # Detect layout
        layout = self.column_detector.detect_layout(words, page_width, page_height)
        
        # Extract text based on layout
        if layout.layout_type == LayoutType.SINGLE_COLUMN:
            text = self._extract_single_column(words, layout)
        elif layout.layout_type == LayoutType.TWO_COLUMN:
            text = self._extract_two_columns(words, layout, page_width)
        else:
            # Fallback for unknown layouts
            text = self._extract_single_column(words, layout)
        
        return text, layout
    
    def _extract_single_column(self, words: List[Dict], layout: PageLayout) -> str:
        """Extract text assuming single column layout"""
        # Filter out header/footer
        body_words = [
            w for w in words
            if w.get('top', 0) > layout.header_zone and 
               w.get('top', 0) < layout.footer_zone
        ]
        
        if not body_words:
            body_words = words
        
        # Sort by Y then X (top to bottom, left to right)
        sorted_words = sorted(body_words, key=lambda w: (
            round(w.get('top', 0) / 3) * 3,  # Group by line (Y tolerance)
            w.get('x0', 0)
        ))
        
        # Group into lines
        lines = []
        current_line = []
        current_y = None
        y_tolerance = 5
        
        for word in sorted_words:
            word_y = round(word.get('top', 0) / y_tolerance) * y_tolerance
            
            if current_y is None:
                current_y = word_y
                current_line = [word]
            elif abs(word_y - current_y) <= y_tolerance:
                current_line.append(word)
            else:
                # New line
                if current_line:
                    line_text = ' '.join(w.get('text', '') for w in 
                                        sorted(current_line, key=lambda w: w.get('x0', 0)))
                    lines.append(line_text)
                current_line = [word]
                current_y = word_y
        
        # Don't forget last line
        if current_line:
            line_text = ' '.join(w.get('text', '') for w in 
                                sorted(current_line, key=lambda w: w.get('x0', 0)))
            lines.append(line_text)
        
        return '\n'.join(lines)
    
    def _extract_two_columns(self, words: List[Dict], layout: PageLayout, 
                             page_width: float) -> str:
        """
        Extract text from two-column layout in reading order.
        
        Reading order: Left column top-to-bottom, then right column top-to-bottom
        """
        if len(layout.columns) < 2:
            return self._extract_single_column(words, layout)
        
        left_col = layout.columns[0]
        right_col = layout.columns[1]
        
        # Split words into columns
        left_words = []
        right_words = []
        
        # Calculate column boundary (midpoint between columns)
        col_boundary = (left_col.x_end + right_col.x_start) / 2
        
        for word in words:
            # Skip header/footer
            word_top = word.get('top', 0)
            if word_top <= layout.header_zone or word_top >= layout.footer_zone:
                continue
            
            word_center = (word.get('x0', 0) + word.get('x1', 0)) / 2
            
            if word_center < col_boundary:
                left_words.append(word)
            else:
                right_words.append(word)
        
        # Extract each column
        left_text = self._words_to_text(left_words)
        right_text = self._words_to_text(right_words)
        
        # Combine: left column first, then right column
        combined = []
        if left_text.strip():
            combined.append(left_text)
        if right_text.strip():
            combined.append(right_text)
        
        return '\n\n'.join(combined)
    
    def _words_to_text(self, words: List[Dict]) -> str:
        """Convert words to text with proper line handling"""
        if not words:
            return ""
        
        # Sort by Y then X
        sorted_words = sorted(words, key=lambda w: (
            round(w.get('top', 0) / 3) * 3,
            w.get('x0', 0)
        ))
        
        # Group into lines
        lines = []
        current_line = []
        current_y = None
        y_tolerance = 5
        
        for word in sorted_words:
            word_y = round(word.get('top', 0) / y_tolerance) * y_tolerance
            
            if current_y is None:
                current_y = word_y
                current_line = [word]
            elif abs(word_y - current_y) <= y_tolerance:
                current_line.append(word)
            else:
                if current_line:
                    line_text = ' '.join(w.get('text', '') for w in 
                                        sorted(current_line, key=lambda w: w.get('x0', 0)))
                    lines.append(line_text)
                current_line = [word]
                current_y = word_y
        
        if current_line:
            line_text = ' '.join(w.get('text', '') for w in 
                                sorted(current_line, key=lambda w: w.get('x0', 0)))
            lines.append(line_text)
        
        return '\n'.join(lines)


# ============================================================================
# PyMuPDF Extractor with Column Detection
# ============================================================================

class PyMuPDFExtractor(PDFExtractor):
    """Extract text using PyMuPDF with adaptive column detection"""
    
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
        """Extract with PyMuPDF using column-aware extraction"""
        if not self.available:
            raise ImportError("PyMuPDF not installed")
        
        start_time = time.time()
        all_text = []
        total_pages = 0
        layout_stats = {
            'single_column_pages': 0,
            'two_column_pages': 0,
            'mixed_pages': 0,
            'detection_confidence_avg': 0.0
        }
        confidence_sum = 0.0
        
        doc = self.fitz.open(str(file_path))
        num_pages = len(doc)
        pages_to_extract = min(num_pages, max_pages) if max_pages else num_pages
        
        for i in range(pages_to_extract):
            total_pages += 1
            page = doc[i]
            
            try:
                text, layout = self._extract_page_with_columns(page)
                
                if text:
                    all_text.append(text)
                
                if layout:
                    confidence_sum += layout.confidence
                    if layout.layout_type == LayoutType.SINGLE_COLUMN:
                        layout_stats['single_column_pages'] += 1
                    elif layout.layout_type == LayoutType.TWO_COLUMN:
                        layout_stats['two_column_pages'] += 1
                    else:
                        layout_stats['mixed_pages'] += 1
                        
            except Exception as e:
                logger.debug(f"Page {i + 1} error: {e}")
                try:
                    text = page.get_text() or ""
                    if text:
                        all_text.append(text)
                except:
                    pass
                continue
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Processed {i + 1}/{pages_to_extract} pages")
        
        doc.close()
        
        if total_pages > 0:
            layout_stats['detection_confidence_avg'] = round(confidence_sum / total_pages, 2)
        
        raw_text = '\n\n'.join(all_text)
        cleaned_text = self.cleaner.clean(raw_text)
        
        time_taken = time.time() - start_time
        info = self._create_info(total_pages, len(cleaned_text), 'PyMuPDF_columns', 
                                 time_taken, layout_stats)
        
        logger.debug(f"Layout detection: {layout_stats}")
        
        return cleaned_text, total_pages, info
    
    def _extract_page_with_columns(self, page) -> Tuple[str, Optional[PageLayout]]:
        """Extract text with column detection using PyMuPDF"""
        # Get words with positions using dict output
        try:
            blocks = page.get_text("dict", flags=self.fitz.TEXT_PRESERVE_WHITESPACE)
            words = self._blocks_to_words(blocks)
        except Exception as e:
            logger.debug(f"Dict extraction failed: {e}")
            words = []
        
        if not words:
            text = page.get_text() or ""
            return text, None
        
        page_width = page.rect.width
        page_height = page.rect.height
        
        layout = self.column_detector.detect_layout(words, page_width, page_height)
        
        if layout.layout_type == LayoutType.SINGLE_COLUMN:
            text = self._extract_single_column(words, layout)
        elif layout.layout_type == LayoutType.TWO_COLUMN:
            text = self._extract_two_columns(words, layout, page_width)
        else:
            text = self._extract_single_column(words, layout)
        
        return text, layout
    
    def _blocks_to_words(self, blocks_dict: Dict) -> List[Dict]:
        """Convert PyMuPDF blocks dict to word list compatible with column detector"""
        words = []
        
        for block in blocks_dict.get('blocks', []):
            if block.get('type') != 0:  # 0 = text block
                continue
            
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    text = span.get('text', '').strip()
                    if not text:
                        continue
                    
                    # Split span into words
                    bbox = span.get('bbox', (0, 0, 0, 0))
                    span_words = text.split()
                    
                    if len(span_words) == 1:
                        words.append({
                            'text': text,
                            'x0': bbox[0],
                            'top': bbox[1],
                            'x1': bbox[2],
                            'bottom': bbox[3]
                        })
                    else:
                        # Estimate word positions within span
                        span_width = bbox[2] - bbox[0]
                        char_width = span_width / max(len(text), 1)
                        
                        pos = 0
                        for word in span_words:
                            word_start = bbox[0] + pos * char_width
                            word_end = word_start + len(word) * char_width
                            
                            words.append({
                                'text': word,
                                'x0': word_start,
                                'top': bbox[1],
                                'x1': word_end,
                                'bottom': bbox[3]
                            })
                            
                            pos += len(word) + 1  # +1 for space
        
        return words
    
    def _extract_single_column(self, words: List[Dict], layout: PageLayout) -> str:
        """Extract single column text"""
        body_words = [
            w for w in words
            if w.get('top', 0) > layout.header_zone and 
               w.get('top', 0) < layout.footer_zone
        ]
        
        if not body_words:
            body_words = words
        
        sorted_words = sorted(body_words, key=lambda w: (
            round(w.get('top', 0) / 3) * 3,
            w.get('x0', 0)
        ))
        
        lines = []
        current_line = []
        current_y = None
        y_tolerance = 5
        
        for word in sorted_words:
            word_y = round(word.get('top', 0) / y_tolerance) * y_tolerance
            
            if current_y is None:
                current_y = word_y
                current_line = [word]
            elif abs(word_y - current_y) <= y_tolerance:
                current_line.append(word)
            else:
                if current_line:
                    line_text = ' '.join(w.get('text', '') for w in 
                                        sorted(current_line, key=lambda w: w.get('x0', 0)))
                    lines.append(line_text)
                current_line = [word]
                current_y = word_y
        
        if current_line:
            line_text = ' '.join(w.get('text', '') for w in 
                                sorted(current_line, key=lambda w: w.get('x0', 0)))
            lines.append(line_text)
        
        return '\n'.join(lines)
    
    def _extract_two_columns(self, words: List[Dict], layout: PageLayout, 
                             page_width: float) -> str:
        """Extract two-column text in reading order"""
        if len(layout.columns) < 2:
            return self._extract_single_column(words, layout)
        
        left_col = layout.columns[0]
        right_col = layout.columns[1]
        col_boundary = (left_col.x_end + right_col.x_start) / 2
        
        left_words = []
        right_words = []
        
        for word in words:
            word_top = word.get('top', 0)
            if word_top <= layout.header_zone or word_top >= layout.footer_zone:
                continue
            
            word_center = (word.get('x0', 0) + word.get('x1', 0)) / 2
            
            if word_center < col_boundary:
                left_words.append(word)
            else:
                right_words.append(word)
        
        left_text = self._words_to_text(left_words)
        right_text = self._words_to_text(right_words)
        
        combined = []
        if left_text.strip():
            combined.append(left_text)
        if right_text.strip():
            combined.append(right_text)
        
        return '\n\n'.join(combined)
    
    def _words_to_text(self, words: List[Dict]) -> str:
        """Convert word list to text"""
        if not words:
            return ""
        
        sorted_words = sorted(words, key=lambda w: (
            round(w.get('top', 0) / 3) * 3,
            w.get('x0', 0)
        ))
        
        lines = []
        current_line = []
        current_y = None
        y_tolerance = 5
        
        for word in sorted_words:
            word_y = round(word.get('top', 0) / y_tolerance) * y_tolerance
            
            if current_y is None:
                current_y = word_y
                current_line = [word]
            elif abs(word_y - current_y) <= y_tolerance:
                current_line.append(word)
            else:
                if current_line:
                    line_text = ' '.join(w.get('text', '') for w in 
                                        sorted(current_line, key=lambda w: w.get('x0', 0)))
                    lines.append(line_text)
                current_line = [word]
                current_y = word_y
        
        if current_line:
            line_text = ' '.join(w.get('text', '') for w in 
                                sorted(current_line, key=lambda w: w.get('x0', 0)))
            lines.append(line_text)
        
        return '\n'.join(lines)