#!/usr/bin/env python3
"""
================================================================================
Date Parsing Utilities for Document Metadata Extraction
corpus_metadata/document_utils/metadata_date_utils.py
================================================================================

Purpose:
    Extract and normalize dates from filenames and document content
    for the rare disease document processing pipeline.

Version: 2.3 - Complete Fix with All Tests Passing
================================================================================
"""

import re
import os
import json
import logging
import logging.config
import threading
from datetime import datetime
from typing import Optional, Union, List, Dict, Any, Pattern, Tuple
from functools import lru_cache
from pathlib import Path

# ================================================================================
# Logging Configuration
# ================================================================================

def setup_logging(name: str = __name__, level: str = None) -> logging.Logger:
    """Set up centralized logging configuration."""
    level = level or os.environ.get('LOG_LEVEL', 'INFO')
    
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(name)

logger = setup_logging(__name__)

# ================================================================================
# Constants
# ================================================================================

MAX_FILENAME_LENGTH = 500
MAX_CONTENT_LENGTH = 10_000_000  # 10MB

DEFAULT_MONTHS = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
    'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
    'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
    'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
    'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
    'dec': 12, 'december': 12
}

DEFAULT_CONFIG = {
    'date_extraction': {
        'content_sample_size': 10000,
        'min_year_offset': -75,  # Extended to include 1950 (2025 - 75 = 1950)
        'max_year_offset': 25,   # Extended to include 2050 (2025 + 25 = 2050)
        'output_format': '%Y-%m-%d',
        'two_digit_year_cutoff': 50,
        'month_year_default_day': 1,
        'max_dates_from_content': 10,
        'strptime_formats': [
            '%Y-%m-%d %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y',
            '%d.%m.%Y',
            '%Y.%m.%d',
        ]
    },
    'masking': {
        'patterns': [
            {'name': 'EudraCT', 'pattern': r'\d{4}-\d{6}-\d{2}-\d{2}'},
            {'name': 'NCT', 'pattern': r'NCT\d{7,8}'},
            {'name': 'IND', 'pattern': r'IND[\s:]*\d{5,6}'},
            {'name': 'Version', 'pattern': r'v?\d+\.\d+\.\d+'},
            {'name': 'ISBN', 'pattern': r'ISBN[\s:]*[\d-]+'}
        ]
    }
}

# ================================================================================
# Configuration Manager
# ================================================================================

class DateParserConfig:
    """Configuration manager for DateParser."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config = self._load_config(config_path)
        logger.info(f"Configuration loaded: {len(self.config)} settings")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        config_path = config_path or os.environ.get('DATE_PARSER_CONFIG_PATH')
        
        if config_path:
            config_file = Path(config_path)
            if config_file.is_dir():
                for filename in ['date_parser.json', 'date_parser.yaml']:
                    potential = config_file / filename
                    if potential.exists():
                        config_file = potential
                        break
            
            if config_file.exists() and config_file.is_file():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        logger.info(f"Loaded configuration from {config_file}")
                        return config
                except Exception as e:
                    logger.error(f"Failed to load config from {config_file}: {e}")
        
        logger.info("Using default configuration")
        return DEFAULT_CONFIG.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

# ================================================================================
# Main DateParser Class
# ================================================================================

class DateParser:
    """Date parser with pattern matching and configuration support."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize DateParser."""
        self.config = DateParserConfig(config_path)
        self._extraction_stats = {'successful': 0, 'failed': 0}
        
        # Load configuration
        self._load_config_parameters()
        
        # Initialize patterns
        self._months = self._load_month_mappings()
        self._compiled_patterns = self._compile_date_patterns()
        self._mask_patterns = self._compile_mask_patterns()
        
        logger.info(f"DateParser initialized with {len(self._compiled_patterns)} date formats")
    
    def _load_config_parameters(self) -> None:
        """Load configuration parameters."""
        cfg = self.config.get('date_extraction', {})
        current_year = datetime.now().year
        
        self.content_sample_size = cfg.get('content_sample_size', 10000)
        self.min_year = current_year + cfg.get('min_year_offset', -50)
        self.max_year = current_year + cfg.get('max_year_offset', 10)
        self.output_format = cfg.get('output_format', '%Y-%m-%d')
        self.two_digit_year_cutoff = cfg.get('two_digit_year_cutoff', 50)
        self.month_year_default_day = cfg.get('month_year_default_day', 1)
        self.strptime_formats = cfg.get('strptime_formats', [])
    
    def _load_month_mappings(self) -> Dict[str, int]:
        """Load month name mappings."""
        return self.config.get('date_extraction.month_mappings', DEFAULT_MONTHS.copy())
    
    def _compile_date_patterns(self) -> List[Tuple[str, Pattern, int]]:
        """Compile date extraction patterns."""
        month_names = '|'.join(self._months.keys())
        
        # Use raw strings and proper escaping
        # Removed \b at start for month patterns to handle underscores
        patterns = [
            ('month_day_year', 
             r'(' + month_names + r')[\s_-]+(\d{1,2})(?:st|nd|rd|th)?[\s_,-]*(\d{4})', 1),
            ('day_month_year',
             r'(\d{1,2})(?:st|nd|rd|th)?[\s_-]+(' + month_names + r')[\s_,-]*(\d{4})', 2),
            ('compact_month',
             r'(\d{1,2})(' + month_names + r')(\d{2,4})', 3),
            ('iso_date',
             r'(20\d{2})[-_.](0[1-9]|1[0-2])[-_.]([0-2]\d|3[01])', 4),
            ('compact_numeric',
             r'(20\d{2})(0[1-9]|1[0-2])([0-2]\d|3[01])', 5),
            ('month_year',
             r'(' + month_names + r')[\s_-]*(\d{4})', 6),
        ]
        
        compiled = []
        for name, pattern, priority in patterns:
            try:
                compiled.append((name, re.compile(pattern, re.IGNORECASE), priority))
                logger.debug(f"Compiled pattern {name}: {pattern}")
            except re.error as e:
                logger.error(f"Failed to compile {name}: {e}")
        
        return sorted(compiled, key=lambda x: x[2])
    
    def _compile_mask_patterns(self) -> List[Tuple[Pattern, str]]:
        """Compile masking patterns."""
        patterns = []
        for mask in self.config.get('masking.patterns', []):
            try:
                pattern = re.compile(mask['pattern'])
                patterns.append((pattern, mask['name']))
            except re.error as e:
                logger.error(f"Failed to compile mask pattern: {e}")
        return patterns
    
    def _normalize_year(self, year_str: str) -> int:
        """Convert 2-digit year to 4-digit year."""
        year = int(year_str)
        if len(year_str) == 2:
            return 2000 + year if year <= self.two_digit_year_cutoff else 1900 + year
        return year
    
    def _validate_input(self, text: Optional[str], max_length: int, name: str) -> Optional[str]:
        """Validate and truncate input if needed."""
        if text is None:
            return None
        if not isinstance(text, str):
            text = str(text)
        if len(text) > max_length:
            logger.warning(f"{name} too long ({len(text)} chars), truncating")
            return text[:max_length]
        return text
    
    def _parse_match(self, match: re.Match, format_name: str) -> Optional[datetime]:
        """Parse a regex match into a datetime object."""
        try:
            groups = match.groups()
            
            if format_name == 'month_day_year':
                mon, day, year = groups
                return datetime(int(year), self._months[mon.lower()], int(day))
            
            elif format_name == 'day_month_year':
                day, mon, year = groups
                return datetime(int(year), self._months[mon.lower()], int(day))
            
            elif format_name == 'compact_month':
                day, mon, yy = groups
                return datetime(self._normalize_year(yy), self._months[mon.lower()], int(day))
            
            elif format_name == 'iso_date':
                y, mo, d = groups
                return datetime(int(y), int(mo), int(d))
            
            elif format_name == 'compact_numeric':
                y, mo, d = groups
                return datetime(int(y), int(mo), int(d))
            
            elif format_name == 'month_year':
                mon, year = groups
                return datetime(int(year), self._months[mon.lower()], self.month_year_default_day)
                
        except (ValueError, KeyError, IndexError) as e:
            logger.debug(f"Failed to parse {format_name}: {e}")
        
        return None
    
    def _is_valid_date(self, date: datetime) -> bool:
        """Check if date is in reasonable range."""
        return self.min_year <= date.year <= self.max_year
    
    @lru_cache(maxsize=1000)
    def extract_date_from_filename(self, filename: Optional[str]) -> Optional[datetime]:
        """Extract date from filename using configured patterns."""
        filename = self._validate_input(filename, MAX_FILENAME_LENGTH, "Filename")
        if not filename:
            return None
        
        # Don't lowercase the entire filename - patterns will handle case insensitivity
        for format_name, pattern, _ in self._compiled_patterns:
            match = pattern.search(filename)
            if match:
                date_obj = self._parse_match(match, format_name)
                if date_obj and self._is_valid_date(date_obj):
                    self._extraction_stats['successful'] += 1
                    logger.debug(f"Extracted date from filename: {date_obj}")
                    return date_obj
        
        logger.debug(f"No date found in filename: {filename}")
        return None
    
    def parse_date_string(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse a date string in various formats."""
        if not date_str:
            return None
        
        # Try filename extraction first
        result = self.extract_date_from_filename.__wrapped__(self, date_str)
        if result:
            return result
        
        # Try strptime formats
        for fmt in self.strptime_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def normalize_date_format(self, date_input: Union[str, datetime, None]) -> Optional[str]:
        """Normalize a date to configured format."""
        if date_input is None:
            return None
        
        if isinstance(date_input, datetime):
            return date_input.strftime(self.output_format)
        
        if isinstance(date_input, str):
            parsed = self.parse_date_string(date_input)
            if parsed:
                return parsed.strftime(self.output_format)
        
        return None
    
    
    
    def extract_dates_from_content(self, content: Optional[str], max_dates: int = 5) -> List[str]:
        """
        Extract dates from document content with enhanced patterns for academic PDFs.
        
        Args:
            content: Text content to search
            max_dates: Maximum number of dates to return
            
        Returns:
            List of normalized date strings in YYYY-MM-DD format
        """
        content = self._validate_input(content, MAX_CONTENT_LENGTH, "Content")
        if not content:
            return []
        
        # Mask non-date patterns first
        for pattern, replacement in self._mask_patterns:
            content = pattern.sub(replacement, content)
        
        # Look at more content for academic papers (they often have dates deeper in)
        sample = content[:min(len(content), 15000)]
        
        found_dates = []
        seen_dates = set()
        
        # First try standard patterns
        for format_name, pattern, _ in self._compiled_patterns:
            for match in pattern.finditer(sample):
                date_obj = self._parse_match(match, format_name)
                
                if date_obj and self._is_valid_date(date_obj):
                    normalized = self.normalize_date_format(date_obj)
                    if normalized and normalized not in seen_dates:
                        seen_dates.add(normalized)
                        found_dates.append(normalized)
                        
                        if len(found_dates) >= max_dates:
                            return found_dates
        
        # Try academic/publication specific patterns
        academic_patterns = [
            # Publication patterns
            (r'Published:?\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})', 'dmy'),
            (r'Published:?\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})', 'ymd'),
            (r'Publication Date:?\s*([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})', 'mdy'),
            (r'Date of Publication:?\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})', 'dmy'),
            
            # Accepted/Received patterns
            (r'Accepted:?\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})', 'dmy'),
            (r'Received:?\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})', 'dmy'),
            (r'Revised:?\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})', 'dmy'),
            
            # Copyright patterns
            (r'©\s*(\d{4})', 'year'),
            (r'Copyright\s*(?:\(c\))?\s*(\d{4})', 'year'),
            
            # DOI patterns (often contain year)
            (r'doi\.org/[\d.]+/[^/]*[._](\d{4})[._]', 'year'),
            (r'DOI:?\s*10\.\d+/[^/\s]*[._](\d{4})', 'year'),
            
            # Journal citation patterns
            (r'Vol(?:ume)?\s*\d+.*?\((\d{4})\)', 'year'),
            (r'Volume\s*\d+,\s*Issue\s*\d+,\s*([A-Za-z]+)\s*(\d{4})', 'my'),
            
            # Year in parentheses (common in references)
            (r'\((\d{4})\)', 'year'),
            
            # Conference patterns
            (r'Conference.*?(\d{4})', 'year'),
            (r'Proceedings.*?(\d{4})', 'year'),
        ]
        
        for pattern_str, date_type in academic_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for match in pattern.finditer(sample):
                try:
                    date_str = self._parse_academic_match(match, date_type)
                    if date_str and date_str not in seen_dates:
                        # Validate year is reasonable
                        year = int(date_str[:4])
                        if 1950 <= year <= 2030:
                            seen_dates.add(date_str)
                            found_dates.append(date_str)
                            if len(found_dates) >= max_dates:
                                return found_dates
                except:
                    continue
        
        # Last resort: find ANY reasonable year
        if not found_dates:
            year_pattern = re.compile(r'\b(19[7-9]\d|20[0-2]\d)\b')
            years = []
            for match in year_pattern.finditer(sample):
                year = int(match.group(1))
                # Filter out unlikely years
                if 1970 <= year <= 2030:
                    years.append(year)
            
            if years:
                # Use the most recent reasonable year
                most_recent = max(years)
                found_dates.append(f"{most_recent}-01-01")
        
        return found_dates

    def _parse_academic_match(self, match, date_type: str) -> Optional[str]:
        """
        Parse academic date patterns into ISO format.
        
        Args:
            match: Regex match object
            date_type: Type of date pattern matched
            
        Returns:
            ISO formatted date string or None
        """
        try:
            if date_type == 'year':
                year = match.group(1)
                return f"{year}-01-01"
                
            elif date_type == 'my':  # Month Year
                month_str = match.group(1)
                year = match.group(2)
                month = self._month_to_number(month_str)
                return f"{year}-{month:02d}-01"
                
            elif date_type == 'mdy':  # Month Day Year
                month_str = match.group(1)
                day = int(match.group(2))
                year = match.group(3)
                month = self._month_to_number(month_str)
                return f"{year}-{month:02d}-{day:02d}"
                
            elif date_type == 'dmy':  # Day Month Year
                day = int(match.group(1))
                month = match.group(2)
                year = match.group(3)
                
                # Check if month is text or number
                if month.isdigit():
                    month = int(month)
                else:
                    month = self._month_to_number(month)
                    
                return f"{year}-{month:02d}-{day:02d}"
                
            elif date_type == 'ymd':  # Year Month Day
                year = match.group(1)
                month = int(match.group(2))
                day = int(match.group(3))
                return f"{year}-{month:02d}-{day:02d}"
                
        except (ValueError, IndexError):
            return None
        
        return None

    def _month_to_number(self, month_str: str) -> int:
        """Convert month name to number."""
        month_map = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }
        return month_map.get(month_str.lower(), 1)

    def get_most_relevant_date(self, filename: str, content: str, 
                            file_metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Extract the most relevant date using multiple sources with fallback.
        
        Priority order:
        1. Date from filename
        2. Date from document content
        3. PDF creation date from metadata
        4. PDF modified date from metadata
        5. File modification date from metadata
        6. File creation date from metadata
        
        Args:
            filename: Document filename
            content: Document text content
            file_metadata: Optional dictionary with file/PDF metadata
            
        Returns:
            Most relevant date in YYYY-MM-DD format or None
        """
        # 1. Try filename first
        filename_date = self.extract_date_from_filename(filename)
        if filename_date:
            return self.normalize_date_format(filename_date)
        
        # 2. Try content extraction
        content_dates = self.extract_dates_from_content(content, max_dates=10)
        if content_dates:
            # Sort dates and return the most recent reasonable one
            content_dates.sort(reverse=True)
            return content_dates[0]
        
        # 3-6. Use file metadata as fallback
        if file_metadata:
            # Try PDF metadata first
            pdf_dates = [
                file_metadata.get('pdf_creation_date'),
                file_metadata.get('pdf_modified_date')
            ]
            
            for date_str in pdf_dates:
                if date_str:
                    # Parse the date string (format: YYYY-MM-DDTHH:MM:SS)
                    try:
                        if 'T' in str(date_str):
                            date_part = str(date_str).split('T')[0]
                            return date_part
                        elif isinstance(date_str, str) and len(date_str) >= 10:
                            return date_str[:10]
                    except:
                        continue
            
            # Try file system dates
            file_dates = [
                file_metadata.get('modified_time'),
                file_metadata.get('created_time')
            ]
            
            for date_str in file_dates:
                if date_str:
                    try:
                        if 'T' in str(date_str):
                            date_part = str(date_str).split('T')[0]
                            return date_part
                        elif isinstance(date_str, str) and len(date_str) >= 10:
                            return date_str[:10]
                    except:
                        continue
        
        return None
        
    def detect_date_format(self, date_str: str) -> Optional[str]:
        """Detect which format pattern matches the date string."""
        if not date_str:
            return None
        
        for format_name, pattern, _ in self._compiled_patterns:
            if pattern.search(date_str):
                return format_name
        
        return None
    
    def get_extraction_stats(self) -> Dict[str, int]:
        """Return extraction statistics."""
        return self._extraction_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset extraction statistics."""
        self._extraction_stats = {'successful': 0, 'failed': 0}
        logger.info("Extraction statistics reset")
    
    def save_config(self, path: str) -> bool:
        """Save current configuration to file."""
        try:
            config_path = Path(path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self.config.config, f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

# ================================================================================
# Singleton Pattern with Thread Safety
# ================================================================================

_default_parser: Optional[DateParser] = None
_parser_lock = threading.Lock()

def _get_default_parser() -> DateParser:
    """Get or create default parser instance (thread-safe)."""
    global _default_parser
    if _default_parser is None:
        with _parser_lock:
            # Double-check pattern
            if _default_parser is None:
                logger.info("Creating default DateParser instance")
                _default_parser = DateParser()
    return _default_parser

# ================================================================================
# Public API Functions
# ================================================================================

def extract_date_from_filename(filename: Optional[str]) -> Optional[datetime]:
    """Extract date from filename."""
    return _get_default_parser().extract_date_from_filename(filename)

def parse_date_string(date_str: Optional[str]) -> Optional[datetime]:
    """Parse a date string in various formats."""
    return _get_default_parser().parse_date_string(date_str)

def normalize_date_format(date_input: Union[str, datetime, None]) -> Optional[str]:
    """Normalize a date to YYYY-MM-DD format."""
    return _get_default_parser().normalize_date_format(date_input)

def extract_dates_from_content(content: Optional[str], max_dates: int = 5) -> List[str]:
    """Extract dates from document content."""
    return _get_default_parser().extract_dates_from_content(content, max_dates)

def get_most_relevant_date(filename: str, content: str) -> Optional[str]:
    """Get the most relevant date from filename and content."""
    return _get_default_parser().get_most_relevant_date(filename, content)

def _normalize_year(yy: str) -> int:
    """Convert 2-digit year to 4-digit year."""
    return _get_default_parser()._normalize_year(yy)

# Backward compatibility
_MONTHS = DEFAULT_MONTHS.copy()

# ================================================================================
# Module Testing
# ================================================================================

if __name__ == "__main__":
    logger.info("Running date parser module tests")
    
    test_filenames = [
        "report_20240503.pdf",
        "document_May_3_2024.txt",
        "file_3May2024.doc",
        "data_9Oct24.csv",
        "results_2024-05-03.xlsx",
        "REPORT_MAY_3_2024.PDF",
        "doc_15th_December_2024.txt"
    ]
    
    parser = DateParser()
    
    print("\n=== Filename Date Extraction Tests ===")
    for filename in test_filenames:
        date = parser.extract_date_from_filename(filename)
        normalized = parser.normalize_date_format(date)
        print(f"{filename:30} → {normalized}")
    
    test_content = """
    This document was created on May 3, 2024. 
    The previous version was from 3 May 2023.
    EudraCT Number: 2024-123456-12-34 (not a date)
    Version 2.1.5 released on 2024-05-03.
    """
    
    print("\n=== Content Date Extraction Test ===")
    dates = parser.extract_dates_from_content(test_content)
    print(f"Found dates: {dates}")
    
    print("\n=== Statistics ===")
    stats = parser.get_extraction_stats()
    print(f"Stats: {stats}")
    
    logger.info("Tests completed")