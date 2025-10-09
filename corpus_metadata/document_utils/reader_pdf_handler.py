#!/usr/bin/env python3
"""
================================
PDF Handler Module - WITH CENTRALIZED LOGGING
================================
Script: corpus_metadata/document_utils/reader_pdf_handler.py

Purpose:
--------
Handle PDF files with two extraction modes:
- intro: First 10 pages (for quick analysis)  
- full: All pages (for complete extraction)

NOW USES CENTRALIZED LOGGING SYSTEM - clean console output
"""

import time
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# ============================================================================
# USE CENTRALIZED LOGGING SYSTEM
# ============================================================================
try:
    from corpus_metadata.document_utils.metadata_logging_config import (
        get_logger,
        log_separator,
        log_file_start,
        log_file_end,
        log_metric
    )
    logger = get_logger('reader_pdf_handler')
except ImportError:
    # Fallback if centralized logging not available
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # Keep console quiet
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('reader_pdf_handler')
    
    def log_separator(logger, style='minor'):
        pass
    
    def log_file_start(logger, filename, file_num=None, total=None):
        logger.info(f"Processing: {filename}")
    
    def log_file_end(logger, filename, success=True, time_taken=None):
        status = "Completed" if success else "Failed"
        logger.info(f"{status}: {filename}")
    
    def log_metric(logger, name, value, unit=''):
        logger.debug(f"{name}: {value}{unit}")


def has_spacing_issues(text: str, sample_size: int = 500) -> bool:
    """Enhanced heuristic to detect spacing issues in a text snippet."""
    sample = text[:sample_size] if text else ""
    
    # More sensitive detection of concatenated words
    concatenated_patterns = [
        r'[a-z]{3,}with[a-z]{3,}',
        r'[a-z]{3,}and[a-z]{3,}',
        r'[a-z]{3,}or[a-z]{3,}',
        r'[a-z]{3,}for[a-z]{3,}',
        r'[a-z]{3,}the[a-z]{3,}',
        r'[a-z]{3,}that[a-z]{3,}',
        r'[a-z]{3,}based[a-z]{3,}',
        r'clinical[a-z]{3,}',
        r'[a-z]{3,}clinical',
    ]
    
    camel_case_names = len(re.findall(r'[A-Z][a-z]+[A-Z][a-z]+', sample))
    long_lowercase = len(re.findall(r'[a-z]{15,}', sample))
    no_space_parens = len(re.findall(r'[a-zA-Z]\(', sample))
    caps_concat = len(re.findall(r'[A-Z]{3,}[a-z]', sample))
    
    concat_words = sum(len(re.findall(pattern, sample, re.IGNORECASE)) for pattern in concatenated_patterns)
    
    return (
        camel_case_names > 0 or
        long_lowercase > 0 or
        no_space_parens > 1 or
        caps_concat > 0 or
        concat_words > 0
    )


class PDFHandler:
    """Simple PDF handler with intro/full modes and text cleaning"""
    
    def __init__(self):
        """Initialize handler"""
        self.intro_pages = 10  # Pages for intro mode
        
        # Import PDFTextCleaner from extractors module
        try:
            from corpus_metadata.document_utils.reader_pdf_extractors import PDFTextCleaner
            self.cleaner = PDFTextCleaner()
        except ImportError:
            logger.debug("PDFTextCleaner not available, using basic cleaning")
            self.cleaner = None
        
        # Initialize available extractors
        self.extractors = []
        self._init_extractors()
        
        logger.debug("PDFHandler initialized")
    
    def _init_extractors(self):
        """Initialize available PDF extractors"""
        # Try to import extractors
        try:
            from corpus_metadata.document_utils.reader_pdf_extractors import (
                PdfplumberExtractor, PyMuPDFExtractor
            )
            
            # Try pdfplumber first (usually better for text)
            try:
                extractor = PdfplumberExtractor()
                if extractor.available:
                    self.extractors.append(extractor)
                    logger.debug("pdfplumber extractor available")
            except:
                pass
            
            # Try PyMuPDF as fallback
            try:
                extractor = PyMuPDFExtractor()
                if extractor.available:
                    self.extractors.append(extractor)
                    logger.debug("PyMuPDF extractor available")
            except:
                pass
                
        except ImportError:
            logger.warning("PDF extractors module not found")
        
        if not self.extractors:
            logger.warning("No PDF extractors available. Install pdfplumber or PyMuPDF")
        else:
            logger.debug(f"PDF handler ready with {len(self.extractors)} extractor(s)")
    
    def can_handle(self, file_extension: str) -> bool:
        """Check if this is a PDF."""
        return file_extension.lower() == '.pdf'
    
    def process(self, file_path: Path, metadata: Dict[str, Any], mode: str = 'intro') -> Dict[str, Any]:
        """
        Process PDF file with improved text extraction.
        
        Args:
            file_path: Path to PDF
            metadata: Basic file metadata
            mode: 'intro' (10 pages) or 'full' (all pages)
            
        Returns:
            Dictionary with content, metadata, and error info
        """
        start_time = time.time()
        
        # Log to file only (debug level)
        log_separator(logger, "major")
        logger.debug(f"PDF EXTRACTION START: {file_path.name}")
        log_separator(logger, "major")
        logger.debug(f"Mode: {mode}")
        logger.debug(f"File size: {metadata.get('size', 0):,} bytes")
        
        # Determine pages to extract
        max_pages = None if mode == 'full' else self.intro_pages
        logger.debug(f"Max pages to extract: {max_pages if max_pages else 'All'}")
        
        # Try each extractor
        last_error = None
        for extractor in self.extractors:
            try:
                logger.debug(f"Attempting {extractor.name} extraction...")
                content, pages, info = extractor.extract(file_path, max_pages)
                
                # Update metadata
                metadata['pages'] = pages
                metadata['extraction_method'] = info.get('extraction_method')
                metadata['extraction_info'] = info
                
                # Log success (to file)
                logger.debug(f"Text cleaning: {info.get('characters_extracted', 0):,} -> {len(content):,} chars ({(1 - len(content)/info.get('characters_extracted', 1))*100:.1f}% reduction)")
                logger.debug(f"Extraction successful in {info.get('extraction_time', 0):.2f}s")
                
                log_separator(logger, "major")
                logger.debug(f"PDF EXTRACTION COMPLETE: {file_path.name}")
                log_separator(logger, "major")
                
                # Record metrics (to file only)
                log_metric(logger, 'extraction_time', info.get('extraction_time', 0), 's')
                log_metric(logger, 'pages_extracted', pages)
                log_metric(logger, 'content_length', len(content), ' chars')
                
                return {
                    'content': content,
                    'metadata': metadata,
                    'error': None
                }
                
            except Exception as e:
                last_error = str(e)
                logger.debug(f"{extractor.name} failed: {e}")
                continue
        
        # No extractors worked
        error_msg = f'PDF extraction failed. Last error: {last_error}' if last_error else 'No PDF extractors available'
        logger.error(error_msg)
        
        log_separator(logger, "major")
        logger.debug(f"PDF EXTRACTION FAILED: {file_path.name}")
        log_separator(logger, "major")
        
        return {
            'content': '',
            'metadata': metadata,
            'error': error_msg
        }
    
    def _basic_clean(self, text: str) -> str:
        """Basic text cleaning if PDFTextCleaner not available"""
        if not text:
            return ""
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Fix hyphenated line breaks
        text = re.sub(r'-\s*\n\s*', '', text)
        
        # Fix regular line breaks within sentences
        text = re.sub(r'(?<=[a-z])\n(?=[a-z])', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        
        # Fix spacing issues if detected
        if has_spacing_issues(text):
            # Add spaces before capitals in camelCase
            text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
            # Add space before parentheses
            text = re.sub(r'([a-zA-Z])(\()', r'\1 \2', text)
            # Fix ALL CAPS followed by lowercase
            text = re.sub(r'([A-Z]{2,})([a-z])', r'\1 \2', text)
        
        # Final cleanup
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()
        
        return text