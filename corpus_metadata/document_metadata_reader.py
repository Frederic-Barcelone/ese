#!/usr/bin/env python3
"""
=======================
Document Reader Module - WITH CENTRALIZED LOGGING
=======================
/Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_metadata_reader.py

Purpose:
--------
Simple document reader that supports two extraction modes for ALL file types:
- intro: Quick extraction (first 10 pages for PDFs, partial content for others)
- full: Complete extraction (all content)

NOW USES CENTRALIZED LOGGING - no separate reader_extraction.log file

What it does:
-------------
1. Reads any document type (PDF, DOCX, TXT, etc.)
2. Extracts content based on mode (intro/full)
3. Returns content and metadata
4. Simple, clean, no over-engineering

Supported formats:
------------------
- PDF: intro (10 pages) / full (all pages)
- DOCX/DOC: intro (first 10k chars) / full (all content)
- TXT: intro (first 10k chars) / full (all content)
- Others: same pattern

Usage:
------
reader = DocumentReader()
result = reader.read_document('myfile.pdf', mode='intro')
result = reader.read_document('myfile.pdf', mode='full')

Author: Corpus Metadata System
Date: 2024
"""

from pathlib import Path
from typing import Dict, Any, Optional

# ============================================================================
# USE CENTRALIZED LOGGING SYSTEM
# ============================================================================
try:
    from corpus_metadata.document_utils.metadata_logging_config import get_logger
    logger = get_logger('corpus_reader')
except ImportError:
    # Fallback if centralized logging not available
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # Keep console quiet
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('corpus_reader')


class DocumentReader:
    """Simple document reader with intro/full extraction modes"""
    
    def __init__(self, config_path: str = "corpus_config/main_config.yaml"):
        """Initialize reader"""
        self.config_path = config_path
        self.handlers = self._init_handlers()
        logger.debug("DocumentReader initialized")
    
    def _init_handlers(self) -> Dict[str, Any]:
        """Initialize file handlers"""
        handlers = {}
        
        # PDF handler
        try:
            from corpus_metadata.document_utils.reader_pdf_handler import PDFHandler
            handlers['.pdf'] = PDFHandler()
            logger.debug("PDF handler initialized")
        except ImportError as e:
            logger.warning(f"PDF handler not available: {e}")
        
        # Office handlers
        try:
            from corpus_metadata.document_utils.reader_office_handler import (
                WordHandler, ExcelHandler, PowerPointHandler
            )
            handlers['.docx'] = WordHandler()
            handlers['.doc'] = WordHandler()
            handlers['.xlsx'] = ExcelHandler()
            handlers['.xls'] = ExcelHandler()
            handlers['.pptx'] = PowerPointHandler()
            handlers['.ppt'] = PowerPointHandler()
            logger.debug("Office handlers initialized")
        except ImportError as e:
            logger.warning(f"Office handlers not available: {e}")
        
        # Text handlers
        try:
            from corpus_metadata.document_utils.reader_text_handler import TextHandler
            handlers['.txt'] = TextHandler()
            handlers['.text'] = TextHandler()
            handlers['.log'] = TextHandler()
            handlers['.md'] = TextHandler()
            logger.debug("Text handlers initialized")
        except ImportError as e:
            logger.warning(f"Text handlers not available: {e}")
        
        logger.debug(f"Initialized {len(handlers)} file handlers")
        return handlers
    
    def read_document(self, file_path: str, mode: str = 'intro') -> Dict[str, Any]:
        """
        Read document and extract content.
        
        Args:
            file_path: Path to document
            mode: 'intro' or 'full' extraction mode
            
        Returns:
            Dictionary with content, metadata, and error info
        """
        file_path = Path(file_path)
        
        # Basic validation
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            return {
                'content': '',
                'metadata': {'filename': file_path.name},
                'error': error_msg
            }
        
        # Get basic metadata
        metadata = {
            'filename': file_path.name,
            'size': file_path.stat().st_size,
            'extension': file_path.suffix.lower(),
            'mode': mode
        }
        
        # Find appropriate handler
        handler = self.handlers.get(metadata['extension'])
        
        if not handler:
            error_msg = f"No handler for file type: {metadata['extension']}"
            logger.warning(error_msg)
            
            # Try to read as text as fallback
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if mode == 'intro':
                        content = content[:10000]
                    
                    logger.debug(f"Read {file_path.name} as text fallback")
                    return {
                        'content': content,
                        'metadata': metadata,
                        'error': None
                    }
            except Exception as e:
                error_msg = f"Failed to read file: {e}"
                logger.error(error_msg)
                return {
                    'content': '',
                    'metadata': metadata,
                    'error': error_msg
                }
        
        # Process with handler
        try:
            logger.debug(f"Processing {file_path.name} with {handler.__class__.__name__} in {mode} mode")
            result = handler.process(file_path, metadata, mode)
            
            # Log success/failure (to file)
            if result.get('error'):
                logger.warning(f"Handler returned error for {file_path.name}: {result['error']}")
            else:
                content_len = len(result.get('content', ''))
                logger.debug(f"Successfully extracted {content_len:,} chars from {file_path.name}")
            
            return result
            
        except Exception as e:
            error_msg = f"Handler failed: {e}"
            logger.error(f"Handler exception for {file_path.name}: {e}")
            return {
                'content': '',
                'metadata': metadata,
                'error': error_msg
            }
    
    def get_supported_extensions(self) -> list:
        """Get list of supported file extensions"""
        return list(self.handlers.keys())