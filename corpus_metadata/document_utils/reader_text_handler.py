#!/usr/bin/env python3
"""
Text File Handler - REFACTORED
==============================
Script: corpus_metadata/document_utils/reader_text_handler.py

Purpose:
--------
Handle text files (TXT, CSV, TSV) with two extraction modes:
- intro: First 10,000 characters
- full: Complete file

What it does:
-------------
1. Reads text files
2. Simple extraction based on mode
3. No over-engineering

Usage:
------
handler = TextHandler()
result = handler.process(txt_path, metadata, mode='intro')

Author: Corpus Metadata System
Date: 2024
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TextHandler:
    """Handle text files"""
    
    def __init__(self):
        """Initialize handler"""
        self.intro_chars = 10000  # Characters for intro mode
    
    def can_handle(self, file_extension: str) -> bool:
        """Check if this is a text file"""
        return file_extension.lower() in ['.txt', '.text', '.csv', '.tsv', '.log', '.md']
    
    def process(self, file_path: Path, metadata: Dict[str, Any], mode: str = 'intro') -> Dict[str, Any]:
        """
        Process text file
        
        Args:
            file_path: Path to text file
            metadata: Basic metadata
            mode: 'intro' or 'full'
            
        Returns:
            Extracted content and metadata
        """
        try:
            # Read file
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Limit content for intro mode
            if mode == 'intro' and len(content) > self.intro_chars:
                content = content[:self.intro_chars]
            
            # Update metadata
            metadata['extraction_method'] = 'direct_read'
            metadata['content_length'] = len(content)
            metadata['encoding'] = 'utf-8'
            
            # For CSV/TSV, add line count
            if file_path.suffix.lower() in ['.csv', '.tsv']:
                lines = content.split('\n')
                metadata['line_count'] = len(lines)
                if mode == 'intro':
                    metadata['lines_extracted'] = min(len(lines), 100)
            
            return {
                'content': content,
                'metadata': metadata,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            return {
                'content': '',
                'metadata': metadata,
                'error': f'Failed to read text file: {str(e)}'
            }