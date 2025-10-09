"""
Reader Metadata Module - FIXED VERSION
=======================================
corpus_metadata/document_utils/reader_metadata.py
Handles extraction of file metadata across different formats.
Key fixes:
1. extract_pdf_metadata is now a proper method, not a boolean
2. extract_metadata accepts both Path objects and strings
3. Added parse_pdf_date method for proper date handling
4. Uses centralized logging
5. Added standalone extract_pdf_metadata function for entity_extraction.py
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib
import mimetypes

# Use centralized logging
try:
    from corpus_metadata.document_utils.metadata_logging_config import get_logger
    logger = get_logger('metadata_extractor')
except ImportError:
    # Fallback if centralized logging not available
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)

# Simple exception class
class ExtractionError(Exception):
    """Simple extraction error exception"""
    pass

# Optional imports
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

class MetadataExtractor:
    """
    Extracts metadata from various file formats.
    Simple, clean, no over-engineering.
    """
    
    def __init__(self, config=None):
        """Initialize with sensible defaults"""
        self.config = config or {}
        logger.debug("MetadataExtractor initialized")
    
    def extract_metadata(self, file_path) -> Dict[str, Any]:
        """
        Main extraction method - works with Path or string
        
        Args:
            file_path: Path to file (string or Path object)
            
        Returns:
            Dictionary containing metadata
        """
        # Convert to Path if string
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        return self.extract_common_metadata(file_path)
    
    def extract_common_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract common metadata from any file
        
        Args:
            file_path: Path object to the file
            
        Returns:
            Dictionary containing common metadata
        """
        try:
            # Ensure it's a Path object
            if isinstance(file_path, str):
                file_path = Path(file_path)
                
            stat_info = file_path.stat()
            
            metadata = {
                'name': file_path.name,
                'path': str(file_path),
                'extension': file_path.suffix.lower(),
                'size_bytes': stat_info.st_size,
                'size_readable': self._format_file_size(stat_info.st_size),
                'created_time': datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                'modified_time': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                'accessed_time': datetime.fromtimestamp(stat_info.st_atime).isoformat(),
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            # Detect MIME type
            mime_type, encoding = mimetypes.guess_type(str(file_path))
            metadata['mime_type'] = mime_type
            metadata['mime_encoding'] = encoding
            
            # Extract PDF metadata if it's a PDF
            if file_path.suffix.lower() == '.pdf' and PYPDF2_AVAILABLE:
                pdf_metadata = self._extract_pdf_file_metadata(file_path)
                metadata.update(pdf_metadata)
            
            logger.debug(f"Extracted metadata for {file_path.name}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {e}")
            raise ExtractionError(f"Failed to extract metadata: {e}")
    
    def extract_pdf_metadata(self, pdf_reader) -> Dict[str, Any]:
        """
        Extract metadata from PyPDF2 reader object
        
        Args:
            pdf_reader: PyPDF2 PdfReader instance
            
        Returns:
            Dictionary containing PDF-specific metadata
        """
        metadata = {}
        
        if not PYPDF2_AVAILABLE:
            logger.warning("PyPDF2 not available")
            return metadata
        
        try:
            # Get metadata from PDF
            if hasattr(pdf_reader, 'metadata') and pdf_reader.metadata:
                info = pdf_reader.metadata
                
                # Standard PDF metadata fields
                metadata_fields = {
                    '/Title': 'pdf_title',
                    '/Author': 'pdf_author', 
                    '/Subject': 'pdf_subject',
                    '/Creator': 'pdf_creator',
                    '/Producer': 'pdf_producer',
                    '/Keywords': 'pdf_keywords',
                    '/CreationDate': 'pdf_creation_date',
                    '/ModDate': 'pdf_modified_date'
                }
                
                for pdf_key, metadata_key in metadata_fields.items():
                    if pdf_key in info:
                        value = info[pdf_key]
                        # Parse dates properly
                        if 'Date' in pdf_key:
                            value = self.parse_pdf_date(value)
                        metadata[metadata_key] = str(value) if value else ''
            
            # Check encryption
            if hasattr(pdf_reader, 'is_encrypted'):
                metadata['pdf_is_encrypted'] = pdf_reader.is_encrypted
            
            # Get page count
            if hasattr(pdf_reader, 'pages'):
                metadata['pdf_page_count'] = len(pdf_reader.pages)
            
            # Get PDF version
            if hasattr(pdf_reader, 'pdf_header'):
                metadata['pdf_version'] = pdf_reader.pdf_header
            
            logger.debug(f"Extracted PDF metadata: {len(metadata)} fields")
            
        except Exception as e:
            logger.error(f"Failed to extract PDF metadata: {e}")
            metadata['pdf_metadata_error'] = str(e)
        
        return metadata
    
    def _extract_pdf_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract PDF metadata from a file path
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with PDF metadata
        """
        metadata = {}
        
        try:
            with open(file_path, 'rb') as f:
                if hasattr(PyPDF2, 'PdfReader'):
                    pdf_reader = PyPDF2.PdfReader(f)
                else:
                    pdf_reader = PyPDF2.PdfFileReader(f)
                
                metadata = self.extract_pdf_metadata(pdf_reader)
                
        except Exception as e:
            logger.error(f"Failed to read PDF file {file_path}: {e}")
            metadata['pdf_error'] = str(e)
        
        return metadata
    
    def parse_pdf_date(self, date_value) -> str:
        """
        Parse PDF date format to ISO format
        
        PDF dates look like: D:20250622103511+03'00'
        
        Args:
            date_value: PDF date string or object
            
        Returns:
            ISO formatted date string
        """
        if not date_value:
            return ""
        
        # Convert to string if needed
        date_str = str(date_value)
        
        # PDF date regex: D:YYYYMMDDHHmmSS[+-Z]HH'mm'
        pattern = r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})"
        match = re.match(pattern, date_str)
        
        if match:
            year, month, day, hour, minute, second = match.groups()
            try:
                dt = datetime(
                    int(year), int(month), int(day),
                    int(hour), int(minute), int(second)
                )
                return dt.isoformat()
            except ValueError:
                logger.warning(f"Invalid date values in PDF date: {date_str}")
                return date_str
        
        # Return original if can't parse
        return date_str
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"


# ============================================================================
# STANDALONE FUNCTION FOR ENTITY_EXTRACTION.PY
# ============================================================================
def extract_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """
    Standalone function to extract PDF metadata from a file path.
    This is what entity_extraction.py imports and uses.
    
    Args:
        file_path: Path to PDF file as string
        
    Returns:
        Dictionary containing all metadata (file + PDF specific)
    """
    try:
        extractor = MetadataExtractor()
        metadata = extractor.extract_metadata(file_path)
        logger.info(f"Extracted {len(metadata)} metadata fields from {Path(file_path).name}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to extract metadata from {file_path}: {e}")
        return {}


# Simple test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
        # Test the standalone function
        print("Testing standalone function:")
        metadata = extract_pdf_metadata(file_path)
        print(f"Extracted {len(metadata)} metadata fields:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")