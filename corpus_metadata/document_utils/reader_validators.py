#
# /Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/reader_validators.py
#

"""
Reader Validators Module - SIMPLIFIED
=====================================
Simple file validation for document reader
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Simple validation error"""
    pass


class FileValidator:
    """
    Simple file validator for document processing
    """
    
    def __init__(self, config: Any = None):
        """Initialize validator with config"""
        self.config = config
        
        # Set defaults if no config
        if not config:
            self.max_file_size_mb = 100
            self.min_file_size_bytes = 10
            self.allowed_extensions = ['.pdf', '.docx', '.doc', '.txt', '.csv', '.json']
        else:
            self.max_file_size_mb = getattr(config, 'max_file_size_mb', 100)
            self.min_file_size_bytes = getattr(config, 'min_file_size_bytes', 10)
            self.allowed_extensions = getattr(config, 'allowed_extensions', ['.pdf', '.docx', '.doc', '.txt'])
    
    def validate_file(self, file_path: Path) -> bool:
        """
        Validate a file for processing
        
        Args:
            file_path: Path to file
            
        Returns:
            True if valid
            
        Raises:
            ValidationError if invalid
        """
        # Check existence
        if not file_path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
        
        # Check if it's a file
        if not file_path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
        
        # Check size
        file_size = file_path.stat().st_size
        if file_size < self.min_file_size_bytes:
            raise ValidationError(f"File too small: {file_size} bytes")
        
        max_size_bytes = self.max_file_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            raise ValidationError(f"File too large: {file_size / (1024*1024):.1f} MB")
        
        # Check extension
        if not self.is_supported_file(file_path):
            raise ValidationError(f"Unsupported file type: {file_path.suffix}")
        
        return True
    
    def is_supported_file(self, file_path: Path) -> bool:
        """Check if file type is supported"""
        extension = file_path.suffix.lower()
        return extension in self.allowed_extensions
    
    def calculate_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file hash"""
        hash_func = getattr(hashlib, algorithm)()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def validate_content(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple content validation
        
        Args:
            content: Extracted text
            metadata: Document metadata
            
        Returns:
            Validation results
        """
        results = {
            'passed': True,
            'flags': {},
            'confidence': 'High'
        }
        
        # Check minimum text length
        if len(content) < 100:
            results['flags']['insufficient_text'] = True
            results['confidence'] = 'Low'
            
        return results


# For backward compatibility
class ContentQualityValidator:
    """Simple content quality validator"""
    
    def __init__(self, quality_config: Dict[str, Any] = None):
        self.config = quality_config or {}
    
    def validate_extraction_confidence(self, extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple confidence validation"""
        return {
            'meets_threshold': True,
            'total_count': len(extractions)
        }