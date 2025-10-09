#
# /Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/reader_utils.py
#
"""
Reader Utilities Module
======================

Common utility functions for the document reader system.
"""

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import mimetypes
import logging

logger = logging.getLogger(__name__)


class FileUtils:
    """Utility functions for file operations."""
    
    @staticmethod
    def calculate_checksum(file_path: Path, algorithm: str = 'sha256') -> str:
        """
        Calculate file checksum.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm to use
            
        Returns:
            Hexadecimal checksum string
        """
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    @staticmethod
    def get_file_type(file_path: Path) -> str:
        """
        Get file type using multiple methods.
        
        Args:
            file_path: Path to file
            
        Returns:
            File type description
        """
        # Try python-magic if available
        try:
            import magic
            return magic.from_file(str(file_path))
        except ImportError:
            pass
        
        # Fallback to mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            return mime_type
        
        # Use extension
        return f"{file_path.suffix.upper()[1:]} file" if file_path.suffix else "Unknown"
    
    @staticmethod
    def safe_file_name(filename: str, max_length: int = 255) -> str:
        """
        Create a safe filename by removing invalid characters.
        
        Args:
            filename: Original filename
            max_length: Maximum filename length
            
        Returns:
            Safe filename
        """
        # Remove non-ASCII characters
        filename = unicodedata.normalize('NFKD', filename)
        filename = filename.encode('ascii', 'ignore').decode('ascii')
        
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove multiple spaces/underscores
        filename = re.sub(r'[_\s]+', '_', filename)
        
        # Trim to max length
        if len(filename) > max_length:
            name, ext = Path(filename).stem, Path(filename).suffix
            max_name_length = max_length - len(ext) - 1
            filename = name[:max_name_length] + ext
        
        return filename.strip('_')


class TextUtils:
    """Utility functions for text processing."""
    
    @staticmethod
    def extract_sentences(text: str, max_sentences: int = 10) -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text: Input text
            max_sentences: Maximum number of sentences to extract
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Simple sentence splitting (can be improved with NLTK if available)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences[:max_sentences]
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 20) -> List[str]:
        """
        Extract potential keywords from text.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords
            
        Returns:
            List of keywords
        """
        if not text:
            return []
        
        # Convert to lowercase and split
        words = text.lower().split()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'cannot'
        }
        
        # Filter words
        keywords = []
        for word in words:
            # Remove punctuation
            word = re.sub(r'[^\w\s]', '', word)
            
            # Skip if too short, stop word, or number
            if len(word) > 3 and word not in stop_words and not word.isdigit():
                keywords.append(word)
        
        # Count frequency
        from collections import Counter
        word_freq = Counter(keywords)
        
        # Return most common
        return [word for word, _ in word_freq.most_common(max_keywords)]
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing extra whitespace and normalizing.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        # Fix quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """
        Truncate text to maximum length at word boundary.
        
        Args:
            text: Input text
            max_length: Maximum length
            suffix: Suffix to add if truncated
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        # Find last space before max_length
        truncate_at = text.rfind(' ', 0, max_length - len(suffix))
        
        if truncate_at == -1:
            # No space found, hard truncate
            return text[:max_length - len(suffix)] + suffix
        
        return text[:truncate_at] + suffix


class MetadataUtils:
    """Utility functions for metadata handling."""
    
    @staticmethod
    def merge_metadata(base: Dict[str, Any], *updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple metadata dictionaries.
        
        Args:
            base: Base metadata dictionary
            *updates: Additional metadata dictionaries to merge
            
        Returns:
            Merged metadata dictionary
        """
        result = base.copy()
        
        for update in updates:
            for key, value in update.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    # Recursively merge dictionaries
                    result[key] = MetadataUtils.merge_metadata(result[key], value)
                else:
                    result[key] = value
        
        return result
    
    @staticmethod
    def filter_metadata(metadata: Dict[str, Any], 
                       exclude_keys: Optional[List[str]] = None,
                       include_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Filter metadata dictionary.
        
        Args:
            metadata: Original metadata
            exclude_keys: Keys to exclude
            include_keys: Keys to include (if specified, only these are included)
            
        Returns:
            Filtered metadata
        """
        if include_keys:
            return {k: v for k, v in metadata.items() if k in include_keys}
        
        if exclude_keys:
            return {k: v for k, v in metadata.items() if k not in exclude_keys}
        
        return metadata.copy()
    
    @staticmethod
    def flatten_metadata(metadata: Dict[str, Any], 
                        prefix: str = "",
                        separator: str = "_") -> Dict[str, Any]:
        """
        Flatten nested metadata dictionary.
        
        Args:
            metadata: Nested metadata dictionary
            prefix: Prefix for keys
            separator: Separator between nested keys
            
        Returns:
            Flattened dictionary
        """
        result = {}
        
        for key, value in metadata.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively flatten
                result.update(
                    MetadataUtils.flatten_metadata(value, new_key, separator)
                )
            else:
                result[new_key] = value
        
        return result