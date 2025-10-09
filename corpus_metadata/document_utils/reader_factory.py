#
# /Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/reader_factory.py
#


"""
Document Reader Factory
======================

Factory for creating document readers.
Bridges the gap between the expected import location and the actual DocumentReader class.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path to find document_metadata_reader
# sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the full document reader
try:
    from corpus_metadata.document_metadata_reader import DocumentReader
    READER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import DocumentReader: {e}")
    READER_AVAILABLE = False
    DocumentReader = None


class DocumentReaderFactory:
    """
    Factory for creating document readers.
    
    This factory creates properly configured DocumentReader instances
    that can handle multiple file formats including PDFs, Office documents,
    and text files.
    """
    
    @staticmethod
    def create_reader(config: Optional[Dict[str, Any]] = None):
        """
        Create a document reader instance.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            DocumentReader instance configured from YAML files
            
        Raises:
            ImportError: If DocumentReader is not available
        """
        if not READER_AVAILABLE:
            raise ImportError(
                "DocumentReader not available. Please ensure document_metadata_reader.py "
                "and all its dependencies are properly installed."
            )
        
        # Create reader with configuration
        if config:
            # If specific config provided, use it
            return DocumentReader(
                max_file_size_mb=config.get('max_file_size_mb'),
                pdf_config=config.get('pdf_config')
            )
        else:
            # Default: Create reader that loads from YAML configuration
            return DocumentReader()
    
    @staticmethod
    def create_reader_with_config(max_file_size_mb: Optional[int] = None,
                                 pdf_config: Optional[Dict[str, Any]] = None):
        """
        Create a document reader with specific configuration overrides.
        
        Args:
            max_file_size_mb: Maximum file size in MB
            pdf_config: PDF-specific configuration
            
        Returns:
            Configured DocumentReader instance
        """
        if not READER_AVAILABLE:
            raise ImportError("DocumentReader not available")
        
        return DocumentReader(
            max_file_size_mb=max_file_size_mb,
            pdf_config=pdf_config
        )