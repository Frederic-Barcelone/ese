#
# dUsers/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/reader_base.py
#

"""
Base classes for document readers to avoid circular imports.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class FileHandler(ABC):
    """
    Abstract base class for all file handlers.
    
    This class defines the interface that all file handlers must implement.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize the file handler.
        
        Args:
            config: Optional configuration object
        """
        self.config = config
    
    @abstractmethod
    def process(self, file_path: Path, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a file and extract its content and metadata.
        
        Args:
            file_path: Path to the file to process
            file_info: Basic file information dictionary
            
        Returns:
            Dictionary containing processed file information
        """
        pass
    
    @abstractmethod
    def supports_file(self, file_path: Path) -> bool:
        """
        Check if this handler supports the given file type.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if this handler can process the file, False otherwise
        """
        pass
    
    def get_file_type(self, file_path: Path) -> str:
        """
        Get a human-readable description of the file type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            String description of the file type
        """
        return file_path.suffix.upper().lstrip('.') + ' File'
    
    def extract_basic_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract basic file information that all handlers can provide.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with basic file information
        """
        try:
            stat = file_path.stat()
            return {
                'name': file_path.name,
                'path': str(file_path),
                'extension': file_path.suffix.lower(),
                'size_bytes': stat.st_size,
                'modified_time': stat.st_mtime,
                'is_binary': True  # Default to binary, handlers can override
            }
        except Exception as e:
            return {
                'name': file_path.name,
                'path': str(file_path),
                'extension': file_path.suffix.lower(),
                'error': f'Failed to get file info: {str(e)}'
            }


class ValidationConfig:
    """Configuration class for file validation."""
    
    def __init__(self, 
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 allowed_extensions: Optional[set] = None,
                 use_magic: bool = True):
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions or set()
        self.use_magic = use_magic