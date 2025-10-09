#
# /Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/reader_file_handlers.py
#

"""
===============================================================================================
File Handlers Module
===============================================================================================

Provides specialized handlers for different document types.
Now integrated with YAML configuration from corpus_config
===============================================================================================
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
import mimetypes
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from corpus_metadata.document_utils.reader_pdf_handler import PDFHandler
from corpus_metadata.document_utils.reader_office_handler import WordHandler, ExcelHandler, PowerPointHandler
from corpus_metadata.document_utils.reader_text_handler import TextHandler, CSVHandler, JSONHandler, XMLHandler
from corpus_metadata.document_utils.reader_image_handler import ImageHandler
from corpus_metadata.document_utils.reader_exceptions import UnsupportedFileTypeError
from corpus_metadata.document_utils.metadata_config_loader import CorpusConfig

logger = logging.getLogger(__name__)


class FileHandler(ABC):
    """
    Abstract base class for all file handlers.
    
    Each handler is responsible for processing a specific type of file
    and extracting its content and metadata.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize the file handler with configuration.
        
        Args:
            config: Optional configuration object (for backward compatibility)
        """
        # Load YAML configuration
        self.corpus_config = CorpusConfig()
        
        # Store legacy config if provided
        self.legacy_config = config
    
    @abstractmethod
    def process(self, file_path: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the file and extract content and metadata.
        
        Args:
            file_path: Path to the file to process
            metadata: Pre-extracted common metadata
            
        Returns:
            Dictionary containing processed content and metadata
        """
        pass
    
    @abstractmethod
    def can_handle(self, file_extension: str) -> bool:
        """
        Check if this handler can process files with the given extension.
        
        Args:
            file_extension: File extension (e.g., '.pdf', '.docx')
            
        Returns:
            True if the handler can process this file type
        """
        pass
    
    def get_mime_type(self, file_path: Path) -> str:
        """
        Get the MIME type of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type string
        """
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"


class FileHandlerFactory:
    """
    Factory class for creating appropriate file handlers based on file type.
    
    This factory maintains a registry of available handlers and selects
    the appropriate one based on file extension.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize the factory with configuration.
        
        Args:
            config: Optional configuration object (for backward compatibility)
        """
        # Load YAML configuration
        self.corpus_config = CorpusConfig()
        
        # Store legacy config if provided
        self.legacy_config = config
        
        self._handlers: Dict[str, Type[FileHandler]] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register all default file handlers based on configuration."""
        # Get handler settings from configuration
        handler_config = self.corpus_config.config.get('extraction', {}).get('file_handlers', {})
        
        # PDF handler
        if handler_config.get('pdf', {}).get('enabled', True):
            self.register_handler('.pdf', PDFHandler)
        
        # Office document handlers
        office_config = handler_config.get('office', {})
        if office_config.get('word_enabled', True):
            self.register_handler('.docx', WordHandler)
            self.register_handler('.doc', WordHandler)
        
        if office_config.get('excel_enabled', True):
            self.register_handler('.xlsx', ExcelHandler)
            self.register_handler('.xls', ExcelHandler)
        
        if office_config.get('powerpoint_enabled', True):
            self.register_handler('.pptx', PowerPointHandler)
            self.register_handler('.ppt', PowerPointHandler)
        
        # Text file handlers
        text_config = handler_config.get('text', {})
        if text_config.get('plain_text_enabled', True):
            self.register_handler('.txt', TextHandler)
            self.register_handler('.text', TextHandler)
            self.register_handler('.log', TextHandler)
            self.register_handler('.md', TextHandler)
        
        if text_config.get('csv_enabled', True):
            self.register_handler('.csv', CSVHandler)
            self.register_handler('.tsv', CSVHandler)
            self.register_handler('.tab', CSVHandler)
        
        if text_config.get('json_enabled', True):
            self.register_handler('.json', JSONHandler)
            self.register_handler('.jsonl', JSONHandler)
            self.register_handler('.geojson', JSONHandler)
        
        if text_config.get('xml_enabled', True):
            self.register_handler('.xml', XMLHandler)
            self.register_handler('.xhtml', XMLHandler)
            self.register_handler('.svg', XMLHandler)
        
        # Image handlers
        image_config = handler_config.get('image', {})
        if image_config.get('enabled', True):
            supported_formats = image_config.get('supported_formats', [
                '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif'
            ])
            for ext in supported_formats:
                self.register_handler(ext, ImageHandler)
        
        logger.info(f"Registered {len(self._handlers)} file handlers")
    
    def register_handler(self, extension: str, handler_class: Type[FileHandler]) -> None:
        """
        Register a new file handler for a specific extension.
        
        Args:
            extension: File extension (e.g., '.pdf')
            handler_class: Handler class that can process this file type
        """
        self._handlers[extension.lower()] = handler_class
        logger.debug(f"Registered handler {handler_class.__name__} for {extension}")
    
    def get_handler(self, extension: str) -> FileHandler:
        """
        Get an appropriate handler instance for the given file extension.
        
        Args:
            extension: File extension (e.g., '.pdf')
            
        Returns:
            FileHandler instance capable of processing the file type
            
        Raises:
            UnsupportedFileTypeError: If no handler is available for the extension
        """
        extension = extension.lower()
        handler_class = self._handlers.get(extension)
        
        if not handler_class:
            # Check if extension is in disabled list
            disabled_extensions = self.corpus_config.config.get(
                'extraction', {}
            ).get('file_handlers', {}).get('disabled_extensions', [])
            
            if extension in disabled_extensions:
                raise UnsupportedFileTypeError(
                    f"File type {extension} is disabled in configuration"
                )
            else:
                raise UnsupportedFileTypeError(
                    f"No handler available for {extension} files"
                )
        
        # Create handler instance
        # Pass legacy config if available for backward compatibility
        if self.legacy_config:
            return handler_class(self.legacy_config)
        else:
            return handler_class()
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of all supported file extensions.
        
        Returns:
            List of supported extensions
        """
        return list(self._handlers.keys())
    
    def get_available_handlers(self) -> Dict[str, str]:
        """
        Get information about available handlers.
        
        Returns:
            Dictionary mapping extensions to handler class names
        """
        return {
            ext: handler.__name__ 
            for ext, handler in self._handlers.items()
        }
    
    def is_format_supported(self, extension: str) -> bool:
        """
        Check if a file format is supported.
        
        Args:
            extension: File extension to check
            
        Returns:
            True if format is supported
        """
        return extension.lower() in self._handlers
    
    def get_handler_info(self, extension: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific handler.
        
        Args:
            extension: File extension
            
        Returns:
            Dictionary with handler information
        """
        extension = extension.lower()
        
        if extension not in self._handlers:
            return {
                'supported': False,
                'extension': extension,
                'reason': 'No handler registered'
            }
        
        handler_class = self._handlers[extension]
        
        # Get handler-specific configuration
        handler_type = None
        if handler_class == PDFHandler:
            handler_type = 'pdf'
        elif handler_class in [WordHandler, ExcelHandler, PowerPointHandler]:
            handler_type = 'office'
        elif handler_class in [TextHandler, CSVHandler, JSONHandler, XMLHandler]:
            handler_type = 'text'
        elif handler_class == ImageHandler:
            handler_type = 'image'
        
        handler_config = {}
        if handler_type:
            handler_config = self.corpus_config.config.get(
                'extraction', {}
            ).get('file_handlers', {}).get(handler_type, {})
        
        return {
            'supported': True,
            'extension': extension,
            'handler_class': handler_class.__name__,
            'handler_type': handler_type,
            'configuration': handler_config,
            'features': self._get_handler_features(handler_class)
        }
    
    def _get_handler_features(self, handler_class: Type[FileHandler]) -> List[str]:
        """
        Get list of features supported by a handler.
        
        Args:
            handler_class: Handler class
            
        Returns:
            List of supported features
        """
        features = []
        
        # Check common features
        if hasattr(handler_class, 'extract_metadata'):
            features.append('metadata_extraction')
        
        if hasattr(handler_class, 'extract_text'):
            features.append('text_extraction')
        
        # Handler-specific features
        if handler_class == PDFHandler:
            features.extend(['ocr', 'table_extraction', 'layout_analysis'])
        elif handler_class == ImageHandler:
            features.extend(['ocr', 'image_analysis'])
        elif handler_class in [ExcelHandler, CSVHandler]:
            features.extend(['data_analysis', 'statistics'])
        elif handler_class == JSONHandler:
            features.extend(['schema_extraction', 'validation'])
        
        return features