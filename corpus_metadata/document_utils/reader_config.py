#
# /Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/reader_config.py
#

"""
Reader Configuration Module
==========================

Provides configuration management for the document reader system.
Now integrated with main_config.yaml and cache_settings.yaml
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import os
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from corpus_metadata.document_utils.metadata_config_loader import CorpusConfig


@dataclass
class CacheConfig:
    """Configuration for document caching system."""
    enabled: bool = True
    max_size_mb: int = 500
    ttl_seconds: int = 3600
    cache_dir: str = ".cache/document_reader"
    
    @classmethod
    def from_corpus_config(cls, corpus_config: CorpusConfig) -> 'CacheConfig':
        """Create CacheConfig from CorpusConfig"""
        cache_storage = corpus_config.get_cache_storage_config()
        cache_behavior = corpus_config.get_cache_behavior_config()
        
        # Convert max_size_gb to mb
        max_size_mb = cache_storage['max_size_gb'] * 1024
        
        return cls(
            enabled=corpus_config.get('features.enable_document_caching', 
                         corpus_config.get('features.enable_caching', True)),
            max_size_mb=max_size_mb,
            ttl_seconds=cache_behavior.get('expiration_days', {}).get('documents', 30) * 86400,
            cache_dir=cache_storage['base_directory']
        )


@dataclass
class ValidationConfig:
    """Configuration for file validation."""
    max_file_size_mb: int = 100
    min_file_size_bytes: int = 10
    check_file_integrity: bool = True
    allowed_extensions: List[str] = field(default_factory=lambda: [
        '.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt',
        '.txt', '.csv', '.json', '.xml', '.png', '.jpg', '.jpeg'
    ])
    
    @classmethod
    def from_corpus_config(cls, corpus_config: CorpusConfig) -> 'ValidationConfig':
        """Create ValidationConfig from CorpusConfig"""
        runtime_config = corpus_config.get_runtime_config()
        quality_config = corpus_config.get_quality_checks_config()
        
        return cls(
            max_file_size_mb=runtime_config.get('max_file_size_mb', 0),
            min_file_size_bytes=quality_config.get('min_text_length', None),
            check_file_integrity=True,  # Always check
            allowed_extensions=[
                '.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt',
                '.txt', '.csv', '.json', '.xml', '.png', '.jpg', '.jpeg'
            ]
        )


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    enabled: bool = True
    dpi: int = 300
    timeout_seconds: int = 45
    language: str = 'eng'
    languages: List[str] = field(default_factory=lambda: ['eng'])
    psm_mode: int = 3  # Page segmentation mode
    oem_mode: int = 3  # OCR engine mode
    enable_preprocessing: bool = True
    confidence_threshold: float = 0.6
    
    @classmethod
    def from_corpus_config(cls, corpus_config: CorpusConfig) -> 'OCRConfig':
        """Create OCRConfig from CorpusConfig"""
        pdf_config = corpus_config.get_pdf_extraction_config()
        
        # Convert language list to primary language
        languages = pdf_config.get('ocr_languages', None)
        primary_language = languages[0] if languages else 'eng'
        
        return cls(
            enabled=pdf_config.get('ocr_enabled', False),
            dpi=300,  # Not in main config, keep default
            timeout_seconds=pdf_config.get('full_timeout_seconds', 0),
            language=primary_language,
            languages=languages,
            psm_mode=3,  # Keep default
            oem_mode=3,  # Keep default
            enable_preprocessing=True,  # Keep default
            confidence_threshold=pdf_config.get('ocr_confidence_threshold', None)
        )


@dataclass
class PDFConfig:
    """Configuration specific to PDF processing."""
    max_pages_preview: int = 5
    extract_images: bool = False
    extract_tables: bool = True
    prefer_text_layer: bool = True
    ocr_threshold_chars_per_page: int = 100
    try_multiple_extractors: bool = True
    default_mode: str = 'smart'
    preview_max_chars: int = 50000
    full_max_chars: int = 10000000
    full_timeout_seconds: int = 120
    enable_layout_analysis: bool = True
    preserve_formatting: bool = True
    extract_headers_footers: bool = False
    table_extraction_mode: str = 'advanced'
    min_table_rows: int = 2
    use_pdf_cache: bool = True
    cache_extracted_text: bool = True
    
    @classmethod
    def from_corpus_config(cls, corpus_config: CorpusConfig) -> 'PDFConfig':
        """Create PDFConfig from CorpusConfig"""
        pdf_config = corpus_config.get_pdf_extraction_config()
        features = corpus_config.get_feature_flags()
        
        return cls(
            max_pages_preview=pdf_config.get('preview_pages', None),
            extract_images=features.get('enable_figure_analysis', False),
            extract_tables=features.get('enable_table_extraction', True),
            prefer_text_layer=True,  # Keep default
            ocr_threshold_chars_per_page=100,  # Keep default
            try_multiple_extractors=True,  # Keep default
            default_mode=pdf_config.get('default_mode', None),
            preview_max_chars=pdf_config.get('preview_max_chars', 0),
            full_max_chars=pdf_config.get('full_max_chars', 0),
            full_timeout_seconds=pdf_config.get('full_timeout_seconds', 0),
            enable_layout_analysis=pdf_config.get('enable_layout_analysis', False),
            preserve_formatting=pdf_config.get('preserve_formatting', None),
            extract_headers_footers=pdf_config.get('extract_headers_footers', None),
            table_extraction_mode=pdf_config.get('table_extraction_mode', None),
            min_table_rows=pdf_config.get('min_table_rows', None),
            use_pdf_cache=pdf_config.get('use_pdf_cache', None),
            cache_extracted_text=pdf_config.get('cache_extracted_text', None)
        )


@dataclass
class ReaderConfig:
    """Main configuration class for the document reader system."""
    # Processing configuration
    batch_size: int = 20
    max_workers: int = field(default_factory=lambda: os.cpu_count() or 4)
    enable_parallel_processing: bool = True
    processing_mode: str = 'batch'
    timeout_seconds: int = 300
    
    # Cache configuration
    enable_cache: bool = True
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    
    # Validation configuration
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    
    # OCR configuration
    ocr_config: OCRConfig = field(default_factory=OCRConfig)
    
    # PDF configuration
    pdf_config: PDFConfig = field(default_factory=PDFConfig)
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = "document_reader.log"
    
    # Feature flags
    enable_content_extraction: bool = True
    enable_metadata_extraction: bool = True
    enable_ocr: bool = True
    enable_scientific_parsing: bool = True
    enable_table_extraction: bool = True
    enable_abbreviation_expansion: bool = True
    
    @classmethod
    def from_corpus_config(cls, corpus_config: Optional[CorpusConfig] = None) -> 'ReaderConfig':
        """
        Create ReaderConfig from CorpusConfig (YAML files).
        
        Args:
            corpus_config: Optional CorpusConfig instance. If not provided, creates one.
            
        Returns:
            ReaderConfig instance with settings from YAML files
        """
        if corpus_config is None:
            corpus_config = CorpusConfig()
        
        runtime_config = corpus_config.get_runtime_config()
        execution_config = corpus_config.get_execution_config()
        features = corpus_config.get_feature_flags()
        logging_config = corpus_config.get_logging_config()
        
        return cls(
            # Processing configuration from runtime
            batch_size=runtime_config.get('batch_size', None),
            max_workers=runtime_config.get('parallel_workers', None),
            enable_parallel_processing=runtime_config.get('parallel_workers', None) > 1,
            processing_mode=execution_config.get('processing_mode', None),
            timeout_seconds=runtime_config.get('timeout_seconds', 0),
            
            # Cache configuration
            enable_cache=features['enable_caching'],
            cache_config=CacheConfig.from_corpus_config(corpus_config),
            
            # Validation configuration
            validation_config=ValidationConfig.from_corpus_config(corpus_config),
            
            # OCR configuration
            ocr_config=OCRConfig.from_corpus_config(corpus_config),
            
            # PDF configuration
            pdf_config=PDFConfig.from_corpus_config(corpus_config),
            
            # Logging configuration
            log_level=logging_config.get('level', None),
            log_file=logging_config.get('file', None),
            
            # Feature flags
            enable_content_extraction=True,  # Always enabled
            enable_metadata_extraction=True,  # Always enabled
            enable_ocr=corpus_config.get_pdf_extraction_config()['ocr_enabled'],
            enable_scientific_parsing=features.get('enable_section_detection', True),
            enable_table_extraction=features['enable_table_extraction'],
            enable_abbreviation_expansion=features['enable_abbreviation_expansion']
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ReaderConfig':
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration JSON file
            
        Returns:
            ReaderConfig instance
        """
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ReaderConfig':
        """
        Create configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            ReaderConfig instance
        """
        # Handle nested configurations
        if 'cache_config' in config_dict and isinstance(config_dict['cache_config'], dict):
            config_dict['cache_config'] = CacheConfig(**config_dict['cache_config'])
        if 'validation_config' in config_dict and isinstance(config_dict['validation_config'], dict):
            config_dict['validation_config'] = ValidationConfig(**config_dict['validation_config'])
        if 'ocr_config' in config_dict and isinstance(config_dict['ocr_config'], dict):
            config_dict['ocr_config'] = OCRConfig(**config_dict['ocr_config'])
        if 'pdf_config' in config_dict and isinstance(config_dict['pdf_config'], dict):
            config_dict['pdf_config'] = PDFConfig(**config_dict['pdf_config'])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return asdict(self)
    
    def save(self, config_path: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            config_path: Path where to save the configuration
        """
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def load_default_config() -> ReaderConfig:
    """
    Load default configuration from YAML files via CorpusConfig.
    
    Returns:
        ReaderConfig instance with values from main_config.yaml and cache_settings.yaml
    """
    # First try to load from YAML configuration
    try:
        corpus_config = CorpusConfig()
        return ReaderConfig.from_corpus_config(corpus_config)
    except Exception as e:
        # If YAML loading fails, check for legacy JSON configs
        
        # Check for user config file
        user_config_path = Path.home() / '.document_reader' / 'config.json'
        if user_config_path.exists():
            return ReaderConfig.from_file(str(user_config_path))
        
        # Check for local config file
        local_config_path = Path('reader_config.json')
        if local_config_path.exists():
            return ReaderConfig.from_file(str(local_config_path))
        
        # Return default configuration
        return ReaderConfig()