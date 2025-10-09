#!/usr/bin/env python3
"""
Document Utils Package
======================
Location: corpus_metadata/document_utils/__init__.py

Utilities for document processing and metadata extraction in the rare disease corpus.

This package provides:
- Document classification
- Type-specific routing
- Metadata extraction
- Caching utilities
- Logging configuration
"""

__version__ = "2.0.0"
__author__ = "Document Processing Team"

import logging

# Set up module logger
logger = logging.getLogger(__name__)

# ============================================================================
# CORE IMPORTS - Always Available
# ============================================================================

# Import logging utilities (always needed)
try:
    from .metadata_logging_config import CorpusConfig
    LOGGING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Logging configuration not available: {e}")
    LOGGING_AVAILABLE = False

# Import cache utilities
try:
    from .reader_extraction_cache import VersionedExtractionCache
    # Alias for backward compatibility
    ExtractionCache = VersionedExtractionCache
    CACHE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Extraction cache not available: {e}")
    CACHE_AVAILABLE = False

# ============================================================================
# OPTIONAL IMPORTS - May Not Be Available
# ============================================================================

# Document classifier (requires anthropic)
try:
    from .metadata_classifier import DocumentClassifier
    CLASSIFIER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Document classifier not available: {e}")
    CLASSIFIER_AVAILABLE = False

# Document router
try:
    from .metadata_document_type_router import DocumentTypeRouter
    ROUTER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Document router not available: {e}")
    ROUTER_AVAILABLE = False

# System initializer
try:
    from .metadata_system_initializer import MetadataSystemInitializer
    SYSTEM_INIT_AVAILABLE = True
except ImportError as e:
    logger.debug(f"System initializer not available: {e}")
    SYSTEM_INIT_AVAILABLE = False

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_extraction_cache():
    """Create a new extraction cache instance"""
    if CACHE_AVAILABLE:
        return VersionedExtractionCache()
    else:
        raise ImportError("Extraction cache is not available")

def get_system_initializer():
    """Get the singleton system initializer instance"""
    if SYSTEM_INIT_AVAILABLE:
        return MetadataSystemInitializer.get_instance()
    else:
        raise ImportError("System initializer is not available")

def setup_logging(**kwargs):
    """Setup centralized logging"""
    if LOGGING_AVAILABLE:
        config = CorpusConfig()
        config.setup(**kwargs)
        return config
    else:
        # Fallback to basic logging
        logging.basicConfig(
            level=kwargs.get('file_level', 'WARNING'),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return None

# ============================================================================
# BUILD EXPORTS LIST
# ============================================================================

__all__ = ['__version__', '__author__']

# Add available components to exports
if CACHE_AVAILABLE:
    __all__.extend([
        'ExtractionCache',
        'VersionedExtractionCache',
        'create_extraction_cache'
    ])

if CLASSIFIER_AVAILABLE:
    __all__.append('DocumentClassifier')

if ROUTER_AVAILABLE:
    __all__.append('DocumentTypeRouter')

if SYSTEM_INIT_AVAILABLE:
    __all__.extend([
        'MetadataSystemInitializer',
        'get_system_initializer'
    ])

if LOGGING_AVAILABLE:
    __all__.extend([
        'LoggingConfig',
        'setup_logging'
    ])

# ============================================================================
# PACKAGE INITIALIZATION MESSAGE (only in debug mode)
# ============================================================================

logger.debug(f"Document utils package v{__version__} initialized")
logger.debug(f"Available components: {', '.join(__all__)}")