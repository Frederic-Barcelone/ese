#!/usr/bin/env python3
"""
Basic Document Metadata Extractor - WITH CONSISTENT AI HANDLING
===========================================================================
Location: corpus_metadata/document_metadata_extraction_basic.py
Version: 2.8.0
Last Updated: 2025-01-17

CHANGES IN VERSION 2.8.0:
- Enhanced date extraction with file metadata fallback
- Uses DateParser.get_most_relevant_date for comprehensive date detection
- Stores file metadata for use in date extraction
- Always provides a date (content -> PDF metadata -> file dates)

CHANGES IN VERSION 2.7.0:
- Consistent AI validation checking for all components
- All extractors respect system initializer's AI validation decision
- No component attempts Claude API if ai_validation is false
- Clean, DRY approach - no duplicate validation

Description:
This script handles basic document processing including:
- Full system initialization (lexicons, models, resources)
- File metadata extraction (size, dates, permissions)
- PDF metadata extraction (author, creator, creation date)
- Text extraction from documents
- Document classification
- Entity extraction using loaded models
- Complete metadata output in JSON

Usage:
    from document_metadata_extraction_basic import RareDiseaseMetadataExtractor
    
    # Initialize extractor with system_initializer
    extractor = RareDiseaseMetadataExtractor(system_initializer=system_initializer)
    
    # Process a document - includes full metadata
    metadata = extractor.extract_from_file("document.pdf")
"""

import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Generator
from dataclasses import dataclass, asdict, field
from datetime import datetime

# Configure logging
logger = logging.getLogger('basic_extractor')
logger.setLevel(logging.WARNING)

# Import PDF processing
try:
    import pymupdf as fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False
        logger.debug("PyMuPDF not available for PDF processing")

# Import the MetadataExtractor for file/PDF metadata
try:
    from corpus_metadata.document_utils.reader_metadata import MetadataExtractor
    METADATA_EXTRACTOR_AVAILABLE = True
    logger.debug("MetadataExtractor loaded successfully")
except ImportError as e:
    logger.warning(f"MetadataExtractor not available: {e}")
    METADATA_EXTRACTOR_AVAILABLE = False
    MetadataExtractor = None

# Safe import helper
def safe_import(module_name, class_name=None):
    """Safely import optional dependencies"""
    try:
        if '.' in module_name:
            parts = module_name.split('.')
            module = __import__(module_name)
            for part in parts[1:]:
                module = getattr(module, part)
        else:
            module = __import__(module_name)
        
        if class_name:
            return getattr(module, class_name)
        return module
    except ImportError:
        return None
    except AttributeError:
        return None

# Import system initializer
MetadataSystemInitializer = safe_import(
    'corpus_metadata.document_utils.metadata_system_initializer', 
    'MetadataSystemInitializer'
)

# Import entity extractors
DrugMetadataExtractor = safe_import(
    'corpus_metadata.document_metadata_extraction_drug',
    'DrugMetadataExtractor'
)

DiseaseMetadataExtractor = safe_import(
    'corpus_metadata.document_metadata_extraction_disease',
    'DiseaseMetadataExtractor'
)

# Import document classifier
DocumentClassifier = safe_import(
    'corpus_metadata.document_utils.metadata_classifier',
    'DocumentClassifier'
)
CLASSIFIER_AVAILABLE = DocumentClassifier is not None

# Import description generator
DescriptionExtractor = safe_import(
    'corpus_metadata.document_utils.metadata_description',
    'DescriptionExtractor'
)
DESCRIPTION_AVAILABLE = DescriptionExtractor is not None

# ============================================================================
# Enhanced Resource Manager with Proper System Handling
# ============================================================================

class ResourceManager:
    """
    Singleton resource manager with comprehensive system initialization.
    Properly handles system_initializer parameter.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, system_initializer=None, config_path=None, verbose=False):
        """
        Initialize resource manager
        
        Args:
            system_initializer: Pre-initialized MetadataSystemInitializer instance
            config_path: Path to configuration file (if system_initializer not provided)
            verbose: Enable verbose output
        """
        if not self._initialized:
            self.verbose = verbose
            self.initializer = system_initializer
            self.config = {}
            self.models = {}
            self.lexicons = {}
            self.status = {
                'initialized': False,
                'lexicons_loaded': False,
                'models_loaded': False,
                'resources_loaded': False,
                'errors': []
            }
            
            # Initialize the system
            self._initialize_system(config_path)
            ResourceManager._initialized = True
    
    def _initialize_system(self, config_path=None):
        """
        Initialize the complete metadata extraction system.
        """
        # If system_initializer was provided, use it
        if self.initializer:
            logger.debug("Using provided system initializer")
            self._extract_system_info()
            return
        
        # Otherwise, try to create one
        if not MetadataSystemInitializer:
            self.status['errors'].append("MetadataSystemInitializer not available")
            logger.debug("MetadataSystemInitializer not available")
            return
        
        try:
            # Use provided config path or default
            if not config_path:
                config_path = "corpus_config/config.yaml"
            
            if self.verbose:
                print("\n" + "="*60)
                print("INITIALIZING METADATA EXTRACTION SYSTEM")
                print("="*60)
            
            # Try to get existing instance first
            try:
                self.initializer = MetadataSystemInitializer.get_instance(config_path)
                logger.debug("Using existing system initializer instance")
            except:
                # Create new instance
                self.initializer = MetadataSystemInitializer(config_path)
                logger.debug("Created new system initializer instance")
            
            # Check if initialization is needed
            if self.initializer and hasattr(self.initializer, 'initialize'):
                try:
                    start_time = time.time()
                    self.initializer.initialize()
                    elapsed = time.time() - start_time
                    if self.verbose:
                        print(f"\n[INFO]  System initialized in {elapsed:.2f}s")
                except Exception as e:
                    logger.debug(f"System already initialized or error: {e}")
            
            self._extract_system_info()
            
        except Exception as e:
            error_msg = f"Failed to initialize system: {str(e)}"
            self.status['errors'].append(error_msg)
            logger.warning(error_msg)
    
    def _extract_system_info(self):
        """Extract information from the initialized system"""
        if not self.initializer:
            return
        
        # Store configuration
        if hasattr(self.initializer, 'config'):
            self.config = self.initializer.config
        
        # Check what was loaded
        if hasattr(self.initializer, 'status'):
            status = self.initializer.status
            
            # Check lexicons
            if status.get('lexicons', {}).get('drugs'):
                self.lexicons['drug'] = True
                if self.verbose and hasattr(self.initializer, 'drug_lexicon'):
                    print(f"  [OK] Drug lexicon loaded: {len(self.initializer.drug_lexicon)} entries")
            
            if status.get('lexicons', {}).get('diseases'):
                self.lexicons['disease'] = True
                if self.verbose and hasattr(self.initializer, 'disease_lexicon'):
                    print(f"  [OK] Disease lexicon loaded: {len(self.initializer.disease_lexicon)} entries")
            
            if status.get('lexicons', {}).get('medical_terms'):
                self.lexicons['medical_terms'] = True
                if self.verbose and hasattr(self.initializer, 'medical_terms_lexicon'):
                    print(f"  [OK] Medical terms loaded: {len(self.initializer.medical_terms_lexicon)} entries")
            
            # Check models
            if status.get('models', {}):
                for model_name, model_info in status['models'].items():
                    if model_info:
                        self.models[model_name] = True
                        if self.verbose:
                            print(f"  [OK] Model loaded: {model_name}")
        
        self.status['initialized'] = True
        self.status['lexicons_loaded'] = bool(self.lexicons)
        self.status['models_loaded'] = bool(self.models)
        self.status['resources_loaded'] = True
        
        if self.verbose:
            print("="*60 + "\n")
    
    @property
    def resources(self):
        """Get loaded resources from initializer"""
        if self.initializer and hasattr(self.initializer, 'resources'):
            return self.initializer.resources
        return {}
    
    @property
    def drug_lexicon(self):
        """Get drug lexicon if loaded"""
        if self.initializer and hasattr(self.initializer, 'drug_lexicon'):
            return self.initializer.drug_lexicon
        return {}
    
    @property
    def disease_lexicon(self):
        """Get disease lexicon if loaded"""
        if self.initializer and hasattr(self.initializer, 'disease_lexicon'):
            return self.initializer.disease_lexicon
        return {}
    
    @property
    def spacy_models(self):
        """Get loaded SpaCy models"""
        if self.initializer and hasattr(self.initializer, 'models'):
            return self.initializer.models
        return {}
    
    def is_ready(self):
        """Check if system is ready for extraction"""
        return self.status['initialized'] and (
            self.status['lexicons_loaded'] or self.status['models_loaded']
        )

# ============================================================================
# Data Classes - Enhanced with file metadata fields
# ============================================================================

@dataclass
class BasicMetadata:
    """Container for basic document metadata with file/PDF metadata fields"""
    document_id: str
    filename: str
    file_path: str
    file_size: int
    pages: int
    text_length: int
    extraction_timestamp: str
    mode: str  # 'intro' or 'full'
    
    # File metadata fields
    file_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    created_time: Optional[str] = None
    modified_time: Optional[str] = None
    accessed_time: Optional[str] = None
    mime_type: Optional[str] = None
    
    # PDF metadata fields  
    pdf_author: Optional[str] = None
    pdf_creator: Optional[str] = None
    pdf_producer: Optional[str] = None
    pdf_title: Optional[str] = None
    pdf_subject: Optional[str] = None
    pdf_keywords: Optional[str] = None
    pdf_creation_date: Optional[str] = None
    pdf_modified_date: Optional[str] = None
    pdf_is_encrypted: Optional[bool] = None
    pdf_page_count: Optional[int] = None
    pdf_version: Optional[str] = None
    
    # Classification
    document_type: Optional[str] = None
    document_subtype: Optional[str] = None
    classification_confidence: Optional[float] = None
    
    # Descriptions
    title: Optional[str] = None
    short_description: Optional[str] = None
    long_description: Optional[str] = None
    
    # Dates
    document_date: Optional[str] = None
    extracted_dates: Optional[List[str]] = None
    
    # Processing info
    processing_time: float = 0.0
    success: bool = True
    error: Optional[str] = None
    partial_results: Optional[Dict[str, Any]] = None
    
    # Entity extraction results
    drugs: Optional[List[Dict[str, Any]]] = None
    diseases: Optional[List[Dict[str, Any]]] = None
    full_text: Optional[str] = None

# ============================================================================
# Main Extractor Class with Consistent AI Handling
# ============================================================================

class RareDiseaseMetadataExtractor:
    """
    Handles document processing with full system initialization and metadata extraction.
    All components respect system initializer's AI validation decision.
    """
    
    def __init__(self, config_path=None, system_initializer=None, verbose=False, **kwargs):
        """
        Initialize the extractor with full system loading
        
        Args:
            config_path: Optional path to configuration file
            system_initializer: Pre-initialized MetadataSystemInitializer instance
            verbose: Enable verbose logging and output
            **kwargs: Additional arguments for compatibility
        """
        self.verbose = verbose
        self.config_path = config_path
        self.system_initializer = system_initializer
        
        # Set logging level based on verbose flag
        if verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
        
        # Initialize the resource manager with system_initializer
        if verbose:
            print("\n" + "="*60)
            print("INITIALIZING EXTRACTION SYSTEM")
            print("="*60)
        
        # Pass system_initializer to resource manager
        self.resource_manager = ResourceManager(
            system_initializer=system_initializer,
            config_path=config_path,
            verbose=verbose
        )
        
        self.config = self.resource_manager.config
        self.initializer = self.resource_manager.initializer or system_initializer
        
        # Check if AI is enabled according to system initializer
        self.ai_enabled = self.config.get('features', {}).get('ai_validation', False)
        if self.ai_enabled and self.initializer:
            # Double check with system initializer status
            if hasattr(self.initializer, 'status'):
                # If there are errors related to API, disable AI
                errors = self.initializer.status.get('errors', [])
                if any('CLAUDE_API_KEY' in str(e) for e in errors):
                    self.ai_enabled = False
                    logger.debug("AI disabled due to API key issues")
        
        # Initialize MetadataExtractor for file/PDF metadata
        self.metadata_extractor = None
        if METADATA_EXTRACTOR_AVAILABLE:
            try:
                self.metadata_extractor = MetadataExtractor(self.config)
                if verbose:
                    logger.info("MetadataExtractor initialized for file/PDF metadata")
            except Exception as e:
                logger.warning(f"Could not initialize MetadataExtractor: {e}")
        
        # Report system status
        if verbose and self.resource_manager.is_ready():
            print("\n[OK] System initialized successfully")
            if self.resource_manager.lexicons:
                print(f"  - Lexicons loaded: {list(self.resource_manager.lexicons.keys())}")
            if self.resource_manager.models:
                print(f"  - Models loaded: {list(self.resource_manager.models.keys())}")
            if self.metadata_extractor:
                print("  - File/PDF metadata extractor loaded")
            print(f"  - AI features: {'enabled' if self.ai_enabled else 'disabled'}")
        
        # Check system status
        if not self.resource_manager.is_ready():
            if not system_initializer:
                logger.warning("System not fully initialized - some features may be unavailable")
            if self.resource_manager.status['errors']:
                for error in self.resource_manager.status['errors']:
                    logger.warning(f"  - {error}")
        
        # Initialize entity extractors
        self.drug_extractor = None
        self.disease_extractor = None
        
        if self.resource_manager.is_ready() or system_initializer:
            self._initialize_entity_extractors()
        
        # Initialize classifier if available
        self.classifier = None
        if CLASSIFIER_AVAILABLE:
            try:
                # Try different initialization patterns
                if system_initializer:
                    try:
                        self.classifier = DocumentClassifier(system_initializer=system_initializer)
                    except:
                        self.classifier = DocumentClassifier()
                else:
                    self.classifier = DocumentClassifier()
                    
                if verbose:
                    logger.info("Document classifier initialized")
            except Exception as e:
                logger.debug(f"Classifier initialization skipped: {e}")
        
        # Initialize description generator based on AI availability
        self.description_generator = None
        if DESCRIPTION_AVAILABLE:
            if self.ai_enabled:
                # AI is enabled by system, try to initialize with API
                try:
                    # Support both config structures: api_configuration.claude (legacy) and api.claude (current)
                    api_config = self.config.get('api_configuration', self.config.get('api', {}))
                    model = api_config.get('claude', {}).get('model', 'claude-sonnet-4-5-20250929')
                    self.description_generator = DescriptionExtractor(model=model, silent=True)
                    
                    if verbose:
                        if self.description_generator.client:
                            logger.info("Description generator initialized with Claude API")
                        else:
                            logger.info("Description generator initialized (fallback mode)")
                except Exception as e:
                    logger.debug(f"Description generator initialization: {e}")
            else:
                # AI not enabled, initialize without API for fallback functionality
                try:
                    self.description_generator = DescriptionExtractor(api_key=None, model="none", silent=True)
                    if verbose:
                        logger.info("Description generator initialized (fallback only)")
                except Exception as e:
                    logger.debug(f"Description generator not available: {e}")
        
        # Initialize storage for current file metadata (for date extraction)
        self._current_filename = None
        self._current_file_metadata = None
        
        # Cache for compiled regex patterns
        self._pattern_cache = {}
        
        if verbose:
            print("="*60 + "\n")
    
    def _initialize_entity_extractors(self):
        """Initialize drug and disease extractors using the loaded system"""
        system_init = self.initializer or self.system_initializer
        
        # Get ALL settings from config
        features = self.config.get('features', {})
        modes = self.config.get('modes', {})
        active_mode = modes.get('active', 'intro')
        mode_config = modes.get(active_mode, {})
        
        # Get thresholds and other settings from config
        thresholds = self.config.get('thresholds', {})
        drug_data = self.config.get('drug_data', {})
        
        # Initialize drug extractor
        if DrugMetadataExtractor and features.get('drug_detection', True):
            try:
                kwargs = {
                    'verbose': self.verbose,
                    'use_claude': self.ai_enabled,  # Use system's AI decision
                    'use_patterns': self.config.get('rules', {}).get('use_suffix_patterns', True),
                    'use_ner': features.get('section_detection', True),
                    'use_lexicon': bool(self.config.get('lexicons', {}).get('drug')),
                    'use_knowledge_base': bool(drug_data),
                    'config_path': self.config_path
                }
                
                # Add system_initializer if available and it accepts it
                if system_init:
                    try:
                        self.drug_extractor = DrugMetadataExtractor(
                            system_initializer=system_init,
                            **kwargs
                        )
                    except TypeError:
                        # If system_initializer not accepted, try without it
                        self.drug_extractor = DrugMetadataExtractor(**kwargs)
                else:
                    self.drug_extractor = DrugMetadataExtractor(**kwargs)
                    
                if self.verbose:
                    logger.info(f"Drug extractor initialized (Claude: {kwargs['use_claude']}, Patterns: {kwargs['use_patterns']})")
            except Exception as e:
                logger.warning(f"Could not initialize drug extractor: {e}")
        
        # Initialize disease extractor
        if DiseaseMetadataExtractor and features.get('disease_detection', True):
            try:
                # Determine mode from config
                detection_mode = 'balanced'  # default
                if thresholds.get('confidence', 0.75) > 0.85:
                    detection_mode = 'precision'
                elif thresholds.get('confidence', 0.75) < 0.6:
                    detection_mode = 'recall'
                
                kwargs = {
                    'mode': detection_mode,
                    'verbose': self.verbose,
                    'use_claude': self.ai_enabled,  # Use system's AI decision
                    'config_path': self.config_path
                }
                
                # DiseaseMetadataExtractor doesn't accept system_initializer
                self.disease_extractor = DiseaseMetadataExtractor(**kwargs)
                    
                if self.verbose:
                    logger.info(f"Disease extractor initialized (Mode: {detection_mode}, Claude: {kwargs['use_claude']})")
            except Exception as e:
                logger.warning(f"Could not initialize disease extractor: {e}")
    
    def extract_file_metadata(self, file_path):
        """
        Extract file and PDF metadata using MetadataExtractor
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of file metadata
        """
        if not self.metadata_extractor:
            return {}
        
        try:
            # Extract metadata using the fixed MetadataExtractor
            metadata = self.metadata_extractor.extract_metadata(file_path)
            
            if self.verbose and metadata:
                logger.info(f"Extracted {len(metadata)} metadata fields from file")
                if 'pdf_author' in metadata:
                    logger.info(f"  PDF Author: {metadata.get('pdf_author')}")
                if 'pdf_creation_date' in metadata:
                    logger.info(f"  PDF Created: {metadata.get('pdf_creation_date')}")
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract file metadata: {e}")
            return {}
    
    def extract_text(self, file_path, mode='intro'):
        """
        Extract text from document
        
        Args:
            file_path: Path to document
            mode: 'intro' (first 10 pages) or 'full' (all pages)
            
        Returns:
            Tuple of (extracted_text, pages_processed)
        """
        file_path = Path(file_path)
        
        if not PYMUPDF_AVAILABLE:
            logger.error("PyMuPDF not available - cannot extract text from PDF")
            return "", 0
        
        text_parts = []
        pages_processed = 0
        
        try:
            with fitz.open(str(file_path)) as pdf:
                total_pages = len(pdf)
                
                # Determine pages to process
                if mode == 'intro':
                    pages_to_process = min(10, total_pages)
                else:
                    pages_to_process = total_pages
                
                # Extract text from pages
                for page_num in range(pages_to_process):
                    page = pdf[page_num]
                    text = page.get_text()
                    if text.strip():
                        text_parts.append(text)
                    pages_processed += 1
                
                if self.verbose:
                    logger.info(f"Extracted text from {pages_processed}/{total_pages} pages")
        
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return "", 0
        
        return '\n'.join(text_parts), pages_processed
    
    def extract_entities(self, text):
        """
        Extract drugs and diseases using loaded models and lexicons
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with 'drugs' and 'diseases' lists
        """
        entities = {'drugs': [], 'diseases': []}
        
        # Extract drugs if extractor available
        if self.drug_extractor:
            try:
                drug_results = self.drug_extractor.extract_drugs_from_text(text)
                if drug_results and isinstance(drug_results, dict):
                    entities['drugs'] = drug_results.get('drugs', [])
                    if self.verbose and entities['drugs']:
                        logger.info(f"Extracted {len(entities['drugs'])} drugs")
            except Exception as e:
                logger.warning(f"Error extracting drugs: {e}")
        
        # Extract diseases if extractor available
        if self.disease_extractor:
            try:
                # Check for correct method name
                if hasattr(self.disease_extractor, 'extract_diseases_from_text'):
                    disease_results = self.disease_extractor.extract_diseases_from_text(text)
                elif hasattr(self.disease_extractor, 'extract'):
                    disease_results = self.disease_extractor.extract(text)
                else:
                    disease_results = None
                
                if disease_results and isinstance(disease_results, dict):
                    entities['diseases'] = disease_results.get('diseases', [])
                    if self.verbose and entities['diseases']:
                        logger.info(f"Extracted {len(entities['diseases'])} diseases")
            except Exception as e:
                logger.warning(f"Error extracting diseases: {e}")
        
        # Fallback to lexicon-based extraction if no extractors
        if not self.drug_extractor and not self.disease_extractor:
            if self.verbose:
                logger.info("Using fallback lexicon-based extraction")
            entities.update(self._lexicon_based_extraction(text))
        
        return entities
    
    def _lexicon_based_extraction(self, text):
        """
        Fallback extraction using lexicons directly
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with extracted entities
        """
        entities = {'drugs': [], 'diseases': []}
        
        if not self.initializer:
            return entities
        
        text_lower = text.lower()
        
        # Extract drugs from lexicon
        if hasattr(self.initializer, 'drug_lexicon'):
            found_drugs = set()
            for term in self.initializer.drug_lexicon:
                if term.lower() in text_lower and len(term) > 3:
                    found_drugs.add(term)
            
            entities['drugs'] = [{'name': drug, 'source': 'lexicon'} for drug in found_drugs]
        
        # Extract diseases from lexicon  
        if hasattr(self.initializer, 'disease_lexicon'):
            found_diseases = set()
            for term in self.initializer.disease_lexicon:
                if term.lower() in text_lower and len(term) > 3:
                    found_diseases.add(term)
            
            entities['diseases'] = [{'name': disease, 'source': 'lexicon'} for disease in found_diseases]
        
        return entities
    
    def get_extracted_text(self, file_path, mode='intro'):
        """
        Compatibility method for other extractors that need text
        
        Args:
            file_path: Path to document
            mode: 'intro' or 'full'
            
        Returns:
            Extracted text string
        """
        text, _ = self.extract_text(file_path, mode)
        return text
    
    def classify_document(self, text, filename):
        """
        Classify document type
        
        Args:
            text: Document text
            filename: Document filename
            
        Returns:
            Classification results dictionary
        """
        if not self.classifier:
            return {
                'document_type': 'unknown',
                'confidence': 0.0,
                'method': 'no_classifier'
            }
        
        try:
            # Check for available methods
            if hasattr(self.classifier, 'classify'):
                classification = self.classifier.classify(text, filename)
            elif hasattr(self.classifier, 'classify_document'):
                classification = self.classifier.classify_document({'content': text, 'name': filename})
            else:
                return {'document_type': 'unknown', 'confidence': 0.0}
            
            return {
                'document_type': classification.get('type', classification.get('doc_type', 'unknown')),
                'document_subtype': classification.get('subtype'),
                'confidence': classification.get('confidence', 0.0),
                'method': classification.get('method', 'unknown')
            }
        except Exception as e:
            logger.warning(f"Classification error: {e}")
            return {
                'document_type': 'unknown',
                'confidence': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    def extract_title(self, text):
        """
        Extract document title from text content.
        
        Attempts to find the title using:
        1. First substantial line of text
        2. Lines with title-like formatting
        3. Common title patterns
        
        Args:
            text: Document text content
            
        Returns:
            Dictionary with 'title' key or None if not found
        """
        if not text or len(text.strip()) < 10:
            return None
        
        try:
            # Split into lines and clean
            lines = text.split('\n')
            candidate_lines = []
            
            for line in lines[:30]:  # Check first 30 lines
                line = line.strip()
                
                # Skip empty lines and very short/long lines
                if not line or len(line) < 5 or len(line) > 200:
                    continue
                
                # Skip lines that look like metadata, headers, or page numbers
                skip_patterns = [
                    r'^page\\s+\\d+',
                    r'^\\d+$',
                    r'^(C)|copyright|all rights reserved',
                    r'^vol\\.\\s+\\d+|volume\\s+\\d+',
                    r'^issn|isbn|doi:',
                    r'^www\\.|http',
                    r'^article\\s+info|abstract:',
                    r'^received:|accepted:|published:',
                    # FIX: Better Keywords/metadata detection
                    r'^keywords?\\s*[:;,]?\\s*[A-Z]',
                    r'^keywords?\\s+[A-Z]{2,}',
                    r'^\\*?correspondence',
                    r'^authors?\\s*[:;]',
                ]
                
                should_skip = False
                for pattern in skip_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        should_skip = True
                        break
                
                if should_skip:
                    continue
                
                candidate_lines.append(line)
                
                # Good title candidate if we have at least one
                if len(candidate_lines) >= 1:
                    break
            
            if candidate_lines:
                # Return the first good candidate
                title = candidate_lines[0]
                
                # Clean up common artifacts
                title = re.sub(r'\s+', ' ', title)
                title = title.strip()
                
                if len(title) >= 10:
                    return {'title': title}
            
            return None
            
        except Exception as e:
            logger.warning(f"Title extraction error: {e}")
            return None
    
    def generate_descriptions(self, text, filename, classification):
        """Generate document descriptions"""
        if not self.description_generator:
            return {
                'title': Path(filename).stem.replace('_', ' ').title(),
                'short_description': f"Document: {filename}",
                'long_description': f"Document {filename} with {len(text)} characters of text"
            }
        
        try:
            descriptions = self.description_generator.generate_descriptions(
                text, filename, classification
            )
            return descriptions
        except Exception as e:
            logger.warning(f"Description generation error: {e}")
            return {
                'title': Path(filename).stem.replace('_', ' ').title(),
                'short_description': f"Document: {filename}",
                'long_description': f"Document with {len(text)} characters"
            }
    
    def extract_dates(self, text):
        """
        Extract dates from document with metadata fallback.
        Uses the enhanced DateParser with academic patterns and file metadata.
        
        Args:
            text: Document text content
            
        Returns:
            Dictionary with document_date and extracted_dates
        """
        try:
            from corpus_metadata.document_utils.metadata_date_utils import DateParser
            
            # Initialize date parser
            date_parser = DateParser()
            
            # Get stored file metadata and filename
            file_metadata = getattr(self, '_current_file_metadata', None)
            filename = getattr(self, '_current_filename', 'document.pdf')
            
            # Extract dates from content using enhanced patterns
            extracted_dates = date_parser.extract_dates_from_content(text, max_dates=10)
            
            # Get most relevant date using all sources with fallback chain
            document_date = date_parser.get_most_relevant_date(
                filename=filename,
                content=text,
                file_metadata=file_metadata
            )
            
            # Log the date source for debugging
            if self.verbose and document_date:
                if document_date in extracted_dates:
                    logger.info(f"Document date from content: {document_date}")
                elif file_metadata:
                    if document_date == (file_metadata.get('pdf_creation_date', '')[:10]):
                        logger.info(f"Document date from PDF metadata: {document_date}")
                    elif document_date == (file_metadata.get('modified_time', '')[:10]):
                        logger.info(f"Document date from file modified time: {document_date}")
            
            return {
                'document_date': document_date,
                'extracted_dates': extracted_dates
            }
            
        except ImportError:
            # Fallback if DateParser not available - use basic extraction
            logger.warning("DateParser not available, using basic date extraction")
            
            dates = []
            document_date = None
            
            # Basic patterns
            patterns = [
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
                r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'
            ]
            
            for pattern_str in patterns:
                if pattern_str not in self._pattern_cache:
                    self._pattern_cache[pattern_str] = re.compile(pattern_str, re.IGNORECASE)
                pattern = self._pattern_cache[pattern_str]
                
                matches = pattern.findall(text[:10000])
                dates.extend(matches)
            
            # Deduplicate dates
            dates = list(set(dates))[:10]
            
            # Try to get document date
            if dates:
                document_date = dates[0]
            elif self._current_file_metadata:
                # Use file metadata as fallback
                for date_field in ['pdf_creation_date', 'pdf_modified_date', 'modified_time', 'created_time']:
                    if date_field in self._current_file_metadata:
                        date_str = str(self._current_file_metadata[date_field])
                        if 'T' in date_str:
                            document_date = date_str.split('T')[0]
                        else:
                            document_date = date_str[:10] if len(date_str) >= 10 else None
                        if document_date:
                            break
            
            return {
                'document_date': document_date,
                'extracted_dates': dates
            }
    
    def extract_from_file(self, file_path, mode='intro', output_dir=None, stage='all'):
        """
        Extract metadata from file with optional stage control
        
        Args:
            file_path: Path to document
            mode: 'intro' (first 10 pages) or 'full' (all pages)
            output_dir: Optional directory to move processed files
            stage: 'metadata', 'entities', or 'all' (default)
                - 'metadata': Only extract document metadata, classification, dates
                - 'entities': Only extract drugs and diseases
                - 'all': Extract everything
        
        Returns:
            Dictionary with extraction results
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        # Store filename for date extraction
        self._current_filename = file_path.name
        
        # Extract file metadata FIRST (always needed)
        file_metadata = self.extract_file_metadata(file_path)
        
        # Store metadata for date extraction
        self._current_file_metadata = file_metadata
        
        # Initialize metadata object with file metadata
        metadata = BasicMetadata(
            document_id=file_path.stem,
            filename=file_path.name,
            file_path=str(file_path),
            file_size=file_metadata.get('size_bytes', file_path.stat().st_size if file_path.exists() else 0),
            pages=0,
            text_length=0,
            extraction_timestamp=datetime.now().isoformat(),
            mode=mode,
            # Add file metadata
            file_metadata=file_metadata,
            created_time=file_metadata.get('created_time'),
            modified_time=file_metadata.get('modified_time'),
            accessed_time=file_metadata.get('accessed_time'),
            mime_type=file_metadata.get('mime_type'),
            # Add PDF metadata
            pdf_author=file_metadata.get('pdf_author'),
            pdf_creator=file_metadata.get('pdf_creator'),
            pdf_producer=file_metadata.get('pdf_producer'),
            pdf_title=file_metadata.get('pdf_title'),
            pdf_subject=file_metadata.get('pdf_subject'),
            pdf_keywords=file_metadata.get('pdf_keywords'),
            pdf_creation_date=file_metadata.get('pdf_creation_date'),
            pdf_modified_date=file_metadata.get('pdf_modified_date'),
            pdf_is_encrypted=file_metadata.get('pdf_is_encrypted'),
            pdf_page_count=file_metadata.get('pdf_page_count'),
            pdf_version=file_metadata.get('pdf_version')
        )
        
        try:
            # Extract text
            text, pages = self.extract_text(file_path, mode)
            metadata.text_length = len(text)
            metadata.pages = pages
            metadata.full_text = text
            
            if not text:
                raise ValueError("No text extracted from document")
            
            # STAGE-SPECIFIC PROCESSING
            
            # Stage 1: Metadata extraction (classification, descriptions, dates)
            if stage in ['metadata', 'all']:
                # Classify document
                classification = self.classify_document(text, file_path.name)
                metadata.document_type = classification.get('document_type')
                metadata.document_subtype = classification.get('document_subtype')
                metadata.classification_confidence = classification.get('confidence')
                
                # Generate descriptions
                descriptions = self.generate_descriptions(text, file_path.name, classification)
                metadata.title = descriptions.get('title')
                metadata.short_description = descriptions.get('short_description')
                metadata.long_description = descriptions.get('long_description')
                
                # Extract dates with enhanced method and metadata fallback
                date_info = self.extract_dates(text)
                metadata.document_date = date_info.get('document_date')
                metadata.extracted_dates = date_info.get('extracted_dates')
                
                # If still no document date, use PDF creation as last resort
                if not metadata.document_date and file_metadata.get('pdf_creation_date'):
                    pdf_date = str(file_metadata['pdf_creation_date'])
                    if 'T' in pdf_date:
                        metadata.document_date = pdf_date.split('T')[0]
                    else:
                        metadata.document_date = pdf_date[:10] if len(pdf_date) >= 10 else None
                        
                    if self.verbose and metadata.document_date:
                        logger.info(f"Using PDF creation date as document date: {metadata.document_date}")
            
            # Stage 2: Entity extraction (drugs and diseases)
            if stage in ['entities', 'all']:
                # Extract entities
                entities = self.extract_entities(text)
                metadata.drugs = entities.get('drugs', [])
                metadata.diseases = entities.get('diseases', [])
            else:
                # If not extracting entities, set empty lists
                metadata.drugs = []
                metadata.diseases = []
            
            metadata.success = True
            metadata.processing_time = time.time() - start_time
            
            if self.verbose:
                logger.info(f"Extraction complete in {metadata.processing_time:.2f}s (stage: {stage})")
                if stage in ['metadata', 'all']:
                    logger.info(f"  - File metadata fields: {len(file_metadata)}")
                    logger.info(f"  - Document date: {metadata.document_date}")
                    logger.info(f"  - Extracted dates: {len(metadata.extracted_dates or [])}")
                if stage in ['entities', 'all']:
                    logger.info(f"  - Drugs found: {len(metadata.drugs)}")
                    logger.info(f"  - Diseases found: {len(metadata.diseases)}")
            
            # Move file if output directory specified
            if output_dir and metadata.success and stage == 'all':
                self._move_processed_file(file_path, output_dir, metadata.document_type)
            
        except Exception as e:
            metadata.success = False
            metadata.error = str(e)
            metadata.drugs = []
            metadata.diseases = []
            metadata.processing_time = time.time() - start_time
            logger.error(f"Extraction failed: {e}")
        
        finally:
            # Clear stored metadata after extraction
            self._current_filename = None
            self._current_file_metadata = None
        
        # Return as dictionary for compatibility
        result = asdict(metadata)
        
        # Clean up the output - remove None values and empty fields
        result = {k: v for k, v in result.items() if v is not None}
        
        # Add stage information
        result['extraction_stage'] = stage
        
        return result

    def extract_metadata_only(self, file_path, mode='intro'):
        """
        Extract only metadata (no entities)
        
        Args:
            file_path: Path to document
            mode: 'intro' or 'full'
        
        Returns:
            Dictionary with metadata results
        """
        return self.extract_from_file(file_path, mode=mode, stage='metadata')

    def extract_entities_only(self, file_path, mode='full'):
        """
        Extract only entities (drugs and diseases)
        
        Args:
            file_path: Path to document
            mode: 'intro' or 'full'
        
        Returns:
            Dictionary with entity results
        """
        return self.extract_from_file(file_path, mode=mode, stage='entities')
    
    def process_document(self, file_path, mode='intro'):
        """Alias for extract_from_file for flexibility"""
        return self.extract_from_file(file_path, mode)
    
    def _move_processed_file(self, file_path, output_dir, doc_type):
        """Move processed file to appropriate directory"""
        try:
            output_path = Path(output_dir)
            
            # Create subdirectory based on document type
            if doc_type and doc_type != 'unknown':
                output_path = output_path / doc_type.lower().replace(' ', '_')
            else:
                output_path = output_path / 'unknown'
            
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Move file
            destination = output_path / file_path.name
            
            # Handle if destination exists
            if destination.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                stem = file_path.stem
                suffix = file_path.suffix
                destination = output_path / f"{stem}_{timestamp}{suffix}"
            
            shutil.move(str(file_path), str(destination))
            
            if self.verbose:
                logger.info(f"Moved file to: {destination}")
            
        except Exception as e:
            logger.error(f"Error moving file: {e}")
    
    def get_system_status(self):
        """Get the status of loaded systems"""
        status = {
            'system_initialized': self.resource_manager.is_ready(),
            'ai_enabled': self.ai_enabled,
            'lexicons': self.resource_manager.lexicons,
            'models': self.resource_manager.models,
            'extractors': {
                'metadata_extractor': self.metadata_extractor is not None,
                'drug_extractor': self.drug_extractor is not None,
                'disease_extractor': self.disease_extractor is not None,
                'classifier': self.classifier is not None,
                'description_generator': self.description_generator is not None
            },
            'errors': self.resource_manager.status.get('errors', [])
        }
        
        # Add counts if available
        if self.resource_manager.drug_lexicon:
            status['drug_lexicon_size'] = len(self.resource_manager.drug_lexicon)
        
        if self.resource_manager.disease_lexicon:
            status['disease_lexicon_size'] = len(self.resource_manager.disease_lexicon)
        
        # Add model names if available
        if self.initializer and hasattr(self.initializer, 'models'):
            status['loaded_models'] = list(self.initializer.models.keys())
        
        return status