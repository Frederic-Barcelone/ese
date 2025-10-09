"""
====================
Document Type Router
/Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/metadata_document_type_router.py
====================

Routes documents to appropriate type-specific extractors based on their classification.
Simple, robust implementation without over-engineering.

PURPOSE:
--------
Different document types (protocols, case report forms, publications) require different
extraction logic. This router directs each document to its specialized extractor.

WHAT IT DOES:
-------------
1. Takes a document type (e.g., 'PRO', 'CRF', 'PUB')
2. Finds the appropriate extractor module
3. Runs the extraction
4. Returns the results

ROUTING MAP:
------------
Clinical Documents:
- PRO/AME → Protocol/Amendment extractornow,
- CRF → Case Report Form extractor
- ICF → Informed Consent Form extractor
- SAP → Statistical Analysis Plan extractor
- CSR → Clinical Study Report extractor
- IBR/IB → Investigator Brochure extractor

Publications:
- PUB → Publication extractor

Regulatory:
- REG/IND → Regulatory extractor

Safety:
- SAE/SUSAR → Safety extractor

Default:
- UNKNOWN/DEFAULT → Generic extractor

DESIGN PRINCIPLES:
------------------
- Simple mapping of document types to extractors
- Graceful fallback to generic extractor
- No complex configuration or profiles
- Clear error handling
- Easy to extend with new document types

HOW TO ADD NEW DOCUMENT TYPE:
------------------------------
1. Add entry to ROUTING_MAP dictionary below
2. Create corresponding extractor module in same directory
3. Module name pattern: rare_disease_metadata_[type].py
4. Extractor class must have extract_metadata() or extract() method

LOGGING:
--------
Uses centralized logging configuration from corpus_metadata.utils.logging_config
Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
Log format: timestamp - logger_name - level - message

AUTHOR: Document Processing Team
VERSION: 2.0
LAST UPDATED: January 2025
"""

import logging
import importlib
from typing import Dict, Any, Optional
from pathlib import Path

# ============================================================================
# CENTRALIZED LOGGING CONFIGURATION
# ============================================================================
try:
    # Try to import centralized logging config from the correct location
    from corpus_metadata.document_utils.metadata_logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    try:
        # Try relative import
        from .metadata_logging_config import get_logger
        logger = get_logger(__name__)
    except ImportError:
        # Fallback to basic logging if centralized config not available
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger(__name__)
        # Don't warn about this - it's expected during early initialization

# ============================================================================
# DOCUMENT TYPE ROUTER CLASS
# ============================================================================

class DocumentTypeRouter:
    """
    Routes documents to appropriate type-specific extractors
    
    Attributes:
        ROUTING_MAP: Dictionary mapping document codes to extractor modules
        model_loader: Optional model loader for NLP-based extractors
        loaded_extractors: Cache of loaded extractor instances
        generic_extractor: Fallback extractor for unknown types
    """
    
    # ========================================================================
    # ROUTING MAP - ADD NEW DOCUMENT TYPES HERE
    # ========================================================================
    # Format: 'DOCUMENT_CODE': 'extractor_module_name'
    # Module must exist at: corpus_metadata/[module_name].py
    
    ROUTING_MAP = {
        # ====================================================================
        # CLINICAL TRIAL DOCUMENTS
        # ====================================================================
        # Protocol and Amendments
        'PRO': 'document_metadata_extraction_basic',  # Clinical Trial Protocol
        'AME': 'document_metadata_extraction_basic',  # Protocol Amendment
        'PCS': 'document_metadata_extraction_basic',  # Protocol Concept Sheet
        
        # Forms and Reports
        'CRF': 'document_metadata_extraction_basic',      # Case Report Form
        'ICF': 'document_metadata_extraction_basic',      # Informed Consent Form
        'SAP': 'document_metadata_extraction_basic',      # Statistical Analysis Plan
        'CSR': 'document_metadata_extraction_basic',      # Clinical Study Report
        
        # Investigator Documents
        'IBR': 'document_metadata_extraction_basic',      # Investigator Brochure
        'IB': 'document_metadata_extraction_basic',       # Investigator Brochure (alternate)
        
        # ====================================================================
        # SCIENTIFIC COMMUNICATIONS
        # ====================================================================
        'PUB': 'document_metadata_extraction_basic',  # Publications & Abstracts
        'MOA': 'document_metadata_extraction_basic',  # Mechanism of Action
        'EFF': 'document_metadata_extraction_basic',  # Efficacy & Outcomes
        
        # ====================================================================
        # REGULATORY & COMPLIANCE
        # ====================================================================
        'REG': 'document_metadata_extraction_basic',   # Regulatory Submissions
        'IND': 'document_metadata_extraction_basic',   # IND/CTA Submissions
        'HTA': 'document_metadata_extraction_basic',   # Health Technology Assessment
        
        # ====================================================================
        # SAFETY & PHARMACOVIGILANCE
        # ====================================================================
        'SAE': 'document_metadata_extraction_basic',       # Serious Adverse Events
        'SUSAR': 'document_metadata_extraction_basic',     # Suspected Unexpected SAR
        
        # ====================================================================
        # SITE & OPERATIONS
        # ====================================================================
        'SIT': 'document_metadata_extraction_basic',      # Site Qualification
        'TMP': 'document_metadata_extraction_basic',      # Management Plans
        'RCE': 'document_metadata_extraction_basic',      # Recruitment & Enrollment
        'TRA': 'document_metadata_extraction_basic',      # Training Materials
        
        # ====================================================================
        # STAKEHOLDER ENGAGEMENT
        # ====================================================================
        'ADV': 'document_metadata_extraction_basic',      # Advisory Boards & KOL
        'COR': 'document_metadata_extraction_basic',      # Correspondence
        
        # ====================================================================
        # MARKET & COMPETITIVE INTELLIGENCE
        # ====================================================================
        'CI': 'document_metadata_extraction_basic',       # Competitive Intelligence
        'CMM': 'document_metadata_extraction_basic',      # Company Marketing Materials
        
        # ====================================================================
        # PATIENT-CENTRIC DOCUMENTS
        # ====================================================================
        'PJM': 'document_metadata_extraction_basic',      # Patient Journey & PROs
        'MED': 'document_metadata_extraction_basic',      # Patient Education
        
        # ====================================================================
        # REAL-WORLD EVIDENCE
        # ====================================================================
        'RWE': 'document_metadata_extraction_basic',      # Real-World Evidence
        'DSC': 'document_metadata_extraction_basic',      # Disease Landscape
        'DLA': 'document_metadata_extraction_basic',      # Disease Analysis
        
        # ====================================================================
        # CLINICAL RECOMMENDATIONS
        # ====================================================================
        'REC': 'document_metadata_extraction_basic',      # Guidelines & Recommendations
        'GOV': 'document_metadata_extraction_basic',      # Governance Reports
        
        # ====================================================================
        # RESEARCH MATERIALS
        # ====================================================================
        'CRM': 'document_metadata_extraction_basic',      # Clinical Research Materials
        'CEM': 'document_metadata_extraction_basic',      # Conference & Exhibition
        
        # ====================================================================
        # DEFAULT HANDLERS
        # ====================================================================
        'UNKNOWN': 'document_metadata_extraction_basic',  # Unknown document type
        'DEFAULT': 'document_metadata_extraction_basic'   # Default fallback
    }
    
    def __init__(self, model_loader=None):
        """
        Initialize router with optional model loader
        
        Args:
            model_loader: Optional model loader for extractors that need NLP models
        """
        logger.info("=" * 70)
        logger.info("Initializing Document Type Router")
        logger.info("=" * 70)
        
        self.model_loader = model_loader
        self.loaded_extractors = {}  # Cache loaded extractors
        
        # Load generic extractor as fallback (using document_metadata_extraction_basic)
        logger.info("Loading generic extractor as fallback...")
        self.generic_extractor = self._load_generic_extractor()
        
        # Log configuration
        logger.info(f"Total routing mappings configured: {len(self.ROUTING_MAP)}")
        logger.info(f"Unique extractors referenced: {len(set(self.ROUTING_MAP.values()))}")
        
        # Log available categories
        categories = {
            'Clinical': ['PRO', 'AME', 'CRF', 'ICF', 'SAP', 'CSR', 'IBR'],
            'Scientific': ['PUB', 'MOA', 'EFF'],
            'Regulatory': ['REG', 'IND', 'HTA'],
            'Safety': ['SAE', 'SUSAR'],
            'Operations': ['SIT', 'TMP', 'RCE', 'TRA'],
            'Stakeholder': ['ADV', 'COR'],
            'Market': ['CI', 'CMM'],
            'Patient': ['PJM', 'MED'],
            'Evidence': ['RWE', 'DSC', 'DLA'],
            'Other': ['REC', 'GOV', 'CRM', 'CEM']
        }
        
        for category, codes in categories.items():
            available = [c for c in codes if c in self.ROUTING_MAP]
            logger.info(f"  {category}: {len(available)} types configured")
        
        logger.info("Document Type Router initialization complete")
        logger.info("=" * 70)
    
    def extract(self, doc_type: str, content: str, filename: str, 
                basic_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract type-specific metadata from document
        
        Args:
            doc_type: Document type code (e.g., 'PRO', 'CRF')
            content: Document text content
            filename: Document filename
            basic_metadata: Optional basic metadata already extracted
            
        Returns:
            Extraction results dictionary with:
            - doc_type: Document type code
            - filename: Document filename
            - extracted metadata fields (varies by type)
            - _extraction_info: Metadata about extraction process
            - error: Error message if extraction failed
        """
        logger.info("-" * 50)
        logger.info(f"Processing document: {filename}")
        logger.info(f"Document type: {doc_type}")
        logger.info(f"Content length: {len(content)} characters")
        
        # Get the extractor module name
        module_name = self.ROUTING_MAP.get(doc_type)
        if not module_name:
            logger.warning(f"Unknown document type '{doc_type}', falling back to generic extractor")
            module_name = 'document_metadata_extraction_basic'
        else:
            logger.info(f"Routing to extractor: {module_name}")
        
        # Get or load the extractor
        extractor = self._get_extractor(module_name)
        
        # If no extractor available, return basic results
        if not extractor:
            logger.error(f"No extractor available for document type '{doc_type}'")
            return {
                'doc_type': doc_type,
                'filename': filename,
                'error': f'No extractor available for type {doc_type}',
                'basic_metadata': basic_metadata or {},
                '_extraction_info': {
                    'extractor_module': module_name,
                    'success': False,
                    'error_type': 'ExtractorNotFound'
                }
            }
        
        # Run extraction
        try:
            logger.info(f"Starting extraction with {extractor.__class__.__name__}")
            
            # Call the extractor with appropriate method
            if hasattr(extractor, 'extract_metadata'):
                results = extractor.extract_metadata(
                    content=content,
                    filename=filename,
                    doc_type=doc_type,
                    basic_metadata=basic_metadata
                )
                logger.debug("Called extract_metadata() method")
            elif hasattr(extractor, 'extract'):
                results = extractor.extract(content, filename)
                logger.debug("Called extract() method")
            else:
                logger.error(f"Extractor {module_name} has no valid extraction method")
                results = {'error': 'Invalid extractor - no extract method found'}
            
            # Ensure results include basic info
            if 'doc_type' not in results:
                results['doc_type'] = doc_type
            if 'filename' not in results:
                results['filename'] = filename
            if basic_metadata and 'basic_metadata' not in results:
                results['basic_metadata'] = basic_metadata
            
            # Add extraction info
            results['_extraction_info'] = {
                'extractor_module': module_name,
                'extractor_class': extractor.__class__.__name__,
                'success': True,
                'fields_extracted': len([k for k in results.keys() if not k.startswith('_')])
            }
            
            logger.info(f"Extraction successful - {results['_extraction_info']['fields_extracted']} fields extracted")
            return results
            
        except Exception as e:
            logger.error(f"Error during {doc_type} extraction: {str(e)}", exc_info=True)
            return {
                'doc_type': doc_type,
                'filename': filename,
                'error': str(e),
                'basic_metadata': basic_metadata or {},
                '_extraction_info': {
                    'extractor_module': module_name,
                    'success': False,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            }
    
    def _get_extractor(self, module_name: str):
        """
        Get or load an extractor module (with caching)
        
        Args:
            module_name: Name of the extractor module
            
        Returns:
            Extractor instance or None if not found
        """
        # Check cache first
        if module_name in self.loaded_extractors:
            logger.debug(f"Using cached extractor: {module_name}")
            return self.loaded_extractors[module_name]
        
        # Try to load the module
        logger.debug(f"Loading new extractor: {module_name}")
        extractor = self._load_extractor(module_name)
        
        if extractor:
            self.loaded_extractors[module_name] = extractor
            logger.info(f"Cached extractor: {module_name}")
        
        return extractor
    
    def _load_extractor(self, module_name: str):
        """
        Load an extractor module as specified in ROUTING_MAP
        The module should contain RareDiseaseMetadataExtractor class
        """
        try:
            # Try to import the module - first from parent, then from document_utils
            module = None
            
            # Try parent directory first (where document_metadata_extraction_basic.py is)
            try:
                import_path = f"corpus_metadata.{module_name}"
                logger.debug(f"Attempting to import: {import_path}")
                module = importlib.import_module(import_path)
                logger.info(f"Successfully imported {module_name} from {import_path}")
            except ImportError as e:
                # Try document_utils as fallback
                try:
                    import_path = f"corpus_metadata.document_utils.{module_name}"
                    logger.debug(f"Attempting to import from document_utils: {import_path}")
                    module = importlib.import_module(import_path)
                    logger.info(f"Successfully imported {module_name} from document_utils")
                except ImportError as e2:
                    logger.warning(f"Could not import {module_name}: {str(e2)}")
                    return None
            
            # Currently all extractors use RareDiseaseMetadataExtractor class
            # When you add new extractors in the future, update this logic
            extractor_class_name = 'RareDiseaseMetadataExtractor'
            
            if hasattr(module, extractor_class_name):
                extractor_class = getattr(module, extractor_class_name)
                logger.debug(f"Found {extractor_class_name} in {module_name}")
            else:
                logger.error(f"{extractor_class_name} not found in {module_name}")
                available_classes = [i for i in dir(module) if not i.startswith('_') and isinstance(getattr(module, i), type)]
                logger.error(f"Available classes: {available_classes}")
                return None
            
            # Instantiate the extractor
            try:
                extractor = extractor_class()
                logger.info(f"Successfully instantiated {extractor_class_name} from {module_name}")
                return extractor
            except Exception as e:
                logger.error(f"Error instantiating {extractor_class_name}: {str(e)}")
                # Try with verbose=False parameter as fallback
                try:
                    extractor = extractor_class(verbose=False)
                    logger.info(f"Successfully instantiated {extractor_class_name} with verbose=False")
                    return extractor
                except Exception as e2:
                    logger.error(f"Failed all instantiation attempts: {str(e2)}")
                    return None
            
        except Exception as e:
            logger.error(f"Unexpected error loading {module_name}: {str(e)}", exc_info=True)
            return None

    def _load_generic_extractor(self):
        """
        Load the generic extractor as fallback
        Using document_metadata_extraction_basic as the generic extractor
        
        Returns:
            Generic extractor instance or None
        """
        # Use document_metadata_extraction_basic as the generic fallback
        generic = self._load_extractor('document_metadata_extraction_basic')
        if not generic:
            logger.warning("Generic extractor not available - extraction may fail for unknown types")
        else:
            logger.info("Generic extractor loaded successfully")
        return generic
    
    def _to_camel_case(self, snake_str: str) -> str:
        """
        Convert snake_case to CamelCase
        
        Args:
            snake_str: String in snake_case format
            
        Returns:
            String in CamelCase format
        """
        return ''.join(word.title() for word in snake_str.split('_'))
    
    def get_available_extractors(self) -> Dict[str, str]:
        """
        Get list of available extractors (checks if modules can be imported)
        
        Returns:
            Dictionary mapping document types to available extractor modules
        """
        logger.info("Checking available extractors...")
        available = {}
        
        for doc_type, module_name in self.ROUTING_MAP.items():
            # Skip checking generics
            if doc_type in ['UNKNOWN', 'DEFAULT']:
                continue
            
            # Try to import the module to check availability
            try:
                # Try the import paths
                imported = False
                for import_path in [f"corpus_metadata.{module_name}", 
                                   f"corpus_metadata.document_utils.{module_name}"]:
                    try:
                        importlib.import_module(import_path)
                        imported = True
                        break
                    except ImportError:
                        continue
                
                if imported:
                    available[doc_type] = module_name
                    logger.debug(f"  ✓ {doc_type}: {module_name}")
                else:
                    logger.debug(f"  ✗ {doc_type}: {module_name} (not found)")
            except Exception:
                logger.debug(f"  ✗ {doc_type}: {module_name} (not found)")
        
        logger.info(f"Found {len(available)} available extractors")
        return available
    
    def add_custom_route(self, doc_type: str, module_name: str):
        """
        Add a custom document type route at runtime
        
        Args:
            doc_type: Document type code to add
            module_name: Extractor module name to map to
        """
        self.ROUTING_MAP[doc_type] = module_name
        logger.info(f"Added custom route: {doc_type} → {module_name}")
    
    def get_route_info(self, doc_type: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific route
        
        Args:
            doc_type: Document type code to query
            
        Returns:
            Dictionary with route information
        """
        module_name = self.ROUTING_MAP.get(doc_type)
        if not module_name:
            return {
                'doc_type': doc_type,
                'status': 'not_configured',
                'message': f'No route configured for {doc_type}'
            }
        
        # Check if module can be imported
        can_import = False
        for import_path in [f"corpus_metadata.{module_name}", 
                           f"corpus_metadata.document_utils.{module_name}"]:
            try:
                importlib.import_module(import_path)
                can_import = True
                break
            except ImportError:
                continue
        
        return {
            'doc_type': doc_type,
            'module_name': module_name,
            'module_available': can_import,
            'is_cached': module_name in self.loaded_extractors,
            'status': 'available' if can_import else 'missing'
        }