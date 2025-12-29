#!/usr/bin/env python3
"""
Document Drug Metadata Extractor - v13.0.0
==========================================
Location: corpus_metadata/document_metadata_extraction_drug.py
Version: 13.0.0 - FIXED CIRCULAR DEPENDENCY
Last Updated: 2025-12-27

CHANGES IN VERSION 13.0.0:
- FIXED infinite recursion / stack overflow
- Replaced RareDiseaseMetadataExtractor with DocumentReader for text extraction
- Eliminated circular dependency between drug and basic extractors

CHANGES IN VERSION 12.0.0:
- REMOVED abbreviation_context parameter from extract_drugs_from_text
- REMOVED _process_abbreviation_candidates method
- REMOVED abbreviation enrichment processing
- REMOVED abbreviation_enriched from results
- Simplified drug detection pipeline

CHANGES IN VERSION 3.3.0:
- Added drug approval status validation and correction
- Fixed METHYLPREDNISOLONE and other misclassified drugs
- Added approved drug database for status lookup
"""

import os
import json
import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from datetime import datetime

# ============================================================================
# CENTRALIZED LOGGING CONFIGURATION
# ============================================================================
try:
    from corpus_metadata.document_utils.metadata_logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('drug_extractor')
    logger.setLevel(logging.INFO)

# ============================================================================
# DRUG APPROVAL STATUS DATABASE
# ============================================================================

APPROVED_DRUGS = {
    # Corticosteroids
    'methylprednisolone': {'type': 'approved', 'class': 'corticosteroid', 'since': 1957},
    'prednisone': {'type': 'approved', 'class': 'corticosteroid', 'since': 1955},
    'prednisolone': {'type': 'approved', 'class': 'corticosteroid', 'since': 1955},
    'dexamethasone': {'type': 'approved', 'class': 'corticosteroid', 'since': 1958},
    'hydrocortisone': {'type': 'approved', 'class': 'corticosteroid', 'since': 1952},
    
    # Immunosuppressants
    'cyclophosphamide': {'type': 'approved', 'class': 'immunosuppressant', 'since': 1959},
    'azathioprine': {'type': 'approved', 'class': 'immunosuppressant', 'since': 1968},
    'mycophenolate': {'type': 'approved', 'class': 'immunosuppressant', 'since': 1995},
    'mycophenolate mofetil': {'type': 'approved', 'class': 'immunosuppressant', 'since': 1995},
    'tacrolimus': {'type': 'approved', 'class': 'immunosuppressant', 'since': 1994},
    'cyclosporine': {'type': 'approved', 'class': 'immunosuppressant', 'since': 1983},
    
    # Monoclonal antibodies
    'rituximab': {'type': 'approved', 'class': 'monoclonal antibody', 'since': 1997},
    'eculizumab': {'type': 'approved', 'class': 'monoclonal antibody', 'since': 2007},
    'ravulizumab': {'type': 'approved', 'class': 'monoclonal antibody', 'since': 2018},
    'avacopan': {'type': 'approved', 'class': 'C5a receptor antagonist', 'since': 2021},
    
    # Biosimilars
    'ruxience': {'type': 'approved', 'class': 'biosimilar', 'since': 2019},
    
    # Antibiotics
    'trimethoprim': {'type': 'approved', 'class': 'antibiotic', 'since': 1980},
    'sulfamethoxazole': {'type': 'approved', 'class': 'antibiotic', 'since': 1961},
    'trimethoprim-sulfamethoxazole': {'type': 'approved', 'class': 'antibiotic', 'since': 1973},
    
    # Common drugs
    'methotrexate': {'type': 'approved', 'class': 'antimetabolite', 'since': 1953},
    'aspirin': {'type': 'approved', 'class': 'NSAID', 'since': 1899},
    'ibuprofen': {'type': 'approved', 'class': 'NSAID', 'since': 1974},
}

DRUG_NAME_VARIANTS = {
    'methylprednisolone': ['methylprednisolone', 'solu-medrol', 'medrol'],
    'rituximab': ['rituximab', 'rituxan', 'mabthera'],
    'mycophenolate': ['mycophenolate', 'mycophenolate mofetil', 'cellcept'],
    'trimethoprim-sulfamethoxazole': ['trimethoprim-sulfamethoxazole', 'tmp-smx', 'bactrim', 'septra'],
}

# ============================================================================
# Imports with fallbacks
# ============================================================================

DETECTOR_AVAILABLE = False
try:
    from corpus_metadata.document_utils.rare_disease_drug_detector import DrugDetector, DrugDetectionResult
    DETECTOR_AVAILABLE = True
except ImportError:
    logger.debug("DrugDetector not available")
    DrugDetector = None
    DrugDetectionResult = None

INITIALIZER_AVAILABLE = False
try:
    from corpus_metadata.document_utils.metadata_system_initializer import MetadataSystemInitializer
    INITIALIZER_AVAILABLE = True
except ImportError:
    logger.debug("MetadataSystemInitializer not available")
    MetadataSystemInitializer = None

# Document reader for text extraction (no circular dependency)
READER_AVAILABLE = False
try:
    from corpus_metadata.document_metadata_reader import DocumentReader
    READER_AVAILABLE = True
except ImportError:
    logger.debug("DocumentReader not available")
    DocumentReader = None

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DrugExtractionResult:
    """Data class for drug extraction results"""
    drugs: List[Dict[str, Any]]
    drug_count: int
    detection_details: Dict[str, Any]
    processing_time: float
    filename: Optional[str] = None
    extraction_mode: Optional[str] = None
    error: Optional[str] = None
    status_corrections: int = 0

# ============================================================================
# Drug Status Validation Functions
# ============================================================================

def normalize_drug_name(name: str) -> str:
    """Normalize drug name for lookup"""
    if not name:
        return ''
    
    normalized = name.lower().strip()
    
    suffixes_to_remove = [
        ' injection', ' tablet', ' capsule', ' solution',
        ' oral', ' iv', ' intravenous', ' mg', ' sodium'
    ]
    
    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)].strip()
    
    return normalized


def get_approved_drug_info(drug_name: str) -> Optional[Dict[str, Any]]:
    """Get approved drug information from database"""
    normalized = normalize_drug_name(drug_name)
    
    if normalized in APPROVED_DRUGS:
        return APPROVED_DRUGS[normalized]
    
    for canonical_name, variants in DRUG_NAME_VARIANTS.items():
        if normalized in [v.lower() for v in variants]:
            return APPROVED_DRUGS.get(canonical_name)
    
    for approved_drug in APPROVED_DRUGS:
        if approved_drug in normalized or normalized in approved_drug:
            return APPROVED_DRUGS[approved_drug]
    
    return None


def validate_drug_status(drug: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Validate and correct drug approval status"""
    drug_name = drug.get('name', '') or drug.get('drug_name', '')
    current_type = drug.get('drug_type', 'unknown')
    
    approved_info = get_approved_drug_info(drug_name)
    
    if approved_info:
        was_corrected = False
        
        if current_type in ['investigational', 'experimental', 'unknown']:
            drug['drug_type'] = approved_info['type']
            was_corrected = True
            logger.info(f"Corrected drug type for '{drug_name}': {current_type} → {approved_info['type']}")
        
        drug['approval_status'] = approved_info['type']
        drug['drug_class'] = approved_info['class']
        
        if 'metadata' not in drug:
            drug['metadata'] = {}
        drug['metadata']['fda_approval_year'] = approved_info.get('since')
        drug['metadata']['status_validated'] = True
        
        return drug, was_corrected
    
    if current_type == 'investigational':
        drug['approval_status'] = 'investigational'
    
    return drug, False

# ============================================================================
# Main Drug Metadata Extractor Class
# ============================================================================

class DrugMetadataExtractor:
    """
    Unified wrapper for drug extraction using specialized modules.
    """
    
    def __init__(self, 
                 system_initializer: Optional[Any] = None,
                 config_path: Optional[str] = None,
                 verbose: bool = False,
                 use_claude: bool = False,
                 use_patterns: bool = True,
                 use_ner: bool = True,
                 use_lexicon: bool = True,
                 use_knowledge_base: bool = True,
                 confidence_threshold: float = 0.5,
                 validate_drug_status: bool = True,
                 **kwargs):
        """Initialize the drug extractor wrapper"""
        try:
            from corpus_metadata.document_utils.metadata_logging_config import get_logger
            self.logger = get_logger(__name__)
        except ImportError:
            self.logger = logging.getLogger(__name__)
        
        self.system_initializer = system_initializer
        self.config_path = config_path
        self.verbose = verbose
        self.use_claude = use_claude
        self.use_patterns = use_patterns
        self.use_ner = use_ner
        self.use_lexicon = use_lexicon
        self.use_knowledge_base = use_knowledge_base
        self.confidence_threshold = confidence_threshold
        self.validate_drug_status = validate_drug_status
        
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)
        
        self.detector = None
        self.pattern_detector = None
        self.knowledge_base = None
        self.validator = None
        self.document_reader = None  # Changed from basic_extractor
        
        self._initialize_components()
        
        self.stats = {
            'documents_processed': 0,
            'total_drugs_detected': 0,
            'total_drugs_validated': 0,
            'total_processing_time': 0,
            'status_corrections': 0
        }
    
    def _initialize_components(self):
        """Initialize all specialized components with proper error handling"""
        
        if not self.system_initializer and INITIALIZER_AVAILABLE and self.use_lexicon:
            try:
                self.system_initializer = MetadataSystemInitializer.get_instance(
                    self.config_path or "corpus_config/main_config.yaml"
                )
                self.logger.debug("Created new MetadataSystemInitializer instance")
            except Exception as e:
                self.logger.warning(f"Could not initialize system initializer: {e}")
        
        # Initialize drug detector
        if DETECTOR_AVAILABLE and DrugDetector:
            try:
                self.detector = DrugDetector(
                    system_initializer=self.system_initializer,
                    use_patterns=self.use_patterns,
                    use_ner=self.use_ner,
                    use_lexicon=self.use_lexicon,
                    use_knowledge_base=self.use_knowledge_base,
                    verbose=self.verbose
                )
                self.logger.debug("DrugDetector initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize DrugDetector: {e}")
        
        # Initialize pattern detector as fallback
        try:
            from corpus_metadata.document_utils.rare_disease_drug_pattern_detector import DrugPatternDetector
            self.pattern_detector = DrugPatternDetector()
            self.logger.debug("DrugPatternDetector initialized")
        except ImportError as e:
            self.logger.debug(f"Could not import DrugPatternDetector: {e}")
        
        # Initialize knowledge base
        if self.use_knowledge_base:
            try:
                from corpus_metadata.document_utils.rare_disease_drug_knowledge_base import DrugKnowledgeBase
                self.knowledge_base = DrugKnowledgeBase(
                    system_initializer=self.system_initializer
                )
                self.logger.debug("DrugKnowledgeBase initialized")
            except Exception as e:
                self.logger.debug(f"Could not initialize DrugKnowledgeBase: {e}")
        
        # Initialize document reader for text extraction
        # NOTE: We use DocumentReader instead of RareDiseaseMetadataExtractor
        # to avoid circular dependency (drug extractor <-> basic extractor)
        if READER_AVAILABLE and DocumentReader:
            try:
                self.document_reader = DocumentReader()
                self.logger.debug("DocumentReader initialized for text extraction")
            except Exception as e:
                self.logger.debug(f"Could not initialize DocumentReader: {e}")
                self.document_reader = None
    
    def _is_duplicate(self, candidate: Dict, existing_drugs: List[Dict]) -> bool:
        """Check if a drug candidate is a duplicate"""
        candidate_name = candidate.get('name', '').lower()
        
        for drug in existing_drugs:
            drug_name = drug.get('name', '').lower()
            
            if candidate_name == drug_name:
                return True
            
            if candidate_name in drug_name or drug_name in candidate_name:
                if candidate.get('confidence', 0) <= drug.get('confidence', 0):
                    return True
        
        return False
    
    def extract_drugs_from_text(self, text: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Extract drugs from text with optional Claude validation.
        
        Args:
            text: Text to extract drugs from
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary containing:
            - drugs: List of validated drug entities
            - drug_count: Number of drugs found
            - drug_detection_details: Processing metadata
            - status_corrections: Number of status corrections made
        """
        try:
            if verbose:
                print(f"Detecting drugs in {len(text)} characters of text...")
            
            detected_drugs = self.detector.detect_drugs(text) if self.detector else []
            
            # Handle DrugDetectionResult object properly
            if DrugDetectionResult and isinstance(detected_drugs, DrugDetectionResult):
                drug_list = detected_drugs.drugs if hasattr(detected_drugs, 'drugs') else []
                initial_count = len(drug_list)
                detected_drugs = [d.to_dict() if hasattr(d, 'to_dict') else d for d in drug_list]
            elif isinstance(detected_drugs, list):
                initial_count = len(detected_drugs)
            else:
                self.logger.warning(f"Unexpected type for detected_drugs: {type(detected_drugs)}")
                detected_drugs = []
                initial_count = 0
            
            if verbose:
                print(f"Initial detection: {initial_count} drug candidates found")
            
            total_candidates = len(detected_drugs)
            
            # Validate drug approval status
            status_corrections = 0
            if self.validate_drug_status:
                if verbose:
                    print(f"Validating drug approval status for {total_candidates} drugs...")
                
                for i, drug in enumerate(detected_drugs):
                    corrected_drug, was_corrected = validate_drug_status(drug)
                    detected_drugs[i] = corrected_drug
                    if was_corrected:
                        status_corrections += 1
                
                if status_corrections > 0:
                    self.logger.info(f"Corrected approval status for {status_corrections} drugs")
                    if verbose:
                        print(f"Corrected status for {status_corrections} drugs")
                
                self.stats['status_corrections'] += status_corrections
            
            # Apply validation based on use_claude flag
            if self.use_claude:
                from corpus_metadata.document_utils.rare_disease_drug_validator import DrugValidator
                
                validator = DrugValidator(
                    system_initializer=self.system_initializer,
                    claude_api_key=None,
                    use_cache=True
                )
                
                claude_available = validator.claude_client is not None
                
                if claude_available:
                    if verbose:
                        print(f"Validating {total_candidates} drugs with Claude AI...")
                    
                    validation_results = validator.validate_drug_list(
                        drug_candidates=detected_drugs,
                        context_text=text,
                        use_all_stages=True
                    )
                    
                    validated_drugs = validation_results.get('final_drugs', [])
                    removed_drugs = validation_results.get('removed_drugs', [])
                    stages = validation_results.get('stages', [])
                    
                    self.logger.debug(f"Claude validation complete: {len(validated_drugs)}/{total_candidates} drugs validated")
                    
                    if verbose:
                        print(f"Validation complete: {len(validated_drugs)} drugs validated")
                    
                    results = {
                        'drugs': validated_drugs,
                        'drug_count': len(validated_drugs),
                        'drug_detection_details': {
                            'initial_candidates': initial_count,
                            'status_corrections': status_corrections,
                            'total_candidates': total_candidates,
                            'final_validated': len(validated_drugs),
                            'removed_count': len(removed_drugs),
                            'reduction_rate': 1 - (len(validated_drugs) / total_candidates) if total_candidates > 0 else 0,
                            'processing_time': validation_results.get('processing_time', 0),
                            'claude_validation_used': True,
                            'validation_stages': [stage.get('name') for stage in stages]
                        },
                        'status_corrections': status_corrections
                    }
                    
                else:
                    # Claude not available, fall back to confidence
                    if verbose:
                        print("Claude validation requested but not available, using confidence threshold")
                    
                    self.logger.warning("Claude validation requested but client not available, falling back to confidence threshold")
                    
                    filtered_drugs = [d for d in detected_drugs if d.get('confidence', 0) > self.confidence_threshold]
                    
                    results = {
                        'drugs': filtered_drugs,
                        'drug_count': len(filtered_drugs),
                        'drug_detection_details': {
                            'initial_candidates': initial_count,
                            'status_corrections': status_corrections,
                            'total_candidates': total_candidates,
                            'final_validated': len(filtered_drugs),
                            'removed_count': total_candidates - len(filtered_drugs),
                            'reduction_rate': 1 - (len(filtered_drugs) / total_candidates) if total_candidates > 0 else 0,
                            'processing_time': 0,
                            'claude_validation_used': False,
                            'fallback_reason': 'Claude client not available'
                        },
                        'status_corrections': status_corrections
                    }
            
            else:
                # use_claude is False, just filter by confidence
                if verbose:
                    print(f"Filtering {total_candidates} drugs by confidence threshold (>{self.confidence_threshold})")
                
                filtered_drugs = [d for d in detected_drugs if d.get('confidence', 0) > self.confidence_threshold]
                
                results = {
                    'drugs': filtered_drugs,
                    'drug_count': len(filtered_drugs),
                    'drug_detection_details': {
                        'initial_candidates': initial_count,
                        'status_corrections': status_corrections,
                        'total_candidates': total_candidates,
                        'final_validated': len(filtered_drugs),
                        'removed_count': total_candidates - len(filtered_drugs),
                        'reduction_rate': 1 - (len(filtered_drugs) / total_candidates) if total_candidates > 0 else 0,
                        'processing_time': 0,
                        'claude_validation_used': False,
                        'validation_method': 'confidence_threshold'
                    },
                    'status_corrections': status_corrections
                }
            
            # Clean up drug data
            for drug in results['drugs']:
                drug.pop('positions', None)
                
                if 'count' in drug:
                    drug['occurrences'] = drug.pop('count')
                elif 'occurrences' not in drug:
                    drug['occurrences'] = 1
            
            self.logger.info(f"Drug extraction complete: {results['drug_count']} drugs from {total_candidates} candidates")
            if status_corrections > 0:
                self.logger.info(f"Status corrections: {status_corrections} drugs corrected")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in drug extraction: {e}", exc_info=True)
            
            return {
                'drugs': [],
                'drug_count': 0,
                'drug_detection_details': {
                    'initial_candidates': 0,
                    'final_validated': 0,
                    'error': str(e),
                    'claude_validation_used': False
                },
                'status_corrections': 0
            }
    
    def extract_drugs_from_file(self, file_path: str, mode: str = 'intro') -> Dict[str, Any]:
        """Extract drugs from a document file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return self._create_error_result(f"File not found: {file_path}")
        
        text = None
        
        # Use DocumentReader for text extraction (no circular dependency)
        if self.document_reader:
            try:
                result = self.document_reader.read_document(str(file_path), mode)
                text = result.get('content', '')
                if text:
                    self.logger.info(f"Extracted {len(text)} characters from {file_path.name}")
            except Exception as e:
                self.logger.error(f"Failed to extract text with DocumentReader: {e}")
        
        # Fallback for text files
        if not text:
            if file_path.suffix.lower() == '.txt':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        if mode == 'intro':
                            text = text[:50000]
                except Exception as e:
                    return self._create_error_result(f"Failed to read file: {e}")
            else:
                return self._create_error_result("Text extraction not available")
        
        results = self.extract_drugs_from_text(text)
        results['filename'] = file_path.name
        results['extraction_mode'] = mode
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return {
            **self.stats,
            'detector_available': self.detector is not None,
            'pattern_detector_available': self.pattern_detector is not None,
            'knowledge_base_available': self.knowledge_base is not None,
            'document_reader_available': self.document_reader is not None
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result dictionary"""
        return {
            'drugs': [],
            'drug_count': 0,
            'drug_detection_details': {
                'initial_candidates': 0,
                'final_validated': 0,
                'error': error_message
            },
            'status_corrections': 0
        }

# ============================================================================
# Convenience Functions
# ============================================================================

def extract_drugs(text: str, verbose: bool = False, use_claude: bool = False,
                 validate_status: bool = True) -> Dict[str, Any]:
    """Simple function to extract drugs from text"""
    extractor = DrugMetadataExtractor(
        verbose=verbose, 
        use_claude=use_claude,
        validate_drug_status=validate_status
    )
    return extractor.extract_drugs_from_text(text, verbose=verbose)


def extract_drugs_from_pdf(pdf_path: str, mode: str = 'intro',
                          verbose: bool = False, use_claude: bool = False,
                          validate_status: bool = True) -> Dict[str, Any]:
    """Simple function to extract drugs from PDF"""
    extractor = DrugMetadataExtractor(
        verbose=verbose, 
        use_claude=use_claude,
        validate_drug_status=validate_status
    )
    return extractor.extract_drugs_from_file(pdf_path, mode)