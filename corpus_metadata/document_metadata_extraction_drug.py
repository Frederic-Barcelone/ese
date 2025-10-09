#!/usr/bin/env python3
"""
Document Drug Metadata Extractor - With Abbreviation Support
===================================================
Location: corpus_metadata/document_metadata_extraction_drug.py
Version: 3.3.0
Last Updated: 2025-01-17

CHANGES IN VERSION 3.3.0:
- Added drug approval status validation and correction
- Fixed METHYLPREDNISOLONE and other misclassified drugs
- Added approved drug database for status lookup
- Enhanced drug type classification logic
- Added investigational vs approved drug detection
- Improved drug metadata enrichment

CHANGES IN VERSION 3.2.0:
- Added abbreviation_context parameter to extract_drugs_from_text
- Implemented abbreviation candidate processing
- Added deduplication logic for abbreviation-sourced drugs
- Enhanced logging for abbreviation enrichment

CHANGES IN VERSION 3.1.1:
- Fixed TypeError with len(detected_drugs) - now handles DrugDetectionResult object properly
- Added centralized logging with self.logger initialization
- Fixed AttributeError when exception handler tries to use self.logger
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
# DRUG APPROVAL STATUS DATABASE (NEW IN v3.3.0)
# ============================================================================

# FDA-approved drugs that should never be marked as investigational
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

# Drug name variants to normalize
DRUG_NAME_VARIANTS = {
    'methylprednisolone': ['methylprednisolone', 'solu-medrol', 'medrol'],
    'rituximab': ['rituximab', 'rituxan', 'mabthera'],
    'mycophenolate': ['mycophenolate', 'mycophenolate mofetil', 'cellcept'],
    'trimethoprim-sulfamethoxazole': ['trimethoprim-sulfamethoxazole', 'tmp-smx', 'bactrim', 'septra'],
}

# ============================================================================
# Import Specialized Modules with Graceful Fallbacks
# ============================================================================

# Import the enhanced drug detector (primary module)
DETECTOR_AVAILABLE = False
try:
    from corpus_metadata.document_utils.rare_disease_drug_detector import (
        EnhancedDrugDetector,
        DrugDetectionResult,
        DrugCandidate
    )
    DETECTOR_AVAILABLE = True
    logger.debug("Enhanced Drug Detector loaded successfully")
except ImportError as e:
    logger.debug(f"Enhanced Drug Detector not available: {e}")
    EnhancedDrugDetector = None

# Import knowledge base for drug information
KNOWLEDGE_BASE_AVAILABLE = False
try:
    from corpus_metadata.document_utils.rare_disease_drug_knowledge_base import (
        DrugKnowledgeBase,
        get_knowledge_base
    )
    KNOWLEDGE_BASE_AVAILABLE = True
    logger.debug("Drug Knowledge Base loaded successfully")
except ImportError as e:
    logger.debug(f"Drug Knowledge Base not available: {e}")
    DrugKnowledgeBase = None
    get_knowledge_base = None

# Import validator for Claude validation
VALIDATOR_AVAILABLE = False
try:
    from corpus_metadata.document_utils.rare_disease_drug_validator import (
        EnhancedDrugValidator,
        ValidationConfig
    )
    VALIDATOR_AVAILABLE = True
    logger.debug("Enhanced Drug Validator loaded successfully")
except ImportError as e:
    logger.debug(f"Enhanced Drug Validator not available: {e}")
    EnhancedDrugValidator = None
    ValidationConfig = None

# Import system initializer for lexicons
INITIALIZER_AVAILABLE = False
try:
    from corpus_metadata.document_utils.metadata_system_initializer import (
        MetadataSystemInitializer
    )
    INITIALIZER_AVAILABLE = True
    logger.debug("System Initializer loaded successfully")
except ImportError as e:
    logger.debug(f"System Initializer not available: {e}")
    MetadataSystemInitializer = None

# Import basic extractor
BASIC_EXTRACTOR_AVAILABLE = False
try:
    from corpus_metadata.document_metadata_extraction_basic import RareDiseaseMetadataExtractor
    BASIC_EXTRACTOR_AVAILABLE = True
    logger.debug("Basic extractor loaded successfully")
except ImportError as e:
    logger.debug(f"Basic extractor not available: {e}")
    RareDiseaseMetadataExtractor = None

# ============================================================================
# Data Classes for Results
# ============================================================================

@dataclass
class Drug:
    """Unified drug entity representation"""
    name: str
    confidence: float
    drug_type: str
    source: str
    normalized_name: str
    mesh_id: Optional[str] = None
    rxcui: Optional[str] = None
    tty: Optional[str] = None
    canonical_name: Optional[str] = None
    claude_decision: Optional[str] = None
    claude_reason: Optional[str] = None
    occurrences: int = 1
    positions: List[Tuple[int, int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    from_abbreviation: Optional[str] = None  # Track if sourced from abbreviation
    approval_status: Optional[str] = None  # NEW: approval status
    drug_class: Optional[str] = None  # NEW: drug class

@dataclass
class ExtractionResult:
    """Complete extraction result"""
    drugs: List[Drug]
    drug_count: int
    detection_details: Dict[str, Any]
    processing_time: float
    filename: Optional[str] = None
    extraction_mode: Optional[str] = None
    error: Optional[str] = None
    abbreviation_enriched: bool = False
    status_corrections: int = 0  # NEW: count of status corrections

# ============================================================================
# Drug Status Validation Functions (NEW IN v3.3.0)
# ============================================================================

def normalize_drug_name(name: str) -> str:
    """
    Normalize drug name for lookup
    
    Args:
        name: Drug name to normalize
        
    Returns:
        Normalized drug name
    """
    if not name:
        return ''
    
    # Convert to lowercase and strip whitespace
    normalized = name.lower().strip()
    
    # Remove common suffixes
    suffixes_to_remove = [
        ' injection', ' tablet', ' capsule', ' solution',
        ' oral', ' iv', ' intravenous', ' mg', ' sodium'
    ]
    
    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)].strip()
    
    return normalized

def get_approved_drug_info(drug_name: str) -> Optional[Dict[str, Any]]:
    """
    Get approved drug information from database
    
    Args:
        drug_name: Name of drug to look up
        
    Returns:
        Dictionary with drug info or None
    """
    normalized = normalize_drug_name(drug_name)
    
    # Direct lookup
    if normalized in APPROVED_DRUGS:
        return APPROVED_DRUGS[normalized]
    
    # Check variants
    for canonical_name, variants in DRUG_NAME_VARIANTS.items():
        if normalized in [v.lower() for v in variants]:
            return APPROVED_DRUGS.get(canonical_name)
    
    # Partial match for combination drugs
    for approved_drug in APPROVED_DRUGS:
        if approved_drug in normalized or normalized in approved_drug:
            return APPROVED_DRUGS[approved_drug]
    
    return None

def validate_drug_status(drug: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Validate and correct drug approval status (NEW IN v3.3.0)
    
    Args:
        drug: Drug dictionary to validate
        
    Returns:
        Tuple of (corrected_drug, was_corrected)
    """
    drug_name = drug.get('name', '') or drug.get('drug_name', '')
    current_type = drug.get('drug_type', 'unknown')
    
    # Look up approved drug info
    approved_info = get_approved_drug_info(drug_name)
    
    if approved_info:
        was_corrected = False
        
        # Check if type needs correction
        if current_type in ['investigational', 'experimental', 'unknown']:
            drug['drug_type'] = approved_info['type']
            was_corrected = True
            logger.info(f"Corrected drug type for '{drug_name}': {current_type} → {approved_info['type']}")
        
        # Add approval status and class
        drug['approval_status'] = approved_info['type']
        drug['drug_class'] = approved_info['class']
        
        # Add approval year to metadata
        if 'metadata' not in drug:
            drug['metadata'] = {}
        drug['metadata']['fda_approval_year'] = approved_info.get('since')
        drug['metadata']['status_validated'] = True
        
        return drug, was_corrected
    
    # Not in approved list - keep as investigational if that's what it was
    if current_type == 'investigational':
        drug['approval_status'] = 'investigational'
    
    return drug, False

# ============================================================================
# Main Drug Metadata Extractor Class
# ============================================================================

class DrugMetadataExtractor:
    """
    Unified wrapper for drug extraction using specialized modules.
    This class orchestrates the detection, validation, and enrichment pipeline.
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
                 validate_drug_status: bool = True,  # NEW parameter
                 **kwargs):
        """
        Initialize the drug extractor wrapper
        
        Args:
            system_initializer: Pre-initialized MetadataSystemInitializer instance
            config_path: Optional path to configuration file
            verbose: Enable verbose logging
            use_claude: Enable Claude AI validation
            use_patterns: Enable pattern-based detection
            use_ner: Enable NER detection
            use_lexicon: Enable RxNorm lexicon
            use_knowledge_base: Enable drug knowledge base
            confidence_threshold: Minimum confidence threshold for drugs
            validate_drug_status: Enable drug approval status validation
            **kwargs: Additional arguments for compatibility
        """
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
        self.validate_drug_status = validate_drug_status  # NEW
        
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)
        
        # Initialize system components
        self.detector = None
        self.pattern_detector = None
        self.knowledge_base = None
        self.validator = None
        self.basic_extractor = None
        
        # Initialize components
        self._initialize_components()
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'total_drugs_detected': 0,
            'total_drugs_validated': 0,
            'total_processing_time': 0,
            'abbreviation_enriched_count': 0,
            'status_corrections': 0  # NEW
        }
    
    def _initialize_components(self):
        """Initialize all specialized components with proper error handling"""
        
        # Get or create system initializer
        if not self.system_initializer and INITIALIZER_AVAILABLE and self.use_lexicon:
            try:
                self.system_initializer = MetadataSystemInitializer.get_instance(
                    self.config_path or "corpus_config/main_config.yaml"
                )
                self.logger.debug("Using existing system initializer instance")
            except Exception as e:
                self.logger.debug(f"Could not get system initializer: {e}")
                try:
                    self.system_initializer = MetadataSystemInitializer(
                        self.config_path or "corpus_config/main_config.yaml"
                    )
                    if hasattr(self.system_initializer, 'initialize'):
                        self.system_initializer.initialize()
                    self.logger.debug("Created new system initializer instance")
                except Exception as e2:
                    self.logger.error(f"Failed to initialize system: {e2}")
                    self.system_initializer = None
        
        # Initialize enhanced drug detector (primary detection)
        if DETECTOR_AVAILABLE and EnhancedDrugDetector:
            try:
                detector_kwargs = {
                    'system_initializer': self.system_initializer,
                    'use_kb': True,
                    'use_patterns': self.use_patterns,
                    'use_ner': self.use_ner,
                    'use_pubtator': True,
                    'use_medical_filter': True,
                    'confidence_threshold': 0.5
                }
                
                if self.system_initializer:
                    detector_kwargs['system_initializer'] = self.system_initializer
                
                self.detector = EnhancedDrugDetector(**detector_kwargs)
                self.logger.debug("Enhanced drug detector initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize drug detector: {e}")
                self.detector = None
        
        # Initialize knowledge base
        if KNOWLEDGE_BASE_AVAILABLE and self.use_knowledge_base and get_knowledge_base:
            try:
                if self.system_initializer:
                    self.knowledge_base = get_knowledge_base(self.system_initializer)
                else:
                    self.knowledge_base = get_knowledge_base()
                self.logger.debug("Drug knowledge base initialized")
            except TypeError:
                try:
                    self.knowledge_base = get_knowledge_base()
                    self.logger.debug("Drug knowledge base initialized (no params)")
                except Exception as e:
                    self.logger.error(f"Failed to initialize knowledge base: {e}")
                    self.knowledge_base = None
            except Exception as e:
                self.logger.error(f"Failed to initialize knowledge base: {e}")
                self.knowledge_base = None
        
        # Initialize validator (for Claude validation)
        if VALIDATOR_AVAILABLE and self.use_claude and EnhancedDrugValidator:
            try:
                claude_api_key = os.environ.get('CLAUDE_API_KEY')
                
                if claude_api_key:
                    validation_config = ValidationConfig(
                        enable_caching=True,
                        use_context=True,
                        apply_medical_filter=True
                    ) if ValidationConfig else None
                    
                    validator_kwargs = {
                        'claude_api_key': claude_api_key
                    }
                    
                    if self.system_initializer:
                        validator_kwargs['system_initializer'] = self.system_initializer
                    
                    if validation_config:
                        validator_kwargs['config'] = validation_config
                    
                    self.validator = EnhancedDrugValidator(**validator_kwargs)
                    self.logger.debug("Claude validator initialized")
                else:
                    self.logger.debug("No Claude API key found - validation disabled")
                    self.use_claude = False
            except Exception as e:
                self.logger.error(f"Failed to initialize validator: {e}")
                self.use_claude = False
                self.validator = None
        
        # Initialize basic extractor for file processing
        if BASIC_EXTRACTOR_AVAILABLE and RareDiseaseMetadataExtractor:
            try:
                extractor_kwargs = {}
                
                if self.config_path:
                    extractor_kwargs['config_path'] = self.config_path
                
                if self.system_initializer:
                    extractor_kwargs['system_initializer'] = self.system_initializer
                
                self.basic_extractor = RareDiseaseMetadataExtractor(**extractor_kwargs)
                self.logger.debug("Basic document extractor initialized")
            except Exception as e:
                self.logger.debug(f"Could not initialize basic extractor: {e}")
                self.basic_extractor = None
    
    def _process_abbreviation_candidates(self, candidates: List[Dict], text: str) -> List[Dict]:
        """
        Convert abbreviation candidates to drug candidates
        
        Args:
            candidates: List of abbreviation candidates
            text: Original text for context
            
        Returns:
            List of drug candidates derived from abbreviations
        """
        drug_candidates = []
        
        for candidate in candidates:
            # Check both abbreviation and expansion as potential drugs
            for term in [candidate.get('abbreviation'), candidate.get('expansion')]:
                if not term:
                    continue
                    
                # Check if this could be a drug
                if self._could_be_drug(term):
                    drug_candidates.append({
                        'name': term,
                        'confidence': candidate.get('confidence', 0.8) * 0.9,  # Slightly reduce confidence
                        'source': 'abbreviation',
                        'occurrences': candidate.get('occurrences', 1),
                        'from_abbreviation': candidate.get('abbreviation'),
                        'context_type': candidate.get('context_type', 'unknown')
                    })
                    self.logger.debug(f"Added drug candidate from abbreviation: {term}")
        
        return drug_candidates
    
    def _could_be_drug(self, term: str) -> bool:
        """
        Check if a term could be a drug
        
        Args:
            term: Term to check
            
        Returns:
            True if term could be a drug
        """
        if not term:
            return False
        
        # Check approved drugs database first (NEW)
        if get_approved_drug_info(term):
            return True
            
        # Check knowledge base if available
        if self.knowledge_base:
            try:
                # Check if term exists in knowledge base
                if hasattr(self.knowledge_base, 'has_drug'):
                    return self.knowledge_base.has_drug(term)
                elif hasattr(self.knowledge_base, 'search'):
                    results = self.knowledge_base.search(term)
                    return len(results) > 0
            except Exception as e:
                self.logger.debug(f"Knowledge base check failed: {e}")
        
        # Common drug patterns
        drug_patterns = [
            r'.*mab$',  # monoclonal antibodies
            r'.*nib$',  # kinase inhibitors
            r'.*cycline$',
            r'.*steroid$',
            r'.*pam$',  # benzodiazepines
            r'.*azole$',  # antifungals
            r'.*cillin$',  # penicillins
            r'.*mycin$',  # antibiotics
            r'.*vir$',  # antivirals
        ]
        
        term_lower = term.lower()
        
        # Check patterns
        for pattern in drug_patterns:
            if re.match(pattern, term_lower):
                return True
        
        # Check for common drug terms
        common_drug_terms = [
            'prednisone', 'methylprednisolone', 'rituximab', 'cyclophosphamide',
            'mycophenolate', 'azathioprine', 'methotrexate', 'eculizumab',
            'avacopan', 'tavneos', 'ravulizumab', 'trimethoprim', 'sulfamethoxazole'
        ]
        
        if any(drug in term_lower for drug in common_drug_terms):
            return True
        
        return False
    
    def _is_duplicate(self, candidate: Dict, existing_drugs: List[Dict]) -> bool:
        """
        Check if a drug candidate is a duplicate
        
        Args:
            candidate: Drug candidate to check
            existing_drugs: List of existing drugs
            
        Returns:
            True if duplicate
        """
        candidate_name = candidate.get('name', '').lower()
        
        for drug in existing_drugs:
            drug_name = drug.get('name', '').lower()
            
            # Exact match
            if candidate_name == drug_name:
                return True
            
            # One contains the other (e.g., "rituximab" and "rituximab biosimilar")
            if candidate_name in drug_name or drug_name in candidate_name:
                # Keep the one with higher confidence
                if candidate.get('confidence', 0) <= drug.get('confidence', 0):
                    return True
        
        return False
    
    def extract_drugs_from_text(self, text: str, verbose: bool = False, 
                               abbreviation_context: Dict = None) -> Dict[str, Any]:
        """
        Extract drugs from text with optional Claude validation and abbreviation enrichment
        
        Args:
            text: Text to extract drugs from
            verbose: Whether to print detailed output
            abbreviation_context: Dictionary with abbreviation candidates
                - drug_candidates: List of drug-related abbreviations
                - all_abbreviations: All detected abbreviations
            
        Returns:
            Dictionary containing:
            - drugs: List of validated drug entities
            - drug_count: Number of drugs found
            - drug_detection_details: Processing metadata
            - abbreviation_enriched: Whether abbreviations were used
            - status_corrections: Number of status corrections made
        """
        try:
            # Step 1: Detect drugs using the detector
            if verbose:
                print(f"Detecting drugs in {len(text)} characters of text...")
            
            detected_drugs = self.detector.detect_drugs(text) if self.detector else []
            
            # Handle DrugDetectionResult object properly
            if isinstance(detected_drugs, DrugDetectionResult):
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
            
            # Step 2: Enrich with abbreviation candidates if provided
            abbreviation_enriched = False
            abbrev_drugs_added = 0
            
            if abbreviation_context and abbreviation_context.get('drug_candidates'):
                if verbose:
                    print(f"Processing {len(abbreviation_context['drug_candidates'])} abbreviation candidates...")
                
                # Process abbreviation candidates
                abbrev_drugs = self._process_abbreviation_candidates(
                    abbreviation_context['drug_candidates'],
                    text
                )
                
                # Add non-duplicate abbreviation drugs
                for abbrev_drug in abbrev_drugs:
                    if not self._is_duplicate(abbrev_drug, detected_drugs):
                        detected_drugs.append(abbrev_drug)
                        abbrev_drugs_added += 1
                        self.logger.debug(f"Added drug from abbreviation: {abbrev_drug['name']}")
                
                abbreviation_enriched = True
                self.logger.info(f"Abbreviation enrichment added {abbrev_drugs_added} new drug candidates")
                
                if verbose:
                    print(f"Added {abbrev_drugs_added} drugs from abbreviations")
            
            # Update count after enrichment
            total_candidates = len(detected_drugs)
            
            # Step 2.5: Validate drug approval status (NEW IN v3.3.0)
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
            
            # Step 3: Apply validation based on use_claude flag
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
                            'abbreviation_candidates': abbrev_drugs_added,
                            'status_corrections': status_corrections,
                            'total_candidates': total_candidates,
                            'final_validated': len(validated_drugs),
                            'removed_count': len(removed_drugs),
                            'reduction_rate': 1 - (len(validated_drugs) / total_candidates) if total_candidates > 0 else 0,
                            'processing_time': validation_results.get('processing_time', 0),
                            'claude_validation_used': True,
                            'validation_stages': [stage.get('name') for stage in stages]
                        },
                        'abbreviation_enriched': abbreviation_enriched,
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
                            'abbreviation_candidates': abbrev_drugs_added,
                            'status_corrections': status_corrections,
                            'total_candidates': total_candidates,
                            'final_validated': len(filtered_drugs),
                            'removed_count': total_candidates - len(filtered_drugs),
                            'reduction_rate': 1 - (len(filtered_drugs) / total_candidates) if total_candidates > 0 else 0,
                            'processing_time': 0,
                            'claude_validation_used': False,
                            'fallback_reason': 'Claude client not available'
                        },
                        'abbreviation_enriched': abbreviation_enriched,
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
                        'abbreviation_candidates': abbrev_drugs_added,
                        'status_corrections': status_corrections,
                        'total_candidates': total_candidates,
                        'final_validated': len(filtered_drugs),
                        'removed_count': total_candidates - len(filtered_drugs),
                        'reduction_rate': 1 - (len(filtered_drugs) / total_candidates) if total_candidates > 0 else 0,
                        'processing_time': 0,
                        'claude_validation_used': False,
                        'validation_method': 'confidence_threshold'
                    },
                    'abbreviation_enriched': abbreviation_enriched,
                    'status_corrections': status_corrections
                }
            
            # Step 4: Clean up drug data
            for drug in results['drugs']:
                drug.pop('positions', None)
                
                if 'count' in drug:
                    drug['occurrences'] = drug.pop('count')
                elif 'occurrences' not in drug:
                    drug['occurrences'] = 1
            
            # Update statistics
            if abbreviation_enriched:
                self.stats['abbreviation_enriched_count'] += 1
            
            self.logger.info(f"Drug extraction complete: {results['drug_count']} drugs from {total_candidates} candidates")
            if abbreviation_enriched:
                self.logger.info(f"Abbreviation enrichment: {abbrev_drugs_added} candidates added")
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
                'abbreviation_enriched': False,
                'status_corrections': 0
            }
    
    def extract_drugs_from_file(self, file_path: str, mode: str = 'intro') -> Dict[str, Any]:
        """
        Extract drugs from a document file
        
        Args:
            file_path: Path to document
            mode: 'intro' (first 10 pages) or 'full'
            
        Returns:
            Dictionary with extraction results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return self._create_error_result(f"File not found: {file_path}")
        
        # Extract text from file
        text = None
        
        if self.basic_extractor:
            try:
                if hasattr(self.basic_extractor, 'get_extracted_text'):
                    text = self.basic_extractor.get_extracted_text(str(file_path), mode)
                self.logger.info(f"Extracted {len(text)} characters from {file_path.name}")
            except Exception as e:
                self.logger.error(f"Failed to extract text: {e}")
        
        if not text:
            # Fallback to simple text reading for .txt files
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
        
        # Extract drugs from text
        results = self.extract_drugs_from_text(text)
        
        # Add file information
        results['filename'] = file_path.name
        results['extraction_mode'] = mode
        results['file_size'] = file_path.stat().st_size
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: str) -> None:
        """
        Save extraction results to JSON file
        
        Args:
            results: Extraction results dictionary
            output_file: Path to output file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        results['metadata'] = {
            'extractor_version': '3.3.0',
            'extraction_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'modules_used': {
                'detector': self.detector is not None,
                'pattern_detector': self.pattern_detector is not None,
                'knowledge_base': self.knowledge_base is not None,
                'validator': self.validator is not None and self.use_claude,
                'abbreviation_enrichment': results.get('abbreviation_enriched', False),
                'status_validation': self.validate_drug_status
            },
            'statistics': self.get_statistics()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        stats = self.stats.copy()
        
        if stats['documents_processed'] > 0:
            stats['avg_drugs_per_doc'] = stats['total_drugs_validated'] / stats['documents_processed']
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['documents_processed']
            stats['validation_rate'] = stats['total_drugs_validated'] / max(1, stats['total_drugs_detected'])
            stats['abbreviation_enrichment_rate'] = stats['abbreviation_enriched_count'] / stats['documents_processed']
            stats['status_correction_rate'] = stats['status_corrections'] / max(1, stats['total_drugs_detected'])
        
        # Add component statistics if available
        if self.detector and hasattr(self.detector, 'get_stats'):
            stats['detector_stats'] = self.detector.get_stats()
        
        if self.validator and hasattr(self.validator, 'get_statistics'):
            stats['validator_stats'] = self.validator.get_statistics()
        
        if self.knowledge_base and hasattr(self.knowledge_base, 'get_statistics'):
            stats['knowledge_base_stats'] = self.knowledge_base.get_statistics()
        
        return stats
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create an error result dictionary"""
        return {
            'drugs': [],
            'drug_count': 0,
            'error': error_message,
            'drug_detection_details': {
                'initial_candidates': 0,
                'final_validated': 0,
                'error': error_message
            },
            'abbreviation_enriched': False,
            'status_corrections': 0
        }

# ============================================================================
# Convenience Functions
# ============================================================================

def extract_drugs(text: str, verbose: bool = False, use_claude: bool = False,
                 abbreviation_context: Dict = None,
                 validate_status: bool = True) -> Dict[str, Any]:
    """
    Simple function to extract drugs from text
    
    Args:
        text: Document text
        verbose: Enable verbose output
        use_claude: Enable Claude AI validation
        abbreviation_context: Abbreviation context dictionary
        validate_status: Validate drug approval status
        
    Returns:
        Dictionary with extraction results
    """
    extractor = DrugMetadataExtractor(
        verbose=verbose, 
        use_claude=use_claude,
        validate_drug_status=validate_status
    )
    return extractor.extract_drugs_from_text(
        text, 
        verbose=verbose, 
        abbreviation_context=abbreviation_context
    )

def extract_drugs_from_pdf(pdf_path: str, mode: str = 'intro',
                          verbose: bool = False, use_claude: bool = False,
                          validate_status: bool = True) -> Dict[str, Any]:
    """
    Simple function to extract drugs from PDF
    
    Args:
        pdf_path: Path to PDF file
        mode: 'intro' (first 10 pages) or 'full'
        verbose: Enable verbose output
        use_claude: Enable Claude AI validation
        validate_status: Validate drug approval status
        
    Returns:
        Dictionary with extraction results
    """
    extractor = DrugMetadataExtractor(
        verbose=verbose, 
        use_claude=use_claude,
        validate_drug_status=validate_status
    )
    return extractor.extract_drugs_from_file(pdf_path, mode)