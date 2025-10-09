#!/usr/bin/env python3
"""
Document Disease Metadata Extractor - With Abbreviation and ID Support
=========================================================
Location: corpus_metadata/document_metadata_extraction_disease.py
Version: 5.4.0
Last Updated: 2025-01-17

CHANGES IN VERSION 5.4.0:
- Added symptom/finding filtering to exclude clinical observations
- Fixed incomplete abbreviation expansions (GPA/MPA/EGPA, SARS, PJP)
- Enhanced disease name validation and completeness checking
- Added semantic type classification (disease vs symptom vs finding)
- Improved ID enrichment for common renal and clinical terms
- Added validation warnings for data quality issues

CHANGES IN VERSION 5.3.0:
- Added ID normalization and surfacing in output
- Include primary_id and all_ids in disease output
- Added canonical_name from KB resolution
- Improved ID formatting for better visibility

CHANGES IN VERSION 5.2.0:
- Fixed abbreviation-based disease validation issue
- Disease names from abbreviations now properly validated
- Improved detection_method assignment for proper validation flow

CHANGES IN VERSION 5.1.0:
- Added abbreviation_context parameter to extract method
- Implemented abbreviation candidate processing for diseases
- Added context-aware disambiguation (MPA, AAV cases)
- Enhanced deduplication logic
- Improved logging for abbreviation enrichment
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import asdict
import os
import sys

# Configure logging
logger = logging.getLogger(__name__)

# Fix import paths - try multiple approaches
RareDiseaseDetector = None
create_detector = None
BasicDocumentExtractor = None

try:
    from corpus_metadata.document_utils.rare_disease_disease_detector import RareDiseaseDetector, create_detector
except ImportError as e:
    logger.error(f"Could not import RareDiseaseDetector: {e}")
    RareDiseaseDetector = None
    create_detector = None

# DO NOT IMPORT RareDiseaseMetadataExtractor - it causes circular import!
RareDiseaseMetadataExtractor = None  # Set to None if needed for compatibility

# ============================================================================
# ID NORMALIZATION (matching entity_abbreviation_promotion.py)
# ============================================================================

# ID key mappings to canonical forms
ID_KEY_MAP = {
    # UMLS variants
    'umls': 'UMLS', 'umls_cui': 'UMLS', 'cui': 'UMLS', 'umls_id': 'UMLS',
    
    # SNOMED variants
    'snomed': 'SNOMED', 'snomed_ct': 'SNOMED', 'snomedct': 'SNOMED', 'snomed_id': 'SNOMED',
    
    # ICD variants
    'icd10': 'ICD10', 'icd-10': 'ICD10', 'icd_10': 'ICD10', 'icd10_code': 'ICD10',
    'icd9': 'ICD9', 'icd-9': 'ICD9', 'icd_9': 'ICD9', 'icd9_code': 'ICD9',
    'icd11': 'ICD11', 'icd-11': 'ICD11', 'icd_11': 'ICD11',
    
    # Orphanet variants
    'orpha': 'ORPHA', 'orpha_code': 'ORPHA', 'orphanet': 'ORPHA', 'orpha_id': 'ORPHA',
    'orphacode': 'ORPHA',  # Additional variant for orphacode field
    
    # Disease Ontology variants
    'doid': 'DOID', 'do_id': 'DOID', 'disease_ontology': 'DOID',
    
    # MONDO variants
    'mondo': 'MONDO', 'mondo_id': 'MONDO',
    
    # MeSH variants
    'mesh': 'MESH', 'mesh_id': 'MESH', 'mesh_code': 'MESH',
    
    # OMIM variants
    'omim': 'OMIM', 'omim_id': 'OMIM', 'mim': 'OMIM',
}

# Required disease IDs (at least one needed)
REQUIRED_DISEASE_IDS = (
    "UMLS", "SNOMED", "ICD10", "ICD9", "ORPHA", 
    "DOID", "MESH", "OMIM", "MONDO"
)

# Preferred ID order for diseases
PREFERRED_DISEASE_KEYS = ('ORPHA', 'DOID', 'UMLS', 'SNOMED', 'MONDO', 'MESH', 'OMIM', 'ICD10', 'ICD9')

# ============================================================================
# SYMPTOM AND FINDING FILTERS (NEW in v5.4.0)
# ============================================================================

# Symptoms and clinical findings to exclude (NOT diseases)
SYMPTOM_KEYWORDS = {
    'pain', 'ache', 'aching',
    'fever', 'fatigue', 'weakness',
    'nausea', 'vomiting', 'diarrhea',
    'cough', 'dyspnea', 'tachycardia',
    'edema', 'swelling', 'rash'
}

# Clinical findings/observations (NOT diseases)
FINDING_KEYWORDS = {
    'impairment', 'insufficiency', 'deficiency',
    'elevation', 'reduction', 'decrease', 'increase',
    'hemorrhage', 'bleeding', 'hematuria',
    'proteinuria', 'azotemia', 'uremia',
    'dysfunction', 'abnormality'
}

# Generic/vague terms that need additional context
GENERIC_TERMS = {
    'neuropathy', 'myopathy', 'arthropathy',
    'infection', 'infections', 'inflammation',
    'disorder', 'condition', 'syndrome'
}

# Terms that should ALWAYS be kept (even if they match filters)
ALWAYS_KEEP_DISEASES = {
    'pneumocystis jirovecii pneumonia',
    'end-stage renal disease',
    'chronic kidney disease',
    'acute kidney injury',
    'diffuse alveolar hemorrhage',
    'peripheral neuropathy',  # When specific
    'diabetic neuropathy',
    'glomerulonephritis',
    'anca-associated vasculitis',
    'microscopic polyangiitis',
    'granulomatosis with polyangiitis',
    'eosinophilic granulomatosis with polyangiitis',
    'myasthenia gravis'
}

# Standalone terms that are diseases when alone
STANDALONE_DISEASE_TERMS = {
    'vasculitis', 'nephritis', 'hepatitis', 'arthritis',
    'pneumonia', 'meningitis', 'encephalitis', 'myocarditis',
    'glomerulonephritis', 'pyelonephritis'
}

# ============================================================================
# ABBREVIATION EXPANSION CORRECTIONS (NEW in v5.4.0)
# ============================================================================

# Corrected abbreviation expansions for common issues
ABBREVIATION_CORRECTIONS = {
    'GPA/MPA/EGPA': 'Granulomatosis with Polyangiitis / Microscopic Polyangiitis / Eosinophilic Granulomatosis with Polyangiitis',
    'GPA': 'Granulomatosis with Polyangiitis',
    'MPA': 'Microscopic Polyangiitis',
    'EGPA': 'Eosinophilic Granulomatosis with Polyangiitis',
    'AAV': 'ANCA-Associated Vasculitis',
    'ANCA': 'Anti-Neutrophil Cytoplasmic Antibody',
    'SARS': 'Severe Acute Respiratory Syndrome',
    'PJP': 'Pneumocystis jirovecii Pneumonia',
    'COVID-19': 'Coronavirus Disease 2019',
    'COPD': 'Chronic Obstructive Pulmonary Disease',
    'CKD': 'Chronic Kidney Disease',
    'ESRD': 'End-Stage Renal Disease',
    'AKI': 'Acute Kidney Injury'
}

# ============================================================================
# ID ENRICHMENT MAPPINGS (NEW in v5.4.0)
# ============================================================================

# Common disease → ID mappings for enrichment
DISEASE_ID_ENRICHMENT = {
    'glomerulonephritis': {
        'ICD10': 'N05.9',
        'SNOMED': '36171008',
        'UMLS': 'C0017658'
    },
    'pneumocystis jirovecii pneumonia': {
        'ICD10': 'B59',
        'SNOMED': '233705009',
        'UMLS': 'C1535939'
    },
    'end-stage renal disease': {
        'ICD10': 'N18.6',
        'SNOMED': '46177005',
        'UMLS': 'C0022661'
    },
    'chronic kidney disease': {
        'ICD10': 'N18.9',
        'SNOMED': '709044004',
        'UMLS': 'C1561643'
    },
    'acute kidney injury': {
        'ICD10': 'N17.9',
        'SNOMED': '14669001',
        'UMLS': 'C2609414'
    },
    'peripheral neuropathy': {
        'ICD10': 'G62.9',
        'SNOMED': '42658009',
        'UMLS': 'C0031117'
    },
    'renal vasculitis': {
        'SNOMED': '266555001',
        'UMLS': 'C0403544'
    }
}

def normalize_ids(entity: Any) -> Dict[str, Any]:
    """
    Normalize all ID keys to canonical form from disease entity
    
    Args:
        entity: Disease entity (dict or object)
        
    Returns:
        Dictionary with normalized ID keys
    """
    normalized = {}
    
    # Handle both dict and object formats
    if isinstance(entity, dict):
        items = entity.items()
    else:
        # Convert object to dict-like items
        items = []
        for attr in dir(entity):
            if not attr.startswith('_'):
                value = getattr(entity, attr, None)
                if value is not None:
                    items.append((attr, value))
    
    for key, value in items:
        # Skip None or empty values
        if value is None:
            continue
        
        # Check if this key maps to an ID
        canonical_key = ID_KEY_MAP.get(key.lower(), None)
        
        if canonical_key:
            # Only add if value is meaningful
            if isinstance(value, str) and value.strip():
                # Normalize ORPHA codes to include prefix
                if canonical_key == 'ORPHA' and not value.upper().startswith('ORPHA:'):
                    normalized[canonical_key] = f"ORPHA:{value.strip()}"
                else:
                    normalized[canonical_key] = value.strip()
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                normalized[canonical_key] = value
            elif value and not isinstance(value, str):
                normalized[canonical_key] = value
    
    return normalized

def get_primary_id(normalized_ids: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Get the primary ID for a disease based on preferred order
    
    Args:
        normalized_ids: Dictionary of normalized IDs
        
    Returns:
        Tuple of (id_type, id_value) or (None, None)
    """
    for key in PREFERRED_DISEASE_KEYS:
        if key in normalized_ids and normalized_ids[key]:
            return key, normalized_ids[key]
    
    # Fallback to any available ID
    for key, value in normalized_ids.items():
        if value:
            return key, value
    
    return None, None

# ============================================================================
# DISEASE CLASSIFICATION AND VALIDATION (NEW in v5.4.0)
# ============================================================================

def classify_semantic_type(name: str, has_ids: bool = False) -> str:
    """
    Classify entity as disease, symptom, finding, or generic
    
    Args:
        name: Entity name
        has_ids: Whether entity has disease identifiers
        
    Returns:
        Semantic type: 'disease', 'symptom', 'finding', 'generic', or 'unknown'
    """
    name_lower = name.lower().strip()
    
    # If it has disease IDs, it's likely a disease
    if has_ids:
        return 'disease'
    
    # Check if it's in always-keep list
    if name_lower in ALWAYS_KEEP_DISEASES:
        return 'disease'
    
    # Check if it's a standalone disease term
    if name_lower in STANDALONE_DISEASE_TERMS:
        return 'disease'
    
    # Check for symptom keywords
    for keyword in SYMPTOM_KEYWORDS:
        if keyword in name_lower:
            return 'symptom'
    
    # Check for finding keywords
    for keyword in FINDING_KEYWORDS:
        if name_lower.endswith(keyword) or keyword in name_lower.split():
            return 'finding'
    
    # Check for generic terms
    for keyword in GENERIC_TERMS:
        if name_lower == keyword:  # Exact match only
            return 'generic'
    
    # Check for disease indicators
    disease_indicators = [
        'vasculitis', 'polyangiitis', 'granulomatosis',
        'syndrome', 'disease', 'disorder',
        'nephritis', 'hepatitis', 'arthritis', 'dermatitis',
        'pneumonia', 'carcinoma', 'lymphoma', 'leukemia',
        'diabetes', 'hypertension', 'anemia', 'thrombosis',
        'fibrosis', 'cirrhosis', 'stenosis', 'sclerosis'
    ]
    
    if any(indicator in name_lower for indicator in disease_indicators):
        return 'disease'
    
    # Check disease suffix patterns
    disease_patterns = [
        r'.*itis$',  # inflammation diseases
        r'.*osis$',  # condition/disease
        r'.*emia$',  # blood conditions
        r'.*pathy$',  # disease/disorder
        r'.*oma$',   # tumors/cancers
    ]
    
    for pattern in disease_patterns:
        if re.match(pattern, name_lower):
            return 'disease'
    
    return 'unknown'

def is_complete_expansion(expansion: str) -> bool:
    """
    Check if an abbreviation expansion is complete (not truncated)
    
    Args:
        expansion: The expansion to check
        
    Returns:
        True if expansion appears complete
    """
    if not expansion or len(expansion) < 3:
        return False
    
    # Patterns indicating incomplete expansions
    incomplete_patterns = [
        r'-Associated$',  # "ANCA-Associated" without the disease
        r'-Onset$',       # "Childhood-Onset" without the disease
        r'^[A-Z]+$',      # All caps (likely still an abbreviation)
        r'^[A-Z][a-z]+ [A-Z]+$',  # "Associated Vasculitis" (partial)
    ]
    
    for pattern in incomplete_patterns:
        if re.search(pattern, expansion):
            logger.debug(f"Incomplete expansion detected: '{expansion}' matches pattern '{pattern}'")
            return False
    
    return True

def enrich_disease_ids(disease_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich disease with standard IDs if missing
    
    Args:
        disease_dict: Disease dictionary
        
    Returns:
        Enriched disease dictionary
    """
    name_lower = disease_dict.get('name', '').lower().strip()
    
    # Check if we have enrichment data for this disease
    if name_lower in DISEASE_ID_ENRICHMENT:
        enrichment_data = DISEASE_ID_ENRICHMENT[name_lower]
        
        # Add IDs if not already present
        for id_type, id_value in enrichment_data.items():
            if id_type not in disease_dict or not disease_dict[id_type]:
                disease_dict[id_type] = id_value
                logger.debug(f"Enriched '{name_lower}' with {id_type}: {id_value}")
    
    return disease_dict

# ============================================================================
# Modern Disease Metadata Extractor with Abbreviation Support
# ============================================================================

class DiseaseMetadataExtractor:
    """
    Modern, clean wrapper for the optimized rare disease detection system
    with abbreviation-first pipeline support and ID normalization.
    """
    
    def __init__(self, 
                 mode: str = "balanced",
                 use_claude: bool = False,
                 claude_api_key: Optional[str] = None,
                 config_path: Optional[str] = None,
                 verbose: bool = False,
                 filter_symptoms: bool = True):  # NEW parameter
        """Initialize the disease extractor with modern defaults."""
        self.mode = mode
        self.use_claude = use_claude
        self.verbose = verbose
        self.filter_symptoms = filter_symptoms  # NEW
        
        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.INFO)
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
        
         
        from corpus_metadata.document_utils.metadata_system_initializer import MetadataSystemInitializer
        self.system_initializer = MetadataSystemInitializer.get_instance(
            config_path or "corpus_config/config.yaml"
        )
        
         
        self._init_detector(mode, use_claude, claude_api_key, config_path)
        
        # Initialize file extractor if available
        self.file_extractor = None
        if BasicDocumentExtractor:
            try:
                self.file_extractor = BasicDocumentExtractor(config_path)
                logger.info("File extraction enabled via BasicDocumentExtractor")
            except Exception as e:
                logger.warning(f"File extraction not available: {e}")
        
        # Statistics tracking
        self.stats = {
            'documents_processed': 0,
            'diseases_detected': 0,
            'abbreviation_enriched_count': 0,
            'diseases_with_ids': 0,
            'symptoms_filtered': 0,  # NEW
            'findings_filtered': 0,  # NEW
            'ids_enriched': 0  # NEW
        }
        
        logger.info(f"DiseaseMetadataExtractor initialized (mode={mode}, claude={use_claude}, filter_symptoms={filter_symptoms})")
        
    def _init_detector(self, mode: str, use_claude: bool, 
                      claude_api_key: Optional[str], config_path: Optional[str]):
        """Initialize the rare disease detector with system initializer"""
        if not RareDiseaseDetector:
            logger.warning("RareDiseaseDetector not found, using fallback detector")
            self.detector = self._create_fallback_detector(mode, config_path)
            return
        
        # Initialize system initializer FIRST
        from corpus_metadata.document_utils.metadata_system_initializer import MetadataSystemInitializer
        system_initializer = MetadataSystemInitializer.get_instance(
            config_path or "corpus_config/config.yaml"
        )
        
 
        orphanet_db_path = None
        doid_db_path = None
        
        if system_initializer:
            # Try get_resource method first (returns string)
            if hasattr(system_initializer, 'get_resource'):
                orphanet_db_path = system_initializer.get_resource('disease_orphanet_db')
                doid_db_path = system_initializer.get_resource('disease_ontology_db')
            
            # Fallback: Extract from config
            if not orphanet_db_path and hasattr(system_initializer, 'config'):
                resources = system_initializer.config.get('resources', {})
                orphanet_resource = resources.get('disease_orphanet_db')
                
                # Handle dict vs string
                if isinstance(orphanet_resource, dict):
                    orphanet_db_path = orphanet_resource.get('path')
                elif isinstance(orphanet_resource, str):
                    orphanet_db_path = orphanet_resource
            
            if not doid_db_path and hasattr(system_initializer, 'config'):
                resources = system_initializer.config.get('resources', {})
                doid_resource = resources.get('disease_ontology_db')
                
                # Handle dict vs string
                if isinstance(doid_resource, dict):
                    doid_db_path = doid_resource.get('path')
                elif isinstance(doid_resource, str):
                    doid_db_path = doid_resource
        
        logger.info(f"Database paths extracted: Orphanet={orphanet_db_path}, DOID={doid_db_path}")
        
        # Prepare Claude client if requested
        claude_client = None
        if use_claude:
            claude_client = self._create_claude_client(claude_api_key)
        
        # Load config if provided
        config = {}
        if config_path and Path(config_path).exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Could not load config: {e}")
        
        # Create detector WITH database paths
        try:
            if create_detector:
                self.detector = create_detector(
                    mode=mode,
                    claude_client=claude_client,
                    verbose=self.verbose,
                    system_initializer=system_initializer,
                    kb_resolver=system_initializer,
                    config_path=config_path,
                    orphanet_db_path=orphanet_db_path,   
                    doid_db_path=doid_db_path,           
                    **config.get('disease_detection', {})
                )
            else:
                self.detector = RareDiseaseDetector(
                    mode=mode,
                    claude_client=claude_client,
                    verbose=self.verbose,
                    system_initializer=system_initializer,
                    kb_resolver=system_initializer,
                    config_path=config_path,
                    orphanet_db_path=orphanet_db_path,   
                    doid_db_path=doid_db_path,           
                    **config.get('disease_detection', {})
                )
            logger.info("RareDiseaseDetector initialized successfully with database paths")
        except Exception as e:
            logger.error(f"Could not initialize RareDiseaseDetector: {e}", exc_info=True)
            logger.warning("Using fallback detector")
            self.detector = self._create_fallback_detector(mode, config_path)


    
    def _create_fallback_detector(self, mode: str, config_path: Optional[str]):
        """Create a simple fallback detector using the metadata system"""
        from corpus_metadata.document_utils.metadata_system_initializer import MetadataSystemInitializer
        
        class FallbackDetector:
            def __init__(self, mode, config_path):
                self.mode = mode
                self.initializer = MetadataSystemInitializer.get_instance(config_path or "corpus_config/config.yaml")
                self.disease_lexicon = self.initializer.get_lexicon('disease') or {}
                logger.info(f"Fallback detector initialized with {len(self.disease_lexicon)} disease entries")
            
            def detect_diseases(self, text):
                """Simple disease detection using lexicon"""
                diseases = []
                if not text or not self.disease_lexicon:
                    return type('Result', (), {'diseases': diseases})()
                
                text_lower = text.lower()
                for disease_name in self.disease_lexicon:
                    if disease_name.lower() in text_lower:
                        disease = type('Disease', (), {
                            'name': disease_name,
                            'confidence': 0.7,
                            'occurrences': text_lower.count(disease_name.lower()),
                            'orphacode': self.disease_lexicon[disease_name].get('orphacode') if isinstance(self.disease_lexicon[disease_name], dict) else None,
                            'detection_method': 'lexicon',
                            'positions': [],
                            'matched_terms': [disease_name]
                        })()
                        diseases.append(disease)
                
                return type('Result', (), {'diseases': diseases})()
            
            def set_mode(self, mode):
                self.mode = mode
            
            def set_confidence_threshold(self, threshold):
                pass
            
            def get_statistics(self):
                return {"mode": self.mode, "lexicon_size": len(self.disease_lexicon)}
            
            def clear_cache(self):
                pass
        
        return FallbackDetector(mode, config_path)
    
    def _create_claude_client(self, api_key: Optional[str]):
        """Create Claude client for validation"""
        api_key = api_key or os.environ.get('CLAUDE_API_KEY')
        if not api_key:
            logger.warning("Claude API key not found, validation disabled")
            return None
        
        try:
            from anthropic import Anthropic
            return Anthropic(api_key=api_key)
        except ImportError:
            logger.warning("Anthropic library not installed, Claude validation disabled")
            return None
    
    # =========================================================================
    # Abbreviation Processing Methods
    # =========================================================================
    
    def _process_abbreviation_candidates(self, candidates: List[Dict], text: str) -> List[Dict]:
        """
        Convert abbreviation candidates to disease candidates with corrections
        
        Args:
            candidates: List of disease-related abbreviation candidates
            text: Original text for context
            
        Returns:
            List of disease candidates derived from abbreviations
        """
        disease_candidates = []
        
        for candidate in candidates:
            # Check both abbreviation and expansion as potential diseases
            abbrev = candidate.get('abbreviation')
            expansion = candidate.get('expansion')
            
            # APPLY ABBREVIATION CORRECTIONS (NEW in v5.4.0)
            if abbrev in ABBREVIATION_CORRECTIONS:
                corrected_expansion = ABBREVIATION_CORRECTIONS[abbrev]
                if expansion != corrected_expansion:
                    logger.info(f"Corrected abbreviation '{abbrev}': '{expansion}' → '{corrected_expansion}'")
                    expansion = corrected_expansion
                    candidate['expansion'] = corrected_expansion
            
            # Get any IDs that came with the abbreviation
            abbrev_ids = {}
            for key, value in candidate.items():
                canonical_key = ID_KEY_MAP.get(key.lower())
                if canonical_key and value:
                    abbrev_ids[canonical_key] = value
            
            # Process expansion first (it's more likely to be the actual disease name)
            if expansion and self._is_actual_disease_name(expansion):
                # Check if expansion is complete
                if not is_complete_expansion(expansion):
                    logger.warning(f"Incomplete expansion detected for '{abbrev}': '{expansion}' - skipping")
                    continue
                
                # This is a real disease name - mark it for proper validation
                disease_dict = {
                    'name': expansion,
                    'confidence': 0.95,  # High confidence for resolved disease names
                    'detection_method': 'pattern',  # Use 'pattern' so it goes through validation
                    'occurrences': candidate.get('occurrences', 1),
                    'source_abbreviation': abbrev,
                    'positions': [],
                    'matched_terms': [expansion, abbrev] if abbrev else [expansion],
                    'canonical_name': candidate.get('canonical_name', expansion)
                }
                # Add any IDs from abbreviation resolution
                disease_dict.update(abbrev_ids)
                
                # Enrich with standard IDs if available
                disease_dict = enrich_disease_ids(disease_dict)
                
                disease_candidates.append(disease_dict)
                logger.debug(f"Added disease from abbreviation expansion: {expansion} (from {abbrev})")
            
            # Also check the abbreviation itself if it could be a disease code
            elif abbrev and self._could_be_disease(abbrev):
                disease_dict = {
                    'name': abbrev,
                    'confidence': candidate.get('confidence', 0.8) * 0.9,
                    'detection_method': 'abbreviation_enrichment',  # Keep as enrichment
                    'occurrences': candidate.get('occurrences', 1),
                    'from_abbreviation': abbrev,
                    'context_type': candidate.get('context_type', 'unknown'),
                    'positions': [],
                    'matched_terms': [abbrev]
                }
                # Add any IDs from abbreviation resolution
                disease_dict.update(abbrev_ids)
                
                disease_candidates.append(disease_dict)
                logger.debug(f"Added disease candidate from abbreviation: {abbrev}")
        
        return disease_candidates
    
    def _is_actual_disease_name(self, term: str) -> bool:
        """
        Check if a term is an actual disease name (not just an abbreviation)
        
        Args:
            term: Term to check
            
        Returns:
            True if term is an actual disease name
        """
        if not term or len(term) < 5:  # Too short to be a disease name
            return False
        
        term_lower = term.lower()
        
        # Strong disease indicators
        strong_indicators = [
            'vasculitis', 'polyangiitis', 'granulomatosis',
            'syndrome', 'disease', 'disorder', 'deficiency',
            'nephritis', 'hepatitis', 'arthritis', 'dermatitis',
            'pneumonia', 'carcinoma', 'lymphoma', 'leukemia',
            'diabetes', 'hypertension', 'anemia', 'thrombosis',
            'fibrosis', 'cirrhosis', 'stenosis', 'sclerosis'
        ]
        
        # Check for strong indicators
        if any(indicator in term_lower for indicator in strong_indicators):
            return True
        
        # Check for disease patterns
        disease_patterns = [
            r'.*itis$',  # inflammation diseases
            r'.*osis$',  # condition/disease
            r'.*emia$',  # blood conditions
            r'.*pathy$',  # disease/disorder
            r'.*oma$',   # tumors/cancers
        ]
        
        for pattern in disease_patterns:
            if re.match(pattern, term_lower):
                return True
        
        return False
    
    def _could_be_disease(self, term: str) -> bool:
        """
        Check if a term could be a disease (looser criteria than _is_actual_disease_name)
        
        Args:
            term: Term to check
            
        Returns:
            True if term could be a disease
        """
        if not term:
            return False
        
        term_lower = term.lower()
        
        # Disease-related terms
        disease_terms = [
            'covid', 'sars', 'mers', 'hiv', 'aids', 'copd',
            'anca', 'gpa', 'mpa', 'aav',  # vasculitis abbreviations
            'ibd', 'ibs', 'ckd', 'esrd',  # GI/kidney diseases
            'ami', 'chf', 'cad', 'pvd',   # cardiovascular
            'ards', 'ild', 'ipf',         # pulmonary
            'ms', 'als', 'pd', 'ad'       # neurological
        ]
        
        if term_lower in disease_terms:
            return True
        
        # Check if it matches disease patterns (less strict)
        if re.match(r'^[A-Z]{2,5}$', term):  # Could be disease abbreviation
            return True
        
        return False
    
    def _resolve_context_conflicts(self, candidates: List[Dict], 
                                  all_abbreviations: List[Dict]) -> List[Dict]:
        """
        Resolve context-dependent conflicts for disease abbreviations
        
        Args:
            candidates: Disease candidates from abbreviations
            all_abbreviations: All detected abbreviations for context
            
        Returns:
            Resolved disease candidates
        """
        # Check for ANCA context (indicates vasculitis document)
        has_anca_context = any(
            'ANCA' in abbr.get('abbreviation', '').upper() 
            for abbr in all_abbreviations
        )
        
        resolved = []
        for candidate in candidates:
            source_abbrev = candidate.get('source_abbreviation', '')
            from_abbrev = candidate.get('from_abbreviation', '')
            abbrev = source_abbrev or from_abbrev
            
            # Resolve specific abbreviations based on context
            if abbrev == 'MPA' and has_anca_context:
                if candidate['name'].lower() != 'microscopic polyangiitis':
                    candidate['name'] = 'Microscopic Polyangiitis'
                    candidate['canonical_name'] = 'Microscopic Polyangiitis'
                candidate['confidence'] = 0.95
                candidate['detection_method'] = 'pattern'  # Ensure proper validation
                # Add standard IDs if known
                candidate['ORPHA'] = 'ORPHA:727'
                logger.info("Resolved MPA as Microscopic Polyangiitis (disease) based on ANCA context")
            
            elif abbrev == 'AAV' and has_anca_context:
                if 'vasculitis' not in candidate['name'].lower():
                    candidate['name'] = 'ANCA-Associated Vasculitis'
                    candidate['canonical_name'] = 'ANCA-Associated Vasculitis'
                candidate['confidence'] = 0.95
                candidate['detection_method'] = 'pattern'  # Ensure proper validation
                candidate['ORPHA'] = 'ORPHA:156152'
                logger.info("Resolved AAV as ANCA-Associated Vasculitis based on context")
            
            elif abbrev == 'GPA' and has_anca_context:
                if 'granulomatosis' not in candidate['name'].lower():
                    candidate['name'] = 'Granulomatosis with Polyangiitis'
                    candidate['canonical_name'] = 'Granulomatosis with Polyangiitis'
                candidate['confidence'] = 0.95
                candidate['detection_method'] = 'pattern'  # Ensure proper validation
                candidate['ORPHA'] = 'ORPHA:900'
                logger.info("Resolved GPA as Granulomatosis with Polyangiitis based on context")
            
            elif abbrev == 'EGPA' and has_anca_context:
                if 'eosinophilic' not in candidate['name'].lower():
                    candidate['name'] = 'Eosinophilic Granulomatosis with Polyangiitis'
                    candidate['canonical_name'] = 'Eosinophilic Granulomatosis with Polyangiitis'
                candidate['confidence'] = 0.95
                candidate['detection_method'] = 'pattern'
                candidate['ORPHA'] = 'ORPHA:183'
                logger.info("Resolved EGPA as Eosinophilic Granulomatosis with Polyangiitis based on context")
            
            resolved.append(candidate)
        
        return resolved
    
    def _filter_symptoms_and_findings(self, diseases: List) -> Tuple[List, Dict[str, int]]:
        """
        Filter out symptoms and clinical findings from disease list (NEW in v5.4.0)
        
        Args:
            diseases: List of disease candidates
            
        Returns:
            Tuple of (filtered diseases, statistics dict)
        """
        if not self.filter_symptoms:
            return diseases, {'symptoms_filtered': 0, 'findings_filtered': 0, 'total_filtered': 0}
        
        filtered = []
        stats = {'symptoms_filtered': 0, 'findings_filtered': 0, 'total_filtered': 0}
        
        for disease in diseases:
            disease_name = self._get_disease_name(disease)
            
            # Check if disease has IDs
            normalized = normalize_ids(disease)
            has_ids = any(key in normalized for key in REQUIRED_DISEASE_IDS)
            
            # Classify semantic type
            semantic_type = classify_semantic_type(disease_name, has_ids)
            
            # Add semantic type to disease
            if isinstance(disease, dict):
                disease['semantic_type'] = semantic_type
            else:
                setattr(disease, 'semantic_type', semantic_type)
            
            # Filter based on semantic type
            if semantic_type == 'symptom':
                logger.debug(f"Filtered symptom: {disease_name}")
                stats['symptoms_filtered'] += 1
                stats['total_filtered'] += 1
                continue
            elif semantic_type == 'finding':
                logger.debug(f"Filtered finding: {disease_name}")
                stats['findings_filtered'] += 1
                stats['total_filtered'] += 1
                continue
            elif semantic_type == 'generic' and not has_ids:
                logger.debug(f"Filtered generic term without IDs: {disease_name}")
                stats['total_filtered'] += 1
                continue
            
            # Keep this disease
            filtered.append(disease)
        
        logger.info(f"Filtered {stats['total_filtered']} non-disease entities "
                   f"({stats['symptoms_filtered']} symptoms, {stats['findings_filtered']} findings)")
        
        return filtered, stats
    
    def _is_duplicate(self, candidate: Dict, existing_diseases: List) -> bool:
        """
        Check if a disease candidate is a duplicate
        
        Args:
            candidate: Disease candidate to check
            existing_diseases: List of existing diseases
            
        Returns:
            True if duplicate
        """
        candidate_name = candidate.get('name', '').lower()
        
        for disease in existing_diseases:
            # Handle both dict and object formats
            if isinstance(disease, dict):
                disease_name = disease.get('name', '').lower()
            else:
                disease_name = getattr(disease, 'name', '').lower()
            
            # Exact match
            if candidate_name == disease_name:
                return True
            
            # One contains the other (for longer names)
            if len(candidate_name) > 5 and len(disease_name) > 5:
                if candidate_name in disease_name or disease_name in candidate_name:
                    # Keep the one with higher confidence
                    disease_conf = disease.get('confidence', 0) if isinstance(disease, dict) else getattr(disease, 'confidence', 0)
                    if candidate.get('confidence', 0) <= disease_conf:
                        return True
        
        return False
    
    # =========================================================================
    # Core Extraction Methods
    # =========================================================================
    
    def extract(self, text: str, filename: Optional[str] = None,
                abbreviation_context: Dict = None) -> Dict[str, Any]:
        """
        Extract diseases from text using the optimized detection pipeline
        with abbreviation enrichment and symptom filtering.
        
        Args:
            text: Input text to analyze
            filename: Optional filename for tracking
            abbreviation_context: Dictionary with abbreviation candidates
                - disease_candidates: List of disease-related abbreviations
                - all_abbreviations: All detected abbreviations
            
        Returns:
            Dictionary with detected diseases and analytics
        """
        if not text:
            return self._empty_result()
        
        try:
            # Step 1: Run standard disease detection
            result = self.detector.detect_diseases(text)
            diseases = result.diseases if hasattr(result, 'diseases') else []
            initial_count = len(diseases)
            
            # Step 2: Enrich with abbreviation candidates if provided
            abbreviation_enriched = False
            abbrev_diseases_added = 0
            
            if abbreviation_context and abbreviation_context.get('disease_candidates'):
                logger.info(f"Processing {len(abbreviation_context['disease_candidates'])} abbreviation candidates for diseases")
                
                # Process abbreviation candidates
                abbrev_diseases = self._process_abbreviation_candidates(
                    abbreviation_context['disease_candidates'],
                    text
                )
                
                # Resolve context conflicts (MPA, AAV, etc.)
                if abbreviation_context.get('all_abbreviations'):
                    abbrev_diseases = self._resolve_context_conflicts(
                        abbrev_diseases,
                        abbreviation_context['all_abbreviations']
                    )
                
                # Add non-duplicate diseases from abbreviations
                for abbrev_disease in abbrev_diseases:
                    if not self._is_duplicate(abbrev_disease, diseases):
                        # Convert dict to disease object if needed
                        disease_obj = type('Disease', (), abbrev_disease)()
                        diseases.append(disease_obj)
                        abbrev_diseases_added += 1
                        logger.debug(f"Added disease from abbreviation: {abbrev_disease['name']}")
                
                abbreviation_enriched = True
                self.stats['abbreviation_enriched_count'] += 1
                logger.info(f"Abbreviation enrichment added {abbrev_diseases_added} new disease candidates")
            
            # Step 3: Filter symptoms and findings (NEW in v5.4.0)
            diseases, filter_stats = self._filter_symptoms_and_findings(diseases)
            self.stats['symptoms_filtered'] += filter_stats['symptoms_filtered']
            self.stats['findings_filtered'] += filter_stats['findings_filtered']
            
            # Step 4: Apply Claude validation if enabled
            if self.use_claude and hasattr(self, 'detector') and hasattr(self.detector, 'validate_diseases'):
                diseases = self.detector.validate_diseases(diseases, text)
            
            # Update statistics
            self.stats['documents_processed'] += 1
            self.stats['diseases_detected'] += len(diseases)
            
            # Count diseases with IDs
            diseases_with_ids = 0
            for d in diseases:
                normalized = normalize_ids(d)
                if any(key in normalized for key in REQUIRED_DISEASE_IDS):
                    diseases_with_ids += 1
            
            self.stats['diseases_with_ids'] += diseases_with_ids
            
            # Format results
            return {
                'diseases': [self._format_disease(d) for d in diseases],
                'summary': {
                    'total': len(diseases),
                    'initial_detected': initial_count,
                    'from_abbreviations': abbrev_diseases_added,
                    'symptoms_filtered': filter_stats['symptoms_filtered'],
                    'findings_filtered': filter_stats['findings_filtered'],
                    'unique': len(set(self._get_disease_name(d) for d in diseases)),
                    'with_orpha': sum(1 for d in diseases if self._get_orphacode(d)),
                    'with_ids': diseases_with_ids,
                    'high_confidence': sum(1 for d in diseases if self._get_confidence(d) >= 0.8)
                },
                'enrichment_info': {
                    'abbreviation_enriched': abbreviation_enriched,
                    'abbreviation_candidates_processed': len(abbreviation_context.get('disease_candidates', [])) if abbreviation_context else 0,
                    'diseases_from_abbreviations': abbrev_diseases_added,
                    'claude_validation': self.use_claude,
                    'symptom_filtering_enabled': self.filter_symptoms
                },
                'mode': self.mode,
                'filename': filename
            }
        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            return self._empty_result(error=str(e))
    
    def extract_from_file(self, file_path: str, 
                         extraction_mode: str = 'intro') -> Dict[str, Any]:
        """
        Extract diseases from a document file.
        
        Args:
            file_path: Path to document (PDF, TXT, etc.)
            extraction_mode: 'intro' (first 10 pages) or 'full'
            
        Returns:
            Dictionary with detected diseases and analytics
        """
        if not self.file_extractor:
            logger.error("File extraction not available")
            return self._empty_result(error="File extraction not available")
        
        # Extract text from file
        try:
            text = self.file_extractor.get_extracted_text(file_path, extraction_mode)
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            return self._empty_result(error=str(e))
        
        # Run disease extraction
        results = self.extract(text, filename=Path(file_path).name)
        results['extraction_mode'] = extraction_mode
        results['file_path'] = str(file_path)
        
        return results
    
    def extract_batch(self, documents: Dict[str, str], 
                     parallel: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            documents: Dictionary mapping document IDs to text
            parallel: Use parallel processing if available
            
        Returns:
            Dictionary mapping document IDs to results
        """
        results = {}
        
        if parallel and hasattr(self.detector, 'detect_diseases_batch'):
            try:
                # Use optimized batch processing
                batch_diseases = self.detector.detect_diseases_batch(documents)
                
                for doc_id, diseases in batch_diseases.items():
                    results[doc_id] = {
                        'diseases': [self._format_disease(d) for d in diseases],
                        'summary': {
                            'total': len(diseases),
                            'unique': len(set(self._get_disease_name(d) for d in diseases)),
                            'with_orpha': sum(1 for d in diseases if self._get_orphacode(d))
                        }
                    }
            except Exception as e:
                logger.warning(f"Batch processing failed: {e}, falling back to sequential")
                parallel = False
        
        if not parallel:
            # Sequential processing
            for doc_id, text in documents.items():
                results[doc_id] = self.extract(text, filename=doc_id)
        
        return results
    
    # =========================================================================
    # Configuration Methods
    # =========================================================================
    
    def set_mode(self, mode: str):
        """
        Change detection mode dynamically.
        
        Args:
            mode: "precision", "balanced", or "recall"
        """
        if mode not in ["precision", "balanced", "recall"]:
            raise ValueError(f"Invalid mode: {mode}")
        
        self.mode = mode
        if hasattr(self.detector, 'set_mode'):
            self.detector.set_mode(mode)
        logger.info(f"Detection mode changed to: {mode}")
    
    def set_confidence_threshold(self, threshold: float):
        """
        Update confidence threshold.
        
        Args:
            threshold: Value between 0 and 1
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        
        if hasattr(self.detector, 'set_confidence_threshold'):
            self.detector.set_confidence_threshold(threshold)
        logger.info(f"Confidence threshold set to: {threshold}")
    
    def set_symptom_filtering(self, enabled: bool):
        """
        Enable or disable symptom/finding filtering (NEW in v5.4.0)
        
        Args:
            enabled: True to filter symptoms, False to keep all
        """
        self.filter_symptoms = enabled
        logger.info(f"Symptom filtering {'enabled' if enabled else 'disabled'}")
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics from the detector"""
        stats = self.stats.copy()
        
        if stats['documents_processed'] > 0:
            stats['avg_diseases_per_doc'] = stats['diseases_detected'] / stats['documents_processed']
            stats['abbreviation_enrichment_rate'] = stats['abbreviation_enriched_count'] / stats['documents_processed']
            stats['diseases_with_ids_rate'] = stats['diseases_with_ids'] / stats['diseases_detected'] if stats['diseases_detected'] > 0 else 0
            stats['symptom_filter_rate'] = (stats['symptoms_filtered'] + stats['findings_filtered']) / (stats['diseases_detected'] + stats['symptoms_filtered'] + stats['findings_filtered']) if (stats['diseases_detected'] + stats['symptoms_filtered'] + stats['findings_filtered']) > 0 else 0
        
        if hasattr(self.detector, 'get_statistics'):
            stats['detector_stats'] = self.detector.get_statistics()
        
        return stats
    
    def clear_cache(self):
        """Clear all caches"""
        if hasattr(self.detector, 'clear_cache'):
            self.detector.clear_cache()
        logger.info("Cache cleared")
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save results to JSON file.
        
        Args:
            results: Extraction results
            output_path: Path for output file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def _format_disease(self, disease) -> Dict[str, Any]:
        """
        Format a disease object for output with normalized IDs
        
        Args:
            disease: Disease entity (dict or object)
            
        Returns:
            Formatted dictionary with IDs included
        """
        # Start with basic fields
        if isinstance(disease, dict):
            formatted = dict(disease)
        else:
            formatted = {
                'name': getattr(disease, 'name', str(disease)),
                'confidence': round(getattr(disease, 'confidence', 0.5), 3),
                'occurrences': getattr(disease, 'occurrences', 1),
                'detection_method': getattr(disease, 'detection_method', 'unknown'),
                'positions': getattr(disease, 'positions', []),
                'matched_terms': getattr(disease, 'matched_terms', []),
                'from_abbreviation': getattr(disease, 'from_abbreviation', None),
                'source_abbreviation': getattr(disease, 'source_abbreviation', None),
                'semantic_type': getattr(disease, 'semantic_type', None),
            }
            
            # Add any existing ID fields from the object
            for attr in dir(disease):
                if not attr.startswith('_'):
                    canonical_key = ID_KEY_MAP.get(attr.lower())
                    if canonical_key:
                        value = getattr(disease, attr, None)
                        if value:
                            formatted[canonical_key] = value
        
        # Enrich with standard IDs if available (NEW in v5.4.0)
        formatted = enrich_disease_ids(formatted)
        
        # Normalize all IDs
        normalized_ids = normalize_ids(formatted)
        
        # Get primary ID
        primary_key, primary_value = get_primary_id(normalized_ids)
        
        # Add ID information to output
        if primary_key and primary_value:
            formatted['primary_id'] = f"{primary_key}:{primary_value}"
        else:
            formatted['primary_id'] = None
        
        # Add all normalized IDs
        formatted['all_ids'] = normalized_ids if normalized_ids else {}
        
        # Add canonical name if available
        if 'canonical_name' not in formatted:
            formatted['canonical_name'] = formatted.get('name')
        
        # Clean up redundant ID fields (keep only normalized ones)
        keys_to_remove = []
        for key in formatted.keys():
            if key.lower() in ID_KEY_MAP and key != ID_KEY_MAP.get(key.lower()):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            formatted.pop(key, None)
        
        # Ensure consistent field order
        ordered = {
            'name': formatted.get('name'),
            'canonical_name': formatted.get('canonical_name'),
            'primary_id': formatted.get('primary_id'),
            'all_ids': formatted.get('all_ids', {}),
            'confidence': formatted.get('confidence', 0.5),
            'occurrences': formatted.get('occurrences', 1),
            'detection_method': formatted.get('detection_method', 'unknown'),
        }
        
        # Add optional fields if present
        optional_fields = [
            'positions', 'matched_terms', 'from_abbreviation', 
            'source_abbreviation', 'context_type', 'semantic_type'
        ]
        for field in optional_fields:
            if field in formatted and formatted[field]:
                ordered[field] = formatted[field]
        
        return ordered
    
    def _get_disease_name(self, disease) -> str:
        """Get disease name from dict or object"""
        if isinstance(disease, dict):
            return disease.get('name', '')
        return getattr(disease, 'name', '')
    
    def _get_confidence(self, disease) -> float:
        """Get confidence from dict or object"""
        if isinstance(disease, dict):
            return disease.get('confidence', 0)
        return getattr(disease, 'confidence', 0)
    
    def _get_orphacode(self, disease) -> Optional[str]:
        """Get orphacode from dict or object (checks normalized IDs too)"""
        if isinstance(disease, dict):
            # Check direct field first
            if disease.get('orphacode'):
                return disease.get('orphacode')
            # Check normalized IDs
            normalized = normalize_ids(disease)
            return normalized.get('ORPHA')
        else:
            # Check object attributes
            orphacode = getattr(disease, 'orphacode', None)
            if orphacode:
                return orphacode
            # Check normalized IDs
            normalized = normalize_ids(disease)
            return normalized.get('ORPHA')
    
    def _empty_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return empty result structure"""
        result = {
            'diseases': [],
            'summary': {
                'total': 0,
                'unique': 0,
                'with_orpha': 0,
                'with_ids': 0,
                'high_confidence': 0,
                'symptoms_filtered': 0,
                'findings_filtered': 0
            },
            'enrichment_info': {
                'abbreviation_enriched': False,
                'abbreviation_candidates_processed': 0,
                'diseases_from_abbreviations': 0,
                'claude_validation': self.use_claude,
                'symptom_filtering_enabled': self.filter_symptoms
            },
            'mode': self.mode
        }
        if error:
            result['error'] = error
        return result
    
    def _get_claude_client(self):
        """Get Claude client from system initializer."""
        if hasattr(self.system_initializer, 'get_claude_client'):
            return self.system_initializer.get_claude_client()
        else:
            # Fallback to creating directly
            api_key = os.getenv('CLAUDE_API_KEY')
            if api_key:
                from anthropic import Anthropic
                return Anthropic(api_key=api_key)
        return None

# ============================================================================
# Convenience Functions
# ============================================================================

def extract_diseases(text: str, mode: str = "balanced", 
                    use_claude: bool = False,
                    abbreviation_context: Dict = None,
                    filter_symptoms: bool = True) -> Dict[str, Any]:
    """
    Quick function to extract diseases from text.
    
    Args:
        text: Input text
        mode: Detection mode
        use_claude: Enable Claude validation
        abbreviation_context: Abbreviation context dictionary
        filter_symptoms: Filter out symptoms and findings
        
    Returns:
        Extraction results
    """
    extractor = DiseaseMetadataExtractor(
        mode=mode, 
        use_claude=use_claude,
        filter_symptoms=filter_symptoms
    )
    return extractor.extract(text, abbreviation_context=abbreviation_context)

def extract_diseases_from_file(file_path: str, mode: str = "balanced",
                              extraction: str = "intro",
                              filter_symptoms: bool = True) -> Dict[str, Any]:
    """
    Quick function to extract diseases from a file.
    
    Args:
        file_path: Path to file
        mode: Detection mode
        extraction: "intro" or "full"
        filter_symptoms: Filter out symptoms and findings
        
    Returns:
        Extraction results
    """
    extractor = DiseaseMetadataExtractor(mode=mode, filter_symptoms=filter_symptoms)
    return extractor.extract_from_file(file_path, extraction)