#!/usr/bin/env python3
"""
rare_disease_disease_detector.py - Enhanced with Truncation Handling
==================================================================================

location: corpus_metadata/document_utils/rare_disease_disease_detector.py

Complete Rare Disease Detection System with:
- Database-backed ID Resolution
- False Positive Filtering
- Synonym Lookup
- Truncated Name Completion (NEW)

Version: 4.3 - Added truncated disease name completion
Last Updated: 2025-10-08
"""

import json
import logging
import time
import re
import sqlite3
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Optional imports with proper checks
SPACY_AVAILABLE = False
INITIALIZER_AVAILABLE = False

# Check for SpaCy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    logger.warning("SpaCy not installed. NER detection will be disabled.")

# Import system initializer for lexicons
try:
    from corpus_metadata.document_utils.metadata_system_initializer import MetadataSystemInitializer
    INITIALIZER_AVAILABLE = True
except ImportError:
    try:
        from .metadata_system_initializer import MetadataSystemInitializer
        INITIALIZER_AVAILABLE = True
    except ImportError:
        INITIALIZER_AVAILABLE = False

# ============================================================================
# Constants
# ============================================================================

# Generic terms to exclude from disease detection
GENERIC_TERMS_TO_EXCLUDE = {
    'disease', 'disorder', 'syndrome', 'condition', 
    'complete', 'sensitive', 'genetic', 'infection',
    'diagnosis', 'treatment', 'symptom', 'patient',
    'therapy', 'medication', 'drug', 'dose',
    'study', 'trial', 'test', 'result',
    'effect', 'response', 'outcome', 'prognosis',
    'assessment', 'evaluation', 'examination', 'screening'
}

# False positive filters
FALSE_POSITIVE_SYMPTOMS = {
    'fever', 'pain', 'toxicity', 'necrosis', 'fatigue',
    'nausea', 'vomiting', 'dizziness', 'headache',
    'myalgias', 'myalgia', 'arthralgia', 'malaise'
}

FALSE_POSITIVE_GENERIC = {
    'disease', 'disorder', 'condition', 'syndrome',
    'aggressive disease', 'chronic disease', 'acute disease',
    'renal disease', 'bone disease', 'liver disease',
    'lung disease', 'heart disease', 'brain disease',
    'genetic disease', 'infectious disease'
}

FALSE_POSITIVE_FRAGMENTS = {
    'anca-associated', 'childhood-onset', 'adult-onset',
    'early-onset', 'late-onset', 'pediatric', 'juvenile'
}

# [OK] NEW: Truncation patterns for common disease name fragments
TRUNCATION_INDICATORS = {
    # Disease type suffixes that suggest continuation
    'with', 'and', 'or', 'type', 'associated',
    # Connector words
    'due', 'to', 'in', 'of', 'from',
    # Incomplete medical terms
    'auto', 'neuro', 'cardio', 'gastro', 'pulmo',
    'hepato', 'reno', 'dermato', 'endo', 'hema'
}

# [OK] NEW: Minimum confidence threshold for truncation completion
TRUNCATION_COMPLETION_MIN_CONFIDENCE = 0.75

# SEMANTIC TYPE CODES FOR UMLS DISORDERS GROUP
DISORDERS_STYS = {
    'T019',  # Congenital Abnormality
    'T020',  # Acquired Abnormality  
    'T037',  # Injury or Poisoning
    'T046',  # Pathologic Function
    'T047',  # Disease or Syndrome
    'T048',  # Mental or Behavioral Dysfunction
    'T049',  # Cell or Molecular Dysfunction
    'T050',  # Experimental Model of Disease
    'T190',  # Anatomical Abnormality
    'T191'   # Neoplastic Process
}

# LEXICAL INDICATORS FOR DISEASE DETECTION
DISEASE_NEG_WORDS = {
    'protein', 'gene', 'component', 'receptor', 'assay', 
    'trial', 'guideline', 'organization', 'test', 'antibody',
    'enzyme', 'cytokine', 'complement', 'biomarker', 'marker',
    'institute', 'foundation', 'society', 'college', 'alliance'
}

DISEASE_POS_WORDS = {
    'disease', 'syndrome', 'vasculitis', 'polyangiitis', 
    'arthritis', 'nephritis', 'pneumonia', 'carcinoma', 
    'lymphoma', 'leukemia', 'infection', 'failure', 
    'deficiency', 'disorder', 'condition', 'granulomatosis',
    'sarcoidosis', 'fibrosis', 'cirrhosis', 'neoplasm'
}

DISEASE_POS_SUFFIXES = {
    'itis', 'oma', 'emia', 'opathy', 'osis', 'iasis',
    'pathy', 'trophy', 'plasia', 'sclerosis', 'stenosis'
}

# Detection confidence thresholds by mode
CONFIDENCE_THRESHOLDS = {
    'precision': 0.8,
    'balanced': 0.6,
    'recall': 0.4
}

# Default confidence scores for different sources
SOURCE_CONFIDENCE = {
    'pattern': 0.95,
    'lexicon': 0.85,
    'ner': 0.75,
    'abbreviation_enrichment': 0.70,
    'truncation_completion': 0.80  # [OK] NEW
}

# Negation confidence reduction factor
NEGATION_CONFIDENCE_FACTOR = 0.2

# Context window sizes
CONTEXT_WINDOW = 50
NEGATION_WINDOW_BEFORE = 150
NEGATION_WINDOW_AFTER = 50

# NER entity labels to detect
DISEASE_LABELS = ['DISEASE', 'DISORDER', 'CONDITION']

# Common disease suffixes for normalization
DISEASE_SUFFIXES = ['disease', 'disorder', 'syndrome', 'condition']

# Compiled regex patterns for negation analysis
SENT_BOUNDARY = re.compile(r'[.!?;:\n]')

CONTRAST = re.compile(
    r'\b(?:however|but|although|though|despite|nevertheless|nonetheless|'
    r'yet|still|except|besides|instead|rather|conversely|otherwise)\b', 
    re.I
)

POSITIVE = re.compile(
    r'\b(?:suspected|suspect|suspicion\s+(?:of|for)|diagnosed|diagnosis\s+of|'
    r'confirmed|confirming|confirmation\s+of|positive\s+for|tested\s+positive|'
    r'consistent\s+with|compatible\s+with|suggestive\s+of|suggests?|'
    r'indicative\s+of|indicates?|concerning\s+for|concern\s+for|'
    r'features?\s+of|signs?\s+of|symptoms?\s+of|presenting\s+with|presents?\s+with|'
    r'differential\s+diagnosis|ddx|(?<!un)(?<!not\s)likely|probable|possible)\b',
    re.I
)

CANNOT_EXCLUDE = re.compile(
    r'\b(?:cannot|can\'t|unable\s+to|not\s+able\s+to|difficult\s+to)\s+(?:exclude|rule\s+out)\b|'
    r'\b(?:cannot|can\'t)\s+be\s+(?:excluded|ruled\s+out)\b|'
    r'\bquestion\s+of\b|\bquery\b',
    re.I
)

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DetectedDisease:
    """Represents a detected disease entity"""
    name: str
    confidence: float
    positions: List[Tuple[int, int]] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    occurrences: int = 1
    source: str = "pattern"
    matched_terms: List[str] = field(default_factory=list)
    orphacode: Optional[str] = None
    mondo_id: Optional[str] = None
    mesh_id: Optional[str] = None
    normalized_name: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    detection_method: str = "pattern"
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    is_negated: bool = False
    validation_status: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    from_abbreviation: Optional[str] = None
    semantic_tui: Optional[str] = None
    # Database ID fields
    primary_id: Optional[str] = None
    all_ids: Dict[str, str] = field(default_factory=dict)
    canonical_name: Optional[str] = None
    identifiers: Dict[str, str] = field(default_factory=dict)
    # [OK] NEW: Truncation handling
    is_truncated: bool = False
    original_fragment: Optional[str] = None
    completion_confidence: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DiseaseDetectionResult:
    """Complete disease detection result"""
    diseases: List[DetectedDisease]
    detection_time: float
    methods_used: List[str]
    pattern_hits: int = 0
    ner_hits: int = 0
    lexicon_hits: int = 0
    validated_count: int = 0
    filtered_count: int = 0
    mode: str = "balanced"
    analytics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        result = {
            'diseases': [d.to_dict() for d in self.diseases],
            'detection_time': self.detection_time,
            'methods_used': self.methods_used,
            'mode': self.mode,
            'analytics': self.analytics
        }
        for d in self.diseases:
            if 'pattern' in d.source:
                result['pattern_hits'] = result.get('pattern_hits', 0) + 1
            if 'ner' in d.source:
                result['ner_hits'] = result.get('ner_hits', 0) + 1
            if 'lexicon' in d.source:
                result['lexicon_hits'] = result.get('lexicon_hits', 0) + 1
        return result

# ============================================================================
# Pattern Definitions
# ============================================================================

class DiseasePatterns:
    """Disease detection patterns - centralized definition"""
    
    PATTERNS = [
        # Genetic diseases
        (r'\b(?:Huntington\'?s?|HD)\s*(?:disease|chorea)?\b', 'Huntington disease'),
        (r'\b(?:cystic\s+fibrosis|CF)\b', 'Cystic fibrosis'),
        (r'\b(?:sickle\s+cell|SCD)\s*(?:disease|anemia)?\b', 'Sickle cell disease'),
        
        # Neurological
        (r'\b(?:ALS|amyotrophic\s+lateral\s+sclerosis)\b', 'Amyotrophic lateral sclerosis'),
        (r'\b(?:multiple\s+sclerosis|MS)\b', 'Multiple sclerosis'),
        (r'\b(?:myasthenia\s+gravis|MG)\b', 'Myasthenia gravis'),
        
        # Autoimmune/Vasculitis
        (r'\b(?:ANCA[\s-]associated\s+vasculitis|AAV)\b', 'ANCA-associated vasculitis'),
        (r'\b(?:GPA|granulomatosis\s+with\s+polyangiitis)\b', 'Granulomatosis with polyangiitis'),
        (r'\b(?:MPA|microscopic\s+polyangiitis)\b', 'Microscopic polyangiitis'),
        (r'\b(?:EGPA|eosinophilic\s+granulomatosis)\b', 'Eosinophilic granulomatosis with polyangiitis'),
    ]
    
    NEGATION_BEFORE = [
        r'\bno\s+(?:evidence|signs?|symptoms?|indication|history|hx)\s+(?:of|for)\b',
        r'\bwithout\s+(?:any\s+)?(?:evidence|signs?|symptoms?)\s+of\b',
        r'\bnot\s+(?:consistent|compatible|suggestive|indicative)\s+(?:with|of)\b',
    ]
    
    NEGATION_AFTER = [
        r'^\s*(?:was|were|has been|have been)?\s*ruled?\s*out',
        r'^\s*(?:is|was|has been|have been)?\s*excluded',
    ]

# ============================================================================
# Helper Functions
# ============================================================================

def _blocked_by_boundary_or_contrast(text_between: str) -> bool:
    """Check if negation is blocked by sentence boundary or contrast word"""
    if SENT_BOUNDARY.search(text_between):
        return True
    if CONTRAST.search(text_between):
        if not re.search(r'\b(?:no|not|without|denies|ruled\s+out|exclude[sd]?)\b', text_between, re.I):
            return True
    return False

def normalize_ids(ids: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize ID keys to canonical form"""
    if not ids:
        return {}
    
    normalized = {}
    id_mappings = {
        'cui': 'UMLS', 'umls': 'UMLS', 'umls_cui': 'UMLS',
        'orpha': 'ORPHA', 'orphanet': 'ORPHA', 'orpha_code': 'ORPHA', 'orphacode': 'ORPHA',
        'doid': 'DOID', 'disease_ontology': 'DOID', 'do_id': 'DOID',
        'mondo': 'MONDO', 'mondo_id': 'MONDO',
        'mesh': 'MESH', 'mesh_id': 'MESH',
        'snomed': 'SNOMED', 'snomed_ct': 'SNOMED', 'snomedct': 'SNOMED',
        'icd10': 'ICD10', 'icd_10': 'ICD10',
        'icd9': 'ICD9', 'icd_9': 'ICD9',
        'omim': 'OMIM', 'omim_id': 'OMIM'
    }
    
    for key, value in ids.items():
        key_lower = key.lower()
        canonical_key = id_mappings.get(key_lower, key.upper())
        if value:
            normalized[canonical_key] = value
    
    return normalized

def _is_likely_disease_by_lexical(name: str) -> bool:
    """Lexical check for disease likelihood"""
    name_lower = name.lower()
    
    non_disease_terms = {
        'protein', 'gene', 'receptor', 'enzyme', 'antibody', 
        'antigen', 'assay', 'test', 'marker', 'biomarker',
        'pathway', 'system', 'component', 'factor', 'ligand'
    }
    
    for term in non_disease_terms:
        if term in name_lower:
            return False
    
    disease_indicators = {
        'disease', 'disorder', 'syndrome', 'cancer', 'carcinoma',
        'lymphoma', 'leukemia', 'itis', 'osis', 'emia', 'pathy'
    }
    
    for indicator in disease_indicators:
        if indicator in name_lower:
            return True
    
    return True

def postfilter_direct_diseases(diseases: List[DetectedDisease], 
                              kb_resolver: Optional[Any] = None) -> Tuple[List[DetectedDisease], List[Dict]]:
    """Post-filter disease candidates using semantic validation"""
    if not kb_resolver or not hasattr(kb_resolver, 'resolve'):
        return diseases, []
    
    filtered = []
    demoted = []
    
    for disease in diseases:
        if disease.source == 'pattern' and disease.confidence > 0.9:
            filtered.append(disease)
            continue
            
        if disease.from_abbreviation or 'abbreviation' in disease.source:
            try:
                resolution = kb_resolver.resolve(disease.name)
                if resolution:
                    candidates = resolution if isinstance(resolution, list) else [resolution]
                    is_valid_disease = False
                    for candidate in candidates:
                        semantic_type = candidate.get('semantic_type', '')
                        if semantic_type in DISORDERS_STYS:
                            is_valid_disease = True
                            if candidate.get('ids'):
                                ids = normalize_ids(candidate['ids'])
                                if 'UMLS' in ids:
                                    disease.metadata['umls_cui'] = ids['UMLS']
                                if 'ORPHA' in ids:
                                    disease.orphacode = ids['ORPHA']
                            disease.semantic_tui = semantic_type
                            break
                    if is_valid_disease:
                        filtered.append(disease)
                    else:
                        demoted.append({
                            'name': disease.name,
                            'reason': 'non_disease_semantic_type',
                            'semantic_type': semantic_type,
                            'from_abbreviation': disease.from_abbreviation
                        })
                else:
                    if _is_likely_disease_by_lexical(disease.name):
                        filtered.append(disease)
                    else:
                        demoted.append({
                            'name': disease.name,
                            'reason': 'failed_lexical_check',
                            'from_abbreviation': disease.from_abbreviation
                        })
            except Exception as e:
                logger.debug(f"KB validation failed for {disease.name}: {e}")
                filtered.append(disease)
        else:
            filtered.append(disease)
    
    if demoted:
        logger.info(f"Semantic filtering: {len(diseases)} -> {len(filtered)} diseases ({len(demoted)} demoted)")
    
    return filtered, demoted

# ============================================================================
# Main Rare Disease Detector Class
# ============================================================================

class RareDiseaseDetector:
    """Complete rare disease detection system with truncation handling"""
    
    def __init__(self, 
                 mode: str = "balanced",
                 use_patterns: bool = True,
                 use_ner: bool = True,
                 use_lexicon: bool = True,
                 use_validation: bool = False,
                 use_truncation_completion: bool = True,  # [OK] NEW
                 orphanet_db_path: Optional[str] = None,
                 doid_db_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 system_initializer: Optional[Any] = None,
                 kb_resolver: Optional[Any] = None,
                 **kwargs):
        """
        Initialize the rare disease detector
        
        Args:
            mode: Detection mode - "precision", "balanced", or "recall"
            use_patterns: Enable pattern-based detection
            use_ner: Enable NER-based detection
            use_lexicon: Enable lexicon-based detection
            use_validation: Enable semantic validation
            use_truncation_completion: Enable truncated name completion (NEW)
            orphanet_db_path: Path to Orphanet database
            doid_db_path: Path to DOID database
            config_path: Path to config file
            system_initializer: System initializer instance
            kb_resolver: Knowledge base resolver
        """
        self.mode = mode
        self.use_patterns = use_patterns
        self.use_ner = use_ner and SPACY_AVAILABLE
        self.use_lexicon = use_lexicon
        self.use_validation = use_validation
        self.use_truncation_completion = use_truncation_completion  # [OK] NEW
        
        # Store KB resolver for semantic filtering
        self.kb_resolver = kb_resolver or system_initializer
        
        # Set confidence threshold based on mode
        self.confidence_threshold = CONFIDENCE_THRESHOLDS.get(mode, 0.6)
        
        # Compile patterns once
        self.compiled_patterns = self._compile_patterns()
        self.negation_patterns_before = [re.compile(p, re.IGNORECASE) for p in DiseasePatterns.NEGATION_BEFORE]
        self.negation_patterns_after = [re.compile(p, re.IGNORECASE) for p in DiseasePatterns.NEGATION_AFTER]
        
        # Initialize data stores
        self.disease_lexicon = {}
        self.orphanet_data = {}
        self.mondo_data = {}
        self.nlp_models = {}
        
        # Initialize semantic type constants
        self.DISORDERS_STYS = DISORDERS_STYS
        
        # Legacy disease name mappings
        self.LEGACY_DISEASE_NAMES = {
            'wegener granulomatosis': 'granulomatosis with polyangiitis',
            "wegener's granulomatosis": 'granulomatosis with polyangiitis',
            'wegener syndrome': 'granulomatosis with polyangiitis',
            'wegener disease': 'granulomatosis with polyangiitis',
            'churg-strauss syndrome': 'eosinophilic granulomatosis with polyangiitis',
            'churg strauss syndrome': 'eosinophilic granulomatosis with polyangiitis',
            'churg-strauss': 'eosinophilic granulomatosis with polyangiitis',
            'css': 'eosinophilic granulomatosis with polyangiitis',
            'lou gehrig disease': 'amyotrophic lateral sclerosis',
            "lou gehrig's disease": 'amyotrophic lateral sclerosis',
        }
        
        # Database Path Initialization
        self.orphanet_db_path = orphanet_db_path
        self.doid_db_path = doid_db_path
        
        # Try to get from system initializer
        if not self.orphanet_db_path and system_initializer:
            if hasattr(system_initializer, 'get_resource'):
                # Try multiple possible resource names
                for key in ['disease_orphanet_db', 'disease_orphanet', 'orphanet_db']:
                    path = system_initializer.get_resource(key)
                    if path:
                        self.orphanet_db_path = path
                        break
            if not self.orphanet_db_path and hasattr(system_initializer, 'config'):
                # Check databases section first (current config structure)
                databases = system_initializer.config.get('databases', {})
                resource = databases.get('disease_orphanet')
                if resource:
                    self.orphanet_db_path = resource if isinstance(resource, str) else resource.get('path')
                # Fall back to resources section (legacy)
                if not self.orphanet_db_path:
                    resources = system_initializer.config.get('resources', {})
                    resource = resources.get('disease_orphanet_db') or resources.get('disease_orphanet')
                    if isinstance(resource, dict):
                        self.orphanet_db_path = resource.get('path')
                    elif isinstance(resource, str):
                        self.orphanet_db_path = resource
        
        if not self.doid_db_path and system_initializer:
            if hasattr(system_initializer, 'get_resource'):
                # Try multiple possible resource names
                for key in ['disease_ontology_db', 'disease_ontology', 'doid_db']:
                    path = system_initializer.get_resource(key)
                    if path:
                        self.doid_db_path = path
                        break
            if not self.doid_db_path and hasattr(system_initializer, 'config'):
                # Check databases section first (current config structure)
                databases = system_initializer.config.get('databases', {})
                resource = databases.get('disease_ontology')
                if resource:
                    self.doid_db_path = resource if isinstance(resource, str) else resource.get('path')
                # Fall back to resources section (legacy)
                if not self.doid_db_path:
                    resources = system_initializer.config.get('resources', {})
                    resource = resources.get('disease_ontology_db') or resources.get('disease_ontology')
                    if isinstance(resource, dict):
                        self.doid_db_path = resource.get('path')
                    elif isinstance(resource, str):
                        self.doid_db_path = resource
        
        # Try config file directly
        if (not self.orphanet_db_path or not self.doid_db_path) and config_path:
            try:
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                
                # Check databases section first
                databases = config.get('databases', {})
                if not self.orphanet_db_path:
                    resource = databases.get('disease_orphanet')
                    if isinstance(resource, dict):
                        self.orphanet_db_path = resource.get('path')
                    elif isinstance(resource, str):
                        self.orphanet_db_path = resource
                
                if not self.doid_db_path:
                    resource = databases.get('disease_ontology')
                    if isinstance(resource, dict):
                        self.doid_db_path = resource.get('path')
                    elif isinstance(resource, str):
                        self.doid_db_path = resource
                
                # Fall back to resources section
                if not self.orphanet_db_path or not self.doid_db_path:
                    resources = config.get('resources', {})
                    if not self.orphanet_db_path:
                        resource = resources.get('disease_orphanet_db') or resources.get('disease_orphanet')
                        if isinstance(resource, dict):
                            self.orphanet_db_path = resource.get('path')
                        elif isinstance(resource, str):
                            self.orphanet_db_path = resource
                    
                    if not self.doid_db_path:
                        resource = resources.get('disease_ontology_db') or resources.get('disease_ontology')
                        if isinstance(resource, dict):
                            self.doid_db_path = resource.get('path')
                        elif isinstance(resource, str):
                            self.doid_db_path = resource
            except Exception as e:
                logger.debug(f"Could not load config file: {e}")
        
        # Resolve relative paths using paths.databases from config
        if self.orphanet_db_path or self.doid_db_path:
            db_base_path = None
            
            # Try to get database base path from system_initializer
            if system_initializer and hasattr(system_initializer, 'config'):
                paths_config = system_initializer.config.get('paths', {})
                db_base_path = paths_config.get('databases')
                
                # Also check for base_path
                if db_base_path and not Path(db_base_path).is_absolute():
                    base = system_initializer.config.get('base_path') or paths_config.get('base')
                    if base:
                        db_base_path = str(Path(base) / db_base_path)
            
            # Try to get from config file if not found
            if not db_base_path and config_path:
                try:
                    import yaml
                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                    db_base_path = config.get('paths', {}).get('databases')
                except:
                    pass
            
            # Resolve paths if we have a base path and files are not absolute
            if db_base_path:
                if self.orphanet_db_path and not Path(self.orphanet_db_path).is_absolute():
                    resolved = Path(db_base_path) / self.orphanet_db_path
                    if resolved.exists():
                        self.orphanet_db_path = str(resolved)
                    else:
                        # Try common parent directories
                        for parent in ['.', '..', '../..']:
                            test_path = Path(parent) / db_base_path / self.orphanet_db_path
                            if test_path.exists():
                                self.orphanet_db_path = str(test_path.resolve())
                                break
                
                if self.doid_db_path and not Path(self.doid_db_path).is_absolute():
                    resolved = Path(db_base_path) / self.doid_db_path
                    if resolved.exists():
                        self.doid_db_path = str(resolved)
                    else:
                        # Try common parent directories
                        for parent in ['.', '..', '../..']:
                            test_path = Path(parent) / db_base_path / self.doid_db_path
                            if test_path.exists():
                                self.doid_db_path = str(test_path.resolve())
                                break
        
        # Ensure strings not dicts
        if isinstance(self.orphanet_db_path, dict):
            self.orphanet_db_path = self.orphanet_db_path.get('path')
        if isinstance(self.doid_db_path, dict):
            self.doid_db_path = self.doid_db_path.get('path')
        
        # Initialize components
        if self.use_lexicon:
            self._initialize_lexicon(system_initializer)
        
        if self.use_ner:
            self._initialize_ner_models()
        
        logger.info(f"RareDiseaseDetector initialized in {mode} mode")
        
        # Verify databases
        logger.info("="*60)
        logger.info("DISEASE DETECTOR INITIALIZATION")
        logger.info("="*60)
        logger.info(f"Orphanet DB: {self.orphanet_db_path}")
        logger.info(f"  Exists: {Path(self.orphanet_db_path).exists() if self.orphanet_db_path else False}")
        logger.info(f"DOID DB: {self.doid_db_path}")
        logger.info(f"  Exists: {Path(self.doid_db_path).exists() if self.doid_db_path else False}")
        logger.info(f"Truncation Completion: {'enabled' if self.use_truncation_completion else 'disabled'}")
        
        if self.orphanet_db_path and Path(self.orphanet_db_path).exists():
            try:
                conn = sqlite3.connect(self.orphanet_db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM core_entities WHERE status='active'")
                count = cursor.fetchone()[0]
                logger.info(f"  [OK] Orphanet: {count:,} active entities")
                conn.close()
            except Exception as e:
                logger.error(f"  [X] Orphanet failed: {e}")
                self.orphanet_db_path = None
        
        if self.doid_db_path and Path(self.doid_db_path).exists():
            try:
                conn = sqlite3.connect(self.doid_db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                if 'diseases' in tables:
                    cursor.execute("SELECT COUNT(*) FROM diseases")
                    count = cursor.fetchone()[0]
                    logger.info(f"  [OK] DOID: {count:,} diseases")
                conn.close()
            except Exception as e:
                logger.error(f"  [X] DOID failed: {e}")
                self.doid_db_path = None
        
        logger.info("="*60)
    
    def _compile_patterns(self) -> List[Tuple]:
        """Compile regex patterns for efficiency"""
        compiled = []
        for pattern_str, name in DiseasePatterns.PATTERNS:
            try:
                compiled.append((re.compile(pattern_str, re.IGNORECASE), name))
            except re.error as e:
                logger.error(f"Failed to compile pattern {pattern_str}: {e}")
        return compiled
    
    def _initialize_lexicon(self, system_initializer):
        """Initialize disease lexicon from system initializer"""
        try:
            initializer = system_initializer or MetadataSystemInitializer.get_instance()
            
            for attr in ['disease_lexicon', 'diseases', 'disease_data']:
                if hasattr(initializer, attr):
                    data = getattr(initializer, attr)
                    if data and isinstance(data, dict):
                        self.disease_lexicon = data
                        logger.info(f"Loaded disease lexicon: {len(self.disease_lexicon)} terms")
                        break
            
            for attr in ['orphanet_diseases', 'orphanet_data', 'orphanet']:
                if hasattr(initializer, attr):
                    data = getattr(initializer, attr)
                    if data and isinstance(data, dict):
                        self.orphanet_data = data
                        logger.info(f"Loaded Orphanet data: {len(self.orphanet_data)} diseases")
                        break
            
            if hasattr(initializer, 'resources') and isinstance(initializer.resources, dict):
                if 'orphanet' in initializer.resources:
                    self.orphanet_data = initializer.resources['orphanet']
                    logger.info(f"Loaded Orphanet from resources: {len(self.orphanet_data)} diseases")
        except Exception as e:
            logger.error(f"Failed to initialize lexicon: {e}")
            self.use_lexicon = False
    
    def _initialize_ner_models(self):
        """Initialize SpaCy NER models"""
        models_to_load = [
            ('en_ner_bc5cdr_md', 'bc5cdr'),
            ('en_core_sci_lg', 'scispacy')
        ]
        
        for model_name, key in models_to_load:
            try:
                if spacy.util.is_package(model_name):
                    self.nlp_models[key] = spacy.load(
                        model_name, 
                        disable=["tagger", "parser", "lemmatizer"]
                    )
                    logger.info(f"Loaded {key} NER model")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
        
        if not self.nlp_models:
            self.use_ner = False
    
    def _is_false_positive_disease(self, name: str) -> bool:
        """Filter out non-diseases before database lookup"""
        if not name or len(name.strip()) < 2:
            return True
        
        name_lower = name.lower().strip()
        
        if name_lower.startswith('http') or '/' in name_lower or '.' in name_lower and len(name_lower.split('.')) > 2:
            logger.debug(f"Filtered URL/fragment: {name}")
            return True
        
        if name_lower in FALSE_POSITIVE_SYMPTOMS:
            logger.debug(f"Filtered symptom: {name}")
            return True
        
        if name_lower in FALSE_POSITIVE_GENERIC:
            logger.debug(f"Filtered generic term: {name}")
            return True
        
        if name_lower in FALSE_POSITIVE_FRAGMENTS:
            logger.debug(f"Filtered fragment: {name}")
            return True
        
        non_disease_patterns = [
            r'^for-',
            r'-associated$',
            r'^\d+$',
            r'^[A-Z\d\-]+$' if len(name) < 4 else None,
        ]
        
        for pattern in non_disease_patterns:
            if pattern and re.match(pattern, name_lower):
                logger.debug(f"Filtered by pattern '{pattern}': {name}")
                return True
        
        return False
    
    # [OK] NEW: Truncation detection and completion methods
    def _is_likely_truncated(self, name: str, text: str, position: Tuple[int, int]) -> bool:
        """
        Detect if a disease name is likely truncated
        
        Conservative approach to avoid overfitting:
        1. Check if name ends with truncation indicators
        2. Verify context suggests continuation
        3. Confirm it's not just a valid partial term
        """
        name_lower = name.lower().strip()
        
        # Check for truncation indicators at end
        ends_with_indicator = False
        for indicator in TRUNCATION_INDICATORS:
            if name_lower.endswith(indicator):
                ends_with_indicator = True
                break
        
        if not ends_with_indicator:
            return False
        
        # Get surrounding context
        start, end = position
        context_after = text[end:end+100] if end < len(text) else ""
        
        # Look for continuation patterns
        continuation_patterns = [
            r'^\s*[a-z]',  # Continues with lowercase
            r'^\s*\(',     # Continues with parenthetical
            r'^\s*-',      # Continues with dash
        ]
        
        for pattern in continuation_patterns:
            if re.match(pattern, context_after):
                logger.debug(f"Detected truncation in '{name}' - continuation pattern matched")
                return True
        
        return False
    
    def _attempt_truncation_completion(self, fragment: str, text: str, position: Tuple[int, int]) -> Optional[str]:
        """
        Attempt to complete a truncated disease name using database
        
        Conservative strategy:
        1. Extract extended context window
        2. Try prefix matching in database
        3. Validate completion makes sense
        4. Return only high-confidence completions
        """
        if not self.use_truncation_completion:
            return None
        
        start, end = position
        
        # Get extended context (up to 200 chars after fragment)
        context_after = text[end:end+200] if end < len(text) else ""
        
        # Extract potential continuation
        continuation_match = re.match(r'^[\s\---]*([a-zA-Z\s\-]+?)(?:\.|,|;|\(|\n|$)', context_after)
        if not continuation_match:
            return None
        
        potential_suffix = continuation_match.group(1).strip()
        if not potential_suffix or len(potential_suffix) > 50:
            return None
        
        # Construct potential complete name
        potential_complete = f"{fragment} {potential_suffix}".strip()
        
        # Try to find in database
        completion_result = self._find_completion_in_database(fragment, potential_complete)
        
        if completion_result and completion_result.get('confidence', 0) >= TRUNCATION_COMPLETION_MIN_CONFIDENCE:
            logger.info(f"[OK] Completed truncation: '{fragment}' -> '{completion_result['completed_name']}'")
            return completion_result['completed_name']
        
        return None
    
    def _find_completion_in_database(self, fragment: str, potential_complete: str) -> Optional[Dict]:
        """
        Find completion candidate in database using conservative prefix matching
        """
        candidates = []
        
        # Try Orphanet prefix search
        if self.orphanet_db_path and Path(self.orphanet_db_path).exists():
            try:
                conn = sqlite3.connect(self.orphanet_db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Search for terms that start with fragment
                cursor.execute("""
                    SELECT ce.orphacode, lr.text_value as name,
                           LENGTH(lr.text_value) as name_length
                    FROM core_entities ce
                    JOIN linguistic_representations lr ON ce.entity_id = lr.entity_id
                    WHERE lr.text_type IN ('preferred_term', 'synonym')
                    AND ce.status = 'active'
                    AND LOWER(lr.text_value) LIKE LOWER(?) || '%'
                    ORDER BY name_length ASC
                    LIMIT 5
                """, (fragment,))
                
                for row in cursor.fetchall():
                    db_name = row['name']
                    
                    # Calculate match confidence
                    confidence = self._calculate_completion_confidence(
                        fragment, potential_complete, db_name
                    )
                    
                    if confidence > 0:
                        candidates.append({
                            'completed_name': db_name,
                            'confidence': confidence,
                            'source': 'orphanet',
                            'orphacode': row['orphacode']
                        })
                
                conn.close()
                
            except Exception as e:
                logger.debug(f"Orphanet completion search failed: {e}")
        
        # Try DOID prefix search
        if self.doid_db_path and Path(self.doid_db_path).exists():
            try:
                conn = sqlite3.connect(self.doid_db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT doid, name, LENGTH(name) as name_length
                    FROM diseases
                    WHERE (is_obsolete = 0 OR is_obsolete IS NULL)
                    AND LOWER(name) LIKE LOWER(?) || '%'
                    ORDER BY name_length ASC
                    LIMIT 5
                """, (fragment,))
                
                for row in cursor.fetchall():
                    db_name = row['name']
                    
                    confidence = self._calculate_completion_confidence(
                        fragment, potential_complete, db_name
                    )
                    
                    if confidence > 0:
                        candidates.append({
                            'completed_name': db_name,
                            'confidence': confidence,
                            'source': 'doid',
                            'doid': row['doid']
                        })
                
                conn.close()
                
            except Exception as e:
                logger.debug(f"DOID completion search failed: {e}")
        
        # Return best candidate
        if candidates:
            best = max(candidates, key=lambda x: x['confidence'])
            return best
        
        return None
    
    def _calculate_completion_confidence(self, fragment: str, potential_complete: str, db_name: str) -> float:
        """
        Calculate confidence for a completion candidate
        
        Factors:
        1. How well potential_complete matches db_name
        2. Length similarity (avoid completing short fragments to very long names)
        3. Word boundary alignment
        """
        fragment_lower = fragment.lower().strip()
        potential_lower = potential_complete.lower().strip()
        db_lower = db_name.lower().strip()
        
        # Must start with fragment
        if not db_lower.startswith(fragment_lower):
            return 0.0
        
        confidence = 0.0
        
        # Exact match with potential complete
        if db_lower == potential_lower:
            confidence = 1.0
        # Very close match (allowing for minor differences)
        elif db_lower.startswith(potential_lower) or potential_lower.startswith(db_lower):
            # Calculate similarity
            shorter = min(len(potential_lower), len(db_lower))
            longer = max(len(potential_lower), len(db_lower))
            similarity = shorter / longer
            confidence = 0.8 * similarity
        # Partial match
        else:
            # Check word boundary alignment
            fragment_words = fragment_lower.split()
            potential_words = potential_lower.split()
            db_words = db_lower.split()
            
            # Count matching words after fragment
            matching_words = 0
            for i, word in enumerate(potential_words[len(fragment_words):]):
                if i < len(db_words) - len(fragment_words) and word == db_words[len(fragment_words) + i]:
                    matching_words += 1
            
            if matching_words > 0:
                confidence = 0.6 + (0.2 * matching_words / max(1, len(potential_words) - len(fragment_words)))
        
        # Penalize very long completions (likely overfitting)
        length_ratio = len(db_name) / max(1, len(fragment))
        if length_ratio > 3:  # If completed name is >3x fragment length
            confidence *= 0.7
        
        return min(confidence, 1.0)
    
    def _query_orphanet_exact(self, original_name: str, normalized_name: str) -> Optional[Dict]:
        """Query Orphanet with exact matching on both preferred terms and synonyms"""
        try:
            conn = sqlite3.connect(self.orphanet_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Try exact match on preferred term
            cursor.execute("""
                SELECT ce.orphacode, lr.text_value as preferred_term
                FROM core_entities ce
                JOIN linguistic_representations lr ON ce.entity_id = lr.entity_id
                WHERE lr.text_type = 'preferred_term'
                AND ce.status = 'active'
                AND (LOWER(lr.text_value) = LOWER(?) OR LOWER(lr.text_value) = LOWER(?))
                ORDER BY 
                    CASE WHEN LOWER(lr.text_value) = LOWER(?) THEN 0 ELSE 1 END,
                    LENGTH(lr.text_value) ASC
                LIMIT 1
            """, (original_name, normalized_name, original_name))
            
            row = cursor.fetchone()
            
            if row:
                orphacode = row['orphacode']
                preferred_term = row['preferred_term']
                
                all_ids = self._get_orphanet_external_ids(cursor, orphacode)
                all_ids['orphanet'] = str(orphacode)
                
                conn.close()
                return {
                    'primary_id': f'ORPHA:{orphacode}',
                    'orphacode': str(orphacode),
                    'canonical_name': preferred_term,
                    'all_ids': all_ids,
                    'match_confidence': 1.0,
                    'match_type': 'exact_preferred'
                }
            
            # Try exact match on synonym
            cursor.execute("""
                SELECT ce.orphacode, 
                    lr_pref.text_value as preferred_term,
                    lr_syn.text_value as matched_synonym
                FROM core_entities ce
                JOIN linguistic_representations lr_syn ON ce.entity_id = lr_syn.entity_id
                JOIN linguistic_representations lr_pref ON ce.entity_id = lr_pref.entity_id
                WHERE lr_syn.text_type = 'synonym'
                AND lr_pref.text_type = 'preferred_term'
                AND ce.status = 'active'
                AND (LOWER(lr_syn.text_value) = LOWER(?) OR LOWER(lr_syn.text_value) = LOWER(?))
                LIMIT 1
            """, (original_name, normalized_name))
            
            row = cursor.fetchone()
            
            if row:
                orphacode = row['orphacode']
                preferred_term = row['preferred_term']
                
                all_ids = self._get_orphanet_external_ids(cursor, orphacode)
                all_ids['orphanet'] = str(orphacode)
                
                logger.debug(f"Found via synonym: '{original_name}' -> '{preferred_term}' (ORPHA:{orphacode})")
                
                conn.close()
                return {
                    'primary_id': f'ORPHA:{orphacode}',
                    'orphacode': str(orphacode),
                    'canonical_name': preferred_term,
                    'all_ids': all_ids,
                    'match_confidence': 0.95,
                    'match_type': 'exact_synonym'
                }
            
            conn.close()
            return None
            
        except Exception as e:
            logger.debug(f"Orphanet exact query failed: {e}")
            return None
    
    def _query_orphanet_fuzzy(self, original_name: str, normalized_name: str) -> Optional[Dict]:
        """Query Orphanet with fuzzy matching"""
        try:
            conn = sqlite3.connect(self.orphanet_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Try FTS if available
            try:
                cursor.execute("""
                    SELECT orphacode, preferred_term, rank
                    FROM entity_fts
                    WHERE preferred_term MATCH ?
                    ORDER BY rank
                    LIMIT 3
                """, (normalized_name,))
                
                rows = cursor.fetchall()
                best_match = None
                best_confidence = 0.0
                
                for row in rows:
                    confidence = 0.85
                    db_term_normalized = self._normalize_disease_name_for_lookup(row['preferred_term'])
                    if db_term_normalized == normalized_name:
                        confidence = 0.95
                    elif normalized_name in db_term_normalized or db_term_normalized in normalized_name:
                        confidence = 0.90
                    
                    if confidence > best_confidence:
                        best_match = row
                        best_confidence = confidence
                
                if best_match:
                    orphacode = best_match['orphacode']
                    preferred_term = best_match['preferred_term']
                    
                    all_ids = self._get_orphanet_external_ids(cursor, orphacode)
                    all_ids['orphanet'] = str(orphacode)
                    
                    conn.close()
                    return {
                        'primary_id': f'ORPHA:{orphacode}',
                        'orphacode': str(orphacode),
                        'canonical_name': preferred_term,
                        'all_ids': all_ids,
                        'match_confidence': best_confidence,
                        'match_type': 'fts'
                    }
            except sqlite3.OperationalError:
                pass
            
            conn.close()
            return None
            
        except Exception as e:
            logger.debug(f"Orphanet fuzzy query failed: {e}")
            return None
    
    def _get_orphanet_external_ids(self, cursor, orphacode: str) -> Dict[str, str]:
        """Helper to fetch external IDs for an Orphanet entity"""
        cursor.execute("""
            SELECT em.external_system, em.external_code
            FROM external_mappings em
            JOIN core_entities ce ON em.entity_id = ce.entity_id
            WHERE ce.orphacode = ?
        """, (orphacode,))
        
        all_ids = {}
        for mapping in cursor.fetchall():
            system = mapping['external_system'].lower()
            code = mapping['external_code']
            
            if 'doid' in system:
                all_ids['doid'] = code if code.startswith('DOID:') else f'DOID:{code}'
            elif 'snomed' in system:
                all_ids['snomed'] = code
            elif 'icd10' in system or 'icd-10' in system:
                all_ids['icd10'] = code
            elif 'mesh' in system:
                all_ids['mesh'] = code
            elif 'umls' in system:
                all_ids['umls'] = code
        
        return all_ids
    
    def _query_doid_enhanced(self, original_name: str, normalized_name: str) -> Optional[Dict]:
        """Query Disease Ontology with enhanced matching"""
        try:
            conn = sqlite3.connect(self.doid_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Try exact match
            cursor.execute("""
                SELECT doid, name 
                FROM diseases
                WHERE LOWER(name) = LOWER(?) OR LOWER(name) = LOWER(?)
                AND (is_obsolete = 0 OR is_obsolete IS NULL)
                LIMIT 1
            """, (original_name, normalized_name))
            
            row = cursor.fetchone()
            
            if row:
                doid = row['doid']
                canonical_name = row['name']
                
                all_ids = {'doid': doid}
                
                try:
                    cursor.execute("""
                        SELECT xref
                        FROM xrefs
                        WHERE doid = ?
                    """, (doid,))
                    
                    for xref_row in cursor.fetchall():
                        xref = xref_row['xref']
                        if ':' in xref:
                            xref_type, xref_id = xref.split(':', 1)
                            xref_type_lower = xref_type.lower()
                            
                            if xref_type_lower in ['orpha', 'orphanet']:
                                all_ids['orphanet'] = xref_id
                            elif xref_type_lower in ['icd10cm', 'icd10']:
                                all_ids['icd10'] = xref_id
                            elif xref_type_lower == 'mesh':
                                all_ids['mesh'] = xref_id
                            elif xref_type_lower in ['snomedct_us_2023_03_01', 'snomedct']:
                                all_ids['snomed'] = xref_id
                            elif xref_type_lower == 'umls_cui':
                                all_ids['umls'] = xref_id
                except sqlite3.OperationalError:
                    pass
                
                conn.close()
                return {
                    'primary_id': doid,
                    'canonical_name': canonical_name,
                    'all_ids': all_ids,
                    'match_confidence': 0.9,
                    'match_type': 'exact'
                }
            
            # Try synonym match
            try:
                cursor.execute("""
                    SELECT d.doid, d.name
                    FROM synonyms s
                    JOIN diseases d ON s.doid = d.doid
                    WHERE LOWER(s.synonym) = LOWER(?) OR LOWER(s.synonym) = LOWER(?)
                    AND (d.is_obsolete = 0 OR d.is_obsolete IS NULL)
                    LIMIT 1
                """, (original_name, normalized_name))
                
                row = cursor.fetchone()
                
                if row:
                    doid = row['doid']
                    canonical_name = row['name']
                    
                    conn.close()
                    return {
                        'primary_id': doid,
                        'canonical_name': canonical_name,
                        'all_ids': {'doid': doid},
                        'match_confidence': 0.85,
                        'match_type': 'synonym'
                    }
            except sqlite3.OperationalError:
                pass
            
            conn.close()
            return None
            
        except Exception as e:
            logger.debug(f"DOID query failed: {e}")
            return None
    
    def _normalize_disease_name_for_lookup(self, name: str) -> str:
        """Normalize disease name for database lookup"""
        normalized = name.lower().strip()
        normalized = re.sub(r"'s\b", '', normalized)
        normalized = re.sub(r'[^\w\s\-]', ' ', normalized)
        normalized = ' '.join(normalized.split())
        
        for suffix in ['disease', 'syndrome', 'disorder', 'condition']:
            if normalized.endswith(f' {suffix}'):
                normalized = normalized[:-len(suffix)-1].strip()
        
        return normalized
    
    def _query_disease_ids(self, disease_name: str) -> Optional[Dict]:
        """Query databases for disease IDs with improved matching strategy"""
        if not disease_name or len(disease_name.strip()) < 2:
            return None
        
        if self._is_false_positive_disease(disease_name):
            logger.debug(f"Skipping false positive: {disease_name}")
            return None
        
        result = {
            'primary_id': None,
            'canonical_name': disease_name,
            'all_ids': {},
            'match_confidence': 0.0,
            'match_type': None
        }
        
        normalized_input = self._normalize_disease_name_for_lookup(disease_name)
        
        # Try Orphanet first
        if self.orphanet_db_path and Path(self.orphanet_db_path).exists():
            orpha_result = self._query_orphanet_exact(disease_name, normalized_input)
            if orpha_result and orpha_result.get('match_confidence', 0) >= 0.9:
                return orpha_result
            
            orpha_fuzzy = self._query_orphanet_fuzzy(disease_name, normalized_input)
            if orpha_fuzzy and orpha_fuzzy.get('match_confidence', 0) > result.get('match_confidence', 0):
                orpha_result = orpha_fuzzy
            
            if orpha_result and orpha_result.get('match_confidence', 0) >= 0.7:
                return orpha_result
            
            if orpha_result:
                result = orpha_result
        
        # Try DOID
        if self.doid_db_path and Path(self.doid_db_path).exists():
            doid_result = self._query_doid_enhanced(disease_name, normalized_input)
            
            if doid_result and doid_result.get('match_confidence', 0) > result.get('match_confidence', 0):
                result = doid_result
        
        if result.get('primary_id') and result.get('match_confidence', 0) >= 0.6:
            return result
        
        return None
    
    # Abbreviation processing methods
    def process_abbreviation_candidates(self, 
                                       abbreviation_context: Dict[str, Any],
                                       text: str) -> List[DetectedDisease]:
        """Process abbreviation candidates with semantic gating"""
        diseases = []
        candidates = abbreviation_context.get('disease_candidates', [])
        
        for candidate in candidates:
            abbrev = candidate.get('abbreviation', '')
            if not self._should_promote_to_disease(candidate):
                logger.debug(f"Rejected abbreviation {abbrev}")
                continue
            
            expansion = self._get_best_expansion(candidate, text)
            if not expansion:
                continue
            
            disease = DetectedDisease(
                name=expansion,
                confidence=self._calculate_abbreviation_confidence(candidate),
                detection_method='abbreviation_enrichment',
                occurrences=candidate.get('occurrences', 1),
                from_abbreviation=abbrev,
                semantic_tui=candidate.get('semantic_tui'),
                matched_terms=[abbrev, expansion],
                source='abbreviation',
                positions=[]
            )
            
            diseases.append(disease)
            logger.info(f"Promoted abbreviation: {abbrev} -> {expansion}")
        
        return diseases
    
    def _should_promote_to_disease(self, candidate: Dict) -> bool:
        """Determine if abbreviation should be promoted"""
        expansion = (candidate.get('local_expansion') or candidate.get('expansion') or '').lower()
        if not expansion:
            return False
        
        semantic_tui = candidate.get('semantic_tui') or candidate.get('semantic_type')
        if semantic_tui:
            return semantic_tui in DISORDERS_STYS
        
        has_neg = any(word in expansion for word in DISEASE_NEG_WORDS)
        if has_neg:
            return False
        
        has_pos_word = any(word in expansion for word in DISEASE_POS_WORDS)
        has_pos_suffix = any(expansion.endswith(suffix) for suffix in DISEASE_POS_SUFFIXES)
        
        return has_pos_word or has_pos_suffix
    
    def _get_best_expansion(self, candidate: Dict, text: str) -> Optional[str]:
        """Get best expansion for abbreviation"""
        abbrev = candidate.get('abbreviation', '')
        local_expansion = candidate.get('local_expansion')
        if local_expansion:
            return local_expansion
        
        patterns = [
            rf'([^(]+?)\s*\(\s*{re.escape(abbrev)}\s*\)',
            rf'{re.escape(abbrev)}\s*\(\s*([^)]+?)\s*\)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                expansion = match.group(1).strip()
                if self._is_valid_expansion(expansion):
                    return expansion
        
        return candidate.get('expansion')
    
    def _is_valid_expansion(self, expansion: str) -> bool:
        """Check if expansion is valid"""
        if not expansion or len(expansion) < 3:
            return False
        if len(expansion) > 100 or len(expansion.split()) > 15:
            return False
        if expansion[0].islower():
            return False
        return True
    
    def _calculate_abbreviation_confidence(self, candidate: Dict) -> float:
        """Calculate confidence for abbreviation-based disease"""
        base_confidence = candidate.get('confidence', 0.8)
        if candidate.get('semantic_tui'):
            base_confidence *= 1.1
        if candidate.get('local_expansion'):
            base_confidence *= 1.05
        base_confidence *= SOURCE_CONFIDENCE.get('abbreviation_enrichment', 0.70) / 0.85
        return min(base_confidence, 1.0)
    
    # Main detection method
    def detect_diseases(self, text: str, abbreviation_context: Optional[Dict] = None) -> DiseaseDetectionResult:
        """Main disease detection method with truncation handling integrated"""
        if text is None or not text:
            return DiseaseDetectionResult(
                diseases=[], detection_time=0, methods_used=[], analytics={}
            )
        
        start_time = time.time()
        methods_used = []
        all_detections = []
        analytics = {'stage_metrics': {}}
        
        # Detection pipeline
        detection_methods = [
            ('patterns', self.use_patterns, self._detect_by_patterns),
            ('lexicon', self.use_lexicon, self._detect_by_lexicon),
            ('ner', self.use_ner, self._detect_by_ner)
        ]
        
        for method_name, enabled, detect_func in detection_methods:
            if enabled:
                stage_start = time.time()
                detections = detect_func(text)
                all_detections.extend(detections)
                methods_used.append(method_name)
                analytics['stage_metrics'][method_name] = {
                    'count': len(detections),
                    'time': round(time.time() - stage_start, 3)
                }
        
        # [OK] NEW: Handle truncation completion BEFORE other processing
        if self.use_truncation_completion:
            stage_start = time.time()
            completed_detections = []
            truncation_count = 0
            
            for detection in all_detections:
                # Check each position for potential truncation
                for position in detection.positions:
                    if self._is_likely_truncated(detection.name, text, position):
                        # Attempt completion
                        completed_name = self._attempt_truncation_completion(
                            detection.name, text, position
                        )
                        
                        if completed_name and completed_name != detection.name:
                            # Create new detection with completed name
                            completed_detection = DetectedDisease(
                                name=completed_name,
                                confidence=detection.confidence * 0.95,  # Slight confidence reduction
                                positions=[position],
                                occurrences=1,
                                source=f"{detection.source}_truncation_completed",
                                detection_method='truncation_completion',
                                is_truncated=True,
                                original_fragment=detection.name,
                                completion_confidence=SOURCE_CONFIDENCE['truncation_completion']
                            )
                            completed_detections.append(completed_detection)
                            truncation_count += 1
                            
                            # Remove this position from original detection
                            detection.positions.remove(position)
                            break  # Only complete once per detection
            
            # Add completed detections
            all_detections.extend(completed_detections)
            
            if truncation_count > 0:
                methods_used.append('truncation_completion')
            
            analytics['stage_metrics']['truncation_completion'] = {
                'count': truncation_count,
                'time': round(time.time() - stage_start, 3)
            }
        
        # Process abbreviations
        if abbreviation_context:
            stage_start = time.time()
            abbrev_diseases = self.process_abbreviation_candidates(abbreviation_context, text)
            all_detections.extend(abbrev_diseases)
            if abbrev_diseases:
                methods_used.append('abbreviation_enrichment')
            analytics['stage_metrics']['abbreviation_enrichment'] = {
                'count': len(abbrev_diseases),
                'time': round(time.time() - stage_start, 3)
            }
        
        # Semantic filtering
        if self.kb_resolver and all_detections:
            stage_start = time.time()
            filtered_detections, demoted_entities = postfilter_direct_diseases(
                all_detections, 
                self.kb_resolver
            )
            all_detections = filtered_detections
            analytics['stage_metrics']['semantic_filtering'] = {
                'before': len(all_detections) + len(demoted_entities),
                'after': len(all_detections),
                'demoted': len(demoted_entities),
                'time': round(time.time() - stage_start, 3)
            }
        
        # Negation analysis
        stage_start = time.time()
        all_detections = self._apply_negation_analysis(text, all_detections)
        analytics['stage_metrics']['negation'] = {
            'time': round(time.time() - stage_start, 3),
            'negated_count': len([d for d in all_detections if d.is_negated])
        }
        
        # Deduplicate
        stage_start = time.time()
        all_detections = self._deduplicate_diseases(all_detections)
        analytics['stage_metrics']['deduplication'] = {
            'time': round(time.time() - stage_start, 3)
        }
        
        # Enrich with database IDs (with false positive filtering)
        enriched_detections = []
        diseases_with_ids = 0
        diseases_filtered = 0
        truncations_completed = 0
        
        for disease in all_detections:
            # Query database for IDs (false positive filter is inside _query_disease_ids)
            db_result = self._query_disease_ids(disease.name)
            
            if db_result and db_result.get('primary_id'):
                # Add IDs to disease
                disease.primary_id = db_result['primary_id']
                disease.all_ids = db_result.get('all_ids', {})
                disease.canonical_name = db_result.get('canonical_name', disease.name)
                
                # Add individual ID fields
                if 'orphanet' in disease.all_ids:
                    disease.orphacode = disease.all_ids['orphanet']
                    disease.identifiers['ORPHA'] = f"ORPHA:{disease.all_ids['orphanet']}"
                if 'doid' in disease.all_ids:
                    disease.identifiers['DOID'] = disease.all_ids['doid']
                if 'snomed' in disease.all_ids:
                    disease.identifiers['SNOMED'] = disease.all_ids['snomed']
                if 'icd10' in disease.all_ids:
                    disease.identifiers['ICD10'] = disease.all_ids['icd10']
                
                # Boost confidence for rare diseases
                if disease.primary_id and disease.primary_id.startswith('ORPHA:'):
                    disease.confidence = min(0.95, disease.confidence + 0.05)
                
                diseases_with_ids += 1
                enriched_detections.append(disease)
                
                if disease.is_truncated:
                    truncations_completed += 1
                    logger.info(f"[OK] Completed truncation with ID: '{disease.original_fragment}' -> '{disease.name}' ({db_result.get('primary_id')})")
                else:
                    logger.info(f"[OK] Found IDs for '{disease.name}': {db_result.get('primary_id')}")
            
            elif db_result is None and self._is_false_positive_disease(disease.name):
                # Filtered out as false positive
                diseases_filtered += 1
                logger.debug(f"[X] Filtered false positive: '{disease.name}'")
            
            else:
                # No IDs found but keep it
                enriched_detections.append(disease)
                if disease.is_truncated:
                    logger.debug(f"[X] No IDs found for completed truncation: '{disease.name}'")
                else:
                    logger.debug(f"[X] No IDs found for '{disease.name}'")
        
        # Log ID coverage and filtering
        if enriched_detections:
            coverage = diseases_with_ids / len(enriched_detections)
            logger.info(f"Disease ID coverage: {diseases_with_ids}/{len(enriched_detections)} ({coverage*100:.1f}%)")
        
        if diseases_filtered > 0:
            logger.info(f"Filtered {diseases_filtered} false positive diseases")
        
        if truncations_completed > 0:
            logger.info(f"Successfully completed {truncations_completed} truncated disease names with IDs")
        
        all_detections = enriched_detections
        
        # Filter by confidence
        filtered_diseases = []
        for disease in all_detections:
            if disease.is_negated:
                if disease.confidence > 0:
                    filtered_diseases.append(disease)
            else:
                if disease.confidence >= self.confidence_threshold:
                    filtered_diseases.append(disease)
        
        # Sort
        filtered_diseases.sort(key=lambda d: (-d.confidence, min(d.positions)[0] if d.positions else 0))
        
        # Analytics
        total_time = time.time() - start_time
        analytics['processing_time_seconds'] = round(total_time, 3)
        analytics['total_diseases'] = len(filtered_diseases)
        analytics['negated_diseases'] = len([d for d in filtered_diseases if d.is_negated])
        analytics['diseases_with_ids'] = diseases_with_ids
        analytics['diseases_filtered'] = diseases_filtered
        analytics['truncations_completed'] = truncations_completed
        analytics['id_coverage'] = round(diseases_with_ids / len(filtered_diseases), 3) if filtered_diseases else 0
        
        return DiseaseDetectionResult(
            diseases=filtered_diseases,
            detection_time=total_time,
            methods_used=methods_used,
            mode=self.mode,
            analytics=analytics
        )
    
    # Detection helper methods
    def _detect_by_patterns(self, text: str) -> List[DetectedDisease]:
        """Detect diseases using regex patterns"""
        return self._generic_pattern_detection(
            text, 
            self.compiled_patterns, 
            SOURCE_CONFIDENCE['pattern'],
            'pattern'
        )
    
    def _detect_by_lexicon(self, text: str) -> List[DetectedDisease]:
        """Lexicon matching"""
        if not self.disease_lexicon:
            return []
        
        detections = []
        text_lower = text.lower()
        
        for term, info in self.disease_lexicon.items():
            if term.lower() in GENERIC_TERMS_TO_EXCLUDE:
                continue
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            matches = list(re.finditer(pattern, text_lower))
            
            if matches:
                positions = [(m.start(), m.end()) for m in matches]
                detections.append(DetectedDisease(
                    name=info.get('preferred_label', term),
                    confidence=SOURCE_CONFIDENCE['lexicon'],
                    positions=positions,
                    occurrences=len(matches),
                    source='lexicon',
                    detection_method='lexicon'
                ))
        
        return detections
    
    def _detect_by_ner(self, text: str) -> List[DetectedDisease]:
        """NER detection"""
        detections = []
        for model_name, nlp in self.nlp_models.items():
            try:
                doc = nlp(text)
                for ent in doc.ents:
                    if ent.label_ in DISEASE_LABELS:
                        if ent.text.lower() not in GENERIC_TERMS_TO_EXCLUDE:
                            detections.append(DetectedDisease(
                                name=ent.text,
                                confidence=SOURCE_CONFIDENCE['ner'],
                                positions=[(ent.start_char, ent.end_char)],
                                occurrences=1,
                                source='ner',
                                detection_method=f'ner_{model_name}'
                            ))
            except Exception as e:
                logger.error(f"NER failed: {e}")
        return detections
    
    def _generic_pattern_detection(self, text: str, patterns: List, confidence: float, source: str) -> List[DetectedDisease]:
        """Generic pattern detection"""
        detections = []
        for pattern, name in patterns:
            matches = list(pattern.finditer(text))
            if matches:
                positions = [(m.start(), m.end()) for m in matches]
                detections.append(DetectedDisease(
                    name=name,
                    confidence=confidence,
                    positions=positions,
                    occurrences=len(matches),
                    source=source,
                    detection_method=source
                ))
        return detections
    
    def _apply_negation_analysis(self, text: str, detections: List[DetectedDisease]) -> List[DetectedDisease]:
        """Negation analysis"""
        if not detections:
            return detections
        
        for detection in detections:
            is_negated = False
            for position in detection.positions:
                start, end = position
                context_before_start = max(0, start - NEGATION_WINDOW_BEFORE)
                context_before = text[context_before_start:start]
                
                for pattern in self.negation_patterns_before:
                    if pattern.search(context_before):
                        if not _blocked_by_boundary_or_contrast(context_before):
                            is_negated = True
                            break
                
                if is_negated:
                    break
            
            detection.is_negated = is_negated
            if is_negated:
                detection.confidence *= NEGATION_CONFIDENCE_FACTOR
        
        return detections
    
    def _deduplicate_diseases(self, detections: List[DetectedDisease]) -> List[DetectedDisease]:
        """
        Deduplicate disease detections
        
        Special handling for truncation completions:
        - If both fragment and completed version exist, keep completed
        - Merge positions and confidence appropriately
        """
        if not detections:
            return []
        
        grouped = defaultdict(list)
        truncation_map = {}  # Map fragments to their completions
        
        # First pass: identify truncation relationships
        for detection in detections:
            if detection.is_truncated and detection.original_fragment:
                fragment_normalized = self._normalize_disease_name(detection.original_fragment)
                completed_normalized = self._normalize_disease_name(detection.name)
                truncation_map[fragment_normalized] = completed_normalized
        
        # Second pass: group by normalized name
        for detection in detections:
            normalized = self._normalize_disease_name(detection.name)
            
            # If this is a fragment that was completed, skip it
            if normalized in truncation_map and not detection.is_truncated:
                logger.debug(f"Skipping fragment '{detection.name}' - completed version exists")
                continue
            
            # If this is a completed version, use its normalized name
            if detection.is_truncated:
                grouped[normalized].append(detection)
            # If this fragment has a completion, group under completion
            elif normalized in truncation_map:
                grouped[truncation_map[normalized]].append(detection)
            else:
                grouped[normalized].append(detection)
        
        # Merge groups
        merged = []
        for normalized_name, group in grouped.items():
            if not group:
                continue
            
            # Prefer completed versions
            completed = [d for d in group if d.is_truncated]
            if completed:
                best = max(completed, key=lambda d: d.confidence)
            else:
                best = max(group, key=lambda d: d.confidence)
            
            # Merge positions from all variants
            all_positions = []
            for detection in group:
                all_positions.extend(detection.positions)
            best.positions = list(set(all_positions))  # Remove duplicates
            
            merged.append(best)
        
        return merged
    
    def _normalize_disease_name(self, name: str) -> str:
        """Normalize disease name"""
        normalized = re.sub(r'[^\w\s]', '', name.lower())
        normalized = ' '.join(normalized.split())
        for suffix in DISEASE_SUFFIXES:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
        return normalized
    
    def _extract_context(self, text: str, start: int, end: int) -> str:
        """Extract context"""
        if not text:
            return ""
        context_start = max(0, start - CONTEXT_WINDOW)
        context_end = min(len(text), end + CONTEXT_WINDOW)
        return text[context_start:context_end]
    
    def detect_diseases_batch(self, documents: Dict[str, str]) -> Dict[str, List[DetectedDisease]]:
        """Batch processing"""
        results = {}
        for doc_id, text in documents.items():
            try:
                result = self.detect_diseases(text)
                results[doc_id] = result.diseases
            except Exception as e:
                logger.error(f"Failed to process {doc_id}: {e}")
                results[doc_id] = []
        return results
    
    def extract_diseases(self, text: str) -> List[DetectedDisease]:
        """Legacy method"""
        result = self.detect_diseases(text)
        return result.diseases
    
    def get_disease_info(self, disease_name: str) -> Optional[Dict]:
        """Get disease info"""
        if not disease_name:
            return None
        return None


# ============================================================================
# Factory Functions
# ============================================================================

def create_detector(mode='balanced', claude_client=None, verbose=False, **kwargs):
    """Factory function to create disease detector with proper initialization"""
    detector_kwargs = {
        'mode': mode
    }
    
    supported_params = [
        'use_patterns', 'use_ner', 'use_lexicon', 'use_validation',
        'use_truncation_completion',  # [OK] NEW
        'system_initializer', 'kb_resolver', 
        'orphanet_db_path', 'doid_db_path', 'config_path'
    ]
    
    for param in supported_params:
        if param in kwargs:
            detector_kwargs[param] = kwargs[param]
    
    if claude_client:
        detector_kwargs['use_validation'] = True
    
    if verbose:
        logger.info(f"Creating detector with kwargs: {list(detector_kwargs.keys())}")
    
    return RareDiseaseDetector(**detector_kwargs)


def get_rare_disease_detector(mode: str = "balanced", kb_resolver: Optional[Any] = None) -> RareDiseaseDetector:
    """Get singleton detector"""
    if not hasattr(get_rare_disease_detector, '_instance'):
        get_rare_disease_detector._instance = create_detector(mode, kb_resolver=kb_resolver)
    return get_rare_disease_detector._instance