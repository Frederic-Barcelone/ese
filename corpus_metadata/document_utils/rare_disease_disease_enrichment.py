#!/usr/bin/env python3
"""
corpus_metadata/document_utils/rare_disease_disease_enrichment.py
===================================

PURPOSE:
--------
This module handles disease data enrichment, deduplication, and database integration:
- Disease name normalization and canonicalization
- Entity deduplication and similarity matching
- Noise filtering and false positive removal
- ORPHA code and ICD mapping enrichment
- Special handling for abbreviation-derived diseases
- ID normalization and semantic type validation
- Integration with MetadataSystemInitializer for resource loading

Version: 3.2 - Added semantic validation and system initializer integration
Last Updated: 2025-01-17
"""

import re
import json
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from functools import lru_cache

# Prefer rapidfuzz for better performance
try:
    from rapidfuzz import fuzz, process
    FUZZY_BACKEND = "rapidfuzz"
except ImportError:
    from fuzzywuzzy import fuzz
    process = None
    FUZZY_BACKEND = "fuzzywuzzy"

# Support for diacritic removal
try:
    from unidecode import unidecode
    HAS_UNIDECODE = True
except ImportError:
    def unidecode(x: str) -> str:
        return x
    HAS_UNIDECODE = False

logger = logging.getLogger(__name__)

# ============================================================================
# ID NORMALIZATION AND VALIDATION CONSTANTS
# ============================================================================

ID_KEY_MAP = {
    'umls': 'UMLS', 'umls_cui': 'UMLS', 'cui': 'UMLS', 'umls_id': 'UMLS',
    'snomed': 'SNOMED', 'snomed_ct': 'SNOMED', 'snomedct': 'SNOMED', 'snomed_id': 'SNOMED',
    'icd10': 'ICD10', 'icd-10': 'ICD10', 'icd_10': 'ICD10', 'icd10_code': 'ICD10',
    'icd9': 'ICD9', 'icd-9': 'ICD9', 'icd_9': 'ICD9', 'icd9_code': 'ICD9',
    'icd11': 'ICD11', 'icd-11': 'ICD11', 'icd_11': 'ICD11',
    'orpha': 'ORPHA', 'orpha_code': 'ORPHA', 'orphanet': 'ORPHA', 'orpha_id': 'ORPHA',
    'orphacode': 'ORPHA',
    'doid': 'DOID', 'do_id': 'DOID', 'disease_ontology': 'DOID',
    'mondo': 'MONDO', 'mondo_id': 'MONDO',
    'mesh': 'MESH', 'mesh_id': 'MESH', 'mesh_code': 'MESH',
    'omim': 'OMIM', 'omim_id': 'OMIM', 'mim': 'OMIM',
}

PREFERRED_DISEASE_KEYS = ('ORPHA', 'DOID', 'UMLS', 'SNOMED', 'MONDO', 'MESH', 'OMIM', 'ICD10', 'ICD9')
REQUIRED_DISEASE_IDS = ('UMLS', 'SNOMED', 'ICD10', 'ICD9', 'ORPHA', 'DOID', 'MESH', 'OMIM', 'MONDO')

DISORDER_SEMANTIC_TYPES = {
    'T019', 'T020', 'T037', 'T046', 'T047', 'T048', 'T049', 'T050', 'T190', 'T191'
}

NON_DISEASE_SEMANTIC_TYPES = {
    'T004', 'T005', 'T007', 'T008', 'T028', 'T059', 'T060', 'T092', 'T116'
}

# ============================================================================
# ID NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_ids(ids: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize all ID keys to canonical form"""
    if not ids:
        return {}
    
    normalized = {}
    for key, value in ids.items():
        if not value:
            continue
        key_lower = key.lower()
        canonical_key = ID_KEY_MAP.get(key_lower, key.upper())
        if canonical_key not in normalized:
            normalized[canonical_key] = value
    return normalized

def pick_primary_id(ids: Dict[str, str], order: Tuple[str, ...] = PREFERRED_DISEASE_KEYS) -> Tuple[Optional[str], Optional[str]]:
    """Select the primary ID based on preferred order"""
    for id_type in order:
        if id_type in ids and ids[id_type]:
            return id_type, ids[id_type]
    return None, None

def is_disorder_entity(resolution: Dict) -> bool:
    """Check if resolved entity is actually a disorder"""
    if not resolution:
        return False
    
    ids = normalize_ids(resolution.get('ids', {}))
    semantic_type = resolution.get('semantic_type')
    
    has_disorder_semantic = semantic_type in DISORDER_SEMANTIC_TYPES
    has_disease_id = any(k in ids for k in REQUIRED_DISEASE_IDS)
    is_non_disease = semantic_type in NON_DISEASE_SEMANTIC_TYPES
    
    return (has_disorder_semantic or has_disease_id) and not is_non_disease

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DetectedDisease:
    """Data class for detected disease entities"""
    name: str
    positions: List[Tuple[int, int]] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    confidence: float = 0.0
    occurrences: int = 1
    source: str = ""
    orphacode: Optional[str] = None
    orpha_code: Optional[str] = None
    doid: Optional[str] = None
    matched_terms: List[str] = field(default_factory=list)
    detection_method: str = ""
    from_abbreviation: Optional[str] = None
    semantic_tui: Optional[str] = None
    
    # Enhanced fields for ID visibility
    ids: Dict[str, str] = field(default_factory=dict)
    canonical_name: Optional[str] = None
    primary_id: Optional[str] = None
    semantic_type: Optional[str] = None
    
    def set_orpha(self, code: Optional[str]):
        """Set ORPHA code with normalization"""
        if not code:
            return
        code = code.strip()
        if code and not code.upper().startswith("ORPHA:"):
            code = f"ORPHA:{code}"
        self.orpha_code = code
        self.orphacode = code
        if not self.ids:
            self.ids = {}
        self.ids['ORPHA'] = code
    
    def get_orpha(self) -> Optional[str]:
        """Get ORPHA code"""
        return self.orpha_code or self.orphacode or self.ids.get('ORPHA')
    
    def set_primary_id(self):
        """Set the primary ID based on available IDs"""
        if self.ids:
            id_type, id_value = pick_primary_id(self.ids)
            if id_type and id_value:
                self.primary_id = f"{id_type}:{id_value}"

# Canonicalization utilities
_ROMAN_MAP = {
    r"\btype\s*i\b": "type 1",
    r"\btype\s*ii\b": "type 2",
    r"\btype\s*iii\b": "type 3",
    r"\btype\s*iv\b": "type 4",
    r"\btype\s*v\b": "type 5",
    r"\bgrades?\s*i\b": "grade 1",
    r"\bgrades?\s*ii\b": "grade 2",
    r"\bgrades?\s*iii\b": "grade 3",
    r"\bgrades?\s*iv\b": "grade 4",
}

_HYPHEN_RE = re.compile(r"\s*[-–—]\s*")
_WS_RE = re.compile(r"\s+")
_APOSTROPHE_RE = re.compile(r"['']s?\b")

@lru_cache(maxsize=50000)
def _canon(text: str) -> str:
    """Cached canonicalization function"""
    if not text:
        return ""
    text = unidecode(text)
    text = text.lower()
    text = _APOSTROPHE_RE.sub("", text)
    text = _HYPHEN_RE.sub("-", text)
    text = re.sub(r"[^\w\s-]", "", text)
    for pattern, replacement in _ROMAN_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = _WS_RE.sub(" ", text).strip()
    return text

# ============================================================================
# MAIN ENRICHMENT CLASS
# ============================================================================

class RareDiseaseEnrichment:
    """
    Disease enrichment with database integration and semantic validation.
    Integrates with MetadataSystemInitializer for resource loading.
    """
    
    DEFAULT_GENERIC_TERMS = {
        "disease", "disorder", "syndrome", "condition", "symptom",
        "diagnosis", "prognosis", "treatment", "therapy", "patient",
        "clinical", "medical", "study", "trial", "case", "infection",
        "inflammation", "lesion", "abnormality", "complication",
        "presentation", "manifestation", "phenotype", "feature"
    }
    
    DEFAULT_NOISE_TERMS = {
        "hepatitis", "pneumonia", "diabetes", "hypertension", "cancer",
        "tumor", "carcinoma", "infection", "sepsis", "shock",
        "failure", "insufficiency", "deficiency", "resistance",
        "arthritis", "gastritis", "dermatitis", "colitis",
        "bronchitis", "sinusitis", "cystitis", "mastitis",
        "encephalitis", "meningitis", "vasculitis", "myocarditis",
        "nephritis", "neuritis", "otitis", "pancreatitis",
        "component", "factor", "complex", "system",
        "pathway", "cascade", "mechanism", "process"
    }
    
    NON_DISEASE_INDICATORS = {
        'complement component', 'protein', 'gene', 'receptor',
        'antibody', 'antigen', 'enzyme', 'trial', 'study',
        'guideline', 'organization', 'test', 'assay', 'method',
        'cytokine', 'chemokine', 'interleukin', 'interferon',
        'biomarker', 'marker', 'institute', 'foundation',
        'society', 'college', 'alliance', 'administration',
        'neutrophil extracellular trap', 'immunosorbent assay',
        'operating characteristic', 'exchange', 'dosing'
    }
    
    DEFAULT_FALSE_POSITIVE_PATTERNS = [
        r"\bno evidence of\b", r"\bruled?\s*out\b", r"\bnegative for\b",
        r"\bexcluded\b", r"\bnot consistent with\b", r"\bdifferential diagnosis\b",
        r"\bunlikely\b", r"\bfamily history of\b|\bfhx\b", r"\bhistory of\b|\bhx of\b",
        r"\bscreening for\b", r"\brisk of\b", r"\bvaccinated against\b",
        r"\bprophylaxis for\b", r"\bsuspected\b", r"\bpossible\b",
        r"\bquestionable\b", r"\buncertain\b", r"\bdenied\b", r"\bdenies\b"
    ]
    
    DISORDERS_STYS = DISORDER_SEMANTIC_TYPES
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 system_initializer=None, kb_resolver=None):
        """
        Initialize with system initializer for resource loading.
        
        Args:
            config: Configuration dictionary
            system_initializer: MetadataSystemInitializer instance
            kb_resolver: KB resolver for semantic validation
        """
        self.config = config or {}
        self.system_initializer = system_initializer
        self.kb_resolver = kb_resolver
        
        # Configuration
        self.similarity_threshold = self.config.get('similarity_threshold', 85)
        self.min_confidence = self.config.get('min_confidence', 0.5)
        
        # Term sets
        self.generic_terms = self.config.get('generic_terms', self.DEFAULT_GENERIC_TERMS)
        self.noise_terms = self.config.get('noise_terms', self.DEFAULT_NOISE_TERMS)
        self.medical_terms = self.generic_terms | self.noise_terms
        
        # False positive patterns
        fp_patterns = self.config.get('false_positive_patterns', self.DEFAULT_FALSE_POSITIVE_PATTERNS)
        self.false_positive_re = [re.compile(p, re.IGNORECASE) for p in fp_patterns]
        
        # Load databases from system initializer
        self.disease_db = self._load_disease_db_from_initializer()
        self.orphanet_data = self._load_orphanet_from_initializer()
        
        # Track demoted entities
        self.demoted_entities = []
        
        logger.info(f"RareDiseaseEnrichment initialized with {len(self.disease_db)} diseases")
        if self.kb_resolver:
            logger.info("KB resolver available for semantic validation")
    
    def canon(self, text: str) -> str:
        """Proxy to cached canonicalization"""
        return _canon(text)
    
    def _load_disease_db_from_initializer(self) -> Dict[str, Dict[str, Any]]:
        """Load disease database from system initializer"""
        disease_db = {}
        
        if not self.system_initializer:
            logger.warning("No system initializer provided")
            return self._get_fallback_diseases()
        
        # Try to get disease lexicon from initializer
        for attr in ['disease_lexicon', 'diseases', 'disease_data']:
            if hasattr(self.system_initializer, attr):
                data = getattr(self.system_initializer, attr)
                if data and isinstance(data, dict):
                    for disease, info in data.items():
                        normalized = self.canon(disease)
                        # Ensure proper structure
                        if isinstance(info, dict):
                            disease_db[normalized] = self._normalize_disease_info(info)
                        else:
                            disease_db[normalized] = {'name': disease, 'confidence': 0.7}
                    logger.info(f"Loaded {len(disease_db)} diseases from initializer.{attr}")
                    break
        
        # Add fallback if empty
        if not disease_db:
            disease_db = self._get_fallback_diseases()
        
        return disease_db
    
    def _load_orphanet_from_initializer(self) -> Dict[str, Dict]:
        """Load Orphanet data from system initializer"""
        orphanet_data = {}
        
        if not self.system_initializer:
            return {}
        
        # Try multiple attribute names
        for attr in ['orphanet_diseases', 'orphanet_data', 'orphanet']:
            if hasattr(self.system_initializer, attr):
                data = getattr(self.system_initializer, attr)
                if data and isinstance(data, dict):
                    orphanet_data = data
                    logger.info(f"Loaded {len(orphanet_data)} Orphanet entries from initializer")
                    break
        
        # Check if nested in resources
        if not orphanet_data and hasattr(self.system_initializer, 'resources'):
            if isinstance(self.system_initializer.resources, dict):
                if 'orphanet' in self.system_initializer.resources:
                    orphanet_data = self.system_initializer.resources['orphanet']
                    logger.info(f"Loaded {len(orphanet_data)} Orphanet entries from resources")
        
        return orphanet_data
    
    def _normalize_disease_info(self, info: Dict) -> Dict:
        """Normalize disease info structure"""
        normalized = info.copy()
        
        # Ensure ORPHA code format
        for key in ['orphacode', 'orpha_code', 'orpha']:
            if key in normalized and normalized[key]:
                code = str(normalized[key]).strip()
                if code and not code.upper().startswith("ORPHA:"):
                    code = f"ORPHA:{code}"
                normalized['orphacode'] = code
                normalized['orpha_code'] = code
                break
        
        # Normalize IDs
        if 'ids' in normalized:
            normalized['ids'] = normalize_ids(normalized['ids'])
        
        return normalized
    
    def _get_fallback_diseases(self) -> Dict[str, Dict]:
        """Get fallback disease entries"""
        fallback = {
            "fabry disease": {
                "name": "Fabry Disease",
                "orphacode": "ORPHA:324",
                "orpha_code": "ORPHA:324",
                "ids": {"ORPHA": "ORPHA:324", "ICD10": "E75.2"},
                "confidence": 0.95,
                "source": "orphanet"
            },
            "gaucher disease": {
                "name": "Gaucher Disease",
                "orphacode": "ORPHA:355",
                "orpha_code": "ORPHA:355",
                "ids": {"ORPHA": "ORPHA:355", "ICD10": "E75.2"},
                "confidence": 0.95,
                "source": "orphanet"
            },
            "pompe disease": {
                "name": "Pompe Disease",
                "orphacode": "ORPHA:365",
                "orpha_code": "ORPHA:365",
                "ids": {"ORPHA": "ORPHA:365", "ICD10": "E74.0"},
                "confidence": 0.95,
                "source": "orphanet"
            }
        }
        
        normalized_fallback = {}
        for disease, info in fallback.items():
            normalized_fallback[self.canon(disease)] = info
        
        return normalized_fallback
    
    def _is_non_disease_from_abbreviation(self, disease: DetectedDisease) -> bool:
        """Check if abbreviation-derived entity is not a disease"""
        name_lower = disease.name.lower()
        
        for indicator in self.NON_DISEASE_INDICATORS:
            if indicator in name_lower:
                return True
        
        if hasattr(disease, 'semantic_tui') and disease.semantic_tui:
            if disease.semantic_tui not in self.DISORDERS_STYS:
                return True
        
        non_disease_patterns = [
            r'\b(CD\d+|HLA-[A-Z]\d*)\b',
            r'\bcomplement\s+(component|system|factor)\s+[C\d]+',
            r'\b(C[0-9]+[a-z]?(?:-[0-9]+)?)\b',
            r'\b(protein|gene|receptor|enzyme|antibody|antigen)\b',
        ]
        
        for pattern in non_disease_patterns:
            if re.search(pattern, name_lower, re.IGNORECASE):
                return True
        
        return False
    
    def postfilter_direct_diseases(self, diseases: List[DetectedDisease]) -> Tuple[List[DetectedDisease], List[Dict]]:
        """Filter diseases through semantic validation"""
        if not self.kb_resolver:
            return diseases, []
        
        kept = []
        demoted = []
        
        for disease in diseases:
            # Skip abbreviation-derived (handled separately)
            if disease.from_abbreviation:
                kept.append(disease)
                continue
            
            try:
                resolution = self.kb_resolver.resolve(disease.name)
            except Exception as e:
                logger.debug(f"KB resolution failed for {disease.name}: {e}")
                resolution = None
            
            if resolution and is_disorder_entity(resolution):
                # Enrich with validated data
                disease.ids = normalize_ids(resolution.get('ids', {}))
                disease.canonical_name = resolution.get('canonical_name', disease.name)
                disease.semantic_type = resolution.get('semantic_type')
                disease.set_primary_id()
                kept.append(disease)
            else:
                # Demote entity
                demoted_entity = {
                    'name': disease.name,
                    'confidence': disease.confidence,
                    'demoted_reason': 'non-disorder-semantic'
                }
                
                if resolution:
                    semantic_type = resolution.get('semantic_type')
                    if semantic_type in {'T005', 'T007', 'T004'}:
                        demoted_entity['reclassified_as'] = 'pathogen'
                    elif semantic_type in {'T116', 'T028'}:
                        demoted_entity['reclassified_as'] = 'biomarker'
                    elif semantic_type in {'T059', 'T060'}:
                        demoted_entity['reclassified_as'] = 'diagnostic_test'
                    elif semantic_type == 'T092':
                        demoted_entity['reclassified_as'] = 'organization'
                
                demoted.append(demoted_entity)
        
        logger.info(f"Semantic filtering: {len(diseases)} -> {len(kept)} diseases ({len(demoted)} demoted)")
        return kept, demoted
    
    def remove_noise(self, text: str, diseases: List[DetectedDisease]) -> List[DetectedDisease]:
        """Remove noise and false positives"""
        filtered = []
        text_lower = text.lower() if text else ""
        
        for disease in diseases:
            canonical = self.canon(disease.name)
            
            # Special handling for abbreviation-derived
            if hasattr(disease, 'from_abbreviation') and disease.from_abbreviation:
                if self._is_non_disease_from_abbreviation(disease):
                    logger.debug(f"Filtered non-disease: {disease.name} (from {disease.from_abbreviation})")
                    continue
            
            # Apply filters
            if canonical in self.medical_terms or len(canonical) < 3 or canonical.isdigit():
                continue
            
            if self._is_false_positive_in_context(disease, text_lower):
                continue
            
            if canonical in {"syndrome", "disease", "disorder"}:
                continue
            
            filtered.append(disease)
        
        logger.info(f"Noise filter: {len(diseases)} -> {len(filtered)} diseases")
        return filtered
    
    def aggregate(self, diseases: List[DetectedDisease]) -> List[DetectedDisease]:
        """Aggregate multiple occurrences"""
        aggregated = {}
        
        for disease in diseases:
            key = self.canon(disease.name)
            
            if key in aggregated:
                existing = aggregated[key]
                existing.positions.extend(disease.positions)
                existing.contexts.extend(disease.contexts)
                existing.occurrences += disease.occurrences
                existing.confidence = max(existing.confidence, disease.confidence)
                existing.matched_terms.extend(disease.matched_terms or [])
                
                # Merge IDs
                if disease.ids:
                    if not existing.ids:
                        existing.ids = {}
                    existing.ids.update(disease.ids)
                
                # Preserve metadata
                if disease.canonical_name and not existing.canonical_name:
                    existing.canonical_name = disease.canonical_name
                if disease.from_abbreviation and not existing.from_abbreviation:
                    existing.from_abbreviation = disease.from_abbreviation
                if disease.semantic_tui and not existing.semantic_tui:
                    existing.semantic_tui = disease.semantic_tui
                
                existing.set_orpha(disease.get_orpha())
            else:
                aggregated[key] = disease
        
        # Deduplicate
        for disease in aggregated.values():
            disease.positions = sorted(set(disease.positions))
            disease.contexts = list(dict.fromkeys(disease.contexts))
            disease.matched_terms = list(dict.fromkeys(disease.matched_terms))
        
        result = list(aggregated.values())
        logger.info(f"Aggregation: {len(diseases)} -> {len(result)} unique diseases")
        return result
    
    def merge_similar(self, diseases: List[DetectedDisease]) -> List[DetectedDisease]:
        """Merge similar diseases using fuzzy matching"""
        if not diseases:
            return []
        
        # Block by first tokens for efficiency
        blocks = defaultdict(list)
        for idx, disease in enumerate(diseases):
            tokens = self.canon(disease.name).split()
            block_key = " ".join(tokens[:2]) if tokens else ""
            blocks[block_key].append(idx)
        
        used = set()
        merged = []
        
        def merge_into(base: DetectedDisease, other: DetectedDisease):
            base.positions.extend(other.positions)
            base.contexts.extend(other.contexts)
            base.occurrences += other.occurrences
            base.matched_terms.extend(other.matched_terms or [])
            
            # Merge IDs
            if other.ids:
                if not base.ids:
                    base.ids = {}
                base.ids.update(other.ids)
            
            # Prefer non-abbreviation names
            if (other.from_abbreviation is None and base.from_abbreviation):
                base.name = other.name
                base.canonical_name = other.canonical_name or other.name
                base.confidence = max(base.confidence, other.confidence)
            elif other.confidence > base.confidence:
                base.name = other.name
                base.confidence = other.confidence
            
            base.set_orpha(other.get_orpha())
        
        # Process blocks
        for block_key, indices in blocks.items():
            # Sort by priority
            def sort_key(i):
                disease = diseases[i]
                is_from_abbrev = bool(disease.from_abbreviation)
                return (is_from_abbrev, -disease.confidence)
            
            indices_sorted = sorted(indices, key=sort_key)
            
            for i in range(len(indices_sorted)):
                if indices_sorted[i] in used:
                    continue
                
                base_disease = diseases[indices_sorted[i]]
                used.add(indices_sorted[i])
                
                # Compare with others in block
                for j in range(i + 1, len(indices_sorted)):
                    if indices_sorted[j] in used:
                        continue
                    
                    candidate = diseases[indices_sorted[j]]
                    similarity = fuzz.token_sort_ratio(
                        self.canon(base_disease.name),
                        self.canon(candidate.name)
                    )
                    
                    if similarity >= self.similarity_threshold:
                        merge_into(base_disease, candidate)
                        used.add(indices_sorted[j])
                        logger.debug(f"Merged '{candidate.name}' into '{base_disease.name}' (sim: {similarity})")
                
                # Deduplicate
                base_disease.positions = sorted(set(base_disease.positions))
                base_disease.contexts = list(dict.fromkeys(base_disease.contexts))
                base_disease.matched_terms = list(dict.fromkeys(base_disease.matched_terms))
                
                merged.append(base_disease)
        
        logger.info(f"Similarity merge: {len(diseases)} -> {len(merged)} diseases")
        return merged
    
    def enrich_with_database(self, diseases: List[DetectedDisease]) -> List[DetectedDisease]:
        """Enrich diseases with database information"""
        enriched = []
        
        for disease in diseases:
            canonical = self.canon(disease.name)
            
            # Look up in disease database
            if canonical in self.disease_db:
                db_info = self.disease_db[canonical]
                
                # Update with database info
                disease.canonical_name = db_info.get('name', disease.name)
                disease.set_orpha(db_info.get('orphacode') or db_info.get('orpha_code'))
                disease.doid = db_info.get('doid', disease.doid)
                
                # Merge IDs
                if db_info.get('ids'):
                    if not disease.ids:
                        disease.ids = {}
                    disease.ids.update(normalize_ids(db_info['ids']))
                else:
                    # Build IDs from fields
                    if not disease.ids:
                        disease.ids = {}
                    if disease.get_orpha():
                        disease.ids['ORPHA'] = disease.get_orpha()
                    if disease.doid:
                        disease.ids['DOID'] = disease.doid
                    for id_key in ['umls', 'mesh', 'snomed', 'icd10']:
                        if db_info.get(id_key):
                            disease.ids[id_key.upper()] = db_info[id_key]
                
                # Boost confidence
                disease.confidence = max(disease.confidence, float(db_info.get('confidence', 0.85)))
                disease.source = db_info.get('source', disease.source)
            
            # Look up in Orphanet data
            if canonical in self.orphanet_data:
                orpha_info = self.orphanet_data[canonical]
                
                disease.set_orpha(disease.get_orpha() or orpha_info.get('orpha_code'))
                
                if orpha_info.get('icd10'):
                    if not disease.ids:
                        disease.ids = {}
                    disease.ids['ICD10'] = orpha_info['icd10']
                
                if not disease.canonical_name:
                    disease.canonical_name = disease.name
                
                if orpha_info.get('synonyms'):
                    disease.matched_terms.extend(orpha_info['synonyms'])
            
            # Set primary ID
            disease.set_primary_id()
            
            # Deduplicate matched terms
            disease.matched_terms = list(dict.fromkeys(disease.matched_terms))
            
            enriched.append(disease)
        
        logger.info(f"Enriched {len(enriched)} diseases with database information")
        return enriched
    
    def normalize_disease_names(self, diseases: List[DetectedDisease]) -> List[DetectedDisease]:
        """Normalize disease names to canonical forms"""
        for disease in diseases:
            if disease.canonical_name:
                disease.name = disease.canonical_name
            else:
                canonical = self.canon(disease.name)
                if canonical in self.disease_db:
                    disease.name = self.disease_db[canonical].get('name', disease.name)
                else:
                    # Standard normalization
                    parts = disease.name.split('-')
                    normalized_parts = []
                    for part in parts:
                        words = part.split()
                        normalized_words = [word.capitalize() for word in words]
                        normalized_parts.append(' '.join(normalized_words))
                    disease.name = '-'.join(normalized_parts)
        
        return diseases
    
    def filter_by_confidence(self, diseases: List[DetectedDisease], 
                           threshold: Optional[float] = None) -> List[DetectedDisease]:
        """Filter by confidence threshold"""
        if threshold is None:
            threshold = self.min_confidence
        
        filtered = []
        for d in diseases:
            # Lower threshold for valid abbreviation-derived diseases
            if (d.from_abbreviation and d.semantic_tui in self.DISORDERS_STYS):
                adjusted_threshold = threshold * 0.9
            else:
                adjusted_threshold = threshold
            
            if d.confidence >= adjusted_threshold:
                filtered.append(d)
        
        logger.info(f"Confidence filter ({threshold}): {len(diseases)} -> {len(filtered)} diseases")
        return filtered
    
    def _is_false_positive_in_context(self, disease: DetectedDisease, text_lower: str) -> bool:
        """Check if disease is false positive based on context"""
        for context in (disease.contexts or []):
            context_lower = context.lower()
            for pattern_re in self.false_positive_re:
                if pattern_re.search(context_lower):
                    return True
        
        if not disease.contexts and text_lower:
            disease_canon = self.canon(disease.name)
            for pattern_re in self.false_positive_re:
                match = pattern_re.search(text_lower)
                if match:
                    start, end = match.span()
                    window = text_lower[max(0, start-50):min(len(text_lower), end+50)]
                    if disease_canon in window:
                        return True
        
        return False
    
    def process(self, text: str, diseases: List[DetectedDisease]) -> List[DetectedDisease]:
        """Main processing pipeline with semantic validation"""
        logger.info(f"Starting disease enrichment pipeline with {len(diseases)} diseases")
        
        # Count abbreviation-derived
        abbrev_count = sum(1 for d in diseases if d.from_abbreviation)
        if abbrev_count > 0:
            logger.info(f"  - {abbrev_count} diseases from abbreviations")
        
        # Step 1: Remove noise
        diseases = self.remove_noise(text, diseases)
        
        # Step 2: Semantic validation
        if self.kb_resolver:
            diseases, demoted = self.postfilter_direct_diseases(diseases)
            if demoted:
                logger.info(f"  - {len(demoted)} entities demoted as non-diseases")
                self.demoted_entities.extend(demoted)
        
        # Step 3: Aggregate
        diseases = self.aggregate(diseases)
        
        # Step 4: Merge similar
        diseases = self.merge_similar(diseases)
        
        # Step 5: Enrich with database
        diseases = self.enrich_with_database(diseases)
        
        # Step 6: Normalize names
        diseases = self.normalize_disease_names(diseases)
        
        # Step 7: Filter by confidence
        diseases = self.filter_by_confidence(diseases)
        
        logger.info(f"Enrichment pipeline complete: {len(diseases)} diseases remaining")
        return diseases
    
    def get_demoted_entities(self) -> List[Dict]:
        """Get list of demoted entities"""
        return self.demoted_entities
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded databases"""
        return {
            "disease_db_size": len(self.disease_db),
            "orphanet_data_size": len(self.orphanet_data),
            "similarity_threshold": self.similarity_threshold,
            "min_confidence": self.min_confidence,
            "fuzzy_backend": FUZZY_BACKEND,
            "has_unidecode": HAS_UNIDECODE,
            "has_kb_resolver": bool(self.kb_resolver),
            "demoted_count": len(self.demoted_entities)
        }