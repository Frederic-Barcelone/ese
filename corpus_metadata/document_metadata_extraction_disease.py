#!/usr/bin/env python3
"""
Document Disease Metadata Extractor - v12.0.0
=============================================
Location: corpus_metadata/document_metadata_extraction_disease.py
Version: 12.0.0 - REMOVED ABBREVIATION PROCESSING
Last Updated: 2025-12-27

CHANGES IN VERSION 12.0.0:
- REMOVED abbreviation_context parameter from extract()
- REMOVED _process_abbreviation_candidates method
- REMOVED _resolve_context_conflicts method
- REMOVED ABBREVIATION_CORRECTIONS dictionary
- SIMPLIFIED extract() method to focus on direct disease detection
- Kept symptom/finding filtering and ID enrichment
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

# Fix import paths
RareDiseaseDetector = None
create_detector = None
BasicDocumentExtractor = None

try:
    from corpus_metadata.document_utils.rare_disease_disease_detector import RareDiseaseDetector, create_detector
except ImportError as e:
    logger.error(f"Could not import RareDiseaseDetector: {e}")
    RareDiseaseDetector = None
    create_detector = None

# ============================================================================
# ID NORMALIZATION
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

REQUIRED_DISEASE_IDS = (
    "UMLS", "SNOMED", "ICD10", "ICD9", "ORPHA", 
    "DOID", "MESH", "OMIM", "MONDO"
)

PREFERRED_DISEASE_KEYS = ('ORPHA', 'DOID', 'UMLS', 'SNOMED', 'MONDO', 'MESH', 'OMIM', 'ICD10', 'ICD9')

# ============================================================================
# SYMPTOM AND FINDING FILTERS
# ============================================================================

SYMPTOM_KEYWORDS = {
    'pain', 'ache', 'aching',
    'fever', 'fatigue', 'weakness',
    'nausea', 'vomiting', 'diarrhea',
    'cough', 'dyspnea', 'tachycardia',
    'edema', 'swelling', 'rash'
}

FINDING_KEYWORDS = {
    'impairment', 'insufficiency', 'deficiency',
    'elevation', 'reduction', 'decrease', 'increase',
    'hemorrhage', 'bleeding', 'hematuria',
    'proteinuria', 'azotemia', 'uremia',
    'dysfunction', 'abnormality'
}

GENERIC_TERMS = {
    'neuropathy', 'myopathy', 'arthropathy',
    'infection', 'infections', 'inflammation',
    'disorder', 'condition', 'syndrome'
}

ALWAYS_KEEP_DISEASES = {
    'pneumocystis jirovecii pneumonia',
    'end-stage renal disease',
    'chronic kidney disease',
    'acute kidney injury',
    'diffuse alveolar hemorrhage',
    'peripheral neuropathy',
    'diabetic neuropathy',
    'glomerulonephritis',
    'anca-associated vasculitis',
    'microscopic polyangiitis',
    'granulomatosis with polyangiitis',
    'eosinophilic granulomatosis with polyangiitis',
    'myasthenia gravis'
}

STANDALONE_DISEASE_TERMS = {
    'vasculitis', 'nephritis', 'hepatitis', 'arthritis',
    'pneumonia', 'meningitis', 'encephalitis', 'myocarditis',
    'glomerulonephritis', 'pyelonephritis'
}

# ============================================================================
# ID ENRICHMENT MAPPINGS
# ============================================================================

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
        'ICD10': 'N18',
        'SNOMED': '709044004',
        'UMLS': 'C1561643'
    },
    'acute kidney injury': {
        'ICD10': 'N17',
        'SNOMED': '14669001',
        'UMLS': 'C0022660'
    },
    'diffuse alveolar hemorrhage': {
        'ICD10': 'R04.89',
        'SNOMED': '233725003',
        'UMLS': 'C0392106'
    }
}


def normalize_ids(entity: Any) -> Dict[str, Any]:
    """Normalize all ID keys to canonical form from disease entity"""
    normalized = {}
    
    if isinstance(entity, dict):
        items = entity.items()
    else:
        items = []
        for attr in dir(entity):
            if not attr.startswith('_'):
                value = getattr(entity, attr, None)
                if value is not None:
                    items.append((attr, value))
    
    for key, value in items:
        if value is None:
            continue
        
        canonical_key = ID_KEY_MAP.get(key.lower(), None)
        
        if canonical_key:
            if isinstance(value, str) and value.strip():
                if canonical_key == 'ORPHA' and not value.upper().startswith('ORPHA:'):
                    value = f'ORPHA:{value}'
                normalized[canonical_key] = value
    
    return normalized


def classify_semantic_type(disease_name: str, has_ids: bool = False) -> str:
    """Classify entity as disease, symptom, finding, or generic"""
    name_lower = disease_name.lower().strip()
    
    if name_lower in ALWAYS_KEEP_DISEASES:
        return 'disease'
    
    if name_lower in STANDALONE_DISEASE_TERMS:
        return 'disease'
    
    symptom_count = sum(1 for kw in SYMPTOM_KEYWORDS if kw in name_lower)
    finding_count = sum(1 for kw in FINDING_KEYWORDS if kw in name_lower)
    generic_count = sum(1 for kw in GENERIC_TERMS if kw in name_lower)
    
    if symptom_count > 0 and 'disease' not in name_lower and 'syndrome' not in name_lower:
        return 'symptom'
    
    if finding_count > 0 and not has_ids:
        return 'finding'
    
    if generic_count > 0 and not has_ids and len(name_lower.split()) <= 2:
        return 'generic'
    
    return 'disease'


def enrich_disease_ids(disease_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich disease dictionary with standard IDs if available"""
    name_lower = disease_dict.get('name', '').lower().strip()
    
    if name_lower in DISEASE_ID_ENRICHMENT:
        enrichment_data = DISEASE_ID_ENRICHMENT[name_lower]
        
        for id_type, id_value in enrichment_data.items():
            if id_type not in disease_dict or not disease_dict[id_type]:
                disease_dict[id_type] = id_value
                logger.debug(f"Enriched '{name_lower}' with {id_type}: {id_value}")
    
    return disease_dict


# ============================================================================
# Disease Metadata Extractor
# ============================================================================

class DiseaseMetadataExtractor:
    """
    Modern disease detection system with ID normalization and symptom filtering.
    """
    
    def __init__(self, 
                 mode: str = "balanced",
                 use_claude: bool = False,
                 claude_api_key: Optional[str] = None,
                 config_path: Optional[str] = None,
                 verbose: bool = False,
                 filter_symptoms: bool = True):
        """Initialize the disease extractor with modern defaults."""
        self.mode = mode
        self.use_claude = use_claude
        self.verbose = verbose
        self.filter_symptoms = filter_symptoms
        
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
        
        self.file_extractor = None
        if BasicDocumentExtractor:
            try:
                self.file_extractor = BasicDocumentExtractor(config_path)
                logger.info("File extraction enabled via BasicDocumentExtractor")
            except Exception as e:
                logger.warning(f"File extraction not available: {e}")
        
        self.stats = {
            'documents_processed': 0,
            'diseases_detected': 0,
            'diseases_with_ids': 0,
            'symptoms_filtered': 0,
            'findings_filtered': 0,
            'ids_enriched': 0
        }
        
        logger.info(f"DiseaseMetadataExtractor initialized (mode={mode}, claude={use_claude}, filter_symptoms={filter_symptoms})")
    
    def _init_detector(self, mode: str, use_claude: bool, 
                      claude_api_key: Optional[str], config_path: Optional[str]):
        """Initialize the rare disease detector with system initializer"""
        if not RareDiseaseDetector:
            logger.warning("RareDiseaseDetector not found, using fallback detector")
            self.detector = self._create_fallback_detector(mode, config_path)
            return
        
        from corpus_metadata.document_utils.metadata_system_initializer import MetadataSystemInitializer
        system_initializer = MetadataSystemInitializer.get_instance(
            config_path or "corpus_config/config.yaml"
        )
        
        orphanet_db_path = None
        doid_db_path = None
        
        if system_initializer:
            if hasattr(system_initializer, 'get_resource'):
                orphanet_db_path = system_initializer.get_resource('disease_orphanet_db')
                doid_db_path = system_initializer.get_resource('disease_ontology_db')
            
            if not orphanet_db_path and hasattr(system_initializer, 'config'):
                databases = system_initializer.config.get('databases', {})
                paths = system_initializer.config.get('paths', {})
                base_path = paths.get('databases', 'corpus_db')
                
                orphanet_file = databases.get('disease_orphanet')
                if orphanet_file:
                    orphanet_db_path = f"{base_path}/{orphanet_file}"
            
            if not doid_db_path and hasattr(system_initializer, 'config'):
                databases = system_initializer.config.get('databases', {})
                paths = system_initializer.config.get('paths', {})
                base_path = paths.get('databases', 'corpus_db')
                
                doid_file = databases.get('disease_ontology')
                if doid_file:
                    doid_db_path = f"{base_path}/{doid_file}"
        
        logger.info(f"Database paths extracted: Orphanet={orphanet_db_path}, DOID={doid_db_path}")
        
        claude_client = None
        if use_claude:
            claude_client = self._create_claude_client(claude_api_key)
        
        config = {}
        if config_path and Path(config_path).exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Could not load config: {e}")
        
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
    
    def _filter_symptoms_and_findings(self, diseases: List) -> Tuple[List, Dict[str, int]]:
        """Filter out symptoms and clinical findings from disease list"""
        if not self.filter_symptoms:
            return diseases, {'symptoms_filtered': 0, 'findings_filtered': 0, 'total_filtered': 0}
        
        filtered = []
        stats = {'symptoms_filtered': 0, 'findings_filtered': 0, 'total_filtered': 0}
        
        for disease in diseases:
            disease_name = self._get_disease_name(disease)
            
            normalized = normalize_ids(disease)
            has_ids = any(key in normalized for key in REQUIRED_DISEASE_IDS)
            
            semantic_type = classify_semantic_type(disease_name, has_ids)
            
            if isinstance(disease, dict):
                disease['semantic_type'] = semantic_type
            else:
                setattr(disease, 'semantic_type', semantic_type)
            
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
            
            filtered.append(disease)
        
        logger.info(f"Filtered {stats['total_filtered']} non-disease entities "
                   f"({stats['symptoms_filtered']} symptoms, {stats['findings_filtered']} findings)")
        
        return filtered, stats
    
    def _get_disease_name(self, disease: Any) -> str:
        """Get disease name from dict or object"""
        if isinstance(disease, dict):
            return disease.get('name', '')
        return getattr(disease, 'name', '')
    
    def _get_orphacode(self, disease: Any) -> Optional[str]:
        """Get orphacode from disease"""
        if isinstance(disease, dict):
            return disease.get('orphacode') or disease.get('ORPHA') or disease.get('orpha_code')
        return getattr(disease, 'orphacode', None) or getattr(disease, 'ORPHA', None)
    
    def _get_confidence(self, disease: Any) -> float:
        """Get confidence from disease"""
        if isinstance(disease, dict):
            return disease.get('confidence', 0.0)
        return getattr(disease, 'confidence', 0.0)
    
    def _format_disease(self, disease: Any) -> Dict[str, Any]:
        """Format disease for output with normalized IDs"""
        if isinstance(disease, dict):
            base_dict = disease.copy()
        else:
            base_dict = {
                'name': getattr(disease, 'name', ''),
                'confidence': getattr(disease, 'confidence', 0.0),
                'detection_method': getattr(disease, 'detection_method', 'unknown'),
                'occurrences': getattr(disease, 'occurrences', 1),
                'positions': getattr(disease, 'positions', []),
                'matched_terms': getattr(disease, 'matched_terms', []),
                'context': getattr(disease, 'context', ''),
                'semantic_type': getattr(disease, 'semantic_type', 'disease')
            }
        
        # Normalize and surface IDs
        normalized = normalize_ids(disease)
        base_dict['all_ids'] = normalized
        
        # Get primary ID
        primary_id = None
        for key in PREFERRED_DISEASE_KEYS:
            if key in normalized:
                primary_id = f"{key}:{normalized[key]}"
                break
        
        base_dict['primary_id'] = primary_id
        
        # Also include individual normalized IDs at top level
        for key, value in normalized.items():
            if key not in base_dict:
                base_dict[key] = value
        
        return base_dict
    
    def _empty_result(self, error: str = None) -> Dict[str, Any]:
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
            'mode': self.mode
        }
        if error:
            result['error'] = error
        return result
    
    def extract(self, text: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract diseases from text using the optimized detection pipeline.
        
        Args:
            text: Input text to analyze
            filename: Optional filename for tracking
            
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
            
            # Step 2: Filter symptoms and findings
            diseases, filter_stats = self._filter_symptoms_and_findings(diseases)
            self.stats['symptoms_filtered'] += filter_stats['symptoms_filtered']
            self.stats['findings_filtered'] += filter_stats['findings_filtered']
            
            # Step 3: Apply Claude validation if enabled
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
                    'symptoms_filtered': filter_stats['symptoms_filtered'],
                    'findings_filtered': filter_stats['findings_filtered'],
                    'unique': len(set(self._get_disease_name(d) for d in diseases)),
                    'with_orpha': sum(1 for d in diseases if self._get_orphacode(d)),
                    'with_ids': diseases_with_ids,
                    'high_confidence': sum(1 for d in diseases if self._get_confidence(d) >= 0.8)
                },
                'mode': self.mode,
                'filename': filename
            }
        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            return self._empty_result(error=str(e))
    
    def extract_from_file(self, file_path: str, 
                         extraction_mode: str = 'intro') -> Dict[str, Any]:
        """Extract diseases from a document file."""
        if not self.file_extractor:
            logger.error("File extraction not available")
            return self._empty_result(error="File extraction not available")
        
        try:
            text = self.file_extractor.get_extracted_text(file_path, extraction_mode)
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            return self._empty_result(error=str(e))
        
        results = self.extract(text, filename=Path(file_path).name)
        results['extraction_mode'] = extraction_mode
        results['file_path'] = str(file_path)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return {
            **self.stats,
            'detector_stats': self.detector.get_statistics() if hasattr(self.detector, 'get_statistics') else {}
        }
    
    def set_mode(self, mode: str):
        """Change detection mode"""
        self.mode = mode
        if hasattr(self.detector, 'set_mode'):
            self.detector.set_mode(mode)
        logger.info(f"Detection mode changed to: {mode}")
    
    def clear_cache(self):
        """Clear any internal caches"""
        if hasattr(self.detector, 'clear_cache'):
            self.detector.clear_cache()
        logger.info("Cache cleared")