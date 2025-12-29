#!/usr/bin/env python3
"""
Entity Extraction Utilities - v12.0.0
=====================================
Location: corpus_metadata/document_utils/entity_extraction_utils.py
Version: 12.0.0
Last Updated: 2025-12-27

CHANGES IN v12.0.0:
[REMOVED] All abbreviation-related functions
[REMOVED] enrich_abbreviations_with_entity_ids()
[REMOVED] deduplicate_entities_across_types() - abbreviation deduplication
[REMOVED] Abbreviation-specific validation
[UPDATED] Simplified adaptive filtering without abbreviation context

Helper functions, classes, and utilities for entity extraction pipeline.
Includes confidence bands, validation, and deduplication logic.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIDENCE BAND SYSTEM
# ============================================================================

class ConfidenceBand(Enum):
    """
    Confidence bands for entity validation
    
    Replaces hardcoded thresholds like 0.65, 0.75, 0.85 with ranges.
    """
    HIGH = (0.80, 1.0)       # Always keep - high quality detections
    MEDIUM = (0.60, 0.80)    # Keep with validation or IDs
    LOW = (0.40, 0.60)       # Flag for review, keep if pattern-detected
    VERY_LOW = (0.0, 0.40)   # Reject unless special case


def get_confidence_band(confidence: float) -> ConfidenceBand:
    """Determine which confidence band a score falls into"""
    if confidence >= 0.80:
        return ConfidenceBand.HIGH
    elif confidence >= 0.60:
        return ConfidenceBand.MEDIUM
    elif confidence >= 0.40:
        return ConfidenceBand.LOW
    else:
        return ConfidenceBand.VERY_LOW


# ============================================================================
# CONFIGURABLE THRESHOLDS
# ============================================================================

class ThresholdConfig:
    """Centralized threshold configuration with provenance tracking"""
    
    # Fuzzy matching
    FUZZY_MATCH_THRESHOLD = 75
    FUZZY_MATCH_PROVENANCE = "Set to 75 based on analysis showing 85 rejected valid matches"
    
    # Entity-specific minimum confidences
    DRUG_MIN_CONFIDENCE = 0.60
    DISEASE_MIN_CONFIDENCE = 0.55
    
    # Detection method quality indicators
    METHOD_QUALITY = {
        'pattern': 'high',
        'kb': 'high',
        'lexicon': 'medium',
        'ner': 'medium',
        'claude': 'high'
    }


# ============================================================================
# VALIDATION CONSTANTS
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

GENERIC_DISEASE_TERMS = {
    'neuropathy', 'myopathy', 'arthropathy',
    'infection', 'infections', 'inflammation',
    'disorder', 'condition'
}

ALWAYS_KEEP_DISEASES = {
    'pneumocystis jirovecii pneumonia',
    'end-stage renal disease',
    'chronic kidney disease',
    'acute kidney injury',
    'diffuse alveolar hemorrhage',
    'peripheral neuropathy',
    'diabetic neuropathy',
    'glomerulonephritis'
}


# ============================================================================
# EXTRACTION CACHING
# ============================================================================

class ExtractionCache:
    """File-based cache for extraction results to avoid re-processing"""
    
    def __init__(self, cache_dir: Path = None):
        if cache_dir is None:
            cache_dir = Path('./cache/extractions')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Extraction cache initialized: {self.cache_dir}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate file hash for cache key"""
        import hashlib
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def get(self, file_path: Path, version: str) -> Optional[Dict]:
        """Get cached results if available"""
        try:
            file_hash = self._get_file_hash(file_path)
            cache_key = f"{file_hash}_{version}.json"
            cache_file = self.cache_dir / cache_key
            
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                    logger.info(f"Cache hit for {file_path.name}")
                    return cached
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        return None
    
    def set(self, file_path: Path, version: str, results: Dict):
        """Store results in cache"""
        try:
            file_hash = self._get_file_hash(file_path)
            cache_key = f"{file_hash}_{version}.json"
            cache_file = self.cache_dir / cache_key
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            logger.debug(f"Cached results for {file_path.name}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fuzzy_match(term: str, candidates: List[str], threshold: int = None) -> Optional[str]:
    """
    Fuzzy string matching using token sort ratio
    """
    if threshold is None:
        threshold = ThresholdConfig.FUZZY_MATCH_THRESHOLD
    
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        logger.warning("fuzzywuzzy not available, fuzzy matching disabled")
        return None
    
    best_match = None
    best_score = 0
    
    for candidate in candidates:
        score = fuzz.token_sort_ratio(term.lower(), candidate.lower())
        if score > best_score and score >= threshold:
            best_score = score
            best_match = candidate
    
    return best_match


def deduplicate_by_key(entities: List[Dict], keys: Tuple[str, ...]) -> List[Dict]:
    """
    Remove duplicates based on specified keys
    
    Args:
        entities: List of entity dictionaries
        keys: Tuple of key fields to check for duplicates
        
    Returns:
        Deduplicated list with empty entries removed
    """
    if not entities:
        return []
    
    seen = set()
    unique = []
    empty_count = 0
    duplicate_count = 0
    
    for entity in entities:
        # Skip None or empty dicts
        if not entity:
            empty_count += 1
            continue
        
        # Get key values
        key_values = tuple(entity.get(k, '').lower() if entity.get(k) else '' for k in keys)
        
        # Skip entities where ALL key values are empty/None
        if all(not v or (isinstance(v, str) and not v.strip()) for v in key_values):
            empty_count += 1
            logger.debug(f"Filtered empty entity with keys {keys}: {entity.get('name', 'N/A')}")
            continue
        
        # Check for duplicates
        if key_values in seen:
            duplicate_count += 1
            logger.debug(f"Filtered duplicate: {key_values}")
            continue
        
        seen.add(key_values)
        unique.append(entity)
    
    # Log results
    original_count = len(entities)
    final_count = len(unique)
    
    if empty_count > 0 or duplicate_count > 0:
        logger.info(
            f"Deduplicated {keys}: {original_count} → {final_count} "
            f"(removed {empty_count} empty + {duplicate_count} duplicates)"
        )
    else:
        logger.debug(f"Deduplicated {keys}: {original_count} → {final_count}")
    
    return unique


def deduplicate_diseases_preserve_hierarchy(diseases: List[Dict]) -> List[Dict]:
    """
    Deduplicate diseases while preserving parent-child relationships
    
    Example: Keep both "vasculitis" and "ANCA-associated vasculitis"
    """
    if not diseases:
        return []
    
    sorted_diseases = sorted(diseases, key=lambda d: len(d.get('name', '')), reverse=True)
    
    kept = []
    kept_names_lower = set()
    
    for disease in sorted_diseases:
        name = disease.get('name', '').lower()
        
        is_parent_of_kept = any(name in kept_name for kept_name in kept_names_lower if name != kept_name)
        
        if not is_parent_of_kept:
            kept.append(disease)
            kept_names_lower.add(name)
    
    logger.debug(f"Deduplicated diseases with hierarchy: {len(diseases)} → {len(kept)}")
    return kept


# ============================================================================
# SIMPLIFIED ADAPTIVE CONFIDENCE FILTERING
# ============================================================================

def apply_adaptive_confidence_filter(
    candidates: List[Dict], 
    entity_type: str,
    use_claude: bool
) -> List[Dict]:
    """
    Simplified adaptive filtering using confidence bands
    
    Strategy:
    1. HIGH band (≥0.80): Always keep
    2. MEDIUM band (0.60-0.80): Keep if has IDs OR pattern/kb detected OR Claude validated
    3. LOW band (0.40-0.60): Keep only if pattern-detected with IDs
    4. VERY_LOW band (<0.40): Reject
    """
    if not candidates:
        return []
    
    min_confidence = {
        'drug': ThresholdConfig.DRUG_MIN_CONFIDENCE,
        'disease': ThresholdConfig.DISEASE_MIN_CONFIDENCE
    }.get(entity_type, 0.60)
    
    logger.debug(f"Filtering {len(candidates)} {entity_type} candidates (min_confidence={min_confidence})")
    
    filtered = []
    stats = {'high': 0, 'medium': 0, 'low': 0, 'very_low': 0, 'rejected': 0}
    
    for candidate in candidates:
        confidence = candidate.get('confidence', 0.0)
        detection_method = candidate.get('detection_method', 'unknown')
        
        band = get_confidence_band(confidence)
        stats[band.name.lower()] += 1
        
        if candidate.get('claude_validated'):
            filtered.append(candidate)
            continue
        
        has_ids = any(
            candidate.get(id_field) 
            for id_field in ['ORPHA', 'DOID', 'orphacode', 'orpha_code', 
                            'rxcui', 'unii', 'drugbank_id', 'mesh_id',
                            'UMLS', 'SNOMED', 'ICD10']
        ) or bool(candidate.get('all_ids'))
        
        keep = False
        
        if band == ConfidenceBand.HIGH:
            keep = True
        elif band == ConfidenceBand.MEDIUM:
            method_quality = ThresholdConfig.METHOD_QUALITY.get(detection_method, 'low')
            keep = has_ids or method_quality == 'high' or use_claude
        elif band == ConfidenceBand.LOW:
            keep = detection_method == 'pattern' and has_ids
        # VERY_LOW band: reject
        
        if keep:
            filtered.append(candidate)
        else:
            stats['rejected'] += 1
    
    logger.info(f"Adaptive filter: {len(candidates)} → {len(filtered)} {entity_type}s")
    logger.debug(f"  Distribution: HIGH={stats['high']}, MEDIUM={stats['medium']}, "
                f"LOW={stats['low']}, VERY_LOW={stats['very_low']}, REJECTED={stats['rejected']}")
    
    return filtered


# ============================================================================
# VALIDATION & QUALITY ASSESSMENT
# ============================================================================

def validate_extraction_results(results: Dict, config: Dict) -> Dict:
    """
    Validate extraction results and assess quality
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'quality_metrics': {},
        'recommendations': [],
        'assessment': 'Good'
    }
    
    entity_stage = None
    for stage in results.get('pipeline_stages', []):
        if stage['stage'] == 'entities':
            entity_stage = stage['results']
            break
    
    if not entity_stage:
        validation['warnings'].append("No entity extraction stage found")
        validation['is_valid'] = False
        return validation
    
    drugs = entity_stage.get('drugs', [])
    diseases = entity_stage.get('diseases', [])
    extraction_summary = entity_stage.get('extraction_summary', {})
    
    # Quality metrics
    if drugs:
        drug_confidences = [d.get('confidence', 0) for d in drugs]
        validation['quality_metrics']['drug_confidence'] = {
            'avg': round(sum(drug_confidences) / len(drug_confidences), 3),
            'count': len(drugs)
        }
    
    if diseases:
        disease_confidences = [d.get('confidence', 0) for d in diseases]
        validation['quality_metrics']['disease_confidence'] = {
            'avg': round(sum(disease_confidences) / len(disease_confidences), 3),
            'min': round(min(disease_confidences), 2),
            'count': len(diseases)
        }
    
    drugs_with_ids = sum(1 for d in drugs if d.get('rxcui') or d.get('mesh_id'))
    diseases_with_ids = sum(1 for d in diseases if any(
        d.get(id_field) for id_field in ['ORPHA', 'DOID', 'UMLS', 'SNOMED', 'orpha_code', 'doid']
    ))
    
    validation['quality_metrics']['drug_id_coverage'] = round(drugs_with_ids / len(drugs), 2) if drugs else 0
    validation['quality_metrics']['disease_id_coverage'] = round(diseases_with_ids / len(diseases), 2) if diseases else 0
    
    text_length = len(results.get('metadata', {}).get('extracted_text', ''))
    if text_length > 0:
        total_entities = len(drugs) + len(diseases)
        validation['quality_metrics']['entities_per_1k_chars'] = round(total_entities / (text_length / 1000), 2)
    
    validation['quality_metrics']['entity_distribution'] = {
        'drugs': len(drugs),
        'diseases': len(diseases),
        'total': len(drugs) + len(diseases)
    }
    
    disease_semantic_types = defaultdict(int)
    for disease in diseases:
        semantic_type = disease.get('semantic_type', 'unknown')
        disease_semantic_types[semantic_type] += 1
    
    validation['quality_metrics']['semantic_type_distribution'] = dict(disease_semantic_types)
    
    # Warnings
    quality_thresholds = config.get('quality_control', {}).get('thresholds', {})
    
    min_drug_id_coverage = quality_thresholds.get('drug_id_coverage', 0.3)
    if validation['quality_metrics']['drug_id_coverage'] < min_drug_id_coverage:
        validation['warnings'].append(
            f"Low drug ID coverage ({validation['quality_metrics']['drug_id_coverage']:.1%})"
        )
    
    min_disease_id_coverage = quality_thresholds.get('disease_id_coverage', 0.25)
    if validation['quality_metrics']['disease_id_coverage'] < min_disease_id_coverage:
        validation['warnings'].append(
            f"Low disease ID coverage ({validation['quality_metrics']['disease_id_coverage']:.1%})"
        )
    
    # Check for incomplete disease names
    incomplete_diseases = [
        d.get('name') for d in diseases 
        if d.get('name') and (
            d['name'].endswith('-Associated') or 
            d['name'].endswith('-Onset') or
            len(d['name']) < 5
        )
    ]
    
    if incomplete_diseases:
        validation['warnings'].append(
            f"Incomplete disease names detected: {len(incomplete_diseases)} entries"
        )
    
    # Check for symptom contamination
    symptom_diseases = [
        d.get('name', '').lower() for d in diseases
        if any(symptom in d.get('name', '').lower() for symptom in SYMPTOM_KEYWORDS)
        and d.get('name', '').lower() not in ALWAYS_KEEP_DISEASES
    ]
    
    if symptom_diseases:
        validation['warnings'].append(
            f"Potential symptom contamination: {len(symptom_diseases)} entries"
        )
    
    # Overall assessment
    warning_count = len(validation['warnings'])
    
    if warning_count == 0:
        validation['assessment'] = "Excellent - No issues detected"
    elif warning_count <= 2:
        validation['assessment'] = "Good - Minor issues detected"
    elif warning_count <= 4:
        validation['assessment'] = "Acceptable - Several issues detected"
    else:
        validation['assessment'] = "Needs Review - Multiple issues detected"
        validation['is_valid'] = False
    
    if validation['warnings']:
        logger.warning(f"Validation warnings for {results.get('filename')}:")
        for warning in validation['warnings']:
            logger.warning(f"  - {warning}")
    
    logger.info(f"Quality assessment: {validation['assessment']}")
    
    return validation