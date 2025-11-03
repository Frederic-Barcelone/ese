#!/usr/bin/env python3
"""
Entity Extraction Utilities - v11.1.2
=====================================
Location: corpus_metadata/document_utils/entity_extraction_utils.py
Version: 11.1.2
Last Updated: 2025-10-14

CHANGES IN v11.1.2:
- FIXED: Empty abbreviations bug - deduplicate_by_key now filters empty entries
- FIXED: Cross-type deduplication now pre-filters empty abbreviations
- IMPROVED: Better logging for filtered entries

Helper functions, classes, and utilities for entity extraction pipeline.
Includes confidence bands, validation, deduplication, and enrichment logic.
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
# CONFIDENCE BAND SYSTEM (NEW IN v11.0.0)
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
# CONFIGURABLE THRESHOLDS (NEW IN v11.0.0)
# ============================================================================

class ThresholdConfig:
    """Centralized threshold configuration with provenance tracking"""
    
    # Fuzzy matching - REDUCED from 85 to 75
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
        'abbreviation': 'medium',
        'claude': 'high'
    }
    
    # Promotion requirements
    PROMOTION_MIN_IDS = 1
    PROMOTION_ALLOW_NO_IDS_IF_HIGH_CONF = True
    PROMOTION_HIGH_CONF_THRESHOLD = 0.85
    
    # Abbreviation context support threshold
    ABBREV_CONTEXT_SUPPORT_MIN = 0.55


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
    
    UPDATED IN v11.0.0: Default threshold reduced from 85 to 75
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
    
    FIXED IN v11.1.2:
    - Now filters out entities with empty/missing key values
    - Special handling for abbreviations (requires both abbr AND expansion)
    - Logs filtered entries for debugging
    
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
        
        # CRITICAL FIX #1: Skip entities where ALL key values are empty/None
        if all(not v or (isinstance(v, str) and not v.strip()) for v in key_values):
            empty_count += 1
            logger.debug(f"Filtered empty entity with keys {keys}: {entity.get('abbreviation', entity.get('name', 'N/A'))}")
            continue
        
        # CRITICAL FIX #2: For abbreviations, BOTH fields must be present
        if keys == ('abbreviation', 'expansion'):
            abbr, exp = key_values
            if not abbr or not str(abbr).strip():
                empty_count += 1
                logger.debug(f"Filtered abbreviation with empty abbreviation field")
                continue
            if not exp or not str(exp).strip():
                empty_count += 1
                logger.debug(f"Filtered abbreviation '{abbr}' with empty expansion")
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
# CROSS-TYPE DEDUPLICATION
# ============================================================================

def deduplicate_entities_across_types(
    abbreviations: List[Dict],
    drugs: List[Dict],
    diseases: List[Dict]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Remove abbreviations whose expansions match detected drugs or diseases
    
    UPDATED IN v11.1.2: Added empty abbreviation filtering
    
    Prevents issues like:
    - AAV → "ANCA-associated vasculitis" appearing as both abbrev and disease
    """
    if not abbreviations:
        return abbreviations, drugs, diseases
    
    # ADDED IN v11.1.2: Filter out empty abbreviations BEFORE processing
    original_count = len(abbreviations)
    abbreviations = [
        a for a in abbreviations 
        if a and a.get('abbreviation') and str(a.get('abbreviation')).strip() 
        and a.get('expansion') and str(a.get('expansion')).strip()
    ]
    
    empty_filtered = original_count - len(abbreviations)
    if empty_filtered > 0:
        logger.info(f"Pre-filtered {empty_filtered} empty abbreviations before cross-type deduplication")
    
    if not abbreviations:
        logger.info("No valid abbreviations after filtering empty entries")
        return [], drugs, diseases
    
    drug_names = {d.get('name', '').lower() for d in drugs if d.get('name')}
    drug_names.update({d.get('normalized_name', '').lower() for d in drugs if d.get('normalized_name')})
    
    disease_names = {d.get('name', '').lower() for d in diseases if d.get('name')}
    disease_names.update({d.get('canonical_name', '').lower() for d in diseases if d.get('canonical_name')})
    
    cleaned_abbreviations = []
    removed_count = 0
    removal_reasons = defaultdict(int)
    
    for abbrev in abbreviations:
        expansion = abbrev.get('expansion', '').lower()
        abbrev_text = abbrev.get('abbreviation', '').lower()
        
        should_remove = False
        reason = None
        
        if expansion in drug_names:
            should_remove = True
            reason = 'expansion_matches_drug'
            removal_reasons['expansion_matches_drug'] += 1
        elif expansion in disease_names:
            should_remove = True
            reason = 'expansion_matches_disease'
            removal_reasons['expansion_matches_disease'] += 1
        elif abbrev_text in disease_names:
            should_remove = True
            reason = 'abbreviation_matches_disease'
            removal_reasons['abbreviation_matches_disease'] += 1
        
        if should_remove:
            removed_count += 1
            logger.debug(
                f"Removed abbreviation '{abbrev.get('abbreviation')}' "
                f"(expansion: '{abbrev.get('expansion')}') - {reason}"
            )
        else:
            cleaned_abbreviations.append(abbrev)
    
    if removed_count > 0:
        logger.info(f"Cross-type deduplication: Removed {removed_count} abbreviations")
        for reason, count in sorted(removal_reasons.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  - {reason}: {count}")
    
    return cleaned_abbreviations, drugs, diseases


# ============================================================================
# SIMPLIFIED ADAPTIVE CONFIDENCE FILTERING (UPDATED IN v11.0.0)
# ============================================================================

def apply_adaptive_confidence_filter(
    candidates: List[Dict], 
    entity_type: str,
    use_claude: bool,
    all_abbreviations: List[Dict] = None
) -> List[Dict]:
    """
    UPDATED IN v11.0.0: Simplified adaptive filtering using confidence bands
    
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
        else:  # VERY_LOW
            if all_abbreviations and entity_type == 'disease':
                entity_name = candidate.get('name', '').lower()
                for abbrev in all_abbreviations:
                    expansion = abbrev.get('expansion', '').lower()
                    if entity_name in expansion or expansion in entity_name:
                        if confidence >= ThresholdConfig.ABBREV_CONTEXT_SUPPORT_MIN:
                            keep = True
                            break
        
        if keep:
            filtered.append(candidate)
        else:
            stats['rejected'] += 1
    
    logger.info(f"Adaptive filter: {len(candidates)} → {len(filtered)} {entity_type}s")
    logger.debug(f"  Distribution: HIGH={stats['high']}, MEDIUM={stats['medium']}, "
                f"LOW={stats['low']}, VERY_LOW={stats['very_low']}, REJECTED={stats['rejected']}")
    
    return filtered


# ============================================================================
# ABBREVIATION ENRICHMENT (UPDATED IN v11.0.0)
# ============================================================================

def enrich_abbreviations_with_entity_ids(
    abbreviations: List[Dict],
    drugs: List[Dict],
    diseases: List[Dict]
) -> Tuple[List[Dict], Dict]:
    """
    Enrich abbreviations with IDs from detected drugs/diseases
    
    UPDATED IN v11.0.0: Fuzzy threshold reduced from 85 to 75
    """
    if not abbreviations:
        return abbreviations, {}
    
    start_time = time.time()
    
    drug_index = {}
    for drug in drugs:
        name = drug.get('name', '') or drug.get('normalized_name', '')
        if name:
            drug_index[name.lower()] = drug
    
    disease_index = {}
    for disease in diseases:
        name = disease.get('name', '') or disease.get('canonical_name', '')
        if name:
            disease_index[name.lower()] = disease
    
    stats = {
        'total_abbreviations': len(abbreviations),
        'drugs_enriched': 0,
        'diseases_enriched': 0,
        'total_enriched': 0,
        'id_types_added': defaultdict(int),
        'exact_matches': 0,
        'fuzzy_matches': 0,
        'no_match': 0
    }
    
    enrichment_details = []
    enriched_abbreviations = []
    
    logger.info(f"Starting abbreviation enrichment: {len(abbreviations)} abbreviations, "
                f"{len(drugs)} drugs, {len(diseases)} diseases")
    
    for abbrev in abbreviations:
        abbreviation_text = abbrev.get('abbreviation', '')
        expansion = abbrev.get('expansion', '')
        
        if not expansion:
            enriched_abbreviations.append(abbrev)
            stats['no_match'] += 1
            continue
        
        expansion_lower = expansion.lower()
        enriched = False
        
        # Try drug match
        matching_drug = None
        match_type = None
        
        if expansion_lower in drug_index:
            matching_drug = drug_index[expansion_lower]
            match_type = 'exact'
            stats['exact_matches'] += 1
        else:
            fuzzy_result = fuzzy_match(expansion, list(drug_index.keys()), 
                                      threshold=ThresholdConfig.FUZZY_MATCH_THRESHOLD)
            if fuzzy_result:
                matching_drug = drug_index[fuzzy_result]
                match_type = 'fuzzy'
                stats['fuzzy_matches'] += 1
        
        if matching_drug:
            if 'metadata' not in abbrev:
                abbrev['metadata'] = {}
            
            drug_id_fields = ['rxcui', 'unii', 'drugbank_id', 'mesh_id', 'atc_code']
            ids_copied = []
            
            for id_field in drug_id_fields:
                if matching_drug.get(id_field):
                    abbrev['metadata'][id_field] = matching_drug[id_field]
                    ids_copied.append(id_field)
                    stats['id_types_added'][id_field] += 1
            
            abbrev['metadata']['enriched_from'] = 'direct_detection'
            abbrev['metadata']['match_type'] = match_type
            
            if ids_copied:
                stats['drugs_enriched'] += 1
                stats['total_enriched'] += 1
                enriched = True
                
                enrichment_details.append({
                    'abbreviation': abbreviation_text,
                    'expansion': expansion,
                    'type': 'drug',
                    'match_type': match_type,
                    'ids_copied': ids_copied
                })
        
        # Try disease match if not enriched
        if not enriched:
            matching_disease = None
            
            if expansion_lower in disease_index:
                matching_disease = disease_index[expansion_lower]
                match_type = 'exact'
                stats['exact_matches'] += 1
            else:
                fuzzy_result = fuzzy_match(expansion, list(disease_index.keys()),
                                          threshold=ThresholdConfig.FUZZY_MATCH_THRESHOLD)
                if fuzzy_result:
                    matching_disease = disease_index[fuzzy_result]
                    match_type = 'fuzzy'
                    stats['fuzzy_matches'] += 1
            
            if matching_disease:
                if 'metadata' not in abbrev:
                    abbrev['metadata'] = {}
                
                disease_id_fields = ['orpha_code', 'orphacode', 'doid', 'umls_cui', 'cui', 
                                   'snomed_ct', 'mesh_id', 'omim_id', 'mondo_id', 'icd10', 'icd9']
                ids_copied = []
                
                for id_field in disease_id_fields:
                    if matching_disease.get(id_field):
                        abbrev['metadata'][id_field] = matching_disease[id_field]
                        ids_copied.append(id_field)
                        stats['id_types_added'][id_field] += 1
                
                if matching_disease.get('all_ids'):
                    for key, value in matching_disease['all_ids'].items():
                        if value and key not in abbrev['metadata']:
                            abbrev['metadata'][key] = value
                            ids_copied.append(key)
                            stats['id_types_added'][key] += 1
                
                abbrev['metadata']['enriched_from'] = 'direct_detection'
                abbrev['metadata']['match_type'] = match_type
                
                if ids_copied:
                    stats['diseases_enriched'] += 1
                    stats['total_enriched'] += 1
                    enriched = True
                    
                    enrichment_details.append({
                        'abbreviation': abbreviation_text,
                        'expansion': expansion,
                        'type': 'disease',
                        'match_type': match_type,
                        'ids_copied': ids_copied
                    })
        
        if not enriched:
            stats['no_match'] += 1
        
        enriched_abbreviations.append(abbrev)
    
    logger.info("=" * 80)
    logger.info("ENRICHMENT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total abbreviations processed: {stats['total_abbreviations']}")
    logger.info(f"Successfully enriched: {stats['total_enriched']} "
                f"({stats['total_enriched']/stats['total_abbreviations']*100:.1f}%)")
    logger.info(f"  - Drugs enriched: {stats['drugs_enriched']}")
    logger.info(f"  - Diseases enriched: {stats['diseases_enriched']}")
    logger.info(f"Match types:")
    logger.info(f"  - Exact matches: {stats['exact_matches']}")
    logger.info(f"  - Fuzzy matches: {stats['fuzzy_matches']}")
    logger.info(f"  - No match: {stats['no_match']}")
    
    if enrichment_details:
        logger.info(f"\nEnrichment details (first 10):")
        for detail in enrichment_details[:10]:
            logger.info(f"  ✓ {detail['abbreviation']} → {detail['expansion']}")
            logger.info(f"    Type: {detail['type']}, Match: {detail['match_type']}, "
                       f"IDs: {', '.join(detail['ids_copied'])}")
    
    logger.info("=" * 80)
    logger.info(f"Enrichment completed in {time.time() - start_time:.2f}s")
    
    return enriched_abbreviations, dict(stats)


# ============================================================================
# VALIDATION & QUALITY ASSESSMENT
# ============================================================================

def validate_extraction_results(results: Dict, config: Dict) -> Dict:
    """
    Validate extraction results and assess quality
    
    UPDATED IN v11.0.0: Updated thresholds for validation
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
    abbreviations = entity_stage.get('abbreviations', [])
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
        total_entities = len(drugs) + len(diseases) + len(abbreviations)
        validation['quality_metrics']['entities_per_1k_chars'] = round(total_entities / (text_length / 1000), 2)
    
    validation['quality_metrics']['entity_distribution'] = {
        'abbreviations': len(abbreviations),
        'drugs': len(drugs),
        'diseases': len(diseases),
        'total': len(abbreviations) + len(drugs) + len(diseases)
    }
    
    promotion_rate = extraction_summary.get('promotion_rate', 0)
    validation['quality_metrics']['promotion_rate'] = round(promotion_rate, 2)
    
    disease_semantic_types = defaultdict(int)
    for disease in diseases:
        semantic_type = disease.get('semantic_type', 'unknown')
        disease_semantic_types[semantic_type] += 1
    
    validation['quality_metrics']['semantic_type_distribution'] = dict(disease_semantic_types)
    
    # Warnings
    quality_thresholds = config.get('quality_control', {}).get('thresholds', {})
    
    min_promotion = quality_thresholds.get('promotion_rate', 0.02)
    if promotion_rate < min_promotion:
        validation['warnings'].append(
            f"Low promotion rate ({promotion_rate:.1%}) - abbreviations may lack identifiers"
        )
    
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
        validation['recommendations'].append("Review abbreviation expansion logic")
    
    # Check for abbreviations extracted as diseases
    abbrev_as_diseases = [
        d.get('name') for d in diseases
        if d.get('name') and len(d['name']) <= 5 and d['name'].isupper()
    ]
    if abbrev_as_diseases:
        validation['warnings'].append(
            f"Abbreviations extracted as diseases: {len(abbrev_as_diseases)} entries"
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