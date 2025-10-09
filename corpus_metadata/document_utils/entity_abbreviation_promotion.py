#!/usr/bin/env python3
"""
Abbreviation to Entity Promotion Logic with ID-Gating
======================================================
Location: corpus_metadata/document_utils/entity_abbreviation_promotion.py

This module handles the ID-gated promotion of abbreviations to drugs/diseases.
Only abbreviations that resolve to entities with valid medical IDs get promoted.
All other abbreviations are kept for context.

Version: 6.1.0 - Enhanced with better ID detection and config integration

CHANGES IN v6.1.0:
- Fixed promotion to properly pass config to load_config_from_yaml
- Enhanced ID detection to check all metadata fields thoroughly
- Improved handling of IDs from enrichment (rxcui, mesh_id, orpha_code, doid)
- Better logging for debugging promotion failures
- Fixed config parameter passing in process_abbreviation_candidates

CHANGES IN v6.0.0:
- Added configurable min_ids_required (supports lenient promotion)
- Added support for partial ID sets via allow_partial_ids
- Integrated with config.yaml promotion settings
- Enhanced ID field matching for enriched abbreviations
- Improved logging for promotion decisions
- Better handling of metadata ID fields from enrichment
"""

from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field

# ============================================================================
# CENTRALIZED LOGGING
# ============================================================================
try:
    from corpus_metadata.document_utils.metadata_logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

# ============================================================================
# IMPORT FROM SHARED MODULES
# ============================================================================
try:
    from corpus_metadata.document_utils.abbreviation_types import (
        AbbreviationCandidate,
        ValidationStatus,
        SourceType
    )
    TYPES_AVAILABLE = True
except ImportError:
    logger.warning("Shared types not available, using local definitions")
    TYPES_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PromotionConfig:
    """Configuration for abbreviation promotion"""
    
    # Minimum number of IDs required for promotion
    # 1 = lenient (any single ID), 2 = moderate, 3+ = strict
    min_ids_required: int = 1
    
    # Allow promotion with partial ID coverage
    allow_partial_ids: bool = True
    
    # Confidence boost for promoted entities
    promotion_confidence_boost: float = 0.05
    
    # Enable detailed promotion logging
    log_promotion_details: bool = True
    
    # Required ID types for promotion (if allow_partial_ids=False, need ALL)
    REQUIRED_DISEASE_IDS: Tuple[str, ...] = (
        "UMLS", "SNOMED", "ICD10", "ICD9", "ORPHA", 
        "DOID", "MESH", "OMIM", "MONDO"
    )
    
    REQUIRED_DRUG_IDS: Tuple[str, ...] = (
        "RxCUI", "UNII", "DrugBank", "MESH", 
        "ChEBI", "ATC", "NDC"
    )
    
    # Preferred ID keys for deduplication (in priority order)
    PREFERRED_DRUG_KEYS: Tuple[str, ...] = (
        'RxCUI', 'MESH', 'ATC', 'DrugBank', 'UNII', 'ChEBI', 'NDC'
    )
    
    PREFERRED_DISEASE_KEYS: Tuple[str, ...] = (
        'ORPHA', 'DOID', 'SNOMED', 'UMLS', 'MONDO', 'MESH', 'ICD10', 'OMIM', 'ICD9'
    )
    
    # UMLS semantic types for disorders
    DISORDER_SEMANTIC_TYPES: Set[str] = field(default_factory=lambda: {
        'T019', 'T020', 'T037', 'T046', 'T047', 
        'T048', 'T049', 'T050', 'T190', 'T191'
    })
    
    # ID key normalization map (handles various input formats)
    # Maps all possible variations to canonical form
    ID_KEY_MAP: Dict[str, str] = field(default_factory=lambda: {
        # UMLS variations
        'umls': 'UMLS', 'umls_cui': 'UMLS', 'cui': 'UMLS',
        
        # SNOMED variations
        'snomed': 'SNOMED', 'snomed_ct': 'SNOMED', 'snomedct': 'SNOMED',
        
        # ICD variations
        'icd10': 'ICD10', 'icd-10': 'ICD10', 'icd_10': 'ICD10',
        'icd9': 'ICD9', 'icd-9': 'ICD9', 'icd_9': 'ICD9',
        
        # Orphanet variations (CRITICAL for enrichment)
        'orpha': 'ORPHA', 'orphanet': 'ORPHA', 'orpha_code': 'ORPHA', 
        'orphacode': 'ORPHA',
        
        # Disease Ontology variations (CRITICAL for enrichment)
        'doid': 'DOID', 'do_id': 'DOID',
        
        # Other disease IDs
        'mondo': 'MONDO', 'mondo_id': 'MONDO',
        'mesh': 'MESH', 'mesh_id': 'MESH',
        'omim': 'OMIM', 'omim_id': 'OMIM',
        
        # Drug ID variations (CRITICAL for enrichment)
        'rxcui': 'RxCUI', 'rxnorm': 'RxCUI',
        'unii': 'UNII', 'fda_unii': 'UNII',
        'drugbank': 'DrugBank', 'drugbank_id': 'DrugBank',
        'chebi': 'ChEBI', 'chebi_id': 'ChEBI',
        'atc': 'ATC', 'atc_code': 'ATC',
        'ndc': 'NDC', 'ndc_code': 'NDC',
    })

# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

def load_config_from_yaml(config_dict: Optional[Dict[str, Any]] = None) -> PromotionConfig:
    """
    Load promotion configuration from config dictionary
    
    Args:
        config_dict: Configuration dictionary from config.yaml
        
    Returns:
        PromotionConfig with settings from config or defaults
    """
    config = PromotionConfig()
    
    if config_dict and 'promotion' in config_dict:
        promotion_settings = config_dict['promotion']
        
        # Load configurable settings
        config.min_ids_required = promotion_settings.get('min_ids_required', 1)
        config.allow_partial_ids = promotion_settings.get('allow_partial_ids', True)
        config.promotion_confidence_boost = promotion_settings.get('promotion_confidence_boost', 0.05)
        config.log_promotion_details = promotion_settings.get('log_promotion_details', True)
        
        # Load preferred ID lists if provided
        if 'preferred_drug_ids' in promotion_settings:
            config.PREFERRED_DRUG_KEYS = tuple(promotion_settings['preferred_drug_ids'])
        
        if 'preferred_disease_ids' in promotion_settings:
            config.PREFERRED_DISEASE_KEYS = tuple(promotion_settings['preferred_disease_ids'])
        
        logger.info(f"Loaded promotion config: min_ids={config.min_ids_required}, "
                   f"allow_partial={config.allow_partial_ids}, "
                   f"confidence_boost={config.promotion_confidence_boost}")
    else:
        logger.debug("Using default promotion configuration (min_ids=1, lenient)")
    
    return config

# Initialize global config with defaults
config = PromotionConfig()

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def normalize_ids(ids: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize all ID keys to canonical form
    
    This is critical for matching IDs from enrichment which may use
    lowercase keys like 'rxcui', 'mesh_id', 'orpha_code', 'doid'
    
    Args:
        ids: Dictionary of identifier keys and values
        
    Returns:
        Dictionary with normalized keys
    """
    if not ids:
        return {}
    
    normalized = {}
    for key, value in ids.items():
        # Skip empty or None values
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        
        # Normalize key using the map
        canonical_key = config.ID_KEY_MAP.get(key.lower(), key)
        
        # Clean value
        clean_value = value.strip() if isinstance(value, str) else value
        
        # Store with canonical key
        normalized[canonical_key] = clean_value
    
    return normalized

def has_required_id(entity: Dict[str, Any], entity_type: str) -> Tuple[bool, List[str]]:
    """
    Check if entity has sufficient identifiers for promotion
    
    NEW IN v6.1.0: Enhanced to thoroughly check all possible ID locations:
    - Root level fields
    - metadata dictionary
    - Handles lowercase field names from enrichment
    
    Args:
        entity: Entity dictionary with potential IDs
        entity_type: "Drug" or "Disease"
        
    Returns:
        Tuple of (has_sufficient_ids, list_of_found_ids)
    """
    required_keys = config.REQUIRED_DRUG_IDS if entity_type == "Drug" else config.REQUIRED_DISEASE_IDS
    
    # Collect all IDs from multiple sources
    all_ids = {}
    
    # 1. Check root level fields
    root_ids = normalize_ids(entity)
    all_ids.update(root_ids)
    
    # 2. Check metadata field (where enrichment puts IDs)
    if 'metadata' in entity and isinstance(entity['metadata'], dict):
        metadata_ids = normalize_ids(entity['metadata'])
        all_ids.update(metadata_ids)
    
    # 3. Find which required IDs are present
    found_ids = []
    for key in required_keys:
        if key in all_ids and all_ids[key]:
            found_ids.append(key)
    
    # Check if we have enough IDs
    has_sufficient = len(found_ids) >= config.min_ids_required
    
    # Debug logging
    if config.log_promotion_details:
        if not has_sufficient:
            logger.debug(f"ID check for {entity_type}: found {found_ids} "
                        f"({len(found_ids)} IDs), required {config.min_ids_required}")
        else:
            logger.debug(f"ID check for {entity_type}: ✓ found {found_ids}")
    
    return has_sufficient, found_ids

def get_entity_key(entity: Dict[str, Any], entity_type: str) -> Tuple:
    """
    Generate unique key for entity deduplication
    
    Args:
        entity: Entity dictionary
        entity_type: "Drug" or "Disease"
        
    Returns:
        Tuple representing unique entity key
    """
    # Collect all IDs from both root and metadata
    all_ids = {}
    
    # Root level IDs
    all_ids.update(normalize_ids(entity))
    
    # Metadata IDs
    if 'metadata' in entity and isinstance(entity['metadata'], dict):
        all_ids.update(normalize_ids(entity['metadata']))
    
    preferred_keys = config.PREFERRED_DRUG_KEYS if entity_type == "Drug" else config.PREFERRED_DISEASE_KEYS
    
    # Try preferred IDs first (in priority order)
    for key in preferred_keys:
        if key in all_ids and all_ids[key]:
            return (entity_type, key, str(all_ids[key]))
    
    # Fall back to name-based key
    name_field = f"{entity_type.lower()}_name"
    name = entity.get(name_field, entity.get('name', entity.get('normalized_name', '')))
    
    if name:
        return (entity_type, 'name', name.lower().strip())
    
    # Last resort: use abbreviation if available
    abbrev = entity.get('source_abbreviation', '')
    if abbrev:
        return (entity_type, 'abbrev', abbrev.upper())
    
    # Ultimate fallback
    return (entity_type, 'unknown', str(id(entity)))

# ============================================================================
# PROMOTION FUNCTIONS
# ============================================================================

def promote_abbreviation_to_entity(
    candidate: 'AbbreviationCandidate',
    entity_type: str
) -> Optional[Dict[str, Any]]:
    """
    Convert an abbreviation candidate to an entity if it has required IDs.
    
    NEW IN v6.1.0: Better handling of enriched abbreviations with IDs in metadata
    
    Args:
        candidate: AbbreviationCandidate with potential entity information
        entity_type: "Drug" or "Disease"
        
    Returns:
        Entity dict if promotable, None otherwise
    """
    # Check if candidate has any metadata at all
    if not hasattr(candidate, 'metadata') or not candidate.metadata:
        if config.log_promotion_details:
            logger.debug(f"Not promoting {candidate.abbreviation}: No metadata")
        return None
    
    # Build entity from candidate
    name_field = f"{entity_type.lower()}_name"
    entity = {
        name_field: candidate.expansion,
        'normalized_name': candidate.expansion.lower(),
        'confidence': min(candidate.confidence + config.promotion_confidence_boost, 1.0),
        'occurrences': 1,  # Will be counted properly later
        'detection_method': 'abbreviation_promotion',
        'source': 'from_abbreviation',
        'source_abbreviation': candidate.abbreviation,
        'context_type': candidate.context_type,
        'domain_context': getattr(candidate, 'domain_context', None),
        'metadata': {}
    }
    
    # Copy ALL metadata fields to entity
    # This is critical for IDs added by enrichment
    for key, value in candidate.metadata.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        
        # Get canonical form of key
        canonical_key = config.ID_KEY_MAP.get(key.lower(), key)
        
        # If it's a recognized ID type, add to both root and metadata
        if canonical_key in config.ID_KEY_MAP.values():
            entity[canonical_key] = value
            entity['metadata'][canonical_key] = value
        else:
            # Non-ID metadata (semantic_types, lexicon_resolved, etc.)
            entity['metadata'][key] = value
    
    # Handle semantic types for disease validation
    if entity_type == "Disease":
        semantic_types = candidate.metadata.get('semantic_types', [])
        if semantic_types:
            entity['semantic_types'] = semantic_types
            entity['metadata']['semantic_types'] = semantic_types
            
            # Verify semantic types match disease
            if not any(st in config.DISORDER_SEMANTIC_TYPES for st in semantic_types):
                if config.log_promotion_details:
                    logger.debug(f"Not promoting {candidate.abbreviation}: "
                               f"Semantic types {semantic_types} don't match Disease")
                return None
    
    # CRITICAL: Check if entity has sufficient IDs for promotion
    has_ids, found_ids = has_required_id(entity, entity_type)
    
    if not has_ids:
        if config.log_promotion_details:
            logger.debug(f"Not promoting {candidate.abbreviation} to {entity_type}: "
                        f"Only {len(found_ids)} IDs found ({found_ids}), "
                        f"need {config.min_ids_required}")
        return None
    
    # Log successful promotion
    if config.log_promotion_details:
        logger.debug(f"✓ Promoting {candidate.abbreviation} → {entity_type}: "
                    f"{candidate.expansion} (IDs: {found_ids})")
    
    return entity

def process_abbreviation_candidates(
    candidates: List['AbbreviationCandidate'],
    direct_drugs: List[Dict],
    direct_diseases: List[Dict],
    promotion_config: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Process abbreviation candidates for promotion to entities.
    
    NEW IN v6.1.0: Fixed to properly use promotion_config parameter
    
    Args:
        candidates: List of AbbreviationCandidate objects
        direct_drugs: Drugs found by direct extraction
        direct_diseases: Diseases found by direct extraction
        promotion_config: Optional config dictionary from config.yaml
        
    Returns:
        Tuple of (remaining_abbreviations, all_drugs, all_diseases, promotion_links)
    """
    global config
    
    # Load configuration if provided (FIXED IN v6.1.0)
    if promotion_config:
        config = load_config_from_yaml(promotion_config)
        logger.info(f"Using custom promotion config: min_ids={config.min_ids_required}")
    else:
        logger.info(f"Using default promotion config: min_ids={config.min_ids_required}")
    
    remaining_abbreviations = []
    promotion_links = []
    
    # Index existing entities by unique key
    drug_index = {get_entity_key(d, "Drug"): d for d in direct_drugs}
    disease_index = {get_entity_key(d, "Disease"): d for d in direct_diseases}
    
    logger.debug(f"Indexed {len(drug_index)} unique drugs and {len(disease_index)} unique diseases")
    
    # Track promotion statistics
    promotion_stats = {
        'total_candidates': len(candidates),
        'promoted_to_drug': 0,
        'promoted_to_disease': 0,
        'merged_with_existing_drug': 0,
        'merged_with_existing_disease': 0,
        'insufficient_ids': 0,
        'wrong_semantic_type': 0,
        'no_context': 0,
        'no_metadata': 0,
        'kept_as_abbrev': 0
    }
    
    for candidate in candidates:
        promoted = False
        
        # Check for metadata first
        if not hasattr(candidate, 'metadata') or not candidate.metadata:
            promotion_stats['no_metadata'] += 1
            if config.log_promotion_details:
                abbrev_text = candidate.abbreviation if hasattr(candidate, 'abbreviation') else candidate.get('abbreviation', 'unknown')
                logger.debug(f"Skipping {abbrev_text}: No metadata")
        else:
            # Determine entity type from context
            entity_type = None
            
            if hasattr(candidate, 'context_type') and candidate.context_type:
                if candidate.context_type in ['drug', 'pharmaceutical']:
                    entity_type = "Drug"
                elif candidate.context_type in ['disease', 'disorder']:
                    entity_type = "Disease"
            
            # If no context, try semantic types
            if not entity_type:
                semantic_types = candidate.metadata.get('semantic_types', [])
                if semantic_types:
                    if any(st in config.DISORDER_SEMANTIC_TYPES for st in semantic_types):
                        entity_type = "Disease"
                    else:
                        entity_type = "Drug"
            
            if not entity_type:
                promotion_stats['no_context'] += 1
                if config.log_promotion_details:
                    logger.debug(f"Cannot determine entity type for {candidate.abbreviation}")
            else:
                # Attempt promotion
                entity = promote_abbreviation_to_entity(candidate, entity_type)
                
                if entity:
                    # Check for duplicates
                    index = drug_index if entity_type == "Drug" else disease_index
                    key = get_entity_key(entity, entity_type)
                    
                    if key in index:
                        # Merge with existing entity
                        _merge_with_existing(index[key], entity, candidate.abbreviation)
                        promotion_stats[f'merged_with_existing_{entity_type.lower()}'] += 1
                        
                        if config.log_promotion_details:
                            logger.debug(f"Merged {candidate.abbreviation} with existing {entity_type}")
                    else:
                        # Add as new entity
                        index[key] = entity
                        if config.log_promotion_details:
                            logger.debug(f"Added new {entity_type} from {candidate.abbreviation}")
                    
                    # Track promotion
                    has_ids, found_ids = has_required_id(entity, entity_type)
                    promotion_links.append({
                        'abbreviation': candidate.abbreviation,
                        'expansion': candidate.expansion,
                        'entity_type': entity_type,
                        'entity_name': entity['normalized_name'],
                        'confidence': entity['confidence'],
                        'ids_found': found_ids
                    })
                    
                    promoted = True
                    promotion_stats[f'promoted_to_{entity_type.lower()}'] += 1
                    
                    logger.info(f"Promoted {candidate.abbreviation} → {entity_type}: "
                              f"{candidate.expansion} (IDs: {found_ids})")
                else:
                    promotion_stats['insufficient_ids'] += 1
        
        if not promoted:
            # Keep as abbreviation
            promotion_stats['kept_as_abbrev'] += 1
            
            # Convert to dict
            if hasattr(candidate, 'to_dict'):
                abbrev_dict = candidate.to_dict()
            else:
                abbrev_dict = {
                    'abbreviation': getattr(candidate, 'abbreviation', ''),
                    'expansion': getattr(candidate, 'expansion', ''),
                    'confidence': getattr(candidate, 'confidence', 0.0),
                    'context_type': getattr(candidate, 'context_type', None),
                    'source': getattr(candidate, 'source', None),
                    'metadata': getattr(candidate, 'metadata', {})
                }
            
            remaining_abbreviations.append(abbrev_dict)
    
    # Convert indexes back to lists
    all_drugs = list(drug_index.values())
    all_diseases = list(disease_index.values())
    
    # Log detailed summary
    logger.info("=" * 80)
    logger.info("PROMOTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  - min_ids_required: {config.min_ids_required}")
    logger.info(f"  - allow_partial_ids: {config.allow_partial_ids}")
    logger.info(f"  - confidence_boost: {config.promotion_confidence_boost}")
    logger.info("")
    logger.info(f"Input:")
    logger.info(f"  - Total candidates: {promotion_stats['total_candidates']}")
    logger.info(f"  - Direct drugs: {len(direct_drugs)}")
    logger.info(f"  - Direct diseases: {len(direct_diseases)}")
    logger.info("")
    logger.info(f"Promotion Results:")
    logger.info(f"  - Promoted to drugs: {promotion_stats['promoted_to_drug']}")
    logger.info(f"  - Promoted to diseases: {promotion_stats['promoted_to_disease']}")
    logger.info(f"  - Merged with existing drugs: {promotion_stats['merged_with_existing_drug']}")
    logger.info(f"  - Merged with existing diseases: {promotion_stats['merged_with_existing_disease']}")
    logger.info(f"  - Kept as abbreviations: {promotion_stats['kept_as_abbrev']}")
    logger.info("")
    logger.info(f"Rejection Reasons:")
    logger.info(f"  - Insufficient IDs: {promotion_stats['insufficient_ids']}")
    logger.info(f"  - No metadata: {promotion_stats['no_metadata']}")
    logger.info(f"  - No context: {promotion_stats['no_context']}")
    logger.info(f"  - Wrong semantic type: {promotion_stats['wrong_semantic_type']}")
    logger.info("")
    logger.info(f"Final Counts:")
    logger.info(f"  - Drugs: {len(all_drugs)} (was {len(direct_drugs)})")
    logger.info(f"  - Diseases: {len(all_diseases)} (was {len(direct_diseases)})")
    logger.info(f"  - Abbreviations: {len(remaining_abbreviations)}")
    logger.info("=" * 80)
    
    return remaining_abbreviations, all_drugs, all_diseases, promotion_links

def _merge_with_existing(existing: Dict, new_entity: Dict, abbrev: str):
    """
    Merge new entity data with existing entity
    
    Args:
        existing: Existing entity dictionary (modified in place)
        new_entity: New entity to merge in
        abbrev: Abbreviation that was promoted
    """
    # Update source to indicate multiple origins
    if existing.get('source') != 'from_abbreviation':
        existing['source'] = 'direct+abbreviation'
    
    # Update occurrences
    existing['occurrences'] = existing.get('occurrences', 1) + new_entity.get('occurrences', 1)
    
    # Track which abbreviations contributed
    abbr_list = existing.setdefault('from_abbr_list', [])
    if abbrev not in abbr_list:
        abbr_list.append(abbrev)
    
    # Merge IDs (add any new IDs not already present)
    # Check both root level and metadata
    new_ids = {}
    new_ids.update(normalize_ids(new_entity))
    if 'metadata' in new_entity and isinstance(new_entity['metadata'], dict):
        new_ids.update(normalize_ids(new_entity['metadata']))
    
    existing_ids = {}
    existing_ids.update(normalize_ids(existing))
    if 'metadata' in existing and isinstance(existing['metadata'], dict):
        existing_ids.update(normalize_ids(existing['metadata']))
    
    for key, value in new_ids.items():
        if key not in existing_ids:
            existing[key] = value
            # Also add to metadata if it exists
            if 'metadata' in existing and isinstance(existing['metadata'], dict):
                existing['metadata'][key] = value
            
            if config.log_promotion_details:
                logger.debug(f"Added {key}={value} to existing entity from abbreviation {abbrev}")
    
    # Update confidence to maximum
    if 'confidence' in new_entity:
        existing['confidence'] = max(
            existing.get('confidence', 0),
            new_entity['confidence']
        )

# ============================================================================
# MAIN EXPORT FUNCTIONS
# ============================================================================

__all__ = [
    'process_abbreviation_candidates',
    'integrate_with_entity_extraction',
    'promote_abbreviation_to_entity',
    'has_required_id',
    'normalize_ids',
    'get_entity_key',
    'PromotionConfig',
    'load_config_from_yaml'
]

# ============================================================================
# INTEGRATION FUNCTION
# ============================================================================

def integrate_with_entity_extraction(
    extraction_results: Dict[str, Any],
    abbreviation_results: Dict[str, Any],
    config_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Integrate abbreviation extraction results with entity extraction.
    
    Args:
        extraction_results: Results from drug/disease extraction
        abbreviation_results: Results from abbreviation extraction
        config_dict: Optional configuration dictionary
        
    Returns:
        Integrated results with promoted entities
    """
    # Get existing entities
    direct_drugs = extraction_results.get('drugs', [])
    direct_diseases = extraction_results.get('diseases', [])
    
    # Convert abbreviation results to candidates if needed
    abbreviations = abbreviation_results.get('abbreviations', [])
    candidates = []
    
    for abbrev in abbreviations:
        if isinstance(abbrev, dict):
            # Convert dict to candidate-like object
            class DictWrapper:
                def __init__(self, d):
                    self.__dict__.update(d)
                    # Ensure metadata exists
                    if 'metadata' not in d:
                        self.metadata = {}
                    else:
                        self.metadata = d['metadata']
                
                def to_dict(self):
                    return self.__dict__
            
            candidate = DictWrapper(abbrev)
            candidates.append(candidate)
        else:
            candidates.append(abbrev)
    
    # Process for promotion (with config)
    remaining_abbrevs, all_drugs, all_diseases, promotion_links = process_abbreviation_candidates(
        candidates, direct_drugs, direct_diseases, config_dict
    )
    
    # Build integrated results
    return {
        'drugs': all_drugs,
        'diseases': all_diseases,
        'abbreviations': remaining_abbrevs,
        'promotion_links': promotion_links,
        'statistics': {
            'original_drugs': len(direct_drugs),
            'original_diseases': len(direct_diseases),
            'original_abbreviations': len(abbreviations),
            'promoted_to_drug': sum(1 for l in promotion_links if l['entity_type'] == 'Drug'),
            'promoted_to_disease': sum(1 for l in promotion_links if l['entity_type'] == 'Disease'),
            'final_drugs': len(all_drugs),
            'final_diseases': len(all_diseases),
            'final_abbreviations': len(remaining_abbrevs)
        }
    }