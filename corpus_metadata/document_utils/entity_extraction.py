#!/usr/bin/env python3
"""
Enhanced Entity Extraction Pipeline - UPDATED v11.1.1
=====================================================
Location: corpus_metadata/document_utils/entity_extraction.py
Version: 11.1.1 - ADDED INTRO TEXT EXTRACTION
Last Updated: 2025-10-14

CHANGES IN v11.1.1:
==================
✓ ADDED intro text file extraction (saves limited text separately)
✓ UPDATED file naming: _full.txt and _intro.txt
✓ IMPROVED logging for text extraction

CHANGES IN v11.1.0:
==================
✓ ADDED citation extraction (Step 2)
✓ ADDED person extraction (Step 3)  
✓ ADDED reference extraction (Step 4)
✓ UPDATED extraction summary to include citation/person/reference counts
✓ Renumbered existing steps (drugs: 2→5, diseases: 3→6, dedup: 3.5→7, etc.)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Import utility functions and classes
from corpus_metadata.document_utils.entity_extraction_utils import (
    ConfidenceBand,
    ThresholdConfig,
    ExtractionCache,
    get_confidence_band,
    fuzzy_match,
    deduplicate_by_key,
    deduplicate_diseases_preserve_hierarchy,
    deduplicate_entities_across_types,
    apply_adaptive_confidence_filter,
    enrich_abbreviations_with_entity_ids,
    validate_extraction_results,
    SYMPTOM_KEYWORDS,
    FINDING_KEYWORDS,
    GENERIC_DISEASE_TERMS,
    ALWAYS_KEEP_DISEASES
)

# Import extraction modules
from corpus_metadata.document_utils.entity_db_extraction import ExtractionDatabase
from corpus_metadata.document_utils.entity_report import generate_extraction_report
from corpus_metadata.document_utils.entity_abbreviation_promotion import (
    process_abbreviation_candidates,
    has_required_id,
    normalize_ids
)

# Configure logging
logger = logging.getLogger(__name__)

# Pipeline version 
PIPELINE_VERSION = 'v11.1.2'


# ============================================================================
# ENTITY PROCESSING WITH ERROR RECOVERY AND ENRICHMENT
# ============================================================================

def process_entities_stage_with_promotion(text_content, file_path, components, stage_config, 
                                         stage_results, abbreviation_context, console, 
                                         features, use_claude):
    """
    Process entities with abbreviation-first approach, ID enrichment, and ID-gated promotion
    
    UPDATED IN v11.1.0:
    - Added citation extraction (Step 2)
    - Added person extraction (Step 3)
    - Added reference extraction (Step 4)
    - Renumbered existing steps accordingly
    
    UPDATED IN v11.0.0:
    - Uses simplified adaptive filtering with confidence bands
    - More permissive promotion logic (OR instead of AND)
    - Reduced fuzzy matching threshold (85 → 75)
    
    Pipeline:
    1. Extract all abbreviations (with error recovery)
    2. Extract citations (with error recovery) - NEW in v11.1.0
    3. Extract persons (with error recovery) - NEW in v11.1.0
    4. Extract references (with error recovery) - NEW in v11.1.0
    5. Extract drugs with adaptive filtering (with error recovery)
    6. Extract diseases with adaptive filtering (with error recovery)
    7. Cross-type deduplication
    8. Enrich abbreviations with IDs from detected drugs/diseases
    9. Apply ID-gated promotion for enriched abbreviations (with error recovery)
    10. Validate and generate summary
    
    Args:
        text_content: Document text
        file_path: Path to document file
        components: Dictionary of initialized extractors
        stage_config: Stage configuration
        stage_results: Previous stage results
        abbreviation_context: Context from abbreviation extraction
        console: Console output handler
        features: Feature flags
        use_claude: Whether Claude validation is enabled
        
    Returns:
        Tuple of (extraction_results, promotion_links)
    """
    stage_name = stage_config['name']
    tasks = stage_config.get('tasks', [])
    
    results = {
        'abbreviations': [],
        'citations': [],
        'persons': [],
        'references': [],
        'drugs': [],
        'diseases': [],
        'processing_errors': [],
        'extraction_summary': {},
        'enrichment_stats': {},
        'promotion_links': []
    }
    
    # ========================================================================
    # STEP 1: Extract abbreviations (with error recovery)
    # ========================================================================
    if 'abbreviation_extraction' in tasks and 'abbreviation_extractor' in components:
        try:
            start = time.time()
            logger.info(f"Starting abbreviation extraction for {file_path.name}")
            
            abbrev_results = components['abbreviation_extractor'].extract_abbreviations(
                text_content
            )
            
            all_abbreviations = abbrev_results.get('abbreviations', [])
            all_abbreviations = deduplicate_by_key(all_abbreviations, ('abbreviation', 'expansion'))
            results['abbreviations'] = all_abbreviations
            
            logger.info(f"Extracted {len(all_abbreviations)} unique abbreviations in {time.time()-start:.1f}s")
            if console:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "Abbreviation extraction", True, 
                                            f"{len(all_abbreviations)} unique", time.time() - start)
        except Exception as e:
            logger.error(f"Abbreviation extraction failed: {type(e).__name__}: {e}", exc_info=True)
            results['processing_errors'].append(f"Abbreviation extraction: {str(e)[:100]}")
            all_abbreviations = []
            if console:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "Abbreviation extraction", False, str(e)[:50])
    else:
        all_abbreviations = []
    
    # ========================================================================
    # STEP 2: Extract citations (with error recovery) - NEW in v11.1.0
    # ========================================================================
    if 'citation_extraction' in tasks and 'citation_extractor' in components:
        try:
            start = time.time()
            logger.info(f"Starting citation extraction for {file_path.name}")
            
            citation_results = components['citation_extractor'].extract_citations(
                text_content,
                doc_id=str(file_path.stem)
            )
            
            results['citations'] = citation_results.to_dict().get('citations', [])
            
            logger.info(f"Extracted {len(results['citations'])} citations in {time.time()-start:.1f}s")
            if console:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "Citation extraction", True, 
                                            f"{len(results['citations'])} found", time.time() - start)
        except Exception as e:
            logger.error(f"Citation extraction failed: {type(e).__name__}: {e}", exc_info=True)
            results['processing_errors'].append(f"Citation extraction: {str(e)[:100]}")
            results['citations'] = []
            if console:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "Citation extraction", False, str(e)[:50])
    else:
        results['citations'] = []
    
    # ========================================================================
    # STEP 3: Extract persons (with error recovery) - NEW in v11.1.0
    # ========================================================================
    if 'person_extraction' in tasks and 'person_extractor' in components:
        try:
            start = time.time()
            logger.info(f"Starting person extraction for {file_path.name}")
            
            person_results = components['person_extractor'].extract_persons(
                text_content,
                doc_id=str(file_path.stem)
            )
            
            results['persons'] = person_results.to_dict().get('persons', [])
            
            logger.info(f"Extracted {len(results['persons'])} persons in {time.time()-start:.1f}s")
            if console:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "Person extraction", True, 
                                            f"{len(results['persons'])} found", time.time() - start)
        except Exception as e:
            logger.error(f"Person extraction failed: {type(e).__name__}: {e}", exc_info=True)
            results['processing_errors'].append(f"Person extraction: {str(e)[:100]}")
            results['persons'] = []
            if console:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "Person extraction", False, str(e)[:50])
    else:
        results['persons'] = []
    
    # ========================================================================
    # STEP 4: Extract references (with error recovery) - NEW in v11.1.0
    # ========================================================================
    if 'reference_extraction' in tasks and 'reference_extractor' in components:
        try:
            start = time.time()
            logger.info(f"Starting reference extraction for {file_path.name}")
            
            reference_results = components['reference_extractor'].extract_references(
                text_content,
                doc_id=str(file_path.stem)
            )
            
            results['references'] = reference_results.to_dict().get('references', [])
            
            logger.info(f"Extracted {len(results['references'])} references in {time.time()-start:.1f}s")
            if console:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "Reference extraction", True, 
                                            f"{len(results['references'])} found", time.time() - start)
        except Exception as e:
            logger.error(f"Reference extraction failed: {type(e).__name__}: {e}", exc_info=True)
            results['processing_errors'].append(f"Reference extraction: {str(e)[:100]}")
            results['references'] = []
            if console:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "Reference extraction", False, str(e)[:50])
    else:
        results['references'] = []
    
    # ========================================================================
    # STEP 5: Extract drugs (with error recovery)
    # ========================================================================
    if 'drug_detection' in tasks and 'drug_extractor' in components:
        try:
            start = time.time()
            logger.info(f"Starting drug detection for {file_path.name}")
            
            drug_results = components['drug_extractor'].extract_drugs_from_text(text_content)
            candidates = drug_results.get('drugs', [])
            logger.info(f"Found {len(candidates)} drug candidates before filtering")
            
            # Apply SIMPLIFIED adaptive confidence filtering
            direct_drugs = apply_adaptive_confidence_filter(
                candidates, 
                entity_type='drug',
                use_claude=use_claude,
                all_abbreviations=all_abbreviations
            )
            
            # Deduplicate
            direct_drugs = deduplicate_by_key(direct_drugs, ('name', 'normalized_name'))
            
            # Remove position fields
            for drug in direct_drugs:
                drug.pop('position', None)
                drug.pop('positions', None)
            
            results['drugs'] = direct_drugs
            
            logger.info(f"Extracted {len(direct_drugs)} drugs after filtering in {time.time()-start:.1f}s")
            if console:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "Drug detection", True, 
                                            f"{len(direct_drugs)} drugs", time.time() - start)
        except Exception as e:
            logger.error(f"Drug detection failed: {type(e).__name__}: {e}", exc_info=True)
            results['processing_errors'].append(f"Drug detection: {str(e)[:100]}")
            results['drugs'] = []
            if console:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "Drug detection", False, str(e)[:50])
    
    direct_drugs = results['drugs']
    
    # ========================================================================
    # STEP 6: Extract diseases (with error recovery)
    # ========================================================================
    if 'disease_detection' in tasks and 'disease_extractor' in components:
        try:
            start = time.time()
            logger.info(f"Starting disease detection for {file_path.name}")
            
            disease_results = components['disease_extractor'].extract(text_content)
            candidates = disease_results.get('diseases', [])
            logger.info(f"Found {len(candidates)} disease candidates before filtering")
            
            # Apply SIMPLIFIED adaptive confidence filtering
            direct_diseases = apply_adaptive_confidence_filter(
                candidates,
                entity_type='disease',
                use_claude=use_claude,
                all_abbreviations=all_abbreviations
            )
            
            # Deduplicate while preserving disease hierarchies
            direct_diseases = deduplicate_diseases_preserve_hierarchy(direct_diseases)
            
            # Remove position fields
            for disease in direct_diseases:
                disease.pop('position', None)
                disease.pop('positions', None)
            
            results['diseases'] = direct_diseases
            
            logger.info(f"Extracted {len(direct_diseases)} diseases after filtering in {time.time()-start:.1f}s")
            if console:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "Disease detection", True, 
                                            f"{len(direct_diseases)} diseases", time.time() - start)
        except Exception as e:
            logger.error(f"Disease detection failed: {type(e).__name__}: {e}", exc_info=True)
            results['processing_errors'].append(f"Disease detection: {str(e)[:100]}")
            results['diseases'] = []
            if console:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "Disease detection", False, str(e)[:50])
    
    direct_diseases = results['diseases']
    
    # ========================================================================
    # STEP 7: CROSS-TYPE DEDUPLICATION
    # ========================================================================
    dedup_enabled = True  # Default to enabled
    
    # Check if we have a config object with deduplication settings
    if hasattr(components, 'config'):
        dedup_enabled = components.config.get('deduplication', {}).get('across_entity_types', True)
    
    if dedup_enabled and all_abbreviations and (direct_drugs or direct_diseases):
        try:
            start = time.time()
            logger.info(f"Starting cross-type deduplication: {len(all_abbreviations)} abbrev, "
                       f"{len(direct_drugs)} drugs, {len(direct_diseases)} diseases")
            
            all_abbreviations, direct_drugs, direct_diseases = deduplicate_entities_across_types(
                all_abbreviations,
                direct_drugs,
                direct_diseases
            )
            
            # Update results with deduplicated entities
            results['abbreviations'] = all_abbreviations
            results['drugs'] = direct_drugs
            results['diseases'] = direct_diseases
            
            logger.info(f"After deduplication: {len(all_abbreviations)} abbrev, "
                       f"{len(direct_drugs)} drugs, {len(direct_diseases)} diseases "
                       f"({time.time()-start:.1f}s)")
            
            if console:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "Cross-type deduplication", True,
                                            f"Cleaned entities", time.time() - start)
                                            
        except Exception as e:
            logger.warning(f"Cross-type deduplication failed: {e}", exc_info=True)
            results['processing_errors'].append(f"Deduplication: {str(e)[:100]}")
    else:
        logger.debug("Cross-type deduplication: Skipped (disabled or no entities)")
    
    # ========================================================================
    # STEP 8: Enrich abbreviations with IDs from detected entities
    # ========================================================================
    if all_abbreviations and (direct_drugs or direct_diseases):
        try:
            start = time.time()
            logger.info(f"Enriching {len(all_abbreviations)} abbreviations with entity IDs")
            
            enriched_abbreviations, enrichment_stats = enrich_abbreviations_with_entity_ids(
                all_abbreviations,
                direct_drugs,
                direct_diseases
            )
            
            # Update abbreviations with enriched versions
            all_abbreviations = enriched_abbreviations
            results['abbreviations'] = enriched_abbreviations
            results['enrichment_stats'] = enrichment_stats
            
            logger.info(f"Enrichment complete: {enrichment_stats['total_enriched']} abbreviations "
                       f"now eligible for promotion ({time.time()-start:.1f}s)")
            
            if console:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "Abbreviation enrichment", True,
                                            f"{enrichment_stats['total_enriched']} enriched", 
                                            time.time() - start)
                                            
        except Exception as e:
            logger.warning(f"Enrichment failed: {e}", exc_info=True)
            results['processing_errors'].append(f"Enrichment: {str(e)[:100]}")
            results['enrichment_stats'] = {}
            if console:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "Abbreviation enrichment", False, str(e)[:50])
    else:
        logger.debug("Enrichment: Skipped (no abbreviations or entities)")
        results['enrichment_stats'] = {}
    
    # ========================================================================
    # STEP 9: ID-Gated Promotion (with error recovery)
    # ========================================================================
    promotion_links = []
    
    if all_abbreviations:
        try:
            start = time.time()
            logger.info(f"Starting ID-gated promotion for {len(all_abbreviations)} abbreviations")
            
            # Get promotion configuration
            promotion_config = components.get('config', {}).get('promotion', {})
            
            # UPDATED IN v11.0.0: More permissive promotion
            kept_abbreviations, promoted_drugs, promoted_diseases, links = process_abbreviation_candidates(
                all_abbreviations,
                direct_drugs,
                direct_diseases,
                promotion_config=promotion_config
            )
            
            promotion_links = links
            
            # Update results
            results['drugs'] = promoted_drugs
            results['diseases'] = promoted_diseases
            results['abbreviations'] = kept_abbreviations
            results['promotion_links'] = promotion_links
            
            drugs_promoted = len(promoted_drugs) - len(direct_drugs)
            diseases_promoted = len(promoted_diseases) - len(direct_diseases)
            
            logger.info(f"Promotion complete: {drugs_promoted} drugs, {diseases_promoted} diseases "
                       f"promoted in {time.time()-start:.1f}s")
            
            if console:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "ID-gated promotion", True,
                                            f"{drugs_promoted + diseases_promoted} promoted", 
                                            time.time() - start)
                                            
        except Exception as e:
            logger.error(f"Promotion failed: {type(e).__name__}: {e}", exc_info=True)
            results['processing_errors'].append(f"Promotion: {str(e)[:100]}")
            promotion_links = []
            if console:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "ID-gated promotion", False, str(e)[:50])
    else:
        logger.debug("Promotion: Skipped (no abbreviations)")
    
    # ========================================================================
    # STEP 10: Generate extraction summary
    # ========================================================================
    final_drugs = results.get('drugs', [])
    final_diseases = results.get('diseases', [])
    final_abbreviations = results.get('abbreviations', [])
    final_citations = results.get('citations', [])      # NEW in v11.1.0
    final_persons = results.get('persons', [])          # NEW in v11.1.0
    final_references = results.get('references', [])    # NEW in v11.1.0
    
    results['extraction_summary'] = {
        'abbreviations_total': len(final_abbreviations),
        'citations_total': len(final_citations),        # NEW in v11.1.0
        'persons_total': len(final_persons),            # NEW in v11.1.0
        'references_total': len(final_references),      # NEW in v11.1.0
        'drugs_direct': len(direct_drugs),
        'drugs_promoted': len(final_drugs) - len(direct_drugs),
        'drugs_total': len(final_drugs),
        'diseases_direct': len(direct_diseases),
        'diseases_promoted': len(final_diseases) - len(direct_diseases),
        'diseases_total': len(final_diseases),
        'promotion_rate': len(promotion_links) / len(all_abbreviations) if all_abbreviations else 0,
        'has_errors': bool(results['processing_errors']),
        'abbreviations_enriched': results.get('enrichment_stats', {}).get('total_enriched', 0)
    }
    
    logger.info(f"Entity extraction complete: {results['extraction_summary']['abbreviations_total']} abbrev, "
               f"{results['extraction_summary']['drugs_total']} drugs, "
               f"{results['extraction_summary']['diseases_total']} diseases, "
               f"{results['extraction_summary']['citations_total']} citations, "
               f"{results['extraction_summary']['persons_total']} persons, "
               f"{results['extraction_summary']['references_total']} references")
    
    return results, promotion_links


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def process_document_two_stage(file_path, components, output_folder, console=None, config=None):
    """
    Process document in two stages: metadata first, then entities
    
    UPDATED IN v11.1.1: Added intro text file extraction
    UPDATED IN v11.1.0: Added citation, person, and reference extraction
    UPDATED IN v11.0.0: Uses new simplified filtering and promotion logic
    
    Args:
        file_path: Path to document
        components: Initialized extraction components
        output_folder: Output directory for extracted files (REQUIRED positional)
        console: Optional console output handler (keyword)
        config: Pipeline configuration (keyword)
        
    Returns:
        Complete extraction results with validation
    """
    start_time = time.time()
    file_path = Path(file_path)
    output_folder = Path(output_folder)
    
    # Handle missing config
    if config is None:
        config = {
            'features': {'ai_validation': False, 'enable_document_caching': False},
            'pipeline': {'stages': []},
            'deduplication': {'across_entity_types': True}
        }
        logger.warning("No config provided, using defaults")
    
    # Store config in components for access in helper functions
    if not hasattr(components, 'config'):
        components['config'] = config
    
    features = config.get('features', {})
    use_claude = features.get('ai_validation', False)
    use_caching = features.get('enable_document_caching', False)
    
    # Check cache first
    cache = None
    if use_caching:
        cache = ExtractionCache()
        cached = cache.get(file_path, PIPELINE_VERSION)
        if cached:
            logger.info(f"Returning cached results for {file_path.name}")
            return cached
    
    final_results = {
        'filename': file_path.name,
        'file_path': str(file_path),
        'extraction_date': datetime.now().isoformat(),
        'pipeline_version': PIPELINE_VERSION,
        'pipeline_stages': [],
        'metadata': {},
        'validation': {},
        'processing_errors': [],
        'validation_method': 'claude' if use_claude else 'threshold',
        'files_saved': [],
        'threshold_config': {
            'fuzzy_match_threshold': ThresholdConfig.FUZZY_MATCH_THRESHOLD,
            'drug_min_confidence': ThresholdConfig.DRUG_MIN_CONFIDENCE,
            'disease_min_confidence': ThresholdConfig.DISEASE_MIN_CONFIDENCE,
            'provenance': ThresholdConfig.FUZZY_MATCH_PROVENANCE
        }
    }
    
    try:
        # Read document text
        if 'document_reader' not in components:
            raise ValueError("DocumentReader not available in components")
        
        result = components['document_reader'].read_document(file_path)
        text_content = result.get('content', '')

        if not text_content or not text_content.strip():
            error_msg = result.get('error', 'No text content extracted')
            raise ValueError(f"No text content extracted from document: {error_msg}")
        
        logger.info(f"Extracted {len(text_content):,} characters from {file_path.name}")
        
        # ====================================================================
        # SAVE EXTRACTED TEXT - BOTH FULL AND INTRO VERSIONS (NEW in v11.1.1)
        # ====================================================================
        text_folder = output_folder / 'extracted_texts'
        text_folder.mkdir(exist_ok=True)
        
        # 1. Save FULL text extraction
        text_file = text_folder / f"{file_path.stem}_full.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        final_results['files_saved'].append(text_file.name)
        logger.info(f"✓ Saved full text to {text_file.name} ({len(text_content):,} chars)")
        
        # 2. Save INTRO text extraction (first N characters)
        # Get limit from first stage config, default to 5000
        intro_limit = 5000  # Default
        metadata_stage = next((s for s in config.get('pipeline', {}).get('stages', []) 
                              if s.get('name') == 'metadata'), None)
        if metadata_stage:
            intro_limit = metadata_stage.get('limits', {}).get('text_chars', 5000)
        
        intro_text = text_content[:intro_limit]
        intro_file = text_folder / f"{file_path.stem}_intro.txt"
        with open(intro_file, 'w', encoding='utf-8') as f:
            f.write(intro_text)
        
        final_results['files_saved'].append(intro_file.name)
        logger.info(f"✓ Saved intro text to {intro_file.name} ({len(intro_text):,} chars)")
        # ====================================================================
        
        # Process each configured stage
        for stage_config in config.get('pipeline', {}).get('stages', []):
            stage_name = stage_config['name']
            logger.info(f"Processing stage: {stage_name}")
            
            # Apply stage limits
            limited_text = text_content
            limits = stage_config.get('limits', {})
            
            if limits.get('text_chars'):
                limited_text = text_content[:limits['text_chars']]
                logger.debug(f"Limited text to {limits['text_chars']} chars for {stage_name} stage")
            
            # Process stage
            stage_results = {}
            
            if stage_name == 'metadata':
                # Process metadata tasks (classification, description, dates, etc.)
                if 'basic_extractor' in components:
                    extractor = components['basic_extractor']
                    
                    # Classification
                    if 'classification' in stage_config.get('tasks', []):
                        try:
                            classification = extractor.classify_document(limited_text, file_path.name)
                            stage_results.update(classification)
                            logger.debug(f"Classification: {classification.get('document_type')}")
                        except Exception as e:
                            logger.error(f"Classification failed: {e}")
                            final_results['processing_errors'].append(f"Classification: {str(e)[:100]}")
                    
                    # Descriptions AND Title (both come from generate_descriptions)
                    if 'description' in stage_config.get('tasks', []) or 'title' in stage_config.get('tasks', []):
                        try:
                            # Call generate_descriptions (PLURAL) and it returns dict with title, short_description, long_description
                            descriptions = extractor.generate_descriptions(limited_text, file_path.name, 
                                                                          stage_results.get('document_type'))
                            stage_results.update(descriptions)
                            
                            # Ensure we have at least a basic title if generation failed
                            if not stage_results.get('title'):
                                logger.warning("No title generated, using filename")
                                stage_results['title'] = file_path.stem.replace('_', ' ')
                        except Exception as e:
                            logger.error(f"Description generation failed: {e}")
                            final_results['processing_errors'].append(f"Description: {str(e)[:100]}")
                            # Fallback title
                            stage_results['title'] = file_path.stem.replace('_', ' ')
                                        
                    # Dates
                    if 'dates' in stage_config.get('tasks', []):
                        try:
                            dates = extractor.extract_dates(limited_text)
                            stage_results.update(dates)
                        except Exception as e:
                            logger.error(f"Date extraction failed: {e}")
                            final_results['processing_errors'].append(f"Dates: {str(e)[:100]}")
                    
                    # Filename proposal
                    if 'filename_proposal' in stage_config.get('tasks', []):
                        try:
                            if 'intelligent_renamer' in components:
                                filename_result = components['intelligent_renamer'].propose_filename(
                                    file_path.name,
                                    stage_results
                                )
                                stage_results['filename_proposal'] = filename_result
                        except Exception as e:
                            logger.error(f"Filename proposal failed: {e}")
                            final_results['processing_errors'].append(f"Filename: {str(e)[:100]}")
                
                # Store metadata stage results
                final_results['pipeline_stages'].append({
                    'stage': stage_name,
                    'sequence': stage_config['sequence'],
                    'tasks': stage_config.get('tasks', []),
                    'results': stage_results
                })
                final_results['metadata'] = stage_results
            
            elif stage_name == 'entities':
                # Process entity extraction stage
                abbreviation_context = {}
                
                entity_results, promotion_links = process_entities_stage_with_promotion(
                    text_content,  # Use full text for entities
                    file_path,
                    components,
                    stage_config,
                    final_results,
                    abbreviation_context,
                    console,
                    features,
                    use_claude
                )
                
                # Store entity stage results
                final_results['pipeline_stages'].append({
                    'stage': stage_name,
                    'sequence': stage_config['sequence'],
                    'tasks': stage_config.get('tasks', []),
                    'results': entity_results
                })
                
                # Add promotion links to results
                final_results['promotion_links'] = promotion_links
        
        # ====================================================================
        # VALIDATION & QUALITY ASSESSMENT
        # ====================================================================
        
        validation_results = validate_extraction_results(final_results, config)
        final_results['validation'] = validation_results
        
        # ====================================================================
        # SAVE RESULTS
        # ====================================================================
        
        # Save JSON results
        output_json = output_folder / f"{file_path.stem}_extracted.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        final_results['files_saved'].append(output_json.name)
        logger.info(f"✓ Results saved to {output_json.name}")
        
        # ====================================================================
        # DATABASE STORAGE & REPORTING
        # ====================================================================

        try:
            # Store in database
            db = ExtractionDatabase()
            document_id, run_id = db.start_extraction(
                filename=file_path.name,
                file_path=str(file_path),
                document_metadata=final_results.get('metadata', {}),
                pipeline_version=PIPELINE_VERSION,
                validation_method='claude' if use_claude else 'threshold'
            )
            
            # Store entities - FIX: Check if methods exist before calling
            entity_stage = next((s for s in final_results.get('pipeline_stages', []) 
                               if s['stage'] == 'entities'), None)
            
            if entity_stage and entity_stage.get('results'):
                entities = entity_stage['results']
                
                # Store abbreviations
                if hasattr(db, 'add_abbreviation'):
                    for abbrev in entities.get('abbreviations', []):
                        try:
                            db.add_abbreviation(run_id, abbrev, text_content)
                        except Exception as e:
                            logger.debug(f"Failed to store abbreviation: {e}")
                
                # Store drugs
                if hasattr(db, 'add_drug'):
                    for drug in entities.get('drugs', []):
                        try:
                            db.add_drug(run_id, drug, text_content)
                        except Exception as e:
                            logger.debug(f"Failed to store drug: {e}")
                
                # Store diseases  
                if hasattr(db, 'add_disease'):
                    for disease in entities.get('diseases', []):
                        try:
                            db.add_disease(run_id, disease, text_content)
                        except Exception as e:
                            logger.debug(f"Failed to store disease: {e}")
            
            # Complete extraction
            extraction_summary = entity_stage.get('results', {}).get('extraction_summary', {}) if entity_stage else {}
            
            if hasattr(db, 'complete_extraction'):
                db.complete_extraction(
                    run_id,
                    extraction_summary.get('abbreviations_total', 0),
                    extraction_summary.get('drugs_total', 0),
                    extraction_summary.get('diseases_total', 0),
                    extraction_summary.get('drugs_promoted', 0),
                    extraction_summary.get('diseases_promoted', 0),
                    time.time() - start_time,
                    len(text_content)
                )
            
            # Generate report
            if hasattr(db, 'generate_report'):
                prefix_manager = components.get('prefix_manager')
                report_path = generate_extraction_report(run_id, output_folder, prefix_manager)
                if report_path:
                    final_results['files_saved'].append(Path(report_path).name)
                    logger.info(f"✓ Generated report: {Path(report_path).name}")
            
            logger.debug(f"Stored extraction in database: document_id={document_id}, run_id={run_id}")
            
        except Exception as e:
            logger.warning(f"Database storage failed: {e}")
            final_results['processing_errors'].append(f"Database: {str(e)[:100]}")
        
        # ====================================================================
        # CACHE RESULTS
        # ====================================================================
        
        if cache:
            cache.set(file_path, PIPELINE_VERSION, final_results)
        
        # Calculate total processing time
        total_time = time.time() - start_time
        logger.info(f"✓ Completed processing {file_path.name} in {total_time:.1f}s")
        
        # Log summary of saved files
        logger.info(f"Files saved ({len(final_results['files_saved'])}):")
        for saved_file in final_results['files_saved']:
            logger.info(f"  - {saved_file}")
        
        return final_results
        
    except Exception as e:
        logger.error(f"Document processing failed: {type(e).__name__}: {e}", exc_info=True)
        final_results['processing_errors'].append(f"Processing: {str(e)[:200]}")
        final_results['validation'] = {
            'is_valid': False,
            'warnings': [f"Processing failed: {str(e)[:100]}"]
        }
        return final_results