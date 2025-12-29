#!/usr/bin/env python3
"""
Enhanced Entity Extraction Pipeline - UPDATED v12.0.0
=====================================================
Location: corpus_metadata/document_utils/entity_extraction.py
Version: 12.0.0 - REMOVED ABBREVIATION EXTRACTION
Last Updated: 2025-12-27

CHANGES IN v12.0.0:
==================
[REMOVED] All abbreviation extraction functionality
[REMOVED] Fallback abbreviation extraction
[REMOVED] Abbreviation conversion functions
[REMOVED] Abbreviation enrichment step
[REMOVED] ID-gated promotion from abbreviations
[REMOVED] Cross-type deduplication with abbreviations
[UPDATED] Pipeline now focuses on DISEASE and DRUG extraction only
[UPDATED] Simplified extraction summary

CHANGES IN v11.3.0:
==================
[+] ADDED provenance_span to all entity types (drugs, diseases, citations, persons, references)
[+] ADDED _create_provenance_span() helper function
[+] ADDED _infer_section_from_position() for section detection
"""
import json
import logging
import re
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
    apply_adaptive_confidence_filter,
    validate_extraction_results,
    SYMPTOM_KEYWORDS,
    FINDING_KEYWORDS,
    GENERIC_DISEASE_TERMS,
    ALWAYS_KEEP_DISEASES
)

# Import extraction modules
from corpus_metadata.document_utils.entity_db_extraction import ExtractionDatabase

# Configure logging
logger = logging.getLogger(__name__)

# Pipeline version 
PIPELINE_VERSION = 'v12.0.0'


# ============================================================================
# PROVENANCE SPAN SUPPORT
# ============================================================================

def _infer_section_from_position(text: str, char_start: int) -> str:
    """
    Infer document section from character position.
    
    Looks backwards from the position to find section headers.
    
    Args:
        text: Full document text
        char_start: Character position to check
        
    Returns:
        Inferred section name or 'Unknown'
    """
    if not text or char_start <= 0:
        return 'Unknown'
    
    # Get text before the position (up to 2000 chars back)
    lookback = min(char_start, 2000)
    preceding_text = text[char_start - lookback:char_start]
    
    # Section patterns (ordered by priority)
    section_patterns = [
        (r'\bAbstract\b', 'Abstract'),
        (r'\bBackground\b', 'Background'),
        (r'\bIntroduction\b', 'Introduction'),
        (r'\bMethods?\b', 'Methods'),
        (r'\bMaterials?\s+and\s+Methods?\b', 'Materials and Methods'),
        (r'\bResults?\b', 'Results'),
        (r'\bDiscussion\b', 'Discussion'),
        (r'\bConclusions?\b', 'Conclusions'),
        (r'\bReferences?\b', 'References'),
        (r'\bAcknowledgements?\b', 'Acknowledgements'),
        (r'\bSupplementary\b', 'Supplementary'),
        (r'\bAppendix\b', 'Appendix'),
        (r'\bFigure\s*\d+', 'Figure'),
        (r'\bTable\s*\d+', 'Table'),
    ]
    
    # Find the latest section header
    latest_section = 'Unknown'
    latest_pos = -1
    
    for pattern, section_name in section_patterns:
        for match in re.finditer(pattern, preceding_text, re.IGNORECASE):
            if match.start() > latest_pos:
                latest_pos = match.start()
                latest_section = section_name
    
    return latest_section


def _create_provenance_span(
    entity: Dict[str, Any],
    full_text: str,
    context_window: int = 100
) -> Optional[Dict[str, Any]]:
    """
    Create a provenance_span structure from entity position/context data.
    
    Args:
        entity: Entity dictionary with position/context fields
        full_text: Full document text for extracting spans
        context_window: Number of characters for context window
        
    Returns:
        Provenance span dictionary or None if no position data
    """
    # Try to get position information
    position = entity.get('position')
    positions = entity.get('positions', [])
    context = entity.get('context', '')
    contexts = entity.get('contexts', [])
    
    # Determine char_start and char_end
    char_start = None
    char_end = None
    
    if position and isinstance(position, (tuple, list)) and len(position) >= 2:
        char_start, char_end = position[0], position[1]
    elif positions and len(positions) > 0:
        # Use first position if multiple
        first_pos = positions[0]
        if isinstance(first_pos, (tuple, list)) and len(first_pos) >= 2:
            char_start, char_end = first_pos[0], first_pos[1]
    
    # If no position data, try to find entity in text
    if char_start is None and full_text:
        entity_name = entity.get('name', '')
        
        if entity_name:
            idx = full_text.find(entity_name)
            if idx >= 0:
                char_start = idx
                char_end = idx + len(entity_name)
    
    # If still no position, return None
    if char_start is None:
        return None
    
    # Extract context text
    if context:
        span_text = context
    elif contexts and len(contexts) > 0:
        span_text = contexts[0]
    elif full_text and char_start is not None:
        # Extract context window around the entity
        window_start = max(0, char_start - context_window)
        window_end = min(len(full_text), char_end + context_window)
        span_text = full_text[window_start:window_end]
        
        # Clean up the span text
        span_text = span_text.strip()
        span_text = re.sub(r'\s+', ' ', span_text)
    else:
        span_text = ''
    
    # Infer section
    section = _infer_section_from_position(full_text, char_start) if full_text else 'Unknown'
    
    return {
        'text': span_text[:500] if span_text else '',  # Limit to 500 chars
        'char_start': char_start,
        'char_end': char_end,
        'section': section
    }


def _add_provenance_to_entity(
    entity: Dict[str, Any],
    full_text: str,
    context_window: int = 100
) -> Dict[str, Any]:
    """
    Add provenance_span to an entity and clean up old position fields.
    
    Args:
        entity: Entity dictionary
        full_text: Full document text
        context_window: Context window size
        
    Returns:
        Entity with provenance_span added
    """
    # Create provenance span
    provenance = _create_provenance_span(entity, full_text, context_window)
    
    if provenance:
        entity['provenance_span'] = provenance
    
    # Remove old position fields (now stored in provenance_span)
    entity.pop('position', None)
    entity.pop('positions', None)
    entity.pop('contexts', None)
    
    return entity


def _add_provenance_to_entities(
    entities: List[Dict[str, Any]],
    full_text: str,
    context_window: int = 100
) -> List[Dict[str, Any]]:
    """
    Add provenance_span to a list of entities.
    
    Args:
        entities: List of entity dictionaries
        full_text: Full document text
        context_window: Context window size
        
    Returns:
        List of entities with provenance_span added
    """
    return [
        _add_provenance_to_entity(entity, full_text, context_window)
        for entity in entities
    ]


# ============================================================================
# CONTEXT-AWARE DRUG FILTERING
# ============================================================================

def _filter_drugs_by_context(
    drugs: List[Dict[str, Any]],
    full_text: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter drug candidates based on context patterns.
    
    Identifies and excludes:
    - Lab reagents (cell culture components)
    - Signaling molecules (intracellular Ca2+, ROS)
    - Gene/protein name fragments
    - Biomarkers used for staining
    
    Args:
        drugs: List of drug candidate dictionaries
        full_text: Full document text for context
        
    Returns:
        Tuple of (kept_drugs, excluded_drugs)
    """
    # Context patterns that indicate NON-drug usage
    lab_context_patterns = [
        re.compile(r'\b(?:medium|media|DMEM|RPMI|MCDB|culture)\b.{0,50}', re.IGNORECASE),
        re.compile(r'\b(?:supplemented|containing)\s+\d+\s*(?:mM|µM|%|mg/mL|U/mL)\b', re.IGNORECASE),
        re.compile(r'\b(?:stimulated|treated|incubated)\s+(?:with|using)\b', re.IGNORECASE),
        re.compile(r'\b(?:buffer|solution|PBS|HEPES)\b', re.IGNORECASE),
    ]
    
    signaling_patterns = [
        re.compile(r'\\b(?:intracellular|cytosolic)\\s*(?:Ca2?\\+|calcium)', re.IGNORECASE),
        re.compile(r'\\[Ca2?\\+\\]', re.IGNORECASE),
        re.compile(r'Ca2?\\+\\s*(?:release|influx|efflux|flux|entry)', re.IGNORECASE),
        re.compile(r'\\b(?:SOCE|TRPC|TRPV|STIM1|ORAI)\\b', re.IGNORECASE),
    ]
    
    gene_patterns = [
        re.compile(r'\s*(?:gene|receptor|channel|transporter|factor)\b', re.IGNORECASE),
        re.compile(r'\b(?:Receptor|Channel)\s+(?:Type\s+)?\d*\s*\(\s*', re.IGNORECASE),
        re.compile(r'\b(?:TRPC|TRPV|TRPM|ORAI|STIM)\d*\b', re.IGNORECASE),
    ]
    
    biomarker_patterns = [
        re.compile(r'\b(?:staining|stained|costaining|immunostaining)\b', re.IGNORECASE),
        re.compile(r'\b(?:marker|markers|indicated)\s+(?:by|for|with)\b', re.IGNORECASE),
        re.compile(r'\b(?:FACS|flow\s+cytometry)\b', re.IGNORECASE),
    ]
    
    kept = []
    excluded = []
    
    for drug in drugs:
        name = drug.get('name', '')
        context = drug.get('context', '')
        
        # Get context from text if not provided
        if not context and full_text:
            match = re.search(re.escape(name), full_text, re.IGNORECASE)
            if match:
                start = max(0, match.start() - 100)
                end = min(len(full_text), match.end() + 100)
                context = full_text[start:end]
        
        # Check for non-drug context
        is_lab_reagent = any(p.search(context) for p in lab_context_patterns)
        is_signaling = any(p.search(context) for p in signaling_patterns)
        is_gene_fragment = any(p.search(context) for p in gene_patterns)
        is_biomarker = any(p.search(context) for p in biomarker_patterns)
        
        if is_lab_reagent or is_signaling or is_gene_fragment or is_biomarker:
            drug['filtered_out'] = True
            drug['filter_reason'] = (
                'lab_reagent' if is_lab_reagent else
                'signaling_molecule' if is_signaling else
                'gene_fragment' if is_gene_fragment else
                'biomarker'
            )
            excluded.append(drug)
        else:
            kept.append(drug)
    
    logger.info(f"Drug context filter: kept {len(kept)}, excluded {len(excluded)}")
    return kept, excluded


# ============================================================================
# DISEASE VALIDATION
# ============================================================================

def _validate_disease_entities(
    diseases: List[Dict[str, Any]],
    full_text: str = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Validate disease entities and filter invalid ones.
    
    Filters out:
    - Cell types (hPASMCs, HUVECs)
    - Incomplete names (peri-, -associated without base)
    - Parsing artifacts
    
    Args:
        diseases: List of disease candidate dictionaries
        full_text: Optional full document text
        
    Returns:
        Tuple of (valid_diseases, invalid_diseases)
    """
    # Cell type patterns (not diseases)
    cell_type_patterns = [
        re.compile(r'^h[A-Z]{2,}[Cc]s?$'),  # hPASMCs, hPAECs
        re.compile(r'^[A-Z]{2,}[Ee][Cc]s?$'),  # HUVECs
        re.compile(r'^[A-Z]{2,}[Ss][Mm][Cc]s?$'),  # PASMCs
        re.compile(r'\b(?:cells?|fibroblasts?|macrophages?|lymphocytes?)\b', re.IGNORECASE),
    ]
    
    # Incomplete name patterns
    incomplete_patterns = [
        re.compile(r'^(?:idiopathic|hereditary|familial|congenital|chronic|acute)$', re.IGNORECASE),
        re.compile(r'^(?:peri|para|hyper|hypo|poly)-?$', re.IGNORECASE),
        re.compile(r'\s+(?:with|and|or|type|associated)$', re.IGNORECASE),
        re.compile(r'^(?:disease|syndrome|disorder)s?$', re.IGNORECASE),
    ]
    
    # Artifact patterns
    artifact_patterns = [
        re.compile(r'^(?:http|https|www|\.com|\.org)$', re.IGNORECASE),
        re.compile(r'^\d+$'),
        re.compile(r'^.{1,2}$'),
        re.compile(r'[|\\/@#$%^&*]'),
    ]
    
    valid = []
    invalid = []
    
    for disease in diseases:
        name = disease.get('name', '').strip()
        
        if not name or len(name) < 3:
            disease['validation_reason'] = 'too_short'
            invalid.append(disease)
            continue
        
        # Check patterns
        is_cell_type = any(p.search(name) for p in cell_type_patterns)
        is_incomplete = any(p.search(name) for p in incomplete_patterns)
        is_artifact = any(p.search(name) for p in artifact_patterns)
        
        if is_artifact:
            disease['validation_reason'] = 'artifact'
            invalid.append(disease)
        elif is_cell_type:
            disease['validation_reason'] = 'cell_type'
            invalid.append(disease)
        elif is_incomplete:
            disease['validation_reason'] = 'incomplete_name'
            invalid.append(disease)
        else:
            disease['validated'] = True
            valid.append(disease)
    
    logger.info(f"Disease validation: {len(valid)} valid, {len(invalid)} filtered")
    return valid, invalid


# ============================================================================
# ENTITY PROCESSING (SIMPLIFIED - NO ABBREVIATIONS)
# ============================================================================

def process_entities_stage(text_content, file_path, components, stage_config, 
                          stage_results, console, features, use_claude):
    """
    Process entities - DRUG and DISEASE extraction only.
    
    v12.0.0: Removed all abbreviation-related processing
    
    Pipeline:
    1. Extract citations (with error recovery)
    2. Extract persons (with error recovery)
    3. Extract references (with error recovery)
    4. Extract drugs with adaptive filtering (with error recovery)
    4.5. Context-aware drug filtering
    5. Extract diseases with adaptive filtering (with error recovery)
    5.5. Disease validation
    6. Generate summary
    
    Args:
        text_content: Document text
        file_path: Path to document file
        components: Dictionary of initialized extractors
        stage_config: Stage configuration
        stage_results: Previous stage results
        console: Console output handler
        features: Feature flags
        use_claude: Whether Claude validation is enabled
        
    Returns:
        Extraction results dictionary
    """
    stage_name = stage_config['name']
    tasks = stage_config.get('tasks', [])
    
    results = {
        'citations': [],
        'persons': [],
        'references': [],
        'drugs': [],
        'diseases': [],
        'processing_errors': [],
        'extraction_summary': {}
    }
    
    # ========================================================================
    # STEP 1: Extract citations (with error recovery)
    # ========================================================================
    if 'citation_extraction' in tasks and 'citation_extractor' in components:
        try:
            start = time.time()
            logger.info(f"Starting citation extraction for {file_path.name}")
            
            citation_results = components['citation_extractor'].extract_citations(
                text_content,
                doc_id=str(file_path.stem)
            )
            
            citations = citation_results.to_dict().get('citations', [])
            
            # Add provenance_span to each citation
            citations = _add_provenance_to_entities(citations, text_content)
            
            results['citations'] = citations
            
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
    # STEP 2: Extract persons (with error recovery)
    # ========================================================================
    if 'person_extraction' in tasks and 'person_extractor' in components:
        try:
            start = time.time()
            logger.info(f"Starting person extraction for {file_path.name}")
            
            person_results = components['person_extractor'].extract_persons(
                text_content,
                doc_id=str(file_path.stem)
            )
            
            persons = person_results.to_dict().get('persons', [])
            
            # Add provenance_span to each person
            persons = _add_provenance_to_entities(persons, text_content)
            
            results['persons'] = persons
            
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
    # STEP 3: Extract references (with error recovery)
    # ========================================================================
    if 'reference_extraction' in tasks and 'reference_extractor' in components:
        try:
            start = time.time()
            logger.info(f"Starting reference extraction for {file_path.name}")
            
            reference_results = components['reference_extractor'].extract_references(
                text_content,
                doc_id=str(file_path.stem)
            )
            
            references = reference_results.to_dict().get('references', [])
            
            # Add provenance_span to each reference
            references = _add_provenance_to_entities(references, text_content)
            
            results['references'] = references
            
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
    # STEP 4: Extract drugs (with error recovery)
    # ========================================================================
    direct_drugs = []
    if 'drug_detection' in tasks and 'drug_extractor' in components:
        try:
            start = time.time()
            logger.info(f"Starting drug detection for {file_path.name}")
            
            drug_results = components['drug_extractor'].extract_drugs_from_text(text_content)
            candidates = drug_results.get('drugs', [])
            logger.info(f"Found {len(candidates)} drug candidates before filtering")
            
            # Apply adaptive confidence filtering
            direct_drugs = apply_adaptive_confidence_filter(
                candidates, 
                entity_type='drug',
                use_claude=use_claude
            )
            
            # Deduplicate
            direct_drugs = deduplicate_by_key(direct_drugs, ('name', 'normalized_name'))
            
            # Add provenance_span to each drug
            direct_drugs = _add_provenance_to_entities(direct_drugs, text_content)
            
            results['drugs'] = direct_drugs
            
            logger.info(f"Extracted {len(direct_drugs)} drugs after confidence filtering in {time.time()-start:.1f}s")
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
    
    # ========================================================================
    # STEP 4.5: Context-aware drug filtering
    # ========================================================================
    direct_drugs = results.get('drugs', [])
    
    if direct_drugs:
        try:
            start = time.time()
            kept_drugs, excluded_drugs = _filter_drugs_by_context(direct_drugs, text_content)
            
            direct_drugs = kept_drugs
            results['drugs'] = kept_drugs
            results['excluded_drugs'] = excluded_drugs
            
            logger.info(f"Drug context filter: kept {len(kept_drugs)}, excluded {len(excluded_drugs)} "
                       f"in {time.time()-start:.1f}s")
            
            if console and excluded_drugs:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "Drug context filter", True, 
                                            f"Excluded {len(excluded_drugs)} non-drugs", time.time() - start)
        except Exception as e:
            logger.warning(f"Drug context filtering failed: {e}")
            results['processing_errors'].append(f"Drug context filter: {str(e)[:100]}")
    
    # ========================================================================
    # STEP 5: Extract diseases (with error recovery)
    # ========================================================================
    direct_diseases = []
    if 'disease_detection' in tasks and 'disease_extractor' in components:
        try:
            start = time.time()
            logger.info(f"Starting disease detection for {file_path.name}")
            
            disease_results = components['disease_extractor'].extract(text_content)
            candidates = disease_results.get('diseases', [])
            logger.info(f"Found {len(candidates)} disease candidates before filtering")
            
            # Apply adaptive confidence filtering
            direct_diseases = apply_adaptive_confidence_filter(
                candidates,
                entity_type='disease',
                use_claude=use_claude
            )
            
            # Deduplicate while preserving disease hierarchies
            direct_diseases = deduplicate_diseases_preserve_hierarchy(direct_diseases)
            
            # Add provenance_span to each disease
            direct_diseases = _add_provenance_to_entities(direct_diseases, text_content)
            
            results['diseases'] = direct_diseases
            
            logger.info(f"Extracted {len(direct_diseases)} diseases after confidence filtering in {time.time()-start:.1f}s")
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
    
    # ========================================================================
    # STEP 5.5: Disease validation
    # ========================================================================
    direct_diseases = results.get('diseases', [])
    
    if direct_diseases:
        try:
            start = time.time()
            valid_diseases, invalid_diseases = _validate_disease_entities(direct_diseases, text_content)
            
            direct_diseases = valid_diseases
            results['diseases'] = valid_diseases
            results['invalid_diseases'] = invalid_diseases
            
            logger.info(f"Disease validation: {len(valid_diseases)} valid, {len(invalid_diseases)} filtered "
                       f"in {time.time()-start:.1f}s")
            
            if console and invalid_diseases:
                console.print_stage_progress(stage_name, stage_config['sequence'], 
                                            "Disease validation", True, 
                                            f"Filtered {len(invalid_diseases)} invalid", time.time() - start)
        except Exception as e:
            logger.warning(f"Disease validation failed: {e}")
            results['processing_errors'].append(f"Disease validation: {str(e)[:100]}")
    
    # ========================================================================
    # STEP 6: Generate extraction summary
    # ========================================================================
    final_drugs = results.get('drugs', [])
    final_diseases = results.get('diseases', [])
    final_citations = results.get('citations', [])
    final_persons = results.get('persons', [])
    final_references = results.get('references', [])
    
    results['extraction_summary'] = {
        'citations_total': len(final_citations),
        'persons_total': len(final_persons),
        'references_total': len(final_references),
        'drugs_total': len(final_drugs),
        'diseases_total': len(final_diseases),
        'has_errors': bool(results['processing_errors']),
        'drugs_context_filtered': len(results.get('excluded_drugs', [])),
        'diseases_validation_filtered': len(results.get('invalid_diseases', []))
    }
    
    logger.info(f"Entity extraction complete: "
               f"{results['extraction_summary']['drugs_total']} drugs, "
               f"{results['extraction_summary']['diseases_total']} diseases, "
               f"{results['extraction_summary']['citations_total']} citations, "
               f"{results['extraction_summary']['persons_total']} persons, "
               f"{results['extraction_summary']['references_total']} references")
    
    return results


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def process_document_two_stage(file_path, components, output_folder, console=None, config=None):
    """
    Process document in two stages: metadata first, then entities
    
    v12.0.0: Removed all abbreviation-related processing
    
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
        # SAVE EXTRACTED TEXT - BOTH FULL AND INTRO VERSIONS
        # ====================================================================
        text_folder = output_folder / 'extracted_texts'
        text_folder.mkdir(exist_ok=True)
        
        # 1. Save FULL text extraction
        text_file = text_folder / f"{file_path.stem}_full.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        final_results['files_saved'].append(f"{file_path.stem}_full.txt")
        logger.info(f"✓ Saved full text: {text_file.name} ({len(text_content):,} chars)")
        
        # 2. Save INTRO text (limited for metadata extraction)
        intro_limit = 10000  # Characters for intro
        intro_text = text_content[:intro_limit]
        intro_file = text_folder / f"{file_path.stem}_intro.txt"
        with open(intro_file, 'w', encoding='utf-8') as f:
            f.write(intro_text)
        final_results['files_saved'].append(f"{file_path.stem}_intro.txt")
        logger.info(f"✓ Saved intro text: {intro_file.name} ({len(intro_text):,} chars)")
        
        # ====================================================================
        # PROCESS PIPELINE STAGES
        # ====================================================================
        
        pipeline_stages = config.get('pipeline', {}).get('stages', [])
        
        for stage_config in pipeline_stages:
            stage_name = stage_config.get('name', '')
            
            if stage_name == 'metadata':
                # Process metadata stage with limited text
                stage_results = {}
                limits = stage_config.get('limits', {})
                char_limit = limits.get('text_chars', 10000)
                limited_text = text_content[:char_limit] if char_limit else text_content
                
                if 'basic_extractor' in components:
                    extractor = components['basic_extractor']
                    
                    # Classification
                    if 'classification' in stage_config.get('tasks', []):
                        try:
                            classification = extractor.classify_document(limited_text, file_path.name)
                            stage_results.update(classification)
                        except Exception as e:
                            logger.error(f"Classification failed: {e}")
                            final_results['processing_errors'].append(f"Classification: {str(e)[:100]}")
                    
                    # Title
                    if 'title_extraction' in stage_config.get('tasks', []):
                        try:
                            title_result = extractor.extract_title(limited_text)
                            if title_result and title_result.get('title'):
                                stage_results['title'] = title_result['title']
                            else:
                                # Fallback title
                                stage_results['title'] = file_path.stem.replace('_', ' ')
                        except Exception as e:
                            logger.error(f"Title extraction failed: {e}")
                            final_results['processing_errors'].append(f"Title: {str(e)[:100]}")
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
                entity_results = process_entities_stage(
                    text_content,  # Use full text for entities
                    file_path,
                    components,
                    stage_config,
                    final_results,
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
            
            # Store entities in database
            entity_stage = next((s for s in final_results.get('pipeline_stages', []) 
                               if s['stage'] == 'entities'), None)
            
            if entity_stage and entity_stage.get('results'):
                entities = entity_stage['results']
                
                # Store drugs
                drugs = entities.get('drugs', [])
                if drugs:
                    try:
                        db.save_drugs_with_counts(
                            run_id, 
                            drugs, 
                            {},  # No abbreviation map
                            text_content
                        )
                        logger.info(f"Saved {len(drugs)} drugs to database")
                    except Exception as e:
                        logger.warning(f"Failed to store drugs: {e}")
                
                # Store diseases
                diseases = entities.get('diseases', [])
                if diseases:
                    try:
                        db.save_diseases_with_counts(
                            run_id, 
                            diseases, 
                            {},  # No abbreviation map
                            text_content
                        )
                        logger.info(f"Saved {len(diseases)} diseases to database")
                    except Exception as e:
                        logger.warning(f"Failed to store diseases: {e}")
            
            # Complete extraction
            extraction_summary = entity_stage.get('results', {}).get('extraction_summary', {}) if entity_stage else {}
            
            db.complete_extraction(
                run_id,
                0,  # No abbreviations
                extraction_summary.get('drugs_total', len(entities.get('drugs', [])) if entity_stage else 0),
                extraction_summary.get('diseases_total', len(entities.get('diseases', [])) if entity_stage else 0),
                0,  # No promoted drugs
                0,  # No promoted diseases
                time.time() - start_time,
                len(text_content) if text_content else 0
            )
            
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