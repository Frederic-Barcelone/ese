#!/usr/bin/env python3
"""
Document Metadata Extraction - Reference & Identifier Extractor (FIXED + VALIDATED)
====================================================================================
Location: corpus_metadata/document_metadata_extraction_references.py
Version: 1.2.0 - WITH AI VALIDATION
Last Updated: 2025-10-13

FEATURES IN v1.2.0:
- ✅ Integrated Claude AI validation after extraction
- ✅ Auto-corrects misclassified references (PMID vs INSPIREHEP)
- ✅ Validation statistics in output
- ✅ Optional validation (can be disabled)

PREVIOUS FIXES IN v1.1.0:
- ✅ Fixed PMID detection with URL context awareness
- ✅ Added numbered reference parsing (1-60 style)
- ✅ Filter false INSPIREHEP matches (actually PMIDs)
- ✅ Extract full reference metadata (title, journal, authors)
- ✅ Better reference section detection
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from collections import defaultdict, Counter
import json

# Import reference patterns
try:
    from document_utils.entity_reference_patterns import (
        REFERENCE_PATTERNS,
        CONTEXT_ROLE_PATTERNS,
        PUBLISHER_METADATA,
        CONFIDENCE_BOOSTERS,
        get_reference_category_stats,
        get_patterns_by_category,
        get_all_categories,
        validate_pattern
    )
except ImportError:
    logging.warning("Could not import entity_reference_patterns. Using fallback patterns.")
    REFERENCE_PATTERNS = {}
    CONTEXT_ROLE_PATTERNS = {}
    PUBLISHER_METADATA = {}
    CONFIDENCE_BOOSTERS = {}

# NEW: Import validation module
try:
    from document_utils.entity_validations import (
        ReferenceValidator,
        apply_validation_corrections,
        BatchValidationResult
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    logging.warning("Could not import entity_validations. AI validation disabled.")
    VALIDATION_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ReferenceCategory(Enum):
    """Categories of references."""
    UNIVERSAL = "universal"
    LITERATURE = "literature"
    PREPRINT = "preprint"
    INDEX = "index"
    REPOSITORY = "repository"
    CLINICAL_TRIAL = "clinical_trial"
    REGULATORY = "regulatory"
    PATENT = "patent"
    GUIDELINE = "guideline"
    DATABASE = "database"
    PUBLISHER = "publisher"
    UNKNOWN = "unknown"


class ReferenceRole(Enum):
    """Role of reference in document context."""
    SUPPORTING_EVIDENCE = "supporting_evidence"
    METHODOLOGY = "methodology"
    GUIDELINE_REFERENCE = "guideline_reference"
    COMPARATIVE = "comparative"
    SAFETY_DATA = "safety_data"
    TRIAL_REGISTRY = "trial_registry"
    BACKGROUND = "background"
    GENERAL = "general"


class ReferenceStatus(Enum):
    """Status of reference extraction."""
    VERIFIED = "verified"
    NORMALIZED = "normalized"
    RAW = "raw"
    FAILED = "failed"


@dataclass
class ExtractedReference:
    """Structured representation of an extracted reference."""
    
    # REQUIRED FIELDS (no defaults) - MUST come first
    reference_id: str
    reference_type: str
    raw_text: str
    normalized_value: str
    category: ReferenceCategory
    start_char: int
    end_char: int
    
    # OPTIONAL FIELDS (with defaults)
    role: ReferenceRole = ReferenceRole.GENERAL
    section: Optional[str] = None
    sentence: Optional[str] = None
    
    # Full reference metadata
    reference_number: Optional[int] = None
    title: Optional[str] = None
    journal: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[str] = None
    
    # Metadata
    url: Optional[str] = None
    source: Optional[str] = None
    confidence: float = 0.0
    status: ReferenceStatus = ReferenceStatus.RAW
    
    # Context
    preceding_context: Optional[str] = None
    following_context: Optional[str] = None
    
    # Validation
    is_preprint: bool = False
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Additional metadata
    metadata: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        result = {
            'reference_id': self.reference_id,
            'reference_type': self.reference_type,
            'raw_text': self.raw_text,
            'normalized_value': self.normalized_value,
            'category': self.category.value,
            'role': self.role.value,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'confidence': round(self.confidence, 3),
            'status': self.status.value,
            'extraction_timestamp': self.extraction_timestamp
        }
        
        # Optional fields
        optional_fields = ['section', 'sentence', 'url', 'source', 
                          'preceding_context', 'following_context',
                          'reference_number', 'title', 'journal', 'authors', 'year']
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        
        if self.is_preprint:
            result['is_preprint'] = True
        
        if self.metadata:
            result['metadata'] = self.metadata
        
        if self.warnings:
            result['warnings'] = self.warnings
        
        return result


@dataclass
class ReferenceExtractionResult:
    """Complete reference extraction result."""
    doc_id: str
    language: str = "en"
    
    # Extracted references
    references: List[ExtractedReference] = field(default_factory=list)
    
    # Numbered references from reference list
    numbered_references: List[Dict] = field(default_factory=list)
    
    # Statistics
    category_counts: Dict[str, int] = field(default_factory=dict)
    type_counts: Dict[str, int] = field(default_factory=dict)
    
    total_references: int = 0
    unique_references: int = 0
    references_with_urls: int = 0
    average_confidence: float = 0.0
    
    # Quality metrics
    has_doi: bool = False
    has_pubmed: bool = False
    has_clinical_trial: bool = False
    preprint_count: int = 0
    
    # NEW: Validation results
    validation_applied: bool = False
    validation_summary: Optional[Dict] = None
    
    # Document classification
    document_type_hints: List[str] = field(default_factory=list)
    
    # Metadata
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        result_dict = {
            'doc_id': self.doc_id,
            'language': self.language,
            'references': [r.to_dict() for r in self.references],
            'numbered_references': self.numbered_references,
            'statistics': {
                'total_references': self.total_references,
                'unique_references': self.unique_references,
                'references_with_urls': self.references_with_urls,
                'average_confidence': round(self.average_confidence, 3),
                'category_counts': self.category_counts,
                'type_counts': self.type_counts
            },
            'quality_metrics': {
                'has_doi': self.has_doi,
                'has_pubmed': self.has_pubmed,
                'has_clinical_trial': self.has_clinical_trial,
                'preprint_count': self.preprint_count
            },
            'document_type_hints': self.document_type_hints,
            'extraction_timestamp': self.extraction_timestamp,
            'processing_time_seconds': round(self.processing_time_seconds, 3),
            'warnings': self.warnings,
            'metadata': self.metadata
        }
        
        # NEW: Add validation info if applied
        if self.validation_applied:
            result_dict['validation'] = {
                'applied': True,
                'summary': self.validation_summary
            }
        
        return result_dict
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ============================================================================
# REFERENCE EXTRACTOR (FIXED VERSION WITH VALIDATION)
# ============================================================================

class ReferenceExtractor:
    """
    Extract and normalize external references from biomedical documents.
    FIXED: Better PMID detection, numbered reference parsing.
    NEW: Integrated Claude AI validation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the reference extractor."""
        self.config = config or {}
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.extract_context = self.config.get('extract_context', True)
        self.context_window = self.config.get('context_window', 100)
        self.classify_roles = self.config.get('classify_roles', True)
        
        # NEW: Validation settings
        self.enable_validation = self.config.get('enable_validation', True)
        self.validation_batch_size = self.config.get('validation_batch_size', 20)
        
        # Compile patterns
        self._compile_patterns()
        
        # NEW: Initialize validator if enabled
        self.validator = None
        if self.enable_validation and VALIDATION_AVAILABLE:
            try:
                self.validator = ReferenceValidator(
                    batch_size=self.validation_batch_size
                )
                logger.info("✓ Reference validator initialized")
            except Exception as e:
                logger.warning(f"Could not initialize validator: {e}")
                self.enable_validation = False
        elif self.enable_validation and not VALIDATION_AVAILABLE:
            logger.warning("Validation requested but entity_validations module not available")
            self.enable_validation = False
        
        logger.info(f"ReferenceExtractor initialized (validation={'enabled' if self.enable_validation else 'disabled'})")
    
    def _compile_patterns(self):
        """Compile all regex patterns for efficiency."""
        self.compiled_patterns = {}
        
        for ref_type, ref_config in REFERENCE_PATTERNS.items():
            try:
                pattern = ref_config['pattern']
                self.compiled_patterns[ref_type] = {
                    'regex': re.compile(pattern, re.IGNORECASE | re.MULTILINE),
                    'config': ref_config
                }
            except re.error as e:
                logger.warning(f"Failed to compile pattern for {ref_type}: {e}")
                continue
        
        logger.info(f"Compiled {len(self.compiled_patterns)} reference patterns")
        
        # Compile context role patterns
        self.compiled_role_patterns = {}
        for role, patterns in CONTEXT_ROLE_PATTERNS.items():
            self.compiled_role_patterns[role] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        # Compile reference section pattern
        self.reference_section_pattern = re.compile(
            r'^\s*(?:REFERENCES?|Bibliography|Literature\s+Cited)\s*$',
            re.MULTILINE | re.IGNORECASE
        )
        
        # Numbered reference pattern
        self.numbered_ref_pattern = re.compile(
            r'^\s*(\d{1,2})\.\s+(.+?)(?=^\s*\d{1,2}\.\s+|$)',
            re.MULTILINE | re.DOTALL
        )
    
    def extract_references(
        self,
        text: str,
        doc_id: str = "unknown",
        section_map: Optional[Dict[str, Tuple[int, int]]] = None,
        validate: Optional[bool] = None
    ) -> ReferenceExtractionResult:
        """
        Extract all references from document text.
        
        Args:
            text: Document text
            doc_id: Document identifier
            section_map: Optional section boundaries
            validate: Override validation setting (None uses config)
        
        Returns:
            ReferenceExtractionResult with extracted and validated references
        """
        start_time = datetime.now()
        
        result = ReferenceExtractionResult(doc_id=doc_id)
        
        try:
            logger.info(f"Extracting references for {doc_id}")
            
            # Extract numbered references from reference section
            numbered_refs = self._extract_numbered_references(text)
            result.numbered_references = numbered_refs
            logger.info(f"Found {len(numbered_refs)} numbered references")
            
            # Extract identifier references by pattern type
            all_references = []
            
            for ref_type, pattern_info in self.compiled_patterns.items():
                refs = self._extract_by_pattern(
                    text,
                    ref_type,
                    pattern_info,
                    section_map
                )
                all_references.extend(refs)
            
            logger.info(f"Found {len(all_references)} raw identifier references")
            
            # Filter out false INSPIREHEP matches
            all_references = self._filter_false_inspirehep(all_references, text)
            logger.info(f"After INSPIREHEP filtering: {len(all_references)} references")
            
            # Deduplicate references
            unique_refs = self._deduplicate_references(all_references)
            logger.info(f"After deduplication: {len(unique_refs)} unique references")
            
            # Filter by confidence
            filtered_refs = [
                r for r in unique_refs 
                if r.confidence >= self.min_confidence
            ]
            logger.info(f"After confidence filter: {len(filtered_refs)} references")
            
            result.references = filtered_refs
            
            # NEW: Apply AI validation if enabled
            should_validate = validate if validate is not None else self.enable_validation
            if should_validate and self.validator and len(result.references) > 0:
                logger.info("Starting AI validation of extracted references...")
                result = self._apply_validation(result, text)
            
            # Calculate statistics
            self._calculate_statistics(result)
            
            # Infer document type
            self._infer_document_type(result)
            
            # Validate references
            self._validate_references(result)
            
        except Exception as e:
            logger.error(f"Reference extraction failed: {e}", exc_info=True)
            result.warnings.append(f"Extraction error: {str(e)}")
        
        # Record processing time
        end_time = datetime.now()
        result.processing_time_seconds = (end_time - start_time).total_seconds()
        
        logger.info(f"Reference extraction complete for {doc_id}: "
                   f"{result.total_references} identifier references, "
                   f"{len(result.numbered_references)} numbered references")
        
        return result
    
    def _apply_validation(
        self,
        result: ReferenceExtractionResult,
        text: str
    ) -> ReferenceExtractionResult:
        """
        NEW: Apply AI validation to extracted references.
        
        Args:
            result: Extraction result with references
            text: Full document text for context
            
        Returns:
            Updated result with validated references
        """
        try:
            # Convert references to dictionaries for validation
            ref_dicts = [r.to_dict() for r in result.references]
            
            # Validate with Claude AI
            validation_result = self.validator.validate_references(
                references=ref_dicts,
                context_text=text[:5000]  # First 5000 chars for context
            )
            
            logger.info(f"Validation complete: {validation_result.valid_count}/{validation_result.total_references} valid, "
                       f"{validation_result.misclassified_count} misclassified")
            
            # Apply corrections
            corrected_dicts = apply_validation_corrections(
                references=ref_dicts,
                validation_result=validation_result
            )
            
            # Convert back to ExtractedReference objects
            corrected_refs = []
            for ref_dict in corrected_dicts:
                # Find original reference
                original_ref = next(
                    (r for r in result.references if r.reference_id == ref_dict['reference_id']),
                    None
                )
                
                if original_ref:
                    # Update fields that may have been corrected
                    if 'validation_corrected' in ref_dict and ref_dict['validation_corrected']:
                        original_ref.reference_type = ref_dict.get('reference_type', original_ref.reference_type)
                        original_ref.url = ref_dict.get('url', original_ref.url)
                        original_ref.source = ref_dict.get('source', original_ref.source)
                        
                        # Update category if changed
                        if 'category' in ref_dict:
                            try:
                                original_ref.category = ReferenceCategory(ref_dict['category'])
                            except ValueError:
                                pass
                    
                    # Add validation metadata
                    if 'validation' in ref_dict:
                        original_ref.metadata['validation'] = ref_dict['validation']
                    
                    corrected_refs.append(original_ref)
            
            result.references = corrected_refs
            result.validation_applied = True
            result.validation_summary = validation_result.to_dict()['summary']
            
            logger.info(f"Applied validation corrections: {len(result.references)} references remain")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            result.warnings.append(f"Validation error: {str(e)}")
            result.validation_applied = False
        
        return result
    
    def _extract_numbered_references(self, text: str) -> List[Dict]:
        """
        Extract numbered references from reference section.
        Handles formats like: 1. Author A, Author B. Title. Journal. 2020;12:34-56.
        """
        numbered_refs = []
        
        # Find reference section
        ref_section_match = self.reference_section_pattern.search(text)
        if not ref_section_match:
            logger.info("No reference section found")
            return numbered_refs
        
        # Extract text after "References" header
        ref_start = ref_section_match.end()
        ref_text = text[ref_start:]
        
        # Find all numbered references
        for match in self.numbered_ref_pattern.finditer(ref_text):
            ref_num = int(match.group(1))
            ref_content = match.group(2).strip()
            
            # Parse reference content
            parsed = self._parse_reference_content(ref_content)
            
            numbered_refs.append({
                'reference_number': ref_num,
                'raw_text': ref_content,
                'title': parsed.get('title'),
                'journal': parsed.get('journal'),
                'authors': parsed.get('authors'),
                'year': parsed.get('year'),
                'url': parsed.get('url'),
                'doi': parsed.get('doi'),
                'pmid': parsed.get('pmid'),
                'pmcid': parsed.get('pmcid'),
                'start_char': ref_start + match.start(),
                'end_char': ref_start + match.end()
            })
        
        return numbered_refs
    
    def _parse_reference_content(self, ref_text: str) -> Dict[str, Optional[str]]:
        """Parse structured information from reference text."""
        parsed = {
            'title': None,
            'journal': None,
            'authors': None,
            'year': None,
            'url': None,
            'doi': None,
            'pmid': None,
            'pmcid': None
        }
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', ref_text)
        if year_match:
            parsed['year'] = year_match.group(0)
        
        # Extract DOI
        doi_match = re.search(r'10\.\d{4,9}/[^\s]+', ref_text)
        if doi_match:
            parsed['doi'] = doi_match.group(0).rstrip('.')
        
        # Extract PMID
        pmid_match = re.search(r'PMID:?\s*(\d{7,8})', ref_text, re.IGNORECASE)
        if pmid_match:
            parsed['pmid'] = pmid_match.group(1)
        
        # Extract PMCID
        pmc_match = re.search(r'PMC(\d{6,9})', ref_text, re.IGNORECASE)
        if pmc_match:
            parsed['pmcid'] = pmc_match.group(1)
        
        # Extract URL
        url_match = re.search(r'https?://[^\s]+', ref_text)
        if url_match:
            parsed['url'] = url_match.group(0).rstrip('.')
        
        # Extract title
        sentences = ref_text.split('.')
        if len(sentences) >= 2:
            potential_title = sentences[1].strip()
            if len(potential_title) > 10:
                parsed['title'] = potential_title
        
        # Extract journal
        if parsed['year']:
            before_year = ref_text.split(parsed['year'])[0]
            parts = before_year.split('.')
            if len(parts) >= 3:
                parsed['journal'] = parts[-1].strip()
        
        # Extract authors
        if sentences:
            authors = sentences[0].strip()
            if len(authors) < 200:
                parsed['authors'] = authors
        
        return parsed
    
    def _filter_false_inspirehep(
        self, 
        references: List[ExtractedReference],
        text: str
    ) -> List[ExtractedReference]:
        """
        Filter out references misclassified as INSPIREHEP that are actually PMIDs.
        """
        filtered_refs = []
        
        for ref in references:
            if ref.reference_type == 'inspirehep':
                # Check context for PubMed indicators
                context = (ref.preceding_context or "") + " " + (ref.following_context or "")
                context_lower = context.lower()
                
                pmid_indicators = [
                    'pubmed',
                    'ncbi.nlm.nih.gov',
                    'PMID',
                    'PubMed',
                    '/pubmed/',
                    'Available from: https://pubmed'
                ]
                
                if any(indicator.lower() in context_lower for indicator in pmid_indicators):
                    # Reclassify as PMID
                    ref.reference_type = 'pmid'
                    ref.category = ReferenceCategory.LITERATURE
                    ref.source = 'PubMed'
                    ref.url = f'https://pubmed.ncbi.nlm.nih.gov/{ref.normalized_value}/'
                    ref.warnings.append('Reclassified from inspirehep to pmid based on context')
                    logger.debug(f"Reclassified INSPIREHEP → PMID: {ref.normalized_value}")
            
            filtered_refs.append(ref)
        
        return filtered_refs
    
    def _extract_by_pattern(
        self,
        text: str,
        ref_type: str,
        pattern_info: Dict,
        section_map: Optional[Dict[str, Tuple[int, int]]]
    ) -> List[ExtractedReference]:
        """Extract references using a specific pattern."""
        references = []
        regex = pattern_info['regex']
        config = pattern_info['config']
        
        for match in regex.finditer(text):
            try:
                ref = self._create_reference(
                    match,
                    ref_type,
                    config,
                    text,
                    section_map
                )
                
                if ref:
                    references.append(ref)
                    
            except Exception as e:
                logger.warning(f"Failed to create reference for {ref_type}: {e}")
                continue
        
        return references
    
    def _create_reference(
        self,
        match: re.Match,
        ref_type: str,
        config: Dict,
        text: str,
        section_map: Optional[Dict[str, Tuple[int, int]]]
    ) -> Optional[ExtractedReference]:
        """Create a structured reference from a regex match."""
        raw_text = match.group(0)
        start_char = match.start()
        end_char = match.end()
        
        # Normalize value
        try:
            normalize_func = config.get('normalize')
            if normalize_func:
                normalized = normalize_func(raw_text)
            else:
                normalized = raw_text.strip()
        except Exception as e:
            logger.warning(f"Normalization failed for {ref_type}: {e}")
            normalized = raw_text.strip()
        
        # Generate reference ID
        ref_id = f"{ref_type}_{start_char}_{end_char}"
        
        # Determine category
        category_str = config.get('category', 'unknown')
        try:
            category = ReferenceCategory(category_str)
        except ValueError:
            category = ReferenceCategory.UNKNOWN
        
        # Create reference
        reference = ExtractedReference(
            reference_id=ref_id,
            reference_type=ref_type,
            raw_text=raw_text,
            normalized_value=normalized,
            category=category,
            start_char=start_char,
            end_char=end_char,
            source=config.get('source'),
            confidence=config.get('confidence', 0.5),
            is_preprint=config.get('is_preprint', False)
        )
        
        # Build URL
        url_template = config.get('url_template')
        if url_template:
            try:
                url_id = normalized
                if match.groups():
                    url_id = match.group(1) if match.group(1) else normalized
                
                reference.url = url_template.format(id=url_id)
            except Exception as e:
                logger.debug(f"URL construction failed for {ref_type}: {e}")
        
        # Determine section
        if section_map:
            reference.section = self._determine_section(start_char, section_map)
        
        # Extract context
        if self.extract_context:
            self._add_context(reference, text)
        
        # Classify role
        if self.classify_roles:
            reference.role = self._classify_reference_role(reference, text)
        
        # Calculate confidence
        reference.confidence = self._calculate_reference_confidence(reference, text)
        
        # Determine status
        reference.status = self._determine_reference_status(reference)
        
        return reference
    
    def _determine_section(
        self,
        position: int,
        section_map: Dict[str, Tuple[int, int]]
    ) -> Optional[str]:
        """Determine which section a reference belongs to."""
        for section_name, (start, end) in section_map.items():
            if start <= position < end:
                return section_name
        return None
    
    def _add_context(self, reference: ExtractedReference, text: str):
        """Add surrounding context to reference."""
        context_start = max(0, reference.start_char - self.context_window)
        preceding = text[context_start:reference.start_char]
        reference.preceding_context = re.sub(r'\s+', ' ', preceding).strip()
        
        context_end = min(len(text), reference.end_char + self.context_window)
        following = text[reference.end_char:context_end]
        reference.following_context = re.sub(r'\s+', ' ', following).strip()
        
        sentence_start = text.rfind('.', context_start, reference.start_char) + 1
        sentence_end = text.find('.', reference.end_char, context_end)
        if sentence_end == -1:
            sentence_end = context_end
        
        sentence = text[sentence_start:sentence_end + 1]
        reference.sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    def _classify_reference_role(
        self,
        reference: ExtractedReference,
        text: str
    ) -> ReferenceRole:
        """Classify the role of the reference in context."""
        context = reference.preceding_context or ""
        
        role_scores = defaultdict(float)
        
        for role_name, patterns in self.compiled_role_patterns.items():
            for pattern in patterns:
                if pattern.search(context):
                    role_scores[role_name] += 1.0
        
        if reference.reference_type in ['clinicaltrials_gov', 'eudract', 'isrctn']:
            role_scores['trial_registry'] += 2.0
        
        if reference.reference_type in ['nice_guideline', 'kdigo_guideline', 'ich_guideline']:
            role_scores['guideline_reference'] += 2.0
        
        if reference.section and 'method' in reference.section.lower():
            role_scores['methodology'] += 1.0
        
        if reference.section and 'result' in reference.section.lower():
            role_scores['supporting_evidence'] += 1.0
        
        if role_scores:
            best_role = max(role_scores.items(), key=lambda x: x[1])[0]
            try:
                return ReferenceRole(best_role)
            except ValueError:
                return ReferenceRole.GENERAL
        
        return ReferenceRole.GENERAL
    
    def _calculate_reference_confidence(
        self,
        reference: ExtractedReference,
        text: str
    ) -> float:
        """Calculate confidence score for reference."""
        confidence = reference.confidence
        
        if reference.section:
            section_lower = reference.section.lower()
            if 'title' in section_lower or 'abstract' in section_lower:
                confidence += CONFIDENCE_BOOSTERS.get('in_title', 0.1)
            elif 'method' in section_lower:
                confidence += CONFIDENCE_BOOSTERS.get('in_methods', 0.05)
        
        if reference.preceding_context:
            citation_markers = ['see', 'ref', 'cited in', 'according to']
            if any(marker in reference.preceding_context.lower() for marker in citation_markers):
                confidence += CONFIDENCE_BOOSTERS.get('explicit_citation', 0.1)
        
        occurrences = text.count(reference.normalized_value)
        if occurrences > 1:
            confidence += CONFIDENCE_BOOSTERS.get('multiple_mentions', 0.05)
        
        if reference.url:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _determine_reference_status(
        self,
        reference: ExtractedReference
    ) -> ReferenceStatus:
        """Determine extraction status of reference."""
        if reference.url and reference.normalized_value != reference.raw_text.strip():
            return ReferenceStatus.VERIFIED
        elif reference.normalized_value != reference.raw_text.strip():
            return ReferenceStatus.NORMALIZED
        elif reference.confidence >= 0.8:
            return ReferenceStatus.NORMALIZED
        else:
            return ReferenceStatus.RAW
    
    def _deduplicate_references(
        self,
        references: List[ExtractedReference]
    ) -> List[ExtractedReference]:
        """Remove duplicate references, keeping highest confidence."""
        groups: Dict[Tuple[str, str], List[ExtractedReference]] = defaultdict(list)
        
        for ref in references:
            key = (ref.reference_type, ref.normalized_value.lower())
            groups[key].append(ref)
        
        unique_references = []
        for group in groups.values():
            best = max(group, key=lambda r: (r.confidence, -r.start_char))
            unique_references.append(best)
        
        return unique_references
    
    def _calculate_statistics(self, result: ReferenceExtractionResult):
        """Calculate statistics for extraction result."""
        result.total_references = len(result.references)
        result.unique_references = len(set(
            r.normalized_value for r in result.references
        ))
        
        result.references_with_urls = sum(
            1 for r in result.references if r.url
        )
        
        if result.references:
            result.average_confidence = sum(
                r.confidence for r in result.references
            ) / len(result.references)
        
        category_counter = Counter(r.category.value for r in result.references)
        result.category_counts = dict(category_counter)
        
        type_counter = Counter(r.reference_type for r in result.references)
        result.type_counts = dict(type_counter)
        
        result.has_doi = any(r.reference_type == 'doi' for r in result.references)
        result.has_pubmed = any(r.reference_type == 'pmid' for r in result.references)
        result.has_clinical_trial = any(
            r.category == ReferenceCategory.CLINICAL_TRIAL 
            for r in result.references
        )
        result.preprint_count = sum(1 for r in result.references if r.is_preprint)
    
    def _infer_document_type(self, result: ReferenceExtractionResult):
        """Infer document type based on extracted references."""
        hints = []
        
        if result.has_clinical_trial:
            hints.append("clinical_trial_related")
        
        cochrane_refs = [r for r in result.references if r.reference_type == 'cochrane_review']
        prospero_refs = [r for r in result.references if r.reference_type == 'prospero']
        if cochrane_refs or prospero_refs:
            hints.append("systematic_review")
        
        regulatory_refs = [
            r for r in result.references 
            if r.category == ReferenceCategory.REGULATORY
        ]
        if regulatory_refs:
            hints.append("regulatory_document")
        
        guideline_refs = [
            r for r in result.references 
            if r.category == ReferenceCategory.GUIDELINE
        ]
        if guideline_refs:
            hints.append("guideline_document")
        
        if result.has_doi and result.has_pubmed:
            hints.append("research_article")
        
        if result.preprint_count > 0:
            hints.append("includes_preprints")
        
        patent_refs = [
            r for r in result.references 
            if r.category == ReferenceCategory.PATENT
        ]
        if patent_refs:
            hints.append("patent_related")
        
        result.document_type_hints = hints
    
    def _validate_references(self, result: ReferenceExtractionResult):
        """Validate and clean references."""
        valid_references = []
        
        for ref in result.references:
            if not ref.normalized_value or ref.normalized_value.strip() == "":
                ref.warnings.append("Empty normalized value")
                continue
            
            if not 0.0 <= ref.confidence <= 1.0:
                ref.warnings.append(f"Invalid confidence: {ref.confidence}")
                ref.confidence = max(0.0, min(1.0, ref.confidence))
            
            if ref.start_char < 0 or ref.end_char <= ref.start_char:
                ref.warnings.append(f"Invalid position: {ref.start_char}-{ref.end_char}")
                continue
            
            if ref.reference_type == 'doi':
                if not re.match(r'^10\.\d{4,9}/', ref.normalized_value):
                    ref.warnings.append(f"Invalid DOI format: {ref.normalized_value}")
            
            elif ref.reference_type == 'pmid':
                if not ref.normalized_value.isdigit() or len(ref.normalized_value) < 6:
                    ref.warnings.append(f"Invalid PMID: {ref.normalized_value}")
            
            elif ref.reference_type == 'clinicaltrials_gov':
                if not re.match(r'^NCT\d{8}$', ref.normalized_value):
                    ref.warnings.append(f"Invalid NCT: {ref.normalized_value}")
            
            valid_references.append(ref)
        
        result.references = valid_references


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_references_from_text(
    text: str,
    doc_id: str = "unknown",
    config: Optional[Dict] = None,
    validate: bool = True
) -> ReferenceExtractionResult:
    """
    Convenience function to extract references from text.
    
    Args:
        text: Document text
        doc_id: Document identifier
        config: Optional extractor configuration
        validate: Enable AI validation (default: True)
        
    Returns:
        ReferenceExtractionResult
    """
    if config is None:
        config = {'enable_validation': validate}
    elif 'enable_validation' not in config:
        config['enable_validation'] = validate
    
    extractor = ReferenceExtractor(config)
    return extractor.extract_references(text, doc_id, validate=validate)


def references_to_json(result: ReferenceExtractionResult, filepath: str):
    """Save reference extraction result to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(result.to_json())
    
    logger.info(f"References saved to {filepath}")


def get_references_by_category(
    result: ReferenceExtractionResult,
    category: ReferenceCategory
) -> List[ExtractedReference]:
    """Filter references by category."""
    return [r for r in result.references if r.category == category]


def get_references_by_type(
    result: ReferenceExtractionResult,
    ref_type: str
) -> List[ExtractedReference]:
    """Filter references by type."""
    return [r for r in result.references if r.reference_type == ref_type]