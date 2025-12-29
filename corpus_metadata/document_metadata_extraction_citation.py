#!/usr/bin/env python3
"""
Document Metadata Extraction - Citation & Reference Extractor
==============================================================
Location: corpus_metadata/document_metadata_extraction_citation.py
Version: 1.0.0
Last Updated: 2025-10-08

Purpose:
    Extract and normalize bibliographic citations and references from biomedical documents.
    Integrates with entity_citation_patterns.py for comprehensive format support.

Features:
    - Multi-format citation extraction (Vancouver, AMA, Harvard, APA, etc.)
    - Inline citation detection and linking
    - Reference section identification
    - Component normalization (authors, journals, DOIs, etc.)
    - Citation graph construction
    - Confidence scoring and validation

Usage:
    extractor = CitationExtractor()
    result = extractor.extract_citations(document_text, doc_id="PMC123456")
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from collections import defaultdict
import json

# Import citation patterns
try:
    from document_utils.entity_citation_patterns import (
        CITATION_STYLES,
        INLINE_CITATION_PATTERNS,
        REFERENCE_SECTION_MARKERS,
        REFERENCE_COMPONENTS,
        JOURNAL_ABBREVIATIONS,
        CONFIDENCE_SCORING,
        detect_citation_style,
        find_reference_section,
        extract_inline_citations,
        normalize_journal_name,
        parse_reference
    )
except ImportError:
    logging.warning("Could not import entity_citation_patterns. Using fallback patterns.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class CitationType(Enum):
    """Types of citations found in documents."""
    JOURNAL_ARTICLE = "journal_article"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    CONFERENCE = "conference_paper"
    CLINICAL_TRIAL = "clinical_trial"
    REGULATORY = "regulatory_document"
    GUIDELINE = "clinical_guideline"
    WEBSITE = "website"
    PATENT = "patent"
    PREPRINT = "preprint"
    THESIS = "thesis"
    DATABASE = "database"
    SOFTWARE = "software"
    UNKNOWN = "unknown"


class CitationStatus(Enum):
    """Citation extraction status."""
    COMPLETE = "complete"
    PARTIAL = "partial"
    MINIMAL = "minimal"
    FAILED = "failed"


@dataclass
class Author:
    """Structured author representation."""
    last_name: str
    first_name: Optional[str] = None
    middle_name: Optional[str] = None
    initials: Optional[str] = None
    suffix: Optional[str] = None
    orcid: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Citation:
    """Structured citation representation."""
    # Core identification
    citation_id: str  # e.g., "ref_1", "smith2021"
    raw_text: str
    citation_type: CitationType
    citation_style: str  # vancouver, apa, harvard, etc.
    
    # Bibliographic components
    authors: List[Author] = field(default_factory=list)
    title: Optional[str] = None
    journal: Optional[str] = None
    journal_abbrev: Optional[str] = None
    year: Optional[int] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    article_id: Optional[str] = None  # For electronic articles
    
    # Identifiers
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmc_id: Optional[str] = None
    isbn: Optional[str] = None
    url: Optional[str] = None
    
    # Registry IDs for clinical trials
    nct_id: Optional[str] = None
    eudract_id: Optional[str] = None
    
    # Document structure
    section: Optional[str] = None
    reference_number: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    # Metadata
    confidence: float = 0.0
    status: CitationStatus = CitationStatus.PARTIAL
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Inline citations linking to this reference
    inline_citation_positions: List[int] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        result = {
            'citation_id': self.citation_id,
            'raw_text': self.raw_text,
            'citation_type': self.citation_type.value,
            'citation_style': self.citation_style,
            'authors': [a.to_dict() for a in self.authors],
            'confidence': round(self.confidence, 3),
            'status': self.status.value,
            'extraction_timestamp': self.extraction_timestamp
        }
        
        # Add optional fields
        optional_fields = [
            'title', 'journal', 'journal_abbrev', 'year', 'volume', 
            'issue', 'pages', 'article_id', 'doi', 'pmid', 'pmc_id',
            'isbn', 'url', 'nct_id', 'eudract_id', 'section', 
            'reference_number', 'start_char', 'end_char'
        ]
        
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        
        if self.inline_citation_positions:
            result['inline_citation_positions'] = self.inline_citation_positions
        
        if self.metadata:
            result['metadata'] = self.metadata
        
        if self.warnings:
            result['warnings'] = self.warnings
        
        return result


@dataclass
class InlineCitation:
    """Inline citation reference within text."""
    citation_text: str
    citation_style: str
    reference_ids: List[str]  # Links to Citation objects
    start_char: int
    end_char: int
    context: Optional[str] = None
    section: Optional[str] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CitationExtractionResult:
    """Complete citation extraction result."""
    doc_id: str
    language: str = "en"
    
    # Citations
    citations: List[Citation] = field(default_factory=list)
    inline_citations: List[InlineCitation] = field(default_factory=list)
    
    # Analysis
    dominant_citation_style: Optional[str] = None
    citation_style_confidence: Dict[str, float] = field(default_factory=dict)
    reference_section_found: bool = False
    reference_section_bounds: Optional[Tuple[int, int]] = None
    
    # Statistics
    total_citations: int = 0
    total_inline_citations: int = 0
    citations_with_doi: int = 0
    citations_with_pmid: int = 0
    average_confidence: float = 0.0
    
    # Metadata
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'doc_id': self.doc_id,
            'language': self.language,
            'citations': [c.to_dict() for c in self.citations],
            'inline_citations': [ic.to_dict() for ic in self.inline_citations],
            'dominant_citation_style': self.dominant_citation_style,
            'citation_style_confidence': self.citation_style_confidence,
            'reference_section_found': self.reference_section_found,
            'reference_section_bounds': self.reference_section_bounds,
            'statistics': {
                'total_citations': self.total_citations,
                'total_inline_citations': self.total_inline_citations,
                'citations_with_doi': self.citations_with_doi,
                'citations_with_pmid': self.citations_with_pmid,
                'average_confidence': round(self.average_confidence, 3)
            },
            'extraction_timestamp': self.extraction_timestamp,
            'processing_time_seconds': round(self.processing_time_seconds, 3),
            'warnings': self.warnings,
            'metadata': self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ============================================================================
# CITATION EXTRACTOR
# ============================================================================

class CitationExtractor:
    """
    Extract and normalize citations from biomedical documents.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the citation extractor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.extract_inline = self.config.get('extract_inline', True)
        self.link_inline_to_refs = self.config.get('link_inline_to_refs', True)
        
        # Compile patterns
        self._compile_identifier_patterns()
        
        logger.info(f"CitationExtractor initialized with config: {self.config}")
    
    def _compile_identifier_patterns(self):
        """Compile regex patterns for identifier extraction."""
        self.doi_pattern = re.compile(
            r'(?:doi:|DOI:|https?://(?:dx\.)?doi\.org/)?\s*'
            r'(10\.\d{4,9}/[^\s\]]+)',
            re.IGNORECASE
        )
        
        self.pmid_pattern = re.compile(
            r'(?:PMID:?\s*|PubMed:?\s*)(\d{7,8})',
            re.IGNORECASE
        )
        
        self.pmc_pattern = re.compile(
            r'(?:PMCID:?\s*)?PMC(\d{6,8})',
            re.IGNORECASE
        )
        
        self.nct_pattern = re.compile(r'\b(NCT\d{8})\b')
        self.eudract_pattern = re.compile(r'\b(\d{4}-\d{6}-\d{2})\b')
        
        self.isbn_pattern = re.compile(
            r'ISBN:?\s*((?:\d{1,5}[-\s]?){4}\d{1,5}|'
            r'(?:\d{1,5}[-\s]?){3}\d{1,5}[-\s]?\d{1,5})',
            re.IGNORECASE
        )
        
        self.url_pattern = re.compile(
            r'https?://[^\s\]]+',
            re.IGNORECASE
        )
    
    def extract_citations(
        self, 
        text: str, 
        doc_id: str = "unknown",
        section_map: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> CitationExtractionResult:
        """
        Extract all citations from document text.
        
        Args:
            text: Document text
            doc_id: Document identifier
            section_map: Optional mapping of section names to character positions
            
        Returns:
            CitationExtractionResult object
        """
        start_time = datetime.now()
        
        result = CitationExtractionResult(doc_id=doc_id)
        
        try:
            # 1. Detect citation style
            logger.info(f"Detecting citation style for {doc_id}")
            style_scores = detect_citation_style(text)
            if style_scores:
                result.citation_style_confidence = style_scores
                result.dominant_citation_style = max(
                    style_scores.items(), 
                    key=lambda x: x[1]
                )[0]
                logger.info(f"Dominant style: {result.dominant_citation_style}")
            
            # 2. Find reference section
            logger.info("Locating reference section")
            ref_section = find_reference_section(text)
            if ref_section:
                result.reference_section_found = True
                result.reference_section_bounds = ref_section
                logger.info(f"Reference section found: {ref_section}")
            else:
                result.warnings.append("No clear reference section found")
                logger.warning("Reference section not identified")
            
            # 3. Extract structured citations from reference section
            if ref_section:
                ref_text = text[ref_section[0]:ref_section[1]]
                citations = self._extract_structured_citations(
                    ref_text,
                    ref_section[0],
                    result.dominant_citation_style
                )
                result.citations.extend(citations)
                logger.info(f"Extracted {len(citations)} structured citations")
            
            # 4. Extract inline citations
            if self.extract_inline:
                logger.info("Extracting inline citations")
                inline_cites = self._extract_inline_citations(text, section_map)
                result.inline_citations.extend(inline_cites)
                logger.info(f"Extracted {len(inline_cites)} inline citations")
            
            # 5. Link inline citations to references
            if self.link_inline_to_refs and result.citations and result.inline_citations:
                logger.info("Linking inline citations to references")
                self._link_inline_to_references(result)
            
            # 6. Calculate statistics
            self._calculate_statistics(result)
            
            # 7. Validate and clean
            self._validate_citations(result)
            
        except Exception as e:
            logger.error(f"Citation extraction failed: {e}", exc_info=True)
            result.warnings.append(f"Extraction error: {str(e)}")
        
        # Record processing time
        end_time = datetime.now()
        result.processing_time_seconds = (end_time - start_time).total_seconds()
        
        logger.info(f"Citation extraction complete for {doc_id}: "
                   f"{result.total_citations} citations, "
                   f"{result.total_inline_citations} inline references")
        
        return result
    
    def _extract_structured_citations(
        self,
        text: str,
        offset: int,
        dominant_style: Optional[str]
    ) -> List[Citation]:
        """
        Extract structured citations from reference section.
        
        Args:
            text: Reference section text
            offset: Character offset of reference section in document
            dominant_style: Detected dominant citation style
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        # Try patterns matching dominant style first
        style_priority = [dominant_style] if dominant_style else []
        style_priority.extend([s for s in ['vancouver', 'ama', 'apa', 'harvard'] 
                               if s != dominant_style])
        
        for style in style_priority:
            style_patterns = {
                name: config for name, config in CITATION_STYLES.items()
                if config['style'] == style
            }
            
            for pattern_name, pattern_config in style_patterns.items():
                try:
                    pattern = re.compile(
                        pattern_config['pattern'],
                        re.MULTILINE | re.IGNORECASE
                    )
                    
                    for match in pattern.finditer(text):
                        citation = self._parse_citation_match(
                            match,
                            pattern_config,
                            offset
                        )
                        
                        if citation and citation.confidence >= self.min_confidence:
                            citations.append(citation)
                            
                except re.error as e:
                    logger.warning(f"Pattern {pattern_name} failed: {e}")
                    continue
        
        # Deduplicate citations
        citations = self._deduplicate_citations(citations)
        
        return citations
    
    def _parse_citation_match(
        self,
        match: re.Match,
        pattern_config: Dict,
        offset: int
    ) -> Optional[Citation]:
        """
        Parse a regex match into a Citation object.
        
        Args:
            match: Regex match object
            pattern_config: Pattern configuration from CITATION_STYLES
            offset: Character offset in document
            
        Returns:
            Citation object or None
        """
        try:
            raw_text = match.group(0)
            groups = match.groupdict()
            
            # Generate citation ID
            ref_num = groups.get('number')
            authors_text = groups.get('authors', '')
            year = groups.get('year')
            
            if ref_num:
                citation_id = f"ref_{ref_num}"
            elif authors_text and year:
                first_author = authors_text.split(',')[0].split()[-1].lower()
                citation_id = f"{first_author}{year}"
            else:
                citation_id = f"ref_{len(raw_text)}"
            
            # Create citation
            citation = Citation(
                citation_id=citation_id,
                raw_text=raw_text,
                citation_type=CitationType.JOURNAL_ARTICLE,  # Default
                citation_style=pattern_config['style'],
                start_char=offset + match.start(),
                end_char=offset + match.end(),
                reference_number=int(ref_num) if ref_num else None
            )
            
            # Parse authors
            if 'authors' in groups:
                citation.authors = self._parse_authors(groups['authors'])
            
            # Extract components
            citation.title = groups.get('title')
            citation.journal = groups.get('journal')
            citation.year = int(groups['year']) if groups.get('year') else None
            citation.volume = groups.get('volume')
            citation.issue = groups.get('issue')
            citation.pages = groups.get('pages')
            citation.article_id = groups.get('article_id')
            
            # Normalize journal name
            if citation.journal:
                citation.journal_abbrev = citation.journal
                full_name = normalize_journal_name(citation.journal)
                if full_name:
                    citation.journal = full_name
            
            # Extract identifiers from raw text
            self._extract_identifiers(raw_text, citation)
            
            # Calculate confidence
            citation.confidence = self._calculate_citation_confidence(
                citation,
                pattern_config
            )
            
            # Determine status
            citation.status = self._determine_citation_status(citation)
            
            return citation
            
        except Exception as e:
            logger.error(f"Failed to parse citation: {e}", exc_info=True)
            return None
    
    def _parse_authors(self, author_text: str) -> List[Author]:
        """
        Parse author string into structured Author objects.
        
        Args:
            author_text: Author string from citation
            
        Returns:
            List of Author objects
        """
        authors = []
        
        # Handle "et al."
        if 'et al' in author_text.lower():
            author_text = re.sub(r',?\s+et\s+al\.?', '', author_text, flags=re.IGNORECASE)
        
        # Split authors
        author_parts = re.split(r'[,;](?:\s+and\s+|(?=[A-Z]))', author_text)
        
        for part in author_parts[:20]:  # Limit to 20 authors
            part = part.strip()
            if not part:
                continue
            
            author = self._parse_single_author(part)
            if author:
                authors.append(author)
        
        return authors
    
    def _parse_single_author(self, author_str: str) -> Optional[Author]:
        """
        Parse a single author string.
        
        Args:
            author_str: Single author string
            
        Returns:
            Author object or None
        """
        author_str = author_str.strip()
        
        # Pattern: LastName FirstInitial(s)
        # e.g., "Smith JA", "Jones BC"
        match = re.match(r'^([A-Z][a-z]+)\s+([A-Z]{1,3})$', author_str)
        if match:
            return Author(
                last_name=match.group(1),
                initials=match.group(2)
            )
        
        # Pattern: LastName, FirstInitial(s)
        # e.g., "Smith, J.A.", "Jones, B. C."
        match = re.match(r'^([A-Z][a-z]+),\s+([A-Z]\.?\s*[A-Z]?\.?)$', author_str)
        if match:
            return Author(
                last_name=match.group(1),
                initials=match.group(2).replace('.', '').replace(' ', '')
            )
        
        # Pattern: FirstInitial(s) LastName
        # e.g., "J. A. Smith"
        match = re.match(r'^([A-Z]\.?\s*[A-Z]?\.?)\s+([A-Z][a-z]+)$', author_str)
        if match:
            return Author(
                last_name=match.group(2),
                initials=match.group(1).replace('.', '').replace(' ', '')
            )
        
        # Fallback: treat as last name
        if re.match(r'^[A-Z][a-z]+$', author_str):
            return Author(last_name=author_str)
        
        return None
    
    def _extract_identifiers(self, text: str, citation: Citation):
        """
        Extract identifiers (DOI, PMID, etc.) from citation text.
        
        Args:
            text: Citation text
            citation: Citation object to update
        """
        # DOI
        doi_match = self.doi_pattern.search(text)
        if doi_match:
            citation.doi = doi_match.group(1).rstrip('.')
        
        # PMID
        pmid_match = self.pmid_pattern.search(text)
        if pmid_match:
            citation.pmid = pmid_match.group(1)
        
        # PMC
        pmc_match = self.pmc_pattern.search(text)
        if pmc_match:
            citation.pmc_id = pmc_match.group(1)
        
        # Clinical trial IDs
        nct_match = self.nct_pattern.search(text)
        if nct_match:
            citation.nct_id = nct_match.group(1)
            citation.citation_type = CitationType.CLINICAL_TRIAL
        
        eudract_match = self.eudract_pattern.search(text)
        if eudract_match:
            citation.eudract_id = eudract_match.group(1)
            citation.citation_type = CitationType.CLINICAL_TRIAL
        
        # ISBN
        isbn_match = self.isbn_pattern.search(text)
        if isbn_match:
            citation.isbn = isbn_match.group(1)
            citation.citation_type = CitationType.BOOK
        
        # URL
        url_match = self.url_pattern.search(text)
        if url_match:
            citation.url = url_match.group(0)
    
    def _calculate_citation_confidence(
        self,
        citation: Citation,
        pattern_config: Dict
    ) -> float:
        """
        Calculate confidence score for citation.
        
        Args:
            citation: Citation object
            pattern_config: Pattern configuration
            
        Returns:
            Confidence score [0.0-1.0]
        """
        # Start with pattern base confidence
        confidence = pattern_config.get('confidence', 0.5)
        
        # Apply scoring rules
        if citation.doi:
            confidence += CONFIDENCE_SCORING.get('has_doi', 0.15)
        if citation.pmid:
            confidence += CONFIDENCE_SCORING.get('has_pmid', 0.15)
        if citation.pmc_id:
            confidence += CONFIDENCE_SCORING.get('has_pmc', 0.10)
        if citation.volume and citation.pages:
            confidence += CONFIDENCE_SCORING.get('has_volume_pages', 0.10)
        if citation.journal and citation.journal in JOURNAL_ABBREVIATIONS.values():
            confidence += CONFIDENCE_SCORING.get('recognized_journal', 0.10)
        if len(citation.authors) > 0:
            confidence += CONFIDENCE_SCORING.get('complete_author_list', 0.08)
        if citation.title:
            confidence += CONFIDENCE_SCORING.get('has_title', 0.08)
        if citation.reference_number:
            confidence += CONFIDENCE_SCORING.get('numbered_format', 0.05)
        if citation.year:
            confidence += CONFIDENCE_SCORING.get('has_year', 0.05)
        
        return min(confidence, 1.0)
    
    def _determine_citation_status(self, citation: Citation) -> CitationStatus:
        """
        Determine completeness status of citation.
        
        Args:
            citation: Citation object
            
        Returns:
            CitationStatus enum
        """
        required_fields = [
            citation.authors,
            citation.title or citation.journal,
            citation.year
        ]
        
        optional_fields = [
            citation.volume,
            citation.pages,
            citation.doi or citation.pmid
        ]
        
        required_present = sum(1 for f in required_fields if f)
        optional_present = sum(1 for f in optional_fields if f)
        
        if required_present == len(required_fields) and optional_present >= 2:
            return CitationStatus.COMPLETE
        elif required_present >= 2:
            return CitationStatus.PARTIAL
        elif required_present >= 1:
            return CitationStatus.MINIMAL
        else:
            return CitationStatus.FAILED
    
    def _extract_inline_citations(
        self,
        text: str,
        section_map: Optional[Dict[str, Tuple[int, int]]]
    ) -> List[InlineCitation]:
        """
        Extract inline citations from text.
        
        Args:
            text: Document text
            section_map: Optional section boundaries
            
        Returns:
            List of InlineCitation objects
        """
        inline_cites = []
        
        # Extract using patterns from entity_citation_patterns
        raw_cites = extract_inline_citations(text)
        
        for cite_dict in raw_cites:
            # Determine section
            section = None
            if section_map:
                for sec_name, (sec_start, sec_end) in section_map.items():
                    if sec_start <= cite_dict['start'] < sec_end:
                        section = sec_name
                        break
            
            # Extract reference IDs
            ref_ids = self._parse_inline_reference_ids(cite_dict)
            
            # Create InlineCitation
            inline_cite = InlineCitation(
                citation_text=cite_dict['text'],
                citation_style=cite_dict['style'],
                reference_ids=ref_ids,
                start_char=cite_dict['start'],
                end_char=cite_dict['end'],
                section=section,
                confidence=cite_dict['confidence'],
                context=self._extract_context(text, cite_dict['start'], cite_dict['end'])
            )
            
            inline_cites.append(inline_cite)
        
        return inline_cites
    
    def _parse_inline_reference_ids(self, cite_dict: Dict) -> List[str]:
        """
        Parse reference IDs from inline citation.
        
        Args:
            cite_dict: Citation dictionary from extract_inline_citations
            
        Returns:
            List of reference IDs
        """
        ref_ids = []
        
        # Numbered citations
        if 'numbers' in cite_dict:
            numbers_str = cite_dict['numbers']
            # Handle ranges: "1-3" -> ["1", "2", "3"]
            # Handle lists: "1,2,5" -> ["1", "2", "5"]
            parts = re.split(r'[,\s]+', numbers_str)
            
            for part in parts:
                if '-' in part or '-' in part:
                    # Range
                    start, end = re.split(r'[--]', part)
                    try:
                        for num in range(int(start), int(end) + 1):
                            ref_ids.append(f"ref_{num}")
                    except ValueError:
                        continue
                else:
                    # Single number
                    try:
                        ref_ids.append(f"ref_{int(part)}")
                    except ValueError:
                        continue
        
        # Author-year citations
        elif 'authors' in cite_dict and 'year' in cite_dict:
            author = cite_dict['authors'].split()[0].lower()
            year = cite_dict['year']
            ref_ids.append(f"{author}{year}")
        
        return ref_ids
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """
        Extract context around inline citation.
        
        Args:
            text: Document text
            start: Citation start position
            end: Citation end position
            window: Context window size
            
        Returns:
            Context string
        """
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        
        context = text[context_start:context_end]
        context = re.sub(r'\s+', ' ', context).strip()
        
        return context
    
    def _link_inline_to_references(self, result: CitationExtractionResult):
        """
        Link inline citations to reference list.
        
        Args:
            result: CitationExtractionResult to update
        """
        # Build reference lookup
        ref_lookup = {c.citation_id: c for c in result.citations}
        
        for inline_cite in result.inline_citations:
            for ref_id in inline_cite.reference_ids:
                if ref_id in ref_lookup:
                    ref_lookup[ref_id].inline_citation_positions.append(
                        inline_cite.start_char
                    )
    
    def _deduplicate_citations(self, citations: List[Citation]) -> List[Citation]:
        """
        Remove duplicate citations, keeping highest confidence.
        
        Args:
            citations: List of citations
            
        Returns:
            Deduplicated list
        """
        # Group by DOI, PMID, or similar raw text
        groups: Dict[str, List[Citation]] = defaultdict(list)
        
        for cite in citations:
            if cite.doi:
                key = f"doi:{cite.doi}"
            elif cite.pmid:
                key = f"pmid:{cite.pmid}"
            else:
                # Use normalized raw text
                key = re.sub(r'\s+', ' ', cite.raw_text[:100]).strip()
            
            groups[key].append(cite)
        
        # Keep best from each group
        unique_citations = []
        for group in groups.values():
            best = max(group, key=lambda c: c.confidence)
            unique_citations.append(best)
        
        return unique_citations
    
    def _calculate_statistics(self, result: CitationExtractionResult):
        """
        Calculate statistics for extraction result.
        
        Args:
            result: CitationExtractionResult to update
        """
        result.total_citations = len(result.citations)
        result.total_inline_citations = len(result.inline_citations)
        
        result.citations_with_doi = sum(1 for c in result.citations if c.doi)
        result.citations_with_pmid = sum(1 for c in result.citations if c.pmid)
        
        if result.citations:
            result.average_confidence = sum(
                c.confidence for c in result.citations
            ) / len(result.citations)
    
    def _validate_citations(self, result: CitationExtractionResult):
        """
        Validate and clean citations.
        
        Args:
            result: CitationExtractionResult to validate
        """
        valid_citations = []
        
        for citation in result.citations:
            # Check minimum requirements
            if not citation.authors and not citation.title:
                citation.warnings.append("Missing both authors and title")
                continue
            
            # Validate year
            if citation.year:
                if citation.year < 1800 or citation.year > datetime.now().year + 1:
                    citation.warnings.append(f"Invalid year: {citation.year}")
                    citation.year = None
            
            # Validate DOI format
            if citation.doi:
                if not re.match(r'^10\.\d{4,9}/', citation.doi):
                    citation.warnings.append(f"Invalid DOI format: {citation.doi}")
                    citation.doi = None
            
            valid_citations.append(citation)
        
        result.citations = valid_citations


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_citations_from_text(
    text: str,
    doc_id: str = "unknown",
    config: Optional[Dict] = None
) -> CitationExtractionResult:
    """
    Convenience function to extract citations from text.
    
    Args:
        text: Document text
        doc_id: Document identifier
        config: Optional extractor configuration
        
    Returns:
        CitationExtractionResult
    """
    extractor = CitationExtractor(config)
    return extractor.extract_citations(text, doc_id)


def citations_to_json(result: CitationExtractionResult, filepath: str):
    """
    Save citation extraction result to JSON file.
    
    Args:
        result: CitationExtractionResult
        filepath: Output file path
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(result.to_json())
    
    logger.info(f"Citations saved to {filepath}")
