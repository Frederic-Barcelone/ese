#!/usr/bin/env python3
"""
Document Metadata Extraction - Person & Author Extractor
=========================================================
Location: corpus_metadata/document_metadata_extraction_persons.py
Version: 1.0.0
Last Updated: 2025-10-08

Purpose:
    Extract and normalize person entities (authors, investigators, experts) from biomedical documents.
    Integrates with entity_person_patterns.py for comprehensive Unicode and multilingual support.

Features:
    - Multi-role extraction (authors, PIs, co-investigators, experts)
    - Unicode support for diacritics (García-López, O'Connor)
    - Surname particle handling (van der Berg, de la Cruz, ibn Rushd)
    - ORCID validation and extraction
    - Affiliation matching and linking
    - Author deduplication and name normalization
    - Confidence scoring based on context

Usage:
    extractor = PersonExtractor()
    result = extractor.extract_persons(document_text, doc_id="PMC123456")
"""

import re
import logging
import unicodedata
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from collections import defaultdict, Counter
import json

# Import person patterns
try:
    from document_utils.entity_person_patterns import (
        PERSON_NAME_PATTERNS,
        AFFILIATION_PATTERNS,
        ROLE_CLASSIFICATION_TRIGGERS,
        NAME_NORMALIZATION_RULES,
        CONFIDENCE_ADJUSTMENTS,
        CONTEXT_SECTIONS,
        get_person_pattern_stats,
        get_patterns_by_role,
        get_all_roles,
        normalize_text_for_person_matching,
        validate_orcid,
        extract_person_components
    )
except ImportError:
    logging.warning("Could not import entity_person_patterns. Using fallback patterns.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class PersonRole(Enum):
    """Roles of persons in documents."""
    AUTHOR = "author"
    PRINCIPAL_INVESTIGATOR = "principal_investigator"
    CO_INVESTIGATOR = "co_investigator"
    CORRESPONDING_AUTHOR = "corresponding_author"
    STUDY_COORDINATOR = "study_coordinator"
    EXPERT_OPINION = "expert_opinion"
    GUIDELINE_AUTHOR = "guideline_author"
    REVIEWER = "reviewer"
    EDITOR = "editor"
    UNKNOWN = "unknown"


class PersonStatus(Enum):
    """Status of person extraction."""
    VERIFIED = "verified"  # Has ORCID or email
    NORMALIZED = "normalized"  # Name normalized
    RAW = "raw"  # As extracted
    AMBIGUOUS = "ambiguous"  # Multiple possible matches


@dataclass
class PersonName:
    """Structured person name components."""
    full_name: str
    surname: Optional[str] = None
    given_names: Optional[str] = None
    initials: Optional[str] = None
    particles: List[str] = field(default_factory=list)  # van, de, etc.
    titles: List[str] = field(default_factory=list)  # Dr., Prof.
    degrees: List[str] = field(default_factory=list)  # MD, PhD
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class Affiliation:
    """Affiliation/institution information."""
    affiliation_id: str
    raw_text: str
    normalized_name: Optional[str] = None
    affiliation_type: Optional[str] = None  # academic, clinical, research
    department: Optional[str] = None
    institution: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ExtractedPerson:
    """Structured representation of an extracted person."""
    # Core identification
    person_id: str  # Unique within document
    name: PersonName
    role: PersonRole
    
    # Location in document
    start_char: int
    end_char: int
    section: Optional[str] = None
    sentence: Optional[str] = None
    
    # Identifiers
    orcid: Optional[str] = None
    email: Optional[str] = None
    
    # Affiliations
    affiliations: List[Affiliation] = field(default_factory=list)
    
    # Context
    preceding_context: Optional[str] = None
    following_context: Optional[str] = None
    
    # Metadata
    confidence: float = 0.0
    status: PersonStatus = PersonStatus.RAW
    extraction_method: Optional[str] = None  # Pattern name used
    
    # Validation
    is_verified: bool = False  # Has ORCID or email
    mention_count: int = 1
    
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Additional metadata
    metadata: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        result = {
            'person_id': self.person_id,
            'name': self.name.to_dict(),
            'role': self.role.value,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'confidence': round(self.confidence, 3),
            'status': self.status.value,
            'mention_count': self.mention_count,
            'extraction_timestamp': self.extraction_timestamp
        }
        
        # Optional fields
        optional_fields = ['section', 'sentence', 'orcid', 'email', 
                          'preceding_context', 'following_context', 'extraction_method']
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        
        if self.affiliations:
            result['affiliations'] = [a.to_dict() for a in self.affiliations]
        
        if self.is_verified:
            result['is_verified'] = True
        
        if self.metadata:
            result['metadata'] = self.metadata
        
        if self.warnings:
            result['warnings'] = self.warnings
        
        return result


@dataclass
class PersonExtractionResult:
    """Complete person extraction result."""
    doc_id: str
    language: str = "en"
    
    # Extracted persons
    persons: List[ExtractedPerson] = field(default_factory=list)
    affiliations: List[Affiliation] = field(default_factory=list)
    
    # Statistics by role
    role_counts: Dict[str, int] = field(default_factory=dict)
    
    # Analysis
    total_persons: int = 0
    unique_persons: int = 0
    verified_persons: int = 0  # With ORCID/email
    average_confidence: float = 0.0
    
    # Author-specific metrics
    author_count: int = 0
    has_corresponding_author: bool = False
    has_principal_investigator: bool = False
    
    # Network analysis
    co_author_network: Dict[str, List[str]] = field(default_factory=dict)
    
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
            'persons': [p.to_dict() for p in self.persons],
            'affiliations': [a.to_dict() for a in self.affiliations],
            'statistics': {
                'total_persons': self.total_persons,
                'unique_persons': self.unique_persons,
                'verified_persons': self.verified_persons,
                'average_confidence': round(self.average_confidence, 3),
                'role_counts': self.role_counts,
                'author_count': self.author_count,
                'has_corresponding_author': self.has_corresponding_author,
                'has_principal_investigator': self.has_principal_investigator
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
# PERSON EXTRACTOR
# ============================================================================

class PersonExtractor:
    """
    Extract and normalize person entities from biomedical documents.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the person extractor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.extract_context = self.config.get('extract_context', True)
        self.context_window = self.config.get('context_window', 100)
        self.extract_affiliations = self.config.get('extract_affiliations', True)
        self.validate_orcid_checksums = self.config.get('validate_orcid_checksums', True)
        
        # Compile patterns
        self._compile_patterns()
        
        logger.info(f"PersonExtractor initialized with config: {self.config}")
    
    def _compile_patterns(self):
        """Compile all regex patterns for efficiency."""
        self.compiled_patterns = {}
        
        for pattern_key, pattern_config in PERSON_NAME_PATTERNS.items():
            try:
                pattern = pattern_config['pattern']
                self.compiled_patterns[pattern_key] = {
                    'regex': re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.UNICODE),
                    'config': pattern_config
                }
            except re.error as e:
                logger.warning(f"Failed to compile pattern for {pattern_key}: {e}")
                continue
        
        logger.info(f"Compiled {len(self.compiled_patterns)} person patterns")
        
        # Compile affiliation patterns
        self.compiled_affiliation_patterns = {}
        for affil_key, affil_config in AFFILIATION_PATTERNS.items():
            try:
                pattern = affil_config['pattern']
                self.compiled_affiliation_patterns[affil_key] = {
                    'regex': re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.UNICODE),
                    'config': affil_config
                }
            except re.error as e:
                logger.warning(f"Failed to compile affiliation pattern for {affil_key}: {e}")
                continue
        
        # Compile role triggers
        self.compiled_role_triggers = {}
        for role, patterns in ROLE_CLASSIFICATION_TRIGGERS.items():
            self.compiled_role_triggers[role] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def extract_persons(
        self,
        text: str,
        doc_id: str = "unknown",
        section_map: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> PersonExtractionResult:
        """
        Extract all persons from document text.
        
        Args:
            text: Document text
            doc_id: Document identifier
            section_map: Optional mapping of section names to character positions
            
        Returns:
            PersonExtractionResult object
        """
        start_time = datetime.now()
        
        result = PersonExtractionResult(doc_id=doc_id)
        
        try:
            # Normalize text for Unicode matching
            normalized_text = normalize_text_for_person_matching(text)
            
            # Extract persons by pattern type
            logger.info(f"Extracting persons for {doc_id}")
            all_persons = []
            
            for pattern_key, pattern_info in self.compiled_patterns.items():
                persons = self._extract_by_pattern(
                    normalized_text,
                    pattern_key,
                    pattern_info,
                    section_map
                )
                all_persons.extend(persons)
            
            logger.info(f"Found {len(all_persons)} raw person mentions")
            
            # Deduplicate persons
            unique_persons = self._deduplicate_persons(all_persons, normalized_text)
            logger.info(f"After deduplication: {len(unique_persons)} unique persons")
            
            # Extract affiliations
            if self.extract_affiliations:
                affiliations = self._extract_affiliations(normalized_text)
                result.affiliations = affiliations
                logger.info(f"Extracted {len(affiliations)} affiliations")
                
                # Link persons to affiliations
                self._link_persons_to_affiliations(unique_persons, affiliations, normalized_text)
            
            # Filter by confidence
            filtered_persons = [
                p for p in unique_persons 
                if p.confidence >= self.min_confidence
            ]
            logger.info(f"After confidence filter: {len(filtered_persons)} persons")
            
            result.persons = filtered_persons
            
            # Calculate statistics
            self._calculate_statistics(result)
            
            # Build co-author network
            self._build_coauthor_network(result)
            
            # Validate persons
            self._validate_persons(result)
            
        except Exception as e:
            logger.error(f"Person extraction failed: {e}", exc_info=True)
            result.warnings.append(f"Extraction error: {str(e)}")
        
        # Record processing time
        end_time = datetime.now()
        result.processing_time_seconds = (end_time - start_time).total_seconds()
        
        logger.info(f"Person extraction complete for {doc_id}: "
                   f"{result.total_persons} persons")
        
        return result
    
    def _extract_by_pattern(
        self,
        text: str,
        pattern_key: str,
        pattern_info: Dict,
        section_map: Optional[Dict[str, Tuple[int, int]]]
    ) -> List[ExtractedPerson]:
        """
        Extract persons using a specific pattern.
        
        Args:
            text: Document text
            pattern_key: Pattern key
            pattern_info: Pattern configuration
            section_map: Optional section boundaries
            
        Returns:
            List of ExtractedPerson objects
        """
        persons = []
        regex = pattern_info['regex']
        config = pattern_info['config']
        
        for match in regex.finditer(text):
            try:
                person = self._create_person(
                    match,
                    pattern_key,
                    config,
                    text,
                    section_map
                )
                
                if person:
                    persons.append(person)
                    
            except Exception as e:
                logger.warning(f"Failed to create person for {pattern_key}: {e}")
                continue
        
        return persons
    
    def _create_person(
        self,
        match: re.Match,
        pattern_key: str,
        config: Dict,
        text: str,
        section_map: Optional[Dict[str, Tuple[int, int]]]
    ) -> Optional[ExtractedPerson]:
        """
        Create a structured person from a regex match.
        
        Args:
            match: Regex match object
            pattern_key: Pattern key
            config: Pattern configuration
            text: Full document text
            section_map: Optional section boundaries
            
        Returns:
            ExtractedPerson object or None
        """
        raw_text = match.group(0)
        start_char = match.start()
        end_char = match.end()
        
        # Extract name components from match groups
        capture_groups = config.get('capture_groups', {})
        groups_dict = match.groupdict()
        
        # Build person name
        name = self._build_person_name(raw_text, groups_dict, capture_groups)
        
        if not name:
            return None
        
        # Generate person ID
        person_id = self._generate_person_id(name, start_char)
        
        # Determine role
        role_str = config.get('role', 'author')
        try:
            role = PersonRole(role_str)
        except ValueError:
            role = PersonRole.UNKNOWN
        
        # Create person
        person = ExtractedPerson(
            person_id=person_id,
            name=name,
            role=role,
            start_char=start_char,
            end_char=end_char,
            confidence=config.get('confidence', 0.5),
            extraction_method=pattern_key
        )
        
        # Extract ORCID if present
        if 'orcid' in groups_dict and groups_dict['orcid']:
            orcid = groups_dict['orcid']
            if self.validate_orcid_checksums:
                if validate_orcid(orcid):
                    person.orcid = orcid
                    person.is_verified = True
                else:
                    person.warnings.append(f"Invalid ORCID checksum: {orcid}")
            else:
                person.orcid = orcid
                person.is_verified = True
        
        # Extract email if present
        if 'email' in groups_dict and groups_dict['email']:
            person.email = groups_dict['email']
            person.is_verified = True
        
        # Determine section
        if section_map:
            person.section = self._determine_section(start_char, section_map)
        
        # Extract context
        if self.extract_context:
            self._add_context(person, text)
        
        # Refine role based on context
        person.role = self._classify_person_role(person, text, config)
        
        # Calculate final confidence
        person.confidence = self._calculate_person_confidence(person, text)
        
        # Determine status
        person.status = self._determine_person_status(person)
        
        return person
    
    def _build_person_name(
        self,
        raw_text: str,
        groups_dict: Dict,
        capture_groups: Dict
    ) -> Optional[PersonName]:
        """
        Build structured person name from match groups.
        
        Args:
            raw_text: Raw matched text
            groups_dict: Named capture groups from regex
            capture_groups: Capture group mapping from config
            
        Returns:
            PersonName object or None
        """
        # Extract name components using capture groups
        full_name = raw_text.strip()
        surname = None
        given_names = None
        initials = None
        
        # Try to extract from named groups
        if 'surname' in groups_dict:
            surname = groups_dict['surname']
        elif 'last_name' in groups_dict:
            surname = groups_dict['last_name']
        
        if 'given_names' in groups_dict:
            given_names = groups_dict['given_names']
        elif 'first_name' in groups_dict:
            given_names = groups_dict['first_name']
        
        if 'initials' in groups_dict:
            initials = groups_dict['initials']
        elif 'middle_initial' in groups_dict:
            initials = groups_dict['middle_initial']
        
        # If no structured extraction, try to parse full name
        if not surname:
            components = extract_person_components(full_name)
            surname_match = re.search(r'[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+', full_name)
            if surname_match:
                surname = surname_match.group(0)
        
        # Extract titles, particles, degrees
        components = extract_person_components(full_name)
        
        name = PersonName(
            full_name=full_name,
            surname=surname,
            given_names=given_names,
            initials=initials,
            particles=components.get('particles', []),
            titles=components.get('titles', []),
            degrees=components.get('degrees', [])
        )
        
        return name
    
    def _generate_person_id(self, name: PersonName, position: int) -> str:
        """
        Generate unique person ID.
        
        Args:
            name: PersonName object
            position: Character position
            
        Returns:
            Person ID string
        """
        # Use surname if available, otherwise use full name
        name_part = name.surname if name.surname else name.full_name
        # Normalize for ID
        name_normalized = re.sub(r'[^\w\s-]', '', name_part.lower())
        name_normalized = re.sub(r'\s+', '_', name_normalized)
        
        return f"person_{name_normalized}_{position}"
    
    def _determine_section(
        self,
        position: int,
        section_map: Dict[str, Tuple[int, int]]
    ) -> Optional[str]:
        """
        Determine which section a person mention belongs to.
        
        Args:
            position: Character position
            section_map: Section boundaries
            
        Returns:
            Section name or None
        """
        for section_name, (start, end) in section_map.items():
            if start <= position < end:
                return section_name
        return None
    
    def _add_context(self, person: ExtractedPerson, text: str):
        """
        Add surrounding context to person.
        
        Args:
            person: ExtractedPerson to update
            text: Full document text
        """
        # Extract preceding context
        context_start = max(0, person.start_char - self.context_window)
        preceding = text[context_start:person.start_char]
        person.preceding_context = re.sub(r'\s+', ' ', preceding).strip()
        
        # Extract following context
        context_end = min(len(text), person.end_char + self.context_window)
        following = text[person.end_char:context_end]
        person.following_context = re.sub(r'\s+', ' ', following).strip()
        
        # Extract sentence
        sentence_start = text.rfind('.', context_start, person.start_char) + 1
        sentence_end = text.find('.', person.end_char, context_end)
        if sentence_end == -1:
            sentence_end = context_end
        
        sentence = text[sentence_start:sentence_end + 1]
        person.sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    def _classify_person_role(
        self,
        person: ExtractedPerson,
        text: str,
        config: Dict
    ) -> PersonRole:
        """
        Classify person role based on context.
        
        Args:
            person: ExtractedPerson object
            text: Full document text
            config: Pattern configuration
            
        Returns:
            PersonRole enum
        """
        # Start with pattern-defined role
        current_role = person.role
        
        # Use preceding context for classification
        context = person.preceding_context or ""
        
        # Score each role based on triggers
        role_scores = defaultdict(float)
        role_scores[current_role.value] = 2.0  # Boost pattern role
        
        for role_name, patterns in self.compiled_role_triggers.items():
            for pattern in patterns:
                if pattern.search(context):
                    role_scores[role_name] += 1.0
        
        # Check section-based clues
        if person.section:
            section_lower = person.section.lower()
            
            # Map sections to likely roles
            if any(ref in section_lower for ref in CONTEXT_SECTIONS.get('references', [])):
                role_scores['author'] += 1.5
            elif any(auth in section_lower for auth in CONTEXT_SECTIONS.get('author_info', [])):
                if 'corresponding' in section_lower:
                    role_scores['corresponding_author'] += 2.0
                else:
                    role_scores['author'] += 1.0
            elif any(meth in section_lower for meth in CONTEXT_SECTIONS.get('methods', [])):
                role_scores['principal_investigator'] += 1.0
        
        # Check for explicit markers
        if person.email:
            role_scores['corresponding_author'] += 1.5
        
        if person.orcid:
            role_scores['author'] += 1.0
        
        # Return highest scoring role
        if role_scores:
            best_role_str = max(role_scores.items(), key=lambda x: x[1])[0]
            try:
                return PersonRole(best_role_str)
            except ValueError:
                return current_role
        
        return current_role
    
    def _calculate_person_confidence(
        self,
        person: ExtractedPerson,
        text: str
    ) -> float:
        """
        Calculate confidence score for person.
        
        Args:
            person: ExtractedPerson object
            text: Full document text
            
        Returns:
            Confidence score [0.0-1.0]
        """
        # Start with base confidence from pattern
        confidence = person.confidence
        
        # Apply adjustments
        if person.orcid:
            confidence += CONFIDENCE_ADJUSTMENTS.get('has_orcid', 0.2)
        
        if person.email:
            confidence += CONFIDENCE_ADJUSTMENTS.get('has_email', 0.15)
        
        if person.affiliations:
            confidence += CONFIDENCE_ADJUSTMENTS.get('has_affiliation', 0.10)
        
        if person.name.degrees:
            confidence += CONFIDENCE_ADJUSTMENTS.get('has_degree', 0.08)
        
        if person.name.titles:
            confidence += CONFIDENCE_ADJUSTMENTS.get('has_title', 0.05)
        
        if person.name.particles:
            confidence += CONFIDENCE_ADJUSTMENTS.get('has_particles', 0.03)
        
        # Check for explicit role markers
        if person.role != PersonRole.UNKNOWN:
            confidence += CONFIDENCE_ADJUSTMENTS.get('explicit_role', 0.10)
        
        # Section-based adjustment
        if person.section:
            section_lower = person.section.lower()
            if any(auth in section_lower for auth in CONTEXT_SECTIONS.get('author_info', [])):
                confidence += CONFIDENCE_ADJUSTMENTS.get('in_author_section', 0.05)
        
        # Multiple mentions (check if name appears multiple times)
        if person.name.surname:
            occurrences = len(re.findall(
                re.escape(person.name.surname), 
                text, 
                re.IGNORECASE
            ))
            person.mention_count = occurrences
            if occurrences > 1:
                confidence += CONFIDENCE_ADJUSTMENTS.get('multiple_mentions', 0.05)
        
        return min(confidence, 1.0)
    
    def _determine_person_status(
        self,
        person: ExtractedPerson
    ) -> PersonStatus:
        """
        Determine extraction status of person.
        
        Args:
            person: ExtractedPerson object
            
        Returns:
            PersonStatus enum
        """
        if person.is_verified:
            return PersonStatus.VERIFIED
        elif person.name.surname and person.name.given_names:
            return PersonStatus.NORMALIZED
        elif person.confidence >= 0.8:
            return PersonStatus.NORMALIZED
        else:
            return PersonStatus.RAW
    
    def _extract_affiliations(self, text: str) -> List[Affiliation]:
        """
        Extract affiliations from text.
        
        Args:
            text: Document text
            
        Returns:
            List of Affiliation objects
        """
        affiliations = []
        affiliation_id = 0
        
        for affil_key, pattern_info in self.compiled_affiliation_patterns.items():
            regex = pattern_info['regex']
            config = pattern_info['config']
            
            for match in regex.finditer(text):
                raw_text = match.group(0)
                
                affiliation = Affiliation(
                    affiliation_id=f"affil_{affiliation_id}",
                    raw_text=raw_text,
                    normalized_name=raw_text.strip(),
                    affiliation_type=config.get('type'),
                    confidence=config.get('confidence', 0.8)
                )
                
                affiliations.append(affiliation)
                affiliation_id += 1
        
        return affiliations
    
    def _link_persons_to_affiliations(
        self,
        persons: List[ExtractedPerson],
        affiliations: List[Affiliation],
        text: str
    ):
        """
        Link persons to nearby affiliations.
        
        Args:
            persons: List of persons
            affiliations: List of affiliations
            text: Document text
        """
        for person in persons:
            # Find affiliations within 500 characters
            nearby_affiliations = []
            
            for affil in affiliations:
                # Get affiliation position (approximate from text search)
                affil_pos = text.find(affil.raw_text)
                if affil_pos == -1:
                    continue
                
                distance = abs(person.start_char - affil_pos)
                if distance < 500:  # Within 500 characters
                    nearby_affiliations.append((distance, affil))
            
            # Add closest affiliations
            nearby_affiliations.sort(key=lambda x: x[0])
            person.affiliations = [affil for _, affil in nearby_affiliations[:3]]
    
    def _deduplicate_persons(
        self,
        persons: List[ExtractedPerson],
        text: str
    ) -> List[ExtractedPerson]:
        """
        Remove duplicate person mentions, keeping highest confidence.
        
        Args:
            persons: List of persons
            text: Document text for context
            
        Returns:
            Deduplicated list
        """
        # Group by normalized name
        groups: Dict[str, List[ExtractedPerson]] = defaultdict(list)
        
        for person in persons:
            # Create normalized key
            if person.name.surname:
                key = person.name.surname.lower()
                if person.name.initials:
                    key += f"_{person.name.initials.lower()}"
            else:
                key = person.name.full_name.lower()[:50]
            
            # Normalize key
            key = re.sub(r'[^\w\s]', '', key)
            key = re.sub(r'\s+', '_', key)
            
            groups[key].append(person)
        
        # Keep best from each group
        unique_persons = []
        for group in groups.values():
            if len(group) == 1:
                unique_persons.append(group[0])
            else:
                # Merge information from duplicates
                best = max(group, key=lambda p: (
                    p.is_verified,
                    p.confidence,
                    len(p.affiliations),
                    -p.start_char
                ))
                
                # Aggregate mention count
                best.mention_count = len(group)
                
                # Collect all affiliations
                all_affiliations = []
                for person in group:
                    all_affiliations.extend(person.affiliations)
                
                # Deduplicate affiliations
                seen_affil = set()
                unique_affil = []
                for affil in all_affiliations:
                    if affil.normalized_name not in seen_affil:
                        seen_affil.add(affil.normalized_name)
                        unique_affil.append(affil)
                
                best.affiliations = unique_affil
                
                unique_persons.append(best)
        
        return unique_persons
    
    def _calculate_statistics(self, result: PersonExtractionResult):
        """
        Calculate statistics for extraction result.
        
        Args:
            result: PersonExtractionResult to update
        """
        result.total_persons = len(result.persons)
        
        # Count unique by surname
        unique_surnames = set()
        for person in result.persons:
            if person.name.surname:
                unique_surnames.add(person.name.surname.lower())
        result.unique_persons = len(unique_surnames)
        
        result.verified_persons = sum(1 for p in result.persons if p.is_verified)
        
        if result.persons:
            result.average_confidence = sum(
                p.confidence for p in result.persons
            ) / len(result.persons)
        
        # Count by role
        role_counter = Counter(p.role.value for p in result.persons)
        result.role_counts = dict(role_counter)
        
        # Author-specific metrics
        result.author_count = sum(
            1 for p in result.persons 
            if p.role == PersonRole.AUTHOR
        )
        
        result.has_corresponding_author = any(
            p.role == PersonRole.CORRESPONDING_AUTHOR 
            for p in result.persons
        )
        
        result.has_principal_investigator = any(
            p.role == PersonRole.PRINCIPAL_INVESTIGATOR 
            for p in result.persons
        )
    
    def _build_coauthor_network(self, result: PersonExtractionResult):
        """
        Build co-author network.
        
        Args:
            result: PersonExtractionResult to update
        """
        # Get all authors
        authors = [p for p in result.persons if p.role == PersonRole.AUTHOR]
        
        # Build network (simplified - all authors are connected)
        for author in authors:
            author_name = author.name.surname or author.name.full_name
            coauthors = [
                p.name.surname or p.name.full_name 
                for p in authors 
                if p.person_id != author.person_id
            ]
            result.co_author_network[author_name] = coauthors[:10]  # Limit to 10
    
    def _validate_persons(self, result: PersonExtractionResult):
        """
        Validate and clean persons.
        
        Args:
            result: PersonExtractionResult to validate
        """
        valid_persons = []
        
        for person in result.persons:
            # Validate confidence range
            if not 0.0 <= person.confidence <= 1.0:
                person.warnings.append(f"Invalid confidence: {person.confidence}")
                person.confidence = max(0.0, min(1.0, person.confidence))
            
            # Validate position
            if person.start_char < 0 or person.end_char <= person.start_char:
                person.warnings.append(f"Invalid position: {person.start_char}-{person.end_char}")
                continue
            
            # Validate name
            if not person.name.full_name or len(person.name.full_name.strip()) < 2:
                person.warnings.append("Name too short or empty")
                continue
            
            # Validate ORCID format if present
            if person.orcid:
                if not re.match(r'^0000-\d{4}-\d{4}-\d{3}[0-9X]$', person.orcid):
                    person.warnings.append(f"Invalid ORCID format: {person.orcid}")
                    person.orcid = None
                    person.is_verified = False
            
            # Validate email format if present
            if person.email:
                if not re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$', person.email):
                    person.warnings.append(f"Invalid email format: {person.email}")
                    person.email = None
            
            valid_persons.append(person)
        
        result.persons = valid_persons


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_persons_from_text(
    text: str,
    doc_id: str = "unknown",
    config: Optional[Dict] = None
) -> PersonExtractionResult:
    """
    Convenience function to extract persons from text.
    
    Args:
        text: Document text
        doc_id: Document identifier
        config: Optional extractor configuration
        
    Returns:
        PersonExtractionResult
    """
    extractor = PersonExtractor(config)
    return extractor.extract_persons(text, doc_id)


def persons_to_json(result: PersonExtractionResult, filepath: str):
    """
    Save person extraction result to JSON file.
    
    Args:
        result: PersonExtractionResult
        filepath: Output file path
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(result.to_json())
    
    logger.info(f"Persons saved to {filepath}")


def get_persons_by_role(
    result: PersonExtractionResult,
    role: PersonRole
) -> List[ExtractedPerson]:
    """
    Filter persons by role.
    
    Args:
        result: PersonExtractionResult
        role: Role to filter by
        
    Returns:
        List of matching persons
    """
    return [p for p in result.persons if p.role == role]


def get_verified_persons(
    result: PersonExtractionResult
) -> List[ExtractedPerson]:
    """
    Get all verified persons (with ORCID or email).
    
    Args:
        result: PersonExtractionResult
        
    Returns:
        List of verified persons
    """
    return [p for p in result.persons if p.is_verified]


def export_author_list(
    result: PersonExtractionResult,
    format: str = "text"
) -> str:
    """
    Export author list in various formats.
    
    Args:
        result: PersonExtractionResult
        format: Output format ('text', 'apa', 'vancouver')
        
    Returns:
        Formatted author list
    """
    authors = get_persons_by_role(result, PersonRole.AUTHOR)
    
    if format == "text":
        return "\n".join(p.name.full_name for p in authors)
    
    elif format == "vancouver":
        formatted = []
        for p in authors:
            if p.name.surname and p.name.initials:
                formatted.append(f"{p.name.surname} {p.name.initials}")
            else:
                formatted.append(p.name.full_name)
        return ", ".join(formatted)
    
    elif format == "apa":
        formatted = []
        for p in authors:
            if p.name.surname and p.name.initials:
                formatted.append(f"{p.name.surname}, {p.name.initials}")
            else:
                formatted.append(p.name.full_name)
        return ", ".join(formatted[:-1]) + (" & " if len(formatted) > 1 else "") + (formatted[-1] if formatted else "")
    
    return "\n".join(p.name.full_name for p in authors)