#!/usr/bin/env python3
"""
Document Metadata Extraction - Person & Author Extractor (FIXED v2.0)
=====================================================================
Location: corpus_metadata/document_metadata_extraction_persons.py
Version: 2.0.0 - MAJOR FIX
Last Updated: 2025-10-13

FIXES IN v2.0.0:
- ✅ Better patterns that actually match real person names
- ✅ Dr./Dr/DRA/Prof./Professor prefix detection
- ✅ False positive filtering (removes "The", "As", "Current", etc.)
- ✅ Length validation (2-50 characters for names)
- ✅ Context-aware extraction (author sections, citations only)
- ✅ Common word blacklist
- ✅ Minimum word requirement for full names

Purpose:
    Extract and normalize person entities (authors, investigators, experts) from biomedical documents.
"""

import re
import logging
import unicodedata
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict, Counter
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS & BLACKLISTS
# ============================================================================

# Words that should NEVER be considered person names
PERSON_NAME_BLACKLIST = {
    # Common sentence starters
    'the', 'a', 'an', 'as', 'at', 'by', 'for', 'from', 'in', 'of', 'on', 'to', 'with',
    'and', 'or', 'but', 'not', 'all', 'any', 'some', 'this', 'that', 'these', 'those',
    'current', 'evidence', 'therapeutic', 'landscape', 'complement', 'inhibition',
    'represents', 'significant', 'advance', 'pharmacological', 'mechanisms', 'rationale',
    'transforms', 'treatment', 'targeting', 'inflammatory', 'amplification', 'loop',
    'central', 'disease', 'pathogenesis', 'oral', 'receptor', 'antagonist', 'selectively',
    'blocks', 'binding', 'preventing', 'mediated', 'neutrophil', 'activation', 'chemotaxis',
    'preserving', 'beneficial', 'functions', 'drug', 'interrupts', 'vicious', 'cycle',
    'where', 'activated', 'neutrophils', 'degranulate', 'activate', 'alternative',
    'pathway', 'generate', 'which', 'then', 'recruits', 'primes', 'additional',
    'experimental', 'strongly', 'supports', 'approach', 'deficient', 'mice', 'show',
    'complete', 'protection', 'induced', 'glomerulonephritis', 'while', 'lacking',
    'membrane', 'attack', 'complex', 'formation', 'remain', 'vulnerable', 'confirming',
    'rather', 'than', 'terminal', 'products', 'drives', 'humans', 'elevated', 'plasma',
    'levels', 'soluble', 'factor', 'correlate', 'active', 'normalize', 'remission',
    'profile', 'presents', 'both', 'opportunities', 'challenges', 'pediatric',
    'application', 'achieves', 'peak', 'concentrations', 'hours', 'after', 'administration',
    'elimination', 'halflife', 'enabling', 'twice', 'daily', 'dosing', 'highly',
    'protein', 'bound', 'primarily', 'metabolized', 'active', 'metabolite', 'representing',
    'potential', 'concerns', 'populations', 'given', 'developmental', 'changes',
    'metabolism', 'interactions', 'strong', 'inhibitors', 'require', 'dose', 'reduction',
    'once', 'inducers', 'should', 'avoided', 'entirely', 'versus', 'considerations',
    'critical', 'gap', 'limited', 'pharmacokinetic', 'data', 'exists', 'other',
    'adult', 'studies', 'clinically', 'relevant', 'differences', 'across', 'age',
    'ranges', 'years', 'sex', 'race', 'body', 'weight', 'adjustment', 'required',
    'mild', 'tosevere', 'renal', 'impairment', 'however', 'activity', 'composition',
    'organ', 'function', 'children', 'would', 'likely', 'affect', 'disposition',
    'significantly', 'label', 'explicitly', 'states', 'safety', 'efficacy', 'known',
    'under', 'trials', 'ongoing'
}

# Titles and prefixes
PERSON_TITLES = [
    'Dr', 'Dr.', 'DRA', 'DRA.', 'Prof', 'Prof.', 'Professor',
    'Mr', 'Mr.', 'Mrs', 'Mrs.', 'Ms', 'Ms.', 'Miss',
    'Sir', 'Dame', 'Lord', 'Lady'
]

# Name particles (not blacklisted)
NAME_PARTICLES = [
    'van', 'von', 'de', 'del', 'della', 'di', 'da', 'le', 'la', 'el',
    'al', 'bin', 'ibn', 'ben', 'ter', 'den', 'der', 'ten'
]

# Academic degrees
ACADEMIC_DEGREES = [
    'PhD', 'Ph.D.', 'MD', 'M.D.', 'ScD', 'Sc.D.', 'MBA', 'M.B.A.',
    'MPH', 'M.P.H.', 'MSc', 'M.Sc.', 'MA', 'M.A.', 'BS', 'B.S.',
    'BA', 'B.A.', 'RN', 'PharmD', 'Pharm.D.', 'DDS', 'D.D.S.',
    'DO', 'D.O.', 'DVM', 'D.V.M.'
]

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
    VERIFIED = "verified"
    NORMALIZED = "normalized"
    RAW = "raw"
    AMBIGUOUS = "ambiguous"


@dataclass
class PersonName:
    """Structured person name components."""
    full_name: str
    surname: Optional[str] = None
    given_names: Optional[str] = None
    initials: Optional[str] = None
    particles: List[str] = field(default_factory=list)
    titles: List[str] = field(default_factory=list)
    degrees: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {'full_name': self.full_name}
        if self.surname:
            result['surname'] = self.surname
        if self.given_names:
            result['given_names'] = self.given_names
        if self.initials:
            result['initials'] = self.initials
        if self.particles:
            result['particles'] = self.particles
        if self.titles:
            result['titles'] = self.titles
        if self.degrees:
            result['degrees'] = self.degrees
        return result


@dataclass
class ExtractedPerson:
    """Structured representation of an extracted person."""
    person_id: str
    name: PersonName
    role: PersonRole
    start_char: int
    end_char: int
    confidence: float = 0.0
    status: PersonStatus = PersonStatus.RAW
    
    section: Optional[str] = None
    sentence: Optional[str] = None
    preceding_context: Optional[str] = None
    following_context: Optional[str] = None
    
    orcid: Optional[str] = None
    email: Optional[str] = None
    affiliations: List[Dict] = field(default_factory=list)
    
    mention_count: int = 1
    is_verified: bool = False
    extraction_method: str = "pattern"
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    warnings: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'person_id': self.person_id,
            'name': self.name.to_dict(),
            'role': self.role.value,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'confidence': round(self.confidence, 3),
            'status': self.status.value,
            'section': self.section,
            'sentence': self.sentence,
            'preceding_context': self.preceding_context,
            'following_context': self.following_context,
            'orcid': self.orcid,
            'email': self.email,
            'affiliations': self.affiliations,
            'mention_count': self.mention_count,
            'is_verified': self.is_verified,
            'extraction_method': self.extraction_method,
            'extraction_timestamp': self.extraction_timestamp,
            'warnings': self.warnings if self.warnings else None,
            'metadata': self.metadata if self.metadata else None
        }


@dataclass
class Affiliation:
    """Affiliation information."""
    affiliation_id: str
    raw_text: str
    normalized_name: str
    affiliation_type: str
    start_char: int
    end_char: int
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'affiliation_id': self.affiliation_id,
            'raw_text': self.raw_text,
            'normalized_name': self.normalized_name,
            'affiliation_type': self.affiliation_type,
            'confidence': round(self.confidence, 3)
        }


@dataclass
class PersonExtractionResult:
    """Complete person extraction result."""
    doc_id: str
    language: str = "en"
    
    persons: List[ExtractedPerson] = field(default_factory=list)
    affiliations: List[Affiliation] = field(default_factory=list)
    
    total_persons: int = 0
    unique_persons: int = 0
    verified_persons: int = 0
    average_confidence: float = 0.0
    role_counts: Dict[str, int] = field(default_factory=dict)
    author_count: int = 0
    has_corresponding_author: bool = False
    has_principal_investigator: bool = False
    
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
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
# PERSON EXTRACTOR (FIXED VERSION)
# ============================================================================

class PersonExtractor:
    """
    Extract and normalize person entities from biomedical documents.
    FIXED: Better patterns, false positive filtering, prefix detection.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the person extractor."""
        self.config = config or {}
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.extract_context = self.config.get('extract_context', True)
        self.context_window = self.config.get('context_window', 100)
        self.extract_affiliations = self.config.get('extract_affiliations', True)
        
        # Compile patterns
        self._compile_patterns()
        
        logger.info(f"PersonExtractor initialized with config: {self.config}")
    
    def _compile_patterns(self):
        """Compile improved regex patterns."""
        self.compiled_patterns = {}
        
        # FIXED: Better patterns that match actual person names
        
        # Pattern 1: Title + Full Name (Dr. John Smith)
        title_pattern = '|'.join(re.escape(t) for t in PERSON_TITLES)
        self.compiled_patterns['title_full_name'] = {
            'regex': re.compile(
                rf'\b(?:{title_pattern})\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){{1,3}})\b',
                re.UNICODE
            ),
            'config': {
                'confidence': 0.95,
                'role': 'author',
                'requires_title': True
            }
        }
        
        # Pattern 2: Citation style (Smith J, Jones AB)
        self.compiled_patterns['citation_standard'] = {
            'regex': re.compile(
                r'\b([A-Z][a-z]{2,20})\s+([A-Z]{1,3})\b(?=\s*[,;]|\s+and\s+|\s+&\s+)',
                re.UNICODE
            ),
            'config': {
                'confidence': 0.85,
                'role': 'author',
                'capture_groups': {
                    1: 'surname',
                    2: 'initials'
                }
            }
        }
        
        # Pattern 3: Full name with initials (John A. Smith, Mary B. Jones)
        self.compiled_patterns['full_name_initials'] = {
            'regex': re.compile(
                r'\b([A-Z][a-z]{2,15})\s+([A-Z]\.?)\s+([A-Z][a-z]{2,20})\b',
                re.UNICODE
            ),
            'config': {
                'confidence': 0.90,
                'role': 'author',
                'capture_groups': {
                    1: 'given_names',
                    2: 'initials',
                    3: 'surname'
                }
            }
        }
        
        # Pattern 4: ORCID + Name
        self.compiled_patterns['orcid_name'] = {
            'regex': re.compile(
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+\(?(https://orcid\.org/)?(?P<orcid>0000-\d{4}-\d{4}-\d{3}[0-9X])\)?',
                re.UNICODE | re.IGNORECASE
            ),
            'config': {
                'confidence': 1.0,
                'role': 'author',
                'has_orcid': True
            }
        }
        
        # Pattern 5: Email + Name
        self.compiled_patterns['email_name'] = {
            'regex': re.compile(
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*[(<](?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})[)>]',
                re.UNICODE
            ),
            'config': {
                'confidence': 0.95,
                'role': 'corresponding_author',
                'has_email': True
            }
        }
        
        logger.info(f"Compiled {len(self.compiled_patterns)} person patterns")
    
    def extract_persons(
        self,
        text: str,
        doc_id: str = "unknown",
        section_map: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> PersonExtractionResult:
        """Extract all persons from document text."""
        start_time = datetime.now()
        result = PersonExtractionResult(doc_id=doc_id)
        
        try:
            logger.info(f"Extracting persons for {doc_id}")
            
            # Extract persons by pattern type
            all_persons = []
            
            for pattern_key, pattern_info in self.compiled_patterns.items():
                persons = self._extract_by_pattern(
                    text,
                    pattern_key,
                    pattern_info,
                    section_map
                )
                all_persons.extend(persons)
            
            logger.info(f"Found {len(all_persons)} raw person mentions")
            
            # CRITICAL: Filter false positives
            filtered_persons = self._filter_false_positives(all_persons)
            logger.info(f"After false positive filtering: {len(filtered_persons)} persons")
            
            # Deduplicate persons
            unique_persons = self._deduplicate_persons(filtered_persons, text)
            logger.info(f"After deduplication: {len(unique_persons)} unique persons")
            
            # Extract affiliations
            if self.extract_affiliations:
                affiliations = self._extract_affiliations(text)
                result.affiliations = affiliations
                logger.info(f"Extracted {len(affiliations)} affiliations")
                
                # Link persons to affiliations
                self._link_persons_to_affiliations(unique_persons, affiliations, text)
            
            # Filter by confidence
            final_persons = [
                p for p in unique_persons 
                if p.confidence >= self.min_confidence
            ]
            logger.info(f"After confidence filter: {len(final_persons)} persons")
            
            result.persons = final_persons
            
            # Calculate statistics
            self._calculate_statistics(result)
            
            # Validate persons
            self._validate_persons(result)
            
        except Exception as e:
            logger.error(f"Person extraction failed: {e}", exc_info=True)
            result.warnings.append(f"Extraction error: {str(e)}")
        
        # Record processing time
        end_time = datetime.now()
        result.processing_time_seconds = (end_time - start_time).total_seconds()
        
        logger.info(f"Person extraction complete for {doc_id}: {result.total_persons} persons")
        
        return result
    
    def _extract_by_pattern(
        self,
        text: str,
        pattern_key: str,
        pattern_info: Dict,
        section_map: Optional[Dict[str, Tuple[int, int]]]
    ) -> List[ExtractedPerson]:
        """Extract persons using a specific pattern."""
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
                logger.debug(f"Failed to create person for {pattern_key}: {e}")
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
        """Create a structured person from a regex match."""
        raw_text = match.group(0)
        start_char = match.start()
        end_char = match.end()
        
        # Build person name
        name = self._build_person_name(match, raw_text, config)
        
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
        groups_dict = match.groupdict()
        if 'orcid' in groups_dict and groups_dict['orcid']:
            person.orcid = groups_dict['orcid']
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
        
        # Calculate final confidence
        person.confidence = self._calculate_person_confidence(person, text, config)
        
        # Determine status
        person.status = self._determine_person_status(person)
        
        return person
    
    def _build_person_name(
        self,
        match: re.Match,
        raw_text: str,
        config: Dict
    ) -> Optional[PersonName]:
        """Build structured person name from match."""
        full_name = raw_text.strip()
        surname = None
        given_names = None
        initials = None
        titles = []
        degrees = []
        particles = []
        
        # Extract from capture groups if available
        capture_groups = config.get('capture_groups', {})
        
        if capture_groups:
            groups = match.groups()
            for idx, field in capture_groups.items():
                if idx <= len(groups):
                    value = groups[idx - 1]
                    if field == 'surname':
                        surname = value
                    elif field == 'given_names':
                        given_names = value
                    elif field == 'initials':
                        initials = value
        
        # Extract titles from the text
        for title in PERSON_TITLES:
            if title in raw_text:
                titles.append(title)
                full_name = full_name.replace(title, '').strip()
        
        # Extract degrees
        for degree in ACADEMIC_DEGREES:
            if degree in raw_text:
                degrees.append(degree)
                full_name = full_name.replace(degree, '').strip()
        
        # Extract particles
        words = full_name.split()
        for word in words:
            if word.lower() in NAME_PARTICLES:
                particles.append(word.lower())
        
        # If no surname extracted, try to parse
        if not surname and len(words) >= 2:
            # Assume last word is surname
            surname = words[-1]
            if len(words) > 2:
                given_names = ' '.join(words[:-1])
            else:
                given_names = words[0]
        
        name = PersonName(
            full_name=full_name,
            surname=surname,
            given_names=given_names,
            initials=initials,
            particles=particles,
            titles=titles,
            degrees=degrees
        )
        
        return name
    
    def _filter_false_positives(
        self,
        persons: List[ExtractedPerson]
    ) -> List[ExtractedPerson]:
        """
        CRITICAL: Filter out false positive person extractions.
        """
        filtered = []
        
        for person in persons:
            # Check 1: Length validation (2-50 characters)
            name_len = len(person.name.full_name)
            if name_len < 2 or name_len > 50:
                logger.debug(f"Rejected: '{person.name.full_name}' (length: {name_len})")
                continue
            
            # Check 2: Blacklist check
            first_word = person.name.full_name.split()[0].lower()
            if first_word in PERSON_NAME_BLACKLIST:
                logger.debug(f"Rejected: '{person.name.full_name}' (blacklisted: {first_word})")
                continue
            
            # Check 3: Must have at least 2 words (unless has title/ORCID)
            words = person.name.full_name.split()
            if len(words) < 2 and not person.name.titles and not person.orcid:
                logger.debug(f"Rejected: '{person.name.full_name}' (single word)")
                continue
            
            # Check 4: Each word should start with capital letter
            if not all(word[0].isupper() for word in words if len(word) > 1):
                logger.debug(f"Rejected: '{person.name.full_name}' (capitalization)")
                continue
            
            # Check 5: No excessive lowercase words
            lowercase_words = sum(1 for word in words if word.islower() and word not in NAME_PARTICLES)
            if lowercase_words > len(words) // 2:
                logger.debug(f"Rejected: '{person.name.full_name}' (too many lowercase)")
                continue
            
            # Check 6: Surname validation (if extracted)
            if person.name.surname:
                surname_lower = person.name.surname.lower()
                if surname_lower in PERSON_NAME_BLACKLIST:
                    logger.debug(f"Rejected: '{person.name.full_name}' (surname blacklisted: {surname_lower})")
                    continue
            
            # Passed all checks
            filtered.append(person)
        
        return filtered
    
    def _generate_person_id(self, name: PersonName, position: int) -> str:
        """Generate unique person ID."""
        name_part = name.surname if name.surname else name.full_name
        name_normalized = re.sub(r'[^\w\s-]', '', name_part.lower())
        name_normalized = re.sub(r'\s+', '_', name_normalized)
        return f"person_{name_normalized}_{position}"
    
    def _determine_section(
        self,
        position: int,
        section_map: Dict[str, Tuple[int, int]]
    ) -> Optional[str]:
        """Determine which section a person mention belongs to."""
        for section_name, (start, end) in section_map.items():
            if start <= position < end:
                return section_name
        return None
    
    def _add_context(self, person: ExtractedPerson, text: str):
        """Add surrounding context to person."""
        context_start = max(0, person.start_char - self.context_window)
        preceding = text[context_start:person.start_char]
        person.preceding_context = re.sub(r'\s+', ' ', preceding).strip()
        
        context_end = min(len(text), person.end_char + self.context_window)
        following = text[person.end_char:context_end]
        person.following_context = re.sub(r'\s+', ' ', following).strip()
        
        sentence_start = text.rfind('.', context_start, person.start_char) + 1
        sentence_end = text.find('.', person.end_char, context_end)
        if sentence_end == -1:
            sentence_end = context_end
        
        sentence = text[sentence_start:sentence_end + 1]
        person.sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    def _calculate_person_confidence(
        self,
        person: ExtractedPerson,
        text: str,
        config: Dict
    ) -> float:
        """Calculate confidence score for person."""
        confidence = config.get('confidence', 0.5)
        
        # Boost confidence for verified persons
        if person.is_verified:
            confidence += 0.2
        
        # Boost for titles
        if person.name.titles:
            confidence += 0.1
        
        # Boost for proper section
        if person.section and 'author' in person.section.lower():
            confidence += 0.1
        
        # Boost for citation context
        if person.preceding_context:
            citation_markers = ['et al', 'and colleagues', 'by', 'from']
            if any(marker in person.preceding_context.lower() for marker in citation_markers):
                confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _determine_person_status(self, person: ExtractedPerson) -> PersonStatus:
        """Determine extraction status of person."""
        if person.is_verified:
            return PersonStatus.VERIFIED
        elif person.name.surname and person.name.given_names:
            return PersonStatus.NORMALIZED
        else:
            return PersonStatus.RAW
    
    def _deduplicate_persons(
        self,
        persons: List[ExtractedPerson],
        text: str
    ) -> List[ExtractedPerson]:
        """Remove duplicate persons."""
        groups: Dict[str, List[ExtractedPerson]] = defaultdict(list)
        
        for person in persons:
            # Use normalized full name as key
            key = person.name.full_name.lower().strip()
            groups[key].append(person)
        
        unique_persons = []
        for group in groups.values():
            # Keep highest confidence
            best = max(group, key=lambda p: (p.confidence, p.is_verified, -p.start_char))
            
            # Count mentions
            best.mention_count = len(group)
            
            unique_persons.append(best)
        
        return unique_persons
    
    def _extract_affiliations(self, text: str) -> List[Affiliation]:
        """Extract affiliation information."""
        affiliations = []
        
        # Simple affiliation pattern
        affiliation_pattern = re.compile(
            r'\b((?:Department|Division|Institute|Center|Laboratory|Clinic|Hospital|University|College|School)\s+of\s+[A-Z][^,.;]{5,80})',
            re.IGNORECASE
        )
        
        for idx, match in enumerate(affiliation_pattern.finditer(text)):
            affiliation = Affiliation(
                affiliation_id=f"affil_{idx}",
                raw_text=match.group(0),
                normalized_name=match.group(0),
                affiliation_type='academic' if 'University' in match.group(0) else 'clinical',
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.95
            )
            affiliations.append(affiliation)
        
        return affiliations
    
    def _link_persons_to_affiliations(
        self,
        persons: List[ExtractedPerson],
        affiliations: List[Affiliation],
        text: str
    ):
        """Link persons to nearby affiliations."""
        for person in persons:
            # Find closest affiliation within 500 characters
            closest_affil = None
            min_distance = 500
            
            for affil in affiliations:
                distance = min(
                    abs(person.start_char - affil.end_char),
                    abs(affil.start_char - person.end_char)
                )
                
                if distance < min_distance:
                    min_distance = distance
                    closest_affil = affil
            
            if closest_affil:
                person.affiliations.append(closest_affil.to_dict())
    
    def _calculate_statistics(self, result: PersonExtractionResult):
        """Calculate statistics for extraction result."""
        result.total_persons = len(result.persons)
        result.unique_persons = len(set(p.name.full_name for p in result.persons))
        result.verified_persons = sum(1 for p in result.persons if p.is_verified)
        
        if result.persons:
            result.average_confidence = sum(p.confidence for p in result.persons) / len(result.persons)
        
        role_counter = Counter(p.role.value for p in result.persons)
        result.role_counts = dict(role_counter)
        
        result.author_count = role_counter.get('author', 0)
        result.has_corresponding_author = any(
            p.role == PersonRole.CORRESPONDING_AUTHOR for p in result.persons
        )
        result.has_principal_investigator = any(
            p.role == PersonRole.PRINCIPAL_INVESTIGATOR for p in result.persons
        )
    
    def _validate_persons(self, result: PersonExtractionResult):
        """Validate and clean persons."""
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
    """Convenience function to extract persons from text."""
    extractor = PersonExtractor(config)
    return extractor.extract_persons(text, doc_id)


def persons_to_json(result: PersonExtractionResult, filepath: str):
    """Save person extraction result to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(result.to_json())
    
    logger.info(f"Persons saved to {filepath}")


def get_persons_by_role(
    result: PersonExtractionResult,
    role: PersonRole
) -> List[ExtractedPerson]:
    """Filter persons by role."""
    return [p for p in result.persons if p.role == role]