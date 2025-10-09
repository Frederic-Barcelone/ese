#!/usr/bin/env python3
"""
Lexicon Integration Module for Abbreviation Extraction
========================================================
Location: corpus_metadata/document_utils/abbreviation_lexicon.py
Version: 1.1.0

Updates in 1.1.0:
- Added system_initializer parameter to access loaded resources
- Initialize abbreviation_dict from system resources
- Improved integration with system-wide abbreviation data

Integrates UMLS/custom lexicon for abbreviation resolution.
Implements decision rules for lexicon vs Claude routing.

Decision Rules:
- Lexicon match + domain match + good acronym match → Accept without Claude
- Lexicon with 2-3 close options → Use margin + keywords; if ambiguous → Claude
- Not in lexicon → Accept if standard pattern (IL-6, TNF-α), else → Claude
- Local definition present → Never send to Claude
- Highly polysemous (RA, MS, CA, ASA) → Require extra signals or → Claude
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import json
from pathlib import Path

# ============================================================================
# CENTRALIZED LOGGING CONFIGURATION
# ============================================================================
try:
    from corpus_metadata.document_utils.metadata_logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    try:
        from .metadata_logging_config import get_logger
        logger = get_logger(__name__)
    except ImportError:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger(__name__)

# ============================================================================
# IMPORT SHARED TYPES
# ============================================================================
try:
    from .abbreviation_types import AbbreviationCandidate, ValidationStatus, SourceType
except ImportError:
    logger.error("Could not import shared types from abbreviation_types")
    raise


# ============================================================================
# LEXICON ENTRY STRUCTURE
# ============================================================================

@dataclass
class LexiconEntry:
    """Represents a lexicon entry for an abbreviation"""
    cui: str  # Concept Unique Identifier (UMLS)
    preferred_term: str
    abbreviation: str
    semantic_types: List[str] = field(default_factory=list)  # STY codes
    source_vocabulary: str = ''  # SAB (Source Abbreviation)
    language: str = 'ENG'
    synonyms: List[str] = field(default_factory=list)
    definition: Optional[str] = None
    domain_keywords: List[str] = field(default_factory=list)
    frequency: float = 0.0  # Usage frequency in corpus
    confidence_boost: float = 0.0  # Extra confidence for this entry
    
    def matches_domain(self, domain: str) -> bool:
        """Check if entry matches a domain"""
        domain_lower = domain.lower()
        
        # Check semantic types
        if domain == 'disease' and any(sty in self.semantic_types for sty in ['T019', 'T020', 'T037', 'T046', 'T047', 'T048', 'T049', 'T050', 'T190', 'T191']):
            return True
        if domain == 'drug' and any(sty in self.semantic_types for sty in ['T109', 'T110', 'T111', 'T112', 'T113', 'T114', 'T115', 'T116', 'T117', 'T118', 'T119', 'T120', 'T121', 'T122', 'T123', 'T124', 'T125', 'T126', 'T127', 'T129', 'T130', 'T131', 'T195', 'T196', 'T197', 'T200']):
            return True
        if domain == 'biological' and any(sty in self.semantic_types for sty in ['T087', 'T088', 'T116', 'T123', 'T124', 'T125', 'T126', 'T129', 'T130', 'T131']):
            return True
        
        # Check keywords
        if self.domain_keywords:
            return any(kw.lower() in domain_lower or domain_lower in kw.lower() 
                      for kw in self.domain_keywords)
        
        return False


# ============================================================================
# LEXICON RESOLVER CLASS
# ============================================================================

class LexiconResolver:
    """Handles lexicon-based abbreviation resolution"""
    
    # Highly polysemous abbreviations requiring extra validation
    POLYSEMOUS_ABBREVS = {
        'RA', 'MS', 'CA', 'ASA', 'DM', 'MI', 'PE', 'PS', 'PT', 
        'LA', 'ER', 'PR', 'AR', 'HD', 'MD', 'PD', 'AD', 'CD'
    }
    
    # Standard patterns that can be accepted without lexicon
    STANDARD_PATTERNS = {
        'interleukin': re.compile(r'^IL-?\d+$', re.IGNORECASE),
        'tnf': re.compile(r'^TNF-?[αβγ]?$', re.IGNORECASE),
        'interferon': re.compile(r'^IFN-?[αβγ]?$', re.IGNORECASE),
        'hla': re.compile(r'^HLA-[A-Z0-9]+(?:\*[0-9:]+)?$', re.IGNORECASE),
        'complement': re.compile(r'^C\d+[a-z]?$'),
        'cluster': re.compile(r'^CD\d+[a-z]?$', re.IGNORECASE),
        'rna_types': re.compile(r'^[mtsir]i?RNA$'),
    }
    
    def __init__(self, lexicon_path: Optional[Path] = None, config: Dict = None, system_initializer=None):
        """
        Initialize lexicon resolver.
        
        Args:
            lexicon_path: Path to lexicon file (JSON or TSV)
            config: Configuration dictionary
            system_initializer: System initializer with loaded resources
        """
        self.config = config or {}
        self.system_initializer = system_initializer
        self.lexicon: Dict[str, List[LexiconEntry]] = {}
        self.overlay: Dict[str, LexiconEntry] = {}  # Project-specific overlay
        
        # Initialize abbreviation_dict from system_initializer
        self.abbreviation_dict = {}
        self._load_abbreviations_from_system()
        
        # Load configuration
        lex_config = self.config.get('lexicon', {})
        self.tau_accept = lex_config.get('tau_accept', 0.85)
        self.delta_margin = lex_config.get('delta_margin', 0.15)
        self.prefer_language = lex_config.get('prefer_language', 'ENG')
        self.source_priority = lex_config.get('source_priority', 
            ['MSH', 'SNOMEDCT_US', 'NCI', 'RXNORM', 'ICD10CM'])
        
        # Load lexicon if path provided - FIX: Add proper type checking
        if lexicon_path:
            # Convert to Path if it's a string
            if isinstance(lexicon_path, str):
                lexicon_path = Path(lexicon_path)
            # Only call .exists() if it's a Path object
            if isinstance(lexicon_path, Path) and lexicon_path.exists():
                self.load_lexicon(lexicon_path)
        
        logger.info(f"LexiconResolver initialized with {len(self.abbreviation_dict)} abbreviations from system")
    
    def _load_abbreviations_from_system(self):
        """Load abbreviations from system_initializer resources"""
        if not self.system_initializer or not hasattr(self.system_initializer, 'resources'):
            logger.debug("No system_initializer or resources available")
            return
        
        resources = self.system_initializer.resources
        loaded_count = 0
        
        # Load abbreviation_general (dict format with detailed structure)
        if 'abbreviation_general' in resources:
            general = resources['abbreviation_general']
            if isinstance(general, dict):
                for abbrev, data in general.items():
                    abbrev_upper = abbrev.upper()
                    
                    # Extract the best expansion
                    expansion = data.get('canonical_expansion', '')
                    if not expansion and 'expansions' in data and data['expansions']:
                        # Get highest scoring expansion
                        best_exp = max(data['expansions'], key=lambda x: x.get('score', 0))
                        expansion = best_exp.get('expansion', '')
                    
                    if expansion:
                        self.abbreviation_dict[abbrev_upper] = {
                            'expansion': expansion,
                            'confidence': 0.9,
                            'source': 'abbreviation_general',
                            'ambiguous': data.get('ambiguous', False),
                            'needs_definition': data.get('needs_definition', False)
                        }
                        loaded_count += 1
                        
                        # Also create lexicon entry
                        entry = LexiconEntry(
                            cui='',  # No CUI in general abbreviations
                            preferred_term=expansion,
                            abbreviation=abbrev_upper,
                            source_vocabulary='GENERAL',
                            confidence_boost=0.1 if not data.get('ambiguous') else 0.0
                        )
                        if abbrev_upper not in self.lexicon:
                            self.lexicon[abbrev_upper] = []
                        self.lexicon[abbrev_upper].append(entry)
        
        # Load UMLS biological abbreviations (list format)
        if 'abbreviation_umls_biological' in resources:
            bio_data = resources['abbreviation_umls_biological']
            if isinstance(bio_data, list):
                for entry in bio_data:
                    if isinstance(entry, dict):
                        abbrev = entry.get('Abbreviation', '').strip().upper()
                        expansion = entry.get('Expansion', '').strip()
                        if abbrev and expansion:
                            # Add to abbreviation_dict if not already there or if better score
                            score = float(entry.get('Score', 0))
                            if abbrev not in self.abbreviation_dict or score > self.abbreviation_dict[abbrev].get('score', 0):
                                self.abbreviation_dict[abbrev] = {
                                    'expansion': expansion,
                                    'score': score,
                                    'source': entry.get('TopSource', 'UMLS_BIO'),
                                    'confidence': min(0.95, 0.5 + score * 0.1)
                                }
                                loaded_count += 1
                            
                            # Create lexicon entry
                            lex_entry = LexiconEntry(
                                cui='',
                                preferred_term=expansion,
                                abbreviation=abbrev,
                                source_vocabulary=entry.get('TopSource', 'UMLS'),
                                semantic_types=['biological'],
                                frequency=score
                            )
                            if abbrev not in self.lexicon:
                                self.lexicon[abbrev] = []
                            self.lexicon[abbrev].append(lex_entry)
        
        # Load UMLS clinical abbreviations (list format)
        if 'abbreviation_umls_clinical' in resources:
            clinical_data = resources['abbreviation_umls_clinical']
            if isinstance(clinical_data, list):
                for entry in clinical_data:
                    if isinstance(entry, dict):
                        abbrev = entry.get('Abbreviation', '').strip().upper()
                        expansion = entry.get('Expansion', '').strip()
                        if abbrev and expansion:
                            score = float(entry.get('Score', 0))
                            if abbrev not in self.abbreviation_dict or score > self.abbreviation_dict[abbrev].get('score', 0):
                                self.abbreviation_dict[abbrev] = {
                                    'expansion': expansion,
                                    'score': score,
                                    'source': entry.get('TopSource', 'UMLS_CLIN'),
                                    'semantic_group': entry.get('SemanticGroup', ''),
                                    'confidence': min(0.95, 0.5 + score * 0.1)
                                }
                                loaded_count += 1
                            
                            # Create lexicon entry
                            lex_entry = LexiconEntry(
                                cui='',
                                preferred_term=expansion,
                                abbreviation=abbrev,
                                source_vocabulary=entry.get('TopSource', 'UMLS'),
                                semantic_types=[entry.get('SemanticGroup', 'clinical')],
                                frequency=score
                            )
                            if abbrev not in self.lexicon:
                                self.lexicon[abbrev] = []
                            self.lexicon[abbrev].append(lex_entry)
        
        # Load rare disease acronyms
        if 'disease_rare_acronyms' in resources:
            rare_data = resources['disease_rare_acronyms']
            if isinstance(rare_data, dict):
                for abbrev, data in rare_data.items():
                    abbrev_upper = abbrev.upper()
                    name = data.get('name', '')
                    if name:
                        self.abbreviation_dict[abbrev_upper] = {
                            'expansion': name,
                            'orphacode': data.get('orphacode'),
                            'icd10': data.get('icd10_code'),
                            'source': 'orphanet',
                            'confidence': 0.95  # High confidence for rare diseases
                        }
                        loaded_count += 1
                        
                        # Create lexicon entry
                        lex_entry = LexiconEntry(
                            cui=f"ORPHA:{data.get('orphacode', '')}" if data.get('orphacode') else '',
                            preferred_term=name,
                            abbreviation=abbrev_upper,
                            source_vocabulary='ORPHANET',
                            semantic_types=['disease', 'rare_disease'],
                            confidence_boost=0.2  # Boost for rare diseases
                        )
                        if abbrev_upper not in self.lexicon:
                            self.lexicon[abbrev_upper] = []
                        self.lexicon[abbrev_upper].append(lex_entry)
        
        logger.info(f"Loaded {loaded_count} abbreviations from system resources")
        logger.debug(f"Total unique abbreviations in dict: {len(self.abbreviation_dict)}")
        logger.debug(f"Total unique abbreviations in lexicon: {len(self.lexicon)}")
    
    def load_lexicon(self, path: Path):
        """
        Load lexicon from file.
        
        Args:
            path: Path to lexicon file
        """
        try:
            if path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._process_json_lexicon(data)
            elif path.suffix in ['.tsv', '.txt']:
                self._process_tsv_lexicon(path)
            else:
                logger.error(f"Unsupported lexicon format: {path.suffix}")
            
            logger.info(f"Loaded lexicon with {len(self.lexicon)} abbreviations")
        except Exception as e:
            logger.error(f"Failed to load lexicon from {path}: {e}")
    
    def _process_json_lexicon(self, data: Dict):
        """Process JSON lexicon data"""
        for abbrev, entries in data.items():
            abbrev_upper = abbrev.upper()
            self.lexicon[abbrev_upper] = []
            
            for entry_data in entries:
                entry = LexiconEntry(
                    cui=entry_data.get('cui', ''),
                    preferred_term=entry_data.get('preferred_term', ''),
                    abbreviation=abbrev_upper,
                    semantic_types=entry_data.get('semantic_types', []),
                    source_vocabulary=entry_data.get('source', ''),
                    language=entry_data.get('language', 'ENG'),
                    synonyms=entry_data.get('synonyms', []),
                    definition=entry_data.get('definition'),
                    domain_keywords=entry_data.get('keywords', []),
                    frequency=entry_data.get('frequency', 0.0),
                    confidence_boost=entry_data.get('confidence_boost', 0.0)
                )
                self.lexicon[abbrev_upper].append(entry)
    
    def _process_tsv_lexicon(self, path: Path):
        """Process TSV lexicon file"""
        with open(path, 'r', encoding='utf-8') as f:
            header = None
            for line in f:
                parts = line.strip().split('\t')
                if not header:
                    header = parts
                    continue
                
                if len(parts) >= 3:
                    abbrev = parts[0].upper()
                    cui = parts[1]
                    term = parts[2]
                    
                    entry = LexiconEntry(
                        cui=cui,
                        preferred_term=term,
                        abbreviation=abbrev,
                        semantic_types=parts[3].split('|') if len(parts) > 3 else [],
                        source_vocabulary=parts[4] if len(parts) > 4 else '',
                        language=parts[5] if len(parts) > 5 else 'ENG'
                    )
                    
                    if abbrev not in self.lexicon:
                        self.lexicon[abbrev] = []
                    self.lexicon[abbrev].append(entry)
    
    def resolve_candidate(self, candidate: AbbreviationCandidate, 
                         context_domain: str = 'general') -> Tuple[bool, Optional[LexiconEntry]]:
        """
        Resolve an abbreviation candidate using lexicon.
        
        Args:
            candidate: The abbreviation candidate
            context_domain: Domain context
            
        Returns:
            Tuple of (should_send_to_claude, best_lexicon_entry)
        """
        abbrev_upper = candidate.abbreviation.upper()
        
        # Rule 1: Local definition present → Never send to Claude
        if candidate.local_expansion and candidate.confidence >= 0.9:
            logger.debug(f"Local definition for {abbrev_upper}, not sending to Claude")
            # Try to find CUI for the local expansion
            best_entry = self._find_cui_for_expansion(abbrev_upper, candidate.local_expansion)
            return False, best_entry
        
        # Rule 2: Check standard patterns
        if self._is_standard_pattern(abbrev_upper):
            logger.debug(f"Standard pattern {abbrev_upper}, accepting without Claude")
            return False, None
        
        # Rule 3: Check lexicon
        if abbrev_upper in self.lexicon:
            entries = self.lexicon[abbrev_upper]
            
            # Check overlay first
            if abbrev_upper in self.overlay:
                entries = [self.overlay[abbrev_upper]] + entries
            
            # Score all entries
            scored_entries = []
            for entry in entries:
                score = self._score_entry(entry, candidate, context_domain)
                scored_entries.append((score, entry))
            
            # Sort by score
            scored_entries.sort(key=lambda x: x[0], reverse=True)
            
            if scored_entries:
                best_score, best_entry = scored_entries[0]
                
                # Rule 4: Highly polysemous abbreviations
                if abbrev_upper in self.POLYSEMOUS_ABBREVS:
                    if best_score < 0.9:  # Need very high confidence
                        logger.debug(f"Polysemous {abbrev_upper} with low score {best_score:.2f}, sending to Claude")
                        return True, None
                
                # Rule 5: Check acceptance threshold and margin
                if best_score >= self.tau_accept:
                    # Check margin to second best
                    if len(scored_entries) > 1:
                        second_score = scored_entries[1][0]
                        if best_score - second_score < self.delta_margin:
                            logger.debug(f"Close scores for {abbrev_upper}: {best_score:.2f} vs {second_score:.2f}, sending to Claude")
                            return True, None
                    
                    # Accept the best entry
                    logger.debug(f"Accepting lexicon entry for {abbrev_upper}: {best_entry.preferred_term} (score: {best_score:.2f})")
                    candidate.expansion = best_entry.preferred_term
                    candidate.confidence = best_score
                    candidate.validation_status = ValidationStatus.DICTIONARY_MATCH
                    candidate.metadata['cui'] = best_entry.cui
                    candidate.metadata['semantic_types'] = best_entry.semantic_types
                    return False, best_entry
                else:
                    logger.debug(f"Low score {best_score:.2f} for {abbrev_upper}, sending to Claude")
                    return True, None
        
        # Not in lexicon, send to Claude
        logger.debug(f"{abbrev_upper} not in lexicon, sending to Claude")
        return True, None
    
    def _score_entry(self, entry: LexiconEntry, candidate: AbbreviationCandidate, 
                    domain: str) -> float:
        """
        Score a lexicon entry against a candidate.
        
        Args:
            entry: Lexicon entry
            candidate: Abbreviation candidate
            domain: Domain context
            
        Returns:
            Score between 0 and 1
        """
        score = 0.5  # Base score
        
        # Domain match
        if entry.matches_domain(domain):
            score += 0.2
        
        # Acronym match
        if candidate.expansion:
            acronym_score = self._calculate_acronym_match(
                candidate.abbreviation, entry.preferred_term
            )
            score += 0.2 * acronym_score
        
        # Source authority
        if entry.source_vocabulary in self.source_priority:
            rank = self.source_priority.index(entry.source_vocabulary)
            score += 0.1 * (1 - rank / len(self.source_priority))
        
        # Language preference
        if entry.language == self.prefer_language:
            score += 0.05
        
        # Keyword match in context
        if entry.domain_keywords and candidate.context:
            context_lower = candidate.context.lower()
            keyword_matches = sum(1 for kw in entry.domain_keywords 
                                 if kw.lower() in context_lower)
            if keyword_matches:
                score += min(0.15, 0.05 * keyword_matches)
        
        # Frequency boost
        score += min(0.1, entry.frequency * 0.1)
        
        # Configured confidence boost
        score += entry.confidence_boost
        
        return min(1.0, score)
    
    def _is_standard_pattern(self, abbrev: str) -> bool:
        """Check if abbreviation matches a standard pattern"""
        for pattern_name, pattern in self.STANDARD_PATTERNS.items():
            if pattern.match(abbrev):
                return True
        return False
    
    def _calculate_acronym_match(self, abbrev: str, expansion: str) -> float:
        """Calculate how well abbreviation matches expansion"""
        abbrev_clean = re.sub(r'[^A-Za-z]', '', abbrev).upper()
        words = [w for w in re.split(r'[\s\-]+', expansion) if w]
        
        if not words:
            return 0.0
        
        # Get initials
        initials = ''.join(w[0].upper() for w in words if w[0].isalpha())
        
        if abbrev_clean == initials:
            return 1.0
        
        # Calculate partial match
        matches = sum(1 for a, i in zip(abbrev_clean, initials) if a == i)
        return matches / max(len(abbrev_clean), len(initials))
    
    def _find_cui_for_expansion(self, abbrev: str, expansion: str) -> Optional[LexiconEntry]:
        """Find CUI for a given expansion"""
        abbrev_upper = abbrev.upper()
        if abbrev_upper in self.lexicon:
            expansion_lower = expansion.lower()
            for entry in self.lexicon[abbrev_upper]:
                if entry.preferred_term.lower() == expansion_lower:
                    return entry
                if any(syn.lower() == expansion_lower for syn in entry.synonyms):
                    return entry
        return None
    
    def add_overlay_entry(self, abbrev: str, entry: LexiconEntry):
        """
        Add project-specific overlay entry.
        
        Args:
            abbrev: Abbreviation
            entry: Lexicon entry
        """
        self.overlay[abbrev.upper()] = entry
        logger.debug(f"Added overlay entry for {abbrev}: {entry.preferred_term}")
    
    def update_frequency(self, abbrev: str, cui: str):
        """
        Update frequency for continuous learning.
        
        Args:
            abbrev: Abbreviation
            cui: CUI that was selected
        """
        abbrev_upper = abbrev.upper()
        if abbrev_upper in self.lexicon:
            for entry in self.lexicon[abbrev_upper]:
                if entry.cui == cui:
                    entry.frequency += 1.0
                    break


# ============================================================================
# INTEGRATION FUNCTION
# ============================================================================

def integrate_lexicon_resolution(candidates: List[AbbreviationCandidate],
                                lexicon_resolver: LexiconResolver,
                                context_domain: str = 'general') -> Tuple[List[AbbreviationCandidate], List[AbbreviationCandidate]]:
    """
    Integrate lexicon resolution into abbreviation pipeline.
    
    Args:
        candidates: List of abbreviation candidates
        lexicon_resolver: Configured lexicon resolver
        context_domain: Domain context
        
    Returns:
        Tuple of (resolved_candidates, candidates_for_claude)
    """
    resolved = []
    for_claude = []
    
    for candidate in candidates:
        should_send_to_claude, lexicon_entry = lexicon_resolver.resolve_candidate(
            candidate, context_domain
        )
        
        if should_send_to_claude:
            for_claude.append(candidate)
        else:
            if lexicon_entry:
                # Update candidate with lexicon information
                candidate.metadata['lexicon_resolved'] = True
                candidate.metadata['cui'] = lexicon_entry.cui
                candidate.metadata['semantic_types'] = lexicon_entry.semantic_types
                candidate.metadata['source_vocabulary'] = lexicon_entry.source_vocabulary
            resolved.append(candidate)
    
    logger.info(f"Lexicon resolved {len(resolved)} candidates, "
                f"sending {len(for_claude)} to Claude")
    
    return resolved, for_claude