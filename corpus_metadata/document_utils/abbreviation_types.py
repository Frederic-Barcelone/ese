#!/usr/bin/env python3
"""
Shared Types Module for Abbreviation Extraction
================================================
Location: corpus_metadata/document_utils/abbreviation_types.py
Version: 1.0.0

Centralized data classes and types for abbreviation extraction modules.
This prevents duplication and ensures consistency across all abbreviation modules.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set

# ============================================================================
# CENTRALIZED LOGGING CONFIGURATION
# ============================================================================
try:
    from corpus_metadata.document_utils.metadata_logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    try:
        # Try relative import
        from .metadata_logging_config import get_logger
        logger = get_logger(__name__)
    except ImportError:
        # Fallback to basic logging if centralized config not available
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger(__name__)


# ============================================================================
# SHARED DATA CLASSES
# ============================================================================

@dataclass
class AbbreviationCandidate:
    """
    Represents a potential abbreviation found in text.
    
    This is the central data structure used across all abbreviation modules:
    - abbreviation_patterns.py: Creates candidates from pattern detection
    - abbreviation_context.py: Enhances candidates with context scoring
    - abbreviation_claude.py: Resolves difficult candidates using AI
    - abbreviation_extraction.py: Orchestrates the full pipeline
    
    Attributes:
        abbreviation: The abbreviated form (e.g., "SMA", "TNF-Î±")
        expansion: The full expansion (e.g., "Spinal muscular atrophy")
        confidence: Confidence score between 0.0 and 1.0
        source: Origin of the candidate ('document', 'dictionary', 'claude_inference', etc.)
        detection_source: Specific pattern that detected it ('definition_parentheses', etc.)
        dictionary_sources: List of dictionaries where found
        context: Text window around the abbreviation
        position: Character position in document
        page_number: Page number in document (-1 if unknown)
        validation_status: Current validation state
        disambiguation_needed: Whether multiple expansions are possible
        alternative_expansions: Other possible expansions
        context_type: Domain context ('medical', 'disease', 'drug', etc.)
        local_expansion: Expansion found in the same document
        context_score: Score from context analysis
        domain_context: Specific domain identified
        claude_resolved: Whether Claude has processed this
        old_confidence: Previous confidence (for audit trail)
        resolution_reason: Reason for resolution/update
        metadata: Additional flexible metadata
    """
    # Core fields
    abbreviation: str
    expansion: str
    confidence: float = 0.0
    
    # Source tracking
    source: str = 'document'
    detection_source: str = ''
    dictionary_sources: List[str] = field(default_factory=list)
    
    # Context information
    context: str = ''
    position: int = -1
    page_number: int = -1
    
    # Validation and disambiguation
    validation_status: str = 'unvalidated'
    disambiguation_needed: bool = False
    alternative_expansions: List[str] = field(default_factory=list)
    
    # Domain and context analysis
    context_type: str = 'general'
    semantic_type: Optional[str] = None  # Inferred semantic category: 'disease', 'drug', 'identifier', etc.
    local_expansion: Optional[str] = None
    context_score: float = 0.0
    domain_context: str = ''
    
    # Claude resolution
    claude_resolved: bool = False
    
    # Audit trail
    old_confidence: Optional[float] = None
    resolution_reason: Optional[str] = None
    
    # Flexible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Ensure confidence is in valid range
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Ensure lists are not None
        if self.dictionary_sources is None:
            self.dictionary_sources = []
        if self.alternative_expansions is None:
            self.alternative_expansions = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'abbreviation': self.abbreviation,
            'expansion': self.expansion,
            'confidence': self.confidence,
            'source': self.source,
            'detection_source': self.detection_source,
            'dictionary_sources': self.dictionary_sources,
            'context': self.context,
            'position': self.position,
            'page_number': self.page_number,
            'validation_status': self.validation_status,
            'disambiguation_needed': self.disambiguation_needed,
            'alternative_expansions': self.alternative_expansions,
            'context_type': self.context_type,
            'semantic_type': self.semantic_type,
            'local_expansion': self.local_expansion,
            'context_score': self.context_score,
            'domain_context': self.domain_context,
            'claude_resolved': self.claude_resolved,
            'old_confidence': self.old_confidence,
            'resolution_reason': self.resolution_reason,
            'metadata': self.metadata
        }
    
    def update_confidence(self, new_confidence: float, reason: str = None):
        """
        Update confidence with audit trail.
        
        Args:
            new_confidence: New confidence value
            reason: Reason for the update
        """
        if self.confidence != new_confidence:
            self.old_confidence = self.confidence
            self.confidence = max(0.0, min(1.0, new_confidence))
            if reason:
                self.resolution_reason = reason
            logger.debug(f"Updated confidence for {self.abbreviation}: "
                        f"{self.old_confidence:.2f} -> {self.confidence:.2f} ({reason})")
    
    def add_alternative(self, expansion: str):
        """
        Add an alternative expansion if not already present.
        
        Args:
            expansion: Alternative expansion to add
        """
        if expansion and expansion not in self.alternative_expansions:
            if expansion != self.expansion:  # Don't add primary as alternative
                self.alternative_expansions.append(expansion)
                self.disambiguation_needed = len(self.alternative_expansions) > 0
    
    def merge_with(self, other: 'AbbreviationCandidate'):
        """
        Merge information from another candidate.
        
        Args:
            other: Another AbbreviationCandidate for the same abbreviation
        """
        # Update confidence to maximum
        if other.confidence > self.confidence:
            self.update_confidence(other.confidence, 'merged_higher_confidence')
        
        # Merge dictionary sources
        for source in other.dictionary_sources:
            if source not in self.dictionary_sources:
                self.dictionary_sources.append(source)
        
        # Merge alternatives
        for alt in other.alternative_expansions:
            self.add_alternative(alt)
        
        # Update expansion if other has higher confidence
        if other.expansion and other.confidence > self.confidence:
            self.expansion = other.expansion
        
        # Merge metadata
        self.metadata.update(other.metadata)
    
    def __str__(self) -> str:
        """String representation"""
        return f"AbbreviationCandidate('{self.abbreviation}' -> '{self.expansion}', confidence={self.confidence:.2f})"
    
    def __repr__(self) -> str:
        """Developer representation"""
        return (f"AbbreviationCandidate(abbreviation='{self.abbreviation}', "
                f"expansion='{self.expansion}', confidence={self.confidence:.2f}, "
                f"source='{self.source}')")


# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

class ValidationStatus:
    """Constants for validation status"""
    UNVALIDATED = 'unvalidated'
    VALIDATED = 'validated'
    INVALID = 'invalid'
    AMBIGUOUS = 'ambiguous'
    CLAUDE_CONTEXT = 'claude_context'
    DICTIONARY_MATCH = 'dictionary_match'
    LOCAL_DEFINITION = 'local_definition'
    PATTERN_MATCH = 'pattern_match'
    CONTEXT_RESOLVED = 'context_resolved'


class SourceType:
    """Constants for source types"""
    DOCUMENT = 'document'
    DICTIONARY = 'dictionary'
    CLAUDE_INFERENCE = 'claude_inference'
    PATTERN = 'pattern'
    CONTEXT = 'context'
    HYBRID = 'hybrid'
    KB_RESOLUTION = 'kb_resolution'
    MANUAL = 'manual'


class DetectionSource:
    """Constants for detection sources"""
    DEFINITION_PARENTHESES = 'definition_parentheses'
    ABBREVIATION_FIRST = 'abbreviation_first'
    COLON_DASH_DEFINITION = 'colon_dash_definition'
    STANDALONE_UPPERCASE = 'standalone_uppercase'
    STANDALONE_PARENTHESES = 'standalone_parentheses'
    ALPHANUMERIC = 'alphanumeric'
    HYPHENATED = 'hyphenated'
    SLASH_SEPARATED = 'slash_separated'
    MIXED_CASE = 'mixed_case'
    COMPLEMENT_SYSTEM = 'complement_system'
    WITH_GREEK = 'with_greek'
    LETTER_DIGIT = 'letter_digit'
    HLA_ALLELE = 'hla_allele'


class ContextType:
    """Constants for context types"""
    GENERAL = 'general'
    MEDICAL = 'medical'
    DISEASE = 'disease'
    DRUG = 'drug'
    PHARMACEUTICAL = 'pharmaceutical'
    BIOLOGICAL = 'biological'
    CLINICAL = 'clinical'
    RESEARCH = 'research'
    REGULATORY = 'regulatory'
    ORGANIZATION = 'organization'
    STATISTICAL = 'statistical'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_candidate(abbreviation: str, expansion: str = '', 
                    confidence: float = 0.0, **kwargs) -> AbbreviationCandidate:
    """
    Factory function to create an AbbreviationCandidate.
    
    Args:
        abbreviation: The abbreviation
        expansion: The expansion (optional)
        confidence: Confidence score (optional)
        **kwargs: Additional fields
        
    Returns:
        New AbbreviationCandidate instance
    """
    return AbbreviationCandidate(
        abbreviation=abbreviation,
        expansion=expansion,
        confidence=confidence,
        **kwargs
    )


def merge_candidates(candidates: List[AbbreviationCandidate]) -> List[AbbreviationCandidate]:
    """
    Merge duplicate candidates for the same abbreviation.
    
    Args:
        candidates: List of candidates to merge
        
    Returns:
        List of merged candidates
    """
    merged = {}
    
    for candidate in candidates:
        key = candidate.abbreviation.upper()
        
        if key in merged:
            merged[key].merge_with(candidate)
        else:
            merged[key] = candidate
    
    return list(merged.values())


def filter_by_confidence(candidates: List[AbbreviationCandidate], 
                        min_confidence: float = 0.5) -> List[AbbreviationCandidate]:
    """
    Filter candidates by minimum confidence threshold.
    
    Args:
        candidates: List of candidates
        min_confidence: Minimum confidence threshold
        
    Returns:
        Filtered list of candidates
    """
    return [c for c in candidates if c.confidence >= min_confidence]


def filter_by_source(candidates: List[AbbreviationCandidate], 
                    sources: Set[str]) -> List[AbbreviationCandidate]:
    """
    Filter candidates by source type.
    
    Args:
        candidates: List of candidates
        sources: Set of allowed source types
        
    Returns:
        Filtered list of candidates
    """
    return [c for c in candidates if c.source in sources]


def filter_by_context_type(candidates: List[AbbreviationCandidate], 
                          context_types: Set[str]) -> List[AbbreviationCandidate]:
    """
    Filter candidates by context type.
    
    Args:
        candidates: List of candidates
        context_types: Set of allowed context types
        
    Returns:
        Filtered list of candidates
    """
    return [c for c in candidates if c.context_type in context_types]


def sort_candidates(candidates: List[AbbreviationCandidate], 
                   by: str = 'confidence') -> List[AbbreviationCandidate]:
    """
    Sort candidates by specified field.
    
    Args:
        candidates: List of candidates
        by: Field to sort by ('confidence', 'abbreviation', 'position')
        
    Returns:
        Sorted list of candidates
    """
    if by == 'confidence':
        return sorted(candidates, key=lambda c: c.confidence, reverse=True)
    elif by == 'abbreviation':
        return sorted(candidates, key=lambda c: c.abbreviation)
    elif by == 'position':
        return sorted(candidates, key=lambda c: c.position)
    else:
        return candidates


# ============================================================================
# EXPORT ALL PUBLIC ITEMS
# ============================================================================

__all__ = [
    # Main data class
    'AbbreviationCandidate',
    
    # Constants
    'ValidationStatus',
    'SourceType',
    'DetectionSource',
    'ContextType',
    
    # Helper functions
    'create_candidate',
    'merge_candidates',
    'filter_by_confidence',
    'filter_by_source',
    'filter_by_context_type',
    'sort_candidates',
]