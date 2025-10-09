#!/usr/bin/env python3
"""
Context Analysis Module for Abbreviation Extraction
===================================================
Location: corpus_metadata/document_utils/abbreviation_context.py
Version: 1.3.0

Handles context analysis, text normalization, and domain-specific disambiguation.

IMPROVEMENTS IN VERSION 1.3.0:
- Added comprehensive article identifier support (NEJM, PMC, DOI, etc.)
- Added _generate_article_identifier_expansion method
- Added _is_non_expandable_term method for standard nomenclature
- Enhanced common word filtering
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple, Any
from functools import lru_cache
from collections import defaultdict
from hashlib import md5

# ============================================================================
# ARTICLE IDENTIFIER PREFIXES
# ============================================================================
ARTICLE_IDENTIFIER_PREFIXES = {
    # --- NCBI / PubMed / Bookshelf ---
    'PMCID': 'PubMed Central ID',
    'PMC': 'PubMed Central article',
    'PMID': 'PubMed ID',
    'NBK': 'NCBI Bookshelf ID (National Bookshelf Knowledge Base)',
    'NIHMS': 'NIH Manuscript Submission ID',

    # --- DOI / publisher item IDs ---
    'DOI': 'Digital Object Identifier',
    'doi:': 'Digital Object Identifier',
    'PII': 'Publisher Item Identifier (Elsevier/others)',
    'S': 'Elsevier PII prefix (S + ISSN + article code)',

    # --- NEJM section codes ---
    'NEJMoa': 'New England Journal of Medicine original article',
    'NEJMra': 'New England Journal of Medicine review article',
    'NEJMcps': 'New England Journal of Medicine clinical problem-solving',
    'NEJMclde': 'New England Journal of Medicine clinical decisions',
    'NEJMvcm': 'New England Journal of Medicine videos in clinical medicine',
    'NEJMicm': 'New England Journal of Medicine images in clinical medicine',
    'NEJMp': 'New England Journal of Medicine perspective',
    'NEJMms': 'New England Journal of Medicine medical student',
    'NEJMc': 'New England Journal of Medicine correspondence',
    'NEJMe': 'New England Journal of Medicine editorial',
    'NEJM': 'New England Journal of Medicine (generic code/prefix)',

    # --- Preprints / repositories ---
    'arXiv:': 'arXiv preprint',
    'BIOXRIV': 'bioRxiv preprint',
    'BIORXIV': 'bioRxiv preprint',
    'biorxiv': 'bioRxiv preprint',
    'MEDXRIV': 'medRxiv preprint',
    'MEDRXIV': 'medRxiv preprint',
    'medrxiv': 'medRxiv preprint',
    'CHEMRXIV': 'ChemRxiv preprint',
    'chemrxiv': 'ChemRxiv preprint',
    'SSRN': 'Social Science Research Network',
    'OSF': 'Open Science Framework preprint',
    'HAL': 'HAL open archive identifier',
    'PPR': 'Europe PMC preprint record ID',

    # --- Clinical trial registries ---
    'NCT': 'ClinicalTrials.gov identifier',
    'ISRCTN': 'ISRCTN registry identifier',
    'EudraCT': 'EU Clinical Trials Register (EudraCT) identifier',
    'EUCTR': 'EU Clinical Trials Register identifier',
    'ChiCTR': 'Chinese Clinical Trial Registry identifier',
    'ANZCTR': 'Australia New Zealand Clinical Trials Registry identifier',
    'UMIN': 'UMIN Clinical Trials Registry (Japan) identifier',
    'JPRN': 'Japan Primary Registries Network identifier',
    'DRKS': 'German Clinical Trials Register identifier',
    'IRCT': 'Iranian Registry of Clinical Trials identifier',
    'CTRI': 'Clinical Trials Registry - India identifier',
    'KCT': 'Korean Clinical Trial Registry identifier',
    'PACTR': 'Pan African Clinical Trials Registry identifier',
    'ReBec': 'Registro Brasileiro de Ensaios Clínicos identifier',

    # --- Evidence syntheses / registries ---
    'PROSPERO': 'International prospective register of systematic reviews ID',
    'CRD': 'Centre for Reviews and Dissemination record ID',
}

# ============================================================================
# ACRONYM STOPWORDS  
# ============================================================================
ACRONYM_STOPWORDS = {"and", "of", "the", "de", "la", "del", "y", "da", "do", "for", "in", "on", "with"}

# ============================================================================
# COMMON WORDS TO FILTER
# ============================================================================
COMMON_WORDS = {
    'the', 'and', 'for', 'with', 'from', 'that', 'this', 'will',
    'would', 'could', 'should', 'have', 'has', 'had', 'been',
    'are', 'was', 'were', 'been', 'being', 'having', 'does', 'did',
    'will', 'would', 'should', 'could', 'may', 'might', 'must',
    'can', 'could', 'shall', 'ought', 'need', 'dare'
}

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
# IMPORT SHARED DATA CLASSES
# ============================================================================
try:
    from .abbreviation_types import AbbreviationCandidate
except ImportError:
    pass


class TokenNormalizer:
    """Handles pre-extraction token normalization"""
    
    PATTERNS = {
        'hyphenation': re.compile(r'(\w+)-\s*\n\s*(\w+)'),
        'soft_hyphen': re.compile(r'\u00AD'),
        'multiple_spaces': re.compile(r'\s{2,}'),
        'tab_spaces': re.compile(r'\t+'),
        'line_continuation': re.compile(r'(\w+)\s*\n\s*(\w+)(?=[a-z])'),
        'bullet_points': re.compile(r'^\s*[•·▪▫◦‣⁃]\s*', re.MULTILINE),
        'list_markers': re.compile(r'^\s*\(?[a-zA-Z0-9]+[.)]\s*', re.MULTILINE),
    }
    
    def normalize(self, text: str) -> str:
        """Apply all normalization patterns"""
        # Remove soft hyphens
        text = self.PATTERNS['soft_hyphen'].sub('', text)
        
        # Fix broken hyphenation
        text = self.PATTERNS['hyphenation'].sub(r'\1\2', text)
        
        # Clean whitespace
        text = self.PATTERNS['multiple_spaces'].sub(' ', text)
        text = self.PATTERNS['tab_spaces'].sub(' ', text)
        
        # Remove list markers
        text = self.PATTERNS['bullet_points'].sub('', text)
        text = self.PATTERNS['list_markers'].sub('', text)
        
        return text.strip()


class StructuralNoiseFilter:
    """Filters out structural noise from documents"""
    
    @classmethod
    def should_filter(cls, text: str, position: int = 0, window: int = 15) -> bool:
        """
        Check if text at position should be filtered as noise.
        
        Args:
            text: Full text
            position: Position to check
            window: Context window size
            
        Returns:
            True if should filter, False otherwise
        """
        # For now, return False to not filter anything
        # This can be enhanced later with actual noise detection
        return False


class DomainContextManager:
    """Manages domain-specific context and disambiguation"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.domain_patterns = self._load_domain_patterns()
        self.common_pairs = self._load_common_pairs()
        self.domain_abbreviations = self._load_domain_abbreviations()
    
    def _load_domain_patterns(self) -> Dict[str, List[str]]:
        """Load domain-specific keyword patterns"""
        return {
            'medical': ['patient', 'treatment', 'therapy', 'clinical', 'disease', 
                       'diagnosis', 'symptoms', 'medication', 'hospital'],
            'pharmacology': ['drug', 'dose', 'pharmacokinetic', 'metabolism', 
                            'receptor', 'inhibitor', 'agonist', 'antagonist'],
            'genetics': ['gene', 'mutation', 'chromosome', 'allele', 'genome',
                        'transcription', 'expression', 'variant'],
            'immunology': ['antibody', 'antigen', 'immune', 'lymphocyte', 
                          'cytokine', 'inflammation', 'autoimmune'],
            'clinical_trial': ['trial', 'randomized', 'placebo', 'endpoint',
                              'efficacy', 'safety', 'adverse', 'phase'],
            'organization': ['organization', 'institute', 'foundation', 'association',
                            'society', 'committee', 'agency', 'university']
        }
    
    def _load_common_pairs(self) -> Dict[str, str]:
        """Load commonly accepted abbreviation-expansion pairs"""
        return {
            'FDA': 'Food and Drug Administration',
            'NIH': 'National Institutes of Health',
            'CDC': 'Centers for Disease Control and Prevention',
            'WHO': 'World Health Organization',
            'EMA': 'European Medicines Agency',
            'MHRA': 'Medicines and Healthcare products Regulatory Agency',
        }
    
    def _load_domain_abbreviations(self) -> Dict[str, Any]:
        """Load domain-specific abbreviations"""
        return {
            'IL': {'default': 'interleukin', 
                   'immunology_context': 'interleukin',
                   'geography_context': 'Illinois'},
            'MS': {'default': 'multiple sclerosis',
                   'medical_context': 'multiple sclerosis',
                   'chemistry_context': 'mass spectrometry'},
            'TNF': 'tumor necrosis factor',
            'EGFR': 'epidermal growth factor receptor',
            'VEGF': 'vascular endothelial growth factor',
        }
    
    def get_domain_context(self, text: str) -> str:
        """Determine the primary domain from text context"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if not domain_scores:
            return 'general'
        
        return max(domain_scores, key=domain_scores.get)
    
    def get_expansion_for_domain(self, abbr: str, domain: str) -> Optional[str]:
        """Get domain-specific expansion"""
        abbr_upper = abbr.upper()
        
        if abbr_upper in self.common_pairs:
            return self.common_pairs[abbr_upper]
        
        if abbr_upper in self.domain_abbreviations:
            abbr_info = self.domain_abbreviations[abbr_upper]
            
            if isinstance(abbr_info, dict):
                context_key = f"{domain}_context"
                return abbr_info.get(context_key, abbr_info.get('default'))
            else:
                return abbr_info
        
        return None


class ContextScoringEngine:
    """Advanced context-aware scoring for abbreviation disambiguation"""
    
    def __init__(self, domain_manager: DomainContextManager):
        self.domain_manager = domain_manager
        self.cache = {}
        self._static_cache = {}
    
    def _get_context_hash(self, context: str) -> str:
        """Create a short hash of context for caching"""
        return md5(context.encode()).hexdigest()[:8]
    
    @lru_cache(maxsize=1000)
    def _score_static_components(self, abbr: str, expansion: str, domain: str) -> Tuple[float, float, float]:
        """Score static components that don't depend on full context"""
        acronym_score = self._score_acronym_match(abbr, expansion)
        domain_score = self._score_domain_relevance(expansion, domain)
        frequency_score = self._score_frequency(abbr, expansion)
        
        return acronym_score, domain_score, frequency_score
    
    def score_expansion(self, abbr: str, expansion: str, context: str, 
                       domain: Optional[str] = None) -> float:
        """Score how well an expansion fits the context"""
        if not expansion:
            return 0.0
        
        if domain is None:
            domain = self.domain_manager.get_domain_context(context)
        
        # Get cached static scores
        acronym_score, domain_score, frequency_score = self._score_static_components(
            abbr, expansion, domain
        )
        
        # Dynamic context scoring
        proximity_score = self._score_proximity(expansion, context)
        semantic_score = self._score_semantic_match(expansion, context)
        
        # Weighted combination
        weights = {
            'acronym': 0.35,
            'proximity': 0.25,
            'semantic': 0.20,
            'domain': 0.15,
            'frequency': 0.05
        }
        
        final_score = (
            weights['acronym'] * acronym_score +
            weights['proximity'] * proximity_score +
            weights['semantic'] * semantic_score +
            weights['domain'] * domain_score +
            weights['frequency'] * frequency_score
        )
        
        return min(1.0, max(0.0, final_score))
    
    def score_context(self, abbr: str, expansion: str, context: str, 
                     domain: Optional[str] = None) -> float:
        """Simplified interface for context scoring"""
        return self.score_expansion(abbr, expansion, context, domain)
    
    def _score_acronym_match(self, abbr: str, expansion: str) -> float:
        """Score acronym pattern matching with stopword filtering"""
        if not expansion:
            return 0.0
        
        # Split expansion into words
        words = expansion.split()
        
        # Filter out stopwords
        significant_words = [w for w in words if w.lower() not in ACRONYM_STOPWORDS]
        
        # Get first letters of significant words
        first_letters = ''.join(w[0].upper() for w in significant_words if w)
        
        # Check exact match
        if first_letters == abbr.upper():
            return 1.0
        
        # Check if abbreviation is contained in first letters
        if abbr.upper() in first_letters:
            return 0.8
        
        # Partial matching
        matches = sum(1 for a, b in zip(abbr.upper(), first_letters) if a == b)
        return matches / max(len(abbr), len(first_letters)) if first_letters else 0.0
    
    def _score_proximity(self, expansion: str, context: str) -> float:
        """Score based on proximity of expansion words to abbreviation"""
        if not expansion or not context:
            return 0.0
        
        expansion_words = expansion.lower().split()
        context_lower = context.lower()
        
        # Check if any expansion words appear in context
        found_words = sum(1 for word in expansion_words if word in context_lower)
        
        if found_words == 0:
            return 0.0
        
        return found_words / len(expansion_words)
    
    def _score_semantic_match(self, expansion: str, context: str) -> float:
        """Score semantic relevance of expansion to context"""
        if not expansion or not context:
            return 0.0
        
        # Simple keyword matching for now
        medical_terms = {'disease', 'syndrome', 'disorder', 'treatment', 
                        'therapy', 'patient', 'clinical'}
        
        expansion_lower = expansion.lower()
        context_lower = context.lower()
        
        # Check medical relevance
        is_medical_expansion = any(term in expansion_lower for term in medical_terms)
        is_medical_context = any(term in context_lower for term in medical_terms)
        
        if is_medical_expansion == is_medical_context:
            return 0.8
        
        return 0.3
    
    def _score_domain_relevance(self, expansion: str, domain: str) -> float:
        """Score expansion relevance to identified domain"""
        if not expansion:
            return 0.0
        
        expansion_lower = expansion.lower()
        
        # Get domain keywords
        domain_keywords = self.domain_manager.domain_patterns.get(domain, [])
        
        # Check keyword matches
        matches = sum(1 for keyword in domain_keywords if keyword in expansion_lower)
        
        if matches > 2:
            return 1.0
        elif matches > 0:
            return 0.5 + (matches * 0.25)
        
        return 0.2
    
    def _score_frequency(self, abbr: str, expansion: str) -> float:
        """Score based on frequency/commonality of the abbreviation"""
        # Check if it's a known common pair
        if abbr.upper() in self.domain_manager.common_pairs:
            if self.domain_manager.common_pairs[abbr.upper()] == expansion:
                return 1.0
        
        # Default middle score
        return 0.5


class ContextAnalyzer:
    """Main context analyzer for abbreviation extraction"""
    
    # Common words that should not be treated as abbreviations
    COMMON_WORDS = COMMON_WORDS
    
    def __init__(self, config: Dict = None):
        """Initialize the context analyzer"""
        self.config = config or {}
        self.normalizer = TokenNormalizer()
        self.domain_manager = DomainContextManager(config)
        self.scoring_engine = ContextScoringEngine(self.domain_manager)
        
        # Preprocessing configuration
        self.preprocess_config = self.config.get('preprocessing', {})
        
        # Use the StructuralNoiseFilter class
        self.noise_filter = StructuralNoiseFilter
        
        logger.info("Initialized ContextAnalyzer")
    
    def preprocess_text(self, text: str, config: Dict = None) -> str:
        """Preprocess text for abbreviation extraction"""
        if config is None:
            config = self.preprocess_config
        
        # Normalize tokens first
        text = self.normalizer.normalize(text)
        
        # Apply additional preprocessing based on config
        if config.get('remove_noise', True):
            text = self._remove_noise(text, config)
        
        if config.get('fix_pdf_artifacts', True):
            text = self._fix_pdf_artifacts(text)
        
        return text
    
    def _remove_noise(self, text: str, preprocess: Dict) -> str:
        """Remove various types of noise from text"""
        if preprocess.get('remove_page_numbers', True):
            text = re.compile(r'^\s*\d{1,4}\s*$', re.MULTILINE).sub('', text)
            text = re.compile(r'Page \d+ of \d+', re.IGNORECASE).sub('', text)
        
        if preprocess.get('remove_references', True):
            # Only remove references section if it's at the end
            m = re.search(r'(References?|Bibliography|Works Cited)\s*(:|\n|$)', text)
            if m and m.start() > int(len(text) * 0.6):
                text = text[:m.start()]
        
        if preprocess.get('remove_formulas', True):
            text = re.sub(r'\$\$[^$]+\$\$', '', text)  # Display math
            text = re.sub(r'\$[^$]+\$', '', text)      # Inline math
        
        return text
    
    def _fix_pdf_artifacts(self, text: str) -> str:
        """Fix PDF extraction artifacts"""
        
        # Fix ANCA- type breaks (abbreviation split across lines)
        text = re.sub(r'\b([A-Z]{2,})-\s*\n\s*', r'\1-', text)
    
        # Fix word breaks like "of-concept" -> "of-concept"
        text = re.sub(r'\b(\w+)-\s*\n\s*(\w+)\b', r'\1-\2', text)
    
        # Fix missing spaces after periods before references
        text = re.sub(r'(\.)(\d+\.\s+[A-Z])', r'\1 \2', text)
        
        # Fix hyphenated words broken across lines
        text = re.sub(r'(?<=[a-z])-\n(?=[a-z])', '', text)
        
        # Medical terms with hyphens
        text = re.sub(r'([A-Z]{2,})-\n([A-Z])', r'\1-\2', text)
        
        # Fix common OCR errors
        text = text.replace('adults-but', 'adults but')
        text = text.replace('renal-limited', 'renal limited')
        
        # Ensure space after periods
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        
        # Join split biomedical tokens
        biomedical_prefixes = ['MPO', 'PR3', 'CD', 'C', 'IL', 'TNF', 'GPA', 'MPA']
        for prefix in biomedical_prefixes:
            text = re.sub(rf'\b({prefix})\s+(ANCA)\b', r'\1-\2', text)
            text = re.sub(rf'\b({prefix})\s+(\d{{1,4}})\b', r'\1\2', text)
        
        # Replace mid-sentence newlines with spaces
        text = re.sub(r'(?<![.!?])\n(?![A-Z])', ' ', text)
        
        return text
    

    def _is_in_references_section(self, text: str, position: int) -> bool:
        """Check if position is in references section"""
        # Find "References" heading
        ref_match = re.search(r'\n(References?|REFERENCES?)\s*\n', text)
        if ref_match and position > ref_match.start():
            return True
        return False


    def analyze_context(self, candidate: 'AbbreviationCandidate', text: str) -> 'AbbreviationCandidate':
        """Analyze context around abbreviation"""
        # Check if abbreviation is a common word
        if candidate.abbreviation.lower() in self.COMMON_WORDS:
            candidate.confidence = 0.1
            candidate.validation_status = 'likely_false_positive'
            logger.debug(f"Marked '{candidate.abbreviation}' as likely false positive (common word)")
            return candidate
        
        # Get context window
        context_start = max(0, candidate.position - 200)
        context_end = min(len(text), candidate.position + 200)
        context = text[context_start:context_end]
        
        # Determine domain
        domain = self.domain_manager.get_domain_context(context)
        candidate.domain_context = domain
        
        # Score the context
        context_score = self.scoring_engine.score_context(
            candidate.abbreviation,
            candidate.expansion,
            context,
            domain
        )
        candidate.context_score = context_score
        
        # Adjust confidence
        if context_score > 0.8:
            candidate.confidence = min(1.0, candidate.confidence * 1.2)
        elif context_score < 0.3:
            candidate.confidence *= 0.8
        
        # Determine context type
        if candidate.expansion:
            candidate.context_type = self.get_context_type(candidate.expansion)
        
        return candidate
    
    def get_context_type(self, expansion: str) -> str:
        """Determine context type from expansion"""
        if not expansion:
            return 'general'
        
        expansion_lower = expansion.lower()
        
        # Check for organization indicators
        org_indicators = ['organization', 'institute', 'foundation', 'association',
                         'society', 'committee', 'agency', 'administration', 
                         'university', 'college', 'alliance']
        
        if any(indicator in expansion_lower for indicator in org_indicators):
            return 'organization'
        
        # Check for disease/medical
        if any(term in expansion_lower for term in ['disease', 'syndrome', 'disorder']):
            return 'disease'
        
        # Check for drug/pharmaceutical
        if any(term in expansion_lower for term in ['drug', 'medicine', 'therapeutic']):
            return 'drug'
        
        # Check for clinical/medical
        if any(term in expansion_lower for term in ['clinical', 'medical', 'patient']):
            return 'clinical'
        
        # Check for genetic
        if any(term in expansion_lower for term in ['gene', 'genetic', 'mutation']):
            return 'genetic'
        
        return 'general'
    
    def _generate_article_identifier_expansion(self, abbrev: str) -> Tuple[Optional[str], float]:
        """
        Generate expansion for article identifiers (PMC, NEJM, etc.)
        
        Args:
            abbrev: The abbreviation to check
            
        Returns:
            Tuple of (expansion, confidence) or (None, 0.0) if not an identifier
        """
        # Check each known prefix
        for prefix, expansion in ARTICLE_IDENTIFIER_PREFIXES.items():
            if abbrev.startswith(prefix):
                # Extract the numeric/alphanumeric suffix
                suffix = abbrev[len(prefix):]
                
                # Validate suffix format
                if suffix and (suffix.isdigit() or suffix.isalnum()):
                    # Generate the full expansion
                    if 'article' in expansion or 'ID' in expansion or 'identifier' in expansion:
                        # Already has descriptive term
                        full_expansion = f"{expansion} {suffix}"
                    else:
                        # Add 'article' for clarity
                        full_expansion = f"{expansion} article {suffix}"
                    
                    # High confidence for known patterns
                    confidence = 0.95
                    
                    logger.debug(f"Generated expansion for article identifier: {abbrev} -> {full_expansion}")
                    return full_expansion, confidence
        
        return None, 0.0
    
    def _is_non_expandable_term(self, term: str) -> bool:
        """
        Check if a term is standardized nomenclature that doesn't need expansion
        
        Args:
            term: The term to check
            
        Returns:
            True if the term is complete as-is
        """
        # CD markers (CD1, CD163, etc.)
        if re.match(r'^CD\d+[a-z]?$', term):
            return True
        
        # Complement components (C3, C5, C5a, C5b-9, etc.)
        if re.match(r'^C\d+[a-z]?(?:-\d+)?$', term):
            return True
        
        # Interleukins (IL-1, IL-6, etc.)
        if re.match(r'^IL-?\d+[a-z]?$', term):
            return True
        
        # HLA types (HLA-B27, HLA-DR4, etc.)
        if re.match(r'^HLA-[A-Z]+\*?\d+(?::\d+)?$', term):
            return True
        
        # Common research platforms (not abbreviations)
        non_expandable_platforms = {
            'ResearchGate', 'PubMed', 'MEDLINE', 'Scopus', 'StatPearls',
            'UpToDate', 'DynaMed', 'ClinicalKey', 'Lexicomp'
        }
        if term in non_expandable_platforms:
            return True
        
        return False
    
    def resolve_ambiguity(self, candidate: 'AbbreviationCandidate', 
                         context: str, alternatives: List[str]) -> str:
        """Resolve ambiguous abbreviations using context"""
        if not alternatives:
            return candidate.expansion
        
        # Score each alternative
        scores = {}
        domain = self.domain_manager.get_domain_context(context)
        
        for alt in alternatives:
            score = self.scoring_engine.score_expansion(
                candidate.abbreviation, alt, context, domain
            )
            scores[alt] = score
        
        # Return the highest scoring alternative
        best_expansion = max(scores, key=scores.get)
        best_score = scores[best_expansion]
        
        # Update candidate confidence based on disambiguation confidence
        if best_score > 0.8:
            candidate.confidence = min(1.0, candidate.confidence * 1.1)
        elif best_score < 0.5:
            candidate.confidence *= 0.9
            candidate.disambiguation_needed = True
        
        return best_expansion


# Export the main class
__all__ = ['ContextAnalyzer', 'DomainContextManager', 
           'ContextScoringEngine', 'TokenNormalizer',
           'ARTICLE_IDENTIFIER_PREFIXES', 'COMMON_WORDS']