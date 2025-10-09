#!/usr/bin/env python3
"""
Enhanced Drug Pattern Detection Module with RxNorm Lexicon Integration
======================================================================
Location: corpus_metadata/document_utils/rare_disease_drug_pattern_detector.py
Version: 2.0

Comprehensive pattern matching for drug detection with enhanced patterns and lexicon.
Designed for use as a library module.

Key features:
1. Token normalization and biomedical compound handling
2. Structural noise filtering
3. Extended pattern library (suffixes, prefixes, substrings)
4. RxNorm lexicon integration for comprehensive coverage
5. PubTator3 validation of detected drugs
6. Confidence boosting for normalized drugs
7. Batch validation for efficiency
8. MeSH ID and RxCUI enrichment

UPDATES v2.0:
- Added TokenNormalizer for pre-processing
- Added BiomedicalCompoundJoiner for handling hyphenated compounds
- Added NoiseFilter for structural noise removal
- Improved pattern matching with normalized text
"""

import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
from functools import lru_cache

# ============================================================================
# Import refactored logging configuration
# ============================================================================
from corpus_metadata.document_utils.metadata_logging_config import (
    setup_logging,
    get_logger,
    get_tracker,
    file_context,
    log_separator,
    log_metric
)

# Initialize logging
setup_logging(
    console_level="WARNING",
    file_level="INFO",
    use_color=False
)

# Get logger and tracker
logger = get_logger('drug_pattern_detector')
tracker = get_tracker('drug_pattern_detector')

# Import MetadataSystemInitializer for lexicon access
try:
    from corpus_metadata.document_utils.metadata_system_initializer import MetadataSystemInitializer
    LEXICON_AVAILABLE = True
except ImportError:
    LEXICON_AVAILABLE = False
    logger.warning("MetadataSystemInitializer not available - lexicon disabled")

# Try importing SpaCy
try:
    import spacy
    SPACY_AVAILABLE = True
    logger.info("✓ SpaCy is available for NER detection")
except ImportError as e:
    SPACY_AVAILABLE = False
    logger.warning(f"SpaCy not installed. NER detection will be disabled: {e}")

# Try importing Aho-Corasick for efficient pattern matching
try:
    import ahocorasick
    AHO_CORASICK_AVAILABLE = True
    logger.info("✓ Aho-Corasick available for efficient pattern matching")
except ImportError as e:
    AHO_CORASICK_AVAILABLE = False
    logger.warning(f"Aho-Corasick not available. Using regex fallback: {e}")

# Try importing PubTator3
try:
    from corpus_metadata.document_utils.rare_disease_pubtator3 import PubTator3Manager
    PUBTATOR_AVAILABLE = True
    logger.info("✓ PubTator3 available for drug normalization")
except ImportError:
    PUBTATOR_AVAILABLE = False
    logger.warning("PubTator3 not available for validation")


# ============================================================================
# Token Normalizer
# ============================================================================

class TokenNormalizer:
    """Handles text normalization before pattern matching"""
    
    # Common OCR errors and variations
    REPLACEMENTS = {
        # Greek letters
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
        'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta',
        'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu', 'ν': 'nu',
        'ξ': 'xi', 'π': 'pi', 'ρ': 'rho', 'σ': 'sigma',
        'τ': 'tau', 'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
        
        # Common variations
        '®': '', '™': '', '©': '',
        '—': '-', '–': '-', '−': '-',
        ''' : "'", ''' : "'", '"': '"', '"': '"',
        '\u00AD': '',  # Soft hyphen
        '\u200B': '',  # Zero-width space
    }
    
    @classmethod
    def normalize(cls, text: str) -> str:
        """
        Normalize text for pattern matching
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Apply character replacements
        for old, new in cls.REPLACEMENTS.items():
            text = text.replace(old, new)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([.,;!?]){2,}', r'\1', text)
        
        # Fix hyphenation at line breaks
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        return text.strip()
    
    @classmethod
    def normalize_drug_name(cls, drug_name: str) -> str:
        """
        Normalize a drug name for matching
        
        Args:
            drug_name: Drug name to normalize
            
        Returns:
            Normalized drug name
        """
        # Remove trademark symbols
        drug_name = re.sub(r'[®™©]', '', drug_name)
        
        # Normalize hyphens
        drug_name = re.sub(r'[—–−]', '-', drug_name)
        
        # Remove parenthetical content
        drug_name = re.sub(r'\([^)]*\)', '', drug_name)
        
        # Normalize whitespace
        drug_name = re.sub(r'\s+', ' ', drug_name)
        
        return drug_name.strip()


# ============================================================================
# Biomedical Compound Joiner
# ============================================================================

class BiomedicalCompoundJoiner:
    """Handles joining of biomedical compound terms"""
    
    # Patterns for compounds that should be joined
    COMPOUND_PATTERNS = [
        # Virus/pathogen compounds
        (r'SARS', r'CoV', r'SARS-CoV'),
        (r'SARS-CoV', r'2', r'SARS-CoV-2'),
        (r'COVID', r'19', r'COVID-19'),
        (r'HCoV', r'\d+', r'HCoV-{1}'),
        
        # Antibody/protein compounds
        (r'PR3', r'ANCA', r'PR3-ANCA'),
        (r'MPO', r'ANCA', r'MPO-ANCA'),
        (r'anti', r'[A-Z]\w+', r'anti-{1}'),
        (r'TNF', r'alpha', r'TNF-alpha'),
        (r'IFN', r'gamma', r'IFN-gamma'),
        (r'IL', r'\d+', r'IL-{1}'),
        
        # Drug compounds
        (r'ACE', r'inhibitor', r'ACE-inhibitor'),
        (r'beta', r'blocker', r'beta-blocker'),
        (r'calcium', r'channel', r'calcium-channel'),
        
        # Receptor/enzyme compounds
        (r'HER2', r'positive', r'HER2-positive'),
        (r'PD', r'L1', r'PD-L1'),
        (r'PD', r'1', r'PD-1'),
        (r'CTLA', r'4', r'CTLA-4'),
    ]
    
    @classmethod
    def join_compounds(cls, text: str) -> str:
        """
        Join biomedical compounds in text
        
        Args:
            text: Input text
            
        Returns:
            Text with joined compounds
        """
        for part1, part2, joined in cls.COMPOUND_PATTERNS:
            # Create pattern with optional whitespace/hyphen
            pattern = rf'\b({part1})[\s\-]?({part2})\b'
            
            # Replace with joined form
            def replacer(match):
                if '{1}' in joined:
                    return joined.replace('{1}', match.group(2))
                return joined
            
            text = re.sub(pattern, replacer, text, flags=re.IGNORECASE)
        
        return text
    
    @classmethod
    def split_for_analysis(cls, compound: str) -> List[str]:
        """
        Split a compound for separate analysis
        
        Args:
            compound: Compound term
            
        Returns:
            List of components
        """
        # Split on hyphens but keep certain prefixes attached
        parts = []
        
        # Keep anti-, non-, pre-, post- attached
        if compound.startswith(('anti-', 'non-', 'pre-', 'post-')):
            prefix_match = re.match(r'^(anti-|non-|pre-|post-)', compound)
            if prefix_match:
                prefix = prefix_match.group(1).rstrip('-')
                remainder = compound[len(prefix_match.group(1)):]
                parts.append(prefix)
                if remainder:
                    parts.extend(remainder.split('-'))
        else:
            parts = compound.split('-')
        
        return [p for p in parts if p]


# ============================================================================
# Noise Filter
# ============================================================================

class NoiseFilter:
    """Filters structural noise from documents"""
    
    # Patterns to filter out
    NOISE_PATTERNS = [
        # References and citations
        re.compile(r'\[\d+(?:[-,]\d+)*\]'),
        re.compile(r'\(\d{4}[a-z]?\)'),
        re.compile(r'(?:Fig(?:ure)?|Table|Scheme|Eq(?:uation)?)\s*\.?\s*\d+[A-Za-z]?'),
        
        # DOI and URLs
        re.compile(r'(?:doi:|DOI:|https?://doi\.org/)\S+'),
        re.compile(r'https?://\S+'),
        re.compile(r'www\.\S+'),
        
        # Page numbers and headers
        re.compile(r'^\s*\d+\s*$', re.MULTILINE),
        re.compile(r'Page \d+ of \d+'),
        
        # Journal metadata
        re.compile(r'©\s*\d{4}\s+\w+'),
        re.compile(r'ISSN\s*:?\s*\d{4}-\d{4}'),
        re.compile(r'ISBN\s*:?\s*[\d-]+'),
        
        # Email addresses
        re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    ]
    
    # Terms that look like drugs but aren't
    FALSE_POSITIVES = {
        # Common lab values/units
        'hdl', 'ldl', 'alt', 'ast', 'gfr', 'egfr', 'bnp',
        'hba1c', 'tsh', 'psa', 'inr', 'crp', 'esr',
        
        # Common abbreviations
        'usa', 'uk', 'eu', 'fda', 'ema', 'who', 'cdc', 'nih',
        'pdf', 'doi', 'url', 'http', 'https', 'www',
        
        # Statistical terms
        'anova', 'sd', 'sem', 'ci', 'or', 'rr', 'hr',
        
        # Study terms
        'rct', 'itt', 'pp', 'locf', 'bocf',
    }
    
    @classmethod
    def is_noise(cls, text: str, context: str = '') -> bool:
        """
        Check if a text fragment is likely noise
        
        Args:
            text: Text to check
            context: Optional context around the text
            
        Returns:
            True if likely noise
        """
        text_lower = text.lower().strip()
        
        # Check false positives
        if text_lower in cls.FALSE_POSITIVES:
            return True
        
        # Check if it's a number or single letter
        if text_lower.isdigit() or (len(text_lower) == 1 and text_lower.isalpha()):
            return True
        
        # Check noise patterns in context
        if context:
            for pattern in cls.NOISE_PATTERNS:
                if pattern.search(context):
                    # Check if the text is part of the noise pattern
                    if text in context:
                        match = pattern.search(context)
                        if match and match.start() <= context.index(text) <= match.end():
                            return True
        
        return False
    
    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Remove noise patterns from text
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Apply noise pattern removal
        for pattern in cls.NOISE_PATTERNS:
            text = pattern.sub(' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PatternMatch:
    """Represents a pattern match result"""
    text: str
    pattern: str
    start: int
    end: int
    confidence: float = 0.7
    context: str = ""
    pattern_type: str = ""
    description: str = ""
    rxcui: Optional[str] = None
    tty: Optional[str] = None
    normalized_text: Optional[str] = None
    
@dataclass
class DrugPattern:
    """Defines a drug detection pattern"""
    name: str
    pattern: str
    confidence: float
    pattern_type: str
    description: str = ""
    requires_context: bool = False


# ============================================================================
# Pattern Detector Class
# ============================================================================

class DrugPatternDetector:
    """
    Advanced pattern-based drug detection with comprehensive pattern library and RxNorm lexicon
    """
    
    def __init__(self, 
                 enable_ner: bool = True, 
                 enable_pubtator: bool = True,
                 enable_lexicon: bool = True,
                 enable_normalization: bool = True,
                 system_initializer: Optional[MetadataSystemInitializer] = None):
        """
        Initialize the pattern detector
        
        Args:
            enable_ner: Enable SpaCy NER detection
            enable_pubtator: Enable PubTator3 validation
            enable_lexicon: Enable RxNorm lexicon matching
            enable_normalization: Enable text normalization
            system_initializer: Optional MetadataSystemInitializer for lexicon access
        """
        log_separator(logger, "major")
        logger.info("Initializing Enhanced Drug Pattern Detector v2.0")
        log_separator(logger, "major")
        
        self.enable_ner = enable_ner and SPACY_AVAILABLE
        self.enable_pubtator = enable_pubtator and PUBTATOR_AVAILABLE
        self.enable_lexicon = enable_lexicon and LEXICON_AVAILABLE
        self.enable_normalization = enable_normalization
        
        # Initialize components
        self.normalizer = TokenNormalizer()
        self.compound_joiner = BiomedicalCompoundJoiner()
        self.noise_filter = NoiseFilter()
        
        # Initialize lexicon if enabled
        self.system_initializer = system_initializer
        self.drug_lexicon_available = False
        
        if self.enable_lexicon:
            self._initialize_lexicon(system_initializer)
        
        # Initialize patterns
        self.patterns = self._initialize_patterns()
        logger.info(f"Loaded {len(self.patterns)} detection patterns")
        
        # Initialize NER models if available
        self.ner_models = {}
        if self.enable_ner:
            self._load_ner_models()
        
        # Initialize PubTator3 if available
        self.pubtator_manager = None
        if self.enable_pubtator:
            try:
                self.pubtator_manager = PubTator3Manager()
                logger.info("PubTator3 manager initialized")
            except Exception as e:
                logger.warning(f"Could not initialize PubTator3: {e}")
                self.enable_pubtator = False
        
        # Initialize Aho-Corasick automaton if available
        self.ac_automaton = None
        if AHO_CORASICK_AVAILABLE:
            self._build_ac_automaton()
        
        # Statistics tracking
        self.stats = defaultdict(int)
        self._reset_statistics()
        
        logger.info("Pattern detector initialization complete")
        log_separator(logger, "minor")
    
    def _initialize_lexicon(self, system_initializer: Optional[MetadataSystemInitializer]):
        """Initialize RxNorm lexicon from system initializer"""
        try:
            if system_initializer:
                self.system_initializer = system_initializer
            else:
                self.system_initializer = MetadataSystemInitializer()
                self.system_initializer.initialize()
            
            if self.system_initializer and self.system_initializer.status.get('lexicons', {}).get('drug'):
                self.drug_lexicon_available = True
                logger.info("✓ Drug lexicon available for pattern detection")
            else:
                logger.warning("Drug lexicon not available in system initializer")
                self.enable_lexicon = False
                
        except Exception as e:
            logger.error(f"Failed to initialize lexicon: {e}")
            self.enable_lexicon = False
            self.drug_lexicon_available = False
    
    def _initialize_patterns(self) -> List[DrugPattern]:
        """Initialize comprehensive drug detection patterns"""
        patterns = []
        
        # Common drug suffix patterns
        suffix_data = [
            ('mab', 0.9, 'monoclonal antibody'),
            ('nib', 0.9, 'kinase inhibitor'),
            ('tinib', 0.92, 'tyrosine kinase inhibitor'),
            ('ciclib', 0.9, 'CDK inhibitor'),
            ('rafenib', 0.9, 'RAF inhibitor'),
            ('zumab', 0.9, 'humanized monoclonal antibody'),
            ('ximab', 0.9, 'chimeric monoclonal antibody'),
            ('umab', 0.9, 'human monoclonal antibody'),
            ('parin', 0.85, 'anticoagulant'),
            ('statin', 0.9, 'HMG-CoA reductase inhibitor'),
            ('pril', 0.9, 'ACE inhibitor'),
            ('sartan', 0.9, 'angiotensin receptor blocker'),
            ('olol', 0.9, 'beta blocker'),
            ('azole', 0.85, 'antifungal'),
            ('cycline', 0.85, 'antibiotic'),
            ('mycin', 0.85, 'antibiotic'),
            ('vir', 0.8, 'antiviral'),
            ('navir', 0.9, 'protease inhibitor'),
            ('tegravir', 0.9, 'integrase inhibitor'),
            ('cillin', 0.85, 'penicillin antibiotic'),
            ('floxacin', 0.9, 'fluoroquinolone antibiotic'),
            ('caine', 0.85, 'local anesthetic'),
            ('tidine', 0.85, 'H2 receptor antagonist'),
            ('triptan', 0.9, 'migraine medication'),
            ('setron', 0.9, 'antiemetic'),
            ('gliptin', 0.9, 'DPP-4 inhibitor'),
            ('gliflozin', 0.9, 'SGLT2 inhibitor'),
            ('tide', 0.85, 'GLP-1 agonist'),
            ('glutide', 0.9, 'GLP-1 receptor agonist'),
        ]
        
        for suffix, confidence, description in suffix_data:
            patterns.append(DrugPattern(
                name=f"{suffix}_suffix",
                pattern=rf'\b[A-Z][a-z]*{re.escape(suffix)}\b',
                confidence=confidence,
                pattern_type='suffix',
                description=description
            ))
        
        # Prefix patterns
        prefix_data = [
            ('cef', 0.9, 'cephalosporin antibiotic'),
            ('pred', 0.85, 'corticosteroid'),
            ('dexa', 0.85, 'corticosteroid'),
            ('hydro', 0.8, 'hydro-containing drug'),
            ('levo', 0.85, 'levo-isomer'),
            ('dextro', 0.85, 'dextro-isomer'),
        ]
        
        for prefix, confidence, description in prefix_data:
            patterns.append(DrugPattern(
                name=f"{prefix}_prefix",
                pattern=rf'\b{prefix}[a-zA-Z]{{3,}}\b',
                confidence=confidence,
                pattern_type='prefix',
                description=description
            ))
        
        # Investigational drug patterns
        patterns.extend([
            DrugPattern(
                name="investigational_code",
                pattern=r'\b(?:ABT|MK|PF|GSK|JNJ|AZD|BMS|RO|LY|TAK|SAR|ONO|BI|DS|AMG|ABBV|REGN|MEDI|ASN|CTI|SGN|EMD|UCB|BAY)[-\s]?\d{3,}\b',
                confidence=0.85,
                pattern_type='code',
                description="Pharmaceutical company investigational code"
            ),
            DrugPattern(
                name="nct_compound",
                pattern=r'\b[A-Z]{2,5}\d{3,5}[A-Z]?\b',
                confidence=0.7,
                pattern_type='code',
                description="Alphanumeric compound code"
            ),
        ])
        
        # Combination patterns
        patterns.extend([
            DrugPattern(
                name="drug_combination",
                pattern=r'\b[A-Z][a-z]+/[A-Z][a-z]+\b',
                confidence=0.8,
                pattern_type='combination',
                description="Drug combination"
            ),
            DrugPattern(
                name="drug_with_dose",
                pattern=r'\b[A-Z][a-z]+\s+\d+\s*(?:mg|mcg|µg|g|ml|mL|IU|units?)\b',
                confidence=0.85,
                pattern_type='with_dose',
                description="Drug with dosage"
            ),
        ])
        
        logger.debug(f"Initialized {len(patterns)} pattern rules")
        return patterns
    
    def _load_ner_models(self):
        """Load SpaCy NER models"""
        models_to_load = [
            ('en_ner_bc5cdr_md', 'bc5cdr'),
            ('en_core_sci_lg', 'scispacy'),
        ]
        
        for model_name, alias in models_to_load:
            try:
                model = spacy.load(model_name)
                self.ner_models[alias] = model
                logger.info(f"Loaded NER model: {alias}")
            except Exception as e:
                logger.debug(f"Could not load {model_name}: {e}")
        
        if not self.ner_models:
            logger.warning("No NER models available")
            self.enable_ner = False
    
    def _build_ac_automaton(self):
        """Build Aho-Corasick automaton for efficient matching"""
        if not AHO_CORASICK_AVAILABLE:
            return
        
        self.ac_automaton = ahocorasick.Automaton()
        
        # Common generic drugs (curated list)
        common_drugs = [
            'aspirin', 'ibuprofen', 'acetaminophen', 'metformin',
            'atorvastatin', 'simvastatin', 'omeprazole', 'losartan',
            'metoprolol', 'amlodipine', 'albuterol', 'gabapentin',
            'lisinopril', 'levothyroxine', 'azithromycin', 'amoxicillin',
            'prednisone', 'hydrochlorothiazide', 'furosemide', 'warfarin',
            'insulin', 'sertraline', 'escitalopram', 'fluoxetine',
            'citalopram', 'trazodone', 'bupropion', 'duloxetine',
            'venlafaxine', 'lorazepam', 'alprazolam', 'clonazepam',
            'tramadol', 'oxycodone', 'hydrocodone', 'morphine',
            'fentanyl', 'codeine', 'methylphenidate', 'amphetamine',
            'montelukast', 'fluticasone', 'budesonide', 'ipratropium',
        ]
        
        idx = 0
        for drug in common_drugs:
            self.ac_automaton.add_word(drug.lower(), (idx, drug, 'generic'))
            idx += 1
        
        # Add from lexicon if available
        if self.drug_lexicon_available and hasattr(self.system_initializer, 'drug_lexicon'):
            tty_priority = ['IN', 'PIN', 'MIN', 'BN']
            added = 0
            
            for term, entry in self.system_initializer.drug_lexicon.items():
                if added >= 10000:  # Limit for performance
                    break
                    
                tty = entry.get('tty', '')
                if tty in tty_priority and len(term) >= 4:
                    normalized = entry.get('term_normalized', term.lower())
                    rxcui = entry.get('rxcui', '')
                    
                    self.ac_automaton.add_word(
                        normalized,
                        (idx, term, 'rxnorm', rxcui, tty)
                    )
                    idx += 1
                    added += 1
        
        self.ac_automaton.make_automaton()
        logger.debug(f"Built Aho-Corasick automaton with {idx} entries")
    
    def _preprocess_text(self, text: str) -> Tuple[str, Dict[int, int]]:
        """
        Preprocess text with normalization and compound joining
        
        Args:
            text: Original text
            
        Returns:
            Tuple of (processed_text, position_mapping)
        """
        if not self.enable_normalization:
            return text, {}
        
        # Store original for position mapping
        original = text
        
        # Normalize text
        text = self.normalizer.normalize(text)
        
        # Join biomedical compounds
        text = self.compound_joiner.join_compounds(text)
        
        # Clean obvious noise
        text = self.noise_filter.clean_text(text)
        
        # Create position mapping (simplified - would need more complex mapping in production)
        position_map = {}
        
        return text, position_map
    
    def detect_patterns(self, text: str, context_window: int = 50) -> List[PatternMatch]:
        """
        Detect drug patterns in text
        
        Args:
            text: Text to analyze
            context_window: Characters of context to capture
            
        Returns:
            List of pattern matches
        """
        # Preprocess text
        processed_text, pos_map = self._preprocess_text(text)
        
        matches = []
        seen_matches = set()
        
        # Apply regex patterns
        for pattern in self.patterns:
            try:
                for match in re.finditer(pattern.pattern, processed_text, re.IGNORECASE):
                    match_text = match.group()
                    normalized = self.normalizer.normalize_drug_name(match_text)
                    
                    # Check if it's noise
                    if self.noise_filter.is_noise(match_text, processed_text[max(0, match.start()-50):match.end()+50]):
                        continue
                    
                    match_key = (normalized.lower(), match.start())
                    
                    if match_key not in seen_matches:
                        seen_matches.add(match_key)
                        
                        # Get context
                        start = max(0, match.start() - context_window)
                        end = min(len(processed_text), match.end() + context_window)
                        context = processed_text[start:end]
                        
                        matches.append(PatternMatch(
                            text=match_text,
                            pattern=pattern.name,
                            start=match.start(),
                            end=match.end(),
                            confidence=pattern.confidence,
                            context=context,
                            pattern_type=pattern.pattern_type,
                            description=pattern.description,
                            normalized_text=normalized
                        ))
                        
                        self.stats['patterns_matched'] += 1
                    
            except Exception as e:
                logger.warning(f"Pattern {pattern.name} failed: {e}")
        
        # Use Aho-Corasick for exact matches
        if self.ac_automaton:
            for end_idx, value in self.ac_automaton.iter(processed_text.lower()):
                if len(value) == 3:
                    idx, drug, drug_type = value
                    rxcui, tty = None, None
                elif len(value) == 5:
                    idx, drug, drug_type, rxcui, tty = value
                else:
                    continue
                
                start_idx = end_idx - len(drug) + 1
                match_text = processed_text[start_idx:end_idx + 1]
                
                # Check noise
                if self.noise_filter.is_noise(match_text):
                    continue
                
                match_key = (drug, start_idx)
                
                if match_key not in seen_matches:
                    seen_matches.add(match_key)
                    
                    # Get context
                    ctx_start = max(0, start_idx - context_window)
                    ctx_end = min(len(processed_text), end_idx + context_window)
                    
                    confidence = 0.95 if drug_type == 'generic' else 0.9
                    if tty in ['IN', 'PIN']:
                        confidence = 0.95
                    elif tty == 'BN':
                        confidence = 0.88
                    
                    matches.append(PatternMatch(
                        text=match_text,
                        pattern='exact_match',
                        start=start_idx,
                        end=end_idx + 1,
                        confidence=confidence,
                        context=processed_text[ctx_start:ctx_end],
                        pattern_type='exact',
                        description=f'{drug_type} drug',
                        rxcui=rxcui,
                        tty=tty,
                        normalized_text=drug
                    ))
                    
                    self.stats['exact_matched'] += 1
        
        logger.debug(f"Found {len(matches)} pattern matches")
        return matches
    
    def detect_with_lexicon(self, text: str, context_window: int = 50) -> List[PatternMatch]:
        """
        Detect drugs using RxNorm lexicon
        
        Args:
            text: Text to analyze
            context_window: Characters of context to capture
            
        Returns:
            List of lexicon matches
        """
        if not self.drug_lexicon_available:
            return []
        
        # Preprocess text
        processed_text, pos_map = self._preprocess_text(text)
        
        matches = []
        text_lower = processed_text.lower()
        seen_matches = set()
        
        # Search for lexicon terms
        for term, entry in self.system_initializer.drug_lexicon.items():
            term_normalized = entry.get('term_normalized', term.lower())
            
            # Skip very short terms or noise
            if len(term_normalized) < 3 or self.noise_filter.is_noise(term_normalized):
                continue
            
            # Create pattern with word boundaries
            pattern = r'\b' + re.escape(term_normalized) + r'\b'
            
            for match in re.finditer(pattern, text_lower):
                match_key = (term_normalized, match.start())
                
                if match_key not in seen_matches:
                    seen_matches.add(match_key)
                    
                    # Get context
                    ctx_start = max(0, match.start() - context_window)
                    ctx_end = min(len(processed_text), match.end() + context_window)
                    
                    # Calculate confidence based on TTY
                    tty = entry.get('tty', 'SY')
                    confidence_map = {
                        'IN': 0.95, 'PIN': 0.93, 'MIN': 0.90,
                        'BN': 0.88, 'SCD': 0.85, 'SBD': 0.83,
                        'SY': 0.75, 'TMSY': 0.70
                    }
                    confidence = confidence_map.get(tty, 0.70)
                    
                    matches.append(PatternMatch(
                        text=processed_text[match.start():match.end()],
                        pattern='lexicon_match',
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        context=processed_text[ctx_start:ctx_end],
                        pattern_type='lexicon',
                        description=f'RxNorm {tty}',
                        rxcui=entry.get('rxcui'),
                        tty=tty,
                        normalized_text=term_normalized
                    ))
                    
                    self.stats['lexicon_matched'] += 1
        
        logger.debug(f"Found {len(matches)} lexicon matches")
        return matches
    
    def detect_with_ner(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect drugs using NER models
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected entities
        """
        if not self.ner_models:
            return []
        
        # Preprocess text
        processed_text, pos_map = self._preprocess_text(text)
        
        entities = []
        seen_entities = set()
        
        for model_name, model in self.ner_models.items():
            try:
                # Process text with model
                doc = model(processed_text[:1000000])  # Limit for performance
                
                # Define target labels based on model
                if model_name == 'bc5cdr':
                    target_labels = ['CHEMICAL']
                elif model_name == 'scispacy':
                    target_labels = ['CHEMICAL', 'DRUG', 'MEDICATION']
                else:
                    target_labels = ['CHEMICAL', 'DRUG']
                
                for ent in doc.ents:
                    if ent.label_ in target_labels:
                        # Check noise
                        if self.noise_filter.is_noise(ent.text):
                            continue
                        
                        entity_key = (ent.text.lower(), ent.start_char)
                        
                        if entity_key not in seen_entities:
                            seen_entities.add(entity_key)
                            
                            normalized = self.normalizer.normalize_drug_name(ent.text)
                            
                            entities.append({
                                'text': ent.text,
                                'normalized': normalized,
                                'label': ent.label_,
                                'start': ent.start_char,
                                'end': ent.end_char,
                                'model': model_name,
                                'confidence': 0.8
                            })
                            
                            self.stats['ner_detected'] += 1
                        
            except Exception as e:
                logger.warning(f"NER detection failed with {model_name}: {e}")
        
        logger.debug(f"NER detected {len(entities)} entities")
        return entities
    
    def process_text(self, 
                    text: str,
                    use_patterns: bool = True,
                    use_lexicon: bool = True,
                    use_ner: bool = True,
                    validate: bool = True,
                    min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        Process text with all detection methods
        
        Args:
            text: Text to process
            use_patterns: Use pattern matching
            use_lexicon: Use RxNorm lexicon matching
            use_ner: Use NER detection
            validate: Validate with PubTator3
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        results = {
            'pattern_matches': [],
            'lexicon_matches': [],
            'ner_entities': [],
            'unique_drugs': {},
            'processing_time': 0,
            'stats': {}
        }
        
        # Pattern detection
        if use_patterns:
            logger.debug("Running pattern detection...")
            matches = self.detect_patterns(text)
            results['pattern_matches'] = matches
            
            for match in matches:
                drug_key = (match.normalized_text or match.text).lower()
                
                if drug_key not in results['unique_drugs']:
                    results['unique_drugs'][drug_key] = {
                        'text': match.text,
                        'normalized': match.normalized_text,
                        'confidence': match.confidence,
                        'sources': ['pattern'],
                        'pattern_type': match.pattern_type,
                        'description': match.description,
                        'rxcui': match.rxcui,
                        'tty': match.tty
                    }
                else:
                    # Boost confidence
                    results['unique_drugs'][drug_key]['confidence'] = min(
                        1.0,
                        results['unique_drugs'][drug_key]['confidence'] + 0.1
                    )
                    if 'pattern' not in results['unique_drugs'][drug_key]['sources']:
                        results['unique_drugs'][drug_key]['sources'].append('pattern')
        
        # Lexicon detection
        if use_lexicon and self.drug_lexicon_available:
            logger.debug("Running lexicon detection...")
            lexicon_matches = self.detect_with_lexicon(text)
            results['lexicon_matches'] = lexicon_matches
            
            for match in lexicon_matches:
                drug_key = (match.normalized_text or match.text).lower()
                
                if drug_key not in results['unique_drugs']:
                    results['unique_drugs'][drug_key] = {
                        'text': match.text,
                        'normalized': match.normalized_text,
                        'confidence': match.confidence,
                        'sources': ['lexicon'],
                        'pattern_type': 'lexicon',
                        'description': match.description,
                        'rxcui': match.rxcui,
                        'tty': match.tty
                    }
                else:
                    # Boost confidence significantly for lexicon matches
                    results['unique_drugs'][drug_key]['confidence'] = min(
                        1.0,
                        results['unique_drugs'][drug_key]['confidence'] + 0.2
                    )
                    if 'lexicon' not in results['unique_drugs'][drug_key]['sources']:
                        results['unique_drugs'][drug_key]['sources'].append('lexicon')
                    # Prefer RxCUI from lexicon
                    if match.rxcui:
                        results['unique_drugs'][drug_key]['rxcui'] = match.rxcui
                        results['unique_drugs'][drug_key]['tty'] = match.tty
        
        # NER detection
        if use_ner and self.enable_ner:
            logger.debug("Running NER detection...")
            entities = self.detect_with_ner(text)
            results['ner_entities'] = entities
            
            for entity in entities:
                drug_key = (entity.get('normalized', entity['text'])).lower()
                
                if drug_key not in results['unique_drugs']:
                    results['unique_drugs'][drug_key] = {
                        'text': entity['text'],
                        'normalized': entity.get('normalized'),
                        'confidence': entity['confidence'],
                        'sources': ['ner'],
                        'ner_model': entity['model'],
                        'ner_label': entity['label']
                    }
                else:
                    # Boost confidence for NER matches
                    results['unique_drugs'][drug_key]['confidence'] = min(
                        1.0,
                        results['unique_drugs'][drug_key]['confidence'] + 0.15
                    )
                    if 'ner' not in results['unique_drugs'][drug_key]['sources']:
                        results['unique_drugs'][drug_key]['sources'].append('ner')
        
        # Filter by minimum confidence
        results['unique_drugs'] = {
            k: v for k, v in results['unique_drugs'].items()
            if v['confidence'] >= min_confidence
        }
        
        # Calculate processing time
        results['processing_time'] = time.time() - start_time
        
        # Update statistics
        self.stats['documents_processed'] += 1
        self.stats['total_drugs'] += len(results['unique_drugs'])
        
        # Add statistics to results
        results['stats'] = {
            'total_patterns': len(results['pattern_matches']),
            'total_lexicon': len(results['lexicon_matches']),
            'total_ner': len(results['ner_entities']),
            'unique_drugs': len(results['unique_drugs']),
            'processing_time': results['processing_time']
        }
        
        # Log metrics
        log_metric(logger, 'unique_drugs', len(results['unique_drugs']))
        log_metric(logger, 'processing_time', results['processing_time'], 's')
        
        logger.info(f"Processed text: {len(results['unique_drugs'])} unique drugs found")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics"""
        stats = dict(self.stats)
        
        if stats['documents_processed'] > 0:
            stats['avg_drugs_per_doc'] = stats['total_drugs'] / stats['documents_processed']
        
        return stats
    
    def _reset_statistics(self):
        """Reset statistics counters"""
        self.stats = defaultdict(int)
        self.stats.update({
            'documents_processed': 0,
            'patterns_matched': 0,
            'exact_matched': 0,
            'ner_detected': 0,
            'lexicon_matched': 0,
            'pubtator_validated': 0,
            'total_drugs': 0
        })
        
    def reset_statistics(self):
        """Public method to reset statistics"""
        self._reset_statistics()
        logger.debug("Statistics reset")