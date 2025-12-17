#!/usr/bin/env python3
"""
Enhanced AbbreviationExtractor with Modular Architecture
=========================================================
Location: corpus_metadata/document_metadata_extraction_abbreviation.py
Version: 5.0.0

CHANGES IN VERSION 5.0:
- Integrated shared types from abbreviation_types.py
- Added lexicon resolution before Claude
- Proper module imports with error handling
- Streamlined pipeline with lexicon → context → Claude flow
- Removed duplicate AbbreviationCandidate definition
"""

import time, re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import asdict

# ============================================================================
# CENTRALIZED LOGGING
# ============================================================================
try:
    from corpus_metadata.document_utils.metadata_logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('abbreviation_extractor')

# ============================================================================
# IMPORTS FROM UTILITY MODULES
# ============================================================================

# Import shared types
try:
    from corpus_metadata.document_utils.abbreviation_types import (
        AbbreviationCandidate,
        ValidationStatus,
        SourceType,
        ContextType,
        merge_candidates,
        filter_by_confidence
    )
except ImportError as e:
    logger.error(f"Could not import shared types: {e}")
    raise

# Import pattern detector
try:
    from corpus_metadata.document_utils.abbreviation_patterns import PatternDetector
except ImportError as e:
    logger.error(f"Could not import pattern detector: {e}")
    raise

# Import context analyzer
try:
    from corpus_metadata.document_utils.abbreviation_context import ContextAnalyzer
except ImportError as e:
    logger.error(f"Could not import context analyzer: {e}")
    raise

# Import lexicon resolver
try:
    from corpus_metadata.document_utils.abbreviation_lexicon import (
        LexiconResolver,
        integrate_lexicon_resolution
    )
    LEXICON_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Lexicon resolver not available: {e}")
    LEXICON_AVAILABLE = False
    LexiconResolver = None

# Import Claude enhancer
try:
    from corpus_metadata.document_utils.abbreviation_claude import ClaudeEnhancer
    CLAUDE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Claude enhancer not available: {e}")
    CLAUDE_AVAILABLE = False
    ClaudeEnhancer = None

# ============================================================================
# RESULT DATA STRUCTURE
# ============================================================================

from dataclasses import dataclass, field

@dataclass
class AbbreviationResult:
    """Final validated abbreviation result"""
    abbreviation: str
    expansion: str
    occurrences: int
    confidence: float
    source: str
    dictionary_sources: List[str]
    first_position: int
    context_type: str
    validated: bool
    validation_method: str
    alternative_meanings: List[str] = field(default_factory=list)
    detection_source: str = ''
    page_number: int = -1
    local_expansion: Optional[str] = None
    domain_context: str = ''
    claude_resolved: bool = False
    lexicon_resolved: bool = False
    cui: Optional[str] = None
    semantic_types: List[str] = field(default_factory=list)

# ============================================================================
# MAIN ABBREVIATION EXTRACTOR CLASS
# ============================================================================

class AbbreviationExtractor:
    def __init__(self, system_initializer=None, use_claude: bool = True, 
                 use_lexicon: bool = True, lexicon_path: Optional[Path] = None,
                 config: Dict = None):
        """
        Initialize the abbreviation extractor.
        
        Args:
            system_initializer: System initializer for resources
            use_claude: Whether to use Claude for disambiguation
            use_lexicon: Whether to use lexicon resolution
            lexicon_path: Path to lexicon file
            config: Configuration dictionary
        """
        self.logger = logger
        self.system_initializer = system_initializer
        
        # FIX: Get config from system_initializer if not provided
        if config is None and system_initializer:
            self.config = getattr(system_initializer, 'config', {})
        else:
            self.config = config or {}
        
        # Now self.config should have the full configuration
        
        # Feature flags
        self.use_claude = use_claude and CLAUDE_AVAILABLE
        self.use_lexicon = use_lexicon and LEXICON_AVAILABLE
        
        # Load configuration
        self._load_config()
        
        # Initialize components
        self._initialize_components(lexicon_path)
        
        # Load resources
        self._load_resources()
        
        # Statistics tracking
        self.stats = defaultdict(int)
    
    def _load_config(self):
        """Load configuration from config dict"""
        abbrev_config = self.config.get('abbreviation_extraction', {})
        
        # Validation config
        validation = abbrev_config.get('validation', {})
        self.min_confidence = validation.get('min_confidence', 0.5)
        
        # Context config
        self.context_window_size = abbrev_config.get('context_window_size', 300)
    
    def _initialize_components(self, lexicon_path: Optional[Path]):
        """Initialize all components"""
        # Pattern detector (already fixed)
        self.pattern_detector = PatternDetector(
            min_abbrev_length=2,
            max_abbrev_length=12,
            context_window_size=200,
            system_initializer=self.system_initializer,
            config=self.config
        )
        logger.info("Pattern detector initialized")
        
        # Context analyzer
        self.context_analyzer = ContextAnalyzer(self.config)
        logger.info("Context analyzer initialized")
        
        # Lexicon resolver - PASS system_initializer
        self.lexicon_resolver = None
        if self.use_lexicon and LexiconResolver:
            try:
                self.lexicon_resolver = LexiconResolver(
                    lexicon_path=lexicon_path,
                    config=self.config,
                    system_initializer=self.system_initializer  # ADD THIS
                )
                logger.info(f"Lexicon resolver initialized with {len(self.lexicon_resolver.abbreviation_dict)} entries")
            except Exception as e:
                logger.warning(f"Could not initialize lexicon resolver: {e}")
                self.use_lexicon = False
        
        # Claude enhancer
        self.claude_enhancer = None
        if self.use_claude and ClaudeEnhancer:
            self._init_claude_enhancer()
    
    def _init_claude_enhancer(self):
        """Initialize Claude enhancement module"""
        if not self.system_initializer:
            logger.warning("No system initializer, Claude not available")
            self.use_claude = False
            return
        
        try:
            claude_client = self.system_initializer.get_claude_client()
            self.claude_enhancer = ClaudeEnhancer(claude_client, self.config)
            logger.info("Claude enhancer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Claude enhancer: {e}")
            self.use_claude = False
    
    def extract_abbreviations(self, text: str, entities: Dict[str, Any] = None,
                             page_number: int = -1) -> Dict[str, Any]:
        """
        Extract and validate abbreviations from text.
        
        Pipeline:
        1. Pattern detection
        2. Local expansion enhancement
        3. Lexicon resolution
        4. Context disambiguation
        5. Claude resolution (for remaining difficult cases)
        6. Entity linking
        7. Final filtering and formatting
        
        Args:
            text: Input text
            entities: Optional extracted entities for linking
            page_number: Page number for tracking
            
        Returns:
            Dictionary with abbreviations and statistics
        """
        start_time = time.time()
        
        # NEW: Pre-process text to fix PDF artifacts BEFORE detection
        processed_text = self._preprocess_text(text)
        
        # Use processed text for pattern detection
        candidates = self.pattern_detector.detect_patterns(
            processed_text, page_number
        )
        
        

        logger.info(f"Found {len(candidates)} initial candidates")
        
        if not candidates:
            return self._empty_results(start_time)
        
        # Step 2: Enhance with local expansions
        candidates = self._enhance_with_local_expansions(candidates, processed_text)
        
        # Step 3: Context analysis for all candidates
        candidates = self._analyze_context(candidates, processed_text)
        
        # Step 4: Lexicon resolution
        resolved_candidates = []
        claude_candidates = []
        
        if self.use_lexicon and self.lexicon_resolver:
            # Get domain from context
            domain = self._get_overall_domain(processed_text)
            
            # Process through lexicon
            for candidate in candidates:
                should_send_to_claude, lexicon_entry = self.lexicon_resolver.resolve_candidate(
                    candidate, domain
                )
                
                if should_send_to_claude:
                    claude_candidates.append(candidate)
                else:
                    if lexicon_entry:
                        # Update with lexicon information
                        candidate.metadata['lexicon_resolved'] = True
                        candidate.metadata['cui'] = lexicon_entry.cui
                        candidate.metadata['semantic_types'] = lexicon_entry.semantic_types
                    resolved_candidates.append(candidate)
            
            logger.info(f"Lexicon resolved {len(resolved_candidates)}, "
                       f"sending {len(claude_candidates)} to Claude")
        else:
            # No lexicon, all candidates go to Claude if needed
            claude_candidates = candidates
        
        # Step 5: Claude resolution for difficult cases
        if self.use_claude and self.claude_enhancer and claude_candidates:
            claude_resolved = self.claude_enhancer.enhance_candidates(
                claude_candidates, processed_text
            )
            resolved_candidates.extend(claude_resolved)
            logger.info(f"Claude processed {len(claude_resolved)} candidates")
        else:
            # Add unresolved candidates
            resolved_candidates.extend(claude_candidates)
        
        # Step 6: Link to entities if provided
        if entities:
            resolved_candidates = self._link_to_entities(resolved_candidates, entities)
        
        # Step 7: Filter and deduplicate
        final_candidates = self._filter_and_deduplicate(resolved_candidates, processed_text)
        
        # Convert to results
        final_results = [self._candidate_to_result(c, processed_text) for c in final_candidates]
        
        # Calculate statistics
        statistics = self._calculate_statistics(final_results, final_candidates)
        
        # Log results
        self._log_extraction_results(final_results, statistics)
        
        return {
            'abbreviations': self._format_results(final_results),
            'statistics': statistics,
            'processing_time': time.time() - start_time
        }
    

    def _preprocess_text(self, text: str) -> str:
        """NEW METHOD: Preprocess text to fix PDF extraction issues"""
        # Fix hyphenated words broken across lines
        # Case 1: lowercase-newline-lowercase -> join without hyphen
        text = re.sub(r'(?<=[a-z])-\n(?=[a-z])', '', text)
        
        # Case 2: Medical terms with hyphens -> keep hyphen, remove newline
        text = re.sub(r'([A-Z]{2,})-\n([A-Z])', r'\1-\2', text)
        
        # Fix common OCR errors
        text = text.replace('adults-but', 'adults but')
        text = text.replace('renal-limited', 'renal limited')
        text = text.replace('treatment-free', 'treatment free')
        text = text.replace('pediatric-specific', 'pediatric specific')
        
        # Ensure space after periods
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        
        # Join split biomedical tokens
        biomedical_prefixes = ['MPO', 'PR3', 'CD', 'C', 'IL', 'TNF']
        for prefix in biomedical_prefixes:
            # Fix "MPO ANCA" -> "MPO-ANCA"
            text = re.sub(rf'\b({prefix})\s+(ANCA)\b', r'\1-\2', text)
            # Fix "CD 163" -> "CD163"
            text = re.sub(rf'\b({prefix})\s+(\d{{1,4}})\b', r'\1\2', text)
        
        return text
    

    def _enhance_with_local_expansions(self, candidates: List[AbbreviationCandidate],
                                      text: str) -> List[AbbreviationCandidate]:
        """Enhance candidates with local document expansions"""
        import re
        
        for candidate in candidates:
            # Search for local definition patterns
            window_size = 500
            start = max(0, candidate.position - window_size)
            end = min(len(text), candidate.position + window_size)
            local_text = text[start:end]
            
            # Pattern 1: Full Form (ABBR)
            pattern1 = rf'([A-Za-z0-9α-ωΑ-Ω][^(){{}}[\]<>]{{2,100}}?)\s*\(\s*{re.escape(candidate.abbreviation)}\s*\)'
            match1 = re.search(pattern1, local_text, re.IGNORECASE)
            
            if match1:
                expansion = match1.group(1).strip()
                candidate.local_expansion = expansion
                if not candidate.expansion:
                    candidate.expansion = expansion
                    candidate.source = SourceType.DOCUMENT
                    candidate.update_confidence(0.95, 'local_definition')
                    candidate.validation_status = ValidationStatus.LOCAL_DEFINITION
        
        return candidates
    
    def _analyze_context(self, candidates: List[AbbreviationCandidate],
                        text: str) -> List[AbbreviationCandidate]:
        """Analyze context for all candidates"""
        for candidate in candidates:
            # Extract context window
            start = max(0, candidate.position - self.context_window_size)
            end = min(len(text), candidate.position + self.context_window_size)
            context = text[start:end]
            
            # Get domain context
            domain = self.context_analyzer.domain_manager.get_domain_context(context)
            candidate.domain_context = domain
            
            # Get context type from expansion
            if candidate.expansion:
                candidate.context_type = self.context_analyzer.get_context_type(candidate.expansion)
            
            # Score context if expansion exists
            if candidate.expansion:
                score = self.context_analyzer.scoring_engine.score_expansion(
                    candidate.abbreviation,
                    candidate.expansion,
                    context,
                    domain
                )
                candidate.context_score = score
        
        return candidates
    
    def _get_overall_domain(self, text: str) -> str:
        """Get overall domain of the document"""
        # Sample from beginning, middle, and end
        sample_size = 500
        samples = []
        
        if len(text) > sample_size * 3:
            samples.append(text[:sample_size])
            mid = len(text) // 2
            samples.append(text[mid-sample_size//2:mid+sample_size//2])
            samples.append(text[-sample_size:])
        else:
            samples.append(text)
        
        # Get domain for each sample
        domains = []
        for sample in samples:
            domain = self.context_analyzer.domain_manager.get_domain_context(sample)
            if domain != 'general':
                domains.append(domain)
        
        # Return most common domain
        if domains:
            return Counter(domains).most_common(1)[0][0]
        return 'general'
    
    def _link_to_entities(self, candidates: List[AbbreviationCandidate],
                         entities: Dict[str, Any]) -> List[AbbreviationCandidate]:
        """Link abbreviations to extracted entities"""
        drug_names = set()
        disease_names = set()
        
        # Collect entity names
        if 'drugs' in entities:
            for drug in entities['drugs']:
                name = drug.get('name', '').lower()
                if name:
                    drug_names.add(name)
        
        if 'diseases' in entities:
            for disease in entities['diseases']:
                name = disease.get('name', '').lower()
                if name:
                    disease_names.add(name)
        
        # Link candidates
        for candidate in candidates:
            if candidate.expansion:
                expansion_lower = candidate.expansion.lower()
                if expansion_lower in drug_names:
                    candidate.context_type = ContextType.DRUG
                    candidate.update_confidence(
                        min(1.0, candidate.confidence + 0.1),
                        'entity_linked'
                    )
                elif expansion_lower in disease_names:
                    candidate.context_type = ContextType.DISEASE
                    candidate.update_confidence(
                        min(1.0, candidate.confidence + 0.1),
                        'entity_linked'
                    )
        
        return candidates
    
    def _filter_and_deduplicate(self, candidates: List[AbbreviationCandidate],
                               text: str) -> List[AbbreviationCandidate]:
        """Filter noise and deduplicate candidates"""
        # Filter by confidence
        filtered = filter_by_confidence(candidates, self.min_confidence)
        
        # Filter noise positions
        final = []
        for candidate in filtered:
            if not self.context_analyzer.noise_filter.should_filter(text, candidate.position):
                final.append(candidate)
        
        # Deduplicate
        return merge_candidates(final)
    
    def _candidate_to_result(self, candidate: AbbreviationCandidate,
                            text: str) -> AbbreviationResult:
        """Convert candidate to result"""
        import re
        
        # Count occurrences
        pattern = re.compile(r'\b' + re.escape(candidate.abbreviation) + r'\b', re.IGNORECASE)
        occurrences = len(pattern.findall(text))
        
        # Determine if validated - handle both string and class-based status
        validation_status = getattr(candidate, 'validation_status', 'unvalidated')
        if isinstance(validation_status, str):
            validated = validation_status in {
                'validated', 'local_definition', 'dictionary_match', 
                'claude_context', 'context_resolved'
            }
        else:
            validated = validation_status in {
                ValidationStatus.VALIDATED,
                ValidationStatus.LOCAL_DEFINITION,
                ValidationStatus.DICTIONARY_MATCH,
                ValidationStatus.CLAUDE_CONTEXT,
                ValidationStatus.CONTEXT_RESOLVED
            }
        
        return AbbreviationResult(
            abbreviation=candidate.abbreviation,
            expansion=candidate.expansion,
            occurrences=occurrences,
            confidence=candidate.confidence,
            source=getattr(candidate, 'source', 'unknown'),
            dictionary_sources=getattr(candidate, 'dictionary_sources', []),
            first_position=getattr(candidate, 'position', -1),
            context_type=getattr(candidate, 'context_type', 'general'),  # FIX: defensive
            validated=validated,
            validation_method=str(validation_status),
            alternative_meanings=getattr(candidate, 'alternative_expansions', [])[:5],
            detection_source=getattr(candidate, 'detection_source', 'unknown'),
            page_number=getattr(candidate, 'page_number', -1),
            local_expansion=getattr(candidate, 'local_expansion', None),
            domain_context=getattr(candidate, 'domain_context', ''),  # FIX: defensive
            claude_resolved=getattr(candidate, 'claude_resolved', False),
            lexicon_resolved=candidate.metadata.get('lexicon_resolved', False) if hasattr(candidate, 'metadata') else False,
            cui=candidate.metadata.get('cui') if hasattr(candidate, 'metadata') else None,
            semantic_types=candidate.metadata.get('semantic_types', []) if hasattr(candidate, 'metadata') else []
        )
    
    def _calculate_statistics(self, results: List[AbbreviationResult],
                            candidates: List[AbbreviationCandidate]) -> Dict[str, Any]:
        """Calculate extraction statistics"""
        return {
            'total_candidates': len(candidates),
            'total_extracted': len(results),
            'validated_count': sum(1 for r in results if r.validated),
            'lexicon_resolved': sum(1 for r in results if r.lexicon_resolved),
            'claude_resolved': sum(1 for r in results if r.claude_resolved),
            'local_definitions': sum(1 for r in results if r.local_expansion),
            'by_domain': Counter(r.domain_context for r in results),
            'by_context_type': Counter(r.context_type for r in results),
            'by_source': Counter(r.source for r in results),
            'with_cui': sum(1 for r in results if r.cui),
            'confidence_distribution': {
                'high': sum(1 for r in results if r.confidence >= 0.8),
                'medium': sum(1 for r in results if 0.6 <= r.confidence < 0.8),
                'low': sum(1 for r in results if r.confidence < 0.6)
            }
        }
    
    def _log_extraction_results(self, results: List[AbbreviationResult],
                               statistics: Dict[str, Any]):
        """Log extraction results"""
        logger.info(f"Extracted {len(results)} abbreviations")
        if statistics.get('lexicon_resolved'):
            logger.info(f"Lexicon resolved {statistics['lexicon_resolved']} abbreviations")
        if statistics.get('claude_resolved'):
            logger.info(f"Claude resolved {statistics['claude_resolved']} abbreviations")
        if statistics.get('with_cui'):
            logger.info(f"Mapped {statistics['with_cui']} abbreviations to CUIs")
        logger.debug(f"Statistics: {statistics}")
    
    def _format_results(self, results: List[AbbreviationResult]) -> List[Dict[str, Any]]:
        """Format results for output"""
        formatted = []
        
        # Sort by confidence and occurrences
        for r in sorted(results, key=lambda x: (-x.confidence, -x.occurrences)):
            result_dict = {
                'abbreviation': r.abbreviation,
                'expansion': r.expansion,
                'occurrences': r.occurrences,
                'confidence': round(r.confidence, 3),
                'source': r.source,
                'context_type': r.context_type,
                'domain_context': r.domain_context,
                'validated': r.validated,
                'validation_method': r.validation_method,
            }
            
            # Add optional fields
            if r.cui:
                result_dict['cui'] = r.cui
            
            if r.semantic_types:
                result_dict['semantic_types'] = r.semantic_types
            
            if r.lexicon_resolved:
                result_dict['lexicon_resolved'] = True
            
            if r.claude_resolved:
                result_dict['claude_resolved'] = True
            
            if r.alternative_meanings:
                result_dict['alternatives'] = r.alternative_meanings
            
            if r.local_expansion:
                result_dict['local_expansion'] = r.local_expansion
            
            formatted.append(result_dict)
        
        return formatted
    
    def _empty_results(self, start_time: float) -> Dict[str, Any]:
        """Return empty results structure"""
        return {
            'abbreviations': [],
            'statistics': {
                'total_candidates': 0,
                'total_extracted': 0,
                'validated_count': 0,
                'lexicon_resolved': 0,
                'claude_resolved': 0,
                'local_definitions': 0,
                'by_domain': {},
                'by_context_type': {},
                'by_source': {},
                'with_cui': 0,
                'confidence_distribution': {
                    'high': 0,
                    'medium': 0,
                    'low': 0
                }
            },
            'processing_time': time.time() - start_time
        }
    
    def _load_resources(self):
        """Load any additional resources from system initializer"""
        if self.system_initializer:
            try:
                # Could load additional dictionaries here if needed
                logger.info("Resources loaded from system initializer")
            except Exception as e:
                logger.warning(f"Could not load resources: {e}")
        else:
            logger.debug("No system initializer, using default resources")