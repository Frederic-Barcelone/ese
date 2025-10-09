#!/usr/bin/env python3
"""
Refactored Rare Disease Drug Detector
=====================================
Location: corpus_metadata/document_utils/rare_disease_drug_detector.py

This refactored version:
1. Uses DrugKnowledgeBase as the primary data source
2. Focuses on text processing and extraction capabilities
3. Adds unique features not in DrugKnowledgeBase:
   - Aho-Corasick automaton for ultra-fast text matching
   - Medical terms filtering for noise reduction
   - SpaCy NER integration
   - PubTator3 normalization
   - Context-aware confidence scoring

Version: 4.0 - Refactored to use DrugKnowledgeBase
"""

import re
import json
import time
import warnings
from typing import List, Dict, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from pathlib import Path
import logging

# Suppress FutureWarning about nested sets in regex
warnings.filterwarnings("ignore", category=FutureWarning, message="Possible nested set")

# Optional imports with fallback
try:
    import ahocorasick
    AHOCORASICK_AVAILABLE = True
except ImportError:
    AHOCORASICK_AVAILABLE = False
    print("Warning: pyahocorasick not installed. Install with: pip install pyahocorasick")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: SpaCy not installed. NER detection will be disabled.")

# Import DrugKnowledgeBase (REQUIRED)
try:
    from corpus_metadata.document_utils.rare_disease_drug_knowledge_base import (
        DrugKnowledgeBase, 
        get_knowledge_base
    )
    KNOWLEDGE_BASE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_BASE_AVAILABLE = False
    raise ImportError("DrugKnowledgeBase is required. Please ensure rare_disease_drug_knowledge_base.py is available.")

# Import MetadataSystemInitializer for medical terms
try:
    from corpus_metadata.document_utils.metadata_system_initializer import MetadataSystemInitializer
    SYSTEM_INIT_AVAILABLE = True
except ImportError:
    SYSTEM_INIT_AVAILABLE = False

# Import PubTator3 manager if available
try:
    from corpus_metadata.document_utils.rare_disease_pubtator3 import PubTator3Manager
    PUBTATOR_AVAILABLE = True
except ImportError:
    PUBTATOR_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Constants and Configuration
# ============================================================================

# Entity labels for drug detection
DRUG_ENTITY_LABELS = {'CHEMICAL', 'DRUG', 'CHEBI', 'SIMPLE_CHEMICAL', 'MEDICATION'}

# Medical acronyms and their expansions
MEDICAL_ACRONYMS = {
    'nsaid': 'nonsteroidal anti-inflammatory drug',
    'nsaids': 'nonsteroidal anti-inflammatory drugs',
    'acei': 'ace inhibitor',
    'aceis': 'ace inhibitors',
    'arb': 'angiotensin receptor blocker',
    'arbs': 'angiotensin receptor blockers',
    'ssri': 'selective serotonin reuptake inhibitor',
    'ssris': 'selective serotonin reuptake inhibitors',
    'snri': 'serotonin norepinephrine reuptake inhibitor',
    'ppi': 'proton pump inhibitor',
    'ppis': 'proton pump inhibitors',
    'tnf': 'tumor necrosis factor',
    'dmard': 'disease-modifying antirheumatic drug',
    'dmards': 'disease-modifying antirheumatic drugs'
}

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DrugCandidate:
    """Drug candidate with comprehensive metadata"""
    name: str
    drug_type: str = 'unknown'
    confidence: float = 0.8
    frequency: int = 1
    source: str = 'text'
    
    # Knowledge base fields
    normalized_name: Optional[str] = None
    rxcui: Optional[str] = None
    kb_drug_type: Optional[str] = None  # from knowledge base
    kb_status: Optional[str] = None
    
    # Detection metadata
    match_type: str = 'exact'  # exact, variant, partial, acronym, ner
    detection_method: str = 'kb'  # kb, pattern, ner, context
    
    # PubTator3 normalization fields
    mesh_id: Optional[str] = None
    pubtator_id: Optional[str] = None
    
    # Context fields
    context: str = ""
    position: Tuple[int, int] = (0, 0)
    
    # Filtering information
    filtered_out: bool = False
    filter_reason: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class DrugDetectionResult:
    """Complete drug detection result"""
    drugs: List[DrugCandidate]
    detection_time: float
    methods_used: List[str]
    kb_hits: int = 0
    pattern_hits: int = 0
    ner_hits: int = 0
    acronym_hits: int = 0
    normalized_count: int = 0
    filtered_count: int = 0
    filtered_terms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'drugs': [d.to_dict() for d in self.drugs],
            'detection_time': self.detection_time,
            'methods_used': self.methods_used,
            'total_drugs': len(self.drugs),
            'kb_hits': self.kb_hits,
            'pattern_hits': self.pattern_hits,
            'ner_hits': self.ner_hits,
            'acronym_hits': self.acronym_hits,
            'normalized_count': self.normalized_count,
            'filtered_count': self.filtered_count,
            'filtered_terms': self.filtered_terms
        }


# ============================================================================
# Refactored Drug Detector Class
# ============================================================================

class EnhancedDrugDetector:
    """
    Refactored drug detector using DrugKnowledgeBase as data source
    """
    
    def __init__(self, 
                 system_initializer: Optional['MetadataSystemInitializer'] = None,
                 use_kb: bool = True,
                 use_patterns: bool = True,
                 use_ner: bool = True,
                 use_pubtator: bool = True,
                 use_medical_filter: bool = True,
                 spacy_models: Optional[List[str]] = None,
                 confidence_threshold: float = 0.5):
        """
        Initialize the drug detector
        
        Args:
            system_initializer: MetadataSystemInitializer instance
            use_kb: Use DrugKnowledgeBase for detection
            use_patterns: Enable pattern-based detection
            use_ner: Enable NER-based detection
            use_pubtator: Enable PubTator3 normalization
            use_medical_filter: Enable medical terms filtering
            spacy_models: List of SpaCy model names to load
            confidence_threshold: Minimum confidence for drug candidates
        """
        self.system_initializer = system_initializer
        self.use_kb = use_kb and KNOWLEDGE_BASE_AVAILABLE
        self.use_patterns = use_patterns
        self.use_ner = use_ner and SPACY_AVAILABLE
        self.use_pubtator = use_pubtator and PUBTATOR_AVAILABLE
        self.use_medical_filter = use_medical_filter
        self.confidence_threshold = confidence_threshold
        
        # Initialize DrugKnowledgeBase
        if self.use_kb:
            self.knowledge_base = get_knowledge_base(system_initializer)
            logger.info(f"DrugKnowledgeBase initialized with {self.knowledge_base.stats['total_drugs']} drugs")
        else:
            self.knowledge_base = None
            logger.warning("DrugKnowledgeBase not available")
        
        # Initialize Aho-Corasick automaton from knowledge base
        self.ac_automaton = None
        if self.use_kb and AHOCORASICK_AVAILABLE:
            self._build_aho_corasick_automaton()
        
        # Initialize medical terms filter
        self.medical_terms_set = set()
        if self.use_medical_filter and system_initializer:
            self._initialize_medical_filter()
        
        # Load SpaCy models
        self.loaded_models = {}
        if self.use_ner and spacy_models:
            self._load_spacy_models(spacy_models)
        
        # Initialize PubTator3
        self.pubtator_manager = None
        if self.use_pubtator:
            self._initialize_pubtator()
        
        # Configuration
        self.config = {
            'context_window': 50,
            'min_term_length': 3,
            'chunk_size': 10000,
            'dosage_boost': 0.1,
            'route_boost': 0.05,
            'frequency_boost': 0.03
        }
        
        # Statistics
        self.stats = defaultdict(int)
        
        logger.info(f"Drug Detector initialized - KB: {self.use_kb}, "
                   f"Patterns: {self.use_patterns}, NER: {self.use_ner}, "
                   f"Medical Filter: {self.use_medical_filter}")
    
    def _build_aho_corasick_automaton(self):
        """Build Aho-Corasick automaton from DrugKnowledgeBase"""
        try:
            self.ac_automaton = ahocorasick.Automaton()
            
            # Add all drug names from knowledge base
            all_drug_names = self.knowledge_base.get_all_drug_names()
            all_variants = self.knowledge_base.get_all_drug_variants()
            
            # Combine all names and variants
            all_terms = all_drug_names | all_variants
            
            # FIX: Use getattr with default or hardcode min_term_length
            min_term_length = getattr(self, 'config', {}).get('min_term_length', 3)
            # Or simply: min_term_length = 3
            
            for term in all_terms:
                term_lower = term.lower()
                if len(term_lower) >= min_term_length:
                    self.ac_automaton.add_word(term_lower, (term_lower, term))
            
            self.ac_automaton.make_automaton()
            logger.info(f"Built Aho-Corasick automaton with {len(all_terms)} patterns")
            
        except Exception as e:
            logger.error(f"Failed to build Aho-Corasick automaton: {e}")
            self.ac_automaton = None
    
    def _initialize_medical_filter(self):
        """Initialize medical terms filter from system initializer"""
        try:
            # Try to get medical terms from system initializer
            if hasattr(self.system_initializer, 'medical_terms_lexicon'):
                terms = self.system_initializer.medical_terms_lexicon
                if isinstance(terms, dict):
                    self.medical_terms_set = set(terms.keys())
                elif isinstance(terms, (set, list)):
                    self.medical_terms_set = set(terms)
                logger.info(f"Medical filter initialized with {len(self.medical_terms_set)} terms")
        except Exception as e:
            logger.error(f"Failed to initialize medical filter: {e}")
    
    def _load_spacy_models(self, model_names: List[str]):
        """Load SpaCy models for NER detection"""
        for model_name in model_names:
            try:
                model = spacy.load(model_name)
                self.loaded_models[model_name] = model
                logger.info(f"Loaded SpaCy model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load SpaCy model {model_name}: {e}")
    
    def _initialize_pubtator(self):
        """Initialize PubTator3 manager"""
        try:
            self.pubtator_manager = PubTator3Manager()
            logger.info("PubTator3 manager initialized")
        except Exception as e:
            logger.warning(f"Could not initialize PubTator3: {e}")
            self.use_pubtator = False
    
    def detect_drugs(self, 
                    text: str, 
                    document_id: Optional[str] = None,
                    normalize: bool = True) -> DrugDetectionResult:
        """
        Detect drugs in text using all available methods
        
        Args:
            text: Text to analyze
            document_id: Optional document identifier
            normalize: Whether to apply PubTator3 normalization
            
        Returns:
            DrugDetectionResult with all detected drugs
        """
        start_time = time.time()
        
        if document_id:
            logger.info(f"Processing document: {document_id}")
        
        # Initialize tracking
        all_candidates = {}
        methods_used = []
        kb_hits = 0
        pattern_hits = 0
        ner_hits = 0
        acronym_hits = 0
        
        # Process text in chunks if too large
        if len(text) > self.config['chunk_size']:
            text_chunks = self._split_text(text, self.config['chunk_size'])
        else:
            text_chunks = [text]
        
        for chunk in text_chunks:
            # METHOD 1: DrugKnowledgeBase detection
            if self.use_kb:
                kb_drugs = self._detect_with_knowledge_base(chunk)
                kb_hits += len(kb_drugs)
                all_candidates.update(kb_drugs)
                if kb_drugs and 'knowledge_base' not in methods_used:
                    methods_used.append('knowledge_base')
            
            # METHOD 2: Pattern-based detection
            if self.use_patterns:
                pattern_drugs = self._detect_with_patterns(chunk)
                pattern_hits += len(pattern_drugs)
                
                # Only add patterns not found by KB
                for key, candidate in pattern_drugs.items():
                    if key not in all_candidates:
                        all_candidates[key] = candidate
                
                if pattern_drugs and 'patterns' not in methods_used:
                    methods_used.append('patterns')
            
            # METHOD 3: Acronym expansion
            acronym_drugs = self._detect_acronyms(chunk)
            acronym_hits += len(acronym_drugs)
            
            for key, candidate in acronym_drugs.items():
                if key not in all_candidates:
                    all_candidates[key] = candidate
            
            if acronym_drugs and 'acronyms' not in methods_used:
                methods_used.append('acronyms')
            
            # METHOD 4: NER detection
            if self.use_ner and self.loaded_models:
                ner_drugs = self._detect_with_ner(chunk)
                ner_hits += len(ner_drugs)
                
                # Only add NER drugs not found by other methods
                for key, candidate in ner_drugs.items():
                    if key not in all_candidates:
                        all_candidates[key] = candidate
                    else:
                        # Boost confidence if found by multiple methods
                        all_candidates[key].confidence = min(1.0, 
                            all_candidates[key].confidence + 0.05)
                
                if ner_drugs and 'ner' not in methods_used:
                    methods_used.append('ner')
        
        # Apply medical terms filtering
        filtered_terms = []
        if self.use_medical_filter:
            all_candidates, filtered_terms = self._filter_medical_terms(all_candidates)
            if filtered_terms:
                methods_used.append('medical_filter')
        
        # Filter by confidence threshold
        filtered_candidates = [
            candidate for candidate in all_candidates.values()
            if candidate.confidence >= self.confidence_threshold
        ]
        
        # Sort by confidence and detection method
        filtered_candidates.sort(key=lambda x: (
            x.detection_method == 'kb',  # KB results first
            x.confidence,
            x.name
        ), reverse=True)
        
        # PubTator3 normalization
        normalized_count = 0
        if self.use_pubtator and normalize and filtered_candidates:
            normalized_count = self._normalize_with_pubtator(filtered_candidates, text)
            if normalized_count > 0:
                methods_used.append('pubtator')
        
        # Calculate detection time
        detection_time = time.time() - start_time
        
        # Update statistics
        self.stats['total_processed'] += 1
        self.stats['kb_detections'] += kb_hits
        self.stats['pattern_detections'] += pattern_hits
        self.stats['ner_detections'] += ner_hits
        self.stats['acronym_detections'] += acronym_hits
        self.stats['normalized_drugs'] += normalized_count
        self.stats['filtered_drugs'] += len(filtered_terms)
        
        logger.info(f"Detection complete: {len(filtered_candidates)} drugs found in {detection_time:.2f}s")
        logger.info(f"Breakdown - KB: {kb_hits}, Patterns: {pattern_hits}, "
                   f"NER: {ner_hits}, Acronyms: {acronym_hits}, Filtered: {len(filtered_terms)}")
        
        return DrugDetectionResult(
            drugs=filtered_candidates,
            detection_time=detection_time,
            methods_used=methods_used,
            kb_hits=kb_hits,
            pattern_hits=pattern_hits,
            ner_hits=ner_hits,
            acronym_hits=acronym_hits,
            normalized_count=normalized_count,
            filtered_count=len(filtered_terms),
            filtered_terms=filtered_terms
        )
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks with overlap"""
        chunks = []
        overlap = 100
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def _detect_with_knowledge_base(self, text: str) -> Dict[str, DrugCandidate]:
        """Detect drugs using DrugKnowledgeBase"""
        candidates = {}
        text_lower = text.lower()
        
        if self.ac_automaton:
            # Use Aho-Corasick for fast matching
            for end_idx, (normalized, original) in self.ac_automaton.iter(text_lower):
                start_idx = end_idx - len(normalized) + 1
                
                if self._check_word_boundaries(text_lower, start_idx, end_idx + 1):
                    # Get drug info from knowledge base
                    drug_info = self.knowledge_base.get_drug_info(original)
                    
                    if drug_info:
                        key = normalized
                        
                        if key not in candidates:
                            confidence = self._calculate_confidence(
                                text, start_idx, end_idx + 1, drug_info.drug_type
                            )
                            
                            candidates[key] = DrugCandidate(
                                name=original,
                                normalized_name=drug_info.name,
                                drug_type=drug_info.drug_type,
                                confidence=confidence,
                                source=drug_info.source,
                                rxcui=drug_info.rxcui,
                                kb_drug_type=drug_info.drug_type,
                                kb_status=drug_info.status,
                                match_type='exact',
                                detection_method='kb',
                                context=self._extract_context(text, start_idx, end_idx + 1),
                                position=(start_idx, end_idx + 1)
                            )
        else:
            # Fallback to direct knowledge base search
            all_drug_names = self.knowledge_base.get_all_drug_names()
            
            for drug_name in all_drug_names:
                pattern = r'\b' + re.escape(drug_name.lower()) + r'\b'
                
                for match in re.finditer(pattern, text_lower):
                    start_idx = match.start()
                    end_idx = match.end()
                    
                    drug_info = self.knowledge_base.get_drug_info(drug_name)
                    
                    if drug_info:
                        key = drug_name.lower()
                        
                        if key not in candidates:
                            confidence = self._calculate_confidence(
                                text, start_idx, end_idx, drug_info.drug_type
                            )
                            
                            candidates[key] = DrugCandidate(
                                name=drug_name,
                                normalized_name=drug_info.name,
                                drug_type=drug_info.drug_type,
                                confidence=confidence,
                                source=drug_info.source,
                                rxcui=drug_info.rxcui,
                                kb_drug_type=drug_info.drug_type,
                                kb_status=drug_info.status,
                                match_type='exact',
                                detection_method='kb',
                                context=self._extract_context(text, start_idx, end_idx),
                                position=(start_idx, end_idx)
                            )
        
        return candidates
    
    def _detect_with_patterns(self, text: str) -> Dict[str, DrugCandidate]:
        """Pattern-based drug detection using knowledge base patterns"""
        candidates = {}
        
        if self.knowledge_base:
            # Use patterns from knowledge base
            pattern_matches = self.knowledge_base.match_patterns(text)
            
            for matched_text, confidence, description in pattern_matches:
                key = matched_text.lower()
                
                if key not in candidates:
                    candidates[key] = DrugCandidate(
                        name=matched_text,
                        drug_type=description if description else 'pattern',
                        confidence=confidence,
                        source='pattern',
                        match_type='pattern',
                        detection_method='pattern'
                    )
        
        return candidates
    
    def _detect_acronyms(self, text: str) -> Dict[str, DrugCandidate]:
        """Detect and expand medical acronyms"""
        candidates = {}
        text_lower = text.lower()
        
        for acronym, full_form in MEDICAL_ACRONYMS.items():
            pattern = r'\b' + re.escape(acronym) + r'\b'
            
            for match in re.finditer(pattern, text_lower):
                start = match.start()
                end = match.end()
                
                # Check if the full form is a known drug in KB
                if self.knowledge_base:
                    drug_info = self.knowledge_base.get_drug_info(full_form)
                    
                    if drug_info:
                        key = acronym
                        candidates[key] = DrugCandidate(
                            name=text[start:end],
                            normalized_name=full_form,
                            drug_type=drug_info.drug_type,
                            confidence=0.85,
                            source='acronym',
                            rxcui=drug_info.rxcui,
                            kb_drug_type=drug_info.drug_type,
                            match_type='acronym',
                            detection_method='acronym',
                            context=self._extract_context(text, start, end),
                            position=(start, end)
                        )
                else:
                    # Still detect as potential drug
                    key = acronym
                    candidates[key] = DrugCandidate(
                        name=text[start:end],
                        normalized_name=full_form,
                        drug_type='class',
                        confidence=0.75,
                        source='acronym',
                        match_type='acronym',
                        detection_method='acronym',
                        context=self._extract_context(text, start, end),
                        position=(start, end)
                    )
        
        return candidates
    
    def _detect_with_ner(self, text: str) -> Dict[str, DrugCandidate]:
        """NER-based drug detection using SpaCy"""
        candidates = {}
        
        for model_name, model in self.loaded_models.items():
            try:
                doc = model(text[:1000000])  # Limit text size for SpaCy
                
                for ent in doc.ents:
                    if ent.label_ in DRUG_ENTITY_LABELS:
                        key = ent.text.lower()
                        
                        if key not in candidates:
                            # Check if it's in knowledge base
                            drug_info = None
                            if self.knowledge_base:
                                drug_info = self.knowledge_base.get_drug_info(ent.text)
                            
                            if drug_info:
                                candidates[key] = DrugCandidate(
                                    name=ent.text,
                                    normalized_name=drug_info.name,
                                    drug_type=drug_info.drug_type,
                                    confidence=0.85,
                                    source=f'ner_{model_name}',
                                    rxcui=drug_info.rxcui,
                                    kb_drug_type=drug_info.drug_type,
                                    match_type='ner',
                                    detection_method='ner',
                                    position=(ent.start_char, ent.end_char)
                                )
                            else:
                                candidates[key] = DrugCandidate(
                                    name=ent.text,
                                    drug_type='unknown',
                                    confidence=0.75,
                                    source=f'ner_{model_name}',
                                    match_type='ner',
                                    detection_method='ner',
                                    position=(ent.start_char, ent.end_char)
                                )
                
            except Exception as e:
                logger.warning(f"NER detection failed with {model_name}: {e}")
        
        return candidates
    
    def _filter_medical_terms(self, candidates: Dict[str, DrugCandidate]) -> Tuple[Dict[str, DrugCandidate], List[str]]:
        """Filter out general medical terms"""
        if not self.medical_terms_set:
            return candidates, []
        
        filtered_candidates = {}
        filtered_terms = []
        
        for key, candidate in candidates.items():
            if candidate.name.lower() in self.medical_terms_set:
                filtered_terms.append(candidate.name)
                candidate.filtered_out = True
                candidate.filter_reason = "general_medical_term"
            else:
                filtered_candidates[key] = candidate
        
        return filtered_candidates, filtered_terms
    
    def _normalize_with_pubtator(self, candidates: List[DrugCandidate], text: str) -> int:
        """Normalize drugs using PubTator3"""
        if not self.pubtator_manager:
            return 0
        
        normalized_count = 0
        
        try:
            drug_names = [c.name for c in candidates]
            normalized_results = self.pubtator_manager.normalize_drugs(drug_names, text)
            
            for candidate, result in zip(candidates, normalized_results):
                if result:
                    candidate.normalized_name = result.get('normalized_name', candidate.name)
                    candidate.mesh_id = result.get('mesh_id')
                    candidate.pubtator_id = result.get('identifier')
                    normalized_count += 1
        
        except Exception as e:
            logger.warning(f"PubTator3 normalization failed: {e}")
        
        return normalized_count
    
    def _check_word_boundaries(self, text: str, start: int, end: int) -> bool:
        """Check if match has proper word boundaries"""
        if start > 0:
            prev_char = text[start - 1]
            if prev_char.isalnum() or prev_char in '-_':
                return False
        
        if end < len(text):
            next_char = text[end]
            if next_char.isalnum() or next_char in '-_':
                return False
        
        return True
    
    def _calculate_confidence(self, text: str, start: int, end: int, drug_type: str) -> float:
        """Calculate confidence score based on context"""
        base_confidence = 0.8
        
        # Boost confidence based on drug type
        if drug_type in ['approved', 'alexion']:
            base_confidence = 0.9
        elif drug_type == 'investigational':
            base_confidence = 0.85
        
        # Check context for dosage information
        context_window = self.config['context_window']
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)
        context = text[context_start:context_end].lower()
        
        # Dosage patterns
        if re.search(r'\d+\s*(?:mg|mcg|g|ml|iu)', context):
            base_confidence += self.config['dosage_boost']
        
        # Route patterns
        if re.search(r'\b(?:oral|iv|im|sc|topical)', context):
            base_confidence += self.config['route_boost']
        
        # Frequency patterns
        if re.search(r'(?:daily|weekly|twice|three times)', context):
            base_confidence += self.config['frequency_boost']
        
        return min(1.0, base_confidence)
    
    def _extract_context(self, text: str, start: int, end: int) -> str:
        """Extract context around the match"""
        window = self.config['context_window']
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        
        context = text[context_start:context_end]
        
        # Mark the matched term
        relative_start = start - context_start
        relative_end = end - context_start
        
        marked_context = (
            context[:relative_start] + 
            '[' + context[relative_start:relative_end] + ']' + 
            context[relative_end:]
        )
        
        return marked_context
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            'total_processed': self.stats['total_processed'],
            'kb_detections': self.stats['kb_detections'],
            'pattern_detections': self.stats['pattern_detections'],
            'ner_detections': self.stats['ner_detections'],
            'acronym_detections': self.stats['acronym_detections'],
            'normalized_drugs': self.stats['normalized_drugs'],
            'filtered_drugs': self.stats['filtered_drugs'],
            'kb_stats': self.knowledge_base.get_statistics() if self.knowledge_base else {},
            'detection_methods': {
                'knowledge_base': self.use_kb,
                'patterns': self.use_patterns,
                'ner': self.use_ner,
                'pubtator': self.use_pubtator,
                'medical_filter': self.use_medical_filter
            }
        }