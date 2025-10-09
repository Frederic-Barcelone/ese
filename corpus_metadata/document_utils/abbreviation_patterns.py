#!/usr/bin/env python3
"""
Enhanced Biomedical Abbreviation Extractor with Lexicon Integration
====================================================================
Version: 3.3.0 - Lexicon-Enhanced Edition
Purpose: Extract and normalize abbreviations from biomedical texts with
         integrated lexicon support for comprehensive coverage.

Key Features:
- Robust pattern matching for multiple abbreviation formats
- Integration with UMLS and medical abbreviation lexicons
- Advanced cleaning to handle PDF extraction artifacts
- Modular disease-specific sections (easily extensible)
- ID normalization for PMC, NBK, and publisher identifiers
- Semantic type inference for better categorization
- Context-aware disambiguation
- Occurrence counting and canonicalization

Data Sources:
- Pattern detection from document text
- UMLS biological/clinical abbreviations
- General medical abbreviation lexicons
- Disease-specific modules (vasculitis, etc.)
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import lexicon components
try:
    from corpus_metadata.document_utils.abbreviation_lexicon import LexiconResolver, LexiconEntry
    LEXICON_AVAILABLE = True
except ImportError:
    try:
        # Fallback to local import
        from abbreviation_lexicon import LexiconResolver, LexiconEntry
        LEXICON_AVAILABLE = True
    except ImportError:
        logger.warning("abbreviation_lexicon module not available - using fallback mode")
        LEXICON_AVAILABLE = False

# Try to import system initializer for resources
try:
    from corpus_metadata.document_utils.metadata_system_initializer import MetadataSystemInitializer
    SYSTEM_INIT_AVAILABLE = True
except ImportError:
    try:
        # Fallback to local import
        from metadata_system_initializer import MetadataSystemInitializer
        SYSTEM_INIT_AVAILABLE = True
    except ImportError:
        logger.warning("MetadataSystemInitializer not available - using standalone mode")
        SYSTEM_INIT_AVAILABLE = False


@dataclass
class AbbreviationCandidate:
    """Data class for abbreviation candidates"""
    abbreviation: str
    expansion: str
    confidence: float
    source: str = 'document'
    detection_source: str = 'unknown'
    context: str = ''
    position: int = -1
    page_number: int = -1
    local_expansion: Optional[str] = None
    validation_status: str = 'unchecked'
    semantic_type: Optional[str] = None
    semantic_tui: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    dictionary_sources: List[str] = field(default_factory=list)
    alternative_expansions: List[str] = field(default_factory=list)
    claude_resolved: bool = False


class PatternDetector:
    """
    Enhanced extractor for biomedical abbreviations with comprehensive
    cleaning and normalization capabilities.
    """
    
    def __init__(self, system_initializer=None, use_lexicon=True, **kwargs):
        """Initialize the abbreviation extractor with patterns and lexicon
        
        Args:
            system_initializer: MetadataSystemInitializer instance with loaded resources
            use_lexicon: Whether to use lexicon for resolution (default: True)
            **kwargs: Accept any parameters for compatibility with existing system
        """
        
        # Log any parameters passed (for debugging)
        if kwargs:
            logger.info(f"PatternDetector initialized with parameters: {kwargs}")
        
        # Initialize lexicon components
        self.lexicon_resolver = None
        self.system_initializer = system_initializer
        self.use_lexicon = use_lexicon and LEXICON_AVAILABLE
        
        if self.use_lexicon:
            try:
                if system_initializer and SYSTEM_INIT_AVAILABLE:
                    # Use provided system initializer
                    self.lexicon_resolver = LexiconResolver(system_initializer)
                    logger.info("Lexicon resolver initialized with system resources")
                elif SYSTEM_INIT_AVAILABLE:
                    # Try to get singleton instance
                    initializer = MetadataSystemInitializer.get_instance()
                    self.lexicon_resolver = LexiconResolver(initializer)
                    logger.info("Lexicon resolver initialized with singleton instance")
                else:
                    # Fallback to empty resolver
                    self.lexicon_resolver = LexiconResolver()
                    logger.warning("Lexicon resolver initialized without resources")
            except Exception as e:
                logger.warning(f"Failed to initialize lexicon resolver: {e}")
                self.use_lexicon = False
        
        # Core abbreviation pattern components
        # Character class for abbreviations (without quantifier)
        self.ABBR_CHARS_CLASS = r'[A-Z0-9α-ωΑ-Ω/\-]'
        
        # Continue with rest of initialization...
        self._compile_patterns()
        self._init_filters()
        self._load_overrides()
        
        self.stats = {
            'total_processed': 0,
            'abbreviations_found': 0,
            'filtered_out': 0,
            'overrides_applied': 0,
            'lexicon_resolved': 0,
            'lexicon_enriched': 0
        }
    
    def _compile_patterns(self):
        """Compile all regex patterns for abbreviation detection"""
        
        # Pattern for "Long Form (ABBR)" - stricter version
        # Captures only capitalized noun phrases, no sentences
        self.definition_pattern = re.compile(
            r'([A-Z][A-Za-z0-9α-ωΑ-Ω]+(?:\s+(?:and|or|of|for|with|in)?\s*[A-Za-z0-9α-ωΑ-Ω\-]+){0,8}?)\s*'
            r'\(\s*([A-Z]' + self.ABBR_CHARS_CLASS + r'{1,11})\s*\)',
            flags=re.MULTILINE
        )
        
        # Pattern for "ABBR (Long Form)"
        self.reverse_pattern = re.compile(
            r'\b([A-Z]' + self.ABBR_CHARS_CLASS + r'{1,11})\s*\(\s*'
            r'([A-Z][A-Za-z0-9α-ωΑ-Ω\-\s]{2,100}?)\s*\)',
            flags=re.MULTILINE
        )
        
        # Pattern for "ABBR: Long Form" or "ABBR - Long Form"
        self.colon_dash_pattern = re.compile(
            r'\b([A-Z]' + self.ABBR_CHARS_CLASS + r'{1,11})\s*[:–—-]\s*'
            r'([A-Z][A-Za-z0-9α-ωΑ-Ω\-\s]{2,100}?)(?=[.;,\n]|$)',
            flags=re.MULTILINE
        )
        
        # Shape guard for abbreviations (needs ≥2 caps or a digit)
        self.ABBR_SHAPE_GUARD = re.compile(r'(?:[A-Z].*[A-Z]|\d)')
        
        # Lead-in phrase removal pattern
        self.LEADIN_CUTS = re.compile(
            r'^(?:'
            r'(?:as\s+(?:a|an|the)\s+)|'
            r'(?:the\s+(?:role|impact|use|study|trial|hallmark\s+trials)\s+of\s+)|'
            r'(?:the\s+(?:role|impact|use|study|trial|hallmark\s+trials)\s+in\s+)|'
            r'(?:role\s+of\s+|impact\s+of\s+|use\s+of\s+|study\s+of\s+|trial\s+of\s+|'
            r'hallmark\s+trials\s+in\s+|guidelines?\s+for\s+|recommendations?\s+for\s+)'
            r')',
            flags=re.IGNORECASE
        )
        
        # Pattern to find capitals in tokens
        self.CAP_IN_TOKEN = re.compile(r'[A-Z]')
        
        # Early verb detection for sentence filtering
        self.EARLY_VERBS = re.compile(
            r'\b(is|are|was|were|be|being|been|has|have|had|shows?|suggests?|'
            r'demonstrates?|presents?)\b',
            re.IGNORECASE
        )
        
        # ID format patterns
        self.PMC_PATTERN = re.compile(r'PMC\d{6,}', flags=re.IGNORECASE)
        self.NBK_PATTERN = re.compile(r'NBK\d+', flags=re.IGNORECASE)
        self.PUBLISHER_ID_PATTERN = re.compile(r'S\d{4}-\d{4}', flags=re.IGNORECASE)
        self.NEJM_PATTERN = re.compile(r'NEJMoa\d+', flags=re.IGNORECASE)
    
    def _init_filters(self):
        """Initialize stopwords and filtering lists"""
        
        # Non-abbreviations to filter out
        self.STOPLIST = {
            'THE', 'THIS', 'THAT', 'WITH', 'FROM', 'INTO', 'OVER', 'UNDER',
            'AND', 'BUT', 'FOR', 'NOR', 'YET', 'ALSO', 'BOTH', 'EITHER',
            'ABOUT', 'ABOVE', 'AFTER', 'BEFORE', 'BELOW', 'BETWEEN',
            'RESEARCHGATE', 'STATPEARLS', 'PUBMED', 'GOOGLE', 'WIKIPEDIA',
            'HTTP', 'HTTPS', 'WWW', 'COM', 'ORG', 'NET', 'EDU', 'GOV',
            'ANTI', 'PRO', 'PRE', 'POST', 'MULTI', 'SEMI', 'AUTO'
        }
        
        # Common words that shouldn't be abbreviations
        self.COMMON_WORDS = {
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
            'saturday', 'sunday', 'today', 'tomorrow', 'yesterday'
        }
        
        # Invalid expansion starts
        self.INVALID_STARTS = {
            'references', 'ref', 'refs', 'reference', 'citation', 'cite',
            'available', 'accessed', 'retrieved', 'downloaded', 'viewed',
            'http', 'https', 'www', 'doi', 'pmid', 'pmc'
        }
        
        # Base valid singletons - will be extended by disease modules
        self.VALID_SINGLETONS = {
            'receptor', 'component', 'antibody', 'protein', 'enzyme',
            'syndrome', 'disease', 'disorder', 'virus', 'assay'
        }
    
    def _load_overrides(self):
        """Load all available override modules - general and all disease-specific"""
        
        # Initialize with empty overrides
        self.OVERRIDES = {}
        
        # Always load general biomedical abbreviations
        self._load_general_overrides()
        
        # Always load ALL available disease-specific modules
        self._load_vasculitis_overrides()
        
        # Add more disease modules here as you implement them
        # self._load_oncology_overrides()
        # self._load_neurology_overrides()
        # self._load_cardiology_overrides()
        # self._load_rare_diseases_overrides()
        
        logger.info(f"Loaded {len(self.OVERRIDES)} total overrides from all modules")
    
    def _load_general_overrides(self):
        """Load general biomedical abbreviations common across all domains"""
        
        general_overrides = {
            # Common clinical/technical
            'IV': 'Intravenous',
            'IM': 'Intramuscular',
            'SC': 'Subcutaneous',
            'PO': 'Per os (by mouth)',
            'BID': 'Twice daily',
            'TID': 'Three times daily',
            'QD': 'Once daily',
            'PRN': 'As needed',
            
            # Common lab tests
            'CBC': 'Complete blood count',
            'CRP': 'C-reactive protein',
            'ESR': 'Erythrocyte sedimentation rate',
            'PCR': 'Polymerase chain reaction',
            'ELISA': 'Enzyme-Linked Immunosorbent Assay',
            'WBC': 'White blood cell',
            'RBC': 'Red blood cell',
            
            # Organizations
            'FDA': 'Food and Drug Administration',
            'EMA': 'European Medicines Agency',
            'WHO': 'World Health Organization',
            'NIH': 'National Institutes of Health',
            'CDC': 'Centers for Disease Control and Prevention',
            'NCBI': 'National Center for Biotechnology Information',
            
            # Common diseases
            'COVID-19': 'Coronavirus Disease 2019',
            'SARS': 'Severe Acute Respiratory Syndrome',
            'SARS-CoV-2': 'Severe Acute Respiratory Syndrome Coronavirus 2',
            'HIV': 'Human immunodeficiency virus',
            'AIDS': 'Acquired immunodeficiency syndrome',
            
            # Study types
            'RCT': 'Randomized controlled trial',
            'DB': 'Double-blind',
            'PC': 'Placebo-controlled',
            
            # Statistical
            'SD': 'Standard deviation',
            'SE': 'Standard error',
            'CI': 'Confidence interval',
            'OR': 'Odds ratio',
            'HR': 'Hazard ratio',
            'RR': 'Relative risk',
        }
        
        self.OVERRIDES.update(general_overrides)
    
    # =========================================================================
    # DISEASE-SPECIFIC SECTIONS - Add new disease modules below
    # =========================================================================
    
    def _load_vasculitis_overrides(self):
        """
        VASCULITIS/ANCA MODULE
        ----------------------
        Specific abbreviations for vasculitis, ANCA-associated conditions,
        and related immunological markers.
        """
        
        vasculitis_overrides = {
            # Core vasculitis diseases
            'AAV': 'ANCA-associated vasculitis',
            'ANCA': 'Antineutrophil cytoplasmic antibodies',
            'GPA': 'Granulomatosis with polyangiitis',
            'MPA': 'Microscopic polyangiitis',
            'EGPA': 'Eosinophilic granulomatosis with polyangiitis',
            'GPA/MPA': 'Granulomatosis with polyangiitis / Microscopic polyangiitis',
            
            # Antibody/protein markers
            'PR3': 'Proteinase 3',
            'MPO': 'Myeloperoxidase',
            'P-ANCA': 'Perinuclear ANCA',
            'C-ANCA': 'Cytoplasmic ANCA',
            'MPO-ANCA': 'Myeloperoxidase-ANCA',
            'PR3-ANCA': 'Proteinase 3-ANCA',
            
            # Complement components
            'C5': 'Complement component 5',
            'C5a': 'Complement component 5a',
            'C5aR': 'C5a receptor',
            'C5aR1': 'C5a receptor 1 (CD88)',
            'CD88': 'C5a receptor (CD88)',
            'C5b-9': 'Membrane attack complex',
            'MAC': 'Membrane attack complex',
            
            # Biomarkers
            'CD163': 'CD163 (hemoglobin-haptoglobin scavenger receptor)',
            'usCD163': 'Urinary soluble CD163',
            'NET': 'Neutrophil extracellular trap',
            'NETs': 'Neutrophil extracellular traps',
            
            # Vasculitis-specific organizations
            'CARRA': 'Childhood Arthritis and Rheumatology Research Alliance',
            'ACR': 'American College of Rheumatology',
            'EULAR': 'European Alliance of Associations for Rheumatology',
            'KDIGO': 'Kidney Disease: Improving Global Outcomes',
            
            # Vasculitis drugs
            'TAVNEOS': 'Avacopan',
            'RITUXAN': 'Rituximab',
            
            # Vasculitis trials
            'PEXIVAS': 'Plasma Exchange in ANCA-Associated Vasculitis',
            'ADVOCATE': 'Avacopan Phase 3 trial (ADVOCATE)',
        }
        
        # Add vasculitis-specific valid singletons
        vasculitis_singletons = {
            'vasculitis', 'rituximab', 'avacopan', 'metabolite'
        }
        
        self.OVERRIDES.update(vasculitis_overrides)
        self.VALID_SINGLETONS.update(vasculitis_singletons)
        
        # Add vasculitis-specific type mappings
        if not hasattr(self, 'VASCULITIS_TYPES'):
            self.VASCULITIS_TYPES = {
                'autoantibody': ['ANCA', 'MPO-ANCA', 'PR3-ANCA', 'P-ANCA', 'C-ANCA'],
                'enzyme/protein': ['MPO', 'PR3', 'CD163'],
                'complement': ['C5', 'C5a', 'C5b-9', 'MAC'],
                'receptor': ['C5aR', 'C5aR1', 'CD88'],
                'biomarker': ['NET', 'NETs', 'usCD163'],
                'disease': ['AAV', 'GPA', 'MPA', 'EGPA'],
                'drug_brand': ['TAVNEOS', 'RITUXAN'],
                'trial': ['PEXIVAS', 'ADVOCATE']
            }
    
    # =========================================================================
    # Add more disease-specific modules here
    # =========================================================================
    
    # def _load_oncology_overrides(self):
    #     """
    #     ONCOLOGY MODULE
    #     ---------------
    #     Specific abbreviations for cancer, oncology treatments, and biomarkers.
    #     """
    #     oncology_overrides = {
    #         # Cancer types
    #         'NSCLC': 'Non-small cell lung cancer',
    #         'SCLC': 'Small cell lung cancer',
    #         'CRC': 'Colorectal cancer',
    #         'HCC': 'Hepatocellular carcinoma',
    #         
    #         # Biomarkers
    #         'EGFR': 'Epidermal growth factor receptor',
    #         'PD-L1': 'Programmed death-ligand 1',
    #         'HER2': 'Human epidermal growth factor receptor 2',
    #         
    #         # Add more oncology abbreviations...
    #     }
    #     
    #     self.OVERRIDES.update(oncology_overrides)
    
    # =========================================================================
    # Core processing methods
    # =========================================================================
    
    def _pre_normalize_text(self, text: str) -> str:
        """Pre-normalize text to canonicalize common variants"""
        
        # General normalizations
        text = re.sub(r'\bCOVID\b', 'COVID-19', text)
        
        # Vasculitis-specific normalizations (always applied)
        text = re.sub(r'\bANCAv\b', 'AAV', text)
        
        return text
    
    def _canonicalize_abbr(self, abbr: str) -> str:
        """Map abbreviation equivalents to canonical form"""
        a = abbr.strip()
        
        # General canonicalizations
        if a == 'COVID':
            return 'COVID-19'
        
        # Vasculitis-specific canonicalizations (always applied)
        if a == 'CD88':
            return 'C5aR1'  # canonical abbr
        
        return a
    
    def _is_identifier_abbr(self, abbr: str) -> Optional[str]:
        """Check if abbreviation is an identifier type"""
        if self.PMC_PATTERN.fullmatch(abbr):
            return 'pmcid'
        if self.NBK_PATTERN.fullmatch(abbr):
            return 'ncbi_bookshelf'
        if self.PUBLISHER_ID_PATTERN.fullmatch(abbr):
            return 'publisher_id'
        if self.NEJM_PATTERN.fullmatch(abbr):
            return 'nejm_id'
        if re.fullmatch(r'[A-Z]+\d{6,}', abbr):
            return 'db_id'
        return None
    
    def _infer_type(self, abbr: str, expansion: str) -> Optional[str]:
        """Infer semantic type from abbreviation and expansion"""
        a, e = abbr, (expansion or '')
        
        # Check if it's an identifier
        id_type = self._is_identifier_abbr(a)
        if id_type:
            return 'identifier'
        
        # Check vasculitis-specific types FIRST (more specific rules)
        if hasattr(self, 'VASCULITIS_TYPES'):
            for type_name, abbr_list in self.VASCULITIS_TYPES.items():
                if a in abbr_list:
                    return type_name
        
        # Then check expansion-based inference
        e_lower = e.lower()
        
        # Specific patterns
        if 'receptor' in e_lower:
            return 'receptor'
        if 'antibod' in e_lower or 'anca' in e_lower:
            return 'autoantibody'
        if any(term in e_lower for term in ['enzyme', 'ase', 'proteinase', 'myeloperoxidase']):
            return 'enzyme/protein'
        if any(term in e_lower for term in ['virus', 'viral']):
            return 'virus'
        if any(term in e_lower for term in ['assay', 'test']):
            return 'assay'
        if 'complement' in e_lower:
            return 'complement'
        
        # Organization patterns
        if any(term in e_lower for term in ['organization', 'association', 'alliance', 
                                              'administration', 'institute', 'institutes',
                                              'college', 'center', 'academy']):
            return 'organization'
        
        # Disease patterns  
        if any(term in e_lower for term in ['disease', 'syndrome', 'disorder', 'cancer', 
                                              'carcinoma', 'vasculitis', 'polyangiitis']):
            return 'disease'
        
        # Drug patterns
        if any(term in e_lower for term in ['drug', 'medication', 'therapeutic']) or a in ['TAVNEOS', 'RITUXAN']:
            return 'drug'
        
        # Trial patterns
        if any(term in e_lower for term in ['trial', 'study', 'phase']) and 'phase 3' in e_lower:
            return 'trial'
        
        # Route of administration
        if a in ['IV', 'IM', 'SC', 'PO'] or 'intravenous' in e_lower or 'intramuscular' in e_lower:
            return 'route'
        
        return None
    
    def _contextual_override(self, abbr: str, expansion: str, context: str) -> Tuple[str, Optional[str], Dict]:
        """Return (expansion, semantic_type, extra_metadata) possibly adjusted by context."""
        meta = {}
        sem = None
        
        # Vasculitis-specific context overrides (always applied)
        # M1 context disambiguation for vasculitis texts
        if abbr == 'M1':
            ctx = context.lower()
            if 'avacopan' in ctx or 'cyp3a4' in ctx:
                return ('avacopan active metabolite (M1)', 'metabolite', meta)
            # else default to macrophage phenotype
            return ('M1 macrophage (classically activated) phenotype', 'enzyme/protein', meta)
        
        # Canonicalize CD88 → C5aR1 expansion
        if abbr in {'CD88', 'C5aR', 'C5aR1'}:
            return ('C5a receptor 1 (CD88)', 'receptor', meta)
        
        return (expansion, sem, meta)
    
    def _post_validate(self, abbr: str, expansion: str) -> str:
        """Repair obviously wrong long forms for known entities."""
        
        # Vasculitis-specific validations (always applied)
        if abbr in {'MPO', 'PR3'}:
            # If the captured long form looks clinical rather than the antigen
            if re.search(r'\bANCA|glomerulonephritis|positiv|involvement\b', expansion, flags=re.I):
                return self.OVERRIDES.get(abbr, expansion)
        
        # General validations
        if abbr == 'SARS' and 'coronavirus 2' in expansion.lower():
            return 'Severe Acute Respiratory Syndrome'  # keep SARS distinct from SARS-CoV-2
        
        return expansion
    
    def _enrich_with_lexicon(self, abbr: str, expansion: str = None, context: str = "") -> Tuple[str, float, Dict]:
        """
        Enrich abbreviation with lexicon information.
        
        Args:
            abbr: Abbreviation to look up
            expansion: Expansion from document (if found)
            context: Context around the abbreviation
            
        Returns:
            Tuple of (expansion, confidence_boost, metadata)
        """
        if not self.use_lexicon or not self.lexicon_resolver:
            return (expansion, 0.0, {})
        
        try:
            # Look up in lexicon
            lexicon_entries = self.lexicon_resolver.resolve(abbr.upper())
            
            if not lexicon_entries:
                return (expansion, 0.0, {})
            
            # If we have a document expansion, verify it against lexicon
            if expansion:
                normalized_exp = expansion.lower().strip()
                for entry in lexicon_entries:
                    if entry.preferred_term.lower().strip() == normalized_exp:
                        # Exact match - boost confidence
                        metadata = {
                            'lexicon_verified': True,
                            'lexicon_source': entry.source_vocabulary,
                            'semantic_types': entry.semantic_types,
                            'cui': entry.cui if hasattr(entry, 'cui') else None
                        }
                        self.stats['lexicon_resolved'] += 1
                        return (expansion, 0.05, metadata)
            
            # No document expansion or no match - use best lexicon entry
            if not expansion and lexicon_entries:
                best_entry = lexicon_entries[0]  # Assume sorted by relevance
                
                # Check if context supports this expansion
                context_score = self._score_context_match(best_entry.preferred_term, context)
                
                if context_score > 0.3 or not context:  # Accept if good context match or no context
                    metadata = {
                        'lexicon_resolved': True,
                        'lexicon_source': best_entry.source_vocabulary,
                        'semantic_types': best_entry.semantic_types,
                        'cui': best_entry.cui if hasattr(best_entry, 'cui') else None,
                        'context_score': context_score
                    }
                    self.stats['lexicon_enriched'] += 1
                    # Lower confidence for lexicon-only expansions
                    return (best_entry.preferred_term, -0.1, metadata)
            
            return (expansion, 0.0, {})
            
        except Exception as e:
            logger.debug(f"Lexicon lookup failed for {abbr}: {e}")
            return (expansion, 0.0, {})
    
    def _score_context_match(self, expansion: str, context: str) -> float:
        """
        Score how well an expansion matches the context.
        
        Args:
            expansion: Proposed expansion
            context: Context string
            
        Returns:
            Score between 0 and 1
        """
        if not context or not expansion:
            return 0.0
        
        context_lower = context.lower()
        expansion_words = expansion.lower().split()
        
        # Check for word overlap
        matches = sum(1 for word in expansion_words 
                     if len(word) > 3 and word in context_lower)
        
        if len(expansion_words) > 0:
            return min(1.0, matches / len(expansion_words))
        return 0.0
    
    def _count_occurrences(self, text: str, abbr: str) -> int:
        """Count occurrences of abbreviation in text"""
        # Count singular/plural (NET/NETs) and canonical alias (CD88→C5aR1)
        pattern = r'\b' + re.escape(abbr) + r's?\b'
        return len(re.findall(pattern, text))
    
    def _abbr_is_plausible(self, abbr: str) -> bool:
        """
        Check if abbreviation has plausible shape.
        Rejects titlecase words like 'Anti' that aren't real abbreviations.
        """
        # Must have 2+ capitals or contain a digit
        if not self.ABBR_SHAPE_GUARD.search(abbr):
            return False
        
        # Filter out known non-abbreviations
        if abbr.upper() in self.STOPLIST:
            return False
        
        # Check for ID formats that should be handled separately
        if (self.PMC_PATTERN.fullmatch(abbr) or 
            self.NBK_PATTERN.fullmatch(abbr) or
            self.PUBLISHER_ID_PATTERN.fullmatch(abbr) or
            self.NEJM_PATTERN.fullmatch(abbr)):
            return True  # These are valid but need special handling
        
        return True
    
    def _strip_leadin(self, text: str) -> str:
        """Remove common lead-in phrases from expansions"""
        return self.LEADIN_CUTS.sub('', text).strip()
    
    def _trim_to_core_np(self, text: str) -> str:
        """
        Trim to the rightmost capital-bearing chunk.
        Keeps "ANCA-associated vasculitis" intact.
        """
        tokens = text.split()
        
        # Find the last token containing an uppercase letter
        last_cap_idx = -1
        for i, tok in enumerate(tokens):
            if self.CAP_IN_TOKEN.search(tok):
                last_cap_idx = i
        
        if last_cap_idx > 0:
            # Check for prepositions before the capital-bearing token
            prefix = ' '.join(tokens[:last_cap_idx])
            prep_words = {'of', 'in', 'for', 'with', 'to', 'from', 'by', 'at'}
            if any(p in prefix.lower().split() for p in prep_words):
                return ' '.join(tokens[last_cap_idx:]).strip()
        
        return text
    
    def _looks_like_sentence(self, text: str) -> bool:
        """
        Detect if expansion looks like a sentence rather than a noun phrase.
        Checks for early verbs which indicate sentence structure.
        """
        # Check first 6 tokens for verbs
        first_words = ' '.join(text.split()[:6])
        return bool(self.EARLY_VERBS.search(first_words))
    
    def _clean_expansion(self, text: str) -> str:
        """
        Comprehensive expansion cleaning:
        - Remove reference artifacts
        - Strip lead-in phrases
        - Trim to core noun phrase
        - Validate structure
        """
        # Basic whitespace normalization
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove reference/citation artifacts
        text = re.sub(r'^(?:references?|ref\.?)\s*\d*\.?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+\.\s*', '', text)
        
        # Remove URL fragments
        text = re.sub(r'^(?:https?://|www\.)\S+\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bhttps?://\S+', '', text, flags=re.IGNORECASE)
        
        # Remove trailing punctuation
        text = re.sub(r'[.,;:]+$', '', text)
        
        # Strip lead-in phrases
        text = self._strip_leadin(text)
        
        # Trim to core noun phrase
        text = self._trim_to_core_np(text)
        
        # Validation checks
        if not text or len(text) < 2:
            return ''
        
        # Must start with capital letter
        if not text[0].isupper():
            return ''
        
        # Reject if looks like a sentence
        if self._looks_like_sentence(text):
            return ''
        
        # Check for invalid starts
        first_word = text.split()[0].lower()
        if first_word in self.INVALID_STARTS:
            return ''
        
        # Prefer multi-word expansions unless known singletons
        if ' ' not in text and text.lower() not in self.VALID_SINGLETONS:
            # Be more strict with single-word expansions
            if len(text) < 5:  # Too short for meaningful singleton
                return ''
        
        return text.strip()
    
    def _normalize_abbrev(self, abbr: str, expansion: str = '') -> str:
        """
        Normalize abbreviation:
        - Handle plurals (NETs → NET)
        - Standardize formatting
        """
        a = abbr.strip()
        e = expansion.strip()
        
        # Handle plural abbreviations
        if a.endswith('s') and not a.endswith('ss'):
            # Check if expansion is also plural
            if e and e.lower().endswith('s'):
                # Both are plural, singularize abbreviation
                singular = a[:-1]
                # But keep some that are always plural
                if singular not in {'CARRA', 'KDIGO', 'PEXIVAS'}:
                    a = singular
        
        return a
    
    def _id_normalize(self, abbr: str) -> Optional[str]:
        """
        Normalize ID-format abbreviations to proper descriptions.
        Returns None if not an ID format.
        """
        if self.PMC_PATTERN.fullmatch(abbr):
            id_num = re.sub(r'^PMC', '', abbr, flags=re.IGNORECASE)
            return f'PubMed Central article {id_num}'
        
        if self.NBK_PATTERN.fullmatch(abbr):
            id_num = re.sub(r'^NBK', '', abbr, flags=re.IGNORECASE)
            return f'NCBI Bookshelf ID {id_num}'
        
        if self.PUBLISHER_ID_PATTERN.fullmatch(abbr):
            return 'Publisher article identifier'
        
        if self.NEJM_PATTERN.fullmatch(abbr):
            return 'New England Journal of Medicine article'
        
        # Check for other journal/database patterns
        if re.fullmatch(r'[A-Z]+\d{6,}', abbr):
            return 'Database identifier'
        
        return None
    
    def _get_context(self, text: str, position: int, window: int = 60) -> str:
        """Extract context window around the match position"""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end].strip()
    
    def _process_definition_pattern(self, text: str, seen: Set[Tuple[str, int]], 
                                   page_number: int = -1) -> List[AbbreviationCandidate]:
        """
        Process 'Long Form (ABBR)' pattern with comprehensive cleaning and lexicon enrichment.
        This is the main pattern for formal definitions.
        """
        candidates = []
        
        for match in self.definition_pattern.finditer(text):
            long_raw = match.group(1).strip()
            abbr_raw = match.group(2).strip()
            
            # Check abbreviation shape
            if not self._abbr_is_plausible(abbr_raw):
                continue
            
            # Clean the expansion
            expansion = self._clean_expansion(long_raw)
            
            if not expansion:
                # Try override or ID normalization
                expansion = (self.OVERRIDES.get(abbr_raw) or 
                           self._id_normalize(abbr_raw) or '')
                if not expansion:
                    # Try lexicon as last resort
                    context = self._get_context(text, match.start())
                    lex_expansion, lex_boost, lex_meta = self._enrich_with_lexicon(abbr_raw, None, context)
                    if lex_expansion:
                        expansion = lex_expansion
                    else:
                        continue
                else:
                    self.stats['overrides_applied'] += 1
            
            # Post-validate expansion
            expansion = self._post_validate(abbr_raw, expansion)
            
            # Normalize abbreviation
            abbr = self._normalize_abbrev(abbr_raw, expansion)
            
            # Context disambiguation
            context = self._get_context(text, match.start())
            expansion, sem_ctx, meta_ctx = self._contextual_override(abbr, expansion, context)
            
            # Enrich with lexicon information
            lex_expansion, conf_boost, lex_metadata = self._enrich_with_lexicon(abbr, expansion, context)
            if lex_expansion and not expansion:
                expansion = lex_expansion
            
            # Merge metadata
            meta_ctx.update(lex_metadata)
            
            # Canonicalize abbreviation
            canonical = self._canonicalize_abbr(abbr)
            aliases = [abbr] if abbr != canonical else []
            
            # Infer semantic type (prefer lexicon semantic types)
            if 'semantic_types' in lex_metadata and lex_metadata['semantic_types']:
                semantic_type = lex_metadata['semantic_types'][0]
            else:
                semantic_type = sem_ctx or self._infer_type(canonical, expansion)
            
            # Adjust confidence based on lexicon verification
            base_confidence = 0.95
            final_confidence = min(1.0, base_confidence + conf_boost)
            
            # Check for duplicates
            pos_key = (canonical, match.start())
            if pos_key not in seen:
                seen.add(pos_key)
                
                candidate = AbbreviationCandidate(
                    abbreviation=canonical,
                    expansion=expansion,
                    confidence=final_confidence,
                    source='document',
                    detection_source='definition_parentheses',
                    context=context,
                    position=match.start(),
                    page_number=page_number,
                    local_expansion=long_raw,  # Keep original text
                    validation_status='local_definition',
                    semantic_type=semantic_type,
                    metadata={'aliases': aliases, **meta_ctx}
                )
                candidates.append(candidate)
                
        return candidates
    
    def _process_reverse_pattern(self, text: str, seen: Set[Tuple[str, int]], 
                                page_number: int = -1) -> List[AbbreviationCandidate]:
        """Process 'ABBR (Long Form)' pattern"""
        candidates = []
        
        for match in self.reverse_pattern.finditer(text):
            abbr_raw = match.group(1).strip()
            long_raw = match.group(2).strip()
            
            # Check abbreviation shape
            if not self._abbr_is_plausible(abbr_raw):
                continue
            
            # Clean the expansion
            expansion = self._clean_expansion(long_raw)
            
            if not expansion:
                # Try override
                expansion = self.OVERRIDES.get(abbr_raw, '')
                if not expansion:
                    continue
                self.stats['overrides_applied'] += 1
            
            # Post-validate expansion
            expansion = self._post_validate(abbr_raw, expansion)
            
            # Normalize abbreviation
            abbr = self._normalize_abbrev(abbr_raw, expansion)
            
            # Context disambiguation
            context = self._get_context(text, match.start())
            expansion, sem_ctx, meta_ctx = self._contextual_override(abbr, expansion, context)
            
            # Canonicalize abbreviation
            canonical = self._canonicalize_abbr(abbr)
            aliases = [abbr] if abbr != canonical else []
            
            # Infer semantic type
            semantic_type = sem_ctx or self._infer_type(canonical, expansion)
            
            # Check for duplicates
            pos_key = (canonical, match.start())
            if pos_key not in seen:
                seen.add(pos_key)
                
                candidate = AbbreviationCandidate(
                    abbreviation=canonical,
                    expansion=expansion,
                    confidence=0.90,
                    source='document',
                    detection_source='reverse_parentheses',
                    context=context,
                    position=match.start(),
                    page_number=page_number,
                    local_expansion=expansion,
                    validation_status='local_definition',
                    semantic_type=semantic_type,
                    metadata={'aliases': aliases, **meta_ctx}
                )
                candidates.append(candidate)
                
        return candidates
    
    def _process_colon_dash_pattern(self, text: str, seen: Set[Tuple[str, int]], 
                                   page_number: int = -1) -> List[AbbreviationCandidate]:
        """Process 'ABBR: Long Form' or 'ABBR - Long Form' pattern"""
        candidates = []
        
        for match in self.colon_dash_pattern.finditer(text):
            abbr_raw = match.group(1).strip()
            long_raw = match.group(2).strip()
            
            # Check abbreviation shape
            if not self._abbr_is_plausible(abbr_raw):
                continue
            
            # Clean the expansion
            expansion = self._clean_expansion(long_raw)
            
            if not expansion:
                # Try override
                expansion = self.OVERRIDES.get(abbr_raw, '')
                if not expansion:
                    continue
                self.stats['overrides_applied'] += 1
            
            # Post-validate expansion
            expansion = self._post_validate(abbr_raw, expansion)
            
            # Normalize abbreviation
            abbr = self._normalize_abbrev(abbr_raw, expansion)
            
            # Context disambiguation
            context = self._get_context(text, match.start())
            expansion, sem_ctx, meta_ctx = self._contextual_override(abbr, expansion, context)
            
            # Canonicalize abbreviation
            canonical = self._canonicalize_abbr(abbr)
            aliases = [abbr] if abbr != canonical else []
            
            # Infer semantic type
            semantic_type = sem_ctx or self._infer_type(canonical, expansion)
            
            # Check for duplicates
            pos_key = (canonical, match.start())
            if pos_key not in seen:
                seen.add(pos_key)
                
                candidate = AbbreviationCandidate(
                    abbreviation=canonical,
                    expansion=expansion,
                    confidence=0.85,
                    source='document',
                    detection_source='colon_dash',
                    context=context,
                    position=match.start(),
                    page_number=page_number,
                    local_expansion=expansion,
                    validation_status='local_definition',
                    semantic_type=semantic_type,
                    metadata={'aliases': aliases, **meta_ctx}
                )
                candidates.append(candidate)
                
        return candidates
    
    def _find_standalone_abbreviations(self, text: str, seen: Set[str], 
                                      page_number: int = -1) -> List[AbbreviationCandidate]:
        """
        Find standalone abbreviations without explicit definitions.
        Use overrides dictionary and lexicon for known expansions.
        """
        candidates = []
        
        # Pattern for standalone abbreviations
        standalone_pattern = re.compile(r'\b([A-Z]' + self.ABBR_CHARS_CLASS + r'{1,11})\b')
        
        for match in standalone_pattern.finditer(text):
            abbr_raw = match.group(1).strip()
            
            # Skip if already seen with definition
            if abbr_raw in seen:
                continue
            
            # Check abbreviation shape
            if not self._abbr_is_plausible(abbr_raw):
                continue
            
            # Try to get expansion from overrides or ID normalization
            expansion = (self.OVERRIDES.get(abbr_raw) or 
                       self._id_normalize(abbr_raw))
            
            source = 'override' if expansion else None
            
            # If no override, try lexicon
            if not expansion and self.use_lexicon:
                context = self._get_context(text, match.start())
                lex_expansion, conf_boost, lex_metadata = self._enrich_with_lexicon(abbr_raw, None, context)
                if lex_expansion:
                    expansion = lex_expansion
                    source = 'lexicon'
            
            if expansion:
                abbr = self._normalize_abbrev(abbr_raw, expansion)
                
                # Context disambiguation
                context = self._get_context(text, match.start())
                expansion, sem_ctx, meta_ctx = self._contextual_override(abbr, expansion, context)
                
                # Canonicalize abbreviation
                canonical = self._canonicalize_abbr(abbr)
                aliases = [abbr] if abbr != canonical else []
                
                # Infer semantic type or check for identifiers
                id_kind = self._is_identifier_abbr(abbr_raw)
                if id_kind:
                    semantic_type = 'identifier'
                    meta_ctx['id_kind'] = id_kind
                else:
                    # Use lexicon semantic types if available
                    if source == 'lexicon' and 'semantic_types' in lex_metadata:
                        semantic_type = lex_metadata['semantic_types'][0] if lex_metadata['semantic_types'] else None
                        meta_ctx.update(lex_metadata)
                    else:
                        semantic_type = sem_ctx or self._infer_type(canonical, expansion)
                
                # Adjust confidence based on source
                if source == 'override':
                    confidence = 0.80
                elif source == 'lexicon':
                    confidence = 0.70  # Lower confidence for lexicon-only
                else:
                    confidence = 0.60
                
                candidate = AbbreviationCandidate(
                    abbreviation=canonical,
                    expansion=expansion,
                    confidence=confidence,
                    source=source,
                    detection_source='standalone',
                    context=context,
                    position=match.start(),
                    page_number=page_number,
                    validation_status=source + '_definition' if source else 'unknown',
                    semantic_type=semantic_type,
                    metadata={'aliases': aliases, **meta_ctx}
                )
                candidates.append(candidate)
                
                if source == 'override':
                    self.stats['overrides_applied'] += 1
        
        return candidates
    
    def detect_patterns(self, text: str, page_number: int = -1) -> List[AbbreviationCandidate]:
        """
        Main detection method - processes text to find all abbreviations.
        Returns a list of AbbreviationCandidate objects for compatibility.
        
        Args:
            text: Input text to process
            page_number: Optional page number for tracking
            
        Returns:
            List of AbbreviationCandidate objects
        """
        self.stats['total_processed'] += 1
        
        # Pre-normalize text
        text = self._pre_normalize_text(text)
        
        # Track seen abbreviations to avoid duplicates
        seen_positions = set()  # (abbr, position) tuples
        seen_abbrevs = set()    # Just abbreviations
        
        all_candidates = []
        
        # Process each pattern type
        candidates = self._process_definition_pattern(text, seen_positions, page_number)
        all_candidates.extend(candidates)
        seen_abbrevs.update(c.abbreviation for c in candidates)
        
        candidates = self._process_reverse_pattern(text, seen_positions, page_number)
        all_candidates.extend(candidates)
        seen_abbrevs.update(c.abbreviation for c in candidates)
        
        candidates = self._process_colon_dash_pattern(text, seen_positions, page_number)
        all_candidates.extend(candidates)
        seen_abbrevs.update(c.abbreviation for c in candidates)
        
        # Find standalone abbreviations with known expansions
        standalone = self._find_standalone_abbreviations(text, seen_abbrevs, page_number)
        all_candidates.extend(standalone)
        
        # Deduplicate by abbreviation, keeping highest confidence
        abbrev_map = {}
        for candidate in all_candidates:
            abbr = candidate.abbreviation
            if abbr not in abbrev_map or candidate.confidence > abbrev_map[abbr].confidence:
                abbrev_map[abbr] = candidate
        
        # Sort by first occurrence
        final_candidates = sorted(abbrev_map.values(), key=lambda x: x.position)
        
        # Count occurrences for each abbreviation
        for c in final_candidates:
            c.metadata['count'] = self._count_occurrences(text, c.abbreviation)
        
        self.stats['abbreviations_found'] = len(final_candidates)
        
        # Log summary
        logger.info(f"Extracted {len(final_candidates)} abbreviations from text")
        if self.stats['overrides_applied'] > 0:
            logger.info(f"Applied {self.stats['overrides_applied']} overrides")
        if self.use_lexicon:
            logger.info(f"Lexicon: {self.stats['lexicon_resolved']} verified, "
                       f"{self.stats['lexicon_enriched']} enriched")
        
        return final_candidates
    
    def format_output(self, results: List[AbbreviationCandidate]) -> str:
        """
        Format extraction results as a ranked table with type, count, and confidence.
        
        Args:
            results: List of AbbreviationCandidate objects
            
        Returns:
            Formatted string table
        """
        lines = []
        lines.append("Rank  Abbrev       Type                 Count  Conf   Source      Expansion")
        lines.append("-" * 120)
        
        for i, c in enumerate(results, 1):
            # Get semantic type, truncate if needed
            t = c.semantic_type or 'general'
            if len(t) > 18:
                t = t[:18] + '..'
            
            # Get count
            cnt = c.metadata.get('count', 1)
            
            # Truncate expansion if too long
            exp = c.expansion
            if len(exp) > 45:
                exp = exp[:42] + '...'
            
            lines.append(
                f"{i:<5} {c.abbreviation:<12} {t:<20} {cnt:<6} {c.confidence:<6.2f} {c.source:<10} {exp}"
            )
        
        return "\n".join(lines)
