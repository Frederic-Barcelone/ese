# corpus_metadata/corpus_metadata/C_generators/C06_strategy_disease.py
"""
Disease mention detection strategy.

Detects disease names in documents using:
1. Specialized disease lexicons (PAH, ANCA, IgAN) - high precision
2. General disease lexicon (29K+ diseases) - with FP filtering
3. Orphanet rare diseases (9.6K diseases) - with FP filtering
4. scispacy NER with UMLS disease types

Includes multi-layer false positive filtering to avoid:
- Chromosome patterns (10p, 46,XX, etc.)
- Gene names used as genes (not disease abbreviations)
- Short matches without disease context
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from flashtext import KeywordProcessor

from A_core.A01_domain_models import Coordinate
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from A_core.A05_disease_models import (
    DiseaseCandidate,
    DiseaseFieldType,
    DiseaseGeneratorType,
    DiseaseIdentifier,
    DiseaseProvenanceMetadata,
)
from B_parsing.B02_doc_graph import DocumentGraph
from B_parsing.B05_section_detector import SectionDetector
from B_parsing.B06_confidence import ConfidenceCalculator
from B_parsing.B07_negation import NegationDetector

# scispacy for biomedical NER
try:
    import spacy
    from scispacy.abbreviation import AbbreviationDetector  # noqa: F401
    from scispacy.linking import EntityLinker  # noqa: F401

    SCISPACY_AVAILABLE = True

    # Suppress scispacy warning about empty matcher patterns
    warnings.filterwarnings(
        "ignore",
        message=r".*The component 'matcher' does not have any patterns defined.*",
        category=UserWarning,
        module="scispacy.abbreviation",
    )
except ImportError:
    spacy = None  # type: ignore
    SCISPACY_AVAILABLE = False


# =============================================================================
# FALSE POSITIVE FILTER
# =============================================================================


class DiseaseFalsePositiveFilter:
    """
    Multi-layer filtering to avoid chromosome/gene false positives.

    The problem: Disease lexicons contain entries like:
    - "10p Deletion Syndrome" -> matches chromosome "10p"
    - "45,X syndrome" -> matches karyotype "45,X"
    - "Chromosome 22q11.2 deletion" -> matches "22q11.2"

    Also filters:
    - Physiological systems (RAAS, RAS) that are not diseases
    - Journal name abbreviations that match disease names
    - Abbreviations that are too ambiguous
    """

    # Physiological systems and pathways (not diseases)
    PHYSIOLOGICAL_SYSTEMS: Set[str] = {
        # Renin-angiotensin-aldosterone system
        "raas",
        "ras",
        "renin-angiotensin system",
        "renin-angiotensin-aldosterone system",
        "renin angiotensin system",
        "renin angiotensin aldosterone system",
        # Other physiological systems
        "hpa axis",
        "hypothalamic-pituitary-adrenal axis",
        "sns",
        "sympathetic nervous system",
        "pns",
        "parasympathetic nervous system",
        "cns",
        "central nervous system",
        "ans",
        "autonomic nervous system",
        "immune system",
        "complement system",
        "coagulation cascade",
        "kinin system",
        "kallikrein-kinin system",
    }

    # Journal name patterns and abbreviations (not diseases)
    JOURNAL_PATTERNS: Set[str] = {
        # Nephrology journals
        "adv chronic kidney dis",
        "advances in chronic kidney disease",
        "kidney int",
        "kidney international",
        "j am soc nephrol",
        "jasn",
        "clin j am soc nephrol",
        "cjasn",
        "nephrol dial transplant",
        "ndt",
        "am j kidney dis",
        "ajkd",
        # General medical journals
        "n engl j med",
        "nejm",
        "lancet",
        "the lancet",
        "jama",
        "bmj",
        "ann intern med",
        "j clin invest",
        "nat med",
        "nature medicine",
        "cell",
        "science",
        "plos one",
        "plos med",
        # Other specialty journals
        "blood",
        "circulation",
        "j immunol",
        "j biol chem",
    }

    # Layer 0: Generic/overly broad terms that are not specific diseases
    GENERIC_TERMS: Set[str] = {
        # Too generic - categories, not specific diseases
        "disease",
        "diseases",
        "syndrome",
        "syndromes",
        "disorder",
        "disorders",
        "condition",
        "conditions",
        "rare diseases",
        "rare disease",
        "orphan disease",
        "orphan diseases",
        "genetic disease",
        "genetic diseases",
        "hereditary disease",
        "hereditary diseases",
        "communicable diseases",
        "infectious disease",
        "infectious diseases",
        "chronic disease",
        "chronic diseases",
        "autoimmune disease",
        "autoimmune diseases",
        "metabolic disease",
        "metabolic diseases",
        "neurological disease",
        "neurological diseases",
        # Generic anatomical terms
        "neoplasm",
        "neoplasms",
        "malignant neoplasms",
        "benign neoplasms",
        "tumor",
        "tumors",
        "cancer",
        "cancers",
        "carcinoma",
        "sarcoma",
        "lymphoma",
        "leukemia",
        # Generic process terms
        "agenesis",
        "aplasia",
        "hypoplasia",
        "hyperplasia",
        "atrophy",
        "hypertrophy",
        "inflammation",
        "infection",
        "deficiency",
        "insufficiency",
        # Other overly generic
        "abnormality",
        "abnormalities",
        "anomaly",
        "anomalies",
        "malformation",
        "malformations",
        "deformity",
        "deformities",
        # Veterinary/animal diseases (FP in human context)
        "newcastle disease",
        # Mental/behavioral (too generic)
        "mental blocking",
        "learning disabilities",
        # Clinical status terms (not diseases)
        "progressive disease",
        "kidney diseases",
        "kidney failure",
        "renal glomerular disease",
        # Generic infection terms
        "infections",
        "pneumonia",
        "meningitis",
        "nephritis",
        # Symptoms/signs, not diseases
        "ascites",
        "hypertensive disease",
        "neoplasm metastasis",
        # Too broad categories
        "complement deficiencies",
        "infections of musculoskeletal system",
        "infection due to encapsulated bacteria",
        # Symptoms/signs that are too generic
        "confusion",
        "erythema",
        "paresis",
        # Behavioral (not diseases in clinical context)
        "firesetting behavior",
        "drug abuse",
        # Genetic/molecular terms (not diseases)
        "transition mutation",
        # Laboratory artifacts or model organisms
        "sarcoma, yoshida",
        "yoshida sarcoma",
        # Anatomical variants (not diseases)
        "short forearm",
        "cavitation",
        # Too generic symptoms
        "mental depression",
    }

    # Layer 1: Chromosome/karyotype patterns to block
    CHROMOSOME_PATTERNS = [
        r"^\d{1,2}[pq]$",  # 10p, 22q, etc.
        r"^\d{1,2}[pq]\d+",  # 10p15, 22q11
        r"^\d{1,2}[pq]\d+\.\d+",  # 22q11.2
        r"^4[0-9],X{1,2}Y?$",  # 45,X, 46,XX, 46,XY
        r"^del\(\d+[pq]?\)",  # del(7q), del(5q)
        r"^t\(\d+;\d+\)",  # t(9;22), t(4;14)
        r"^inv\(\d+\)",  # inv(16)
        r"^dup\(\d+\)",  # dup(7)
        r"^\+\d{1,2}$",  # +21, +13 (trisomy notation)
        r"^-\d{1,2}$",  # -7, -5 (monosomy notation)
    ]

    # Layer 2: Context keywords for disambiguation
    CHROMOSOME_CONTEXT_KEYWORDS = [
        "chromosome",
        "karyotype",
        "cytogenetic",
        "translocation",
        "deletion",
        "duplication",
        "trisomy",
        "monosomy",
        "band",
        "breakpoint",
        "FISH",
        "CGH",
        "array",
        "copy number",
        "ploidy",
        "aneuploidy",
        "mosaicism",
    ]

    DISEASE_CONTEXT_KEYWORDS = [
        "syndrome",
        "disease",
        "disorder",
        "condition",
        "patient",
        "diagnosis",
        "diagnosed",
        "treatment",
        "therapy",
        "symptom",
        "clinical",
        "prognosis",
        "affected",
        "prevalence",
        "incidence",
        "rare",
        "orphan",
        "trial",
        "study",
    ]

    # Layer 3: Gene name patterns (genes used as genes, not diseases)
    GENE_PATTERN = r"^[A-Z][A-Z0-9]{1,6}$"  # BRCA1, TP53, EGFR, etc.

    GENE_CONTEXT_KEYWORDS = [
        "mutation",
        "variant",
        "expression",
        "gene",
        "protein",
        "encoded",
        "pathway",
        "receptor",
        "kinase",
        "transcription",
        "allele",
        "polymorphism",
        "genotype",
    ]

    # Short match threshold - matches <= this length need disease context
    SHORT_MATCH_THRESHOLD = 4

    def __init__(self):
        self._compiled_chr_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.CHROMOSOME_PATTERNS
        ]
        self._gene_pattern = re.compile(self.GENE_PATTERN)
        self._generic_terms_lower = {t.lower() for t in self.GENERIC_TERMS}
        self._physiological_systems_lower = {t.lower() for t in self.PHYSIOLOGICAL_SYSTEMS}
        self._journal_patterns_lower = {t.lower() for t in self.JOURNAL_PATTERNS}

    def should_filter(
        self, matched_text: str, context: str, is_abbreviation: bool = False
    ) -> Tuple[bool, str]:
        """
        Determine if a match should be filtered out.

        Returns:
            (should_filter, reason)
        """
        matched_clean = matched_text.strip()
        matched_lower = matched_clean.lower()
        ctx_lower = context.lower()

        # Layer 0: Filter generic/overly broad terms
        if matched_lower in self._generic_terms_lower:
            return True, "generic_term"

        # Filter physiological systems (RAAS, RAS, etc.)
        if matched_lower in self._physiological_systems_lower:
            return True, "physiological_system"

        # Filter journal names and abbreviations
        if matched_lower in self._journal_patterns_lower:
            return True, "journal_name"

        # Check if the matched text is part of a journal citation
        if self._is_journal_citation_context(matched_lower, ctx_lower):
            return True, "journal_citation_context"

        # Layer 1: Check chromosome patterns
        for pattern in self._compiled_chr_patterns:
            if pattern.match(matched_clean):
                # Check if context suggests disease vs chromosome
                if self._is_chromosome_context(ctx_lower):
                    return True, "chromosome_pattern_in_chromosome_context"

        # Layer 2: Short matches need disease context
        if len(matched_clean) <= self.SHORT_MATCH_THRESHOLD and not is_abbreviation:
            if not self._has_disease_context(ctx_lower):
                return True, "short_match_no_disease_context"

        # Layer 3: Check if it's a gene name used as gene (not disease)
        if self._is_gene_as_gene(matched_clean, ctx_lower):
            return True, "gene_name_not_disease"

        return False, ""

    def _is_chromosome_context(self, ctx_lower: str) -> bool:
        """Check if context suggests chromosome/cytogenetic usage."""
        chr_score = sum(1 for kw in self.CHROMOSOME_CONTEXT_KEYWORDS if kw in ctx_lower)
        dis_score = sum(1 for kw in self.DISEASE_CONTEXT_KEYWORDS if kw in ctx_lower)
        return chr_score > dis_score

    def _has_disease_context(self, ctx_lower: str) -> bool:
        """Check if context contains disease-related keywords."""
        return any(kw in ctx_lower for kw in self.DISEASE_CONTEXT_KEYWORDS)

    def _is_gene_as_gene(self, matched_text: str, ctx_lower: str) -> bool:
        """Check if text is a gene name being used as a gene (not disease abbreviation)."""
        if not self._gene_pattern.match(matched_text):
            return False

        # Count gene vs disease context keywords
        gene_score = sum(1 for kw in self.GENE_CONTEXT_KEYWORDS if kw in ctx_lower)
        dis_score = sum(1 for kw in self.DISEASE_CONTEXT_KEYWORDS if kw in ctx_lower)

        # If strong gene context and weak disease context, filter it
        return gene_score >= 2 and gene_score > dis_score

    def _is_journal_citation_context(self, matched_lower: str, ctx_lower: str) -> bool:
        """
        Check if the match appears in a journal citation context.

        This catches cases like "Adv Chronic Kidney Dis" where a disease name
        is actually a journal abbreviation in a reference/citation.

        Args:
            matched_lower: The matched text (lowercase).
            ctx_lower: The context around the match (lowercase).

        Returns:
            True if the match is likely in a citation context.
        """
        # Citation indicators
        citation_indicators = [
            # Volume/issue patterns
            r"\d{4};\s*\d+",  # year; volume
            r"vol\.\s*\d+",   # vol. number
            r"pp?\.\s*\d+",   # p. or pp. page numbers
            r"doi:",          # DOI reference
            r"pmid:",         # PubMed ID
            r"\[\d+\]",       # Reference numbers like [1], [23]
            # Journal context words
            "published in",
            "et al",
            "authors",
            "reference",
            "citation",
            "bibliography",
        ]

        for indicator in citation_indicators:
            if indicator in ctx_lower:
                return True

        # Check for common citation patterns near the match
        # Pattern: "Journal Name Year;Volume:Pages"
        if re.search(r"\d{4}\s*;\s*\d+\s*:\s*\d+", ctx_lower):
            return True

        return False


# =============================================================================
# DISEASE LEXICON ENTRY
# =============================================================================


class DiseaseEntry:
    """Loaded disease entry from lexicon."""

    __slots__ = (
        "key",
        "preferred_label",
        "abbreviation",
        "synonyms",
        "patterns",
        "identifiers",
        "context_keywords",
        "exclude_contexts",
        "is_rare_disease",
        "prevalence",
        "confidence_boost",
        "parent",
        "source",
    )

    def __init__(
        self,
        key: str,
        preferred_label: str,
        abbreviation: Optional[str] = None,
        synonyms: Optional[List[str]] = None,
        patterns: Optional[List[str]] = None,
        identifiers: Optional[Dict[str, str]] = None,
        context_keywords: Optional[List[str]] = None,
        exclude_contexts: Optional[List[str]] = None,
        is_rare_disease: bool = False,
        prevalence: Optional[str] = None,
        confidence_boost: float = 0.0,
        parent: Optional[str] = None,
        source: str = "unknown",
    ):
        self.key = key
        self.preferred_label = preferred_label
        self.abbreviation = abbreviation
        self.synonyms = synonyms or []
        self.patterns = patterns or []
        self.identifiers = identifiers or {}
        self.context_keywords = context_keywords or []
        self.exclude_contexts = exclude_contexts or []
        self.is_rare_disease = is_rare_disease
        self.prevalence = prevalence
        self.confidence_boost = confidence_boost
        self.parent = parent
        self.source = source


# =============================================================================
# DISEASE DETECTOR
# =============================================================================


class DiseaseDetector:
    """
    Disease mention detector.

    Detects diseases using multiple strategies:
    1. Specialized lexicons (PAH, ANCA, IgAN) - highest priority
    2. General disease lexicon - with FP filtering
    3. Orphanet rare diseases - with FP filtering
    4. scispacy NER - for unknown diseases
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

        # Lexicon paths
        lexicon_base = Path(
            self.config.get(
                "lexicon_base_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources",
            )
        )

        self.specialized_lexicons = {
            "pah": lexicon_base / "disease_lexicon_pah.json",
            "anca": lexicon_base / "disease_lexicon_anca.json",
            "igan": lexicon_base / "disease_lexicon_igan.json",
            "c3g": lexicon_base / "disease_lexicon_c3g.json",
        }

        self.general_disease_path = lexicon_base / "2025_08_lexicon_disease.json"
        self.orphanet_path = lexicon_base / "2025_08_orphanet_diseases.json"

        # Context window for snippets
        self.context_window = int(self.config.get("context_window", 300))

        # Shared parsing utilities from B_parsing
        self.section_detector = SectionDetector()
        self.negation_detector = NegationDetector()
        self.confidence_calculator = ConfidenceCalculator()

        # FP filter
        self.fp_filter = DiseaseFalsePositiveFilter()

        # Disease entries storage
        self.specialized_entries: Dict[str, DiseaseEntry] = {}  # key -> DiseaseEntry
        self.general_entries: Dict[str, DiseaseEntry] = {}

        # FlashText for fast matching
        self.specialized_kp = KeywordProcessor(case_sensitive=False)
        self.general_kp = KeywordProcessor(case_sensitive=False)

        # Compiled regex patterns
        self.specialized_patterns: List[Tuple[re.Pattern, DiseaseEntry]] = []
        self.general_patterns: List[Tuple[re.Pattern, DiseaseEntry]] = []

        # Provenance
        self.pipeline_version = str(
            self.config.get("pipeline_version") or get_git_revision_hash()
        )
        self.run_id = str(self.config.get("run_id") or generate_run_id("DIS"))
        self.doc_fingerprint_default = str(
            self.config.get("doc_fingerprint") or "unknown-doc-fingerprint"
        )

        # Stats: (name, count, filename)
        self._lexicon_stats: List[Tuple[str, int, str]] = []

        # Load lexicons
        self._load_specialized_lexicons()
        self._load_general_lexicon()
        self._load_orphanet_lexicon()

        # Initialize scispacy
        self.scispacy_nlp = None
        self.umls_linker = None
        self._init_scispacy()

    def _load_specialized_lexicons(self) -> None:
        """Load specialized disease lexicons (PAH, ANCA, IgAN)."""
        for name, path in self.specialized_lexicons.items():
            if not path.exists():
                continue

            data = json.loads(path.read_text(encoding="utf-8"))
            diseases = data.get("diseases", {})
            loaded = 0

            for key, entry_data in diseases.items():
                if not isinstance(entry_data, dict):
                    continue

                entry = self._parse_disease_entry(key, entry_data, path.name)
                if not entry:
                    continue

                self.specialized_entries[key] = entry

                # Register preferred label and synonyms for FlashText
                self.specialized_kp.add_keyword(entry.preferred_label, key)
                for syn in entry.synonyms:
                    if len(syn) >= 3:  # Skip very short synonyms
                        self.specialized_kp.add_keyword(syn, key)

                # Compile regex patterns
                for pattern_str in entry.patterns:
                    try:
                        pattern = re.compile(pattern_str, re.IGNORECASE)
                        self.specialized_patterns.append((pattern, entry))
                    except re.error:
                        pass

                loaded += 1

            self._lexicon_stats.append(
                (f"Specialized ({name.upper()})", loaded, path.name)
            )

    def _load_general_lexicon(self) -> None:
        """Load general disease lexicon with FP awareness."""
        if not self.general_disease_path.exists():
            return

        data = json.loads(self.general_disease_path.read_text(encoding="utf-8"))
        loaded = 0

        for entry_data in data:
            if not isinstance(entry_data, dict):
                continue

            label = (entry_data.get("label") or "").strip()
            if not label or len(label) < 3:
                continue

            # Skip entries that look like chromosome patterns
            if self._looks_like_chromosome(label):
                continue

            key = f"general_{loaded}"
            identifiers = {}

            # Extract MONDO ID if present
            sources = entry_data.get("sources", [])
            for src in sources:
                if isinstance(src, dict):
                    src_name = src.get("source", "")
                    src_id = src.get("id", "")
                    if src_name and src_id:
                        identifiers[src_name] = src_id

            entry = DiseaseEntry(
                key=key,
                preferred_label=label,
                identifiers=identifiers,
                is_rare_disease=False,
                source=self.general_disease_path.name,
            )

            self.general_entries[key] = entry
            self.general_kp.add_keyword(label, key)
            loaded += 1

        self._lexicon_stats.append(
            ("General diseases", loaded, self.general_disease_path.name)
        )

    def _load_orphanet_lexicon(self) -> None:
        """Load Orphanet rare disease lexicon."""
        if not self.orphanet_path.exists():
            return

        data = json.loads(self.orphanet_path.read_text(encoding="utf-8"))
        loaded = 0

        for entry_data in data:
            if not isinstance(entry_data, dict):
                continue

            name = (entry_data.get("name") or "").strip()
            if not name or len(name) < 3:
                continue

            # Skip entries that look like chromosome patterns
            if self._looks_like_chromosome(name):
                continue

            orphacode = entry_data.get("orphacode")
            synonyms = entry_data.get("synonyms", [])

            key = f"orphanet_{orphacode or loaded}"

            identifiers = {}
            if orphacode:
                identifiers["ORPHA"] = str(orphacode)

            entry = DiseaseEntry(
                key=key,
                preferred_label=name,
                synonyms=[s for s in synonyms if len(s) >= 3],
                identifiers=identifiers,
                is_rare_disease=True,
                source=self.orphanet_path.name,
            )

            self.general_entries[key] = entry
            self.general_kp.add_keyword(name, key)
            for syn in entry.synonyms:
                if not self._looks_like_chromosome(syn):
                    self.general_kp.add_keyword(syn, key)

            loaded += 1

        self._lexicon_stats.append(
            ("Orphanet diseases", loaded, self.orphanet_path.name)
        )

    def _looks_like_chromosome(self, text: str) -> bool:
        """Quick check if text looks like a chromosome/karyotype pattern."""
        text = text.strip()
        # Common chromosome patterns
        if re.match(r"^\d{1,2}[pq]\d*", text):
            return True
        if re.match(r"^4[0-9],X", text):
            return True
        if re.match(r"^(del|dup|inv|t)\(", text):
            return True
        # Pure numbers (chromosome numbers)
        if text.isdigit() and int(text) <= 50:
            return True
        return False

    def _parse_disease_entry(
        self, key: str, data: dict, source: str
    ) -> Optional[DiseaseEntry]:
        """Parse a disease entry from lexicon data."""
        preferred_label = (data.get("preferred_label") or "").strip()
        if not preferred_label:
            return None

        return DiseaseEntry(
            key=key,
            preferred_label=preferred_label,
            abbreviation=data.get("abbreviation"),
            synonyms=data.get("synonyms", []),
            patterns=data.get("patterns", []),
            identifiers=data.get("identifiers", {}),
            context_keywords=data.get("context_keywords", []),
            exclude_contexts=data.get("exclude_contexts", []),
            is_rare_disease=bool(data.get("rare_disease", False)),
            prevalence=data.get("prevalence"),
            confidence_boost=float(data.get("confidence_boost", 0.0)),
            parent=data.get("parent"),
            source=source,
        )

    def _init_scispacy(self) -> None:
        """Initialize scispacy NER with UMLS linker."""
        if not SCISPACY_AVAILABLE or spacy is None:
            return

        try:
            try:
                self.scispacy_nlp = spacy.load("en_core_sci_lg")
                model_name = "en_core_sci_lg"
            except OSError:
                self.scispacy_nlp = spacy.load("en_core_sci_sm")
                model_name = "en_core_sci_sm"

            # Add UMLS linker for disease identification
            try:
                self.scispacy_nlp.add_pipe(
                    "scispacy_linker",
                    config={"resolve_abbreviations": True, "linker_name": "umls"},
                )
                self.umls_linker = self.scispacy_nlp.get_pipe("scispacy_linker")
                print(f"  Disease detector: loaded scispacy {model_name} + UMLS linker")
                self._lexicon_stats.append(("scispacy NER", 1, model_name))
            except Exception as e:
                print(
                    f"  Disease detector: loaded scispacy {model_name} (no UMLS: {e})"
                )
                self._lexicon_stats.append(("scispacy NER", 1, model_name))
        except OSError as e:
            print(f"  Disease detector: scispacy not available: {e}")

    def _print_summary(self) -> None:
        """Print loading summary grouped by category."""
        if not self._lexicon_stats:
            return

        total = sum(count for _, count, _ in self._lexicon_stats)
        print(
            f"\nDisease lexicons: {len(self._lexicon_stats)} sources, {total:,} entries"
        )
        print("─" * 70)
        print(f"  Disease ({total:,} entries)")

        for name, count, filename in self._lexicon_stats:
            # Clean up display name
            display_name = name.replace("Specialized ", "")
            print(f"    • {display_name:<26} {count:>8,}  {filename}")
        print()

    def extract(self, doc_structure: DocumentGraph) -> List[DiseaseCandidate]:
        """
        Extract disease mentions from document.

        Args:
            doc_structure: Parsed document graph

        Returns:
            List of DiseaseCandidate objects
        """
        doc = doc_structure
        candidates: List[DiseaseCandidate] = []
        seen: Set[Tuple[str, str]] = (
            set()
        )  # (matched_text_lower, preferred_label_lower)

        for block in doc.iter_linear_blocks(skip_header_footer=True):
            text = (block.text or "").strip()
            if not text:
                continue

            # 1. Specialized lexicon matches (highest priority)
            candidates.extend(self._extract_specialized(text, block, doc, seen))

            # 2. General lexicon matches (with FP filtering)
            candidates.extend(self._extract_general(text, block, doc, seen))

            # 3. Regex pattern matches from specialized lexicons
            candidates.extend(self._extract_patterns(text, block, doc, seen))

            # 4. scispacy NER (for unknown diseases)
            if self.scispacy_nlp is not None:
                candidates.extend(self._extract_scispacy(text, block, doc, seen))

        # Final deduplication by preferred_name to avoid duplicates like "Sarcoma" x3
        return self._deduplicate_by_name(candidates)

    def _extract_specialized(
        self,
        text: str,
        block,
        doc: DocumentGraph,
        seen: Set[Tuple[str, str]],
    ) -> List[DiseaseCandidate]:
        """Extract from specialized lexicons (PAH, ANCA, IgAN)."""
        candidates = []

        hits = self.specialized_kp.extract_keywords(text, span_info=True)
        for matched_text, start, end in hits:
            key = self.specialized_kp.get_keyword(matched_text)
            if not key or key not in self.specialized_entries:
                continue

            entry = self.specialized_entries[key]

            # Check exclude contexts
            context = self._make_context(text, start, end)
            if self._should_exclude(context, entry.exclude_contexts):
                continue

            # Dedup
            dedup_key = (matched_text.lower(), entry.preferred_label.lower())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Build candidate
            candidates.append(
                self._make_candidate(
                    doc=doc,
                    block=block,
                    matched_text=matched_text,
                    entry=entry,
                    context=context,
                    generator_type=DiseaseGeneratorType.LEXICON_SPECIALIZED,
                )
            )

        return candidates

    def _extract_general(
        self,
        text: str,
        block,
        doc: DocumentGraph,
        seen: Set[Tuple[str, str]],
    ) -> List[DiseaseCandidate]:
        """Extract from general lexicons with FP filtering."""
        candidates = []

        hits = self.general_kp.extract_keywords(text, span_info=True)
        for matched_text, start, end in hits:
            key = self.general_kp.get_keyword(matched_text)
            if not key or key not in self.general_entries:
                continue

            entry = self.general_entries[key]
            context = self._make_context(text, start, end)

            # Apply FP filter to matched text
            should_filter, reason = self.fp_filter.should_filter(
                matched_text, context, is_abbreviation=False
            )
            if should_filter:
                continue

            # Also filter by preferred_label (lexicon entry might have generic name)
            should_filter_label, _ = self.fp_filter.should_filter(
                entry.preferred_label, context, is_abbreviation=False
            )
            if should_filter_label:
                continue

            # Dedup
            dedup_key = (matched_text.lower(), entry.preferred_label.lower())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            gen_type = (
                DiseaseGeneratorType.LEXICON_ORPHANET
                if entry.is_rare_disease
                else DiseaseGeneratorType.LEXICON_GENERAL
            )

            candidates.append(
                self._make_candidate(
                    doc=doc,
                    block=block,
                    matched_text=matched_text,
                    entry=entry,
                    context=context,
                    generator_type=gen_type,
                )
            )

        return candidates

    def _extract_patterns(
        self,
        text: str,
        block,
        doc: DocumentGraph,
        seen: Set[Tuple[str, str]],
    ) -> List[DiseaseCandidate]:
        """Extract using regex patterns from specialized lexicons."""
        candidates = []

        for pattern, entry in self.specialized_patterns:
            for match in pattern.finditer(text):
                matched_text = match.group(0)
                start, end = match.start(), match.end()
                context = self._make_context(text, start, end)

                # Check exclude contexts
                if self._should_exclude(context, entry.exclude_contexts):
                    continue

                # Dedup
                dedup_key = (matched_text.lower(), entry.preferred_label.lower())
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                candidates.append(
                    self._make_candidate(
                        doc=doc,
                        block=block,
                        matched_text=matched_text,
                        entry=entry,
                        context=context,
                        generator_type=DiseaseGeneratorType.LEXICON_SPECIALIZED,
                        field_type=DiseaseFieldType.PATTERN_MATCH,
                    )
                )

        return candidates

    def _extract_scispacy(
        self,
        text: str,
        block,
        doc: DocumentGraph,
        seen: Set[Tuple[str, str]],
    ) -> List[DiseaseCandidate]:
        """Extract using scispacy NER with UMLS linking."""
        candidates = []

        if self.scispacy_nlp is None:
            return candidates

        try:
            spacy_doc = self.scispacy_nlp(text)

            # Disease-related UMLS semantic types
            disease_semantic_types = {
                "T047",  # Disease or Syndrome
                "T048",  # Mental or Behavioral Dysfunction
                "T191",  # Neoplastic Process
                "T019",  # Congenital Abnormality
                "T190",  # Anatomical Abnormality
                "T049",  # Cell or Molecular Dysfunction
            }

            for ent in spacy_doc.ents:
                ent_text = ent.text.strip()

                # Skip very short or very long entities
                if len(ent_text) < 4 or len(ent_text) > 100:
                    continue

                # Check UMLS linking
                if not hasattr(ent._, "kb_ents") or not ent._.kb_ents:
                    continue

                top_match = ent._.kb_ents[0]
                cui = top_match[0]
                score = top_match[1]

                # Skip low confidence matches
                if score < 0.7:
                    continue

                # Check if it's a disease type
                kb_entry = None
                if self.umls_linker and hasattr(self.umls_linker, "kb"):
                    kb_entry = self.umls_linker.kb.cui_to_entity.get(cui)

                if not kb_entry:
                    continue

                # Check semantic types
                entity_types = set(kb_entry.types) if kb_entry.types else set()
                if not entity_types.intersection(disease_semantic_types):
                    continue

                preferred_label = kb_entry.canonical_name or ent_text
                context = self._make_context(text, ent.start_char, ent.end_char)

                # Apply FP filter to matched text
                should_filter, _ = self.fp_filter.should_filter(
                    ent_text, context, is_abbreviation=False
                )
                if should_filter:
                    continue

                # Also filter by preferred_label (UMLS canonical name might be generic)
                should_filter_label, _ = self.fp_filter.should_filter(
                    preferred_label, context, is_abbreviation=False
                )
                if should_filter_label:
                    continue

                # Dedup
                dedup_key = (ent_text.lower(), preferred_label.lower())
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                # Create entry for NER match
                entry = DiseaseEntry(
                    key=f"ner_{cui}",
                    preferred_label=preferred_label,
                    identifiers={"UMLS_CUI": cui},
                    source="scispacy_ner",
                )

                candidates.append(
                    self._make_candidate(
                        doc=doc,
                        block=block,
                        matched_text=ent_text,
                        entry=entry,
                        context=context,
                        generator_type=DiseaseGeneratorType.SCISPACY_NER,
                        field_type=DiseaseFieldType.NER_DETECTION,
                        initial_confidence=score,
                    )
                )

        except Exception:
            # Don't fail entire extraction if scispacy has issues
            pass

        return candidates

    def _make_candidate(
        self,
        doc: DocumentGraph,
        block,
        matched_text: str,
        entry: DiseaseEntry,
        context: str,
        generator_type: DiseaseGeneratorType,
        field_type: DiseaseFieldType = DiseaseFieldType.EXACT_MATCH,
        initial_confidence: float = 0.85,
    ) -> DiseaseCandidate:
        """Create a DiseaseCandidate from match data."""
        # Build identifiers list
        identifiers = self._build_identifiers(entry.identifiers)

        # Calculate confidence with boost
        confidence = min(1.0, initial_confidence + entry.confidence_boost)

        # Build provenance
        provenance = DiseaseProvenanceMetadata(
            pipeline_version=self.pipeline_version,
            run_id=self.run_id,
            doc_fingerprint=self.doc_fingerprint_default,
            generator_name=generator_type,
            rule_version="disease_v1.0",
            lexicon_source=entry.source,
            lexicon_ids=identifiers if identifiers else None,
        )

        # Build coordinate
        location = Coordinate(
            page_num=int(block.page_num),
            block_id=str(block.id),
            bbox=block.bbox,
        )

        return DiseaseCandidate(
            doc_id=doc.doc_id,
            matched_text=matched_text,
            preferred_label=entry.preferred_label,
            abbreviation=entry.abbreviation,
            synonyms=entry.synonyms,
            field_type=field_type,
            generator_type=generator_type,
            identifiers=identifiers,
            context_text=context,
            context_location=location,
            is_rare_disease=entry.is_rare_disease,
            prevalence=entry.prevalence,
            parent_disease=entry.parent,
            initial_confidence=confidence,
            confidence_boost=entry.confidence_boost,
            provenance=provenance,
        )

    def _build_identifiers(
        self, raw_identifiers: Dict[str, str]
    ) -> List[DiseaseIdentifier]:
        """Convert raw identifier dict to DiseaseIdentifier list."""
        identifiers = []

        # Map of raw keys to standardized system names
        system_map = {
            "ORPHA": "ORPHA",
            "ICD10": "ICD-10",
            "ICD10CM": "ICD-10-CM",
            "ICD11": "ICD-11",
            "SNOMED_CT": "SNOMED-CT",
            "UMLS_CUI": "UMLS",
            "MESH": "MeSH",
            "MONDO": "MONDO",
        }

        for raw_key, code in raw_identifiers.items():
            if not code or raw_key.endswith("_label") or raw_key.startswith("_"):
                continue

            system = system_map.get(raw_key, raw_key)

            # Format code properly
            if raw_key == "ORPHA" and not code.startswith("ORPHA:"):
                code = f"ORPHA:{code}"

            identifiers.append(DiseaseIdentifier(system=system, code=code))

        return identifiers

    def _make_context(self, text: str, start: int, end: int) -> str:
        """Create context snippet around match."""
        left = max(0, start - self.context_window)
        right = min(len(text), end + self.context_window)
        return text[left:right].replace("\n", " ").strip()

    def _should_exclude(self, context: str, exclude_contexts: List[str]) -> bool:
        """Check if context contains any exclude keywords."""
        if not exclude_contexts:
            return False
        ctx_lower = context.lower()
        return any(exc.lower() in ctx_lower for exc in exclude_contexts)

    def _deduplicate_by_name(
        self, candidates: List[DiseaseCandidate]
    ) -> List[DiseaseCandidate]:
        """
        Final deduplication by preferred_label to avoid duplicates.

        Keeps the highest confidence candidate for each unique preferred_label.
        """
        if not candidates:
            return candidates

        # Group by preferred_label (case-insensitive)
        by_name: Dict[str, List[DiseaseCandidate]] = {}
        for c in candidates:
            key = c.preferred_label.lower()
            if key not in by_name:
                by_name[key] = []
            by_name[key].append(c)

        # Keep highest confidence for each name
        deduped = []
        for name_key, group in by_name.items():
            # Sort by confidence descending, then by generator type priority
            group.sort(
                key=lambda x: (
                    -x.initial_confidence,
                    0 if x.generator_type == DiseaseGeneratorType.LEXICON_SPECIALIZED else 1,
                )
            )
            deduped.append(group[0])

        return deduped
