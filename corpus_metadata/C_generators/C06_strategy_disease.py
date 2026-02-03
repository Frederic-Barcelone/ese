"""
Disease mention detection using lexicons and NER.

This module detects disease names in clinical documents using a multi-layered
approach combining specialized lexicons, general disease vocabularies, and
biomedical NER. Features confidence-based false positive filtering to balance
precision and recall across different document types.

Key Components:
    - DiseaseDetector: Main detector combining multiple detection strategies
    - Lexicon layers:
        - Specialized disease lexicons (PAH, ANCA, IgAN) - highest precision
        - General disease lexicon (29K+ diseases) - with FP filtering
        - Orphanet rare diseases (9.6K diseases) - with FP filtering
    - scispacy NER with UMLS disease semantic types
    - DiseaseFalsePositiveFilter: Confidence-based filtering (C24)

Example:
    >>> from C_generators.C06_strategy_disease import DiseaseDetector
    >>> detector = DiseaseDetector(config={"lexicon_base_path": "lexicons/"})
    >>> candidates = detector.detect(doc_graph, "doc_123", "fingerprint")
    >>> for c in candidates:
    ...     print(f"{c.canonical_name} (confidence: {c.confidence:.2f})")
    pulmonary arterial hypertension (confidence: 0.95)

Dependencies:
    - A_core.A01_domain_models: Coordinate
    - A_core.A03_provenance: Provenance tracking utilities
    - A_core.A05_disease_models: DiseaseCandidate, DiseaseFieldType, DiseaseGeneratorType
    - A_core.A15_domain_profile: Domain-specific configuration
    - B_parsing.B02_doc_graph: DocumentGraph
    - B_parsing.B05_section_detector: Section classification
    - B_parsing.B06_confidence: Confidence scoring
    - B_parsing.B07_negation: Negation detection
    - C_generators.C24_disease_fp_filter: False positive filtering
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from flashtext import KeywordProcessor

from A_core.A01_domain_models import Coordinate

logger = logging.getLogger(__name__)
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from A_core.A05_disease_models import (
    DiseaseCandidate,
    DiseaseFieldType,
    DiseaseGeneratorType,
    DiseaseIdentifier,
    DiseaseProvenanceMetadata,
)
from A_core.A15_domain_profile import load_domain_profile
from B_parsing.B02_doc_graph import DocumentGraph
from B_parsing.B05_section_detector import SectionDetector
from B_parsing.B06_confidence import ConfidenceCalculator
from B_parsing.B07_negation import NegationDetector

from .C24_disease_fp_filter import DiseaseFalsePositiveFilter

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
    spacy = None
    SCISPACY_AVAILABLE = False


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
                "ouput_datasources",
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
        self.mondo_path = lexicon_base / "2025_mondo_diseases.json"
        self.rare_disease_acronyms_path = (
            lexicon_base / "2025_08_rare_disease_acronyms.json"
        )

        # Context window for snippets
        self.context_window = int(self.config.get("context_window", 300))

        # Shared parsing utilities from B_parsing
        self.section_detector = SectionDetector()
        self.negation_detector = NegationDetector()
        self.confidence_calculator = ConfidenceCalculator()

        # Load domain profile for confidence adjustments
        # Check disease_detection.domain_profile first, then domain_profile.active
        profile_name = self.config.get("domain_profile")
        if profile_name is None:
            # Try to get from nested config structure
            domain_profile_cfg = self.config.get("domain_profile_config", {})
            profile_name = domain_profile_cfg.get("active", "generic")
        self.domain_profile = load_domain_profile(profile_name, self.config)

        # FP filter with domain profile
        self.fp_filter = DiseaseFalsePositiveFilter(domain_profile=self.domain_profile)

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

        # Lexicon toggles (from disease_detection config section)
        self._enable_general = bool(self.config.get("enable_general_lexicon", True))
        self._enable_orphanet = bool(self.config.get("enable_orphanet", True))
        self._enable_mondo = bool(self.config.get("enable_mondo", True))
        self._enable_acronyms = bool(self.config.get("enable_rare_disease_acronyms", True))
        self._enable_scispacy = bool(self.config.get("enable_scispacy", True))

        # Stats: (name, count, filename)
        self._lexicon_stats: List[Tuple[str, int, str]] = []

        # Load lexicons (specialized always loaded; others gated by config)
        self._load_specialized_lexicons()
        if self._enable_general:
            self._load_general_lexicon()
        if self._enable_orphanet:
            self._load_orphanet_lexicon()
        if self._enable_mondo:
            self._load_mondo_lexicon()
        if self._enable_acronyms:
            self._load_rare_disease_acronyms()

        # Initialize scispacy
        self.scispacy_nlp = None
        self.umls_linker = None
        if self._enable_scispacy:
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

                # Register preferred label, abbreviation, and synonyms for FlashText
                self.specialized_kp.add_keyword(entry.preferred_label, key)
                if entry.abbreviation and len(entry.abbreviation) >= 2:
                    self.specialized_kp.add_keyword(entry.abbreviation, key)
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

    def _load_mondo_lexicon(self) -> None:
        """Load MONDO unified disease ontology lexicon.

        MONDO contains ~97K disease entries with synonyms and cross-references
        to MESH, UMLS, ICD, SNOMED, NCIT, etc.  Same format as the general
        disease lexicon (label / sources / synonyms) but much broader coverage
        including common diseases like "stroke" and "autoimmune disorder" that
        are absent from the rare-disease-focused lexicons.
        """
        if not self.mondo_path.exists():
            return

        data = json.loads(self.mondo_path.read_text(encoding="utf-8"))
        loaded = 0

        for entry_data in data:
            if not isinstance(entry_data, dict):
                continue

            label = (entry_data.get("label") or "").strip()
            if not label or len(label) < 3:
                continue

            if self._looks_like_chromosome(label):
                continue

            key = f"mondo_{loaded}"
            identifiers: Dict[str, str] = {}

            sources = entry_data.get("sources", [])
            for src in sources:
                if isinstance(src, dict):
                    src_name = src.get("source", "")
                    src_id = src.get("id", "")
                    if src_name and src_id:
                        identifiers[src_name] = src_id

            synonyms_raw = entry_data.get("synonyms", [])
            synonyms = [
                s for s in synonyms_raw
                if isinstance(s, str) and len(s) >= 3
                and not self._looks_like_chromosome(s)
            ]

            entry = DiseaseEntry(
                key=key,
                preferred_label=label,
                synonyms=synonyms,
                identifiers=identifiers,
                is_rare_disease=False,
                source=self.mondo_path.name,
            )

            self.general_entries[key] = entry
            self.general_kp.add_keyword(label, key)
            for syn in synonyms:
                self.general_kp.add_keyword(syn, key)

            loaded += 1

        self._lexicon_stats.append(
            ("MONDO diseases", loaded, self.mondo_path.name)
        )

    def _load_rare_disease_acronyms(self) -> None:
        """Load rare disease acronyms into general keyword processor.

        Loads acronym-to-disease mappings from the rare disease acronyms file.
        Each acronym (e.g., "APS") maps to a full disease name with Orphanet
        and ICD codes, enabling detection of disease abbreviations in text.
        """
        if not self.rare_disease_acronyms_path.exists():
            return

        data = json.loads(
            self.rare_disease_acronyms_path.read_text(encoding="utf-8")
        )
        loaded = 0

        for acronym, entry_data in data.items():
            if not isinstance(entry_data, dict):
                continue

            name = (entry_data.get("name") or "").strip()
            if not name:
                continue

            # Skip very short acronyms (single char) to avoid noise
            if len(acronym) < 2:
                continue

            key = f"acronym_{acronym}"

            identifiers: Dict[str, str] = {}
            orphacode = entry_data.get("orphacode")
            if orphacode:
                identifiers["ORPHA"] = str(orphacode)
            icd10 = entry_data.get("icd10_code")
            if icd10:
                identifiers["ICD10"] = icd10
            icd11 = entry_data.get("icd11_code")
            if icd11:
                identifiers["ICD11"] = icd11

            entry = DiseaseEntry(
                key=key,
                preferred_label=name,
                abbreviation=acronym,
                identifiers=identifiers,
                is_rare_disease=True,
                source=self.rare_disease_acronyms_path.name,
            )

            self.general_entries[key] = entry
            self.general_kp.add_keyword(acronym, key)
            loaded += 1

        self._lexicon_stats.append(
            ("Rare disease acronyms", loaded, self.rare_disease_acronyms_path.name)
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
            assert self.scispacy_nlp is not None  # Type narrowing for mypy
            try:
                self.scispacy_nlp.add_pipe(
                    "scispacy_linker",
                    config={"resolve_abbreviations": True, "linker_name": "umls"},
                )
                self.umls_linker = self.scispacy_nlp.get_pipe("scispacy_linker")
                logger.debug("Disease detector: loaded scispacy %s + UMLS linker", model_name)
                self._lexicon_stats.append(("scispacy NER", 1, model_name))
            except Exception as e:
                logger.debug("Disease detector: loaded scispacy %s (no UMLS: %s)", model_name, e)
                self._lexicon_stats.append(("scispacy NER", 1, model_name))
        except OSError as e:
            logger.debug("Disease detector: scispacy not available: %s", e)

    def _print_summary(self) -> None:
        """Print loading summary grouped by category."""
        if not self._lexicon_stats:
            return

        total = sum(count for _, count, _ in self._lexicon_stats)
        logger.info("Disease lexicons: %d sources, %d entries", len(self._lexicon_stats), total)
        logger.info("  Disease (%d entries)", total)

        for name, count, filename in self._lexicon_stats:
            # Clean up display name
            display_name = name.replace("Specialized ", "")
            logger.debug("    • %-26s %8d  %s", display_name, count, filename)

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

        # FlashText returns (key, start, end), not (matched_text, start, end)
        hits = self.specialized_kp.extract_keywords(text, span_info=True)
        for key, start, end in hits:
            if not key or key not in self.specialized_entries:
                continue

            # Get actual matched text from original text using span positions
            matched_text = text[start:end]
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
        """
        Extract from general lexicons with confidence-based scoring.

        CHANGED: Uses confidence adjustments instead of hard filtering.
        Only catastrophic FPs are hard-filtered; everything else gets
        a confidence adjustment that downstream components can use.
        """
        candidates = []

        # FlashText returns (key, start, end), not (matched_text, start, end)
        hits = self.general_kp.extract_keywords(text, span_info=True)
        for key, start, end in hits:
            if not key or key not in self.general_entries:
                continue

            # Get actual matched text from original text using span positions
            matched_text = text[start:end]
            entry = self.general_entries[key]
            context = self._make_context(text, start, end)

            # Hard filter only catastrophic FPs (chromosomes, strong gene context)
            should_filter, reason = self.fp_filter.should_filter(
                matched_text, context, is_abbreviation=False
            )
            if should_filter:
                continue

            # Also hard-filter by preferred_label for catastrophic FPs
            should_filter_label, _ = self.fp_filter.should_filter(
                entry.preferred_label, context, is_abbreviation=False
            )
            if should_filter_label:
                continue

            # Calculate confidence adjustment (replaces most filtering)
            adjustment, _ = self.fp_filter.score_adjustment(
                matched_text, context, is_abbreviation=False
            )

            # Skip if adjustment is extremely negative (very likely FP)
            # but less strict than hard filtering
            if adjustment < -0.5:
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

            # Apply confidence adjustment to initial confidence
            base_confidence = 0.85
            adjusted_confidence = max(0.1, min(1.0, base_confidence + adjustment))

            candidates.append(
                self._make_candidate(
                    doc=doc,
                    block=block,
                    matched_text=matched_text,
                    entry=entry,
                    context=context,
                    generator_type=gen_type,
                    initial_confidence=adjusted_confidence,
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
        candidates: list[DiseaseCandidate] = []

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

                # Hard filter only catastrophic FPs
                should_filter, _ = self.fp_filter.should_filter(
                    ent_text, context, is_abbreviation=False
                )
                if should_filter:
                    continue

                should_filter_label, _ = self.fp_filter.should_filter(
                    preferred_label, context, is_abbreviation=False
                )
                if should_filter_label:
                    continue

                # Calculate confidence adjustment
                adjustment, _ = self.fp_filter.score_adjustment(
                    ent_text, context, is_abbreviation=False
                )

                # Skip if extremely negative
                if adjustment < -0.5:
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

                # Apply adjustment to NER confidence score
                adjusted_score = max(0.1, min(1.0, score + adjustment))

                candidates.append(
                    self._make_candidate(
                        doc=doc,
                        block=block,
                        matched_text=ent_text,
                        entry=entry,
                        context=context,
                        generator_type=DiseaseGeneratorType.SCISPACY_NER,
                        field_type=DiseaseFieldType.NER_DETECTION,
                        initial_confidence=adjusted_score,
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
