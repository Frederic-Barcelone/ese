"""
Drug and chemical entity detection using lexicons and NER.

This module detects drug and chemical entity names in clinical documents using
a multi-layered approach with prioritized lexicon matching. Combines specialized
drug databases, FDA-approved drugs, RxNorm vocabulary, and scispacy NER with
false positive filtering for high precision.

Key Components:
    - DrugDetector: Main detector combining multiple detection strategies
    - Lexicon layers (in priority order):
        1. Alexion drugs (specialized, highest priority)
        2. Investigational drugs (compound IDs + lexicon)
        3. FDA approved drugs (brand + generic)
        4. RxNorm general terms
        5. scispacy NER (CHEMICAL semantic type, fallback)
    - DrugFalsePositiveFilter: Filters common false positives (C25)

Example:
    >>> from C_generators.C07_strategy_drug import DrugDetector
    >>> detector = DrugDetector(config={"lexicon_base_path": "lexicons/"})
    >>> candidates = detector.detect(doc_graph, "doc_123", "fingerprint")
    >>> for c in candidates:
    ...     print(f"{c.canonical_name} (source: {c.source})")
    ravulizumab (source: alexion_drugs)

Dependencies:
    - A_core.A01_domain_models: Coordinate
    - A_core.A03_provenance: Provenance tracking utilities
    - A_core.A06_drug_models: DrugCandidate, DrugFieldType, DrugGeneratorType
    - B_parsing.B01_pdf_to_docgraph: DocumentGraph
    - B_parsing.B05_section_detector: Section classification
    - B_parsing.B06_confidence: Confidence scoring
    - B_parsing.B07_negation: Negation detection
    - C_generators.C25_drug_fp_filter: False positive filtering
    - flashtext: KeywordProcessor for fast lexicon matching
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from E_normalization.E10_biomedical_ner_all import BiomedicalNERResult

from flashtext import KeywordProcessor

from A_core.A01_domain_models import Coordinate

logger = logging.getLogger(__name__)
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from A_core.A06_drug_models import (
    DrugCandidate,
    DrugFieldType,
    DrugGeneratorType,
    DrugIdentifier,
    DrugProvenanceMetadata,
)
from B_parsing.B01_pdf_to_docgraph import DocumentGraph
from B_parsing.B05_section_detector import SectionDetector
from B_parsing.B06_confidence import ConfidenceCalculator
from B_parsing.B07_negation import NegationDetector

# Import false positive filter and abbreviations from modularized file
from C_generators.C25_drug_fp_filter import (
    DRUG_ABBREVIATIONS,
    DrugFalsePositiveFilter,
)
from C_generators.C26_drug_fp_constants import (
    CONSUMER_DRUG_PATTERNS,
    CONSUMER_DRUG_VARIANTS,
)
from Z_utils.Z12_data_loader import load_term_set

# Optional scispacy import
try:
    import spacy
    from scispacy.linking import EntityLinker  # noqa: F401

    SCISPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SCISPACY_AVAILABLE = False


# -------------------------
# Drug Detector
# -------------------------


class DrugDetector:
    """
    Multi-layered drug mention detection.

    Layers (in priority order):
    1. Alexion drugs (specialized, auto-validated)
    2. Investigational drugs (compound IDs + lexicon)
    3. FDA approved drugs
    4. RxNorm general terms
    5. scispacy NER (fallback)
    """

    # Terms to skip when loading lexicons (prevent indexing obvious false positives)
    # These are filtered at load time for efficiency - no runtime overhead
    # Loaded from G_config/data/drug_fp_terms.yaml
    LEXICON_LOAD_BLACKLIST: Set[str] = load_term_set("drug_fp_terms.yaml", "lexicon_load_blacklist")

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.run_id = str(self.config.get("run_id") or generate_run_id("DRUG"))
        self.pipeline_version = (
            self.config.get("pipeline_version") or get_git_revision_hash()
        )
        self.doc_fingerprint_default = (
            self.config.get("doc_fingerprint") or "unknown-doc-fingerprint"
        )

        # Context window for evidence extraction
        self.context_window = int(self.config.get("context_window", 300))

        # Shared parsing utilities from B_parsing
        self.section_detector = SectionDetector()
        self.negation_detector = NegationDetector()
        self.confidence_calculator = ConfidenceCalculator()

        # Lexicon base path
        self.lexicon_base_path = Path(
            self.config.get("lexicon_base_path", "ouput_datasources")
        )

        # Initialize FlashText processors
        self.alexion_processor: Optional[KeywordProcessor] = None
        self.investigational_processor: Optional[KeywordProcessor] = None
        self.fda_processor: Optional[KeywordProcessor] = None
        self.rxnorm_processor: Optional[KeywordProcessor] = None
        self.consumer_processor: Optional[KeywordProcessor] = None
        self.bioactive_processor: Optional[KeywordProcessor] = None

        # Drug metadata dictionaries
        self.alexion_drugs: Dict[str, Dict] = {}
        self.investigational_drugs: Dict[str, Dict] = {}
        self.fda_drugs: Dict[str, Dict] = {}
        self.rxnorm_drugs: Dict[str, Dict] = {}
        self.consumer_drugs: Dict[str, Dict] = {}
        self.bioactive_drugs: Dict[str, Dict] = {}

        # Lexicon loading stats (for summary output)
        self._lexicon_stats: List[Tuple[str, int, str]] = []

        # Load lexicons
        self._load_lexicons()

        # Compound ID patterns for investigational drugs
        self.compound_patterns = [
            re.compile(r"\b([A-Z]{2,4})[-]?(\d{3,6})\b"),  # LNP023, BMS-986278
            re.compile(r"\b([A-Z]{2,4})[-]?([A-Z]?\d{4,})\b"),  # ABT199, GS-9973
            re.compile(r"\b(ALXN\d{3,6})\b", re.IGNORECASE),  # ALXN1720
        ]

        # False positive filter
        allow_bioactive = bool(self.config.get("allow_bioactive_compounds", False))
        self.fp_filter = DrugFalsePositiveFilter(
            allow_bioactive_compounds=allow_bioactive,
        )

        # scispacy NER model
        self.nlp = None
        if SCISPACY_AVAILABLE:
            self._init_scispacy()

    def _load_lexicons(self) -> None:
        """Load all drug lexicons."""
        self._load_alexion_lexicon()
        self._load_investigational_lexicon()
        self._load_fda_lexicon()
        self._load_rxnorm_lexicon()
        self._load_consumer_variants()
        if self.config.get("allow_bioactive_compounds", False):
            self._load_bioactive_compounds()

    def _load_alexion_lexicon(self) -> None:
        """Load Alexion specialized drug lexicon."""
        path = self.lexicon_base_path / "2025_08_alexion_drugs.json"
        if not path.exists():
            logger.warning("Alexion lexicon not found: %s", path)
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            alexion_proc = KeywordProcessor(case_sensitive=False)
            self.alexion_processor = alexion_proc
            known_drugs = data.get("known_drugs", {})
            drug_types = data.get("drug_types", {})

            for drug_name, drug_info in known_drugs.items():
                # Store metadata
                self.alexion_drugs[drug_name.lower()] = {
                    "preferred_name": drug_name,
                    "info": drug_info,
                    "drug_type": drug_types.get(drug_name, {}),
                    "source": "alexion",
                }
                # Add to FlashText
                alexion_proc.add_keyword(drug_name, drug_name.lower())

                # Add variations (brand names, compound IDs if available)
                if isinstance(drug_info, dict):
                    for alias in drug_info.get("aliases", []):
                        alexion_proc.add_keyword(alias, drug_name.lower())

            self._lexicon_stats.append(
                ("Alexion drugs", len(known_drugs), "2025_08_alexion_drugs.json")
            )

        except Exception as e:
            logger.warning("Failed to load Alexion lexicon: %s", e)

    def _load_investigational_lexicon(self) -> None:
        """Load investigational drugs from ClinicalTrials.gov data."""
        path = self.lexicon_base_path / "2025_08_investigational_drugs.json"
        if not path.exists():
            logger.warning("Investigational lexicon not found: %s", path)
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            inv_proc = KeywordProcessor(case_sensitive=False)
            self.investigational_processor = inv_proc
            count = 0

            skipped = 0
            for entry in data:
                drug_name = entry.get("interventionName", "").strip()
                if not drug_name or len(drug_name) < 3:
                    continue

                # Skip non-drug interventions
                if entry.get("interventionType") != "DRUG":
                    continue

                drug_key = drug_name.lower()

                # Skip blacklisted terms at load time (efficiency)
                if drug_key in self.LEXICON_LOAD_BLACKLIST:
                    skipped += 1
                    continue

                if drug_key not in self.investigational_drugs:
                    self.investigational_drugs[drug_key] = {
                        "preferred_name": drug_name,
                        "nct_id": entry.get("nctId"),
                        "conditions": entry.get("conditions", []),
                        "status": entry.get("overallStatus"),
                        "title": entry.get("title"),
                        "source": "investigational",
                    }
                    inv_proc.add_keyword(drug_name, drug_key)
                    count += 1

            if skipped > 0:
                logger.debug("Skipped %d blacklisted investigational terms", skipped)
            self._lexicon_stats.append(
                ("Investigational drugs", count, "2025_08_investigational_drugs.json")
            )

        except Exception as e:
            logger.warning("Failed to load investigational lexicon: %s", e)

    def _load_fda_lexicon(self) -> None:
        """Load FDA approved drugs lexicon."""
        path = self.lexicon_base_path / "2025_08_fda_approved_drugs.json"
        if not path.exists():
            logger.warning("FDA lexicon not found: %s", path)
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            fda_proc = KeywordProcessor(case_sensitive=False)
            self.fda_processor = fda_proc
            count = 0
            skipped = 0

            for entry in data:
                drug_name = entry.get("key", "").strip()
                if not drug_name or len(drug_name) < 3:
                    continue

                drug_key = drug_name.lower()

                # Skip blacklisted terms at load time (efficiency)
                if drug_key in self.LEXICON_LOAD_BLACKLIST:
                    skipped += 1
                    continue

                meta = entry.get("meta", {})

                if drug_key not in self.fda_drugs:
                    self.fda_drugs[drug_key] = {
                        "preferred_name": drug_name,
                        "brand_name": meta.get("brand_name"),
                        "drug_class": entry.get("drug_class"),
                        "dosage_form": meta.get("dosage_form"),
                        "route": meta.get("route"),
                        "marketing_status": meta.get("marketing_status"),
                        "application_number": meta.get("application_number"),
                        "source": "fda",
                    }
                    fda_proc.add_keyword(drug_name, drug_key)
                    count += 1

                    # Also add brand name if different
                    brand = meta.get("brand_name", "")
                    if brand and brand.lower() != drug_key:
                        brand_key = brand.lower()
                        if brand_key not in self.fda_drugs:
                            fda_proc.add_keyword(brand, drug_key)

            if skipped > 0:
                logger.debug("Skipped %d blacklisted FDA terms", skipped)
            self._lexicon_stats.append(
                ("FDA approved drugs", count, "2025_08_fda_approved_drugs.json")
            )

        except Exception as e:
            logger.warning("Failed to load FDA lexicon: %s", e)

    def _load_rxnorm_lexicon(self) -> None:
        """Load RxNorm general drug lexicon."""
        path = self.lexicon_base_path / "2025_08_lexicon_drug.json"
        if not path.exists():
            logger.warning("RxNorm lexicon not found: %s", path)
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            rxnorm_proc = KeywordProcessor(case_sensitive=False)
            self.rxnorm_processor = rxnorm_proc
            count = 0

            skipped = 0
            for entry in data:
                term = entry.get("term", "").strip()
                if not term or len(term) < 3:
                    continue

                term_key = term.lower()

                # Skip blacklisted terms at load time (efficiency)
                if term_key in self.LEXICON_LOAD_BLACKLIST:
                    skipped += 1
                    continue

                if term_key not in self.rxnorm_drugs:
                    self.rxnorm_drugs[term_key] = {
                        "preferred_name": term,
                        "term_normalized": entry.get("term_normalized"),
                        "rxcui": entry.get("rxcui"),
                        "tty": entry.get("tty"),
                        "source": "rxnorm",
                    }
                    rxnorm_proc.add_keyword(term, term_key)
                    count += 1

            if skipped > 0:
                logger.debug("Skipped %d blacklisted RxNorm terms", skipped)
            self._lexicon_stats.append(
                ("RxNorm terms", count, "2025_08_lexicon_drug.json")
            )

        except Exception as e:
            logger.warning("Failed to load RxNorm lexicon: %s", e)

    def _load_consumer_variants(self) -> None:
        """Load consumer drug misspellings and multi-word patterns."""
        consumer_proc = KeywordProcessor(case_sensitive=False)
        count = 0

        # Add misspelling → canonical mappings
        for variant, canonical in CONSUMER_DRUG_VARIANTS.items():
            canonical_key = canonical.lower()
            if canonical_key not in self.consumer_drugs:
                self.consumer_drugs[canonical_key] = {
                    "preferred_name": canonical.title(),
                    "source": "consumer_variant",
                }
            consumer_proc.add_keyword(variant, canonical_key)
            count += 1

        # Add multi-word consumer patterns
        for pattern in CONSUMER_DRUG_PATTERNS:
            pattern_key = pattern.lower()
            if pattern_key not in self.consumer_drugs:
                self.consumer_drugs[pattern_key] = {
                    "preferred_name": pattern.title(),
                    "source": "consumer_pattern",
                }
            consumer_proc.add_keyword(pattern, pattern_key)
            count += 1

        self.consumer_processor = consumer_proc
        self._lexicon_stats.append(
            ("Consumer variants", count, "C26_drug_fp_constants.py")
        )

    def _load_bioactive_compounds(self) -> None:
        """Load bioactive compounds as detectable drug keywords.

        Only loaded when allow_bioactive_compounds=True (e.g., for BC5CDR).
        These compounds (dopamine, calcium, etc.) are valid pharmaceutical
        agents that are normally filtered as biological entities.
        """
        from C_generators.C25_drug_fp_filter import DrugFalsePositiveFilter

        bio_proc = KeywordProcessor(case_sensitive=False)
        count = 0
        for compound in DrugFalsePositiveFilter.BIOACTIVE_DRUG_COMPOUNDS:
            key = compound.lower()
            if key not in self.bioactive_drugs:
                self.bioactive_drugs[key] = {
                    "preferred_name": compound.title(),
                    "source": "bioactive_compound",
                }
            bio_proc.add_keyword(compound, key)
            count += 1

        self.bioactive_processor = bio_proc
        self._lexicon_stats.append(
            ("Bioactive compounds", count, "C25_drug_fp_filter.py")
        )

    def _init_scispacy(self) -> None:
        """Initialize scispacy NER model."""
        if not SCISPACY_AVAILABLE or spacy is None:
            return

        try:
            # Try large model first, fall back to small
            try:
                self.nlp = spacy.load("en_core_sci_lg")
            except OSError:
                self.nlp = spacy.load("en_core_sci_sm")

            # Add UMLS linker for chemical entities
            assert self.nlp is not None  # Type narrowing for mypy
            if "scispacy_linker" not in self.nlp.pipe_names:
                self.nlp.add_pipe(
                    "scispacy_linker",
                    config={
                        "resolve_abbreviations": True,
                        "linker_name": "umls",
                        "threshold": 0.7,
                    },
                )
            self._lexicon_stats.append(("scispacy NER", 1, "en_core_sci_lg"))

        except Exception as e:
            logger.warning("Failed to initialize scispacy for drugs: %s", e)
            self.nlp = None

    def _print_lexicon_summary(self) -> None:
        """Print compact summary of loaded drug lexicons."""
        if not self._lexicon_stats:
            return

        # All drug lexicons go under "Drug" category
        total = sum(count for _, count, _ in self._lexicon_stats if count > 1)
        file_count = len([s for s in self._lexicon_stats if s[1] > 0])
        logger.info("Drug lexicons: %d sources, %d entries", file_count, total)
        logger.info("  Drug (%d entries)", total)

        for name, count, filename in self._lexicon_stats:
            if count > 1:
                logger.debug("    • %-26s %8d  %s", name, count, filename)
            else:
                logger.debug("    • %-26s %8s  %s", name, "enabled", filename)

    def detect(
        self,
        doc_graph: DocumentGraph,
        biomedical_ner_result: Optional["BiomedicalNERResult"] = None,
    ) -> List[DrugCandidate]:
        """
        Detect drug mentions in document.

        Args:
            doc_graph: Parsed document graph
            biomedical_ner_result: Optional BiomedicalNERResult from E10

        Returns list of DrugCandidate objects.
        """
        candidates: List[DrugCandidate] = []
        doc_fingerprint = getattr(
            doc_graph, "fingerprint", self.doc_fingerprint_default
        )

        # Get full text for detection by concatenating all blocks
        full_text = "\n\n".join(
            block.text for block in doc_graph.iter_linear_blocks(skip_header_footer=True)
            if block.text
        )

        # Layer 1: Alexion drugs (specialized, highest priority)
        if self.alexion_processor:
            candidates.extend(
                self._detect_with_lexicon(
                    full_text,
                    doc_graph,
                    doc_fingerprint,
                    self.alexion_processor,
                    self.alexion_drugs,
                    DrugGeneratorType.LEXICON_ALEXION,
                    "2025_08_alexion_drugs.json",
                )
            )

        # Layer 2: Compound ID patterns
        candidates.extend(
            self._detect_compound_patterns(full_text, doc_graph, doc_fingerprint)
        )

        # Layer 3: Investigational drugs
        if self.investigational_processor:
            candidates.extend(
                self._detect_with_lexicon(
                    full_text,
                    doc_graph,
                    doc_fingerprint,
                    self.investigational_processor,
                    self.investigational_drugs,
                    DrugGeneratorType.LEXICON_INVESTIGATIONAL,
                    "2025_08_investigational_drugs.json",
                )
            )

        # Layer 4: FDA approved drugs
        if self.fda_processor:
            candidates.extend(
                self._detect_with_lexicon(
                    full_text,
                    doc_graph,
                    doc_fingerprint,
                    self.fda_processor,
                    self.fda_drugs,
                    DrugGeneratorType.LEXICON_FDA,
                    "2025_08_fda_approved_drugs.json",
                )
            )

        # Layer 5: RxNorm general (more selective)
        if self.rxnorm_processor:
            candidates.extend(
                self._detect_with_lexicon(
                    full_text,
                    doc_graph,
                    doc_fingerprint,
                    self.rxnorm_processor,
                    self.rxnorm_drugs,
                    DrugGeneratorType.LEXICON_RXNORM,
                    "2025_08_lexicon_drug.json",
                )
            )

        # Layer 6: Consumer drug variants (misspellings + multi-word patterns)
        if self.consumer_processor:
            candidates.extend(
                self._detect_with_lexicon(
                    full_text,
                    doc_graph,
                    doc_fingerprint,
                    self.consumer_processor,
                    self.consumer_drugs,
                    DrugGeneratorType.LEXICON_RXNORM,  # Treat as general lexicon
                    "consumer_variants",
                )
            )

        # Layer 7: Bioactive compounds (only when enabled)
        if self.bioactive_processor:
            candidates.extend(
                self._detect_with_lexicon(
                    full_text,
                    doc_graph,
                    doc_fingerprint,
                    self.bioactive_processor,
                    self.bioactive_drugs,
                    DrugGeneratorType.LEXICON_RXNORM,  # Treat as general lexicon
                    "bioactive_compounds",
                )
            )

        # Layer 8: scispacy NER fallback
        if self.nlp:
            candidates.extend(
                self._detect_with_ner(full_text, doc_graph, doc_fingerprint)
            )

        # Layer 9: BiomedNER-All (document-level gap-filler, lowest priority)
        if biomedical_ner_result is not None:
            candidates.extend(
                self._detect_biomedical_ner(
                    biomedical_ner_result, doc_graph, doc_fingerprint
                )
            )

        # Deduplicate
        candidates = self._deduplicate(candidates)

        return candidates

    def _detect_with_lexicon(
        self,
        text: str,
        doc_graph: DocumentGraph,
        doc_fingerprint: str,
        processor: KeywordProcessor,
        drug_dict: Dict[str, Dict],
        generator_type: DrugGeneratorType,
        lexicon_source: str,
    ) -> List[DrugCandidate]:
        """Detect drugs using FlashText lexicon matching."""
        candidates = []

        # Extract keywords with positions
        matches = processor.extract_keywords(text, span_info=True)

        for keyword, start, end in matches:
            drug_info = drug_dict.get(keyword, {})
            if not drug_info:
                continue

            matched_text = text[start:end]

            # Apply false positive filter
            context = self._extract_context(text, start, end)
            if self.fp_filter.is_false_positive(matched_text, context, generator_type):
                continue

            # Build identifiers
            identifiers = self._build_identifiers(drug_info)

            # Determine if investigational
            is_investigational = generator_type in {
                DrugGeneratorType.LEXICON_ALEXION,
                DrugGeneratorType.LEXICON_INVESTIGATIONAL,
                DrugGeneratorType.PATTERN_COMPOUND_ID,
            }

            # Use canonical name if the preferred_name is a known abbreviation
            raw_preferred = drug_info.get("preferred_name", matched_text)
            canonical_name = DRUG_ABBREVIATIONS.get(raw_preferred.lower())
            final_preferred = canonical_name.title() if canonical_name else raw_preferred

            candidate = DrugCandidate(
                doc_id=doc_graph.doc_id,
                matched_text=matched_text,
                preferred_name=final_preferred,
                brand_name=drug_info.get("brand_name"),
                compound_id=drug_info.get("compound_id"),
                field_type=DrugFieldType.EXACT_MATCH,
                generator_type=generator_type,
                identifiers=identifiers,
                context_text=context,
                context_location=Coordinate(page_num=1),  # Simplified
                drug_class=drug_info.get("drug_class"),
                mechanism=drug_info.get("mechanism"),
                development_phase=drug_info.get("status"),
                is_investigational=is_investigational,
                sponsor=drug_info.get("sponsor"),
                conditions=drug_info.get("conditions", []),
                nct_id=drug_info.get("nct_id"),
                dosage_form=drug_info.get("dosage_form"),
                route=drug_info.get("route"),
                marketing_status=drug_info.get("marketing_status"),
                initial_confidence=0.85 if is_investigational else 0.7,
                provenance=DrugProvenanceMetadata(
                    pipeline_version=self.pipeline_version,
                    run_id=self.run_id,
                    doc_fingerprint=doc_fingerprint,
                    generator_name=generator_type,
                    lexicon_source=lexicon_source,
                ),
            )
            candidates.append(candidate)

        return candidates

    def _detect_compound_patterns(
        self,
        text: str,
        doc_graph: DocumentGraph,
        doc_fingerprint: str,
    ) -> List[DrugCandidate]:
        """Detect compound IDs using regex patterns."""
        candidates = []
        seen_positions: Set[Tuple[int, int]] = set()

        for pattern in self.compound_patterns:
            for match in pattern.finditer(text):
                start, end = match.span()

                # Skip if already matched
                if (start, end) in seen_positions:
                    continue
                seen_positions.add((start, end))

                matched_text = match.group(0)
                context = self._extract_context(text, start, end)

                # Check if this matches a known investigational drug
                drug_info = self.investigational_drugs.get(matched_text.lower(), {})
                if not drug_info:
                    # Create basic entry for unknown compound
                    drug_info = {
                        "preferred_name": matched_text,
                        "compound_id": matched_text,
                    }

                # Apply false positive filter for compound patterns too
                if self.fp_filter.is_false_positive(
                    matched_text, context, DrugGeneratorType.PATTERN_COMPOUND_ID
                ):
                    continue

                conditions_raw = drug_info.get("conditions", [])
                conditions_list: List[str] = (
                    conditions_raw if isinstance(conditions_raw, list) else []
                )
                candidate = DrugCandidate(
                    doc_id=doc_graph.doc_id,
                    matched_text=matched_text,
                    preferred_name=drug_info.get("preferred_name", matched_text),
                    compound_id=matched_text,
                    field_type=DrugFieldType.PATTERN_MATCH,
                    generator_type=DrugGeneratorType.PATTERN_COMPOUND_ID,
                    identifiers=[],
                    context_text=context,
                    context_location=Coordinate(page_num=1),
                    is_investigational=True,
                    conditions=conditions_list,
                    nct_id=drug_info.get("nct_id"),
                    initial_confidence=0.8,
                    provenance=DrugProvenanceMetadata(
                        pipeline_version=self.pipeline_version,
                        run_id=self.run_id,
                        doc_fingerprint=doc_fingerprint,
                        generator_name=DrugGeneratorType.PATTERN_COMPOUND_ID,
                        lexicon_source="pattern:compound_id",
                    ),
                )
                candidates.append(candidate)

        return candidates

    def _detect_with_ner(
        self,
        text: str,
        doc_graph: DocumentGraph,
        doc_fingerprint: str,
    ) -> List[DrugCandidate]:
        """Detect drugs using scispacy NER."""
        candidates: list[DrugCandidate] = []

        if not self.nlp:
            return candidates

        # Process text (limit to avoid memory issues)
        max_chars = 100000
        if len(text) > max_chars:
            text = text[:max_chars]

        try:
            doc = self.nlp(text)

            # CHEMICAL semantic type in UMLS
            CHEMICAL_TYPES = {"T109", "T116", "T121", "T123", "T195", "T200"}

            for ent in doc.ents:
                # Check if entity has UMLS linking
                if not hasattr(ent, "_") or not hasattr(ent._, "kb_ents"):
                    continue

                kb_ents = ent._.kb_ents
                if not kb_ents:
                    continue

                # Get best UMLS match
                best_cui, best_score = kb_ents[0]
                if best_score < 0.7:
                    continue

                # Check semantic type - use cui_to_entity API
                linker = self.nlp.get_pipe("scispacy_linker")
                entity_info = linker.kb.cui_to_entity.get(best_cui)
                if entity_info is None:
                    continue

                types = set(entity_info.types)

                # Only keep chemical entities
                if not types.intersection(CHEMICAL_TYPES):
                    continue

                matched_text = ent.text
                context = self._extract_context(text, ent.start_char, ent.end_char)

                # Skip if too short or common word
                if self.fp_filter.is_false_positive(
                    matched_text, context, DrugGeneratorType.SCISPACY_NER
                ):
                    continue

                # Also check the UMLS canonical name (preferred_name) against blacklist
                # This catches cases where matched_text is "MuSK" but UMLS returns
                # "Musk secretion from Musk Deer" which is a false positive
                canonical_name = entity_info.canonical_name or matched_text
                if self.fp_filter.is_false_positive(
                    canonical_name, context, DrugGeneratorType.SCISPACY_NER
                ):
                    continue

                # Additional check: if canonical_name contains known FP substrings
                canonical_lower = canonical_name.lower()
                skip_entity = False
                for fp_substr in self.fp_filter.fp_substrings_lower:
                    if fp_substr in canonical_lower:
                        skip_entity = True
                        break
                # Check for musk deer specifically (common UMLS mislinking)
                if "musk deer" in canonical_lower or "musk secretion" in canonical_lower:
                    skip_entity = True
                if skip_entity:
                    continue

                candidate = DrugCandidate(
                    doc_id=doc_graph.doc_id,
                    matched_text=matched_text,
                    preferred_name=entity_info.canonical_name or matched_text,
                    field_type=DrugFieldType.NER_DETECTION,
                    generator_type=DrugGeneratorType.SCISPACY_NER,
                    identifiers=[
                        DrugIdentifier(
                            system="UMLS_CUI",
                            code=best_cui,
                            display=entity_info.canonical_name,
                        )
                    ],
                    context_text=context,
                    context_location=Coordinate(page_num=1),
                    initial_confidence=best_score,
                    provenance=DrugProvenanceMetadata(
                        pipeline_version=self.pipeline_version,
                        run_id=self.run_id,
                        doc_fingerprint=doc_fingerprint,
                        generator_name=DrugGeneratorType.SCISPACY_NER,
                        lexicon_source="ner:scispacy_umls",
                    ),
                )
                candidates.append(candidate)

        except Exception as e:
            logger.warning("scispacy drug detection error: %s", e)

        return candidates

    def _detect_biomedical_ner(
        self,
        ner_result: "BiomedicalNERResult",
        doc_graph: DocumentGraph,
        doc_fingerprint: str,
    ) -> List[DrugCandidate]:
        """Detect drugs from BiomedNER-All (d4data) Medication entities.

        Lexicon-verified gap-filler: only accepts BiomedNER entities that
        also match a drug lexicon as a full-term match. This prevents
        subword/fragment FPs from the BERT tokenizer while recovering drugs
        that were missed by the block-level lexicon scan.
        """
        candidates: list[DrugCandidate] = []

        try:
            medication_entities = ner_result.get_by_type("Medication")
            if not medication_entities:
                return candidates

            for entity in medication_entities:
                matched_text = entity.text.strip()

                # Skip very short or very long
                if len(matched_text) < 4 or len(matched_text) > 100:
                    continue

                # Require minimum confidence
                if entity.score < 0.7:
                    continue

                # Skip fragments with special characters (BERT tokenization artifacts)
                if re.search(r"[)\[\]{}<>]", matched_text):
                    continue

                # Require lexicon verification — only accept if the entity
                # matches a known drug in our lexicons
                match = self.is_known_drug(matched_text)
                if match is None:
                    continue

                lexicon_name, lexicon_gen_type = match

                context = matched_text

                # FP filter
                if self.fp_filter.is_false_positive(
                    matched_text, context, DrugGeneratorType.BIOMEDICAL_NER
                ):
                    continue

                candidate = DrugCandidate(
                    doc_id=doc_graph.doc_id,
                    matched_text=matched_text,
                    preferred_name=lexicon_name,
                    field_type=DrugFieldType.NER_DETECTION,
                    generator_type=DrugGeneratorType.BIOMEDICAL_NER,
                    identifiers=[],
                    context_text=context,
                    context_location=Coordinate(page_num=1),
                    initial_confidence=entity.score,
                    provenance=DrugProvenanceMetadata(
                        pipeline_version=self.pipeline_version,
                        run_id=self.run_id,
                        doc_fingerprint=doc_fingerprint,
                        generator_name=DrugGeneratorType.BIOMEDICAL_NER,
                        lexicon_source="ner:biomedical_ner_all",
                    ),
                )
                candidates.append(candidate)

        except Exception as e:
            logger.warning("BiomedNER drug detection error: %s", e)

        return candidates

    def _extract_context(self, text: str, start: int, end: int) -> str:
        """Extract context around a match."""
        ctx_start = max(0, start - self.context_window // 2)
        ctx_end = min(len(text), end + self.context_window // 2)
        return text[ctx_start:ctx_end]

    def _build_identifiers(self, drug_info: Dict) -> List[DrugIdentifier]:
        """Build identifier list from drug info."""
        identifiers = []

        if drug_info.get("rxcui"):
            identifiers.append(
                DrugIdentifier(system="RxCUI", code=str(drug_info["rxcui"]))
            )
        if drug_info.get("nct_id"):
            identifiers.append(DrugIdentifier(system="NCT", code=drug_info["nct_id"]))
        if drug_info.get("application_number"):
            identifiers.append(
                DrugIdentifier(system="FDA_NDA", code=drug_info["application_number"])
            )

        return identifiers

    def _get_lexicon_processors(
        self,
    ) -> List[Tuple[KeywordProcessor, DrugGeneratorType]]:
        """Return loaded FlashText processors in priority order."""
        processors: List[Tuple[KeywordProcessor, DrugGeneratorType]] = []
        if self.alexion_processor:
            processors.append((self.alexion_processor, DrugGeneratorType.LEXICON_ALEXION))
        if self.investigational_processor:
            processors.append((self.investigational_processor, DrugGeneratorType.LEXICON_INVESTIGATIONAL))
        if self.fda_processor:
            processors.append((self.fda_processor, DrugGeneratorType.LEXICON_FDA))
        if self.rxnorm_processor:
            processors.append((self.rxnorm_processor, DrugGeneratorType.LEXICON_RXNORM))
        if self.consumer_processor:
            processors.append((self.consumer_processor, DrugGeneratorType.LEXICON_RXNORM))
        if self.bioactive_processor:
            processors.append((self.bioactive_processor, DrugGeneratorType.LEXICON_RXNORM))
        return processors

    def is_known_drug(self, term: str) -> Optional[Tuple[str, DrugGeneratorType]]:
        """Check if a term matches any drug lexicon as a full-term match.

        Only returns a match when the lexicon keyword covers the entire term
        (not just a substring). Returns (matched_name, generator_type) or None.
        """
        term_lower = term.lower().strip()
        if not term_lower or len(term_lower) < 3:
            return None
        term_len = len(term_lower)
        for processor, gen_type in self._get_lexicon_processors():
            matches = processor.extract_keywords(term_lower, span_info=True)
            for keyword, start, end in matches:
                # Require the match to cover the entire term
                if start == 0 and end == term_len:
                    return (keyword, gen_type)
        return None

    def is_known_drug_substring(self, term: str) -> Optional[Tuple[str, DrugGeneratorType]]:
        """Check if a term contains a known drug as a substring match.

        Returns the longest (matched_name, generator_type) found within the term,
        or None. Requires the matched key to be at least 6 characters.
        """
        term_lower = term.lower().strip()
        if not term_lower or len(term_lower) < 6:
            return None
        best: Optional[Tuple[str, DrugGeneratorType, int]] = None
        for processor, gen_type in self._get_lexicon_processors():
            matches = processor.extract_keywords(term_lower, span_info=True)
            for keyword, _start, _end in matches:
                kw_len = len(keyword)
                if kw_len < 6:
                    continue
                if best is None or kw_len > best[2]:
                    best = (keyword, gen_type, kw_len)
        if best is not None:
            return (best[0], best[1])
        return None

    def _deduplicate(self, candidates: List[DrugCandidate]) -> List[DrugCandidate]:
        """
        Deduplicate candidates, preferring specialized sources.

        Priority: Alexion > Investigational > FDA > RxNorm > NER
        Deduplicates by both matched_text AND preferred_name.
        Also links common abbreviations to their full forms (e.g., MTX = methotrexate).
        """
        # Priority order
        priority = {
            DrugGeneratorType.LEXICON_ALEXION: 0,
            DrugGeneratorType.PATTERN_COMPOUND_ID: 1,
            DrugGeneratorType.LEXICON_INVESTIGATIONAL: 2,
            DrugGeneratorType.LEXICON_FDA: 3,
            DrugGeneratorType.LEXICON_RXNORM: 4,
            DrugGeneratorType.SCISPACY_NER: 5,
            DrugGeneratorType.BIOMEDICAL_NER: 6,
        }

        # Sort all candidates by priority first
        candidates.sort(key=lambda c: priority.get(c.generator_type, 99))

        # Track seen names (both matched_text and preferred_name)
        seen_names: Set[str] = set()
        deduped: List[DrugCandidate] = []

        def get_canonical_name(name: str) -> Optional[str]:
            """Get canonical drug name if input is a known abbreviation."""
            name_lower = name.lower().strip()
            return DRUG_ABBREVIATIONS.get(name_lower)

        def is_seen(name: str) -> bool:
            """Check if name or its canonical form has been seen."""
            name_lower = name.lower().strip()
            if name_lower in seen_names:
                return True
            # Check if this is an abbreviation whose canonical form is seen
            canonical = get_canonical_name(name_lower)
            if canonical and canonical in seen_names:
                return True
            return False

        def mark_as_seen(name: str) -> None:
            """Mark name and its canonical form as seen."""
            name_lower = name.lower().strip()
            seen_names.add(name_lower)
            # Also add canonical form if this is an abbreviation
            canonical = get_canonical_name(name_lower)
            if canonical:
                seen_names.add(canonical)

        for c in candidates:
            matched_key = c.matched_text.lower().strip()
            preferred_key = (c.preferred_name or "").lower().strip()

            # Skip if we've seen this name already (including abbreviation links)
            if is_seen(matched_key) or (preferred_key and is_seen(preferred_key)):
                continue

            # Add to result and mark as seen
            deduped.append(c)
            mark_as_seen(matched_key)
            if preferred_key:
                mark_as_seen(preferred_key)

        return deduped
