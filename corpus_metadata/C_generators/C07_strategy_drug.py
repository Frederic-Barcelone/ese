# corpus_metadata/corpus_metadata/C_generators/C07_strategy_drug.py
"""
Drug/chemical entity detection strategy.

Multi-layered approach:
1. Alexion drugs (specialized, highest priority)
2. Investigational drugs (compound IDs + lexicon)
3. FDA approved drugs (brand + generic)
4. RxNorm general terms
5. scispacy NER (CHEMICAL semantic type, fallback)

Uses FlashText for fast keyword matching.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from flashtext import KeywordProcessor

from A_core.A01_domain_models import Coordinate
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from A_core.A06_drug_models import (
    DrugCandidate,
    DrugFieldType,
    DrugGeneratorType,
    DrugIdentifier,
    DrugProvenanceMetadata,
)
from B_parsing.B01_pdf_to_docgraph import DocumentGraph

# Optional scispacy import
try:
    import spacy
    from scispacy.linking import EntityLinker  # noqa: F401

    SCISPACY_AVAILABLE = True
except ImportError:
    SCISPACY_AVAILABLE = False


# -------------------------
# Drug False Positive Filter
# -------------------------


class DrugFalsePositiveFilter:
    """Filter false positive drug matches."""

    # Common words that might match drug names
    COMMON_WORDS: Set[str] = {
        # Dosage forms
        "oral",
        "tablet",
        "capsule",
        "injection",
        "solution",
        "cream",
        "gel",
        "patch",
        "spray",
        "drops",
        # Generic substances
        "water",
        "salt",
        "acid",
        "base",
        "oil",
        "sugar",
        "fat",
        # Metals (often false positives)
        "iron",
        "gold",
        "silver",
        "lead",
        "zinc",
        # Drug-related terms
        "dose",
        "drug",
        "agent",
        "compound",
        "product",
        "formula",
        "active",
        "inactive",
        "medications",
        "medication",
        "medicine",
        "medicines",
        "treatment",
        "therapy",
        # Generic/vague terms
        "various",
        "other",
        "others",
        "complete",
        "unknown",
        "none",
        "same",
        "different",
        "several",
        "many",
        "some",
        "all",
        "any",
        # Journal/publication names
        "lancet",
        "nature",
        "science",
        "cell",
        # Generic process/action words
        "via",
        "met",
        "food",
        "duration",
        "support",
        "root",
        "process",
        "his",
        "central",
        "ensure",
        "blockade",
        "therapeutic",
        "monotherapy",
        "targeted therapy",
        "soc",
        # Biomolecules (not drugs per se)
        "protein",
        "creatinine",
        "glucose",
        "angiotensin",
        "aldosterone",
        "renin",
        "complement",
        "factor b",
        "serum",
        # Too generic drug terms
        "inhibitor",
        "inhibitors",
        "antagonist",
        "antagonists",
        "agonist",
        "agonists",
        "receptor",
        "receptors",
        "pharmaceutical preparations",
        "activation product",
        # Anatomical/biological structures
        "nephron",
        "membrane attack complex",
        "com",
        "importal",
    }

    # Body parts and organs (not drugs)
    BODY_PARTS: Set[str] = {
        "liver",
        "kidney",
        "heart",
        "lung",
        "brain",
        "blood",
        "bone",
        "skin",
        "muscle",
        "nerve",
        "eye",
        "ear",
        "stomach",
        "intestine",
        "colon",
        "bladder",
        "spleen",
        "pancreas",
        "thyroid",
        "adrenal",
        "ovary",
        "uterus",
        "prostate",
        "breast",
        "tongue",
        "teeth",
        "gum",
        "nail",
        "hair",
    }

    # Clinical trial status terms (leaked from trial data)
    TRIAL_STATUS_TERMS: Set[str] = {
        "not_yet_recruiting",
        "recruiting",
        "active",
        "completed",
        "suspended",
        "terminated",
        "withdrawn",
        "enrolling",
        "available",
        "approved",
        "no_longer_available",
        "withheld",
        "unknown",
        "not yet recruiting",
        "active, not recruiting",
        "enrolling by invitation",
    }

    # Medical equipment/procedures (not drugs)
    EQUIPMENT_PROCEDURES: Set[str] = {
        "ultrasound",
        "mri",
        "ct",
        "xray",
        "x-ray",
        "scan",
        "surgery",
        "biopsy",
        "endoscopy",
        "catheter",
        "stent",
        "implant",
        "pacemaker",
        "ventilator",
        "dialysis",
        "ecg",
        "ekg",
        "eeg",
        "emg",
    }

    # Terms that should ALWAYS be filtered, even from specialized lexicons
    # These are generic placeholders that sometimes appear in trial data
    ALWAYS_FILTER: Set[str] = {
        "medications",
        "medication",
        "other",
        "others",
        "placebo",
        "control",
        "standard of care",
        "standard care",
        "best supportive care",
        "usual care",
        "no intervention",
        "observation",
        "watchful waiting",
        "dietary supplement",
        "behavioral",
        "device",
        "procedure",
        "radiation",
        "biological",
        "combination product",
        "diagnostic test",
        "genetic",
        "various",
        "multiple",
        "unspecified",
        "investigational",
        "experimental",
        "study drug",
        "study treatment",
        "test drug",
        "test product",
        "active comparator",
        "sham comparator",
    }

    # Generic all-caps words that are not drugs
    NON_DRUG_ALLCAPS: Set[str] = {
        "information",
        "complete",
        "ring",
        "same",
        "other",
        "none",
        "all",
        "any",
        "new",
        "old",
        "high",
        "low",
        "full",
        "empty",
        "open",
        "closed",
        "start",
        "end",
        "first",
        "last",
        "next",
        "previous",
        "current",
        "total",
        "average",
        "mean",
        "median",
        "normal",
        "abnormal",
        "positive",
        "negative",
        "present",
        "absent",
        "available",
        "unavailable",
        "required",
        "optional",
        "primary",
        "secondary",
        "additional",
    }

    # Minimum drug name length
    MIN_LENGTH = 3

    def __init__(self):
        self.common_words_lower = {w.lower() for w in self.COMMON_WORDS}
        self.body_parts_lower = {w.lower() for w in self.BODY_PARTS}
        self.trial_status_lower = {w.lower() for w in self.TRIAL_STATUS_TERMS}
        self.equipment_lower = {w.lower() for w in self.EQUIPMENT_PROCEDURES}
        self.non_drug_allcaps_lower = {w.lower() for w in self.NON_DRUG_ALLCAPS}
        self.always_filter_lower = {w.lower() for w in self.ALWAYS_FILTER}

    def is_false_positive(
        self, matched_text: str, context: str, generator_type: DrugGeneratorType
    ) -> bool:
        """
        Check if a drug match is likely a false positive.

        Returns True if the match should be filtered out.
        """
        text_lower = matched_text.lower().strip()
        text_stripped = matched_text.strip()

        # Skip very short matches
        if len(text_lower) < self.MIN_LENGTH:
            return True

        # Always filter generic placeholder terms (even from specialized lexicons)
        if text_lower in self.always_filter_lower:
            return True

        # Always filter trial status terms (even from specialized lexicons)
        if text_lower in self.trial_status_lower:
            return True

        # Check if text contains any trial status term (handles various formats)
        text_normalized = text_lower.replace("_", " ")
        for status in self.trial_status_lower:
            if status in text_normalized:
                return True

        # Filter text containing trial status in parentheses like "Medications (NOT_YET_RECRUITING)"
        if "(" in text_stripped and ")" in text_stripped:
            # Also check the base word before parentheses
            base_word = text_stripped[:text_stripped.find("(")].strip().lower()
            if base_word in self.common_words_lower:
                return True

        # Always filter body parts
        if text_lower in self.body_parts_lower:
            return True

        # Always filter equipment/procedures
        if text_lower in self.equipment_lower:
            return True

        # Skip common words (unless from specialized lexicon)
        if generator_type not in {
            DrugGeneratorType.LEXICON_ALEXION,
            DrugGeneratorType.LEXICON_INVESTIGATIONAL,
        }:
            if text_lower in self.common_words_lower:
                return True

            # Filter generic all-caps words that aren't drugs
            if text_lower in self.non_drug_allcaps_lower:
                return True

        return False


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

        # Lexicon base path
        self.lexicon_base_path = Path(
            self.config.get("lexicon_base_path", "ouput_datasources")
        )

        # Feature flags
        self.enable_alexion = self.config.get("enable_alexion_lexicon", True)
        self.enable_investigational = self.config.get(
            "enable_investigational_lexicon", True
        )
        self.enable_fda = self.config.get("enable_fda_lexicon", True)
        self.enable_rxnorm = self.config.get("enable_rxnorm_lexicon", True)
        self.enable_scispacy = self.config.get("enable_scispacy", True)
        self.enable_patterns = self.config.get("enable_patterns", True)

        # Initialize FlashText processors
        self.alexion_processor: Optional[KeywordProcessor] = None
        self.investigational_processor: Optional[KeywordProcessor] = None
        self.fda_processor: Optional[KeywordProcessor] = None
        self.rxnorm_processor: Optional[KeywordProcessor] = None

        # Drug metadata dictionaries
        self.alexion_drugs: Dict[str, Dict] = {}
        self.investigational_drugs: Dict[str, Dict] = {}
        self.fda_drugs: Dict[str, Dict] = {}
        self.rxnorm_drugs: Dict[str, Dict] = {}

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
        self.fp_filter = DrugFalsePositiveFilter()

        # scispacy NER model
        self.nlp = None
        if self.enable_scispacy and SCISPACY_AVAILABLE:
            self._init_scispacy()

    def _load_lexicons(self) -> None:
        """Load all drug lexicons."""
        if self.enable_alexion:
            self._load_alexion_lexicon()
        if self.enable_investigational:
            self._load_investigational_lexicon()
        if self.enable_fda:
            self._load_fda_lexicon()
        if self.enable_rxnorm:
            self._load_rxnorm_lexicon()

    def _load_alexion_lexicon(self) -> None:
        """Load Alexion specialized drug lexicon."""
        path = self.lexicon_base_path / "2025_08_alexion_drugs.json"
        if not path.exists():
            print(f"[WARN] Alexion lexicon not found: {path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.alexion_processor = KeywordProcessor(case_sensitive=False)
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
                self.alexion_processor.add_keyword(drug_name, drug_name.lower())

                # Add variations (brand names, compound IDs if available)
                if isinstance(drug_info, dict):
                    for alias in drug_info.get("aliases", []):
                        self.alexion_processor.add_keyword(alias, drug_name.lower())

            self._lexicon_stats.append(
                ("Alexion drugs", len(known_drugs), "2025_08_alexion_drugs.json")
            )

        except Exception as e:
            print(f"[WARN] Failed to load Alexion lexicon: {e}")

    def _load_investigational_lexicon(self) -> None:
        """Load investigational drugs from ClinicalTrials.gov data."""
        path = self.lexicon_base_path / "2025_08_investigational_drugs.json"
        if not path.exists():
            print(f"[WARN] Investigational lexicon not found: {path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.investigational_processor = KeywordProcessor(case_sensitive=False)
            count = 0

            for entry in data:
                drug_name = entry.get("interventionName", "").strip()
                if not drug_name or len(drug_name) < 3:
                    continue

                # Skip non-drug interventions
                if entry.get("interventionType") != "DRUG":
                    continue

                drug_key = drug_name.lower()
                if drug_key not in self.investigational_drugs:
                    self.investigational_drugs[drug_key] = {
                        "preferred_name": drug_name,
                        "nct_id": entry.get("nctId"),
                        "conditions": entry.get("conditions", []),
                        "status": entry.get("overallStatus"),
                        "title": entry.get("title"),
                        "source": "investigational",
                    }
                    self.investigational_processor.add_keyword(drug_name, drug_key)
                    count += 1

            self._lexicon_stats.append(
                ("Investigational drugs", count, "2025_08_investigational_drugs.json")
            )

        except Exception as e:
            print(f"[WARN] Failed to load investigational lexicon: {e}")

    def _load_fda_lexicon(self) -> None:
        """Load FDA approved drugs lexicon."""
        path = self.lexicon_base_path / "2025_08_fda_approved_drugs.json"
        if not path.exists():
            print(f"[WARN] FDA lexicon not found: {path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.fda_processor = KeywordProcessor(case_sensitive=False)
            count = 0

            for entry in data:
                drug_name = entry.get("key", "").strip()
                if not drug_name or len(drug_name) < 3:
                    continue

                drug_key = drug_name.lower()
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
                    self.fda_processor.add_keyword(drug_name, drug_key)
                    count += 1

                    # Also add brand name if different
                    brand = meta.get("brand_name", "")
                    if brand and brand.lower() != drug_key:
                        brand_key = brand.lower()
                        if brand_key not in self.fda_drugs:
                            self.fda_processor.add_keyword(brand, drug_key)

            self._lexicon_stats.append(
                ("FDA approved drugs", count, "2025_08_fda_approved_drugs.json")
            )

        except Exception as e:
            print(f"[WARN] Failed to load FDA lexicon: {e}")

    def _load_rxnorm_lexicon(self) -> None:
        """Load RxNorm general drug lexicon."""
        path = self.lexicon_base_path / "2025_08_lexicon_drug.json"
        if not path.exists():
            print(f"[WARN] RxNorm lexicon not found: {path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.rxnorm_processor = KeywordProcessor(case_sensitive=False)
            count = 0

            for entry in data:
                term = entry.get("term", "").strip()
                if not term or len(term) < 3:
                    continue

                term_key = term.lower()
                if term_key not in self.rxnorm_drugs:
                    self.rxnorm_drugs[term_key] = {
                        "preferred_name": term,
                        "term_normalized": entry.get("term_normalized"),
                        "rxcui": entry.get("rxcui"),
                        "tty": entry.get("tty"),
                        "source": "rxnorm",
                    }
                    self.rxnorm_processor.add_keyword(term, term_key)
                    count += 1

            self._lexicon_stats.append(
                ("RxNorm terms", count, "2025_08_lexicon_drug.json")
            )

        except Exception as e:
            print(f"[WARN] Failed to load RxNorm lexicon: {e}")

    def _init_scispacy(self) -> None:
        """Initialize scispacy NER model."""
        if not SCISPACY_AVAILABLE:
            return

        try:
            # Try large model first, fall back to small
            try:
                self.nlp = spacy.load("en_core_sci_lg")
            except OSError:
                self.nlp = spacy.load("en_core_sci_sm")

            # Add UMLS linker for chemical entities
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
            print(f"[WARN] Failed to initialize scispacy for drugs: {e}")
            self.nlp = None

    def _print_lexicon_summary(self) -> None:
        """Print compact summary of loaded drug lexicons."""
        if not self._lexicon_stats:
            return

        # All drug lexicons go under "Drug" category
        total = sum(count for _, count, _ in self._lexicon_stats if count > 1)
        file_count = len([s for s in self._lexicon_stats if s[1] > 0])
        print(f"\nDrug lexicons: {file_count} sources, {total:,} entries")
        print("─" * 70)
        print(f"  Drug ({total:,} entries)")

        for name, count, filename in self._lexicon_stats:
            if count > 1:
                print(f"    • {name:<26} {count:>8,}  {filename}")
            else:
                print(f"    • {name:<26} {'enabled':>8}  {filename}")
        print()

    def detect(self, doc_graph: DocumentGraph) -> List[DrugCandidate]:
        """
        Detect drug mentions in document.

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
        if self.enable_patterns:
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

        # Layer 6: scispacy NER fallback
        if self.nlp:
            candidates.extend(
                self._detect_with_ner(full_text, doc_graph, doc_fingerprint)
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

            candidate = DrugCandidate(
                doc_id=doc_graph.doc_id,
                matched_text=matched_text,
                preferred_name=drug_info.get("preferred_name", matched_text),
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
                    conditions=drug_info.get("conditions", []),
                    nct_id=drug_info.get("nct_id"),
                    initial_confidence=0.8,
                    provenance=DrugProvenanceMetadata(
                        pipeline_version=self.pipeline_version,
                        run_id=self.run_id,
                        doc_fingerprint=doc_fingerprint,
                        generator_name=DrugGeneratorType.PATTERN_COMPOUND_ID,
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
        candidates = []

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
                    ),
                )
                candidates.append(candidate)

        except Exception as e:
            print(f"[WARN] scispacy drug detection error: {e}")

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

    def _deduplicate(self, candidates: List[DrugCandidate]) -> List[DrugCandidate]:
        """
        Deduplicate candidates, preferring specialized sources.

        Priority: Alexion > Investigational > FDA > RxNorm > NER
        """
        # Group by matched text (case-insensitive)
        by_text: Dict[str, List[DrugCandidate]] = {}
        for c in candidates:
            key = c.matched_text.lower()
            if key not in by_text:
                by_text[key] = []
            by_text[key].append(c)

        # Priority order
        priority = {
            DrugGeneratorType.LEXICON_ALEXION: 0,
            DrugGeneratorType.PATTERN_COMPOUND_ID: 1,
            DrugGeneratorType.LEXICON_INVESTIGATIONAL: 2,
            DrugGeneratorType.LEXICON_FDA: 3,
            DrugGeneratorType.LEXICON_RXNORM: 4,
            DrugGeneratorType.SCISPACY_NER: 5,
        }

        # Keep highest priority for each text
        deduped = []
        for text_key, group in by_text.items():
            # Sort by priority
            group.sort(key=lambda c: priority.get(c.generator_type, 99))
            # Keep the highest priority (first after sort)
            deduped.append(group[0])

        return deduped
