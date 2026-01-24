# corpus_metadata/corpus_metadata/C_generators/C04_strategy_flashtext.py

from __future__ import annotations

import csv
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from flashtext import KeywordProcessor

# scispacy for biomedical NER (identifies entities like C3G, eGFR, KDIGO)
try:
    import spacy
    from scispacy.abbreviation import AbbreviationDetector  # noqa: F401
    from scispacy.linking import EntityLinker  # noqa: F401

    SCISPACY_AVAILABLE = True

    # Suppress scispacy warning about empty matcher patterns (W036)
    # This occurs when abbreviation_detector has no global patterns defined
    warnings.filterwarnings(
        "ignore",
        message=r".*The component 'matcher' does not have any patterns defined.*",
        category=UserWarning,
        module="scispacy.abbreviation",
    )
except ImportError:
    spacy = None
    SCISPACY_AVAILABLE = False

from A_core.A02_interfaces import BaseCandidateGenerator

# =============================================================================
# LIGHT NOISE FILTERING (High Recall - Let Validation Layer Judge)
# =============================================================================
# Philosophy: Generators should be EXHAUSTIVE. Only block OBVIOUS noise.
# Claude (D_validation) will handle borderline cases with context awareness.

# Obvious non-abbreviations: single letters, basic English function words
OBVIOUS_NOISE: set = {
    # Single letters (never valid abbreviations alone)
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    # Basic English function words (articles, prepositions, conjunctions)
    # NOTE: "or" and "us" removed - they can be valid abbreviations (Odds Ratio, United States)
    "an",
    "as",
    "at",
    "be",
    "by",
    "do",
    "go",
    "he",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "no",
    "of",
    "on",
    "so",
    "to",
    "up",
    "we",
    "the",
    "and",
    "for",
    "but",
    "not",
    "are",
    "was",
    "were",
    "been",
    "have",
    "has",
    "had",
    "will",
    "would",
    "could",
    "should",
    "this",
    "that",
    "these",
    "those",
    "with",
    "from",
    "into",
    # Citation artifacts
    "et",
    "al",
    # Measurement units (not abbreviations, just units)
    "dl",
    "ml",
    "mg",
    "kg",
    "mm",
    "cm",
    "hz",
    "khz",
    "mhz",
    "mmhg",
    "kpa",
    "mol",
    "mmol",
    "umol",
    "nmol",
    # Full English words mistakenly in lexicons (NOT abbreviations)
    "investigator",
    "investigators",
    "sponsor",
    "protocol",
    "study",
    "patient",
    "patients",
    "subject",
    "subjects",
    "article",
    "articles",
    # Geographic (context-dependent, high FP rate in pharma docs)
    "nj",
    "ny",
    "ca",
    "tx",  # US states - usually location, not abbreviation
    # Company names (not abbreviations, even if in UMLS)
    "roche",
    "novartis",
    "pfizer",
    "merck",
    "bayer",
    "sanofi",
    "gsk",
    "astrazeneca",
    "amgen",
    "gilead",
    "biogen",
    "regeneron",
    "vertex",
    "alexion",
    "takeda",
    "abbvie",
    "lilly",
    "bristol",
    "johnson",
}

# Minimum length (allow 2-char if uppercase like CT, MR, IV)
MIN_ABBREV_LENGTH = 2

# =============================================================================
# WRONG EXPANSION BLACKLIST
# =============================================================================
# Some UMLS/lexicon entries have clearly wrong or contextually inappropriate
# expansions. These SF -> LF pairs should never be used.
#
# Format: (short_form_lower, bad_long_form_lower)
WRONG_EXPANSION_BLACKLIST: Set[Tuple[str, str]] = {
    # UMLS mapping errors
    ("task", "product"),  # TASK in protocols = schedule/activity, not SNOMED "product"
    ("musk", "musk secretion from musk deer"),  # MuSK = muscle-specific kinase
    # Clinical trial context - these expansions are wrong in protocol context
    ("et", "essential thrombocythemia"),  # ET in protocols = Early Termination
    ("sc", "subcutaneous"),  # Often correct, but sometimes wrong
    # Generic wrong mappings
    ("exam", "examination"),  # Too generic, not an abbreviation
    ("dose", "dosage"),  # Too generic
    ("task", "kcnk3 gene"),  # TASK is not usually a gene reference in protocols
}

# Long forms that are ALWAYS wrong (regardless of short form)
# These are UMLS artifacts or clearly incorrect expansions
BAD_LONG_FORMS: Set[str] = {
    "product",  # Too generic, SNOMED artifact
    "musk secretion from musk deer",  # Wrong MuSK expansion
    "essential thrombocythemia",  # Often wrong in clinical trial context
    "ambulatory care facilities",  # Wrong expansion for "Clinic"
    "simultaneous",  # Wrong expansion for "CONCOMITANT"
    "kit dosing unit",  # Wrong expansion for "Kits"
    "medical devices",  # Wrong expansion for "Device"
    "planum polare",  # Wrong expansion for "PP" (usually Per Protocol)
    "follicle stimulating hormone injectable",  # Wrong for FSH in most contexts
    "multiple sulfatase deficiency",  # MSD in pharma context = Merck Sharp & Dohme
}

from A_core.A01_domain_models import (
    Candidate,
    Coordinate,
    FieldType,
    GeneratorType,
    LexiconIdentifier,
    ProvenanceMetadata,
)
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from B_parsing.B02_doc_graph import DocumentGraph


class LexiconEntry:
    """Compiled lexicon entry with regex pattern and source provenance."""

    __slots__ = ("sf", "lf", "pattern", "source", "lexicon_ids", "preserve_case")

    def __init__(
        self,
        sf: str,
        lf: str,
        pattern: re.Pattern,
        source: str,
        lexicon_ids: Optional[List[Dict[str, str]]] = None,
        preserve_case: bool = True,
    ):
        self.sf = sf
        self.lf = lf
        self.pattern = pattern
        self.source = source  # Lexicon file name for provenance
        self.lexicon_ids = lexicon_ids or []  # External IDs [{source, id}, ...]
        self.preserve_case = preserve_case  # If True, use matched text as SF


class RegexLexiconGenerator(BaseCandidateGenerator):
    """
    LEXICON MATCHER using:
    - Regex patterns for abbreviation lexicon (handles spacing/case variations)
    - FlashText for disease lexicon (fast exact matching)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

        # Lexicon paths
        self.abbrev_lexicon_path = Path(
            self.config.get(
                "abbrev_lexicon_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_abbreviation_general.json",
            )
        )
        self.clinical_research_abbrev_path = Path(
            self.config.get(
                "clinical_research_abbrev_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/clinical_research_abbreviations.json",
            )
        )
        self.disease_lexicon_path = Path(
            self.config.get(
                "disease_lexicon_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_lexicon_disease.json",
            )
        )
        self.orphanet_lexicon_path = Path(
            self.config.get(
                "orphanet_lexicon_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_orphanet_diseases.json",
            )
        )
        self.rare_disease_acronyms_path = Path(
            self.config.get(
                "rare_disease_acronyms_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_rare_disease_acronyms.json",
            )
        )
        self.umls_abbrev_path = Path(
            self.config.get(
                "umls_abbrev_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_umls_biological_abbreviations_v5.tsv",
            )
        )
        self.umls_clinical_path = Path(
            self.config.get(
                "umls_clinical_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_umls_clinical_abbreviations_v5.tsv",
            )
        )
        self.anca_lexicon_path = Path(
            self.config.get(
                "anca_lexicon_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/disease_lexicon_anca.json",
            )
        )
        self.igan_lexicon_path = Path(
            self.config.get(
                "igan_lexicon_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/disease_lexicon_igan.json",
            )
        )
        self.pah_lexicon_path = Path(
            self.config.get(
                "pah_lexicon_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/disease_lexicon_pah.json",
            )
        )
        self.trial_acronyms_path = Path(
            self.config.get(
                "trial_acronyms_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/trial_acronyms_lexicon.json",
            )
        )
        self.pro_scales_path = Path(
            self.config.get(
                "pro_scales_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/pro_scales_lexicon.json",
            )
        )
        self.pharma_companies_path = Path(
            self.config.get(
                "pharma_companies_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/pharma_companies_lexicon.json",
            )
        )

        # NEW LEXICONS: Meta-Inventory, MONDO, ChEMBL
        self.meta_inventory_path = Path(
            self.config.get(
                "meta_inventory_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/2025_meta_inventory_abbreviations.json",
            )
        )
        self.mondo_lexicon_path = Path(
            self.config.get(
                "mondo_lexicon_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/2025_mondo_diseases.json",
            )
        )
        self.chembl_lexicon_path = Path(
            self.config.get(
                "chembl_lexicon_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/2025_chembl_drugs.json",
            )
        )

        self.context_window = int(self.config.get("context_window", 300))

        # Abbreviation entries (regex-based)
        self.abbrev_entries: List[LexiconEntry] = []

        # Disease/entity entries (FlashText-based)
        self.entity_kp = KeywordProcessor(case_sensitive=False)
        self.entity_canonical: Dict[str, str] = {}  # matched_term -> canonical_name
        self.entity_source: Dict[str, str] = {}  # matched_term -> source file
        self.entity_ids: Dict[
            str, List[Dict[str, str]]
        ] = {}  # matched_term -> [{source, id}, ...]

        # Provenance
        self.pipeline_version = str(
            self.config.get("pipeline_version") or get_git_revision_hash()
        )
        self.run_id = str(self.config.get("run_id") or generate_run_id("ABBR"))
        self.doc_fingerprint_default = str(
            self.config.get("doc_fingerprint") or "unknown-doc-fingerprint"
        )

        # Lexicon loading stats (for summary output)
        self._lexicon_stats: List[tuple] = []

        # Load lexicons
        # Note: Disease/Orphanet lexicons contain disease NAMES, not abbreviations
        # They cause FPs by matching chromosome numbers (45, 46, 10p) as "abbreviations"
        self._load_abbrev_lexicon(self.abbrev_lexicon_path)
        self._load_abbrev_lexicon(
            self.clinical_research_abbrev_path, "Clinical research"
        )
        # self._load_disease_lexicon(self.disease_lexicon_path)  # Disabled: names, not abbreviations
        # self._load_orphanet_lexicon(self.orphanet_lexicon_path)  # Disabled: names, not abbreviations
        self._load_rare_disease_acronyms(self.rare_disease_acronyms_path)
        self._load_umls_tsv(self.umls_abbrev_path)
        self._load_umls_tsv(self.umls_clinical_path)
        self._load_anca_lexicon(self.anca_lexicon_path)
        self._load_igan_lexicon(self.igan_lexicon_path)
        self._load_pah_lexicon(self.pah_lexicon_path)
        self._load_trial_acronyms(self.trial_acronyms_path)
        self._load_pro_scales(self.pro_scales_path)
        # Pharma companies now extracted via dedicated C12_strategy_pharma detector
        # self._load_pharma_companies(self.pharma_companies_path)

        # NEW LEXICONS: Meta-Inventory, MONDO, ChEMBL
        # Meta-Inventory: 104K+ clinical abbreviations (increases coverage 28-52%)
        self._load_meta_inventory(self.meta_inventory_path)
        # MONDO: Unified disease ontology with precise semantic mappings
        self._load_mondo_lexicon(self.mondo_lexicon_path)
        # ChEMBL: Open drug database with approved drugs
        self._load_chembl_lexicon(self.chembl_lexicon_path)

        # Initialize scispacy NER for biomedical entity recognition
        self.scispacy_nlp = None
        self.umls_linker = None
        if SCISPACY_AVAILABLE and spacy is not None:
            try:
                # Try lg model first (better for linking), fall back to sm
                try:
                    self.scispacy_nlp = spacy.load("en_core_sci_lg")
                    model_name = "en_core_sci_lg"
                except OSError:
                    self.scispacy_nlp = spacy.load("en_core_sci_sm")
                    model_name = "en_core_sci_sm"

                # Add abbreviation detector for Schwartz-Hearst pattern matching
                self.scispacy_nlp.add_pipe("abbreviation_detector")

                # Add UMLS EntityLinker for abbreviation resolution
                try:
                    self.scispacy_nlp.add_pipe(
                        "scispacy_linker",
                        config={"resolve_abbreviations": True, "linker_name": "umls"},
                    )
                    self.umls_linker = self.scispacy_nlp.get_pipe("scispacy_linker")
                    print(
                        f"Loaded scispacy {model_name} with abbreviation detector + UMLS linker"
                    )
                except Exception as e:
                    print(
                        f"Loaded scispacy {model_name} with abbreviation detector (no UMLS linker: {e})"
                    )
            except OSError as e:
                print(f"Warning: Could not load scispacy model: {e}")

    @property
    def generator_type(self) -> GeneratorType:
        return GeneratorType.LEXICON_MATCH

    def extract(self, doc_structure: DocumentGraph) -> List[Candidate]:
        doc = doc_structure  # Alias for readability
        out: List[Candidate] = []
        seen: Set[Tuple[str, str]] = set()  # (SF_upper, LF_lower) dedup

        # Collect blocks for batch scispacy processing
        blocks_data: List[
            Tuple[Any, str, int, int]
        ] = []  # (block, text, start_offset, end_offset)
        full_text_parts: List[str] = []
        current_offset = 0

        for block in doc.iter_linear_blocks(skip_header_footer=True):
            text = (block.text or "").strip()
            if not text:
                continue

            # Track block position in concatenated text
            blocks_data.append(
                (block, text, current_offset, current_offset + len(text))
            )
            full_text_parts.append(text)
            current_offset += len(text) + 2  # +2 for "\n\n" separator

            # 1) Abbreviation matches (regex) - per block (fast)
            for entry in self.abbrev_entries:
                # Skip blacklisted/invalid abbreviations
                if not self._is_valid_match(entry.sf):
                    continue

                for match in entry.pattern.finditer(text):
                    start, end = match.start(), match.end()

                    # Use matched text to preserve original case (e.g., IgA, MedDRA)
                    matched_text = text[start:end].strip()
                    if not matched_text:
                        continue  # Skip empty matches

                    sf_to_use = matched_text if entry.preserve_case else entry.sf

                    key = (sf_to_use.upper(), entry.lf.lower())
                    if key in seen:
                        continue
                    seen.add(key)

                    # Convert dict lexicon_ids to LexiconIdentifier objects
                    lex_ids = None
                    if entry.lexicon_ids:
                        lex_ids = [
                            LexiconIdentifier(source=d["source"], id=d["id"])
                            for d in entry.lexicon_ids
                        ]

                    out.append(
                        self._make_candidate(
                            doc=doc,
                            block=block,
                            short_form=sf_to_use,  # Use matched text with original case
                            long_form=entry.lf,
                            start=start,
                            end=end,
                            text=text,
                            rule_version="abbrev_regex::v1",
                            lexicon_source=entry.source,
                            lexicon_ids=lex_ids,
                        )
                    )

            # 2) Entity matches (FlashText - diseases, orphanet terms)
            entity_hits = self.entity_kp.extract_keywords(text, span_info=True)
            for matched_term, start, end in entity_hits:
                # Skip blacklisted/invalid terms
                if not self._is_valid_match(matched_term):
                    continue

                canonical = self.entity_canonical.get(matched_term) or matched_term
                source = self.entity_source.get(matched_term, "unknown")

                key = (matched_term.upper(), canonical.lower())
                if key in seen:
                    continue
                seen.add(key)

                # Convert dict lexicon_ids to LexiconIdentifier objects
                entity_lex_ids = None
                raw_ids = self.entity_ids.get(matched_term, [])
                if raw_ids:
                    entity_lex_ids = [
                        LexiconIdentifier(source=d["source"], id=d["id"])
                        for d in raw_ids
                    ]

                out.append(
                    self._make_candidate(
                        doc=doc,
                        block=block,
                        short_form=matched_term,
                        long_form=canonical,
                        start=start,
                        end=end,
                        text=text,
                        rule_version="entity_exact::v1",
                        lexicon_source=source,
                        lexicon_ids=entity_lex_ids,
                    )
                )

        # 3) BATCH scispacy NER + abbreviation detection + UMLS linking
        # Process entire document at once instead of per-block (5-10x faster)
        if self.scispacy_nlp is not None and blocks_data:
            try:
                # Concatenate all text with separators
                full_text = "\n\n".join(full_text_parts)

                # Run scispacy ONCE on full document
                spacy_doc = self.scispacy_nlp(full_text)

                # Helper to find which block contains a character offset
                def find_block_for_offset(char_offset: int):
                    for block, text, start, end in blocks_data:
                        if start <= char_offset < end:
                            return block, text, start
                    return None, None, 0

                # Extract abbreviations found by Schwartz-Hearst detector
                for abrv in spacy_doc._.abbreviations:
                    sf = abrv.text
                    lf = str(abrv._.long_form) if abrv._.long_form else None

                    if not sf or not lf:
                        continue
                    if not self._is_valid_match(sf):
                        continue

                    key = (sf.upper(), lf.lower())
                    if key in seen:
                        continue
                    seen.add(key)

                    # Find the block this abbreviation belongs to
                    block, block_text, block_start = find_block_for_offset(
                        abrv.start_char
                    )
                    if block is None or block_text is None:
                        continue

                    # Adjust offsets to be relative to block
                    local_start = abrv.start_char - block_start
                    local_end = abrv.end_char - block_start

                    out.append(
                        self._make_candidate(
                            doc=doc,
                            block=block,
                            short_form=sf,
                            long_form=lf,
                            start=local_start,
                            end=local_end,
                            text=block_text,
                            rule_version="scispacy_abbrev::v1",
                            lexicon_source="scispacy",
                            lexicon_ids=None,
                        )
                    )

                # Extract NER entities with UMLS linking
                for ent in spacy_doc.ents:
                    ent_text = ent.text.strip()

                    # Only consider short, uppercase-containing tokens as abbreviations
                    if len(ent_text) < 2 or len(ent_text) > 12:
                        continue
                    if " " in ent_text:  # Skip multi-word entities
                        continue
                    if not any(c.isupper() for c in ent_text):
                        continue
                    if not self._is_valid_match(ent_text):
                        continue

                    # Try UMLS linker first for expansion
                    lf_from_umls = None
                    umls_cui = None
                    if hasattr(ent._, "kb_ents") and ent._.kb_ents:
                        # kb_ents is [(CUI, score), ...] - take top match
                        top_match = ent._.kb_ents[0]
                        umls_cui = top_match[0]
                        # Get canonical name from linker's knowledge base
                        if self.umls_linker and hasattr(self.umls_linker, "kb"):
                            try:
                                kb_entry = self.umls_linker.kb.cui_to_entity.get(
                                    umls_cui
                                )
                                if kb_entry:
                                    candidate_lf = kb_entry.canonical_name
                                    # Filter out known wrong expansions from UMLS
                                    sf_lower = ent_text.lower()
                                    lf_lower = candidate_lf.lower() if candidate_lf else ""
                                    if (sf_lower, lf_lower) not in WRONG_EXPANSION_BLACKLIST and lf_lower not in BAD_LONG_FORMS:
                                        lf_from_umls = candidate_lf
                            except Exception:
                                pass

                    # Fall back to our lexicons if UMLS didn't provide expansion
                    lf_from_lexicon = lf_from_umls
                    if not lf_from_lexicon:
                        lf_from_lexicon = self.entity_canonical.get(
                            ent_text
                        ) or self.entity_canonical.get(ent_text.upper())
                    if not lf_from_lexicon:
                        # Try to find in abbrev_entries
                        for entry in self.abbrev_entries:
                            if entry.sf.upper() == ent_text.upper():
                                lf_from_lexicon = entry.lf
                                break

                    if not lf_from_lexicon:
                        continue  # Only report if we have a known expansion
                    assert lf_from_lexicon is not None  # Type narrowing for pyright

                    key = (ent_text.upper(), lf_from_lexicon.lower())
                    if key in seen:
                        continue
                    seen.add(key)

                    # Find the block this entity belongs to
                    block, block_text, block_start = find_block_for_offset(
                        ent.start_char
                    )
                    if block is None or block_text is None:
                        continue

                    # Adjust offsets to be relative to block
                    local_start = ent.start_char - block_start
                    local_end = ent.end_char - block_start

                    # Build lexicon IDs from UMLS CUI if available
                    lex_ids = None
                    if umls_cui:
                        lex_ids = [LexiconIdentifier(source="UMLS", id=umls_cui)]

                    source = "scispacy+umls" if lf_from_umls else "scispacy+lexicon"
                    out.append(
                        self._make_candidate(
                            doc=doc,
                            block=block,
                            short_form=ent_text,
                            long_form=lf_from_lexicon,
                            start=local_start,
                            end=local_end,
                            text=block_text,
                            rule_version="scispacy_ner::v1",
                            lexicon_source=source,
                            lexicon_ids=lex_ids,
                        )
                    )
            except Exception:
                # Don't fail entire extraction if scispacy has issues
                pass

        # 4) Regex-based inline definition detector
        # Catches patterns scispacy's Schwartz-Hearst might miss:
        # - Mixed-case abbreviations like "LoE", "LoA"
        # - Reversed patterns "ABBREV (long form)"
        # - Comma-separated definitions "ABBREV, the long form"
        if blocks_data:
            full_text = "\n\n".join(full_text_parts)
            inline_matches = self._extract_inline_definitions(full_text)

            for sf, lf, start, end in inline_matches:
                if not self._is_valid_match(sf):
                    continue

                key = (sf.upper(), lf.lower())
                if key in seen:
                    continue
                seen.add(key)

                # Find the block this definition belongs to
                block = None  # type: ignore[assignment]
                block_text = None
                block_start = 0
                for b, text, s, e in blocks_data:
                    if s <= start < e:
                        block = b
                        block_text = text
                        block_start = s
                        break

                if block is None or block_text is None:
                    continue

                # Adjust offsets to be relative to block
                local_start = start - block_start
                local_end = end - block_start

                out.append(
                    self._make_candidate(
                        doc=doc,
                        block=block,
                        short_form=sf,
                        long_form=lf,
                        start=local_start,
                        end=local_end,
                        text=block_text,
                        rule_version="inline_regex::v1",
                        lexicon_source="inline_definition",
                        lexicon_ids=None,
                    )
                )

        return out

    def _make_candidate(
        self,
        doc: DocumentGraph,
        block,
        short_form: str,
        long_form: str,
        start: int,
        end: int,
        text: str,
        rule_version: str,
        lexicon_source: str,
        lexicon_ids: Optional[List[LexiconIdentifier]] = None,
    ) -> Candidate:
        ctx = self._make_context(text, start, end)

        loc = Coordinate(
            page_num=int(block.page_num),
            block_id=str(block.id),
            bbox=block.bbox,
        )

        prov = ProvenanceMetadata(
            pipeline_version=self.pipeline_version,
            run_id=self.run_id,
            doc_fingerprint=self.doc_fingerprint_default,
            generator_name=self.generator_type,
            rule_version=rule_version,
            lexicon_source=lexicon_source,
            lexicon_ids=lexicon_ids,
        )

        return Candidate(
            doc_id=doc.doc_id,
            field_type=FieldType.DEFINITION_PAIR,
            generator_type=self.generator_type,
            short_form=short_form,
            long_form=long_form,
            context_text=ctx,
            context_location=loc,
            initial_confidence=0.95,
            provenance=prov,
        )

    def _print_lexicon_summary(self) -> None:
        """Print compact summary of loaded lexicons grouped by category."""
        if not self._lexicon_stats:
            return

        # Categorize lexicons with explicit ordering: Abbreviation, Drug, Disease, Other
        categories: list[tuple[str, list[tuple[str, int, str]]]] = [
            ("Abbreviation", []),
            ("Drug", []),
            ("Disease", []),
            ("Other", []),
        ]
        cat_dict = {name: items for name, items in categories}

        # Map lexicon names to categories
        category_map = {
            # Abbreviation
            "Abbreviations": "Abbreviation",
            "Clinical research": "Abbreviation",
            "UMLS biological": "Abbreviation",
            "UMLS clinical": "Abbreviation",
            "Meta-Inventory": "Abbreviation",
            # Drug
            "ChEMBL drugs": "Drug",
            # Disease
            "Rare disease acronyms": "Disease",
            "ANCA disease": "Disease",
            "IgAN disease": "Disease",
            "PAH disease": "Disease",
            "MONDO diseases": "Disease",
            # Other
            "Trial acronyms": "Other",
            "PRO scales": "Other",
            "Pharma companies": "Other",
        }

        for name, count, filename in self._lexicon_stats:
            cat = category_map.get(name, "Abbreviation")
            if cat in cat_dict:
                cat_dict[cat].append((name, count, filename))

        total = sum(count for _, count, _ in self._lexicon_stats)
        file_count = len([s for s in self._lexicon_stats if s[1] > 0])
        print(f"\nLexicons loaded: {file_count} files, {total:,} terms")
        print("─" * 70)

        for cat_name, items in categories:
            if not items:
                continue
            cat_total = sum(count for _, count, _ in items)
            print(f"  {cat_name} ({cat_total:,} terms)")
            for name, count, filename in items:
                print(f"    • {name:<26} {count:>8,}  {filename}")
        print()

    def _load_abbrev_lexicon(self, path: Path, label: str = "Abbreviations") -> None:
        if not path.exists():
            return

        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return

        data = json.loads(content)
        if not data:  # Skip empty files
            return

        loaded = 0

        for sf, entry in data.items():
            if not sf or not isinstance(entry, dict):
                continue

            lf = entry.get("canonical_expansion")
            regex_str = entry.get("regex")

            if not lf or not regex_str:
                continue

            try:
                case_insensitive = bool(entry.get("case_insensitive", False))
                flags = re.IGNORECASE if case_insensitive else 0
                pattern = re.compile(regex_str, flags)

                self.abbrev_entries.append(
                    LexiconEntry(sf=sf, lf=lf, pattern=pattern, source=path.name)
                )
                loaded += 1
            except re.error:
                pass

        self._lexicon_stats.append((label, loaded, path.name))

    def _load_disease_lexicon(self, path: Path) -> None:
        if not path.exists():
            print(f"Disease lexicon not found: {path}")
            return

        print(f"Loading disease lexicon: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name

        loaded = 0

        for entry in data:
            if not isinstance(entry, dict):
                continue

            label = (entry.get("label") or "").strip()
            if not label or len(label) < 3:
                continue

            # Extract external IDs (Orphanet, MONDO, etc.)
            sources_list = entry.get("sources", [])
            lexicon_ids = [
                {"source": s.get("source", ""), "id": s.get("id", "")}
                for s in sources_list
                if isinstance(s, dict) and s.get("id")
            ]

            self.entity_kp.add_keyword(label, label)
            self.entity_canonical[label] = label
            self.entity_source[label] = source
            self.entity_ids[label] = lexicon_ids
            loaded += 1

        print(f"Loaded {loaded} disease terms from {path.name}")

    def _load_orphanet_lexicon(self, path: Path) -> None:
        if not path.exists():
            print(f"Orphanet lexicon not found: {path}")
            return

        print(f"Loading Orphanet lexicon: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name

        loaded = 0

        for entry in data:
            if not isinstance(entry, dict):
                continue

            name = (entry.get("name") or "").strip()
            if not name or len(name) < 3:
                continue

            synonyms = entry.get("synonyms") or []

            # Build lexicon IDs from orphacode
            orphacode = entry.get("orphacode")
            lexicon_ids = []
            if orphacode:
                lexicon_ids.append({"source": "Orphanet", "id": f"ORPHA:{orphacode}"})

            # Register main name
            self.entity_kp.add_keyword(name, name)
            self.entity_canonical[name] = name
            self.entity_source[name] = source
            self.entity_ids[name] = lexicon_ids
            loaded += 1

            # Register synonyms (map to canonical name)
            for syn in synonyms:
                syn = (syn or "").strip()
                if not syn or len(syn) < 2:
                    continue
                self.entity_kp.add_keyword(syn, syn)
                self.entity_canonical[syn] = name  # maps to canonical
                self.entity_source[syn] = source
                self.entity_ids[syn] = lexicon_ids  # Same IDs as canonical
                loaded += 1

        print(f"Loaded {loaded} Orphanet terms from {path.name}")

    def _load_rare_disease_acronyms(self, path: Path) -> None:
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name
        loaded = 0

        for acronym, entry in data.items():
            if not acronym or not isinstance(entry, dict):
                continue

            acronym = acronym.strip()
            name = (entry.get("name") or "").strip()

            if not acronym or len(acronym) < 2 or not name:
                continue

            # Build lexicon IDs from orphacode
            orphacode = entry.get("orphacode")
            lexicon_ids = []
            if orphacode:
                lexicon_ids.append({"source": "Orphanet", "id": f"ORPHA:{orphacode}"})

            self.entity_kp.add_keyword(acronym, acronym)
            self.entity_canonical[acronym] = name
            self.entity_source[acronym] = source
            self.entity_ids[acronym] = lexicon_ids
            loaded += 1

        self._lexicon_stats.append(("Rare disease acronyms", loaded, path.name))

    def _load_umls_tsv(self, path: Path) -> None:
        if not path.exists():
            return

        source = path.name
        loaded = 0
        skipped_wrong_expansion = 0

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                abbrev = (row.get("Abbreviation") or "").strip()
                expansion = (row.get("Expansion") or "").strip()
                top_source = (row.get("TopSource") or "").strip()

                if not abbrev or len(abbrev) < 2 or not expansion:
                    continue

                # Filter out known wrong SF -> LF pairs
                abbrev_lower = abbrev.lower()
                expansion_lower = expansion.lower()
                if (abbrev_lower, expansion_lower) in WRONG_EXPANSION_BLACKLIST:
                    skipped_wrong_expansion += 1
                    continue

                # Filter out known bad long forms (regardless of short form)
                if expansion_lower in BAD_LONG_FORMS:
                    skipped_wrong_expansion += 1
                    continue

                # Build lexicon IDs from UMLS source
                lexicon_ids = []
                if top_source:
                    lexicon_ids.append({"source": f"UMLS:{top_source}", "id": abbrev})

                self.entity_kp.add_keyword(abbrev, abbrev)
                self.entity_canonical[abbrev] = expansion
                self.entity_source[abbrev] = source
                self.entity_ids[abbrev] = lexicon_ids
                loaded += 1

        # Extract a short name from filename for display
        name = "UMLS biological" if "biological" in source else "UMLS clinical"
        self._lexicon_stats.append((name, loaded, path.name))
        if skipped_wrong_expansion > 0:
            print(f"    [INFO] Skipped {skipped_wrong_expansion} wrong expansions from {path.name}")

    def _extract_identifiers(self, identifiers: Dict) -> List[Dict[str, str]]:
        """Extract lexicon IDs from an identifiers dict."""
        lexicon_ids: list[dict[str, str]] = []
        if not identifiers or not isinstance(identifiers, dict):
            return lexicon_ids

        # Map of identifier keys to source names
        source_map = {
            "ORPHA": "Orphanet",
            "ICD11": "ICD-11",
            "ICD10": "ICD-10",
            "ICD10CM": "ICD-10-CM",
            "SNOMED_CT": "SNOMED-CT",
            "MESH": "MeSH",
            "UMLS": "UMLS",
            "UMLS_CUI": "UMLS",
            "MONDO": "MONDO",
        }
        for key, source_name in source_map.items():
            if key in identifiers and identifiers[key]:
                id_val = identifiers[key]
                # Format the ID properly
                if key == "ORPHA":
                    id_val = f"ORPHA:{id_val}"
                elif key == "MESH":
                    id_val = f"MESH:{id_val}"
                lexicon_ids.append({"source": source_name, "id": str(id_val)})

        return lexicon_ids

    def _load_anca_lexicon(self, path: Path) -> None:
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name
        loaded = 0

        # 1) Diseases section
        diseases = data.get("diseases", {})
        for disease_key, entry in diseases.items():
            if not isinstance(entry, dict):
                continue

            preferred = (entry.get("preferred_label") or "").strip()
            abbrev = (entry.get("abbreviation") or "").strip()
            synonyms = entry.get("synonyms") or []
            identifiers = entry.get("identifiers", {})

            if not preferred:
                continue

            # Extract lexicon IDs
            lexicon_ids = self._extract_identifiers(identifiers)

            # Register preferred label
            self.entity_kp.add_keyword(preferred, preferred)
            self.entity_canonical[preferred] = preferred
            self.entity_source[preferred] = source
            self.entity_ids[preferred] = lexicon_ids
            loaded += 1

            # Register abbreviation
            if abbrev and len(abbrev) >= 2:
                self.entity_kp.add_keyword(abbrev, abbrev)
                self.entity_canonical[abbrev] = preferred
                self.entity_source[abbrev] = source
                self.entity_ids[abbrev] = lexicon_ids
                loaded += 1

            # Register synonyms
            for syn in synonyms:
                syn = (syn or "").strip()
                if not syn or len(syn) < 2 or syn == abbrev:
                    continue
                self.entity_kp.add_keyword(syn, syn)
                self.entity_canonical[syn] = preferred
                self.entity_source[syn] = source
                self.entity_ids[syn] = lexicon_ids
                loaded += 1

        # 2) Abbreviation expansions section
        abbrev_expansions = data.get("abbreviation_expansions", {})
        for abbrev, entry in abbrev_expansions.items():
            if not abbrev or not isinstance(entry, dict):
                continue

            preferred = (entry.get("preferred") or "").strip()
            if not preferred or len(abbrev) < 2:
                continue

            self.entity_kp.add_keyword(abbrev, abbrev)
            self.entity_canonical[abbrev] = preferred
            self.entity_source[abbrev] = source
            self.entity_ids[abbrev] = []  # No IDs for abbreviation expansions
            loaded += 1

        # 3) Composite terms section
        composite = data.get("composite_terms", {})
        for term, entry in composite.items():
            if not term or not isinstance(entry, dict):
                continue

            expansion = (entry.get("expansion") or "").strip()
            if not expansion or len(term) < 2:
                continue

            self.entity_kp.add_keyword(term, term)
            self.entity_canonical[term] = expansion
            self.entity_source[term] = source
            self.entity_ids[term] = []  # No IDs for composite terms
            loaded += 1

        self._lexicon_stats.append(("ANCA disease", loaded, path.name))

    def _load_igan_lexicon(self, path: Path) -> None:
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name
        loaded = 0

        # 1) Diseases section
        diseases = data.get("diseases", {})
        for disease_key, entry in diseases.items():
            if not isinstance(entry, dict):
                continue

            preferred = (entry.get("preferred_label") or "").strip()
            abbrev = (entry.get("abbreviation") or "").strip()
            synonyms = entry.get("synonyms") or []
            identifiers = entry.get("identifiers", {})

            if not preferred:
                continue

            # Extract lexicon IDs
            lexicon_ids = self._extract_identifiers(identifiers)

            # Register preferred label
            self.entity_kp.add_keyword(preferred, preferred)
            self.entity_canonical[preferred] = preferred
            self.entity_source[preferred] = source
            self.entity_ids[preferred] = lexicon_ids
            loaded += 1

            # Register abbreviation
            if abbrev and len(abbrev) >= 2:
                self.entity_kp.add_keyword(abbrev, abbrev)
                self.entity_canonical[abbrev] = preferred
                self.entity_source[abbrev] = source
                self.entity_ids[abbrev] = lexicon_ids
                loaded += 1

            # Register synonyms
            for syn in synonyms:
                syn = (syn or "").strip()
                if not syn or len(syn) < 2 or syn == abbrev:
                    continue
                self.entity_kp.add_keyword(syn, syn)
                self.entity_canonical[syn] = preferred
                self.entity_source[syn] = source
                self.entity_ids[syn] = lexicon_ids
                loaded += 1

        # 2) Abbreviation expansions section
        abbrev_expansions = data.get("abbreviation_expansions", {})
        for abbrev, entry in abbrev_expansions.items():
            if not abbrev or not isinstance(entry, dict):
                continue

            preferred = (entry.get("preferred") or "").strip()
            if not preferred or len(abbrev) < 2:
                continue

            self.entity_kp.add_keyword(abbrev, abbrev)
            self.entity_canonical[abbrev] = preferred
            self.entity_source[abbrev] = source
            self.entity_ids[abbrev] = []
            loaded += 1

        # 3) Composite terms section
        composite = data.get("composite_terms", {})
        for term, entry in composite.items():
            if not term or not isinstance(entry, dict):
                continue

            expansion = (entry.get("expansion") or "").strip()
            if not expansion or len(term) < 2:
                continue

            self.entity_kp.add_keyword(term, term)
            self.entity_canonical[term] = expansion
            self.entity_source[term] = source
            self.entity_ids[term] = []
            loaded += 1

        # 4) Renal terms section (specific to IgAN lexicon)
        renal_terms = data.get("renal_terms", {})
        for term, entry in renal_terms.items():
            if not term or not isinstance(entry, dict):
                continue

            expansion = (entry.get("expansion") or "").strip()
            if not expansion or len(term) < 2:
                continue

            self.entity_kp.add_keyword(term, term)
            self.entity_canonical[term] = expansion
            self.entity_source[term] = source
            self.entity_ids[term] = []
            loaded += 1

        self._lexicon_stats.append(("IgAN disease", loaded, path.name))

    def _load_pah_lexicon(self, path: Path) -> None:
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name
        loaded = 0

        # 1) Diseases section
        diseases = data.get("diseases", {})
        for disease_key, entry in diseases.items():
            if not isinstance(entry, dict):
                continue

            preferred = (entry.get("preferred_label") or "").strip()
            abbrev = (entry.get("abbreviation") or "").strip()
            synonyms = entry.get("synonyms") or []
            identifiers = entry.get("identifiers", {})

            if not preferred:
                continue

            # Extract lexicon IDs once per disease entry
            lexicon_ids = self._extract_identifiers(identifiers)

            # Register preferred label
            self.entity_kp.add_keyword(preferred, preferred)
            self.entity_canonical[preferred] = preferred
            self.entity_source[preferred] = source
            self.entity_ids[preferred] = lexicon_ids
            loaded += 1

            # Register abbreviation
            if abbrev and len(abbrev) >= 2:
                self.entity_kp.add_keyword(abbrev, abbrev)
                self.entity_canonical[abbrev] = preferred
                self.entity_source[abbrev] = source
                self.entity_ids[abbrev] = lexicon_ids
                loaded += 1

            # Register synonyms
            for syn in synonyms:
                syn = (syn or "").strip()
                if not syn or len(syn) < 2 or syn == abbrev:
                    continue
                self.entity_kp.add_keyword(syn, syn)
                self.entity_canonical[syn] = preferred
                self.entity_source[syn] = source
                self.entity_ids[syn] = lexicon_ids
                loaded += 1

        # 2) Abbreviation expansions section
        abbrev_expansions = data.get("abbreviation_expansions", {})
        for abbrev, entry in abbrev_expansions.items():
            if not abbrev or not isinstance(entry, dict):
                continue

            preferred = (entry.get("preferred") or "").strip()
            if not preferred or len(abbrev) < 2:
                continue

            self.entity_kp.add_keyword(abbrev, abbrev)
            self.entity_canonical[abbrev] = preferred
            self.entity_source[abbrev] = source
            self.entity_ids[abbrev] = []
            loaded += 1

        # 3) Composite terms section
        composite = data.get("composite_terms", {})
        for term, entry in composite.items():
            if not term or not isinstance(entry, dict):
                continue

            expansion = (entry.get("expansion") or "").strip()
            if not expansion or len(term) < 2:
                continue

            self.entity_kp.add_keyword(term, term)
            self.entity_canonical[term] = expansion
            self.entity_source[term] = source
            self.entity_ids[term] = []
            loaded += 1

        # 4) Hemodynamic terms section (specific to PAH lexicon)
        hemodynamic_terms = data.get("hemodynamic_terms", {})
        for term, entry in hemodynamic_terms.items():
            if not term or not isinstance(entry, dict):
                continue

            expansion = (entry.get("expansion") or "").strip()
            if not expansion or len(term) < 2:
                continue

            self.entity_kp.add_keyword(term, term)
            self.entity_canonical[term] = expansion
            self.entity_source[term] = source
            self.entity_ids[term] = []
            loaded += 1

        self._lexicon_stats.append(("PAH disease", loaded, path.name))

    def _load_trial_acronyms(self, path: Path) -> None:
        """Load clinical trial acronyms lexicon (RADAR, APPEAR-C3G, MAINRITSAN, etc.)."""
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        loaded = 0

        for acronym, entry in data.items():
            if not acronym or not isinstance(entry, dict):
                continue

            # Get expansion and regex pattern
            lf = entry.get("canonical_expansion")
            regex_str = entry.get("regex")

            if not lf or not regex_str:
                continue

            # Skip very long expansions (full trial titles)
            # We want abbreviations, not full protocol titles
            if len(lf) > 150:
                lf = f"{acronym} (clinical trial)"

            try:
                case_insensitive = bool(entry.get("case_insensitive", False))
                flags = re.IGNORECASE if case_insensitive else 0
                pattern = re.compile(regex_str, flags)

                # Build lexicon IDs from NCT ID
                lexicon_ids = []
                nct_id = entry.get("nct_id")
                if nct_id:
                    lexicon_ids.append({"source": "ClinicalTrials.gov", "id": nct_id})

                self.abbrev_entries.append(
                    LexiconEntry(
                        sf=acronym,
                        lf=lf,
                        pattern=pattern,
                        source=path.name,
                        lexicon_ids=lexicon_ids,
                        preserve_case=True,  # Preserve case for trial names
                    )
                )
                loaded += 1
            except re.error:
                pass

        self._lexicon_stats.append(("Trial acronyms", loaded, path.name))

    def _load_pro_scales(self, path: Path) -> None:
        """Load PRO scales lexicon (SF-36, PHQ-9, EORTC-QLQ-C30, etc.)."""
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        loaded = 0

        for scale_name, entry in data.items():
            if not scale_name or not isinstance(entry, dict):
                continue

            # Get expansion and regex pattern
            lf = entry.get("canonical_expansion")
            regex_str = entry.get("regex")

            if not lf or not regex_str:
                continue

            try:
                case_insensitive = bool(entry.get("case_insensitive", False))
                flags = re.IGNORECASE if case_insensitive else 0
                pattern = re.compile(regex_str, flags)

                # Build lexicon IDs from example NCT IDs (use first one)
                lexicon_ids = []
                example_ncts = entry.get("example_nct_ids", [])
                if example_ncts and len(example_ncts) > 0:
                    lexicon_ids.append(
                        {"source": "ClinicalTrials.gov", "id": example_ncts[0]}
                    )

                self.abbrev_entries.append(
                    LexiconEntry(
                        sf=scale_name,
                        lf=lf,
                        pattern=pattern,
                        source=path.name,
                        lexicon_ids=lexicon_ids,
                        preserve_case=True,  # Preserve case for PRO scale names
                    )
                )
                loaded += 1
            except re.error:
                pass

        self._lexicon_stats.append(("PRO scales", loaded, path.name))

    def _load_pharma_companies(self, path: Path) -> None:
        """Load pharma companies lexicon (Roche, Novartis, Pfizer, etc.)."""
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name
        loaded = 0

        for company_key, entry in data.items():
            if not company_key or not isinstance(entry, dict):
                continue

            canonical = (entry.get("canonical_name") or "").strip()
            full_name = (entry.get("full_name") or "").strip()
            variants = entry.get("variants") or []

            if not canonical:
                continue

            # Use full_name as the long form if available, otherwise canonical
            lf = full_name if full_name else canonical

            # Register all variants (including canonical name)
            all_terms = set(variants)
            all_terms.add(canonical)
            if full_name:
                all_terms.add(full_name)

            for term in all_terms:
                term = (term or "").strip()
                if not term or len(term) < 2:
                    continue

                self.entity_kp.add_keyword(term, term)
                self.entity_canonical[term] = lf
                self.entity_source[term] = source
                self.entity_ids[term] = []  # No external IDs for pharma companies
                loaded += 1

        self._lexicon_stats.append(("Pharma companies", loaded, path.name))

    # =========================================================================
    # NEW LEXICONS: Meta-Inventory, MONDO, ChEMBL
    # =========================================================================

    def _load_meta_inventory(self, path: Path) -> None:
        """
        Load Meta-Inventory clinical abbreviations (104K+ abbreviations).

        Source: https://github.com/lisavirginia/clinical-abbreviations
        Paper: https://www.nature.com/articles/s41597-021-00929-4

        Format: {SF: {canonical_expansion, regex, expansions: [...]}}
        """
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        loaded = 0

        for sf, entry in data.items():
            if not sf or not isinstance(entry, dict):
                continue

            lf = entry.get("canonical_expansion")
            regex_str = entry.get("regex")

            if not lf or not regex_str:
                continue

            # Skip very short abbreviations
            if len(sf) < 2:
                continue

            try:
                case_insensitive = bool(entry.get("case_insensitive", True))
                flags = re.IGNORECASE if case_insensitive else 0
                pattern = re.compile(regex_str, flags)

                self.abbrev_entries.append(
                    LexiconEntry(
                        sf=sf,
                        lf=lf,
                        pattern=pattern,
                        source="meta-inventory"
                    )
                )
                loaded += 1
            except re.error:
                pass

        self._lexicon_stats.append(("Meta-Inventory", loaded, path.name))

    def _load_mondo_lexicon(self, path: Path) -> None:
        """
        Load MONDO disease ontology.

        Source: https://mondo.monarchinitiative.org/
        Provides unified disease mappings with precise semantics.

        Format: [{label, sources: [{source, id}], synonyms: [...]}]
        """
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name
        loaded = 0

        for entry in data:
            if not isinstance(entry, dict):
                continue

            label = (entry.get("label") or "").strip()
            if not label or len(label) < 3:
                continue

            # Extract MONDO and cross-reference IDs
            sources_list = entry.get("sources", [])
            lexicon_ids = [
                {"source": s.get("source", ""), "id": s.get("id", "")}
                for s in sources_list
                if isinstance(s, dict) and s.get("id")
            ]

            # Use canonical label if this is a synonym entry
            canonical = entry.get("canonical", label)

            self.entity_kp.add_keyword(label, label)
            self.entity_canonical[label] = canonical
            self.entity_source[label] = source
            self.entity_ids[label] = lexicon_ids
            loaded += 1

        self._lexicon_stats.append(("MONDO diseases", loaded, path.name))

    def _load_chembl_lexicon(self, path: Path) -> None:
        """
        Load ChEMBL approved drugs.

        Source: https://www.ebi.ac.uk/chembl/
        Open data drug database with bioactivity information.

        Format: [{label, chembl_id, max_phase, synonyms: [...]}]
        """
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name
        loaded = 0

        # Handle placeholder format
        if isinstance(data, dict) and "drugs" in data:
            drugs = data.get("drugs", [])
        elif isinstance(data, list):
            drugs = data
        else:
            return

        for entry in drugs:
            if not isinstance(entry, dict):
                continue

            label = (entry.get("label") or "").strip()
            if not label or len(label) < 2:
                continue

            chembl_id = entry.get("chembl_id", "")
            synonyms = entry.get("synonyms", [])

            # Build identifier list
            lexicon_ids = []
            if chembl_id:
                lexicon_ids.append({"source": "ChEMBL", "id": chembl_id})

            # Add main drug name
            self.entity_kp.add_keyword(label, label)
            self.entity_canonical[label] = label
            self.entity_source[label] = source
            self.entity_ids[label] = lexicon_ids
            loaded += 1

            # Add synonyms
            for syn in synonyms[:5]:  # Limit synonyms per drug
                syn = (syn or "").strip()
                if syn and len(syn) >= 2 and syn != label:
                    self.entity_kp.add_keyword(syn, syn)
                    self.entity_canonical[syn] = label
                    self.entity_source[syn] = source
                    self.entity_ids[syn] = lexicon_ids
                    loaded += 1

        self._lexicon_stats.append(("ChEMBL drugs", loaded, path.name))

    def _make_context(self, text: str, start: int, end: int) -> str:
        left = max(0, start - self.context_window)
        right = min(len(text), end + self.context_window)
        return text[left:right].replace("\n", " ").strip()

    def _is_valid_match(self, term: str) -> bool:
        """
        Light filter - only block OBVIOUS noise.
        Let the Validation layer (Claude) handle borderline cases.
        """
        if not term:
            return False

        term_lower = term.lower().strip()

        # Block obvious noise (single letters, function words)
        if term_lower in OBVIOUS_NOISE:
            return False

        # Minimum length
        if len(term) < MIN_ABBREV_LENGTH:
            return False

        # Block pure numbers (not abbreviations)
        if term.isdigit():
            return False

        # Block short alphanumeric codes starting with digit (e.g., 1A, 2B, 45)
        if len(term) <= 3 and term[0].isdigit():
            return False

        return True

    def _extract_inline_definitions(
        self, text: str
    ) -> List[Tuple[str, str, int, int]]:
        """
        Extract inline abbreviation definitions using regex patterns.

        Catches patterns that scispacy's Schwartz-Hearst detector might miss:
        1. "long form (ABBREV)" - standard pattern with mixed-case abbreviations
        2. "ABBREV (long form)" - reversed pattern
        3. "ABBREV, the/or/i.e. long form" - comma-separated definitions
        4. "ABBREV = long form" - equals-separated definitions

        Returns:
            List of (short_form, long_form, start, end) tuples
        """
        results: List[Tuple[str, str, int, int]] = []

        # Pattern 1a: "Long Form (ABBREV)" - title case long form
        # Catches mixed-case abbreviations like LoE, LoA
        # Updated to handle hyphenated words like "Five-Factor Score"
        pattern1a = re.compile(
            r"\b((?:[A-Z][a-z]+(?:-[A-Z]?[a-z]+)?\s+){1,7}[A-Za-z]+)\s*"  # Long form: capitalized words (with optional hyphen)
            r"\(([A-Za-z][A-Za-z0-9/-]{1,9})\)",  # (ABBREV) - mixed case allowed
            re.UNICODE
        )

        for match in pattern1a.finditer(text):
            lf = match.group(1).strip()
            sf = match.group(2).strip()

            # Validate: SF should have at least one uppercase
            if not any(c.isupper() for c in sf):
                continue

            # Validate: SF should look like an abbreviation (not a word)
            if sf.lower() == sf:  # All lowercase
                continue

            # Check if SF could plausibly be an abbreviation of LF
            if self._could_be_abbreviation(sf, lf):
                results.append((sf, lf, match.start(), match.end()))

        # Pattern 1b: "long form (ABBREV)" - lowercase long form (common in clinical text)
        # e.g., "level of agreement (LoA)", "eosinophilic GPA (EGPA)"
        pattern1b = re.compile(
            r"\b([a-z][a-z\s]{3,60})\s*"  # Long form: lowercase words
            r"\(([A-Z][A-Za-z0-9/-]{1,9})\)",  # (ABBREV) - must start with uppercase
            re.UNICODE
        )

        for match in pattern1b.finditer(text):
            lf = match.group(1).strip()
            sf = match.group(2).strip()

            # Clean up long form - remove common lead-in phrases that aren't part of the term
            lead_in_patterns = [
                r"^(?:developed\s+for|known\s+as|called|termed|named|referred\s+to\s+as)\s+",
                r"^(?:including|such\s+as|like|e\.?g\.?)\s+",
                r"^(?:is\s+a|was\s+a|are|were)\s+",
            ]
            for pattern in lead_in_patterns:
                lf = re.sub(pattern, "", lf, flags=re.IGNORECASE).strip()

            # Validate: long form should have multiple words
            if len(lf.split()) < 2:
                continue

            # Skip if LF looks like it could be part of a sentence (too long)
            if len(lf.split()) > 8:
                continue

            # Check if SF could plausibly be an abbreviation of LF
            if self._could_be_abbreviation(sf, lf):
                results.append((sf, lf, match.start(), match.end()))

        # Pattern 2: "ABBREV, (the|or|i.e.|ie) long form"
        # e.g., "FV, the final vote" or "GPA, or granulomatosis with polyangiitis"
        pattern2 = re.compile(
            r"\b([A-Z][A-Za-z0-9/-]{1,9})"  # ABBREV
            r",?\s+(?:the|or|i\.?e\.?|namely|meaning)\s+"  # separator
            r"([a-z][a-z\s,/-]{5,60})"  # long form (lowercase start)
            r"(?=[.,;:\)\]\s]|$)",  # followed by punctuation or end
            re.UNICODE
        )

        for match in pattern2.finditer(text):
            sf = match.group(1).strip()
            lf = match.group(2).strip()

            # Clean up long form - remove trailing punctuation
            lf = re.sub(r"[.,;:]+$", "", lf).strip()

            if len(lf) < 5:
                continue

            if self._could_be_abbreviation(sf, lf):
                results.append((sf, lf, match.start(), match.end()))

        # Pattern 3: "ABBREV (long form)" - reversed Schwartz-Hearst
        # e.g., "LoE (level of evidence)"
        pattern3 = re.compile(
            r"\b([A-Z][A-Za-z0-9/-]{1,9})\s*"  # ABBREV
            r"\(([a-z][a-z\s,/-]{5,60})\)",  # (long form)
            re.UNICODE
        )

        for match in pattern3.finditer(text):
            sf = match.group(1).strip()
            lf = match.group(2).strip()

            if self._could_be_abbreviation(sf, lf):
                results.append((sf, lf, match.start(), match.end()))

        # Pattern 4: "ABBREV = long form" or "ABBREV: long form"
        pattern4 = re.compile(
            r"\b([A-Z][A-Za-z0-9/-]{1,9})\s*"  # ABBREV
            r"[=:]\s*"  # = or :
            r"([a-z][a-z\s,/-]{5,60})"  # long form
            r"(?=[.,;:\)\]\s]|$)",
            re.UNICODE
        )

        for match in pattern4.finditer(text):
            sf = match.group(1).strip()
            lf = match.group(2).strip()

            # Clean up long form
            lf = re.sub(r"[.,;:]+$", "", lf).strip()

            if len(lf) < 5:
                continue

            if self._could_be_abbreviation(sf, lf):
                results.append((sf, lf, match.start(), match.end()))

        return results

    def _could_be_abbreviation(self, sf: str, lf: str) -> bool:
        """
        Check if short form could plausibly be an abbreviation of long form.

        Uses a simple heuristic: at least half of the SF letters should
        appear in LF (in order), OR the SF appears as initials of LF words.
        """
        sf_upper = sf.upper()
        lf_lower = lf.lower()
        lf_words = lf_lower.split()

        # Check 1: Initials match
        # Get first letter of each word in LF
        initials = "".join(w[0] for w in lf_words if w)
        if initials.upper() == sf_upper:
            return True

        # Check 2: Partial initials match (for abbreviations that skip words)
        # Allow some flexibility - at least 50% of SF chars match LF initials
        matching = sum(1 for c in sf_upper if c.lower() in initials)
        if len(sf) >= 2 and matching >= len(sf) * 0.5:
            return True

        # Check 3: Letters appear in order in LF
        lf_idx = 0
        matches = 0
        for c in sf_upper:
            # Find this character in the remaining LF
            found = lf_lower.find(c.lower(), lf_idx)
            if found >= 0:
                matches += 1
                lf_idx = found + 1

        # At least half the SF letters should be found in order
        if matches >= len(sf) * 0.5:
            return True

        return False
