# corpus_metadata/C_generators/C04_strategy_flashtext.py
"""
FlashText-based lexicon matching for abbreviation and entity extraction.

Uses:
- Regex patterns for abbreviation lexicon (handles spacing/case variations)
- FlashText for disease/entity lexicons (fast exact matching)
- scispacy NER for biomedical entity recognition
"""

from __future__ import annotations

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

from A_core.A01_domain_models import (
    Candidate,
    Coordinate,
    FieldType,
    GeneratorType,
    LexiconIdentifier,
    ProvenanceMetadata,
)
from A_core.A02_interfaces import BaseCandidateGenerator
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from B_parsing.B02_doc_graph import DocumentGraph

# Import noise filters and constants from modularized file
from C_generators.C04a_noise_filters import (
    BAD_LONG_FORMS,
    LexiconEntry,
    MIN_ABBREV_LENGTH,
    OBVIOUS_NOISE,
    WRONG_EXPANSION_BLACKLIST,
)

# Import lexicon loading and inline definition detection from modularized files
from C_generators.C04b_lexicon_loaders import LexiconLoaderMixin
from C_generators.C04c_inline_definition_detector import InlineDefinitionDetectorMixin


class RegexLexiconGenerator(
    LexiconLoaderMixin, InlineDefinitionDetectorMixin, BaseCandidateGenerator
):
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
        self.umls_max_blocks = int(self.config.get("umls_max_blocks", 500))

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
        # Pharma companies now extracted via dedicated C18_strategy_pharma detector
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

                    # Skip matches that are part of a longer hyphenated term
                    # e.g., skip "APPEAR" when the text contains "APPEAR-C3G"
                    if end < len(text) and text[end] == '-':
                        # Check if hyphen is followed by alphanumeric chars
                        rest = text[end + 1:]
                        if rest and (rest[0].isalnum()):
                            continue  # Part of a longer hyphenated term

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
                # For large documents, only do UMLS lookup for first N blocks to avoid O(n^2) scaling
                umls_char_limit = blocks_data[min(self.umls_max_blocks, len(blocks_data)) - 1][3] if blocks_data else 0
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

                    # Try UMLS linker first for expansion (only for first N blocks in large docs)
                    lf_from_umls = None
                    umls_cui = None
                    if ent.start_char < umls_char_limit and hasattr(ent._, "kb_ents") and ent._.kb_ents:
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
                        generator_type=GeneratorType.INLINE_DEFINITION,
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
        generator_type: Optional[GeneratorType] = None,
    ) -> Candidate:
        ctx = self._make_context(text, start, end)

        loc = Coordinate(
            page_num=int(block.page_num),
            block_id=str(block.id),
            bbox=block.bbox,
        )

        # Use provided generator_type or fall back to self.generator_type
        gen_type = generator_type or self.generator_type

        prov = ProvenanceMetadata(
            pipeline_version=self.pipeline_version,
            run_id=self.run_id,
            doc_fingerprint=self.doc_fingerprint_default,
            generator_name=gen_type,
            rule_version=rule_version,
            lexicon_source=lexicon_source,
            lexicon_ids=lexicon_ids,
        )

        return Candidate(
            doc_id=doc.doc_id,
            field_type=FieldType.DEFINITION_PAIR,
            generator_type=gen_type,
            short_form=short_form,
            long_form=long_form,
            context_text=ctx,
            context_location=loc,
            initial_confidence=0.95,
            provenance=prov,
        )

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


__all__ = ["RegexLexiconGenerator"]
