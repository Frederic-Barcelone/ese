# corpus_metadata/H_pipeline/H02_abbreviation_pipeline.py
"""
Abbreviation extraction pipeline.

Handles the complete abbreviation extraction workflow including PDF parsing,
candidate generation, filtering, LLM validation, and normalization.
"""

from __future__ import annotations

import heapq
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser
    from B_parsing.B02_doc_graph import DocumentGraph
    from B_parsing.B03_table_extractor import TableExtractor
    from A_core.A01_domain_models import (
        Candidate,
        ExtractedEntity,
        ValidationStatus,
    )
    from A_core.A04_heuristics_config import HeuristicsConfig, HeuristicsCounters
    from D_validation.D02_llm_engine import ClaudeClient, LLMEngine
    from D_validation.D03_validation_logger import ValidationLogger
    from E_normalization.E01_term_mapper import TermMapper
    from E_normalization.E02_disambiguator import Disambiguator
    from E_normalization.E07_deduplicator import Deduplicator
    from E_normalization.E06_nct_enricher import NCTEnricher
    from C_generators.C15_vlm_table_extractor import VLMTableExtractor

from A_core.A03_provenance import hash_string
from Z_utils.Z02_text_helpers import (
    extract_context_snippet,
    normalize_lf_for_dedup,
    has_numeric_evidence,
    is_valid_sf_form,
    score_lf_quality,
)


class AbbreviationPipeline:
    """
    Pipeline for extracting and validating abbreviations from documents.

    Coordinates the entire abbreviation extraction workflow including:
    - PDF parsing into document graphs
    - Candidate generation using multiple strategies
    - Candidate filtering and deduplication
    - Heuristic-based auto-approve/reject
    - LLM-based validation
    - Normalization and final deduplication
    """

    def __init__(
        self,
        run_id: str,
        pipeline_version: str,
        parser: "PDFToDocGraphParser",
        table_extractor: "TableExtractor",
        generators: List[Any],
        heuristics: "HeuristicsConfig",
        term_mapper: "TermMapper",
        disambiguator: "Disambiguator",
        deduplicator: "Deduplicator",
        logger: "ValidationLogger",
        claude_client: Optional["ClaudeClient"] = None,
        llm_engine: Optional["LLMEngine"] = None,
        vlm_table_extractor: Optional["VLMTableExtractor"] = None,
        nct_enricher: Optional["NCTEnricher"] = None,
        rare_disease_lookup: Optional[Dict[str, str]] = None,
        use_vlm_tables: bool = False,
        use_normalization: bool = True,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        """
        Initialize the abbreviation pipeline.

        Args:
            run_id: Unique identifier for this pipeline run
            pipeline_version: Version string for the pipeline
            parser: PDF parser component
            table_extractor: Table extraction component
            generators: List of candidate generators
            heuristics: Heuristics configuration
            term_mapper: Term mapping/normalization component
            disambiguator: Disambiguation component
            deduplicator: Deduplication component
            logger: Validation logger
            claude_client: Optional Claude API client
            llm_engine: Optional LLM validation engine
            vlm_table_extractor: Optional VLM table extractor
            nct_enricher: Optional NCT enricher
            rare_disease_lookup: Optional rare disease SF->LF lookup
            use_vlm_tables: Whether to use VLM for table extraction
            use_normalization: Whether to run normalization
            model: Model name for LLM calls
        """
        self.run_id = run_id
        self.pipeline_version = pipeline_version
        self.parser = parser
        self.table_extractor = table_extractor
        self.generators = generators
        self.heuristics = heuristics
        self.term_mapper = term_mapper
        self.disambiguator = disambiguator
        self.deduplicator = deduplicator
        self.logger = logger
        self.claude_client = claude_client
        self.llm_engine = llm_engine
        self.vlm_table_extractor = vlm_table_extractor
        self.nct_enricher = nct_enricher
        self.rare_disease_lookup = rare_disease_lookup or {}
        self.use_vlm_tables = use_vlm_tables
        self.use_normalization = use_normalization
        self.model = model

    def parse_pdf(self, pdf_path: Path, output_dir: Path) -> "DocumentGraph":
        """Stage 1: Parse PDF into DocumentGraph."""
        print("\n[1/12] Parsing PDF...")
        start = time.time()

        doc = self.parser.parse(str(pdf_path), image_output_dir=str(output_dir))

        # Extract tables with VLM if available (300 DPI for optimal VLM reading)
        doc = self.table_extractor.populate_document_graph(
            doc,
            str(pdf_path),
            render_images=True,
            use_vlm=self.use_vlm_tables and self.vlm_table_extractor is not None,
            vlm_extractor=self.vlm_table_extractor,
        )

        total_blocks = sum(len(p.blocks) for p in doc.pages.values())
        total_tables = sum(len(p.tables) for p in doc.pages.values())

        print(f"  Pages: {len(doc.pages)}")
        print(f"  Blocks: {total_blocks}")
        print(f"  Tables: {total_tables}")
        print(f"  Time: {time.time() - start:.2f}s")

        return doc

    def generate_candidates(self, doc: "DocumentGraph") -> Tuple[List["Candidate"], str]:
        """Stage 2: Generate and deduplicate candidates."""

        print("\n[2/12] Generating candidates...")
        start = time.time()

        all_candidates = []
        for gen in self.generators:
            candidates = gen.extract(doc)
            print(f"  {gen.generator_type.value}: {len(candidates)} candidates")
            all_candidates.extend(candidates)

        # Deduplicate using normalized LF to handle punctuation variants
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            # Use normalize_lf_for_dedup to handle hyphen/dash variants
            key = (c.short_form.upper(), normalize_lf_for_dedup(c.long_form or ""))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        # Build full text
        full_text = " ".join(
            block.text for block in doc.iter_linear_blocks() if block.text
        )

        print(f"  Total unique: {len(unique_candidates)}")
        print(f"  Time: {time.time() - start:.2f}s")

        return unique_candidates, full_text

    def filter_candidates(
        self,
        candidates: List["Candidate"],
        full_text: str,
    ) -> Tuple[List["Candidate"], set, Counter, int]:
        """Filter and reduce candidates before validation."""
        from A_core.A01_domain_models import GeneratorType

        # Corroborated SFs (found by non-lexicon generators)
        corroborated_sfs = {
            c.short_form.upper()
            for c in candidates
            if c.generator_type != GeneratorType.LEXICON_MATCH
        }

        # Count SF occurrences
        abbrev_pattern = r"\b[A-Za-z][A-Za-z0-9]{1,11}\b"
        all_tokens = re.findall(abbrev_pattern, full_text)
        word_counts = Counter(
            t.upper() for t in all_tokens if any(c.isupper() for c in t)
        )

        # Keep lexicon matches if corroborated or frequent
        def should_keep(c: "Candidate") -> bool:
            if c.generator_type != GeneratorType.LEXICON_MATCH:
                return True
            sf_upper = c.short_form.upper()
            return sf_upper in corroborated_sfs or word_counts.get(sf_upper, 0) >= 2

        needs_validation = [c for c in candidates if should_keep(c)]

        # Reduce lexicon matches (dedup by SF, filter by form)
        allowed_2letter = self.heuristics.allowed_2letter_sfs
        allowed_mixed = self.heuristics.allowed_mixed_case

        lexicon_by_sf: Dict[str, List["Candidate"]] = {}
        non_lexicon = []
        sf_form_rejected = 0

        for c in needs_validation:
            if c.generator_type == GeneratorType.LEXICON_MATCH:
                ctx = c.context_text or ""
                if not is_valid_sf_form(
                    c.short_form, ctx, allowed_2letter, allowed_mixed
                ):
                    sf_form_rejected += 1
                    continue
                sf_upper = c.short_form.upper()
                lexicon_by_sf.setdefault(sf_upper, []).append(c)
            else:
                non_lexicon.append(c)

        # Keep top 2 LFs per SF (cache lowercase for performance)
        full_text_lower = full_text.lower()
        deduped_lexicon = []
        for sf, cands in lexicon_by_sf.items():
            if len(cands) <= 2:
                deduped_lexicon.extend(cands)
            else:
                # Use heapq.nlargest instead of full sort for O(n) vs O(n log n)
                top_2 = heapq.nlargest(
                    2,
                    cands,
                    key=lambda c: score_lf_quality(c, full_text, full_text_lower),
                )
                deduped_lexicon.extend(top_2)

        lexicon_before = sum(len(v) for v in lexicon_by_sf.values()) + sf_form_rejected
        print(
            f"  LEXICON_MATCH reduction: {lexicon_before} -> {len(deduped_lexicon)} "
            f"(dedup: {lexicon_before - sf_form_rejected - len(deduped_lexicon)}, form filter: {sf_form_rejected})"
        )

        return (
            non_lexicon + deduped_lexicon,
            corroborated_sfs,
            word_counts,
            len(candidates) - len(non_lexicon + deduped_lexicon),
        )

    def apply_heuristics(
        self,
        candidates: List["Candidate"],
        counters: "HeuristicsCounters",
    ) -> Tuple[List[Tuple["Candidate", "ExtractedEntity"]], List["Candidate"]]:
        """Apply auto-approve/reject heuristics. Returns (auto_results, llm_candidates)."""
        from A_core.A01_domain_models import ValidationStatus
        from A_core.A04_heuristics_config import check_context_match, check_trial_id

        auto_results = []
        llm_candidates = []

        for c in candidates:
            sf_upper = c.short_form.upper()
            ctx = c.context_text or ""

            # Auto-reject blacklisted
            if sf_upper in self.heuristics.sf_blacklist:
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.REJECTED,
                    0.95,
                    "Blacklisted: not a valid abbreviation",
                    ["auto_rejected_blacklist"],
                    {"auto": "blacklist"},
                )
                auto_results.append((c, entity))
                counters.blacklisted_fp_count += 1
                continue

            # Auto-reject Figure/Table references
            if re.match(r"^(Figure|Table|Fig|Figures|Tables)\s*S?\d*[A-Za-z]?(-S?\d+)?$", c.short_form, re.IGNORECASE):
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.REJECTED,
                    0.95,
                    "Figure/Table reference, not an abbreviation",
                    ["auto_rejected_form"],
                    {"auto": "figure_table_ref"},
                )
                auto_results.append((c, entity))
                counters.form_filter_rejected += 1
                continue

            # Auto-reject author initials pattern
            if re.match(r"^[A-Z]\.[A-Z]\.$", c.short_form):
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.REJECTED,
                    0.95,
                    "Author initials pattern, not an abbreviation",
                    ["auto_rejected_form"],
                    {"auto": "author_initials"},
                )
                auto_results.append((c, entity))
                counters.form_filter_rejected += 1
                continue

            # Auto-reject context mismatch
            if not check_context_match(c.short_form, ctx, self.heuristics):
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.REJECTED,
                    0.90,
                    "Rejected: ambiguous SF without required context",
                    ["auto_rejected_context"],
                    {"auto": "context_mismatch"},
                )
                auto_results.append((c, entity))
                counters.context_rejected += 1
                continue

            # Auto-reject trial IDs
            if check_trial_id(c.short_form, self.heuristics):
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.REJECTED,
                    0.90,
                    "Excluded: trial identifier (NCT number)",
                    ["auto_rejected_trial_id"],
                    {"auto": "trial_id_excluded"},
                )
                auto_results.append((c, entity))
                counters.trial_id_excluded += 1
                continue

            # Auto-approve stats with numeric evidence
            if sf_upper in self.heuristics.stats_abbrevs and has_numeric_evidence(
                ctx, c.short_form
            ):
                canonical_lf = self.heuristics.stats_abbrevs.get(sf_upper)
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.VALIDATED,
                    0.90,
                    "Stats abbreviation with numeric evidence",
                    ["auto_approved_stats"],
                    {"auto": "stats_whitelist"},
                    long_form_override=canonical_lf,
                )
                auto_results.append((c, entity))
                counters.recovered_by_stats_whitelist += 1
                continue

            # Auto-approve country codes
            if sf_upper in self.heuristics.country_abbrevs:
                canonical_lf = self.heuristics.country_abbrevs.get(sf_upper)
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.VALIDATED,
                    0.90,
                    "Country code abbreviation",
                    ["auto_approved_country"],
                    {"auto": "country_code"},
                    long_form_override=canonical_lf,
                )
                auto_results.append((c, entity))
                counters.recovered_by_country_code += 1
                continue

            # Auto-reject common words
            if sf_upper in self.heuristics.common_words:
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.REJECTED,
                    0.95,
                    "Common English word, not an abbreviation",
                    ["auto_rejected"],
                    {"auto": "rejected_common_word"},
                )
                auto_results.append((c, entity))
                counters.common_word_rejected += 1
                continue

            # Auto-reject malformed long forms
            lf = c.long_form or ""
            if lf:
                lf_lower = lf.lower()
                partial_starters = (
                    "and ", "or ", "our ", "the ", "a ", "an ",
                    "whereas ", "although ", "because ", "while ",
                    "from the ", "in the ", "of the ", "to the ",
                )
                if any(lf_lower.startswith(s) for s in partial_starters):
                    entity = self._create_entity_from_candidate(
                        c,
                        ValidationStatus.REJECTED,
                        0.90,
                        "Malformed LF: starts with article/conjunction",
                        ["auto_rejected_form"],
                        {"auto": "malformed_lf"},
                    )
                    auto_results.append((c, entity))
                    counters.form_filter_rejected += 1
                    continue

                # Reject LFs with unclosed brackets
                if lf.count("[") != lf.count("]") or lf.count("(") != lf.count(")"):
                    entity = self._create_entity_from_candidate(
                        c,
                        ValidationStatus.REJECTED,
                        0.90,
                        "Malformed LF: unclosed brackets",
                        ["auto_rejected_form"],
                        {"auto": "malformed_lf"},
                    )
                    auto_results.append((c, entity))
                    counters.form_filter_rejected += 1
                    continue

            llm_candidates.append(c)

        return auto_results, llm_candidates

    def validate_with_llm(
        self,
        llm_candidates: List["Candidate"],
        batch_delay_ms: float,
    ) -> List["ExtractedEntity"]:
        """Validate candidates using LLM."""
        assert self.llm_engine is not None, (
            "LLM engine must be initialized for validation"
        )
        results = []

        # Split by generator type
        explicit_candidates = []
        lexicon_candidates = []

        for c in llm_candidates:
            gen = c.generator_type.value if c.generator_type else ""
            if "syntax" in gen.lower() or "glossary" in gen.lower():
                explicit_candidates.append(c)
            else:
                lexicon_candidates.append(c)

        print(f"  Batch (explicit pairs): {len(explicit_candidates)}")
        print(f"  Individual (lexicon): {len(lexicon_candidates)}")

        # Batch validate explicit pairs
        if explicit_candidates:
            try:
                val_start = time.time()
                batch_results = self.llm_engine.verify_candidates_batch(
                    explicit_candidates, batch_size=10, delay_ms=batch_delay_ms
                )
                elapsed_ms = (time.time() - val_start) * 1000

                for candidate, entity in zip(explicit_candidates, batch_results):
                    self.logger.log_validation(
                        candidate,
                        entity,
                        entity.raw_llm_response,
                        elapsed_ms / len(explicit_candidates),
                    )
                    results.append(entity)
            except Exception as e:
                print(f"  [WARN] Batch error, falling back to individual: {e}")
                for candidate in explicit_candidates:
                    try:
                        entity = self.llm_engine.verify_candidate(candidate)
                        self.logger.log_validation(
                            candidate, entity, entity.raw_llm_response, 0
                        )
                        results.append(entity)
                    except Exception as e2:
                        self.logger.log_error(candidate, str(e2))

        # Batch validation for lexicon matches
        if lexicon_candidates:
            try:
                val_start = time.time()
                batch_results = self.llm_engine.verify_candidates_batch(
                    lexicon_candidates, batch_size=15, delay_ms=batch_delay_ms
                )
                elapsed_ms = (time.time() - val_start) * 1000

                for candidate, entity in zip(lexicon_candidates, batch_results):
                    self.logger.log_validation(
                        candidate,
                        entity,
                        entity.raw_llm_response,
                        elapsed_ms / len(lexicon_candidates),
                    )
                    results.append(entity)
            except Exception as e:
                print(f"  [WARN] Batch error, falling back to individual: {e}")
                for candidate in lexicon_candidates:
                    try:
                        entity = self.llm_engine.verify_candidate(candidate)
                        self.logger.log_validation(
                            candidate, entity, entity.raw_llm_response, 0
                        )
                        results.append(entity)
                    except Exception as e2:
                        self.logger.log_error(candidate, str(e2))

        return results

    def search_missing_abbreviations(
        self,
        doc_id: str,
        full_text: str,
        found_sfs: set,
        counters: "HeuristicsCounters",
    ) -> List["ExtractedEntity"]:
        """Search for abbreviations missed by generators (PASO C/D)."""
        from E_normalization.E06_nct_enricher import enrich_trial_acronym

        results = []

        # PASO C: Hyphenated abbreviations
        for hyph_sf, hyph_lf in self.heuristics.hyphenated_abbrevs.items():
            if hyph_sf.upper() in found_sfs:
                continue
            pattern = rf"\b{re.escape(hyph_sf)}\b"
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                # If long form is "trial name", enrich from ClinicalTrials.gov API
                actual_lf = hyph_lf
                if hyph_lf.lower() == "trial name":
                    enriched_title = enrich_trial_acronym(hyph_sf)
                    if enriched_title:
                        actual_lf = enriched_title
                        print(f"    Enriched trial '{hyph_sf}': {enriched_title[:60]}...")

                entity = self._create_entity_from_search(
                    doc_id,
                    full_text,
                    match,
                    actual_lf,
                    0.85,
                    ["auto_approved_hyphenated"],
                    "hyphenated_detector:v1.0",
                    "hyphenated",
                )
                results.append(entity)
                found_sfs.add(hyph_sf.upper())
                counters.recovered_by_hyphen += 1

        if counters.recovered_by_hyphen > 0:
            print(f"  Hyphenated detected (PASO C): {counters.recovered_by_hyphen}")

        # Direct search abbreviations
        for direct_sf, direct_lf in self.heuristics.direct_search_abbrevs.items():
            if direct_sf.upper() in found_sfs:
                continue
            pattern = (
                re.escape(direct_sf)
                if " " in direct_sf
                else rf"\b{re.escape(direct_sf)}\b"
            )
            match = re.search(pattern, full_text)
            if match:
                entity = self._create_entity_from_search(
                    doc_id,
                    full_text,
                    match,
                    direct_lf,
                    0.85,
                    ["auto_approved_direct_search"],
                    "direct_search:v1.0",
                    "direct_search",
                )
                results.append(entity)
                found_sfs.add(direct_sf.upper())
                counters.recovered_by_direct_search += 1

        if counters.recovered_by_direct_search > 0:
            print(f"  Direct search detected: {counters.recovered_by_direct_search}")

        return results

    def extract_sf_only_with_llm(
        self,
        doc_id: str,
        full_text: str,
        found_sfs: set,
        counters: "HeuristicsCounters",
        delay_ms: float = 100,
    ) -> List["ExtractedEntity"]:
        """PASO D: Extract SF-only using LLM."""
        from A_core.A01_domain_models import (
            Coordinate,
            EvidenceSpan,
            ExtractedEntity,
            FieldType,
            GeneratorType,
            ProvenanceMetadata,
            ValidationStatus,
        )
        from C_generators.C04_strategy_flashtext import (
            BAD_LONG_FORMS,
            WRONG_EXPANSION_BLACKLIST,
        )

        print("\n  Running LLM SF-only extractor (PASO D)...")
        if self.claude_client is None:
            return []
        results = []

        # Sample chunks
        chunk_size = 3000
        text_chunks = [
            full_text[i : i + chunk_size]
            for i in range(0, min(len(full_text), 15000), chunk_size)
        ]

        sf_extraction_prompt = """You are an expert at identifying medical/scientific abbreviations and their definitions in text.

Given the following text, identify ALL abbreviations and their long form definitions. An abbreviation is typically:
- 2-10 characters long
- Contains uppercase letters (e.g., "FDA", "eGFR", "IL-6", "C3a", "AN")
- May contain numbers or hyphens
- Represents a longer term/concept

IMPORTANT RULES:
1. Only return abbreviations that EXACTLY appear in the text (case-sensitive)
2. Extract the full long form/definition if it appears in the text (e.g., "Acanthosis nigricans (AN)" -> sf: "AN", lf: "Acanthosis nigricans")
3. Do NOT include common words (the, and, for, etc.)
4. Do NOT include author names, months, or country names
5. Include mixed-case abbreviations like "eGFR", "IgA", "C3a"
6. If the long form is not explicitly defined in the text, set lf to null

Already found (DO NOT include these): {already_found}

Text to analyze:
{text}

Return a JSON array of objects with "sf" (short form) and "lf" (long form, or null if not defined).
Example: [{{"sf": "AN", "lf": "Acanthosis nigricans"}}, {{"sf": "FDA", "lf": null}}]
Return ONLY the JSON array, nothing else."""

        llm_sf_candidates: Dict[str, Optional[str]] = {}  # sf -> lf mapping
        llm_errors = 0

        for i, chunk in enumerate(text_chunks):
            if i > 0 and delay_ms > 0:
                time.sleep(delay_ms / 1000)
            try:
                prompt = sf_extraction_prompt.format(
                    already_found=", ".join(sorted(found_sfs)[:50]), text=chunk
                )
                response = self.claude_client.complete_json_any(
                    system_prompt="You are an abbreviation extraction assistant. Return only valid JSON arrays of objects.",
                    user_prompt=prompt,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=1000,
                    top_p=1.0,
                )
                if isinstance(response, list):
                    for item in response:
                        if isinstance(item, dict):
                            sf = item.get("sf", "")
                            lf = item.get("lf")
                            if isinstance(sf, str) and sf.strip():
                                sf_clean = sf.strip()
                                # Keep the LF if we found one, or update if new one is better
                                if sf_clean not in llm_sf_candidates or (
                                    lf and not llm_sf_candidates.get(sf_clean)
                                ):
                                    llm_sf_candidates[sf_clean] = (
                                        lf.strip()
                                        if isinstance(lf, str) and lf
                                        else None
                                    )
                        elif isinstance(item, str) and item.strip():
                            # Backward compatibility: handle plain strings
                            sf_clean = item.strip()
                            if sf_clean not in llm_sf_candidates:
                                llm_sf_candidates[sf_clean] = None
            except Exception:
                llm_errors += 1

        print(
            f"    LLM chunks: {len(text_chunks)}, errors: {llm_errors}, candidates: {len(llm_sf_candidates)}"
        )

        # Validate and add LLM-found SFs
        blacklist = self.heuristics.common_words | self.heuristics.sf_blacklist

        for sf_candidate, lf_candidate in llm_sf_candidates.items():
            sf_upper = sf_candidate.upper()

            if sf_upper in found_sfs or sf_upper in blacklist:
                continue

            # Verify SF exists in text
            if sf_candidate not in full_text:
                match = re.search(
                    rf"\b{re.escape(sf_candidate)}\b", full_text, re.IGNORECASE
                )
                if not match:
                    continue
                sf_candidate = match.group()

            # Basic form validation
            if (
                len(sf_candidate) < 2
                or len(sf_candidate) > 15
                or not any(c.isupper() for c in sf_candidate)
            ):
                continue

            # Get context for form validation
            idx = full_text.find(sf_candidate)
            if idx == -1:
                idx = full_text.lower().find(sf_candidate.lower())
            temp_ctx = full_text[max(0, idx - 200):idx + len(sf_candidate) + 200] if idx >= 0 else ""

            # Apply the same form validation used for other candidates
            if not is_valid_sf_form(
                sf_candidate,
                temp_ctx,
                self.heuristics.allowed_2letter_sfs,
                self.heuristics.allowed_mixed_case
            ):
                continue

            # Skip likely author initials: short (2-5 char) all-uppercase with no expansion
            # appearing in author/contributor context
            if (
                lf_candidate is None
                and len(sf_candidate) <= 5
                and sf_candidate.isupper()
            ):
                ctx_lower = temp_ctx.lower()
                author_keywords = [
                    "contributor", "author", "academic", "steering committee",
                    "wrote the", "drafted", "reviewed", "edited", "et al",
                    "declaration of interests", "conflicts of interest"
                ]
                if any(kw in ctx_lower for kw in author_keywords):
                    continue  # Likely author initials, skip

            # Find context
            idx = full_text.find(sf_candidate)
            if idx == -1:
                idx = full_text.lower().find(sf_candidate.lower())

            context_snippet = extract_context_snippet(
                full_text, idx, idx + len(sf_candidate)
            )
            ctx_hash = hash_string(context_snippet)

            primary = EvidenceSpan(
                text=context_snippet,
                location=Coordinate(page_num=1),
                scope_ref=ctx_hash,
                start_char_offset=max(0, idx - (idx - 100 if idx > 100 else 0)),
                end_char_offset=max(0, idx - (idx - 100 if idx > 100 else 0))
                + len(sf_candidate),
            )

            # Lexicon fallback: if LLM found SF but no LF, check rare disease lexicon
            lexicon_fallback_used = False
            if not lf_candidate:
                sf_upper = sf_candidate.upper()
                sf_lower = sf_candidate.lower()
                if sf_upper in self.rare_disease_lookup:
                    candidate_lf = self.rare_disease_lookup[sf_upper]
                    lf_lower = candidate_lf.lower()
                    # Filter out known wrong SF -> LF pairs and bad long forms
                    if (sf_lower, lf_lower) not in WRONG_EXPANSION_BLACKLIST and lf_lower not in BAD_LONG_FORMS:
                        lf_candidate = candidate_lf
                        lexicon_fallback_used = True

            prov = ProvenanceMetadata(
                pipeline_version=self.pipeline_version,
                run_id=self.run_id,
                doc_fingerprint="llm_sf_extraction",
                generator_name=GeneratorType.LEXICON_MATCH,
                rule_version="llm_sf_lf_extractor:v2.0",
                lexicon_source="orchestrator:llm_extraction"
                + (":lexicon_fallback" if lexicon_fallback_used else ""),
            )

            # Determine field type based on whether we have LF
            has_lf = bool(lf_candidate)
            field_type = (
                FieldType.DEFINITION_PAIR if has_lf else FieldType.SHORT_FORM_ONLY
            )
            confidence = 0.85 if has_lf else 0.75

            entity = ExtractedEntity(
                candidate_id=uuid.uuid4(),
                doc_id=doc_id,
                field_type=field_type,
                short_form=sf_candidate,
                long_form=lf_candidate,
                primary_evidence=primary,
                supporting_evidence=[],
                status=ValidationStatus.VALIDATED,
                confidence_score=confidence,
                rejection_reason=None,
                validation_flags=["llm_extracted"] if has_lf else ["llm_sf_extracted"],
                provenance=prov,
                raw_llm_response={"auto": "llm_sf_lf_extractor"},
            )
            results.append(entity)
            found_sfs.add(sf_upper)
            counters.recovered_by_llm_sf_only += 1

        print(f"  LLM extracted (PASO D): {counters.recovered_by_llm_sf_only}")
        return results

    def normalize_results(
        self, results: List["ExtractedEntity"], full_text: str
    ) -> List["ExtractedEntity"]:
        """Stage 3.5: Normalize, disambiguate, and deduplicate results."""
        if not self.use_normalization:
            print("\n[4/12] Normalization SKIPPED (disabled in config)")
            return results

        print("\n[4/12] Normalizing, disambiguating & deduplicating...")

        # Step 1: Normalize
        normalized_count = 0
        for i, entity in enumerate(results):
            normalized = self.term_mapper.normalize(entity)
            if "normalized" in (normalized.validation_flags or []):
                normalized_count += 1
            results[i] = normalized

        # Step 1.5: NCT enrichment
        nct_enriched_count = 0
        if self.nct_enricher is not None:
            nct_enriched_count = self._enrich_nct_entities(results)

        # Step 2: Disambiguate
        results = self.disambiguator.resolve(results, full_text)
        disambiguated_count = sum(
            1 for r in results if "disambiguated" in (r.validation_flags or [])
        )

        # Step 3: Deduplicate
        count_before = len(results)
        results = self.deduplicator.deduplicate(results)
        count_after = len(results)
        dedup_removed = count_before - count_after

        print(f"  Normalized: {normalized_count}")
        if nct_enriched_count > 0:
            print(f"  NCT enriched: {nct_enriched_count}")
        print(f"  Disambiguated: {disambiguated_count}")
        print(f"  Deduplicated: {dedup_removed} duplicates merged")

        return results

    def _enrich_nct_entities(self, results: List["ExtractedEntity"]) -> int:
        """Enrich NCT identifiers with trial titles."""
        if self.nct_enricher is None:
            return 0

        nct_pattern = re.compile(r"^NCT\d{8}$", re.IGNORECASE)
        enriched_count = 0

        for i, entity in enumerate(results):
            if entity.long_form:
                continue

            sf = (entity.short_form or "").strip().upper()
            if not nct_pattern.match(sf):
                continue

            info = self.nct_enricher.enrich(sf)
            if info and info.long_form:
                new_flags = list(entity.validation_flags or [])
                if "nct_enriched" not in new_flags:
                    new_flags.append("nct_enriched")

                results[i] = entity.model_copy(update={
                    "long_form": info.long_form,
                    "validation_flags": new_flags,
                })
                enriched_count += 1

        return enriched_count

    def _create_entity_from_candidate(
        self,
        candidate: "Candidate",
        status: "ValidationStatus",
        confidence: float,
        reason: str,
        flags: List[str],
        raw_response: Dict,
        long_form_override: Optional[str] = None,
    ) -> "ExtractedEntity":
        """Create ExtractedEntity from a Candidate."""
        from A_core.A01_domain_models import EvidenceSpan, ExtractedEntity, ValidationStatus

        context = (candidate.context_text or "").strip()
        ctx_hash = hash_string(context) if context else "no_context"
        primary = EvidenceSpan(
            text=context,
            location=candidate.context_location,
            scope_ref=ctx_hash,
            start_char_offset=0,
            end_char_offset=len(context),
        )
        return ExtractedEntity(
            candidate_id=candidate.id,
            doc_id=candidate.doc_id,
            field_type=candidate.field_type,
            short_form=candidate.short_form.strip(),
            long_form=long_form_override
            or (candidate.long_form.strip() if candidate.long_form else None),
            primary_evidence=primary,
            supporting_evidence=[],
            status=status,
            confidence_score=confidence,
            rejection_reason=reason if status == ValidationStatus.REJECTED else None,
            validation_flags=flags,
            provenance=candidate.provenance,
            raw_llm_response=raw_response,
        )

    def _create_entity_from_search(
        self,
        doc_id: str,
        full_text: str,
        match: re.Match,
        long_form: Optional[str],
        confidence: float,
        flags: List[str],
        rule_version: str,
        lexicon_source: str,
    ) -> "ExtractedEntity":
        """Create ExtractedEntity from a text search match."""
        from A_core.A01_domain_models import (
            Coordinate,
            EvidenceSpan,
            ExtractedEntity,
            FieldType,
            GeneratorType,
            ProvenanceMetadata,
            ValidationStatus,
        )

        context_snippet = extract_context_snippet(full_text, match.start(), match.end())
        ctx_hash = hash_string(context_snippet)

        primary = EvidenceSpan(
            text=context_snippet,
            location=Coordinate(page_num=1),
            scope_ref=ctx_hash,
            start_char_offset=match.start() - max(0, match.start() - 100),
            end_char_offset=match.end() - max(0, match.start() - 100),
        )

        prov = ProvenanceMetadata(
            pipeline_version=self.pipeline_version,
            run_id=self.run_id,
            doc_fingerprint=lexicon_source,
            generator_name=GeneratorType.LEXICON_MATCH,
            rule_version=rule_version,
            lexicon_source=f"orchestrator:{lexicon_source}",
        )

        return ExtractedEntity(
            candidate_id=uuid.uuid4(),
            doc_id=doc_id,
            field_type=FieldType.DEFINITION_PAIR,
            short_form=match.group(),
            long_form=long_form,
            primary_evidence=primary,
            supporting_evidence=[],
            status=ValidationStatus.VALIDATED,
            confidence_score=confidence,
            rejection_reason=None,
            validation_flags=flags,
            provenance=prov,
            raw_llm_response={"auto": lexicon_source},
        )


__all__ = ["AbbreviationPipeline"]
