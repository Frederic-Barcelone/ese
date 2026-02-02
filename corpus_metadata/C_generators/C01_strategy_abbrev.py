"""
Schwartz-Hearst abbreviation extraction from running text.

This module extracts abbreviation definitions using the Schwartz-Hearst algorithm
and related pattern matching strategies. It detects explicit short form/long form
pairs in clinical trial documents, handling both standard and reversed patterns.

Key Components:
    - SyntaxMatcherGenerator: Main generator implementing Schwartz-Hearst algorithm
    - Pattern strategies for multiple definition formats:
        - Standard: "Tumor Necrosis Factor (TNF)"
        - Reversed: "TNF (Tumor Necrosis Factor)"
        - Implicit: "TNF, defined as Tumor Necrosis Factor"
        - Inline: "SF=Long Form" or "SF: Long Form"
        - Comma: "SF, Long Form" (common in tables/legends)

Example:
    >>> from C_generators.C01_strategy_abbrev import SyntaxMatcherGenerator
    >>> generator = SyntaxMatcherGenerator(config={})
    >>> candidates = generator.generate(doc_graph)
    >>> for c in candidates:
    ...     print(f"{c.short_form} = {c.long_form}")
    TNF = Tumor Necrosis Factor

References:
    - Schwartz & Hearst (2003) "A Simple Algorithm for Identifying
      Abbreviation Definitions in Biomedical Text"

Dependencies:
    - A_core.A01_domain_models: Candidate, Coordinate, FieldType, GeneratorType
    - A_core.A02_interfaces: BaseCandidateGenerator
    - A_core.A03_provenance: Provenance tracking utilities
    - B_parsing.B02_doc_graph: DocumentGraph, ContentRole
    - C_generators.C20_abbrev_patterns: Pattern matching helpers
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from A_core.A01_domain_models import (
    Candidate,
    Coordinate,
    FieldType,
    GeneratorType,
    ProvenanceMetadata,
)
from A_core.A02_interfaces import BaseCandidateGenerator
from A_core.A03_provenance import (
    generate_run_id,
    get_git_revision_hash,
)
from B_parsing.B02_doc_graph import (
    ContentRole,
    DocumentGraph,
)

from .C20_abbrev_patterns import (
    _is_likely_author_initial,
    _clean_ws,
    _normalize_long_form,
    _looks_like_short_form,
    _context_window,
    _truncate_at_breaks,
    _schwartz_hearst_extract,
    _space_sf_extract,
    _extract_preceding_name,
    _validate_sf_in_lf,
)


class AbbrevSyntaxCandidateGenerator(BaseCandidateGenerator):
    """
    Abbreviation extraction from running text.

    Strategy A (Explicit):  Long Form (SF)
    Strategy B (Explicit):  SF (Long Form)
    Strategy C (Implicit):  SF, defined as/stands for/... Long Form
    Strategy D (Inline):    SF=Long Form or SF: Long Form
    Strategy E (Comma):     SF, Long Form (common in tables/legends)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        self.min_sf_length = int(self.config.get("min_sf_length", 2))
        self.max_sf_length = int(self.config.get("max_sf_length", 10))

        # Capture any parentheses content; we decide SF vs LF by heuristics
        self.parens_any = re.compile(r"\(([^)]+)\)")

        # Implicit phrasing patterns (case-insensitive phrases, SF stays strict-ish)
        # LF capture is non-greedy and stops before punctuation via lookahead.
        # Use lookahead instead of \b for end boundary to handle hyphenated abbreviations (e.g., APPEAR-C3G)
        self.implicit_patterns = [
            re.compile(
                r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9\-\+/().]{1,14})(?=[,\s])\s*,?\s*(?:defined\s+as)\s+(.{5,160}?)(?=[.;:\n\r\)\]]|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9\-\+/().]{1,14})(?=[,\s])\s*,?\s*(?:stands\s+for)\s+(.{5,160}?)(?=[.;:\n\r\)\]]|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9\-\+/().]{1,14})(?=[,\s])\s*,?\s*(?:abbreviated\s+as)\s+(.{5,160}?)(?=[.;:\n\r\)\]]|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9\-\+/().]{1,14})(?=[,\s])\s*,?\s*(?:also\s+known\s+as)\s+(.{5,160}?)(?=[.;:\n\r\)\]]|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9\-\+/().]{1,14})(?=[,\s])\s*,?\s*(?:short\s+for)\s+(.{5,160}?)(?=[.;:\n\r\)\]]|$)",
                re.IGNORECASE,
            ),
        ]

        # Explicit inline definition patterns: "SF=long form" or "SF: long form"
        # Captures patterns like "RR=relative reduction" or "UPCR: urine protein-creatinine ratio"
        # NOTE: Use [ \t] instead of \s to avoid matching newlines in long forms
        self.inline_definition_patterns = [
            # SF=long form or SF: long form (e.g., "RR=relative reduction")
            # Uses greedy match for LF, stopping at period, comma, newline, or uppercase letter
            re.compile(
                r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9\-]{1,10})\s*[=:]\s*([a-z][a-z \t\-/]{3,60})(?=[.,;)\]A-Z\n\r]|$)",
            ),
            # SF: Capitalized Long Form (e.g., "LOA: Level of Agreement", "SOR: Strength of Recommendation")
            # Handles title-case definitions common in methodology tables
            re.compile(
                r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9\-]{1,10})\s*[=:]\s*([A-Z][a-z]+(?:[ \t]+[a-zA-Z][a-z]*){1,8})(?=[.,;)\]\n\r]|$)",
            ),
        ]

        # Comma-separated definition patterns: "SF, Long Form" (common in tables/legends)
        # E.g., "FV, Final Vote" or "LOA, Level of Agreement"
        self.comma_definition_patterns = [
            # SF, Capitalized Long Form
            re.compile(
                r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9\-]{1,10})\s*,\s*([A-Z][a-z]+(?:[ \t]+[a-zA-Z][a-z]*){1,8})(?=[.,;)\]\n\r]|$)",
            ),
            # SF, lowercase long form
            re.compile(
                r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9\-]{1,10})\s*,\s*([a-z][a-z \t\-/]{3,60})(?=[.,;)\]A-Z\n\r]|$)",
            ),
        ]

        self.context_window_chars = int(self.config.get("context_window_chars", 400))
        self.carryover_chars = int(
            self.config.get("carryover_chars", 140)
        )  # helps cross-block edge cases
        self.max_candidates_per_block = int(
            self.config.get("max_candidates_per_block", 200)
        )

        # Provenance defaults (prefer orchestrator to pass these in)
        self.pipeline_version = str(
            self.config.get("pipeline_version") or get_git_revision_hash()
        )
        self.run_id = str(self.config.get("run_id") or generate_run_id("ABBR"))
        self.doc_fingerprint_default = str(
            self.config.get("doc_fingerprint") or "unknown-doc-fingerprint"
        )

    @property
    def generator_type(self) -> GeneratorType:
        return GeneratorType.SYNTAX_PATTERN

    def extract(self, doc_structure: DocumentGraph) -> List[Candidate]:
        doc = doc_structure
        out: List[Candidate] = []
        seen: Set[Tuple[str, str, str]] = set()

        prev_tail = ""

        for block in doc.iter_linear_blocks(skip_header_footer=True):
            # Avoid extracting inside section headers; keep those for context elsewhere
            if block.role == ContentRole.SECTION_HEADER:
                prev_tail = _clean_ws(block.text)[-self.carryover_chars :]
                continue

            text_block = _clean_ws(block.text)
            if not text_block:
                prev_tail = ""
                continue

            combined = (
                (prev_tail + " " + text_block).strip() if prev_tail else text_block
            )

            added_this_block = 0

            # -------------------------
            # Strategy A/B: Parentheses explicit
            # -------------------------
            for m in self.parens_any.finditer(combined):
                if added_this_block >= self.max_candidates_per_block:
                    break

                inside = _clean_ws(m.group(1))
                if not inside:
                    continue

                # Prefer the content inside parens as SF if it looks like SF
                if _looks_like_short_form(
                    inside, self.min_sf_length, self.max_sf_length
                ):
                    sf = inside
                    preceding = combined[: m.start()]

                    # Skip author initials (e.g., "BH", "JM" in author lists)
                    if _is_likely_author_initial(sf, combined):
                        continue

                    # Try Schwartz-Hearst first (works for single-token SFs)
                    lf = _schwartz_hearst_extract(sf, preceding)

                    # Fallback for space-separated SFs like "CC BY"
                    if not lf and " " in sf:
                        lf = _space_sf_extract(sf, preceding)

                    # HIGH PRIORITY FIX: Handle compound IDs like "iptacopan (LNP023)"
                    # where the code doesn't spell out the drug name
                    if not lf:
                        lf = _extract_preceding_name(sf, preceding)

                    if not lf:
                        continue

                    key = (doc.doc_id, sf.upper(), lf.lower())
                    if key in seen:
                        continue
                    seen.add(key)

                    method = "explicit_lf_sf" if " " not in sf else "explicit_space_sf"
                    out.append(
                        self._make_candidate(
                            doc,
                            block,
                            sf,
                            lf,
                            combined,
                            m.start(),
                            m.end(),
                            method=method,
                            confidence=0.95 if " " in sf else 0.98,
                        )
                    )
                    added_this_block += 1
                    continue

                # Otherwise, treat as possible LF and look for a preceding SF token
                lf_candidate = inside
                preceding = combined[: m.start()].rstrip()

                # Grab last token before "(" as SF
                prev_token_match = re.search(
                    r"([A-Za-z0-9\-\+/().]{2,15})\s*$", preceding
                )
                if not prev_token_match:
                    continue
                sf_candidate = prev_token_match.group(1)

                if not _looks_like_short_form(
                    sf_candidate, self.min_sf_length, self.max_sf_length
                ):
                    continue

                # Skip author initials (e.g., "BH", "JM" in author lists)
                if _is_likely_author_initial(sf_candidate, combined):
                    continue

                lf_clean = _truncate_at_breaks(lf_candidate)
                if not lf_clean:
                    continue

                if not _validate_sf_in_lf(sf_candidate, lf_clean):
                    continue

                key = (doc.doc_id, sf_candidate.upper(), lf_clean.lower())
                if key in seen:
                    continue
                seen.add(key)

                out.append(
                    self._make_candidate(
                        doc,
                        block,
                        sf_candidate,
                        lf_clean,
                        combined,
                        m.start(),
                        m.end(),
                        method="explicit_sf_lf",
                        confidence=0.96,
                    )
                )
                added_this_block += 1

            # -------------------------
            # Strategy C: Implicit phrasing
            # -------------------------
            for pat in self.implicit_patterns:
                if added_this_block >= self.max_candidates_per_block:
                    break

                for m in pat.finditer(
                    text_block
                ):  # use block text for implicit; avoids cross-block weirdness
                    if added_this_block >= self.max_candidates_per_block:
                        break

                    sf = _clean_ws(m.group(1))
                    raw_lf = _clean_ws(m.group(2))

                    if not _looks_like_short_form(
                        sf, self.min_sf_length, self.max_sf_length
                    ):
                        continue

                    # Skip author initials (e.g., "BH", "JM" in author lists)
                    if _is_likely_author_initial(sf, text_block):
                        continue

                    lf = _truncate_at_breaks(raw_lf)
                    if not lf:
                        continue

                    # validate implicit LF (prevents greedy junk)
                    if not _validate_sf_in_lf(sf, lf):
                        continue

                    key = (doc.doc_id, sf.upper(), lf.lower())
                    if key in seen:
                        continue
                    seen.add(key)

                    out.append(
                        self._make_candidate(
                            doc,
                            block,
                            sf,
                            lf,
                            text_block,
                            m.start(),
                            m.end(),
                            method="implicit_phrasing",
                            confidence=0.90,
                        )
                    )
                    added_this_block += 1

            # -------------------------
            # Strategy D: Inline definitions (SF=LF or SF: LF)
            # HIGH PRIORITY: Catches "RR=relative reduction" patterns
            # -------------------------
            for pat in self.inline_definition_patterns:
                if added_this_block >= self.max_candidates_per_block:
                    break

                for m in pat.finditer(text_block):
                    if added_this_block >= self.max_candidates_per_block:
                        break

                    sf = _clean_ws(m.group(1))
                    lf = _clean_ws(m.group(2))

                    if not _looks_like_short_form(
                        sf, self.min_sf_length, self.max_sf_length
                    ):
                        continue

                    # Skip author initials (e.g., "BH", "JM" in author lists)
                    if _is_likely_author_initial(sf, text_block):
                        continue

                    # LF should be lowercase phrase (not another abbreviation)
                    if not lf or len(lf) < 4:
                        continue

                    # Skip if LF is all uppercase (likely another abbreviation)
                    if lf.isupper():
                        continue

                    # Skip if LF contains common noise patterns
                    noise_patterns = [
                        "sponsored by", "workshop", "scientific", "facts",
                        "uncertainties", "study", "trial", "published",
                    ]
                    lf_lower = lf.lower()
                    if any(noise in lf_lower for noise in noise_patterns):
                        continue

                    # Skip if LF is too long (likely noise)
                    if len(lf) > 40:
                        continue

                    key = (doc.doc_id, sf.upper(), lf.lower())
                    if key in seen:
                        continue
                    seen.add(key)

                    out.append(
                        self._make_candidate(
                            doc,
                            block,
                            sf,
                            lf,
                            text_block,
                            m.start(),
                            m.end(),
                            method="inline_definition",
                            confidence=0.97,  # High confidence - explicit definition
                        )
                    )
                    added_this_block += 1

            # -------------------------
            # Strategy E: Comma-separated definitions (SF, Long Form)
            # HIGH PRIORITY: Catches "LOA, Level of Agreement" patterns in tables/legends
            # -------------------------
            for pat in self.comma_definition_patterns:
                if added_this_block >= self.max_candidates_per_block:
                    break

                for m in pat.finditer(text_block):
                    if added_this_block >= self.max_candidates_per_block:
                        break

                    sf = _clean_ws(m.group(1))
                    lf = _clean_ws(m.group(2))

                    if not _looks_like_short_form(
                        sf, self.min_sf_length, self.max_sf_length
                    ):
                        continue

                    # Skip author initials
                    if _is_likely_author_initial(sf, text_block):
                        continue

                    # LF should be a proper phrase (not another abbreviation)
                    if not lf or len(lf) < 4:
                        continue

                    # Skip if LF is all uppercase (likely another abbreviation)
                    if lf.isupper():
                        continue

                    # Skip if LF is too long (likely noise)
                    if len(lf) > 50:
                        continue

                    key = (doc.doc_id, sf.upper(), lf.lower())
                    if key in seen:
                        continue
                    seen.add(key)

                    out.append(
                        self._make_candidate(
                            doc,
                            block,
                            sf,
                            lf,
                            text_block,
                            m.start(),
                            m.end(),
                            method="comma_definition",
                            confidence=0.92,  # Slightly lower - comma patterns can be ambiguous
                        )
                    )
                    added_this_block += 1

            # update tail for cross-block continuity
            prev_tail = (
                combined[-self.carryover_chars :] if self.carryover_chars > 0 else ""
            )

        # -------------------------
        # Strategy F: Image/Figure Captions
        # Process captions from ImageBlocks which are NOT included in iter_linear_blocks
        # This catches patterns like "RR = relative reduction" in figure captions
        # -------------------------
        for img in doc.iter_images():
            caption = img.caption
            if not caption:
                continue

            caption_text = _clean_ws(caption)
            if not caption_text or len(caption_text) < 10:
                continue

            # Apply inline definition patterns (SF=LF, SF: LF) to captions
            # These are common in figure/table legends
            for pat in self.inline_definition_patterns:
                for m in pat.finditer(caption_text):
                    sf = m.group(1).strip()
                    lf = m.group(2).strip()

                    if not sf or not lf:
                        continue

                    # Skip if LF is too long
                    if len(lf) > 60:
                        continue

                    # Clean long form
                    lf = _normalize_long_form(lf)
                    if not lf or len(lf) < 3:
                        continue

                    key = (doc.doc_id, sf.upper(), lf.lower())
                    if key in seen:
                        continue
                    seen.add(key)

                    out.append(
                        self._make_candidate_from_image(
                            doc,
                            img,
                            sf,
                            lf,
                            caption_text,
                            m.start(),
                            m.end(),
                            method="caption_inline_definition",
                            confidence=0.95,  # High confidence - explicit definition in caption
                        )
                    )

            # Also apply comma definition patterns to captions
            for pat in self.comma_definition_patterns:
                for m in pat.finditer(caption_text):
                    sf = m.group(1).strip()
                    lf = m.group(2).strip()

                    if not sf or not lf:
                        continue

                    if len(lf) > 60:
                        continue

                    lf = _normalize_long_form(lf)
                    if not lf or len(lf) < 3:
                        continue

                    key = (doc.doc_id, sf.upper(), lf.lower())
                    if key in seen:
                        continue
                    seen.add(key)

                    out.append(
                        self._make_candidate_from_image(
                            doc,
                            img,
                            sf,
                            lf,
                            caption_text,
                            m.start(),
                            m.end(),
                            method="caption_comma_definition",
                            confidence=0.92,
                        )
                    )

        return out

    def _make_candidate(
        self,
        doc: DocumentGraph,
        block,
        sf: str,
        lf: str,
        context_source_text: str,
        match_start: int,
        match_end: int,
        method: str,
        confidence: float,
    ) -> Candidate:
        ctx = _context_window(
            context_source_text,
            match_start,
            match_end,
            window=self.context_window_chars,
        )

        loc = Coordinate(
            page_num=int(block.page_num),
            block_id=str(block.id),
            bbox=block.bbox,
        )

        prov = ProvenanceMetadata(
            pipeline_version=self.pipeline_version,
            run_id=self.run_id,
            doc_fingerprint=str(
                self.config.get("doc_fingerprint") or self.doc_fingerprint_default
            ),
            generator_name=self.generator_type,
            # keep method here (cheap + no schema changes)
            rule_version=f"abbrev_syntax::{method}",
        )

        return Candidate(
            doc_id=doc.doc_id,
            field_type=FieldType.DEFINITION_PAIR,
            generator_type=self.generator_type,
            short_form=_clean_ws(sf),
            long_form=_normalize_long_form(lf),  # Dehyphenate line-break artifacts
            context_text=ctx if ctx else context_source_text,
            context_location=loc,
            initial_confidence=float(max(0.0, min(1.0, confidence))),
            provenance=prov,
        )

    def _make_candidate_from_image(
        self,
        doc: DocumentGraph,
        img,
        sf: str,
        lf: str,
        context_source_text: str,
        match_start: int,
        match_end: int,
        method: str,
        confidence: float,
    ) -> Candidate:
        """Create a candidate from an ImageBlock's caption."""
        ctx = _context_window(
            context_source_text,
            match_start,
            match_end,
            window=self.context_window_chars,
        )

        loc = Coordinate(
            page_num=int(img.page_num),
            block_id=str(img.id),
            bbox=img.bbox,
        )

        prov = ProvenanceMetadata(
            pipeline_version=self.pipeline_version,
            run_id=self.run_id,
            doc_fingerprint=str(
                self.config.get("doc_fingerprint") or self.doc_fingerprint_default
            ),
            generator_name=self.generator_type,
            rule_version=f"abbrev_syntax::{method}",
        )

        return Candidate(
            doc_id=doc.doc_id,
            field_type=FieldType.DEFINITION_PAIR,
            generator_type=self.generator_type,
            short_form=_clean_ws(sf),
            long_form=_normalize_long_form(lf),
            context_text=ctx if ctx else context_source_text,
            context_location=loc,
            initial_confidence=float(max(0.0, min(1.0, confidence))),
            provenance=prov,
        )


# Re-export for backward compatibility
