#  corpus_metadata/corpus_abbreviations/C_generators/C01_strategy_abbrev.py

from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

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
    TableType,
)


# ----------------------------
# Helpers (pure functions)
# ----------------------------

_ABBREV_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-\+/().]{0,14}$")


def _clean_ws(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _looks_like_short_form(sf: str, min_len: int = 2, max_len: int = 10) -> bool:
    sf = (sf or "").strip()
    if len(sf) < min_len or len(sf) > max_len:
        return False
    if sf.isdigit():
        return False
    if " " in sf:
        return False
    if not _ABBREV_TOKEN_RE.match(sf):
        return False
    # Must contain at least one uppercase letter (strong pharma heuristic)
    if not any(ch.isupper() for ch in sf):
        return False
    return True


def _truncate_at_breaks(text: str) -> str:
    """
    For implicit LF captures: cut at punctuation / obvious clause starters.
    """
    t = (text or "").strip()

    # Stop at punctuation
    t = re.split(r"[.;:\n\r\)\]]", t, maxsplit=1)[0].strip()

    # Stop at clause starters that often introduce extra commentary
    t = re.split(r"\b(which|that|who|where|when|while)\b", t, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    # Drop trailing connector words
    t = re.sub(r"\b(and|or|as|by)\s*$", "", t, flags=re.IGNORECASE).strip()

    return _clean_ws(t)


def _context_window(text: str, start: int, end: int, window: int = 240) -> str:
    """
    Create a short context window around a match.
    """
    if not text:
        return ""
    s = max(0, start - window)
    e = min(len(text), end + window)
    return _clean_ws(text[s:e])


def _schwartz_hearst_extract(short_form: str, preceding_text: str) -> Optional[str]:
    """
    Schwartz-Hearst-ish backward character alignment.
    Returns a best-effort long form (LF) extracted from preceding_text, or None.

    Works well for: "... Tumor Necrosis Factor (TNF)"
    """
    sf = re.sub(r"[^A-Za-z0-9]", "", short_form or "")
    if len(sf) < 2:
        return None

    # Limit lookback window (characters)
    txt = (preceding_text or "").rstrip()
    if not txt:
        return None

    window_size = min(len(txt), len(sf) * 18 + 30)
    txt = txt[-window_size:]

    i = len(sf) - 1
    j = len(txt) - 1

    # Align SF chars backwards into txt backwards
    while i >= 0:
        c = sf[i].lower()
        while j >= 0 and txt[j].lower() != c:
            j -= 1
        if j < 0:
            return None
        i -= 1
        j -= 1

    # j is now before the earliest matched character; pick LF start at word boundary
    start = txt.rfind(" ", 0, j + 1) + 1
    lf = _clean_ws(txt[start:])

    if not lf:
        return None

    # Strong safety: first letter alignment (prevents many false positives)
    if lf[0].lower() != short_form[0].lower():
        return None

    # Avoid absurdly long LFs
    if len(lf) > 120:
        return None

    return lf


def _validate_sf_in_lf(short_form: str, long_form: str) -> bool:
    """
    Lightweight validation that SF letters appear in LF (in order, not necessarily contiguous).
    This is used for:
      - validating implicit captures
      - validating reverse explicit pattern: "SF (Long Form)"
    """
    sf = re.sub(r"[^A-Za-z0-9]", "", short_form or "")
    lf = re.sub(r"[^A-Za-z0-9 ]", " ", long_form or "")
    lf = _clean_ws(lf)

    if not sf or not lf:
        return False

    if lf[0].lower() != short_form[0].lower():
        return False

    # In-order scan across LF
    sf_idx = 0
    for ch in lf:
        if ch.lower() == sf[sf_idx].lower():
            sf_idx += 1
            if sf_idx == len(sf):
                return True

    return False


# ----------------------------
# Generator 1: Explicit + Implicit (syntax)
# ----------------------------

class AbbrevSyntaxCandidateGenerator(BaseCandidateGenerator):
    """
    Abbreviation extraction from running text.

    Strategy A (Explicit):  Long Form (SF)
    Strategy B (Explicit):  SF (Long Form)
    Strategy C (Implicit):  SF, defined as/stands for/... Long Form
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        self.min_sf_length = int(self.config.get("min_sf_length", 2))
        self.max_sf_length = int(self.config.get("max_sf_length", 10))

        # Capture any parentheses content; we decide SF vs LF by heuristics
        self.parens_any = re.compile(r"\(([^)]+)\)")

        # Implicit phrasing patterns (case-insensitive phrases, SF stays strict-ish)
        # LF capture is non-greedy and stops before punctuation via lookahead.
        self.implicit_patterns = [
            re.compile(
                r"\b([A-Z][A-Za-z0-9\-\+/().]{1,14})\b\s*,?\s*(?:defined\s+as)\s+(.{5,160}?)(?=[.;:\n\r\)\]]|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b([A-Z][A-Za-z0-9\-\+/().]{1,14})\b\s*,?\s*(?:stands\s+for)\s+(.{5,160}?)(?=[.;:\n\r\)\]]|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b([A-Z][A-Za-z0-9\-\+/().]{1,14})\b\s*,?\s*(?:abbreviated\s+as)\s+(.{5,160}?)(?=[.;:\n\r\)\]]|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b([A-Z][A-Za-z0-9\-\+/().]{1,14})\b\s*,?\s*(?:also\s+known\s+as)\s+(.{5,160}?)(?=[.;:\n\r\)\]]|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b([A-Z][A-Za-z0-9\-\+/().]{1,14})\b\s*,?\s*(?:short\s+for)\s+(.{5,160}?)(?=[.;:\n\r\)\]]|$)",
                re.IGNORECASE,
            ),
        ]

        self.context_window_chars = int(self.config.get("context_window_chars", 240))
        self.carryover_chars = int(self.config.get("carryover_chars", 140))  # helps cross-block edge cases
        self.max_candidates_per_block = int(self.config.get("max_candidates_per_block", 200))

        # Provenance defaults (prefer orchestrator to pass these in)
        self.pipeline_version = str(self.config.get("pipeline_version") or get_git_revision_hash())
        self.run_id = str(self.config.get("run_id") or generate_run_id("ABBR"))
        self.doc_fingerprint_default = str(self.config.get("doc_fingerprint") or "unknown-doc-fingerprint")

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

            combined = (prev_tail + " " + text_block).strip() if prev_tail else text_block

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
                if _looks_like_short_form(inside, self.min_sf_length, self.max_sf_length):
                    sf = inside
                    preceding = combined[: m.start()]
                    lf = _schwartz_hearst_extract(sf, preceding)
                    if not lf:
                        continue

                    key = (doc.doc_id, sf.upper(), lf.lower())
                    if key in seen:
                        continue
                    seen.add(key)

                    out.append(self._make_candidate(doc, block, sf, lf, combined, m.start(), m.end(), method="explicit_lf_sf", confidence=0.98))
                    added_this_block += 1
                    continue

                # Otherwise, treat as possible LF and look for a preceding SF token
                lf_candidate = inside
                preceding = combined[: m.start()].rstrip()

                # Grab last token before "(" as SF
                prev_token_match = re.search(r"([A-Za-z0-9\-\+/().]{2,15})\s*$", preceding)
                if not prev_token_match:
                    continue
                sf_candidate = prev_token_match.group(1)

                if not _looks_like_short_form(sf_candidate, self.min_sf_length, self.max_sf_length):
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

                out.append(self._make_candidate(doc, block, sf_candidate, lf_clean, combined, m.start(), m.end(), method="explicit_sf_lf", confidence=0.96))
                added_this_block += 1

            # -------------------------
            # Strategy C: Implicit phrasing
            # -------------------------
            for pat in self.implicit_patterns:
                if added_this_block >= self.max_candidates_per_block:
                    break

                for m in pat.finditer(text_block):  # use block text for implicit; avoids cross-block weirdness
                    if added_this_block >= self.max_candidates_per_block:
                        break

                    sf = _clean_ws(m.group(1))
                    raw_lf = _clean_ws(m.group(2))

                    if not _looks_like_short_form(sf, self.min_sf_length, self.max_sf_length):
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

                    out.append(self._make_candidate(doc, block, sf, lf, text_block, m.start(), m.end(), method="implicit_phrasing", confidence=0.90))
                    added_this_block += 1

            # update tail for cross-block continuity
            prev_tail = combined[-self.carryover_chars :] if self.carryover_chars > 0 else ""

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
        ctx = _context_window(context_source_text, match_start, match_end, window=self.context_window_chars)

        loc = Coordinate(
            page_num=int(block.page_num),
            block_id=str(block.id),
            bbox=block.bbox,
        )

        prov = ProvenanceMetadata(
            pipeline_version=self.pipeline_version,
            run_id=self.run_id,
            doc_fingerprint=str(self.config.get("doc_fingerprint") or self.doc_fingerprint_default),
            generator_name=self.generator_type,
            # keep method here (cheap + no schema changes)
            rule_version=f"abbrev_syntax::{method}",
        )

        return Candidate(
            doc_id=doc.doc_id,
            field_type=FieldType.DEFINITION_PAIR,
            generator_type=self.generator_type,
            short_form=_clean_ws(sf),
            long_form=_clean_ws(lf),
            context_text=ctx if ctx else context_source_text,
            context_location=loc,
            initial_confidence=float(max(0.0, min(1.0, confidence))),
            provenance=prov,
        )


# ----------------------------
# Generator 2: Glossary tables (high value)
# ----------------------------

class GlossaryTableCandidateGenerator(BaseCandidateGenerator):
    """
    Extract SF/LF pairs from tables classified as GLOSSARY.
    Uses the Table dual-view (logical_rows + physical cells).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_rows_per_table = int(self.config.get("max_rows_per_table", 500))

        self.pipeline_version = str(self.config.get("pipeline_version") or get_git_revision_hash())
        self.run_id = str(self.config.get("run_id") or generate_run_id("ABBR"))
        self.doc_fingerprint_default = str(self.config.get("doc_fingerprint") or "unknown-doc-fingerprint")

    @property
    def generator_type(self) -> GeneratorType:
        return GeneratorType.GLOSSARY_TABLE

    def extract(self, doc_structure: DocumentGraph) -> List[Candidate]:
        doc = doc_structure
        out: List[Candidate] = []
        seen: Set[Tuple[str, str, str]] = set()

        for table in doc.iter_tables(table_type=TableType.GLOSSARY):
            count = 0
            for sf, lf, sf_cell, lf_cell in table.iter_glossary_pairs():
                if count >= self.max_rows_per_table:
                    break

                sf = _clean_ws(sf)
                lf = _clean_ws(lf)
                if not sf or not lf:
                    continue

                key = (doc.doc_id, sf.upper(), lf.lower())
                if key in seen:
                    continue
                seen.add(key)

                # Prefer SF cell for pinpointing; fallback to table bbox
                bbox = (sf_cell.bbox if sf_cell else table.bbox)
                cell_row = int(sf_cell.row_index) if sf_cell else None
                cell_col = int(sf_cell.col_index) if sf_cell else None

                loc = Coordinate(
                    page_num=int(table.page_num),
                    table_id=str(table.id),
                    cell_row=cell_row,
                    cell_col=cell_col,
                    bbox=bbox,
                )

                prov = ProvenanceMetadata(
                    pipeline_version=self.pipeline_version,
                    run_id=self.run_id,
                    doc_fingerprint=str(self.config.get("doc_fingerprint") or self.doc_fingerprint_default),
                    generator_name=self.generator_type,
                    rule_version="glossary_table::v1",
                )

                # Context: small markdown table for LLM/debug
                ctx = table.to_markdown(max_rows=20)

                out.append(
                    Candidate(
                        doc_id=doc.doc_id,
                        field_type=FieldType.GLOSSARY_ENTRY,
                        generator_type=self.generator_type,
                        short_form=sf,
                        long_form=lf,
                        context_text=ctx,
                        context_location=loc,
                        initial_confidence=0.99,
                        provenance=prov,
                    )
                )

                count += 1

        return out