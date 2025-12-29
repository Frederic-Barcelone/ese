# corpus_metadata/corpus_abbreviations/C_generators/C02_strategy_regex.py
# Abbreviation-only orphan finder (SHORT_FORM_ONLY)

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

from A_core.A01_domain_models import (
    Candidate,
    Coordinate,
    FieldType,
    GeneratorType,
    ProvenanceMetadata,
)
from A_core.A02_interfaces import BaseCandidateGenerator
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from B_parsing.B02_doc_graph import ContentRole, DocumentGraph


def _clean_ws(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _norm_section(s: str) -> str:
    return _clean_ws(s).lower()


class RegexCandidateGenerator(BaseCandidateGenerator):
    """
    Abbreviation-Only "Orphan" Finder.

    Goal:
      - Find acronym-like tokens that appear WITHOUT a local definition in the text.
      - This complements strategy_abbrev.py (definitions) and glossary extraction.

    Approach:
      1) Scan for acronym-like tokens (AE, SAE, TNF-alpha, IL-6, eGFR, HbA1c, etc.)
      2) Global frequency filter (orphans must repeat to be credible)
      3) Section awareness (skip References/Bibliography/ToC/Appendix)
      4) Optional suppression: known_short_forms (already defined elsewhere)
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}

        # Matches: TNF, IL-6, TNF-alpha, HbA1c, eGFR, SAE, AE
        # (First char uppercase; rest can include lowercase/digits; optional hyphen suffix)
        self.token_pattern = re.compile(
            cfg.get("token_regex", r"\b[A-Z][A-Za-z0-9]{1,11}(?:-[A-Za-z0-9]{1,11})?\b")
        )

        # Filters
        self.min_len = int(cfg.get("min_len", 2))
        self.max_len = int(cfg.get("max_len", 12))
        self.min_occurrences = int(cfg.get("min_occurrences", 3))  # stricter for orphans

        # Context window around match (chars)
        self.ctx_window = int(cfg.get("ctx_window", 80))

        # Section blacklist (substring match)
        self.blacklisted_sections = {
            _norm_section(s)
            for s in cfg.get(
                "blacklisted_sections",
                ["references", "bibliography", "table of contents", "appendix", "appendices"],
            )
        }

        # Token blacklist: compare in UPPER
        self.blacklist_upper = {
            str(x).strip().upper()
            for x in cfg.get(
                "blacklist",
                ["THE", "AND", "FOR", "NOT", "BUT", "FIG", "TABLE", "PAGE", "SEE", "SECTION"],
            )
        }

        # Optional: suppress abbreviations already defined by other generators
        # Orchestrator should pass: {"known_short_forms": ["TNF", "AE", ...]}
        self.known_short_forms: Set[str] = {
            str(x).strip().upper()
            for x in cfg.get("known_short_forms", [])
            if str(x).strip()
        }

        # Emit mode: only first occurrence per token (default)
        self.emit_first_only = bool(cfg.get("emit_first_only", True))

        # Confidence base for orphans (definition-less)
        self.base_confidence = float(cfg.get("base_confidence", 0.60))
        self.max_confidence = float(cfg.get("max_confidence", 0.85))

        # Provenance defaults
        self.pipeline_version = str(cfg.get("pipeline_version") or get_git_revision_hash())
        self.run_id = str(cfg.get("run_id") or generate_run_id("ABBR"))
        self.doc_fingerprint_default = str(cfg.get("doc_fingerprint") or "unknown-doc-fingerprint")

    @property
    def generator_type(self) -> GeneratorType:
        return GeneratorType.SYNTAX_PATTERN

    def extract(self, doc_structure: DocumentGraph) -> List[Candidate]:
        doc = doc_structure

        # First pass: count + store occurrences
        counts: DefaultDict[str, int] = defaultdict(int)
        occurrences: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

        current_section = "UNKNOWN"

        for block in doc.iter_linear_blocks(skip_header_footer=True):
            # Section awareness
            if block.role == ContentRole.SECTION_HEADER:
                current_section = _clean_ws(block.text) or "UNKNOWN"
                continue

            if self._is_blacklisted_section(current_section):
                continue

            text = block.text or ""
            if not text.strip():
                continue

            for match in self.token_pattern.finditer(text):
                token_raw = match.group(0)
                token = token_raw.strip()
                token_upper = token.upper()

                if token_upper in self.known_short_forms:
                    continue

                if not self._is_valid_acronym(token):
                    continue

                counts[token_upper] += 1
                occurrences[token_upper].append(
                    {
                        "block": block,
                        "match_start": match.start(),
                        "match_end": match.end(),
                        "section": current_section,
                        "token_original": token_raw,
                    }
                )

        # Second pass: emit candidates
        candidates: List[Candidate] = []

        for token_upper, count in counts.items():
            if count < self.min_occurrences:
                continue

            # Compute confidence as a function of frequency
            conf = self._confidence_from_count(count)

            if self.emit_first_only:
                inst = occurrences[token_upper][0]
                candidates.append(self._make_candidate(doc, token_upper, count, conf, inst))
            else:
                for inst in occurrences[token_upper]:
                    candidates.append(self._make_candidate(doc, token_upper, count, conf, inst))

        return candidates

    # -------------------------
    # Internals
    # -------------------------

    def _make_candidate(
        self,
        doc: DocumentGraph,
        token_upper: str,
        count: int,
        conf: float,
        inst: Dict[str, Any],
    ) -> Candidate:
        block = inst["block"]
        start = int(inst["match_start"])
        end = int(inst["match_end"])
        section = str(inst["section"])

        # Context snippet from the same block
        text = block.text or ""
        s = max(0, start - self.ctx_window)
        e = min(len(text), end + self.ctx_window)
        context_snippet = _clean_ws(text[s:e])

        # Build Coordinate (matches A01 schema)
        loc = Coordinate(
            page_num=int(block.page_num),
            block_id=str(block.id),
            bbox=block.bbox,
        )

        # Build ProvenanceMetadata (matches A01 schema)
        prov = ProvenanceMetadata(
            pipeline_version=self.pipeline_version,
            run_id=self.run_id,
            doc_fingerprint=self.doc_fingerprint_default,
            generator_name=self.generator_type,
            rule_version=f"regex_orphan::v1::freq={count}",
        )

        return Candidate(
            doc_id=doc.doc_id,
            field_type=FieldType.SHORT_FORM_ONLY,
            generator_type=self.generator_type,
            short_form=token_upper,
            long_form=None,  # Orphans have no LF
            context_text=context_snippet,
            context_location=loc,
            initial_confidence=float(max(0.0, min(1.0, conf))),
            provenance=prov,
        )

    def _is_blacklisted_section(self, section_header: str) -> bool:
        s = _norm_section(section_header)
        if not s:
            return False
        return any(bad in s for bad in self.blacklisted_sections)

    def _is_valid_acronym(self, token: str) -> bool:
        """
        Filter obvious noise:
          - too short/long
          - purely numeric
          - common word-like tokens ("The") that slip through because regex allows lowercase
          - tokens in blacklist
          - tokens with weak "abbreviation-ness"
        """
        t = (token or "").strip()
        if not t:
            return False

        if len(t) < self.min_len or len(t) > self.max_len:
            return False

        if t.isdigit():
            return False

        tu = t.upper()
        if tu in self.blacklist_upper:
            return False

        # Must contain at least one uppercase
        uppers = sum(1 for ch in t if ch.isupper())
        letters = sum(1 for ch in t if ch.isalpha())

        if uppers == 0:
            return False

        # Avoid "The", "This" style: 1 uppercase and mostly lowercase letters
        # Rule: either >=2 uppercase letters, OR has a digit, OR has hyphen, OR uppercase ratio strong.
        has_digit = any(ch.isdigit() for ch in t)
        has_hyphen = "-" in t
        upper_ratio = (uppers / letters) if letters else 1.0

        if not (uppers >= 2 or has_digit or has_hyphen or upper_ratio >= 0.60):
            return False

        # Extra: filter header/footer-esque tokens if any slipped through
        if "PAGE" in tu or "DATE" in tu:
            return False

        return True

    def _confidence_from_count(self, count: int) -> float:
        """
        Confidence grows with frequency, capped.
        """
        conf = self.base_confidence + 0.05 * max(0, count - self.min_occurrences)
        return float(min(self.max_confidence, max(0.0, conf)))