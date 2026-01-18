# corpus_metadata/C_generators/C14_strategy_citation.py
"""
Citation/reference mention detection.

Uses regex patterns to detect citations with identifiers (PMID, PMCID, DOI, NCT, URL).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from A_core.A01_domain_models import Coordinate, EvidenceSpan, ValidationStatus
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from A_core.A11_citation_models import (
    CitationCandidate,
    CitationExportDocument,
    CitationExportEntry,
    CitationGeneratorType,
    CitationIdentifierType,
    CitationProvenanceMetadata,
    ExtractedCitation,
)
from B_parsing.B02_doc_graph import DocumentGraph


class CitationDetector:
    """
    Detect citation/reference mentions using regex patterns.

    Detects PMID, PMCID, DOI, NCT identifiers, and URLs in clinical documents.
    """

    # Citation identifier patterns
    CITATION_PATTERNS: Dict[CitationIdentifierType, List[re.Pattern]] = {
        CitationIdentifierType.PMID: [
            # Explicit PMID prefix required - no standalone number matching to avoid NCT confusion
            re.compile(r"PMID[:\s]*(\d{7,8})", re.IGNORECASE),
            re.compile(r"PubMed\s*(?:ID)?[:\s]*(\d{7,8})", re.IGNORECASE),
        ],
        CitationIdentifierType.PMCID: [
            re.compile(r"PMC\s*(\d{6,8})", re.IGNORECASE),
            re.compile(r"PMCID[:\s]*PMC?(\d{6,8})", re.IGNORECASE),
        ],
        CitationIdentifierType.DOI: [
            # DOI pattern - handles both bare DOIs and doi.org URLs
            re.compile(r"(?:doi[:\s]*)(10\.\d{4,9}/[^\s\])<>\"',;\n]+)", re.IGNORECASE),
            re.compile(r"(?<!\w)10\.\d{4,9}/[^\s\])<>\"',;\n]+"),  # Bare DOI starting with 10.
            re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s\])<>\"',;\n]+)"),
        ],
        CitationIdentifierType.NCT: [
            re.compile(r"NCT[:\s]*(\d{8})", re.IGNORECASE),
            re.compile(r"ClinicalTrials\.gov[:\s]*(?:NCT)?(\d{8})", re.IGNORECASE),
        ],
        CitationIdentifierType.URL: [
            # URL pattern - exclude doi.org URLs (handled by DOI pattern)
            re.compile(r"https?://(?!(?:dx\.)?doi\.org)(?:www\.)?[^\s\])<>\"'\n]+"),
        ],
    }

    # NCT pattern for filtering false PMID matches
    NCT_NUMBER_PATTERN = re.compile(r"NCT\s*(\d{8})", re.IGNORECASE)

    # Reference section header patterns
    REFERENCE_SECTION_PATTERNS = [
        re.compile(r"^References?\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^Bibliography\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^Literature\s+Cited\s*$", re.IGNORECASE | re.MULTILINE),
    ]

    # Numbered reference pattern (e.g., "1. Smith J et al...")
    NUMBERED_REF_PATTERN = re.compile(
        r"^\s*(\d{1,3})\.\s+(.+?)(?=\n\s*\d{1,3}\.|$)",
        re.MULTILINE | re.DOTALL,
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.run_id = str(self.config.get("run_id") or generate_run_id("CITATION"))
        self.pipeline_version = (
            self.config.get("pipeline_version") or get_git_revision_hash()
        )

    def detect(
        self,
        doc_graph: DocumentGraph,
        doc_id: str,
        doc_fingerprint: str,
        full_text: Optional[str] = None,
    ) -> List[CitationCandidate]:
        """
        Detect citation mentions in document.

        Args:
            doc_graph: Document graph
            doc_id: Document identifier
            doc_fingerprint: Document fingerprint
            full_text: Optional pre-built full text

        Returns:
            List of CitationCandidate objects
        """
        candidates: List[CitationCandidate] = []
        seen_identifiers: Set[str] = set()

        # Build full text if not provided
        if not full_text:
            full_text = " ".join(
                block.text
                for block in doc_graph.iter_linear_blocks()
                if block.text
            )

        if not full_text:
            return candidates

        # Detect each identifier type
        for id_type, patterns in self.CITATION_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(full_text):
                    identifier = self._extract_identifier(match, id_type)
                    if not identifier:
                        continue

                    # Create unique key for deduplication
                    unique_key = f"{id_type.value}:{identifier}"
                    if unique_key in seen_identifiers:
                        continue

                    # Validate identifier format
                    if not self._is_valid_identifier(identifier, id_type):
                        continue

                    seen_identifiers.add(unique_key)

                    # Get context
                    context_start = max(0, match.start() - 150)
                    context_end = min(len(full_text), match.end() + 150)
                    context_text = full_text[context_start:context_end]

                    # Extract citation text (try to get full citation line)
                    citation_text = self._extract_citation_text(
                        full_text, match.start(), match.end()
                    )

                    provenance = CitationProvenanceMetadata(
                        pipeline_version=self.pipeline_version,
                        run_id=self.run_id,
                        doc_fingerprint=doc_fingerprint,
                        generator_name=CitationGeneratorType.REGEX_PATTERN,
                    )

                    # Build candidate with appropriate identifier field
                    candidate_kwargs: Dict[str, Any] = {
                        "doc_id": doc_id,
                        "citation_text": citation_text,
                        "generator_type": CitationGeneratorType.REGEX_PATTERN,
                        "identifier_types": [id_type],
                        "context_text": context_text,
                        "context_location": Coordinate(page_num=1, block_id="unknown"),
                        "initial_confidence": self._get_confidence_for_type(id_type),
                        "provenance": provenance,
                    }

                    # Set the appropriate identifier field
                    if id_type == CitationIdentifierType.PMID:
                        candidate_kwargs["pmid"] = identifier
                    elif id_type == CitationIdentifierType.PMCID:
                        candidate_kwargs["pmcid"] = f"PMC{identifier}"
                    elif id_type == CitationIdentifierType.DOI:
                        candidate_kwargs["doi"] = identifier
                    elif id_type == CitationIdentifierType.NCT:
                        candidate_kwargs["nct"] = f"NCT{identifier}" if not identifier.startswith("NCT") else identifier
                    elif id_type == CitationIdentifierType.URL:
                        candidate_kwargs["url"] = identifier

                    candidate = CitationCandidate(**candidate_kwargs)
                    candidates.append(candidate)

        # Try to merge candidates that refer to the same citation
        candidates = self._merge_related_citations(candidates)

        return candidates

    def _extract_identifier(
        self, match: re.Match, id_type: CitationIdentifierType
    ) -> Optional[str]:
        """Extract the identifier value from a regex match."""
        if id_type == CitationIdentifierType.DOI:
            # Try to get captured group first (for patterns with capture groups)
            try:
                identifier = match.group(1)
            except IndexError:
                identifier = match.group(0)
            # Clean up DOI - remove any prefix, keep just 10.xxxx/yyyy
            identifier = re.sub(r"^(?:doi[:\s]*|https?://(?:dx\.)?doi\.org/)", "", identifier, flags=re.IGNORECASE)
            # Remove trailing punctuation and whitespace
            identifier = identifier.rstrip(".,;: \t\n")
            return identifier
        elif id_type == CitationIdentifierType.URL:
            identifier = match.group(0)
            # Clean up URL - remove trailing punctuation
            identifier = identifier.rstrip(".,;:)")
            # Skip URLs that are just fragments (truncated)
            if len(identifier) < 15 or identifier.endswith("."):
                return None
            return identifier
        else:
            # For PMID, PMCID, NCT - extract captured group
            try:
                return match.group(1)
            except IndexError:
                return match.group(0)

    def _is_valid_identifier(
        self, identifier: str, id_type: CitationIdentifierType
    ) -> bool:
        """Validate identifier format."""
        if id_type == CitationIdentifierType.PMID:
            # PMID should be 7-8 digits
            return bool(re.match(r"^\d{7,8}$", identifier))
        elif id_type == CitationIdentifierType.PMCID:
            # PMCID should be 6-8 digits
            return bool(re.match(r"^\d{6,8}$", identifier))
        elif id_type == CitationIdentifierType.DOI:
            # DOI should start with 10. and have content after the slash
            if not identifier.startswith("10."):
                return False
            # Must have registrant code and suffix
            parts = identifier.split("/", 1)
            if len(parts) < 2 or len(parts[1]) < 2:
                return False
            return True
        elif id_type == CitationIdentifierType.NCT:
            # NCT should be 8 digits (with or without NCT prefix)
            clean = identifier.replace("NCT", "")
            return bool(re.match(r"^\d{8}$", clean))
        elif id_type == CitationIdentifierType.URL:
            # Basic URL validation - must start with http(s) and have meaningful content
            if not identifier.startswith(("http://", "https://")):
                return False
            # Must have domain (at least x.xx)
            domain_part = identifier.split("://", 1)[1] if "://" in identifier else ""
            if len(domain_part) < 4 or "." not in domain_part:
                return False
            # Reject truncated URLs (ending with incomplete domain)
            if domain_part.endswith("."):
                return False
            return True
        return False

    def _get_confidence_for_type(self, id_type: CitationIdentifierType) -> float:
        """Get confidence score based on identifier type."""
        confidence_map = {
            CitationIdentifierType.PMID: 0.95,
            CitationIdentifierType.PMCID: 0.95,
            CitationIdentifierType.DOI: 0.98,
            CitationIdentifierType.NCT: 0.95,
            CitationIdentifierType.URL: 0.80,
        }
        return confidence_map.get(id_type, 0.85)

    def _extract_citation_text(
        self, full_text: str, match_start: int, match_end: int
    ) -> str:
        """Extract focused citation text around a match."""
        # Get a reasonable window around the match (50 chars before, 80 after)
        context_before = 50
        context_after = 80

        citation_start = max(0, match_start - context_before)
        citation_end = min(len(full_text), match_end + context_after)

        # Try to start at a word boundary
        if citation_start > 0:
            # Look for space or newline to start at word boundary
            prefix = full_text[max(0, citation_start - 20):citation_start]
            space_pos = prefix.rfind(" ")
            newline_pos = prefix.rfind("\n")
            boundary = max(space_pos, newline_pos)
            if boundary >= 0:
                citation_start = citation_start - (len(prefix) - boundary - 1)

        # Try to end at a word/sentence boundary
        if citation_end < len(full_text):
            suffix = full_text[citation_end:min(len(full_text), citation_end + 30)]
            # Prefer sentence end
            for end_char in [".", ")", " "]:
                pos = suffix.find(end_char)
                if pos >= 0:
                    citation_end = citation_end + pos + 1
                    break

        citation_text = full_text[citation_start:citation_end].strip()

        # Clean up - remove excessive whitespace
        citation_text = re.sub(r"\s+", " ", citation_text)

        # Truncate if still too long
        if len(citation_text) > 200:
            citation_text = citation_text[:200]

        return citation_text

    def _merge_related_citations(
        self, candidates: List[CitationCandidate]
    ) -> List[CitationCandidate]:
        """
        Merge candidates that appear to be from the same citation.

        E.g., if PMID and DOI are found near each other, combine them.
        """
        if len(candidates) <= 1:
            return candidates

        # Group by proximity in context
        merged: List[CitationCandidate] = []
        used: Set[int] = set()

        for i, c1 in enumerate(candidates):
            if i in used:
                continue

            # Find other candidates with overlapping context
            related: List[CitationCandidate] = [c1]
            used.add(i)

            for j, c2 in enumerate(candidates):
                if j in used:
                    continue

                # Check if contexts overlap significantly
                if self._contexts_overlap(c1.context_text, c2.context_text):
                    related.append(c2)
                    used.add(j)

            # Merge related candidates
            if len(related) > 1:
                merged_candidate = self._merge_candidates(related)
                merged.append(merged_candidate)
            else:
                merged.append(c1)

        return merged

    def _contexts_overlap(self, ctx1: str, ctx2: str) -> bool:
        """Check if two context strings overlap significantly."""
        # Simple overlap check based on common substring length
        shorter = min(ctx1, ctx2, key=len)
        longer = max(ctx1, ctx2, key=len)

        # Check if significant portion of shorter is in longer
        for i in range(len(shorter) - 20):
            if shorter[i : i + 20] in longer:
                return True
        return False

    def _merge_candidates(
        self, candidates: List[CitationCandidate]
    ) -> CitationCandidate:
        """Merge multiple candidates into one."""
        base = candidates[0]

        # Collect all identifiers
        pmid = base.pmid
        pmcid = base.pmcid
        doi = base.doi
        nct = base.nct
        url = base.url
        identifier_types = list(base.identifier_types)

        for c in candidates[1:]:
            if c.pmid and not pmid:
                pmid = c.pmid
            if c.pmcid and not pmcid:
                pmcid = c.pmcid
            if c.doi and not doi:
                doi = c.doi
            if c.nct and not nct:
                nct = c.nct
            if c.url and not url:
                url = c.url

            for it in c.identifier_types:
                if it not in identifier_types:
                    identifier_types.append(it)

        # Use the longest citation text
        citation_text = max((c.citation_text for c in candidates), key=len)

        # Use highest confidence
        confidence = max(c.initial_confidence for c in candidates)

        return CitationCandidate(
            doc_id=base.doc_id,
            pmid=pmid,
            pmcid=pmcid,
            doi=doi,
            nct=nct,
            url=url,
            citation_text=citation_text,
            generator_type=base.generator_type,
            identifier_types=identifier_types,
            context_text=base.context_text,
            context_location=base.context_location,
            initial_confidence=confidence,
            provenance=base.provenance,
        )

    def validate_candidates(
        self, candidates: List[CitationCandidate]
    ) -> List[ExtractedCitation]:
        """
        Validate citation candidates.

        Since these are pattern matches for well-defined identifiers,
        we auto-validate with the initial confidence.
        """
        validated: List[ExtractedCitation] = []

        for candidate in candidates:
            # Build evidence text
            evidence_parts = []
            if candidate.pmid:
                evidence_parts.append(f"PMID:{candidate.pmid}")
            if candidate.pmcid:
                evidence_parts.append(f"PMCID:{candidate.pmcid}")
            if candidate.doi:
                evidence_parts.append(f"DOI:{candidate.doi}")
            if candidate.nct:
                evidence_parts.append(f"NCT:{candidate.nct}")
            if candidate.url:
                evidence_parts.append(candidate.url)

            evidence_text = " | ".join(evidence_parts) if evidence_parts else candidate.citation_text

            evidence = EvidenceSpan(
                text=evidence_text,
                location=candidate.context_location,
                scope_ref="citation_detection",
                start_char_offset=0,
                end_char_offset=len(evidence_text),
            )

            extracted = ExtractedCitation(
                candidate_id=candidate.id,
                doc_id=candidate.doc_id,
                pmid=candidate.pmid,
                pmcid=candidate.pmcid,
                doi=candidate.doi,
                nct=candidate.nct,
                url=candidate.url,
                citation_text=candidate.citation_text,
                citation_number=candidate.citation_number,
                primary_evidence=evidence,
                status=ValidationStatus.VALIDATED,
                confidence_score=candidate.initial_confidence,
                validation_flags=["pattern_match"],
                provenance=candidate.provenance,
            )
            validated.append(extracted)

        return validated

    def export_to_json(
        self,
        validated: List[ExtractedCitation],
        doc_id: str,
        doc_path: Optional[str] = None,
    ) -> CitationExportDocument:
        """
        Create export document for citations.
        """
        entries: List[CitationExportEntry] = []
        unique_ids: Set[str] = set()

        for citation in validated:
            # Track unique identifiers
            if citation.pmid:
                unique_ids.add(f"pmid:{citation.pmid}")
            if citation.pmcid:
                unique_ids.add(f"pmcid:{citation.pmcid}")
            if citation.doi:
                unique_ids.add(f"doi:{citation.doi}")
            if citation.nct:
                unique_ids.add(f"nct:{citation.nct}")

            entry = CitationExportEntry(
                pmid=citation.pmid,
                pmcid=citation.pmcid,
                doi=citation.doi,
                nct=citation.nct,
                url=citation.url,
                citation_text=citation.citation_text,
                citation_number=citation.citation_number,
                confidence=citation.confidence_score,
                page=citation.primary_evidence.location.page_num,
            )
            entries.append(entry)

        return CitationExportDocument(
            run_id=self.run_id,
            timestamp=self.config.get("timestamp", ""),
            document=doc_id,
            document_path=doc_path,
            pipeline_version=self.pipeline_version,
            total_detected=len(validated),
            unique_identifiers=len(unique_ids),
            citations=entries,
        )

    def print_summary(self) -> None:
        """Print loading summary."""
        print("  Citation detector: regex-based detection initialized")
