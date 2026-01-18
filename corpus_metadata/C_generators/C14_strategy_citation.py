# corpus_metadata/C_generators/C14_strategy_citation.py
"""
Citation/reference mention detection.

Uses regex patterns to detect citations with identifiers (PMID, PMCID, DOI, NCT, URL).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

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
            re.compile(r"PMID[:\s]*(\d{7,8})", re.IGNORECASE),
            re.compile(r"PubMed\s*(?:ID)?[:\s]*(\d{7,8})", re.IGNORECASE),
            re.compile(r"(?:^|[^\d])(\d{8})(?:[^\d]|$)"),  # Standalone 8-digit numbers in reference context
        ],
        CitationIdentifierType.PMCID: [
            re.compile(r"PMC\s*(\d{6,8})", re.IGNORECASE),
            re.compile(r"PMCID[:\s]*PMC?(\d{6,8})", re.IGNORECASE),
        ],
        CitationIdentifierType.DOI: [
            re.compile(r"(?:doi[:\s]*)?10\.\d{4,9}/[^\s\])<>\"',;]+", re.IGNORECASE),
            re.compile(r"https?://doi\.org/10\.\d{4,9}/[^\s\])<>\"',;]+"),
        ],
        CitationIdentifierType.NCT: [
            re.compile(r"NCT\s*(\d{8})", re.IGNORECASE),
            re.compile(r"ClinicalTrials\.gov[:\s]*(?:NCT)?(\d{8})", re.IGNORECASE),
        ],
        CitationIdentifierType.URL: [
            re.compile(r"https?://(?:www\.)?[^\s\])<>\"']+"),
        ],
    }

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
        if id_type in (CitationIdentifierType.DOI, CitationIdentifierType.URL):
            # For DOI and URL, take the full match
            identifier = match.group(0)
            # Clean up DOI prefix
            if id_type == CitationIdentifierType.DOI:
                identifier = re.sub(r"^(?:doi[:\s]*|https?://doi\.org/)", "", identifier, flags=re.IGNORECASE)
                # Remove trailing punctuation
                identifier = identifier.rstrip(".,;:")
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
            # DOI should start with 10.
            return identifier.startswith("10.")
        elif id_type == CitationIdentifierType.NCT:
            # NCT should be 8 digits (with or without NCT prefix)
            clean = identifier.replace("NCT", "")
            return bool(re.match(r"^\d{8}$", clean))
        elif id_type == CitationIdentifierType.URL:
            # Basic URL validation
            return identifier.startswith(("http://", "https://"))
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
        """Extract the full citation text around a match."""
        # Look for the start of the citation (numbered reference or line start)
        search_start = max(0, match_start - 300)
        prefix = full_text[search_start:match_start]

        # Find line start or reference number
        line_match = re.search(r"(?:^|\n)\s*(\d{1,3}\.)?\s*", prefix[::-1])
        if line_match:
            citation_start = match_start - line_match.end()
        else:
            citation_start = max(0, match_start - 100)

        # Look for end of citation (next reference number or double newline)
        search_end = min(len(full_text), match_end + 300)
        suffix = full_text[match_end:search_end]

        end_match = re.search(r"\n\s*\d{1,3}\.|(\n\s*\n)", suffix)
        if end_match:
            citation_end = match_end + end_match.start()
        else:
            citation_end = min(len(full_text), match_end + 200)

        citation_text = full_text[citation_start:citation_end].strip()

        # Truncate if too long
        if len(citation_text) > 500:
            citation_text = citation_text[:500] + "..."

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
        overlap_threshold = len(shorter) * 0.5
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
