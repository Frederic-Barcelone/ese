"""
Citation and reference detection using regex patterns.

This module detects academic citations and identifier references in clinical
documents, extracting structured identifiers for validation and linking to
external databases. Handles complex DOI formats and PDF extraction artifacts.

Key Components:
    - CitationDetector: Main detector for citation extraction
    - Supported identifier types:
        - PMID: PubMed identifiers (7-8 digit numbers)
        - PMCID: PubMed Central identifiers (PMC + 6-8 digits)
        - DOI: Digital Object Identifiers (10.xxxx/yyyy format)
        - NCT: ClinicalTrials.gov identifiers (NCT + 8 digits)
        - URL: Web URLs (excluding doi.org, handled as DOIs)
    - Features:
        - Handles DOIs with parentheses (e.g., Lancet S0140-6736(25)01148-1)
        - Fixes broken DOIs from PDF line breaks
        - Validates and rejects truncated URLs
        - Merges related citations near each other

Example:
    >>> from C_generators.C14_strategy_citation import CitationDetector
    >>> detector = CitationDetector()
    >>> candidates = detector.detect(doc_graph, "doc123", "fingerprint")
    >>> for c in candidates:
    ...     print(f"{c.identifier_type}: {c.doi or c.pmid or c.nct}")
    DOI: 10.1016/S0140-6736(25)01148-1
    PMID: 12345678

Dependencies:
    - A_core.A00_logging: Logging utilities
    - A_core.A01_domain_models: Coordinate, EvidenceSpan, ValidationStatus
    - A_core.A03_provenance: Provenance tracking utilities
    - A_core.A11_citation_models: CitationCandidate, CitationIdentifierType
    - B_parsing.B02_doc_graph: DocumentGraph for text extraction
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from A_core.A00_logging import get_logger
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

if TYPE_CHECKING:
    from A_core.A23_doc_graph_models import DocumentGraph

# Module logger
logger = get_logger(__name__)


class CitationDetector:
    """
    Detect citation and reference mentions using regex patterns.

    This class scans document text for academic identifiers (PMID, DOI, NCT, etc.)
    and extracts structured citation candidates for downstream validation.

    The detector handles common edge cases in PDF extraction:
        - DOIs broken by line breaks
        - Parentheses in DOI suffixes (e.g., Lancet format)
        - Truncated URLs from PDF rendering issues

    Attributes:
        CITATION_PATTERNS: Compiled regex patterns for each identifier type.
        config: Configuration dictionary.
        run_id: Unique identifier for this detection run.
        pipeline_version: Version string for provenance tracking.

    Example:
        >>> detector = CitationDetector({"run_id": "RUN_123"})
        >>> candidates = detector.detect(doc_graph, "doc_id", "fingerprint")
        >>> validated = detector.validate_candidates(candidates)
        >>> export = detector.export_to_json(validated, "doc_id")
    """

    # Citation identifier patterns
    # Note: DOIs can contain parentheses (e.g., 10.1016/S0140-6736(25)01148-1)
    # so we allow () in patterns but strip trailing unbalanced ) later
    CITATION_PATTERNS: Dict[CitationIdentifierType, List[re.Pattern[str]]] = {
        CitationIdentifierType.PMID: [
            # Explicit PMID prefix required - no standalone number matching
            # to avoid confusion with NCT numbers
            re.compile(r"PMID[:\s]*(\d{7,8})", re.IGNORECASE),
            re.compile(r"PubMed\s*(?:ID)?[:\s]*(\d{7,8})", re.IGNORECASE),
        ],
        CitationIdentifierType.PMCID: [
            re.compile(r"PMC\s*(\d{6,8})", re.IGNORECASE),
            re.compile(r"PMCID[:\s]*PMC?(\d{6,8})", re.IGNORECASE),
        ],
        CitationIdentifierType.DOI: [
            # DOI pattern - allow parentheses (common in Lancet DOIs)
            re.compile(r"(?:doi[:\s]*)(10\.\d{4,9}/[^\s\]<>\"',;\n]+)", re.IGNORECASE),
            re.compile(r"(?<!\w)10\.\d{4,9}/[^\s\]<>\"',;\n]+"),  # Bare DOI
            re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s\]<>\"',;\n]+)"),
        ],
        CitationIdentifierType.NCT: [
            re.compile(r"NCT[:\s]*(\d{8})", re.IGNORECASE),
            re.compile(r"ClinicalTrials\.gov[:\s]*(?:NCT)?(\d{8})", re.IGNORECASE),
        ],
        CitationIdentifierType.URL: [
            # URL pattern - exclude doi.org URLs (handled by DOI pattern)
            re.compile(r"https?://(?!(?:dx\.)?doi\.org)(?:www\.)?[^\s\]<>\"'\n]+"),
        ],
    }

    # NCT pattern for filtering false PMID matches
    NCT_NUMBER_PATTERN: re.Pattern[str] = re.compile(r"NCT\s*(\d{8})", re.IGNORECASE)

    # Pattern to fix broken DOIs (line breaks in PDF extraction)
    BROKEN_DOI_PATTERN: re.Pattern[str] = re.compile(
        r"(10\.\d{4,9}/)\s+(\S+)",  # DOI prefix, whitespace, then continuation
    )

    # Reference section header patterns
    REFERENCE_SECTION_PATTERNS: List[re.Pattern[str]] = [
        re.compile(r"^References?\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^Bibliography\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^Literature\s+Cited\s*$", re.IGNORECASE | re.MULTILINE),
    ]

    # Numbered reference pattern (e.g., "1. Smith J et al...")
    NUMBERED_REF_PATTERN: re.Pattern[str] = re.compile(
        r"^\s*(\d{1,3})\.\s+(.+?)(?=\n\s*\d{1,3}\.|$)",
        re.MULTILINE | re.DOTALL,
    )

    # Known truncated TLDs (incomplete government/organization domains)
    TRUNCATED_TLDS: Set[str] = {"fda", "nih", "cdc", "cms", "hhs", "europa", "who", "ema"}

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the citation detector.

        Args:
            config: Optional configuration dictionary with keys:
                - run_id: Unique run identifier (auto-generated if not provided)
                - pipeline_version: Version string for provenance
                - timestamp: ISO timestamp for exports
        """
        self.config = config or {}
        self.run_id = str(self.config.get("run_id") or generate_run_id("CITATION"))
        self.pipeline_version = (
            self.config.get("pipeline_version") or get_git_revision_hash()
        )
        logger.debug(f"CitationDetector initialized with run_id={self.run_id}")

    def _normalize_broken_dois(self, text: str) -> str:
        """
        Fix DOIs that are broken by line breaks in PDF text extraction.

        PDF extraction can introduce line breaks/spaces within DOIs, e.g.:
        "10.1016/ S0140-6736(25)01148-1" should become
        "10.1016/S0140-6736(25)01148-1"

        Args:
            text: Input text potentially containing broken DOIs.

        Returns:
            Text with DOIs rejoined.
        """
        return self.BROKEN_DOI_PATTERN.sub(r"\1\2", text)

    def detect(
        self,
        doc_graph: "DocumentGraph",
        doc_id: str,
        doc_fingerprint: str,
        full_text: Optional[str] = None,
    ) -> List[CitationCandidate]:
        """
        Detect citation mentions in a document.

        Scans the document text for all supported identifier types,
        validates their format, and creates structured candidates.

        Args:
            doc_graph: Parsed document graph containing text blocks.
            doc_id: Unique document identifier.
            doc_fingerprint: Hash fingerprint for provenance.
            full_text: Optional pre-built full text (extracted if not provided).

        Returns:
            List of CitationCandidate objects, deduplicated and potentially merged.
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
            logger.debug(f"No text found in document {doc_id}")
            return candidates

        # Preprocess: Fix broken DOIs
        full_text = self._normalize_broken_dois(full_text)

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
                        logger.debug(f"Invalid {id_type.value}: {identifier}")
                        continue

                    seen_identifiers.add(unique_key)

                    # Get context
                    context_start = max(0, match.start() - 150)
                    context_end = min(len(full_text), match.end() + 150)
                    context_text = full_text[context_start:context_end]

                    # Extract citation text
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
                        nct_value = identifier if identifier.startswith("NCT") else f"NCT{identifier}"
                        candidate_kwargs["nct"] = nct_value
                    elif id_type == CitationIdentifierType.URL:
                        candidate_kwargs["url"] = identifier

                    candidate = CitationCandidate(**candidate_kwargs)
                    candidates.append(candidate)

        logger.debug(f"Found {len(candidates)} citation candidates before merging")

        # Merge candidates that refer to the same citation
        candidates = self._merge_related_citations(candidates)

        logger.debug(f"Found {len(candidates)} citation candidates after merging")
        return candidates

    def _extract_identifier(
        self,
        match: re.Match[str],
        id_type: CitationIdentifierType,
    ) -> Optional[str]:
        """
        Extract the identifier value from a regex match.

        Handles cleanup for each identifier type:
        - DOIs: Remove prefixes, balance parentheses
        - URLs: Clean trailing punctuation, validate completeness
        - Others: Extract captured group

        Args:
            match: Regex match object.
            id_type: Type of identifier being extracted.

        Returns:
            Cleaned identifier string, or None if invalid.
        """
        if id_type == CitationIdentifierType.DOI:
            # Try to get captured group first
            try:
                identifier = match.group(1)
            except IndexError:
                identifier = match.group(0)

            # Clean up DOI - remove any prefix
            identifier = re.sub(
                r"^(?:doi[:\s]*|https?://(?:dx\.)?doi\.org/)",
                "",
                identifier,
                flags=re.IGNORECASE,
            )
            # Remove trailing punctuation
            identifier = identifier.rstrip(".,;: \t\n")
            # Balance parentheses
            identifier = self._balance_parentheses(identifier)
            return identifier

        elif id_type == CitationIdentifierType.URL:
            identifier = match.group(0)
            # Clean up URL
            identifier = identifier.rstrip(".,;:")
            identifier = self._balance_parentheses(identifier)

            # Skip truncated URLs
            if len(identifier) < 15 or identifier.endswith("."):
                return None
            return identifier

        else:
            # For PMID, PMCID, NCT - extract captured group
            try:
                return match.group(1)
            except IndexError:
                return match.group(0)

    def _balance_parentheses(self, text: str) -> str:
        """
        Remove trailing unbalanced closing parentheses.

        DOIs like 10.1016/S0140-6736(25)01148-1 have balanced ()
        but we may capture extra ) from surrounding text.

        Args:
            text: Input string potentially with unbalanced parens.

        Returns:
            String with balanced parentheses.
        """
        if not text:
            return text

        open_count = text.count("(")
        close_count = text.count(")")

        while close_count > open_count and text.endswith(")"):
            text = text[:-1]
            close_count -= 1

        return text

    def _is_valid_identifier(
        self,
        identifier: str,
        id_type: CitationIdentifierType,
    ) -> bool:
        """
        Validate identifier format.

        Each identifier type has specific format requirements:
        - PMID: 7-8 digits
        - PMCID: 6-8 digits
        - DOI: 10.xxxx/yyyy with meaningful suffix
        - NCT: 8 digits
        - URL: Valid scheme, domain, and TLD

        Args:
            identifier: The identifier string to validate.
            id_type: Type of identifier.

        Returns:
            True if valid, False otherwise.
        """
        if id_type == CitationIdentifierType.PMID:
            return bool(re.match(r"^\d{7,8}$", identifier))

        elif id_type == CitationIdentifierType.PMCID:
            return bool(re.match(r"^\d{6,8}$", identifier))

        elif id_type == CitationIdentifierType.DOI:
            if not identifier.startswith("10."):
                return False
            parts = identifier.split("/", 1)
            if len(parts) < 2 or len(parts[1]) < 2:
                return False
            return True

        elif id_type == CitationIdentifierType.NCT:
            clean = identifier.replace("NCT", "")
            return bool(re.match(r"^\d{8}$", clean))

        elif id_type == CitationIdentifierType.URL:
            # Must start with http(s)
            if not identifier.startswith(("http://", "https://")):
                return False

            # Must have domain
            domain_part = identifier.split("://", 1)[1] if "://" in identifier else ""
            if len(domain_part) < 4 or "." not in domain_part:
                return False

            # Reject incomplete domains
            if domain_part.endswith("."):
                return False

            # Extract TLD
            domain_only = domain_part.split("/")[0].split("?")[0]
            parts = domain_only.split(".")
            if len(parts) < 2:
                return False

            tld = parts[-1].lower()

            # Reject truncated government/organization domains
            if tld in self.TRUNCATED_TLDS:
                return False

            # TLD should be at least 2 chars
            if len(tld) < 2:
                return False

            return True

        return False

    def _get_confidence_for_type(self, id_type: CitationIdentifierType) -> float:
        """
        Get confidence score based on identifier type.

        DOIs have highest confidence (0.98) as they're well-structured.
        URLs have lowest confidence (0.80) due to truncation risk.

        Args:
            id_type: Type of identifier.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        confidence_map = {
            CitationIdentifierType.PMID: 0.95,
            CitationIdentifierType.PMCID: 0.95,
            CitationIdentifierType.DOI: 0.98,
            CitationIdentifierType.NCT: 0.95,
            CitationIdentifierType.URL: 0.80,
        }
        return confidence_map.get(id_type, 0.85)

    def _extract_citation_text(
        self,
        full_text: str,
        match_start: int,
        match_end: int,
    ) -> str:
        """
        Extract focused citation text around a match.

        Attempts to find natural word/sentence boundaries for cleaner
        extraction.

        Args:
            full_text: Complete document text.
            match_start: Start position of the identifier match.
            match_end: End position of the identifier match.

        Returns:
            Extracted citation text, max 200 characters.
        """
        context_before = 50
        context_after = 80

        citation_start = max(0, match_start - context_before)
        citation_end = min(len(full_text), match_end + context_after)

        # Try to start at a word boundary
        if citation_start > 0:
            prefix = full_text[max(0, citation_start - 20):citation_start]
            space_pos = prefix.rfind(" ")
            newline_pos = prefix.rfind("\n")
            boundary = max(space_pos, newline_pos)
            if boundary >= 0:
                citation_start = citation_start - (len(prefix) - boundary - 1)

        # Try to end at a word/sentence boundary
        if citation_end < len(full_text):
            suffix = full_text[citation_end:min(len(full_text), citation_end + 30)]
            for end_char in [".", ")", " "]:
                pos = suffix.find(end_char)
                if pos >= 0:
                    citation_end = citation_end + pos + 1
                    break

        citation_text = full_text[citation_start:citation_end].strip()
        citation_text = re.sub(r"\s+", " ", citation_text)

        if len(citation_text) > 200:
            citation_text = citation_text[:200]

        return citation_text

    def _merge_related_citations(
        self,
        candidates: List[CitationCandidate],
    ) -> List[CitationCandidate]:
        """
        Merge candidates that appear to be from the same citation.

        If PMID and DOI are found near each other, combine them into
        a single citation record.

        Args:
            candidates: List of citation candidates.

        Returns:
            List with related candidates merged.
        """
        if len(candidates) <= 1:
            return candidates

        merged: List[CitationCandidate] = []
        used: Set[int] = set()

        for i, c1 in enumerate(candidates):
            if i in used:
                continue

            related: List[CitationCandidate] = [c1]
            used.add(i)

            for j, c2 in enumerate(candidates):
                if j in used:
                    continue

                if self._contexts_overlap(c1.context_text, c2.context_text):
                    related.append(c2)
                    used.add(j)

            if len(related) > 1:
                merged_candidate = self._merge_candidates(related)
                merged.append(merged_candidate)
            else:
                merged.append(c1)

        return merged

    def _contexts_overlap(self, ctx1: str, ctx2: str) -> bool:
        """
        Check if two context strings overlap significantly.

        Uses substring matching to detect overlapping context windows.

        Args:
            ctx1: First context string.
            ctx2: Second context string.

        Returns:
            True if significant overlap detected.
        """
        shorter = min(ctx1, ctx2, key=len)
        longer = max(ctx1, ctx2, key=len)

        for i in range(len(shorter) - 20):
            if shorter[i : i + 20] in longer:
                return True
        return False

    def _merge_candidates(
        self,
        candidates: List[CitationCandidate],
    ) -> CitationCandidate:
        """
        Merge multiple candidates into one.

        Combines all identifiers, uses longest citation text,
        and takes highest confidence.

        Args:
            candidates: List of candidates to merge.

        Returns:
            Single merged CitationCandidate.
        """
        base = candidates[0]

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

        citation_text = max((c.citation_text for c in candidates), key=len)
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
        self,
        candidates: List[CitationCandidate],
    ) -> List[ExtractedCitation]:
        """
        Validate citation candidates.

        Since these are pattern matches for well-defined identifiers,
        candidates are auto-validated with their initial confidence.

        Args:
            candidates: List of candidates to validate.

        Returns:
            List of ExtractedCitation objects.
        """
        validated: List[ExtractedCitation] = []

        for candidate in candidates:
            evidence_parts: List[str] = []
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

        Args:
            validated: List of validated citations.
            doc_id: Document identifier.
            doc_path: Optional file path for provenance.

        Returns:
            CitationExportDocument ready for JSON serialization.
        """
        entries: List[CitationExportEntry] = []
        unique_ids: Set[str] = set()

        for citation in validated:
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
        """Print loading summary to console."""
        logger.info("Citation detector: regex-based detection initialized")
