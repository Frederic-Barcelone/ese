# corpus_metadata/C_generators/C13_strategy_author.py
"""
Author/investigator mention detection.

Uses regex patterns to detect author names, roles, and affiliations.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from A_core.A01_domain_models import Coordinate, EvidenceSpan, ValidationStatus
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from A_core.A10_author_models import (
    AuthorCandidate,
    AuthorExportDocument,
    AuthorExportEntry,
    AuthorGeneratorType,
    AuthorProvenanceMetadata,
    AuthorRoleType,
    ExtractedAuthor,
)
from B_parsing.B02_doc_graph import DocumentGraph


class AuthorDetector:
    """
    Detect author/investigator mentions using regex patterns.

    Detects names with academic credentials, role-prefixed names,
    and author blocks in clinical documents.
    """

    # Name with credentials pattern (e.g., "John Smith, MD, PhD")
    NAME_WITH_CREDENTIALS = re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)"  # Name
        r"(?:,\s*(?:MD|PhD|MPH|DO|PharmD|RN|MSN|DrPH|ScD|MBBS|MBChB|FRCPC?|FACP))+",
        re.MULTILINE,
    )

    # Role-prefixed name patterns
    ROLE_PATTERNS: List[Tuple[re.Pattern, AuthorRoleType]] = [
        (
            re.compile(
                r"Principal\s+[Ii]nvestigator[s]?[:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)",
                re.MULTILINE,
            ),
            AuthorRoleType.PRINCIPAL_INVESTIGATOR,
        ),
        (
            re.compile(
                r"Corresponding\s+[Aa]uthor[s]?[:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)",
                re.MULTILINE,
            ),
            AuthorRoleType.CORRESPONDING_AUTHOR,
        ),
        (
            re.compile(
                r"Co-?[Ii]nvestigator[s]?[:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)",
                re.MULTILINE,
            ),
            AuthorRoleType.CO_INVESTIGATOR,
        ),
        (
            re.compile(
                r"Study\s+[Cc]hair[:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)",
                re.MULTILINE,
            ),
            AuthorRoleType.STUDY_CHAIR,
        ),
        (
            re.compile(
                r"Steering\s+[Cc]ommittee[:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)",
                re.MULTILINE,
            ),
            AuthorRoleType.STEERING_COMMITTEE,
        ),
    ]

    # Email pattern for affiliation extraction
    EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")

    # ORCID pattern
    ORCID_PATTERN = re.compile(r"(?:ORCID[:\s]*)?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])")

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.run_id = str(self.config.get("run_id") or generate_run_id("AUTHOR"))
        self.pipeline_version = (
            self.config.get("pipeline_version") or get_git_revision_hash()
        )

    def detect(
        self,
        doc_graph: DocumentGraph,
        doc_id: str,
        doc_fingerprint: str,
        full_text: Optional[str] = None,
    ) -> List[AuthorCandidate]:
        """
        Detect author mentions in document.

        Args:
            doc_graph: Document graph
            doc_id: Document identifier
            doc_fingerprint: Document fingerprint
            full_text: Optional pre-built full text

        Returns:
            List of AuthorCandidate objects
        """
        candidates: List[AuthorCandidate] = []
        seen_names: Set[str] = set()

        # Build full text if not provided
        if not full_text:
            full_text = " ".join(
                block.text
                for block in doc_graph.iter_linear_blocks()
                if block.text
            )

        if not full_text:
            return candidates

        # Detect role-prefixed names (higher confidence)
        for pattern, role in self.ROLE_PATTERNS:
            for match in pattern.finditer(full_text):
                name = match.group(1).strip()
                normalized_name = self._normalize_name(name)

                if normalized_name in seen_names:
                    continue
                if not self._is_valid_name(name):
                    continue

                seen_names.add(normalized_name)

                context_start = max(0, match.start() - 100)
                context_end = min(len(full_text), match.end() + 100)
                context_text = full_text[context_start:context_end]

                # Extract email if present in context
                email_match = self.EMAIL_PATTERN.search(context_text)
                email = email_match.group(0) if email_match else None

                # Extract ORCID if present
                orcid_match = self.ORCID_PATTERN.search(context_text)
                orcid = orcid_match.group(1) if orcid_match else None

                provenance = AuthorProvenanceMetadata(
                    pipeline_version=self.pipeline_version,
                    run_id=self.run_id,
                    doc_fingerprint=doc_fingerprint,
                    generator_name=AuthorGeneratorType.REGEX_PATTERN,
                )

                candidate = AuthorCandidate(
                    doc_id=doc_id,
                    full_name=name,
                    role=role,
                    email=email,
                    orcid=orcid,
                    generator_type=AuthorGeneratorType.REGEX_PATTERN,
                    context_text=context_text,
                    context_location=Coordinate(page_num=1, block_id="unknown"),
                    initial_confidence=0.95,
                    provenance=provenance,
                )
                candidates.append(candidate)

        # Detect names with credentials (lower confidence, general authors)
        for match in self.NAME_WITH_CREDENTIALS.finditer(full_text):
            name = match.group(1).strip()
            normalized_name = self._normalize_name(name)

            if normalized_name in seen_names:
                continue
            if not self._is_valid_name(name):
                continue

            seen_names.add(normalized_name)

            context_start = max(0, match.start() - 100)
            context_end = min(len(full_text), match.end() + 100)
            context_text = full_text[context_start:context_end]

            # Try to determine role from context
            role = self._infer_role_from_context(context_text)

            # Extract email if present
            email_match = self.EMAIL_PATTERN.search(context_text)
            email = email_match.group(0) if email_match else None

            # Extract ORCID if present
            orcid_match = self.ORCID_PATTERN.search(context_text)
            orcid = orcid_match.group(1) if orcid_match else None

            provenance = AuthorProvenanceMetadata(
                pipeline_version=self.pipeline_version,
                run_id=self.run_id,
                doc_fingerprint=doc_fingerprint,
                generator_name=AuthorGeneratorType.HEADER_PATTERN,
            )

            candidate = AuthorCandidate(
                doc_id=doc_id,
                full_name=name,
                role=role,
                email=email,
                orcid=orcid,
                generator_type=AuthorGeneratorType.HEADER_PATTERN,
                context_text=context_text,
                context_location=Coordinate(page_num=1, block_id="unknown"),
                initial_confidence=0.85,
                provenance=provenance,
            )
            candidates.append(candidate)

        return candidates

    def _normalize_name(self, name: str) -> str:
        """Normalize name for deduplication."""
        return " ".join(name.lower().split())

    def _is_valid_name(self, name: str) -> bool:
        """Check if name appears to be a valid person name."""
        parts = name.split()
        if len(parts) < 2:
            return False

        # Filter out common false positives
        invalid_names = {
            "study design",
            "data analysis",
            "statistical analysis",
            "new england",
            "journal medicine",
            "clinical trial",
        }
        if self._normalize_name(name) in invalid_names:
            return False

        # Name should have reasonable length
        if len(name) < 5 or len(name) > 60:
            return False

        return True

    def _infer_role_from_context(self, context: str) -> AuthorRoleType:
        """Infer author role from surrounding context."""
        context_lower = context.lower()

        if "principal investigator" in context_lower:
            return AuthorRoleType.PRINCIPAL_INVESTIGATOR
        if "corresponding author" in context_lower:
            return AuthorRoleType.CORRESPONDING_AUTHOR
        if "co-investigator" in context_lower or "coinvestigator" in context_lower:
            return AuthorRoleType.CO_INVESTIGATOR
        if "steering committee" in context_lower:
            return AuthorRoleType.STEERING_COMMITTEE
        if "study chair" in context_lower:
            return AuthorRoleType.STUDY_CHAIR
        if "data safety" in context_lower or "dsmb" in context_lower:
            return AuthorRoleType.DATA_SAFETY_BOARD

        return AuthorRoleType.AUTHOR

    def validate_candidates(
        self, candidates: List[AuthorCandidate]
    ) -> List[ExtractedAuthor]:
        """
        Validate author candidates.

        Since these are pattern matches, we auto-validate with the initial confidence.
        """
        validated: List[ExtractedAuthor] = []

        for candidate in candidates:
            evidence = EvidenceSpan(
                text=candidate.full_name,
                location=candidate.context_location,
                scope_ref="author_detection",
                start_char_offset=0,
                end_char_offset=len(candidate.full_name),
            )

            extracted = ExtractedAuthor(
                candidate_id=candidate.id,
                doc_id=candidate.doc_id,
                full_name=candidate.full_name,
                role=candidate.role,
                affiliation=candidate.affiliation,
                email=candidate.email,
                orcid=candidate.orcid,
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
        validated: List[ExtractedAuthor],
        doc_id: str,
        doc_path: Optional[str] = None,
    ) -> AuthorExportDocument:
        """
        Create export document for authors.
        """
        entries: List[AuthorExportEntry] = []
        unique_names: Set[str] = set()

        for author in validated:
            unique_names.add(self._normalize_name(author.full_name))

            entry = AuthorExportEntry(
                full_name=author.full_name,
                role=author.role.value,
                affiliation=author.affiliation,
                email=author.email,
                orcid=author.orcid,
                confidence=author.confidence_score,
                context=author.primary_evidence.text,
                page=author.primary_evidence.location.page_num,
            )
            entries.append(entry)

        return AuthorExportDocument(
            run_id=self.run_id,
            timestamp=self.config.get("timestamp", ""),
            document=doc_id,
            document_path=doc_path,
            pipeline_version=self.pipeline_version,
            total_detected=len(validated),
            unique_authors=len(unique_names),
            authors=entries,
        )

    def print_summary(self) -> None:
        """Print loading summary."""
        print("  Author detector: regex-based detection initialized")
