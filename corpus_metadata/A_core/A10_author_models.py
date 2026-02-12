# corpus_metadata/A_core/A10_author_models.py
"""
Pydantic domain models for author and investigator extraction.

This module defines data structures for extracting author and investigator information
from clinical documents. It supports multiple author roles (principal investigator,
corresponding author, steering committee), tracks affiliations and contact information,
and handles ORCID identifiers for researcher disambiguation.

Key Components:
    - AuthorCandidate: Pre-validation author mention with role and affiliation
    - ExtractedAuthor: Validated author entity with evidence
    - AuthorExportEntry: Simplified export format for JSON output
    - AuthorExportDocument: Complete extraction results for a document
    - AuthorProvenanceMetadata: Audit trail for author detection
    - AuthorRoleType: Author role enumeration (PI, corresponding, steering committee)
    - AuthorGeneratorType: Source generator tracking (header, affiliation block, regex)

Example:
    >>> from A_core.A10_author_models import AuthorCandidate, AuthorRoleType, AuthorGeneratorType
    >>> candidate = AuthorCandidate(
    ...     doc_id="doc_001",
    ...     full_name="Jane Smith, MD, PhD",
    ...     role=AuthorRoleType.PRINCIPAL_INVESTIGATOR,
    ...     affiliation="Harvard Medical School",
    ...     orcid="0000-0001-2345-6789",
    ...     generator_type=AuthorGeneratorType.HEADER_PATTERN,
    ...     context_text="Principal Investigator: Jane Smith, MD, PhD",
    ...     context_location=Coordinate(page=1, bbox=[100, 200, 300, 220]),
    ...     provenance=provenance,
    ... )

Dependencies:
    - A_core.A01_domain_models: For BaseProvenanceMetadata, Coordinate, EvidenceSpan, ValidationStatus
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from A_core.A01_domain_models import (
    BaseProvenanceMetadata,
    Coordinate,
    EvidenceSpan,
    ValidationStatus,
)


# -------------------------
# Author-specific enums
# -------------------------


class AuthorRoleType(str, Enum):
    """Role of the author in the document."""

    AUTHOR = "author"
    PRINCIPAL_INVESTIGATOR = "principal_investigator"
    CO_INVESTIGATOR = "co_investigator"
    CORRESPONDING_AUTHOR = "corresponding_author"
    STEERING_COMMITTEE = "steering_committee"
    STUDY_CHAIR = "study_chair"
    DATA_SAFETY_BOARD = "data_safety_board"
    UNKNOWN = "unknown"


class AuthorGeneratorType(str, Enum):
    """Tracks which strategy produced the author candidate."""

    HEADER_PATTERN = "gen:author_header"  # From document header/title section
    AFFILIATION_BLOCK = "gen:author_affiliation"  # From affiliation section
    CONTRIBUTION_SECTION = "gen:author_contribution"  # From contributions section
    INVESTIGATOR_LIST = "gen:author_investigator"  # From investigator listing
    REGEX_PATTERN = "gen:author_regex"  # Regex pattern match


# -------------------------
# Author provenance
# -------------------------


class AuthorProvenanceMetadata(BaseProvenanceMetadata):
    """
    Provenance metadata for author detection.

    Inherits common provenance fields from BaseProvenanceMetadata and adds:
    - generator_name: AuthorGeneratorType (overrides base Enum type)

    Note: No lexicon_ids field - author detection uses pattern-based extraction.
    """

    generator_name: AuthorGeneratorType


# -------------------------
# AuthorCandidate (pre-verification)
# -------------------------


class AuthorCandidate(BaseModel):
    """Pre-validation author mention."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    doc_id: str

    # Author identification
    full_name: str  # Full name as found
    role: AuthorRoleType = AuthorRoleType.UNKNOWN
    affiliation: Optional[str] = None
    email: Optional[str] = None
    orcid: Optional[str] = None

    # Detection metadata
    generator_type: AuthorGeneratorType

    # Context
    context_text: str
    context_location: Coordinate

    # Confidence
    initial_confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    provenance: AuthorProvenanceMetadata

    model_config = ConfigDict(extra="forbid")


# -------------------------
# ExtractedAuthor (post-verification)
# -------------------------


class ExtractedAuthor(BaseModel):
    """Validated author entity for output."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    candidate_id: uuid.UUID

    doc_id: str
    schema_version: str = "1.0.0"

    # Author identification
    full_name: str
    role: AuthorRoleType
    affiliation: Optional[str] = None
    email: Optional[str] = None
    orcid: Optional[str] = None

    # Evidence
    primary_evidence: EvidenceSpan
    supporting_evidence: List[EvidenceSpan] = Field(default_factory=list)

    # Verdict
    status: ValidationStatus
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    validation_flags: List[str] = Field(default_factory=list)

    # Audit trail
    provenance: AuthorProvenanceMetadata

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Output schema for JSON export
# -------------------------


class AuthorExportEntry(BaseModel):
    """Simplified author entry for JSON export."""

    full_name: str
    role: str
    affiliation: Optional[str] = None
    email: Optional[str] = None
    orcid: Optional[str] = None
    context: Optional[str] = None
    page: Optional[int] = None
    confidence: float

    model_config = ConfigDict(extra="forbid")


class AuthorExportDocument(BaseModel):
    """Complete author extraction output for a document."""

    run_id: str
    timestamp: str  # ISO format
    document: str
    document_path: Optional[str] = None
    pipeline_version: str

    # Counts
    total_detected: int
    unique_authors: int

    # Disease association
    primary_disease: Optional[str] = None

    # Results
    authors: List[AuthorExportEntry]

    model_config = ConfigDict(extra="forbid")
