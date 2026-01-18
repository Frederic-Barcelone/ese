# corpus_metadata/A_core/A10_author_models.py
"""
Domain models for author/investigator extraction.

Detects authors, principal investigators, and contributors from clinical documents.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from A_core.A01_domain_models import Coordinate, EvidenceSpan, ValidationStatus


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


class AuthorProvenanceMetadata(BaseModel):
    """Provenance metadata for author detection."""

    pipeline_version: str
    run_id: str
    doc_fingerprint: str
    generator_name: AuthorGeneratorType
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True, extra="forbid")


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

    # Results
    authors: List[AuthorExportEntry]

    model_config = ConfigDict(extra="forbid")
