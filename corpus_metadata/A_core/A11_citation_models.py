# corpus_metadata/A_core/A11_citation_models.py
"""
Domain models for citation/reference extraction.

Detects citations with identifiers (PMID, PMCID, DOI, NCT, URL) from clinical documents.
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
# Citation-specific enums
# -------------------------


class CitationIdentifierType(str, Enum):
    """Type of citation identifier found."""

    PMID = "pmid"
    PMCID = "pmcid"
    DOI = "doi"
    NCT = "nct"
    URL = "url"


class CitationGeneratorType(str, Enum):
    """Tracks which strategy produced the citation candidate."""

    REGEX_PATTERN = "gen:citation_regex"  # Regex pattern match
    REFERENCE_SECTION = "gen:citation_reference"  # From reference section
    INLINE_CITATION = "gen:citation_inline"  # Inline citation marker


# -------------------------
# Citation provenance
# -------------------------


class CitationProvenanceMetadata(BaseProvenanceMetadata):
    """
    Provenance metadata for citation detection.

    Inherits common provenance fields from BaseProvenanceMetadata and adds:
    - generator_name: CitationGeneratorType (overrides base Enum type)

    Note: No lexicon_ids field - citation detection uses regex-based extraction.
    """

    generator_name: CitationGeneratorType


# -------------------------
# CitationCandidate (pre-verification)
# -------------------------


class CitationCandidate(BaseModel):
    """Pre-validation citation mention."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    doc_id: str

    # Citation identifiers (at least one should be set)
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    nct: Optional[str] = None
    url: Optional[str] = None

    # Citation text
    citation_text: str  # Full citation string if available
    citation_number: Optional[int] = None  # Reference number [1], [2], etc.

    # Detection metadata
    generator_type: CitationGeneratorType
    identifier_types: List[CitationIdentifierType] = Field(default_factory=list)

    # Context
    context_text: str
    context_location: Coordinate

    # Confidence
    initial_confidence: float = Field(default=0.9, ge=0.0, le=1.0)

    provenance: CitationProvenanceMetadata

    model_config = ConfigDict(extra="forbid")


# -------------------------
# ExtractedCitation (post-verification)
# -------------------------


class ExtractedCitation(BaseModel):
    """Validated citation entity for output."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    candidate_id: uuid.UUID

    doc_id: str
    schema_version: str = "1.0.0"

    # Citation identifiers
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    nct: Optional[str] = None
    url: Optional[str] = None

    # Citation text
    citation_text: str
    citation_number: Optional[int] = None

    # Evidence
    primary_evidence: EvidenceSpan
    supporting_evidence: List[EvidenceSpan] = Field(default_factory=list)

    # Verdict
    status: ValidationStatus
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    validation_flags: List[str] = Field(default_factory=list)

    # Audit trail
    provenance: CitationProvenanceMetadata

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Output schema for JSON export
# -------------------------


class CitationValidation(BaseModel):
    """Validation result for a citation identifier."""

    is_valid: bool
    resolved_url: Optional[str] = None
    title: Optional[str] = None
    status: Optional[str] = None  # For NCT: trial status
    error: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class CitationExportEntry(BaseModel):
    """Simplified citation entry for JSON export."""

    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    nct: Optional[str] = None
    url: Optional[str] = None
    citation_text: str
    citation_number: Optional[int] = None
    page: Optional[int] = None
    confidence: float

    # API validation results
    validation: Optional[CitationValidation] = None

    model_config = ConfigDict(extra="forbid")


class CitationValidationSummary(BaseModel):
    """Summary of citation validation results."""

    total_validated: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    error_count: int = 0

    model_config = ConfigDict(extra="forbid")


class CitationExportDocument(BaseModel):
    """Complete citation extraction output for a document."""

    run_id: str
    timestamp: str  # ISO format
    document: str
    document_path: Optional[str] = None
    pipeline_version: str

    # Counts
    total_detected: int
    unique_identifiers: int

    # Validation summary
    validation_summary: Optional[CitationValidationSummary] = None

    # Results
    citations: List[CitationExportEntry]

    model_config = ConfigDict(extra="forbid")
