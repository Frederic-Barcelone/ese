# corpus_metadata/A_core/A09_pharma_models.py
"""
Pydantic domain models for pharmaceutical company entity extraction.

This module defines data structures for detecting and extracting pharmaceutical company
mentions from clinical documents using lexicon-based matching. It tracks canonical company
names, headquarters locations, and parent-subsidiary relationships to enable sponsor
identification and corporate entity resolution.

Key Components:
    - PharmaCandidate: Pre-validation pharma company mention with metadata
    - ExtractedPharma: Validated pharma company entity with evidence
    - PharmaExportEntry: Simplified export format for JSON output
    - PharmaExportDocument: Complete extraction results for a document
    - PharmaProvenanceMetadata: Audit trail for pharma detection
    - PharmaGeneratorType: Source generator tracking (lexicon match)

Example:
    >>> from A_core.A09_pharma_models import PharmaCandidate, PharmaGeneratorType
    >>> candidate = PharmaCandidate(
    ...     doc_id="doc_001",
    ...     matched_text="Novartis",
    ...     canonical_name="Novartis",
    ...     full_name="Novartis AG",
    ...     headquarters="Basel, Switzerland",
    ...     generator_type=PharmaGeneratorType.LEXICON_MATCH,
    ...     context_text="Study sponsored by Novartis Pharmaceuticals",
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
# Pharma-specific enums
# -------------------------


class PharmaGeneratorType(str, Enum):
    """Tracks which strategy produced the pharma company candidate."""

    LEXICON_MATCH = "gen:pharma_lexicon"  # Matched from pharma_companies_lexicon.json


# -------------------------
# Pharma provenance
# -------------------------


class PharmaProvenanceMetadata(BaseProvenanceMetadata):
    """
    Provenance metadata for pharma company detection.

    Inherits common provenance fields from BaseProvenanceMetadata and adds:
    - generator_name: PharmaGeneratorType (overrides base Enum type)

    Note: No lexicon_ids field - pharma detection uses simpler lexicon matching.
    """

    generator_name: PharmaGeneratorType


# -------------------------
# PharmaCandidate (pre-verification)
# -------------------------


class PharmaCandidate(BaseModel):
    """Pre-validation pharma company mention."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    doc_id: str

    # Company identification
    matched_text: str  # Exact text matched in document
    canonical_name: str  # Canonical company name (e.g., "Novartis")
    full_name: Optional[str] = None  # Full legal name (e.g., "Novartis AG")

    # Company metadata
    headquarters: Optional[str] = None  # HQ location
    parent_company: Optional[str] = None  # Parent if subsidiary
    subsidiaries: List[str] = Field(default_factory=list)

    # Detection metadata
    generator_type: PharmaGeneratorType

    # Context
    context_text: str
    context_location: Coordinate

    # Confidence
    initial_confidence: float = Field(default=0.9, ge=0.0, le=1.0)

    provenance: PharmaProvenanceMetadata

    model_config = ConfigDict(extra="forbid")


# -------------------------
# ExtractedPharma (post-verification)
# -------------------------


class ExtractedPharma(BaseModel):
    """Validated pharma company entity for output."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    candidate_id: uuid.UUID

    doc_id: str
    schema_version: str = "1.0.0"

    # Company identification
    matched_text: str
    canonical_name: str
    full_name: Optional[str] = None

    # Company metadata
    headquarters: Optional[str] = None
    parent_company: Optional[str] = None
    subsidiaries: List[str] = Field(default_factory=list)

    # Evidence
    primary_evidence: EvidenceSpan
    supporting_evidence: List[EvidenceSpan] = Field(default_factory=list)

    # Verdict
    status: ValidationStatus
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    validation_flags: List[str] = Field(default_factory=list)

    # Audit trail
    provenance: PharmaProvenanceMetadata

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Output schema for JSON export
# -------------------------


class PharmaExportEntry(BaseModel):
    """Simplified pharma company entry for JSON export."""

    matched_text: str
    canonical_name: str
    full_name: Optional[str] = None
    headquarters: Optional[str] = None
    parent_company: Optional[str] = None
    subsidiaries: List[str] = Field(default_factory=list)
    confidence: float
    context: Optional[str] = None
    page: Optional[int] = None
    lexicon_source: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class PharmaExportDocument(BaseModel):
    """Complete pharma company extraction output for a document."""

    run_id: str
    timestamp: str  # ISO format
    document: str
    document_path: Optional[str] = None
    pipeline_version: str

    # Counts
    total_detected: int
    unique_companies: int

    # Results
    companies: List[PharmaExportEntry]

    model_config = ConfigDict(extra="forbid")
