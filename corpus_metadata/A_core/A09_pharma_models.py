# corpus_metadata/A_core/A09_pharma_models.py
"""
Domain models for pharmaceutical company entity extraction.

Simple lexicon-based detection of pharma company mentions.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from A_core.A01_domain_models import Coordinate, EvidenceSpan, ValidationStatus


# -------------------------
# Pharma-specific enums
# -------------------------


class PharmaGeneratorType(str, Enum):
    """Tracks which strategy produced the pharma company candidate."""

    LEXICON_MATCH = "gen:pharma_lexicon"  # Matched from pharma_companies_lexicon.json


# -------------------------
# Pharma provenance
# -------------------------


class PharmaProvenanceMetadata(BaseModel):
    """Provenance metadata for pharma company detection."""

    pipeline_version: str
    run_id: str
    doc_fingerprint: str
    generator_name: PharmaGeneratorType
    lexicon_source: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True, extra="forbid")


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
