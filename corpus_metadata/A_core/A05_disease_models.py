# corpus_metadata/corpus_metadata/A_core/A05_disease_models.py
"""
Domain models for disease mention detection.
Separate from abbreviation models (A01) as diseases have fundamentally different semantics:
- No short_form/long_form relationship
- Multiple code systems (ICD-10, ICD-11, SNOMED-CT, MONDO, ORPHA)
- Disease-specific fields (is_rare_disease, category, parent_disease)
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from A_core.A01_domain_models import (
    BaseProvenanceMetadata,
    Coordinate,
    EvidenceSpan,
    ValidationStatus,
)


# -------------------------
# Disease-specific enums
# -------------------------


class DiseaseFieldType(str, Enum):
    """How the disease mention was detected."""

    EXACT_MATCH = "EXACT_MATCH"  # Exact string match from lexicon
    PATTERN_MATCH = "PATTERN_MATCH"  # Regex pattern match (e.g., "type 2 diabetes")
    NER_DETECTION = "NER_DETECTION"  # scispacy/NER detected
    ABBREVIATION_EXPAND = "ABBREV_EXPAND"  # Expanded from disease abbreviation


class DiseaseGeneratorType(str, Enum):
    """Tracks which strategy produced the disease candidate."""

    LEXICON_SPECIALIZED = "gen:disease_lexicon_specialized"  # PAH, ANCA, IgAN lexicons
    LEXICON_GENERAL = "gen:disease_lexicon_general"  # General disease lexicon
    LEXICON_ORPHANET = "gen:disease_lexicon_orphanet"  # Orphanet rare diseases
    SCISPACY_NER = "gen:disease_scispacy_ner"  # scispacy entity recognition


# -------------------------
# Disease identifiers (medical codes)
# -------------------------


class DiseaseIdentifier(BaseModel):
    """Medical code from a standard ontology."""

    system: str  # "ICD-10", "ICD-11", "ICD-10-CM", "SNOMED-CT", "MONDO", "ORPHA", "MeSH", "UMLS"
    code: str  # e.g., "I27.0", "11399002", "MONDO_0011055", "ORPHA:182090"
    display: Optional[str] = None  # Human-readable name from the ontology

    model_config = ConfigDict(frozen=True, extra="forbid")


# -------------------------
# Disease provenance (extends base provenance)
# -------------------------


class DiseaseProvenanceMetadata(BaseProvenanceMetadata):
    """
    Provenance metadata for disease detection.

    Inherits common provenance fields from BaseProvenanceMetadata and adds:
    - generator_name: DiseaseGeneratorType (overrides base Enum type)
    - lexicon_ids: Disease-specific medical codes (DiseaseIdentifier)
    """

    generator_name: DiseaseGeneratorType
    lexicon_ids: Optional[List[DiseaseIdentifier]] = None  # Medical codes


# -------------------------
# DiseaseCandidate (pre-verification)
# -------------------------


class DiseaseCandidate(BaseModel):
    """
    Pre-validation disease mention.
    Unlike Candidate (abbreviations), diseases have:
    - matched_text + preferred_label (not short_form/long_form)
    - Multiple medical code identifiers
    - Disease-specific flags (is_rare_disease, category)
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)

    doc_id: str

    # Disease info
    matched_text: str  # Exact text matched in document
    preferred_label: str  # Canonical disease name from lexicon
    abbreviation: Optional[str] = None  # Disease abbreviation if any (PAH, IgAN)
    synonyms: List[str] = Field(default_factory=list)  # Known synonyms

    # Detection metadata
    field_type: DiseaseFieldType
    generator_type: DiseaseGeneratorType

    # Medical codes
    identifiers: List[DiseaseIdentifier] = Field(default_factory=list)

    # Context
    context_text: str
    context_location: Coordinate

    # Disease-specific flags
    is_rare_disease: bool = False
    prevalence: Optional[str] = None  # e.g., "<1/1000000"
    parent_disease: Optional[str] = None  # e.g., "vasculitis" for GPA
    disease_category: Optional[str] = None  # e.g., "nephrology", "pulmonology"

    # Confidence
    initial_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_boost: float = Field(
        default=0.0, ge=0.0, le=0.5
    )  # Boost from context keywords

    provenance: DiseaseProvenanceMetadata

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_disease_candidate(self):
        matched = (self.matched_text or "").strip()
        label = (self.preferred_label or "").strip()

        if not matched:
            raise ValueError("DiseaseCandidate.matched_text must be non-empty.")
        if not label:
            raise ValueError("DiseaseCandidate.preferred_label must be non-empty.")

        return self


# -------------------------
# ExtractedDisease (post-verification)
# -------------------------


class ExtractedDisease(BaseModel):
    """
    Validated disease entity for output.
    Contains all medical codes and disease-specific metadata.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    candidate_id: uuid.UUID

    doc_id: str
    schema_version: str = "1.0.0"

    # Disease info
    matched_text: str
    preferred_label: str
    abbreviation: Optional[str] = None

    # All medical codes (full list)
    identifiers: List[DiseaseIdentifier] = Field(default_factory=list)

    # Primary code for each system (convenience accessors)
    icd10_code: Optional[str] = None
    icd11_code: Optional[str] = None
    snomed_code: Optional[str] = None
    mondo_id: Optional[str] = None
    orpha_code: Optional[str] = None
    umls_cui: Optional[str] = None
    mesh_id: Optional[str] = None

    # Evidence
    primary_evidence: EvidenceSpan
    supporting_evidence: List[EvidenceSpan] = Field(default_factory=list)

    # Mention frequency (populated during deduplication)
    mention_count: int = Field(default=1, ge=1, description="Number of times this disease appears in the document")
    pages_mentioned: List[int] = Field(default_factory=list, description="List of page numbers where disease appears")

    # Verdict
    status: ValidationStatus
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    rejection_reason: Optional[str] = None
    validation_flags: List[str] = Field(default_factory=list)

    # Disease-specific metadata
    is_rare_disease: bool = False
    disease_category: Optional[str] = None  # e.g., "nephrology", "pulmonology"

    # PubTator enrichment fields
    mesh_aliases: List[str] = Field(default_factory=list)
    pubtator_normalized_name: Optional[str] = None
    enrichment_source: Optional[str] = None

    # Audit trail
    provenance: DiseaseProvenanceMetadata
    raw_llm_response: Optional[Union[Dict[str, Any], str]] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_extracted_disease(self):
        matched = (self.matched_text or "").strip()
        label = (self.preferred_label or "").strip()

        if not matched:
            raise ValueError("ExtractedDisease.matched_text must be non-empty.")
        if not label:
            raise ValueError("ExtractedDisease.preferred_label must be non-empty.")

        return self


# -------------------------
# Output schema for JSON export
# -------------------------


class DiseaseExportEntry(BaseModel):
    """
    Simplified disease entry for JSON export.
    Optimized for downstream consumption (no internal UUIDs).
    """

    matched_text: str
    preferred_label: str
    abbreviation: Optional[str] = None
    confidence: float
    is_rare_disease: bool
    category: Optional[str] = None

    # Medical codes (flat structure for easy access)
    codes: Dict[str, Optional[str]]  # {"icd10": "I27.0", "snomed": "11399002", ...}

    # Full identifiers list for completeness
    all_identifiers: List[Dict[str, Optional[str]]]

    # Context
    context: Optional[str] = None
    page: Optional[int] = None

    # Mention frequency
    mention_count: int = Field(default=1, ge=1)
    pages_mentioned: List[int] = Field(default_factory=list)

    # Provenance
    lexicon_source: Optional[str] = None
    validation_flags: List[str] = Field(default_factory=list)

    # PubTator enrichment
    mesh_aliases: List[str] = Field(default_factory=list)
    pubtator_normalized_name: Optional[str] = None
    enrichment_source: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class DiseaseExportDocument(BaseModel):
    """
    Complete disease extraction output for a document.
    """

    run_id: str
    timestamp: str  # ISO format
    document: str  # Filename
    document_path: Optional[str] = None  # Full path
    pipeline_version: str

    # Counts
    total_candidates: int
    total_validated: int
    total_rejected: int
    total_ambiguous: int

    # Results
    diseases: List[DiseaseExportEntry]

    model_config = ConfigDict(extra="forbid")
