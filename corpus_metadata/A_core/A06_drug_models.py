# corpus_metadata/corpus_metadata/A_core/A06_drug_models.py
"""
Domain models for drug/chemical entity extraction.

Supports:
- Investigational drugs (compound IDs like LNP023, ALXN1720)
- FDA-approved drugs (brand + generic names)
- General drug terms (RxNorm)
- PubTator enrichment (MeSH IDs)
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
# Drug-specific enums
# -------------------------


class DrugFieldType(str, Enum):
    """How the drug mention was detected."""

    EXACT_MATCH = "EXACT_MATCH"  # Exact string match from lexicon
    PATTERN_MATCH = "PATTERN_MATCH"  # Compound ID pattern (LNP023)
    NER_DETECTION = "NER_DETECTION"  # scispacy/NER detected


class DrugGeneratorType(str, Enum):
    """Tracks which strategy produced the drug candidate."""

    LEXICON_ALEXION = "gen:drug_lexicon_alexion"  # Alexion pipeline drugs
    LEXICON_INVESTIGATIONAL = "gen:drug_lexicon_investigational"  # Clinical trial drugs
    LEXICON_FDA = "gen:drug_lexicon_fda"  # FDA approved drugs
    LEXICON_RXNORM = "gen:drug_lexicon_rxnorm"  # RxNorm general terms
    PATTERN_COMPOUND_ID = "gen:drug_pattern_compound"  # Compound ID regex
    SCISPACY_NER = "gen:drug_scispacy_ner"  # scispacy CHEMICAL detection


class DevelopmentPhase(str, Enum):
    """Drug development phase."""

    PRECLINICAL = "Preclinical"
    PHASE_1 = "Phase 1"
    PHASE_2 = "Phase 2"
    PHASE_3 = "Phase 3"
    APPROVED = "Approved"
    WITHDRAWN = "Withdrawn"
    UNKNOWN = "Unknown"


# -------------------------
# Drug identifiers (codes)
# -------------------------


class DrugIdentifier(BaseModel):
    """Drug code from a standard database."""

    system: str  # "RxCUI", "NDC", "MeSH", "DrugBank", "ChEMBL", "NCT", "UNII"
    code: str  # e.g., "12345", "D000123", "NCT04817618"
    display: Optional[str] = None  # Human-readable name

    model_config = ConfigDict(frozen=True, extra="forbid")


# -------------------------
# Drug provenance (extends base provenance)
# -------------------------


class DrugProvenanceMetadata(BaseProvenanceMetadata):
    """
    Provenance metadata for drug detection.

    Inherits common provenance fields from BaseProvenanceMetadata and adds:
    - generator_name: DrugGeneratorType (overrides base Enum type)
    - lexicon_ids: Drug-specific identifiers (DrugIdentifier)
    """

    generator_name: DrugGeneratorType
    lexicon_ids: Optional[List[DrugIdentifier]] = None


# -------------------------
# DrugCandidate (pre-verification)
# -------------------------


class DrugCandidate(BaseModel):
    """
    Pre-validation drug mention.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    doc_id: str

    # Drug identification
    matched_text: str  # Exact text matched in document
    preferred_name: str  # Generic/canonical drug name
    brand_name: Optional[str] = None  # Brand name (e.g., FABHALTA)
    compound_id: Optional[str] = None  # Development code (e.g., LNP023)

    # Detection metadata
    field_type: DrugFieldType
    generator_type: DrugGeneratorType

    # Drug codes
    identifiers: List[DrugIdentifier] = Field(default_factory=list)

    # Context
    context_text: str
    context_location: Coordinate

    # Drug-specific metadata
    drug_class: Optional[str] = None  # "Factor B inhibitor", "anti-C5 antibody"
    mechanism: Optional[str] = None  # Mechanism of action
    development_phase: Optional[str] = None  # "Phase 3", "Approved"
    is_investigational: bool = False
    sponsor: Optional[str] = None  # Pharmaceutical company
    conditions: List[str] = Field(default_factory=list)  # Target indications
    nct_id: Optional[str] = None  # ClinicalTrials.gov NCT ID

    # FDA metadata (if from FDA lexicon)
    dosage_form: Optional[str] = None  # "TABLET", "INJECTABLE"
    route: Optional[str] = None  # "ORAL", "INJECTION"
    marketing_status: Optional[str] = None  # "Prescription", "Discontinued"

    # Confidence
    initial_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    provenance: DrugProvenanceMetadata

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_drug_candidate(self):
        matched = (self.matched_text or "").strip()
        name = (self.preferred_name or "").strip()

        if not matched:
            raise ValueError("DrugCandidate.matched_text must be non-empty.")
        if not name:
            raise ValueError("DrugCandidate.preferred_name must be non-empty.")

        return self


# -------------------------
# ExtractedDrug (post-verification)
# -------------------------


class ExtractedDrug(BaseModel):
    """
    Validated drug entity for output.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    candidate_id: uuid.UUID

    doc_id: str
    schema_version: str = "1.0.0"

    # Drug identification
    matched_text: str
    preferred_name: str  # Generic name
    brand_name: Optional[str] = None
    compound_id: Optional[str] = None

    # All drug codes (full list)
    identifiers: List[DrugIdentifier] = Field(default_factory=list)

    # Primary codes (convenience accessors)
    rxcui: Optional[str] = None
    mesh_id: Optional[str] = None
    ndc_code: Optional[str] = None
    drugbank_id: Optional[str] = None
    unii: Optional[str] = None  # FDA Unique Ingredient Identifier

    # Evidence
    primary_evidence: EvidenceSpan
    supporting_evidence: List[EvidenceSpan] = Field(default_factory=list)

    # Mention frequency (populated during deduplication)
    mention_count: int = Field(default=1, ge=1, description="Number of times this drug appears in the document")
    pages_mentioned: List[int] = Field(default_factory=list, description="List of page numbers where drug appears")

    # Verdict
    status: ValidationStatus
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    rejection_reason: Optional[str] = None
    validation_flags: List[str] = Field(default_factory=list)

    # Drug-specific metadata
    drug_class: Optional[str] = None
    mechanism: Optional[str] = None
    development_phase: Optional[str] = None
    is_investigational: bool = False
    sponsor: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)
    nct_id: Optional[str] = None

    # FDA metadata
    dosage_form: Optional[str] = None
    route: Optional[str] = None
    marketing_status: Optional[str] = None

    # PubTator enrichment fields
    mesh_aliases: List[str] = Field(default_factory=list)
    pubtator_normalized_name: Optional[str] = None
    enrichment_source: Optional[str] = None

    # Audit trail
    provenance: DrugProvenanceMetadata
    raw_llm_response: Optional[Union[Dict[str, Any], str]] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_extracted_drug(self):
        matched = (self.matched_text or "").strip()
        name = (self.preferred_name or "").strip()

        if not matched:
            raise ValueError("ExtractedDrug.matched_text must be non-empty.")
        if not name:
            raise ValueError("ExtractedDrug.preferred_name must be non-empty.")

        return self


# -------------------------
# Output schema for JSON export
# -------------------------


class DrugExportEntry(BaseModel):
    """
    Simplified drug entry for JSON export.
    """

    matched_text: str
    preferred_name: str
    brand_name: Optional[str] = None
    compound_id: Optional[str] = None
    confidence: float
    is_investigational: bool

    # Drug metadata
    drug_class: Optional[str] = None
    mechanism: Optional[str] = None
    development_phase: Optional[str] = None
    sponsor: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)
    nct_id: Optional[str] = None

    # FDA metadata
    dosage_form: Optional[str] = None
    route: Optional[str] = None
    marketing_status: Optional[str] = None

    # Drug codes (flat structure)
    codes: Dict[str, Optional[str]]  # {"rxcui": "12345", "mesh": "D000123", ...}

    # Full identifiers list
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


class DrugExportDocument(BaseModel):
    """
    Complete drug extraction output for a document.
    """

    run_id: str
    timestamp: str  # ISO format
    document: str
    document_path: Optional[str] = None
    pipeline_version: str

    # Counts
    total_candidates: int
    total_validated: int
    total_rejected: int
    total_investigational: int

    # Results
    drugs: List[DrugExportEntry]

    model_config = ConfigDict(extra="forbid")
