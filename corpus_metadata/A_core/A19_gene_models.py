# corpus_metadata/A_core/A19_gene_models.py
"""
Domain models for gene and protein entity extraction.

This module provides Pydantic models for the complete gene extraction pipeline,
from candidate generation through validation to export. Use these models when
extracting gene mentions from documents, linking to standard identifiers (HGNC,
Entrez, Ensembl), and tracking gene-disease associations from Orphadata.

Key Components:
    - GeneFieldType: Enum for detection method (EXACT_MATCH, PATTERN_MATCH, NER)
    - GeneGeneratorType: Enum tracking which strategy produced the candidate
    - GeneAssociationType: Enum for Orphadata gene-disease association types
    - GeneIdentifier: Gene code from standard databases (HGNC, Entrez, Ensembl, etc.)
    - GeneDiseaseLinkage: Link between a gene and an associated rare disease
    - GeneCandidate: Pre-validation gene mention with context and identifiers
    - ExtractedGene: Validated gene entity with evidence and confidence scores
    - GeneExportEntry: Simplified gene entry for JSON export
    - GeneExportDocument: Complete gene extraction output for a document

Example:
    >>> from A_core.A19_gene_models import GeneCandidate, GeneIdentifier, GeneFieldType
    >>> identifier = GeneIdentifier(system="HGNC", code="HGNC:1100", display="BRCA1")
    >>> candidate = GeneCandidate(
    ...     doc_id="doc_001", matched_text="BRCA1", hgnc_symbol="BRCA1",
    ...     field_type=GeneFieldType.EXACT_MATCH,
    ...     generator_type=GeneGeneratorType.LEXICON_HGNC_ALIAS,
    ...     context_text="mutations in BRCA1 are associated with...",
    ...     context_location=Coordinate(page=1, start=100, end=105),
    ...     identifiers=[identifier], provenance=...
    ... )

Dependencies:
    - A_core.A01_domain_models: For BaseProvenanceMetadata, Coordinate, EvidenceSpan
    - pydantic: For model validation and serialization
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
# Gene-specific enums
# -------------------------


class GeneFieldType(str, Enum):
    """How the gene mention was detected."""

    EXACT_MATCH = "EXACT_MATCH"  # Exact string match from lexicon
    PATTERN_MATCH = "PATTERN_MATCH"  # Gene symbol pattern (BRCA1, TP53)
    NER_DETECTION = "NER_DETECTION"  # scispacy/NER detected


class GeneGeneratorType(str, Enum):
    """Tracks which strategy produced the gene candidate."""

    LEXICON_ORPHADATA = "gen:gene_lexicon_orphadata"  # Orphadata rare disease genes
    LEXICON_HGNC_ALIAS = "gen:gene_lexicon_hgnc_alias"  # HGNC alias/synonym
    PATTERN_GENE_SYMBOL = "gen:gene_pattern_symbol"  # Gene symbol pattern match
    SCISPACY_NER = "gen:gene_scispacy_ner"  # scispacy GENE detection


class GeneAssociationType(str, Enum):
    """Type of gene-disease association from Orphadata."""

    DISEASE_CAUSING = "Disease-causing germline mutation(s) in"
    DISEASE_CAUSING_SOMATIC = "Disease-causing somatic mutation(s) in"
    MAJOR_SUSCEPTIBILITY = "Major susceptibility factor in"
    MODIFYING = "Modifying germline mutation in"
    ROLE_PATHOGENESIS = "Role in the phenotype of"
    CANDIDATE = "Candidate gene tested in"
    BIOMARKER = "Biomarker tested in"
    UNKNOWN = "Unknown"


# -------------------------
# Gene identifiers (codes)
# -------------------------


class GeneIdentifier(BaseModel):
    """Gene code from a standard database."""

    system: str  # "HGNC", "ENTREZ", "ENSEMBL", "OMIM", "UNIPROT", "ORPHACODE"
    code: str  # e.g., "HGNC:1100", "672", "ENSG00000012048"
    display: Optional[str] = None  # Human-readable name

    model_config = ConfigDict(frozen=True, extra="forbid")


# -------------------------
# Disease association
# -------------------------


class GeneDiseaseLinkage(BaseModel):
    """Link between a gene and an associated rare disease."""

    orphacode: str
    disease_name: str
    association_type: Optional[str] = None
    association_status: Optional[str] = None

    model_config = ConfigDict(frozen=True, extra="forbid")


# -------------------------
# Gene provenance (extends base provenance)
# -------------------------


class GeneProvenanceMetadata(BaseProvenanceMetadata):
    """
    Provenance metadata for gene detection.

    Inherits common provenance fields from BaseProvenanceMetadata and adds:
    - generator_name: GeneGeneratorType (overrides base Enum type)
    - lexicon_ids: Gene-specific identifiers (GeneIdentifier)
    """

    generator_name: GeneGeneratorType
    lexicon_ids: Optional[List[GeneIdentifier]] = None  # Gene codes


# -------------------------
# GeneCandidate (pre-verification)
# -------------------------


class GeneCandidate(BaseModel):
    """Pre-validation gene mention."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    doc_id: str

    # Gene identification
    matched_text: str  # Exact text matched in document
    hgnc_symbol: str  # Official HGNC symbol (canonical)
    full_name: Optional[str] = None  # Full gene name
    is_alias: bool = False  # True if matched via alias/synonym
    alias_of: Optional[str] = None  # If alias, what's the canonical symbol

    # Detection metadata
    field_type: GeneFieldType
    generator_type: GeneGeneratorType

    # Gene codes
    identifiers: List[GeneIdentifier] = Field(default_factory=list)

    # Context
    context_text: str
    context_location: Coordinate

    # Gene-specific metadata
    locus_type: Optional[str] = None  # "protein-coding", "ncRNA", "pseudogene"
    chromosome: Optional[str] = None

    # Associated rare diseases (from Orphadata)
    associated_diseases: List[GeneDiseaseLinkage] = Field(default_factory=list)

    # Confidence
    initial_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    provenance: GeneProvenanceMetadata

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_gene_candidate(self):
        matched = (self.matched_text or "").strip()
        symbol = (self.hgnc_symbol or "").strip()

        if not matched:
            raise ValueError("GeneCandidate.matched_text must be non-empty.")
        if not symbol:
            raise ValueError("GeneCandidate.hgnc_symbol must be non-empty.")

        return self


# -------------------------
# ExtractedGene (post-verification)
# -------------------------


class ExtractedGene(BaseModel):
    """Validated gene entity for output."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    candidate_id: uuid.UUID

    doc_id: str
    schema_version: str = "1.0.0"

    # Gene identification
    matched_text: str
    hgnc_symbol: str  # Official symbol
    full_name: Optional[str] = None
    is_alias: bool = False
    alias_of: Optional[str] = None

    # All gene codes (full list)
    identifiers: List[GeneIdentifier] = Field(default_factory=list)

    # Primary codes (convenience accessors)
    hgnc_id: Optional[str] = None
    entrez_id: Optional[str] = None
    ensembl_id: Optional[str] = None
    omim_id: Optional[str] = None
    uniprot_id: Optional[str] = None

    # Evidence
    primary_evidence: EvidenceSpan
    supporting_evidence: List[EvidenceSpan] = Field(default_factory=list)

    # Mention frequency (populated during deduplication)
    mention_count: int = Field(default=1, ge=1, description="Number of times this gene appears in the document")
    pages_mentioned: List[int] = Field(default_factory=list, description="List of page numbers where gene appears")

    # Verdict
    status: ValidationStatus
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    rejection_reason: Optional[str] = None
    validation_flags: List[str] = Field(default_factory=list)

    # Gene-specific metadata
    locus_type: Optional[str] = None
    chromosome: Optional[str] = None

    # Associated rare diseases
    associated_diseases: List[GeneDiseaseLinkage] = Field(default_factory=list)

    # Audit trail
    provenance: GeneProvenanceMetadata
    raw_llm_response: Optional[Union[Dict[str, Any], str]] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_extracted_gene(self):
        matched = (self.matched_text or "").strip()
        symbol = (self.hgnc_symbol or "").strip()

        if not matched:
            raise ValueError("ExtractedGene.matched_text must be non-empty.")
        if not symbol:
            raise ValueError("ExtractedGene.hgnc_symbol must be non-empty.")

        return self


# -------------------------
# Output schema for JSON export
# -------------------------


class GeneExportEntry(BaseModel):
    """Simplified gene entry for JSON export."""

    matched_text: str
    hgnc_symbol: str
    full_name: Optional[str] = None
    confidence: float
    is_alias: bool = False

    # Gene metadata
    locus_type: Optional[str] = None
    chromosome: Optional[str] = None

    # Gene codes (flat structure)
    codes: Dict[str, Optional[str]]  # {"hgnc_id": "HGNC:1100", "entrez": "672", ...}

    # Full identifiers list
    all_identifiers: List[Dict[str, Optional[str]]]

    # Associated diseases (simplified)
    associated_diseases: List[Dict[str, str]]

    # Context
    context: Optional[str] = None
    page: Optional[int] = None

    # Mention frequency
    mention_count: int = Field(default=1, ge=1)
    pages_mentioned: List[int] = Field(default_factory=list)

    # Provenance
    lexicon_source: Optional[str] = None
    validation_flags: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class GeneExportDocument(BaseModel):
    """Complete gene extraction output for a document."""

    run_id: str
    timestamp: str  # ISO format
    document: str
    document_path: Optional[str] = None
    pipeline_version: str

    # Counts
    total_candidates: int
    total_validated: int
    total_rejected: int

    # Results
    genes: List[GeneExportEntry]

    model_config = ConfigDict(extra="forbid")
