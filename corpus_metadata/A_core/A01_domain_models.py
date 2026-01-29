# corpus_metadata/A_core/A01_domain_models.py
"""
Domain models for the ESE extraction pipeline.

Provides Pydantic models for:
- Pipeline stages and field types (enums)
- Abbreviation categories and validation status
- Candidates, extracted entities, and bounding boxes
- Drug, disease, and other entity-specific models
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# -------------------------
# Pipeline enums
# -------------------------


class PipelineStage(str, Enum):
    GENERATION = "GENERATION"
    VERIFICATION = "VERIFICATION"
    NORMALIZATION = "NORMALIZATION"


class FieldType(str, Enum):
    """
    Distinguishes *how* the abbreviation was presented.
    This drives which verifier rules apply.
    """

    # "Tumor Necrosis Factor (TNF)" or "TNF (Tumor Necrosis Factor)"
    DEFINITION_PAIR = "DEFINITION_PAIR"

    # "AE | Adverse Event" (table/section glossary)
    GLOSSARY_ENTRY = "GLOSSARY_ENTRY"

    # "The patient received TNF..." (no definition in doc)
    SHORT_FORM_ONLY = "SHORT_FORM_ONLY"


class AbbreviationCategory(str, Enum):
    """
    Semantic category for abbreviations.

    Used to distinguish the domain type of an abbreviation without
    conflating it with the entity type (which is always ABBREVIATION).
    This prevents confusing metrics in abbreviations_only mode.
    """

    # General medical/clinical abbreviations
    ABBREV = "ABBREV"

    # Statistical abbreviations (CI, SD, HR, etc.)
    STATISTICAL = "STATISTICAL"

    # Disease-related abbreviations (MS, PD, etc.)
    DISEASE = "DISEASE"

    # Drug-related abbreviations (compound IDs, etc.)
    DRUG = "DRUG"

    # Gene-related abbreviations
    GENE = "GENE"

    # Study/trial-related abbreviations
    STUDY = "STUDY"

    # Organizational abbreviations (FDA, EMA, etc.)
    ORGANIZATION = "ORGANIZATION"

    # Uncategorized
    UNKNOWN = "UNKNOWN"


class GeneratorType(str, Enum):
    """
    Tracks which strategy produced the candidate.
    Useful for Recall-by-Strategy analysis.
    """

    SYNTAX_PATTERN = "gen:syntax_pattern"  # C01: Schwartz-Hearst abbreviations
    SECTION_PARSER = "gen:section_parser"  # (reserved)
    GLOSSARY_TABLE = "gen:glossary_table"  # C01b: Glossary tables
    RIGID_PATTERN = "gen:rigid_pattern"  # C02: DOI, trial IDs, doses, etc.
    TABLE_LAYOUT = "gen:table_layout"  # C03: Spatial extraction
    LEXICON_MATCH = "gen:lexicon_match"  # C04: Dictionary matching
    INLINE_DEFINITION = "gen:inline_definition"  # C04: Explicit inline definitions (SF=LF)


class ValidationStatus(str, Enum):
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    AMBIGUOUS = "AMBIGUOUS"


# -------------------------
# Geometry (shared)
# -------------------------


class BoundingBox(BaseModel):
    """
    Strict BBox used across the pipeline.
    coords = (x0, y0, x1, y1)

    If is_normalized=True: coords are in [0,1]
    If is_normalized=False: coords are absolute units (pixels/points), >= 0
    """

    coords: Tuple[float, float, float, float]

    page_width: Optional[float] = None
    page_height: Optional[float] = None
    is_normalized: bool = False

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _validate_bbox(self):
        x0, y0, x1, y1 = self.coords

        if x0 < 0 or y0 < 0:
            raise ValueError("BoundingBox coordinates cannot be negative.")
        if x1 < x0 or y1 < y0:
            raise ValueError(f"Invalid BoundingBox geometry: ({x0},{y0},{x1},{y1})")

        if self.is_normalized:
            if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
                raise ValueError(
                    "Normalized BoundingBox must have coords in [0.0, 1.0]."
                )

        # Optional soft bounds check (kept permissive)
        if (not self.is_normalized) and self.page_width and self.page_height:
            # Some extractors overshoot slightly; don't hard-fail by default.
            pass

        return self


class Coordinate(BaseModel):
    """
    Minimal but audit-friendly location.
    - page_num is 1-based.
    - bbox is optional (but recommended for highlighting).
    - block_id/table_id/cell indices allow linking back to doc_graph objects.
    """

    page_num: int = Field(..., ge=1, description="1-based page index (human readable)")

    block_id: Optional[str] = None
    table_id: Optional[str] = None
    cell_row: Optional[int] = Field(default=None, ge=0)
    cell_col: Optional[int] = Field(default=None, ge=0)

    bbox: Optional[BoundingBox] = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class EvidenceSpan(BaseModel):
    """
    Proof snippet with offsets scoped to a known text context (scope_ref).
    Offsets are optional-ish in practice, but we keep them required for strictness.
    """

    text: str
    location: Coordinate

    # E.g. hash of context window OR block_id/table_id; must be stable for audit.
    scope_ref: str

    start_char_offset: int = Field(..., ge=0)
    end_char_offset: int = Field(..., ge=0)

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _validate_offsets(self):
        if self.end_char_offset < self.start_char_offset:
            raise ValueError("end_char_offset must be >= start_char_offset")
        return self


# -------------------------
# Provenance (audit trail)
# -------------------------


class LLMParameters(BaseModel):
    model_name: str
    temperature: float
    max_tokens: int
    top_p: float
    seed: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None  # e.g. {"type": "json_object"}

    model_config = ConfigDict(frozen=True, extra="forbid")


class LexiconIdentifier(BaseModel):
    """External identifier from a lexicon source (e.g., Orphanet, MONDO, UMLS)."""

    source: str  # e.g. "Orphanet", "MONDO", "UMLS"
    id: str  # e.g. "ORPHA:2453", "MONDO_0011055", "C0001234"

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProvenanceMetadata(BaseModel):
    """
    Minimal reproducibility + compliance fingerprints.
    """

    pipeline_version: str  # e.g. git commit hash
    run_id: str  # e.g. RUN_20250101_120000_ab12cd34ef56
    doc_fingerprint: str  # SHA256 of source PDF bytes

    generator_name: GeneratorType
    rule_version: Optional[str] = None

    # Lexicon provenance (which dictionary file the match came from)
    lexicon_source: Optional[str] = None  # e.g. "2025_08_abbreviation_general.json"
    lexicon_ids: Optional[List[LexiconIdentifier]] = (
        None  # External IDs (Orphanet, MONDO, etc.)
    )

    # Populated during verification
    prompt_bundle_hash: Optional[str] = None
    context_hash: Optional[str] = None
    llm_config: Optional[LLMParameters] = None

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True, extra="forbid")


# -------------------------
# Candidate (pre-verification)
# -------------------------


class Candidate(BaseModel):
    """
    Noisy pre-LLM object.
    Keep it abbreviation-specific: SF always required; LF optional depending on FieldType.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)

    doc_id: str
    field_type: FieldType
    generator_type: GeneratorType

    short_form: str
    long_form: Optional[str] = None

    # Context sent to verifier (LLM or deterministic)
    context_text: str
    context_location: Coordinate

    initial_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    provenance: ProvenanceMetadata

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_sf_lf(self):
        sf = (self.short_form or "").strip()
        lf = (self.long_form or "").strip() if self.long_form else ""

        if not sf:
            raise ValueError("Candidate.short_form must be non-empty.")

        if self.field_type in (FieldType.DEFINITION_PAIR, FieldType.GLOSSARY_ENTRY):
            if not lf:
                raise ValueError(
                    f"Candidate.long_form is required for field_type={self.field_type}."
                )
        return self


# -------------------------
# ExtractedEntity (post-verification)
# -------------------------


class ExtractedEntity(BaseModel):
    """
    Verified output object. Suitable for export + audit.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    candidate_id: uuid.UUID

    doc_id: str
    schema_version: str = "1.0.0"

    field_type: FieldType

    short_form: str
    long_form: Optional[str] = None

    # Optional post-processing (normalization)
    normalized_value: Optional[Union[str, Dict[str, Any]]] = None
    standard_id: Optional[str] = None

    # Evidence
    primary_evidence: EvidenceSpan
    supporting_evidence: List[EvidenceSpan] = Field(default_factory=list)

    # Verdict
    status: ValidationStatus
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    rejection_reason: Optional[str] = None
    validation_flags: List[str] = Field(default_factory=list)

    # Optional semantic category (for distinguishing abbreviation domains)
    # This is separate from entity type - an abbreviation can have a disease category
    # without being counted as a disease entity in metrics.
    category: Optional[AbbreviationCategory] = None

    # Audit trail
    provenance: ProvenanceMetadata
    raw_llm_response: Optional[Union[Dict[str, Any], str]] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_entity_sf_lf(self):
        sf = (self.short_form or "").strip()
        lf = (self.long_form or "").strip() if self.long_form else ""

        if not sf:
            raise ValueError("ExtractedEntity.short_form must be non-empty.")

        if self.field_type in (FieldType.DEFINITION_PAIR, FieldType.GLOSSARY_ENTRY):
            if self.status == ValidationStatus.VALIDATED and not lf:
                raise ValueError(
                    f"Validated entity requires long_form for field_type={self.field_type}."
                )
        return self
