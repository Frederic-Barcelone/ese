# corpus_metadata/A_core/A14_extraction_result.py
"""
Universal extraction output contract for deterministic pipeline results.

This module defines the canonical output format that all extraction strategies
must emit. The design ensures determinism for regression testing: IDs are computed
from content hashes (not UUIDs), timestamps are excluded from hashing, and all
collections use immutable tuples. Use these types when building generators or
consuming extraction results.

Key Components:
    - EntityType: Enum of all extractable entity types (ABBREVIATION, DISEASE, etc.)
    - Provenance: Immutable location and audit trail (page, bbox, strategy, version)
    - ExtractionResult: Universal output with value, confidence, evidence, and status
    - compute_result_id: Deterministic SHA256-based ID generation
    - compute_regression_hash: Stable hash for regression test comparison
    - to_canonical_dict: Convert result to hashable dictionary format

Example:
    >>> from A_core.A14_extraction_result import ExtractionResult, EntityType, Provenance
    >>> provenance = Provenance(page_num=1, strategy_id="lexicon_disease")
    >>> result = ExtractionResult(
    ...     doc_id="doc123",
    ...     entity_type=EntityType.DISEASE,
    ...     field_name="disease",
    ...     value="IgA nephropathy",
    ...     provenance=provenance
    ... )
    >>> result.id  # Deterministic hash-based ID

Dependencies:
    - None (standalone module using only stdlib dataclasses, hashlib, json)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, List, Optional, Tuple


class EntityType(str, Enum):
    """All extractable entity types."""

    ABBREVIATION = "abbreviation"
    DISEASE = "disease"
    DRUG = "drug"
    GENE = "gene"
    PHARMA = "pharma_company"
    AUTHOR = "author"
    CITATION = "citation"
    FEASIBILITY = "feasibility"
    METADATA = "metadata"


@dataclass(frozen=True)
class Provenance:
    """
    Immutable provenance - location + audit trail.

    Required fields have no defaults. Optional fields follow.
    """

    # Required fields (no defaults)
    page_num: int
    strategy_id: str

    # Optional location fields
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    node_ids: Tuple[str, ...] = ()  # Immutable tuple of block_id, table_id, etc.
    char_span: Optional[Tuple[int, int]] = None  # (start, end) within node

    # Audit trail
    strategy_version: str = "1.0.0"
    doc_fingerprint: str = ""
    lexicon_source: Optional[str] = None

    # Non-deterministic fields (excluded from regression hash)
    pipeline_version: str = ""
    run_id: str = ""
    timestamp: Optional[datetime] = None  # Optional, not auto-generated


@dataclass(frozen=True)
class ExtractionResult:
    """
    Universal output contract for ALL extraction strategies.
    Immutable and deterministic.

    INVARIANT: Confidence is set ONLY by UnifiedConfidenceCalculator, never by strategies.
    """

    # Required fields (no defaults)
    doc_id: str
    entity_type: EntityType
    field_name: str  # e.g., "disease", "drug_name", "abbreviation"
    value: str  # Primary extracted value
    provenance: Provenance  # Required, no default

    # Optional fields
    normalized_value: Optional[str] = None

    # Confidence (set ONLY by UnifiedConfidenceCalculator)
    confidence: float = 0.0
    confidence_features: Tuple[Tuple[str, float], ...] = ()  # Immutable feature breakdown

    # Evidence
    evidence_text: str = ""
    supporting_evidence: Tuple[str, ...] = ()  # Immutable

    # Standard identifiers (ontology codes)
    standard_ids: Tuple[Tuple[str, str], ...] = ()  # e.g., (("ORPHA", "182090"), ("ICD10", "I27.0"))
    extensions: Tuple[Tuple[str, Any], ...] = ()  # Domain-specific extras

    # Validation status
    status: str = "pending"  # "pending", "validated", "rejected"
    rejection_reason: Optional[str] = None

    @property
    def id(self) -> str:
        """Stable hash-based ID for determinism."""
        return compute_result_id(self)

    def with_confidence(
        self,
        confidence: float,
        confidence_features: Tuple[Tuple[str, float], ...],
    ) -> "ExtractionResult":
        """Return a new ExtractionResult with updated confidence (immutable pattern)."""
        return ExtractionResult(
            doc_id=self.doc_id,
            entity_type=self.entity_type,
            field_name=self.field_name,
            value=self.value,
            provenance=self.provenance,
            normalized_value=self.normalized_value,
            confidence=confidence,
            confidence_features=confidence_features,
            evidence_text=self.evidence_text,
            supporting_evidence=self.supporting_evidence,
            standard_ids=self.standard_ids,
            extensions=self.extensions,
            status=self.status,
            rejection_reason=self.rejection_reason,
        )

    def with_status(self, status: str, rejection_reason: Optional[str] = None) -> "ExtractionResult":
        """Return a new ExtractionResult with updated status (immutable pattern)."""
        return ExtractionResult(
            doc_id=self.doc_id,
            entity_type=self.entity_type,
            field_name=self.field_name,
            value=self.value,
            provenance=self.provenance,
            normalized_value=self.normalized_value,
            confidence=self.confidence,
            confidence_features=self.confidence_features,
            evidence_text=self.evidence_text,
            supporting_evidence=self.supporting_evidence,
            standard_ids=self.standard_ids,
            extensions=self.extensions,
            status=status,
            rejection_reason=rejection_reason,
        )


# -----------------------------------------------------------------------------
# Deterministic ID and Hash Utilities
# -----------------------------------------------------------------------------


def _round_bbox(bbox: Optional[Tuple[float, ...]]) -> Optional[Tuple[int, ...]]:
    """Round bbox to integers for stable hashing."""
    if bbox is None:
        return None
    return tuple(int(round(v)) for v in bbox)


def _to_id_dict(result: ExtractionResult) -> dict:
    """Canonical dict for ID computation. JSON-serializable."""
    return {
        "doc_id": result.doc_id,
        "entity_type": result.entity_type.value,
        "field_name": result.field_name,
        "value": result.value,
        "normalized_value": result.normalized_value,
        "page_num": result.provenance.page_num,
        "bbox": _round_bbox(result.provenance.bbox),
        "node_ids": list(result.provenance.node_ids),
        "char_span": list(result.provenance.char_span) if result.provenance.char_span else None,
        "strategy_id": result.provenance.strategy_id,
    }


def compute_result_id(result: ExtractionResult) -> str:
    """
    Compute deterministic ID via JSON serialization.

    Uses SHA256 hash of canonical JSON representation.
    Same content always produces same ID across runs.
    """
    content = json.dumps(_to_id_dict(result), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def to_canonical_dict(result: ExtractionResult) -> dict:
    """
    Convert to canonical dict for regression hashing.
    Excludes non-deterministic fields: run_id, pipeline_version, timestamp.
    """
    return {
        "doc_id": result.doc_id,
        "entity_type": result.entity_type.value,
        "field_name": result.field_name,
        "value": result.value,
        "normalized_value": result.normalized_value,
        "page_num": result.provenance.page_num,
        "bbox": _round_bbox(result.provenance.bbox),
        "node_ids": list(result.provenance.node_ids),
        "char_span": list(result.provenance.char_span) if result.provenance.char_span else None,
        "strategy_id": result.provenance.strategy_id,
        "strategy_version": result.provenance.strategy_version,
        "confidence": round(result.confidence, 4),  # Stable float formatting
        "status": result.status,
    }


def compute_regression_hash(results: List[ExtractionResult]) -> str:
    """
    Compute stable hash for regression testing.

    Same inputs must produce same hash across runs.
    Sorts results by deterministic key before hashing.
    """
    canonical = [to_canonical_dict(r) for r in results]
    # Sort by stable key
    canonical.sort(
        key=lambda d: (
            d["doc_id"],
            d["entity_type"],
            d["field_name"],
            d["value"],
            d["page_num"],
        )
    )
    content = json.dumps(canonical, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
