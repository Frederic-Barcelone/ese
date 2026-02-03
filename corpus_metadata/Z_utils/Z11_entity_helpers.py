# corpus_metadata/Z_utils/Z11_entity_helpers.py
"""
Shared entity creation helpers for pipeline components.

Provides common functions for creating ExtractedEntity objects from
candidates and text search matches. Used by both AbbreviationPipeline
and EntityProcessor to avoid code duplication.

Key Components:
    - create_entity_from_candidate: Convert Candidate to ExtractedEntity
    - create_entity_from_search: Create ExtractedEntity from regex match

Example:
    >>> from Z_utils.Z11_entity_helpers import create_entity_from_candidate
    >>> entity = create_entity_from_candidate(
    ...     candidate, ValidationStatus.VALIDATED, 0.9,
    ...     "Auto-approved", ["auto_approved"], {"auto": "stats"},
    ... )

Dependencies:
    - A_core.A01_domain_models: Candidate, ExtractedEntity, ValidationStatus
    - A_core.A03_provenance: hash_string
    - Z_utils.Z02_text_helpers: extract_context_snippet
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from A_core.A03_provenance import hash_string
from Z_utils.Z02_text_helpers import extract_context_snippet

if TYPE_CHECKING:
    from A_core.A01_domain_models import (
        Candidate,
        ExtractedEntity,
        FieldType,
        ValidationStatus,
    )


def create_entity_from_candidate(
    candidate: "Candidate",
    status: "ValidationStatus",
    confidence: float,
    reason: str,
    flags: List[str],
    raw_response: Dict[str, Any],
    long_form_override: Optional[str] = None,
) -> "ExtractedEntity":
    """
    Create ExtractedEntity from a Candidate.

    Used for auto-approve/reject decisions where no LLM validation is needed.

    Args:
        candidate: Source candidate with detection metadata
        status: Validation status (VALIDATED, REJECTED, AMBIGUOUS)
        confidence: Confidence score (0.0-1.0)
        reason: Reason for the decision (stored in rejection_reason if rejected)
        flags: Validation flags to attach
        raw_response: Raw response dict for audit trail
        long_form_override: Override the long form from candidate

    Returns:
        ExtractedEntity with populated evidence and provenance
    """
    from A_core.A01_domain_models import EvidenceSpan, ExtractedEntity, ValidationStatus

    context = (candidate.context_text or "").strip()
    ctx_hash = hash_string(context) if context else "no_context"
    primary = EvidenceSpan(
        text=context,
        location=candidate.context_location,
        scope_ref=ctx_hash,
        start_char_offset=0,
        end_char_offset=len(context),
    )
    return ExtractedEntity(
        candidate_id=candidate.id,
        doc_id=candidate.doc_id,
        field_type=candidate.field_type,
        short_form=candidate.short_form.strip(),
        long_form=long_form_override
        or (candidate.long_form.strip() if candidate.long_form else None),
        primary_evidence=primary,
        supporting_evidence=[],
        status=status,
        confidence_score=confidence,
        rejection_reason=reason if status == ValidationStatus.REJECTED else None,
        validation_flags=flags,
        provenance=candidate.provenance,
        raw_llm_response=raw_response,
    )


def create_entity_from_search(
    doc_id: str,
    full_text: str,
    match: re.Match,
    long_form: Optional[str],
    field_type: "FieldType",
    confidence: float,
    flags: List[str],
    rule_version: str,
    lexicon_source: str,
    pipeline_version: str,
    run_id: str,
) -> "ExtractedEntity":
    """
    Create ExtractedEntity from a text search match.

    Used for PASO C/D heuristics where abbreviations are found by direct
    text search rather than candidate generation.

    Args:
        doc_id: Document identifier
        full_text: Full document text for context extraction
        match: Regex match object with start/end positions
        long_form: Long form expansion (or None)
        field_type: Field type (DEFINITION_PAIR, SHORT_FORM_ONLY, etc.)
        confidence: Confidence score (0.0-1.0)
        flags: Validation flags to attach
        rule_version: Version of the detection rule
        lexicon_source: Source identifier for provenance
        pipeline_version: Pipeline version string
        run_id: Pipeline run identifier

    Returns:
        ExtractedEntity with evidence from surrounding context
    """
    from A_core.A01_domain_models import (
        Coordinate,
        EvidenceSpan,
        ExtractedEntity,
        GeneratorType,
        ProvenanceMetadata,
        ValidationStatus,
    )

    context_snippet = extract_context_snippet(full_text, match.start(), match.end())
    ctx_hash = hash_string(context_snippet)

    primary = EvidenceSpan(
        text=context_snippet,
        location=Coordinate(page_num=1),
        scope_ref=ctx_hash,
        start_char_offset=match.start() - max(0, match.start() - 100),
        end_char_offset=match.end() - max(0, match.start() - 100),
    )

    prov = ProvenanceMetadata(
        pipeline_version=pipeline_version,
        run_id=run_id,
        doc_fingerprint=lexicon_source,
        generator_name=GeneratorType.LEXICON_MATCH,
        rule_version=rule_version,
        lexicon_source=f"orchestrator:{lexicon_source}",
    )

    return ExtractedEntity(
        candidate_id=uuid.uuid4(),
        doc_id=doc_id,
        field_type=field_type,
        short_form=match.group(),
        long_form=long_form,
        primary_evidence=primary,
        supporting_evidence=[],
        status=ValidationStatus.VALIDATED,
        confidence_score=confidence,
        rejection_reason=None,
        validation_flags=flags,
        provenance=prov,
        raw_llm_response={"auto": lexicon_source},
    )


__all__ = ["create_entity_from_candidate", "create_entity_from_search"]
