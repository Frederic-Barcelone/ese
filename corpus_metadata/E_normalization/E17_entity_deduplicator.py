"""
Entity deduplicator for diseases, drugs, and genes.

This module merges duplicate entities by canonical identifier after normalization,
tracking mention frequency and page distribution. Uses a generic implementation
supporting any entity type with configurable key and scoring functions.

Key Components:
    - _deduplicate_entities: Generic deduplication function
    - deduplicate_diseases: Disease-specific deduplication
    - deduplicate_drugs: Drug-specific deduplication
    - deduplicate_genes: Gene-specific deduplication
    - Features:
        - Canonical ID-based grouping
        - Mention count tracking
        - Pages mentioned aggregation
        - Score-based best entity selection

Example:
    >>> from E_normalization.E17_entity_deduplicator import deduplicate_diseases
    >>> deduplicated = deduplicate_diseases(disease_entities)
    >>> for entity in deduplicated:
    ...     print(f"{entity.canonical_name}: {entity.mention_count} mentions")
    pulmonary arterial hypertension: 15 mentions

Dependencies:
    - A_core.A01_domain_models: ValidationStatus
    - collections: defaultdict for grouping
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Protocol, Set, TypeVar, runtime_checkable

from A_core.A01_domain_models import ValidationStatus


@runtime_checkable
class DeduplicableEntity(Protocol):
    """Protocol describing the attributes needed for entity deduplication."""

    status: str
    id: str
    primary_evidence: Any
    supporting_evidence: Any
    validation_flags: Any

    def model_copy(self, **kwargs: Any) -> DeduplicableEntity: ...


T = TypeVar("T", bound=DeduplicableEntity)


def _deduplicate_entities(
    entities: List[T],
    get_key: Callable[[T], str],
    get_score: Callable[[T], tuple],
) -> List[T]:
    """
    Generic entity deduplication with mention tracking.

    Args:
        entities: List of entities to deduplicate
        get_key: Function to extract grouping key from entity
        get_score: Function to score entity for selection (higher = better)

    Returns:
        Deduplicated list with mention_count and pages_mentioned populated
    """
    if not entities:
        return []

    # Split validated vs non-validated
    validated = [e for e in entities if e.status == ValidationStatus.VALIDATED]
    non_validated = [e for e in entities if e.status != ValidationStatus.VALIDATED]

    # Group by key
    groups: Dict[str, List[T]] = defaultdict(list)
    for entity in validated:
        groups[get_key(entity)].append(entity)

    # Merge each group
    deduped = []
    for group in groups.values():
        merged = _merge_group(group, get_score)
        deduped.append(merged)

    return deduped + non_validated


def _merge_group(group: List[T], get_score: Callable[[T], tuple]) -> T:
    """Merge entity group into one with mention tracking."""
    # Collect all pages
    pages: Set[int] = set()
    for entity in group:
        if entity.primary_evidence:
            pages.add(entity.primary_evidence.location.page_num)
        for ev in getattr(entity, "supporting_evidence", []) or []:
            pages.add(ev.location.page_num)

    # Select best entity
    best = max(group, key=get_score)

    # Collect supporting evidence from merged entities
    all_supporting = list(getattr(best, "supporting_evidence", []) or [])
    for entity in group:
        if entity.id != best.id and entity.primary_evidence:
            all_supporting.append(entity.primary_evidence)

    # Update with mention tracking
    flags = list(getattr(best, "validation_flags", []) or [])
    if "deduplicated" not in flags:
        flags.append("deduplicated")

    return best.model_copy(update={  # type: ignore[return-value]
        "mention_count": len(group),
        "pages_mentioned": sorted(pages),
        "supporting_evidence": all_supporting,
        "validation_flags": flags,
    })


class EntityDeduplicator:
    """Deduplicates entities by canonical identifier and tracks mention frequency."""

    def deduplicate_diseases(self, diseases: List) -> List:
        """Deduplicate diseases by MONDO > ORPHA > UMLS > MeSH > text, then merge hierarchical."""
        def get_key(d):
            if d.mondo_id:
                return f"mondo:{d.mondo_id}"
            if d.orpha_code:
                return f"orpha:{d.orpha_code}"
            if d.umls_cui:
                return f"umls:{d.umls_cui}"
            if d.mesh_id:
                return f"mesh:{d.mesh_id}"
            return f"text:{d.preferred_label.lower().strip()}"

        def get_score(d):
            return (d.confidence_score, len(d.identifiers), bool(d.mondo_id), bool(d.orpha_code))

        deduped = _deduplicate_entities(diseases, get_key, get_score)
        return self._merge_hierarchical_diseases(deduped, get_score)

    @staticmethod
    def _merge_hierarchical_diseases(diseases: List, get_score) -> List:
        """Merge diseases where one preferred_label is a substring of another.

        Only merges when the shorter (more generic) term shares the same ontology
        ID as the longer term, or has no ontology ID at all. Keeps the more
        specific entity and aggregates mention counts.
        """
        if len(diseases) <= 1:
            return diseases

        validated = [d for d in diseases if d.status == ValidationStatus.VALIDATED]
        non_validated = [d for d in diseases if d.status != ValidationStatus.VALIDATED]

        if len(validated) <= 1:
            return diseases

        # Sort by label length descending so we process specific → generic
        sorted_diseases = sorted(
            validated,
            key=lambda d: len(d.preferred_label),
            reverse=True,
        )

        merged_into: set[str] = set()  # ids of entities absorbed
        result = []

        for i, disease in enumerate(sorted_diseases):
            if str(disease.id) in merged_into:
                continue

            label_lower = disease.preferred_label.lower().strip()
            absorbed_mentions = 0

            for j in range(i + 1, len(sorted_diseases)):
                other = sorted_diseases[j]
                if str(other.id) in merged_into:
                    continue

                other_label = other.preferred_label.lower().strip()

                # Check if other is a substring of disease
                if other_label not in label_lower:
                    continue

                # Only merge if same ontology ID or other has no ID
                same_id = False
                other_has_no_id = not (other.mondo_id or other.orpha_code or other.umls_cui or other.mesh_id)

                if other_has_no_id:
                    same_id = True
                elif disease.mondo_id and other.mondo_id and disease.mondo_id == other.mondo_id:
                    same_id = True
                elif disease.orpha_code and other.orpha_code and disease.orpha_code == other.orpha_code:
                    same_id = True
                elif disease.umls_cui and other.umls_cui and disease.umls_cui == other.umls_cui:
                    same_id = True

                if same_id:
                    merged_into.add(str(other.id))
                    absorbed_mentions += getattr(other, "mention_count", 1) or 1

            # Update mention count if we absorbed anything
            if absorbed_mentions > 0:
                current_mentions = getattr(disease, "mention_count", 1) or 1
                flags = list(getattr(disease, "validation_flags", []) or [])
                if "hierarchical_merge" not in flags:
                    flags.append("hierarchical_merge")
                disease = disease.model_copy(update={
                    "mention_count": current_mentions + absorbed_mentions,
                    "validation_flags": flags,
                })

            result.append(disease)

        return result + non_validated

    def deduplicate_drugs(self, drugs: List) -> List:
        """Deduplicate drugs by RxCUI > DrugBank > compound_id > name."""
        def get_key(d):
            if d.rxcui:
                return f"rxcui:{d.rxcui}"
            if d.drugbank_id:
                return f"drugbank:{d.drugbank_id}"
            if d.compound_id:
                return f"compound:{d.compound_id.upper()}"
            return f"text:{d.preferred_name.lower().strip()}"

        def get_score(d):
            return (d.confidence_score, d.is_investigational, len(d.identifiers), bool(d.rxcui))

        return _deduplicate_entities(drugs, get_key, get_score)

    def deduplicate_genes(self, genes: List) -> List:
        """Deduplicate genes by HGNC ID > Entrez > symbol."""
        def get_key(g):
            if g.hgnc_id:
                return f"hgnc:{g.hgnc_id}"
            if g.entrez_id:
                return f"entrez:{g.entrez_id}"
            return f"symbol:{g.hgnc_symbol.upper()}"

        def get_score(g):
            return (not g.is_alias, g.confidence_score, len(g.identifiers), len(g.associated_diseases or []))

        return _deduplicate_entities(genes, get_key, get_score)


__all__ = ["EntityDeduplicator"]
