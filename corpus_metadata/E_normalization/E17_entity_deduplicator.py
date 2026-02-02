# corpus_metadata/E_normalization/E17_entity_deduplicator.py
"""
Entity deduplicator for diseases, drugs, and genes.

Merges duplicate entities by canonical identifier and tracks mention frequency.
Runs AFTER normalization to ensure proper ID-based grouping.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Set, TypeVar

from A_core.A01_domain_models import ValidationStatus

T = TypeVar("T")


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

    return best.model_copy(update={
        "mention_count": len(group),
        "pages_mentioned": sorted(pages),
        "supporting_evidence": all_supporting,
        "validation_flags": flags,
    })


class EntityDeduplicator:
    """Deduplicates entities by canonical identifier and tracks mention frequency."""

    def deduplicate_diseases(self, diseases: List) -> List:
        """Deduplicate diseases by MONDO > ORPHA > UMLS > MeSH > text."""
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

        return _deduplicate_entities(diseases, get_key, get_score)

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
