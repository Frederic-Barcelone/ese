# corpus_metadata/H_pipeline/H04_merge_resolver.py
"""
Deterministic Merge & Resolve Layer.

Handles deduplication, conflict resolution, and evidence selection
for multiple strategies extracting the same field.

INVARIANT: Same inputs always produce same outputs (deterministic).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from A_core.A02_interfaces import RawExtraction


@dataclass
class MergeConfig:
    """
    Configuration for merge/resolve behavior.

    All configuration is immutable and deterministic.
    """

    # Deduplication key fields
    # Extractions with the same values for these fields are considered duplicates
    dedupe_key_fields: Tuple[str, ...] = ("doc_id", "field_name", "value", "normalized_value")

    # Conflict resolution priority (higher = preferred)
    # Strategies with higher priority win when resolving duplicates
    strategy_priority: Dict[str, int] = field(default_factory=dict)
    # e.g., {"disease_lexicon_specialized": 10, "disease_lexicon_orphanet": 5}

    # Mutual exclusivity constraints
    # If multiple fields in a group are present, keep only one
    mutually_exclusive_fields: Tuple[Tuple[str, ...], ...] = ()
    # e.g., (("is_rare_disease_true", "is_rare_disease_false"),)

    # Canonical form rules
    prefer_table_evidence: bool = True  # Table extractions preferred over text
    prefer_longer_evidence: bool = True  # Longer evidence text preferred

    @staticmethod
    def default() -> "MergeConfig":
        """Create default merge configuration."""
        return MergeConfig(
            strategy_priority={
                # Disease strategies (specialized > orphanet > general)
                "disease_lexicon_specialized": 10,
                "disease_lexicon_orphanet": 5,
                "disease_lexicon_general": 1,
                # Abbreviation strategies (syntax > glossary > lexicon)
                "abbreviation_syntax": 10,
                "abbreviation_glossary": 8,
                "abbreviation_lexicon": 5,
                # Drug strategies
                "drug_lexicon_specialized": 10,
                "drug_lexicon_general": 5,
                # Gene strategies
                "gene_lexicon_hgnc": 10,
                "gene_lexicon_general": 5,
            }
        )


class MergeResolver:
    """
    Deterministic merge & resolve layer.

    Takes outputs from multiple extraction strategies and produces
    a deduplicated, conflict-resolved list of extractions.

    INVARIANT: Same inputs always produce same outputs.

    Usage:
        >>> config = MergeConfig.default()
        >>> resolver = MergeResolver(config)
        >>> merged = resolver.merge(raw_extractions)
    """

    def __init__(self, config: MergeConfig = None):
        """
        Initialize the merge resolver.

        Args:
            config: Merge configuration. Uses default if None.
        """
        self.config = config or MergeConfig.default()

    def merge(self, raw_extractions: List[RawExtraction]) -> List[RawExtraction]:
        """
        Deduplicate, resolve conflicts, enforce constraints, select best evidence.

        All operations are deterministic:
        - Inputs are sorted by stable key before processing
        - Tie-breaking uses deterministic sort order
        - Outputs are sorted by stable key

        Args:
            raw_extractions: List of RawExtraction from all strategies.

        Returns:
            Merged and deduplicated list of RawExtraction.
        """
        if not raw_extractions:
            return []

        # Step 1: Sort inputs for determinism
        sorted_raw = sorted(raw_extractions, key=self._sort_key)

        # Step 2: Group by dedupe key
        groups = self._group_by_key(sorted_raw)

        # Step 3: Resolve each group to single result
        resolved = []
        for key in sorted(groups.keys()):  # Sort keys for determinism
            candidates = groups[key]
            winner = self._resolve_group(candidates)
            resolved.append(winner)

        # Step 4: Enforce mutual exclusivity constraints
        resolved = self._enforce_constraints(resolved)

        # Step 5: Sort output for determinism
        return sorted(resolved, key=self._sort_key)

    def _sort_key(self, r: RawExtraction) -> tuple:
        """
        Deterministic sort key for extractions.

        Returns tuple of all relevant fields for stable ordering.
        """
        return (
            r.doc_id,
            r.entity_type.value,
            r.field_name,
            r.value,
            r.normalized_value or "",
            r.page_num,
            r.strategy_id,
        )

    def _group_by_key(
        self, extractions: List[RawExtraction]
    ) -> Dict[tuple, List[RawExtraction]]:
        """
        Group extractions by dedupe key fields.

        Args:
            extractions: Sorted list of extractions.

        Returns:
            Dict mapping dedupe key to list of extractions.
        """
        groups: Dict[tuple, List[RawExtraction]] = defaultdict(list)
        for r in extractions:
            key = tuple(
                getattr(r, f) if f != "normalized_value" else (getattr(r, f) or "")
                for f in self.config.dedupe_key_fields
            )
            groups[key].append(r)
        return dict(groups)

    def _resolve_group(self, candidates: List[RawExtraction]) -> RawExtraction:
        """
        Resolve a group of duplicate extractions to a single winner.

        Resolution priority (in order):
        1. Strategy priority (from config)
        2. Table evidence preference
        3. Longer evidence text
        4. First in sort order (deterministic tie-break)

        Args:
            candidates: List of duplicate extractions (same dedupe key).

        Returns:
            Single winning extraction with merged supporting evidence.
        """
        if len(candidates) == 1:
            return candidates[0]

        def score(r: RawExtraction) -> tuple:
            priority = self.config.strategy_priority.get(r.strategy_id, 0)
            table_bonus = 1 if (self.config.prefer_table_evidence and r.from_table) else 0
            evidence_len = len(r.evidence_text) if self.config.prefer_longer_evidence else 0
            return (priority, table_bonus, evidence_len)

        # Sort by score descending, then by sort_key for deterministic tie-break
        # We want highest score first, but for tie-breaking we want lowest sort_key
        scored = sorted(
            candidates,
            key=lambda r: (score(r), tuple(-ord(c) for c in str(self._sort_key(r)))),
            reverse=True,
        )
        winner = scored[0]

        # Merge supporting evidence from all candidates
        all_evidence: set = set()
        for c in candidates:
            if c.evidence_text:
                all_evidence.add(c.evidence_text)
            all_evidence.update(c.supporting_evidence)
        # Remove winner's primary evidence from supporting (it's already primary)
        all_evidence.discard(winner.evidence_text)

        # Create new RawExtraction with merged supporting evidence
        return RawExtraction(
            doc_id=winner.doc_id,
            entity_type=winner.entity_type,
            field_name=winner.field_name,
            value=winner.value,
            page_num=winner.page_num,
            strategy_id=winner.strategy_id,
            normalized_value=winner.normalized_value,
            bbox=winner.bbox,
            node_ids=winner.node_ids,
            char_span=winner.char_span,
            strategy_version=winner.strategy_version,
            doc_fingerprint=winner.doc_fingerprint,
            lexicon_source=winner.lexicon_source,
            evidence_text=winner.evidence_text,
            supporting_evidence=tuple(sorted(all_evidence)),  # Sorted for determinism
            standard_ids=winner.standard_ids,
            extensions=winner.extensions,
            section_name=winner.section_name,
            from_table=winner.from_table,
            lexicon_matched=winner.lexicon_matched,
            externally_validated=winner.externally_validated,
            pattern_strength=winner.pattern_strength,
            negated=winner.negated,
        )

    def _enforce_constraints(
        self, resolved: List[RawExtraction]
    ) -> List[RawExtraction]:
        """
        Remove extractions that violate mutual exclusivity constraints.

        For each exclusive group, if multiple fields are present,
        keep only the first one in sort order.

        Args:
            resolved: List of resolved extractions.

        Returns:
            List with constraint violations removed.
        """
        if not self.config.mutually_exclusive_fields:
            return resolved

        result = list(resolved)
        for exclusive_group in self.config.mutually_exclusive_fields:
            # Find all extractions with field_name in this exclusive group
            present = [r for r in result if r.field_name in exclusive_group]
            if len(present) > 1:
                # Keep only the first in sort order
                present_sorted = sorted(present, key=self._sort_key)
                keep = present_sorted[0]
                # Remove all others
                result = [r for r in result if r not in present or r == keep]

        return result


# Singleton instance
_DEFAULT_RESOLVER: MergeResolver = None


def get_merge_resolver() -> MergeResolver:
    """Get the singleton MergeResolver instance with default config."""
    global _DEFAULT_RESOLVER
    if _DEFAULT_RESOLVER is None:
        _DEFAULT_RESOLVER = MergeResolver(MergeConfig.default())
    return _DEFAULT_RESOLVER
