# corpus_metadata/E_normalization/E07_deduplicator.py

"""
Deduplicator for abbreviation entities.

Merges multiple entities with the same short_form into a single canonical entry,
selecting the best long_form based on quality ranking and preserving alternatives.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from A_core.A01_domain_models import (
    ExtractedEntity,
    ValidationStatus,
    FieldType,
    GeneratorType,
)


# Pattern for normalizing various dash/hyphen characters
_DASH_PATTERN = re.compile(r"[\u2010-\u2015\u2212\u002D\uFE58\uFE63\uFF0D]")


def _normalize_lf(lf: str) -> str:
    """
    Normalize long form for comparison.
    Handles hyphen/dash variants, whitespace, and case.
    """
    if not lf:
        return ""
    normalized = _DASH_PATTERN.sub("-", lf)
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.lower().strip()


class Deduplicator:
    """
    Merges duplicate abbreviation entities (same SF, different LFs).

    Strategy:
      1. Group validated entities by normalized short_form
      2. Rank long_forms by quality (source, confidence, specificity)
      3. Select best LF, store alternatives in normalized_value
      4. Return deduplicated list

    Quality ranking factors (in order of priority):
      - Generator type: GLOSSARY_TABLE > SYNTAX_PATTERN > LEXICON_MATCH
      - Field type: GLOSSARY_ENTRY > DEFINITION_PAIR > SHORT_FORM_ONLY
      - Confidence score
      - Long form specificity (longer = more specific, up to a point)
    """

    # Generator priority (higher = better)
    # Key principle: LFs found IN THE TEXT have priority over lexicon lookups
    GENERATOR_PRIORITY = {
        GeneratorType.GLOSSARY_TABLE: 200,   # Author-provided glossary (highest trust)
        GeneratorType.SYNTAX_PATTERN: 180,   # Schwartz-Hearst from text (author-defined)
        GeneratorType.TABLE_LAYOUT: 150,     # Extracted from document tables
        GeneratorType.RIGID_PATTERN: 100,    # Structured patterns in text
        GeneratorType.SECTION_PARSER: 80,    # Section-based extraction
        GeneratorType.LEXICON_MATCH: 30,     # Dictionary lookup (lowest - not in text)
    }

    # Field type priority (higher = better)
    # DEFINITION_PAIR and GLOSSARY_ENTRY mean the LF was in the document
    FIELD_TYPE_PRIORITY = {
        FieldType.GLOSSARY_ENTRY: 200,    # Author-provided (highest trust)
        FieldType.DEFINITION_PAIR: 180,   # Explicit definition in text
        FieldType.SHORT_FORM_ONLY: 20,    # No LF found in document
    }

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

        # Whether to keep rejected/ambiguous entities as-is
        self.only_validated = bool(self.config.get("only_validated", True))

        # Minimum confidence difference to prefer one LF over another
        self.confidence_margin = float(self.config.get("confidence_margin", 0.1))

        # Whether to store alternatives in normalized_value
        self.store_alternatives = bool(self.config.get("store_alternatives", True))

        # Maximum number of alternatives to store
        self.max_alternatives = int(self.config.get("max_alternatives", 5))

    def deduplicate(
        self, entities: List[ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """
        Deduplicate entities by short_form.

        Args:
            entities: List of extracted entities (post-validation)

        Returns:
            Deduplicated list with best LF selected for each SF
        """
        # Separate validated from non-validated
        validated = []
        non_validated = []

        for e in entities:
            if e.status == ValidationStatus.VALIDATED:
                validated.append(e)
            else:
                if not self.only_validated:
                    non_validated.append(e)

        # Group validated entities by normalized SF
        sf_groups: Dict[str, List[ExtractedEntity]] = defaultdict(list)
        for e in validated:
            sf_key = (e.short_form or "").strip().upper()
            if sf_key:
                sf_groups[sf_key].append(e)

        # Deduplicate each group
        deduped = []
        for sf_key, group in sf_groups.items():
            if len(group) == 1:
                # No deduplication needed, but add mention tracking
                entity = group[0]
                page = entity.primary_evidence.location.page_num if entity.primary_evidence else None
                deduped.append(self._add_dedup_flag(entity, [], 1, [page] if page else []))
            else:
                # Multiple entities for same SF - merge them
                merged = self._merge_group(sf_key, group)
                deduped.append(merged)

        # Add non-validated entities back (unchanged)
        deduped.extend(non_validated)

        return deduped

    def _merge_group(
        self, sf_key: str, group: List[ExtractedEntity]
    ) -> ExtractedEntity:
        """
        Merge a group of entities with the same SF into one.

        Selects the best LF and stores alternatives.
        """
        # Score and rank all entities
        scored: List[Tuple[float, ExtractedEntity]] = []
        for e in group:
            score = self._score_entity(e)
            scored.append((score, e))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Best entity becomes the canonical one
        best_score, best_entity = scored[0]

        # Collect all pages where the abbreviation appears
        pages: set = set()
        for _, e in scored:
            if e.primary_evidence:
                pages.add(e.primary_evidence.location.page_num)
            for ev in e.supporting_evidence or []:
                pages.add(ev.location.page_num)

        # Collect unique alternative LFs (excluding the best one)
        best_lf_norm = _normalize_lf(best_entity.long_form or "")
        alternatives = []
        seen_lf_norms = {best_lf_norm}

        for score, e in scored[1:]:
            lf = e.long_form or ""
            lf_norm = _normalize_lf(lf)

            # Skip if same normalized LF or empty
            if not lf_norm or lf_norm in seen_lf_norms:
                continue

            seen_lf_norms.add(lf_norm)
            alternatives.append({
                "long_form": lf,
                "score": round(score, 3),
                "generator": e.provenance.generator_name.value if e.provenance else None,
                "field_type": e.field_type.value if e.field_type else None,
            })

            if len(alternatives) >= self.max_alternatives:
                break

        return self._add_dedup_flag(best_entity, alternatives, len(group), sorted(pages))

    def _score_entity(self, entity: ExtractedEntity) -> float:
        """
        Score an entity for ranking.

        Higher score = better quality.

        Key principle: Long forms extracted FROM THE TEXT (Schwartz-Hearst,
        glossary tables) have priority over lexicon lookups. Author-provided
        definitions are the ground truth for that document.
        """
        score = 0.0

        # Generator priority (0-200 points) - TEXT EXTRACTION > LEXICON
        gen_type = entity.provenance.generator_name if entity.provenance else None
        if gen_type:
            score += self.GENERATOR_PRIORITY.get(gen_type, 30)

        # Field type priority (0-200 points) - EXPLICIT DEFINITION > SF-ONLY
        if entity.field_type:
            score += self.FIELD_TYPE_PRIORITY.get(entity.field_type, 10)

        # Confidence score (0-100 points)
        score += entity.confidence_score * 100

        # Long form specificity bonus (0-50 points)
        # Prefer longer expansions up to ~50 chars, then diminishing returns
        lf = entity.long_form or ""
        if lf:
            lf_len = len(lf)
            if lf_len <= 50:
                score += lf_len
            else:
                score += 50 + (lf_len - 50) * 0.1  # Diminishing returns

        # Penalty for generic/vague expansions
        lf_lower = lf.lower()
        if lf_lower in ("no expansion", "unknown", "n/a", ""):
            score -= 200

        # Bonus for having standard_id (indicates normalization worked)
        if entity.standard_id:
            score += 20

        return score

    def _add_dedup_flag(
        self, entity: ExtractedEntity, alternatives: List[Dict[str, Any]],
        mention_count: int = 1, pages_mentioned: Optional[List[int]] = None
    ) -> ExtractedEntity:
        """
        Add deduplication metadata to entity.
        """
        updates: Dict[str, Any] = {
            "mention_count": mention_count,
            "pages_mentioned": pages_mentioned or [],
        }

        # Update validation flags
        flags = list(entity.validation_flags or [])
        if "deduplicated" not in flags:
            flags.append("deduplicated")
        updates["validation_flags"] = flags

        # Store alternatives in normalized_value if enabled
        if self.store_alternatives and alternatives:
            dedup_payload = {
                "deduplication": {
                    "method": "quality_ranking",
                    "alternatives_count": len(alternatives),
                    "alternatives": alternatives,
                }
            }
            updates["normalized_value"] = self._merge_normalized_value(
                entity.normalized_value, dedup_payload
            )

        return entity.model_copy(update=updates)

    def _merge_normalized_value(
        self, existing: Any, patch: Dict[str, Any]
    ) -> Any:
        """
        Merge patch into existing normalized_value.
        """
        if existing is None:
            return patch
        if isinstance(existing, dict):
            merged = dict(existing)
            for k, v in patch.items():
                merged[k] = v
            return merged
        # If existing is string or other type, wrap it
        return {"previous_normalized_value": existing, **patch}


class DeduplicationStats:
    """
    Statistics from deduplication process.
    """

    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.groups_merged = 0
        self.alternatives_found = 0

    def __str__(self) -> str:
        removed = self.total_input - self.total_output
        return (
            f"Deduplication: {self.total_input} -> {self.total_output} "
            f"(merged {self.groups_merged} groups, {removed} duplicates removed, "
            f"{self.alternatives_found} alternatives preserved)"
        )
