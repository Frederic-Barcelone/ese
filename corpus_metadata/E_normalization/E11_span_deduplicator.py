# corpus_metadata/E_normalization/E11_span_deduplicator.py
"""
Span deduplication for multi-source NER outputs.

Merges overlapping spans from multiple NER enrichers:
- EpiExtract4GARD-v2
- ZeroShotBioNER
- BiomedicalNER

Keeps highest confidence score when spans overlap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class NERSpan:
    """Unified span representation from any NER source."""

    text: str
    category: str  # e.g., "symptom", "drug_dosage", "epidemiology"
    confidence: float
    source: str  # e.g., "EpiExtract4GARD-v2", "ZeroShotBioNER", "BiomedicalNER"
    start: Optional[int] = None
    end: Optional[int] = None
    entity_type: Optional[str] = None  # Original entity type from source
    metadata: Dict[str, Any] = field(default_factory=dict)

    def overlaps_with(self, other: "NERSpan", threshold: float = 0.5) -> bool:
        """Check if this span overlaps with another span."""
        # If positions are available, use positional overlap
        if self.start is not None and self.end is not None and \
           other.start is not None and other.end is not None:
            overlap_start = max(self.start, other.start)
            overlap_end = min(self.end, other.end)
            if overlap_start < overlap_end:
                overlap_len = overlap_end - overlap_start
                min_len = min(self.end - self.start, other.end - other.start)
                return (overlap_len / min_len) >= threshold
            return False

        # Fall back to text similarity
        return self._text_overlap(other.text) >= threshold

    def _text_overlap(self, other_text: str) -> float:
        """Calculate text overlap ratio."""
        self_normalized = self.text.lower().strip()
        other_normalized = other_text.lower().strip()

        # Exact match
        if self_normalized == other_normalized:
            return 1.0

        # One contains the other
        if self_normalized in other_normalized:
            return len(self_normalized) / len(other_normalized)
        if other_normalized in self_normalized:
            return len(other_normalized) / len(self_normalized)

        # Word overlap
        self_words = set(self_normalized.split())
        other_words = set(other_normalized.split())
        if not self_words or not other_words:
            return 0.0

        intersection = self_words & other_words
        union = self_words | other_words
        return len(intersection) / len(union) if union else 0.0


@dataclass
class DeduplicationResult:
    """Result of span deduplication."""

    unique_spans: List[NERSpan] = field(default_factory=list)
    merged_count: int = 0
    total_input: int = 0
    by_category: Dict[str, List[NERSpan]] = field(default_factory=dict)
    by_source: Dict[str, int] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        """Convert to summary dict."""
        return {
            "total_input": self.total_input,
            "unique_spans": len(self.unique_spans),
            "merged_count": self.merged_count,
            "dedup_ratio": f"{(1 - len(self.unique_spans) / self.total_input) * 100:.1f}%" if self.total_input > 0 else "0%",
            "by_category": {k: len(v) for k, v in self.by_category.items()},
            "by_source": self.by_source,
        }


class SpanDeduplicator:
    """
    Deduplicates overlapping NER spans from multiple sources.

    Strategy:
    1. Group spans by category (similar entity types grouped)
    2. Within each group, find overlapping spans
    3. Keep span with highest confidence
    4. Preserve source attribution for provenance
    """

    # Category mappings to normalize across sources
    CATEGORY_GROUPS = {
        # Epidemiology
        "epidemiology": ["epidemiology", "prevalence", "incidence", "demographics"],
        # Adverse events
        "adverse_event": ["adverse_event", "ADE", "side_effect"],
        # Drug administration
        "drug_admin": ["drug_dosage", "drug_frequency", "drug_route", "treatment_duration",
                       "dosage", "frequency", "route", "duration", "strength", "form"],
        # Clinical
        "symptom": ["symptom", "Sign_symptom", "sign"],
        "procedure": ["diagnostic_procedure", "therapeutic_procedure", "procedure"],
        "lab_value": ["lab_value", "Lab_value", "laboratory"],
        "outcome": ["outcome", "Outcome", "clinical_event"],
        # Demographics
        "demographics": ["demographics_age", "demographics_sex", "demographics_family_history",
                        "age", "sex", "family_history", "personal_background"],
        # Patient Journey
        "diagnostic_delay": ["diagnostic_delay"],
        "treatment_history": ["treatment_line", "prior_therapy"],
        "care_pathway": ["care_pathway_step", "care_pathway"],
        "trial_burden": ["surveillance_frequency", "visit_frequency"],
        "retention_risks": ["pain_point", "recruitment_touchpoint"],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.overlap_threshold = config.get("overlap_threshold", 0.5)
        self.min_confidence = config.get("min_confidence", 0.3)

        # Build reverse mapping
        self._category_map = {}
        for group, categories in self.CATEGORY_GROUPS.items():
            for cat in categories:
                self._category_map[cat.lower()] = group

    def normalize_category(self, category: str) -> str:
        """Normalize category to standard group."""
        return self._category_map.get(category.lower(), category.lower())

    def deduplicate(self, spans: List[NERSpan]) -> DeduplicationResult:
        """
        Deduplicate a list of NER spans.

        Args:
            spans: List of NERSpan from various sources

        Returns:
            DeduplicationResult with unique spans
        """
        if not spans:
            return DeduplicationResult()

        result = DeduplicationResult(total_input=len(spans))

        # Filter by minimum confidence
        valid_spans = [s for s in spans if s.confidence >= self.min_confidence]

        # Group by normalized category
        by_category: Dict[str, List[NERSpan]] = {}
        for span in valid_spans:
            norm_cat = self.normalize_category(span.category)
            if norm_cat not in by_category:
                by_category[norm_cat] = []
            by_category[norm_cat].append(span)

        # Deduplicate within each category
        unique_spans = []
        for category, category_spans in by_category.items():
            deduped = self._deduplicate_category(category_spans)
            unique_spans.extend(deduped)
            result.by_category[category] = deduped

        # Count sources
        for span in unique_spans:
            result.by_source[span.source] = result.by_source.get(span.source, 0) + 1

        result.unique_spans = unique_spans
        result.merged_count = len(valid_spans) - len(unique_spans)

        return result

    def _deduplicate_category(self, spans: List[NERSpan]) -> List[NERSpan]:
        """Deduplicate spans within a single category."""
        if not spans:
            return []

        if len(spans) == 1:
            return spans

        # Sort by confidence descending (keep highest first)
        sorted_spans = sorted(spans, key=lambda s: s.confidence, reverse=True)

        unique = []
        used_indices: Set[int] = set()

        for i, span in enumerate(sorted_spans):
            if i in used_indices:
                continue

            # Find all spans that overlap with this one
            overlapping_indices = {i}
            for j, other in enumerate(sorted_spans):
                if j != i and j not in used_indices:
                    if span.overlaps_with(other, self.overlap_threshold):
                        overlapping_indices.add(j)

            # Mark all overlapping as used
            used_indices.update(overlapping_indices)

            # Keep the highest confidence span (already sorted)
            # But merge source information
            if len(overlapping_indices) > 1:
                sources = [sorted_spans[idx].source for idx in overlapping_indices]
                span.metadata["merged_from"] = list(set(sources))
                span.metadata["merged_count"] = len(overlapping_indices)

            unique.append(span)

        return unique


def deduplicate_feasibility_candidates(
    candidates: List[Any],  # List of FeasibilityCandidate
) -> Tuple[List[Any], DeduplicationResult]:
    """
    Deduplicate FeasibilityCandidate objects from multiple NER sources.

    Args:
        candidates: List of FeasibilityCandidate objects

    Returns:
        Tuple of (deduplicated candidates, deduplication result)
    """
    # Convert to NERSpan
    spans = []
    candidate_map: Dict[int, Any] = {}  # span index -> original candidate

    for i, cand in enumerate(candidates):
        # Only deduplicate NER-sourced candidates
        source = getattr(cand, 'source', None)
        if source not in ["EpiExtract4GARD-v2", "ZeroShotBioNER", "BiomedicalNER", "PatientJourneyNER"]:
            continue

        span = NERSpan(
            text=getattr(cand, 'text', ''),
            category=getattr(cand, 'category', 'unknown'),
            confidence=getattr(cand, 'confidence', 0.5),
            source=source,
            entity_type=getattr(cand, 'entity_type', None),
        )
        spans.append(span)
        candidate_map[len(spans) - 1] = cand

    # Deduplicate
    deduplicator = SpanDeduplicator()
    result = deduplicator.deduplicate(spans)

    # Map back to candidates
    # Keep non-NER candidates + deduplicated NER candidates
    non_ner_candidates = [
        c for c in candidates
        if getattr(c, 'source', None) not in ["EpiExtract4GARD-v2", "ZeroShotBioNER", "BiomedicalNER", "PatientJourneyNER"]
    ]

    # Find original candidates for unique spans
    unique_candidates = []
    for span in result.unique_spans:
        # Find matching original candidate
        for idx, orig_cand in candidate_map.items():
            if (getattr(orig_cand, 'text', '') == span.text and
                getattr(orig_cand, 'source', '') == span.source):
                unique_candidates.append(orig_cand)
                break

    return non_ner_candidates + unique_candidates, result
