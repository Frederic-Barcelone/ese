# corpus_metadata/A_core/A16_pipeline_metrics.py
"""
Unified metrics tracking system for extraction pipeline observability.

This module provides a single source of truth for all pipeline metrics, ensuring
consistency across generation, heuristics, validation, normalization, export, and
scoring stages. Use PipelineMetrics to track progress, validate invariants, and
generate summary reports. All logs and displays should read from this object.

Key Components:
    - PipelineMetrics: Top-level container with all stage metrics and validation
    - GenerationMetrics: Candidate counts from C_generators
    - HeuristicsMetrics: PASO rule filtering breakdown (auto-approved/rejected/LLM)
    - ValidationMetrics: LLM validation results and SF-only extraction counts
    - NormalizationMetrics: Disambiguation and deduplication counts
    - ExportMetrics: Final export counts by entity type and status
    - ScoringMetrics: Precision/recall/F1 and TP/FP/FN counts

Example:
    >>> from A_core.A16_pipeline_metrics import PipelineMetrics
    >>> metrics = PipelineMetrics(run_id="RUN_001", doc_id="study.pdf")
    >>> metrics.generation.generated_candidates = 500
    >>> metrics.validation.llm_approved = 73
    >>> errors = metrics.validate_invariants()
    >>> if not errors:
    ...     print(metrics.summary())

Dependencies:
    - pydantic: For model validation and configuration
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class GenerationMetrics(BaseModel):
    """Metrics from candidate generation stage (C_generators)."""

    generated_candidates: int = 0
    by_generator: Dict[str, int] = Field(default_factory=dict)
    unique_short_forms: int = 0
    filtered_lexicon_only: int = 0

    model_config = ConfigDict(extra="forbid")

    @property
    def total(self) -> int:
        """Total candidates generated."""
        return self.generated_candidates


class HeuristicsMetrics(BaseModel):
    """Metrics from heuristics filtering stage (PASO rules)."""

    total_processed: int = 0
    auto_approved: int = 0
    auto_rejected: int = 0
    sent_to_llm: int = 0

    # Detailed breakdown
    approved_by_stats_whitelist: int = 0
    approved_by_country_code: int = 0
    rejected_by_blacklist: int = 0
    rejected_by_context: int = 0
    rejected_by_trial_id: int = 0
    rejected_by_common_word: int = 0

    model_config = ConfigDict(extra="forbid")

    def validate_totals(self) -> Optional[str]:
        """Check that auto_approved + auto_rejected + sent_to_llm == total_processed."""
        expected = self.auto_approved + self.auto_rejected + self.sent_to_llm
        if self.total_processed > 0 and expected != self.total_processed:
            return (
                f"Heuristics totals mismatch: "
                f"auto_approved({self.auto_approved}) + "
                f"auto_rejected({self.auto_rejected}) + "
                f"sent_to_llm({self.sent_to_llm}) = {expected} "
                f"!= total_processed({self.total_processed})"
            )
        return None


class ValidationMetrics(BaseModel):
    """Metrics from LLM validation stage (D_validation)."""

    total_validated: int = 0
    llm_approved: int = 0
    llm_rejected: int = 0
    llm_ambiguous: int = 0
    llm_calls: int = 0
    llm_errors: int = 0

    # SF-only extraction metrics
    sf_only_extracted: int = 0
    sf_only_from_llm: int = 0

    model_config = ConfigDict(extra="forbid")

    def validate_totals(self) -> Optional[str]:
        """Check that llm_approved + llm_rejected + llm_ambiguous == total_validated."""
        if self.total_validated > 0:
            expected = self.llm_approved + self.llm_rejected + self.llm_ambiguous
            if expected != self.total_validated:
                return (
                    f"Validation totals mismatch: "
                    f"llm_approved({self.llm_approved}) + "
                    f"llm_rejected({self.llm_rejected}) + "
                    f"llm_ambiguous({self.llm_ambiguous}) = {expected} "
                    f"!= total_validated({self.total_validated})"
                )
        return None


class NormalizationMetrics(BaseModel):
    """Metrics from normalization stage (E_normalization)."""

    input_entities: int = 0
    disambiguated: int = 0
    deduplicated: int = 0
    term_mapped: int = 0
    output_entities: int = 0

    model_config = ConfigDict(extra="forbid")


class ExportMetrics(BaseModel):
    """Metrics from export stage (J_export) - ground truth for scoring."""

    validated: int = 0
    rejected: int = 0
    ambiguous: int = 0

    # Entity type breakdown
    by_entity_type: Dict[str, int] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @property
    def total_exported(self) -> int:
        """Total entities exported (all statuses)."""
        return self.validated + self.rejected + self.ambiguous


class ScoringMetrics(BaseModel):
    """Metrics from scoring/evaluation (F_evaluation)."""

    is_scored: bool = True
    unscored_reason: Optional[str] = None

    gold_count: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None

    model_config = ConfigDict(extra="forbid")

    def validate_against_export(self, export: ExportMetrics) -> Optional[str]:
        """Check that TP + FP == export.validated."""
        if not self.is_scored:
            return None

        expected_validated = self.true_positives + self.false_positives
        if expected_validated != export.validated:
            return (
                f"Scoring/export mismatch: "
                f"TP({self.true_positives}) + FP({self.false_positives}) = "
                f"{expected_validated} != export.validated({export.validated})"
            )
        return None

    def validate_against_gold(self) -> Optional[str]:
        """Check that TP + FN == gold_count."""
        if not self.is_scored:
            return None

        expected_gold = self.true_positives + self.false_negatives
        if expected_gold != self.gold_count:
            return (
                f"Scoring/gold mismatch: "
                f"TP({self.true_positives}) + FN({self.false_negatives}) = "
                f"{expected_gold} != gold_count({self.gold_count})"
            )
        return None


class PipelineMetrics(BaseModel):
    """
    Single source of truth for all pipeline metrics.

    This class tracks metrics across all pipeline stages and provides
    validation to ensure consistency. All other metrics displays
    (logs, summaries, reports) should read from this object.
    """

    run_id: str
    doc_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Stage metrics
    generation: GenerationMetrics = Field(default_factory=GenerationMetrics)
    heuristics: HeuristicsMetrics = Field(default_factory=HeuristicsMetrics)
    validation: ValidationMetrics = Field(default_factory=ValidationMetrics)
    normalization: NormalizationMetrics = Field(default_factory=NormalizationMetrics)
    export: ExportMetrics = Field(default_factory=ExportMetrics)
    scoring: Optional[ScoringMetrics] = None

    model_config = ConfigDict(extra="forbid")

    def validate_invariants(self) -> List[str]:
        """
        Validate all metrics invariants.

        Returns a list of error messages for any violations.
        Empty list means all invariants hold.
        """
        errors: List[str] = []

        # Check heuristics totals
        if heuristics_error := self.heuristics.validate_totals():
            errors.append(heuristics_error)

        # Check validation totals
        if validation_error := self.validation.validate_totals():
            errors.append(validation_error)

        # Check scoring against export
        if self.scoring:
            if export_error := self.scoring.validate_against_export(self.export):
                errors.append(export_error)

            if gold_error := self.scoring.validate_against_gold():
                errors.append(gold_error)

        return errors

    def summary(self) -> Dict[str, int]:
        """Return a summary dict suitable for logging."""
        summary = {
            "generated": self.generation.generated_candidates,
            "auto_approved": self.heuristics.auto_approved,
            "auto_rejected": self.heuristics.auto_rejected,
            "sent_to_llm": self.heuristics.sent_to_llm,
            "llm_approved": self.validation.llm_approved,
            "llm_rejected": self.validation.llm_rejected,
            "exported_validated": self.export.validated,
            "exported_rejected": self.export.rejected,
        }

        if self.scoring and self.scoring.is_scored:
            summary.update({
                "gold_count": self.scoring.gold_count,
                "true_positives": self.scoring.true_positives,
                "false_positives": self.scoring.false_positives,
                "false_negatives": self.scoring.false_negatives,
            })

        return summary


__all__ = [
    "PipelineMetrics",
    "GenerationMetrics",
    "HeuristicsMetrics",
    "ValidationMetrics",
    "NormalizationMetrics",
    "ExportMetrics",
    "ScoringMetrics",
]
