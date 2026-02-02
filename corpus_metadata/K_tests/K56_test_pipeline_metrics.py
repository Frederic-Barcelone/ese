# corpus_metadata/K_tests/K56_test_pipeline_metrics.py
"""
Tests for A_core.A16_pipeline_metrics module.

Tests pipeline metrics tracking and invariant validation.
"""

from __future__ import annotations

from datetime import datetime


from A_core.A16_pipeline_metrics import (
    GenerationMetrics,
    HeuristicsMetrics,
    ValidationMetrics,
    NormalizationMetrics,
    ExportMetrics,
    ScoringMetrics,
    PipelineMetrics,
)


class TestGenerationMetrics:
    """Tests for GenerationMetrics class."""

    def test_default_values(self):
        metrics = GenerationMetrics()
        assert metrics.generated_candidates == 0
        assert metrics.by_generator == {}
        assert metrics.unique_short_forms == 0
        assert metrics.total == 0

    def test_with_values(self):
        metrics = GenerationMetrics(
            generated_candidates=500,
            by_generator={"syntax": 200, "lexicon": 300},
            unique_short_forms=150,
        )
        assert metrics.total == 500

    def test_total_property(self):
        metrics = GenerationMetrics(generated_candidates=100)
        assert metrics.total == 100


class TestHeuristicsMetrics:
    """Tests for HeuristicsMetrics class."""

    def test_default_values(self):
        metrics = HeuristicsMetrics()
        assert metrics.total_processed == 0
        assert metrics.auto_approved == 0
        assert metrics.auto_rejected == 0
        assert metrics.sent_to_llm == 0

    def test_validate_totals_success(self):
        metrics = HeuristicsMetrics(
            total_processed=100,
            auto_approved=30,
            auto_rejected=20,
            sent_to_llm=50,
        )
        assert metrics.validate_totals() is None

    def test_validate_totals_failure(self):
        metrics = HeuristicsMetrics(
            total_processed=100,
            auto_approved=30,
            auto_rejected=20,
            sent_to_llm=40,  # Sum = 90, not 100
        )
        error = metrics.validate_totals()
        assert error is not None
        assert "mismatch" in error.lower()

    def test_validate_totals_zero_total(self):
        metrics = HeuristicsMetrics(total_processed=0)
        assert metrics.validate_totals() is None

    def test_detailed_breakdown(self):
        metrics = HeuristicsMetrics(
            total_processed=100,
            auto_approved=30,
            auto_rejected=20,
            sent_to_llm=50,
            approved_by_stats_whitelist=15,
            approved_by_country_code=15,
            rejected_by_blacklist=10,
            rejected_by_context=10,
        )
        assert metrics.approved_by_stats_whitelist == 15


class TestValidationMetrics:
    """Tests for ValidationMetrics class."""

    def test_default_values(self):
        metrics = ValidationMetrics()
        assert metrics.total_validated == 0
        assert metrics.llm_approved == 0
        assert metrics.llm_rejected == 0

    def test_validate_totals_success(self):
        metrics = ValidationMetrics(
            total_validated=100,
            llm_approved=70,
            llm_rejected=25,
            llm_ambiguous=5,
        )
        assert metrics.validate_totals() is None

    def test_validate_totals_failure(self):
        metrics = ValidationMetrics(
            total_validated=100,
            llm_approved=70,
            llm_rejected=20,  # Sum = 90, not 100
            llm_ambiguous=0,
        )
        error = metrics.validate_totals()
        assert error is not None
        assert "mismatch" in error.lower()

    def test_sf_only_metrics(self):
        metrics = ValidationMetrics(
            total_validated=50,
            llm_approved=45,
            llm_rejected=5,
            sf_only_extracted=10,
            sf_only_from_llm=8,
        )
        assert metrics.sf_only_extracted == 10


class TestNormalizationMetrics:
    """Tests for NormalizationMetrics class."""

    def test_default_values(self):
        metrics = NormalizationMetrics()
        assert metrics.input_entities == 0
        assert metrics.output_entities == 0

    def test_with_values(self):
        metrics = NormalizationMetrics(
            input_entities=100,
            disambiguated=20,
            deduplicated=15,
            term_mapped=80,
            output_entities=85,
        )
        assert metrics.deduplicated == 15


class TestExportMetrics:
    """Tests for ExportMetrics class."""

    def test_default_values(self):
        metrics = ExportMetrics()
        assert metrics.validated == 0
        assert metrics.total_exported == 0

    def test_total_exported_property(self):
        metrics = ExportMetrics(
            validated=70,
            rejected=25,
            ambiguous=5,
        )
        assert metrics.total_exported == 100

    def test_by_entity_type(self):
        metrics = ExportMetrics(
            validated=100,
            by_entity_type={"abbreviation": 60, "drug": 25, "disease": 15},
        )
        assert metrics.by_entity_type["abbreviation"] == 60


class TestScoringMetrics:
    """Tests for ScoringMetrics class."""

    def test_default_values(self):
        metrics = ScoringMetrics()
        assert metrics.is_scored is True
        assert metrics.gold_count == 0

    def test_unscored(self):
        metrics = ScoringMetrics(
            is_scored=False,
            unscored_reason="No gold standard available",
        )
        assert metrics.is_scored is False

    def test_validate_against_export_success(self):
        scoring = ScoringMetrics(
            is_scored=True,
            true_positives=60,
            false_positives=10,
            false_negatives=5,
            gold_count=65,
        )
        export = ExportMetrics(validated=70)
        assert scoring.validate_against_export(export) is None

    def test_validate_against_export_failure(self):
        scoring = ScoringMetrics(
            is_scored=True,
            true_positives=60,
            false_positives=10,
        )
        export = ExportMetrics(validated=80)  # Doesn't match TP+FP=70
        error = scoring.validate_against_export(export)
        assert error is not None
        assert "mismatch" in error.lower()

    def test_validate_against_export_unscored(self):
        scoring = ScoringMetrics(is_scored=False)
        export = ExportMetrics(validated=100)
        assert scoring.validate_against_export(export) is None

    def test_validate_against_gold_success(self):
        scoring = ScoringMetrics(
            is_scored=True,
            gold_count=65,
            true_positives=60,
            false_negatives=5,
        )
        assert scoring.validate_against_gold() is None

    def test_validate_against_gold_failure(self):
        scoring = ScoringMetrics(
            is_scored=True,
            gold_count=65,
            true_positives=60,
            false_negatives=10,  # TP+FN=70, not 65
        )
        error = scoring.validate_against_gold()
        assert error is not None

    def test_precision_recall_f1(self):
        scoring = ScoringMetrics(
            is_scored=True,
            gold_count=100,
            true_positives=80,
            false_positives=20,
            false_negatives=20,
            precision=0.8,
            recall=0.8,
            f1=0.8,
        )
        assert scoring.precision == 0.8


class TestPipelineMetrics:
    """Tests for PipelineMetrics class."""

    def test_create_minimal(self):
        metrics = PipelineMetrics(run_id="RUN_001", doc_id="test.pdf")
        assert metrics.run_id == "RUN_001"
        assert isinstance(metrics.timestamp, datetime)

    def test_all_stages_initialized(self):
        metrics = PipelineMetrics(run_id="RUN_001", doc_id="test.pdf")
        assert isinstance(metrics.generation, GenerationMetrics)
        assert isinstance(metrics.heuristics, HeuristicsMetrics)
        assert isinstance(metrics.validation, ValidationMetrics)
        assert isinstance(metrics.normalization, NormalizationMetrics)
        assert isinstance(metrics.export, ExportMetrics)
        assert metrics.scoring is None  # Optional

    def test_validate_invariants_all_pass(self):
        metrics = PipelineMetrics(run_id="RUN_001", doc_id="test.pdf")
        metrics.generation.generated_candidates = 100
        metrics.heuristics.total_processed = 100
        metrics.heuristics.auto_approved = 30
        metrics.heuristics.auto_rejected = 20
        metrics.heuristics.sent_to_llm = 50
        metrics.validation.total_validated = 50
        metrics.validation.llm_approved = 40
        metrics.validation.llm_rejected = 10
        metrics.export.validated = 70

        errors = metrics.validate_invariants()
        assert len(errors) == 0

    def test_validate_invariants_heuristics_mismatch(self):
        metrics = PipelineMetrics(run_id="RUN_001", doc_id="test.pdf")
        metrics.heuristics.total_processed = 100
        metrics.heuristics.auto_approved = 30
        metrics.heuristics.auto_rejected = 20
        metrics.heuristics.sent_to_llm = 40  # Sum=90, not 100

        errors = metrics.validate_invariants()
        assert len(errors) == 1
        assert "heuristics" in errors[0].lower()

    def test_validate_invariants_validation_mismatch(self):
        metrics = PipelineMetrics(run_id="RUN_001", doc_id="test.pdf")
        metrics.validation.total_validated = 100
        metrics.validation.llm_approved = 70
        metrics.validation.llm_rejected = 20  # Sum=90, not 100

        errors = metrics.validate_invariants()
        assert len(errors) == 1
        assert "validation" in errors[0].lower()

    def test_validate_invariants_scoring_export_mismatch(self):
        metrics = PipelineMetrics(run_id="RUN_001", doc_id="test.pdf")
        metrics.export.validated = 100
        metrics.scoring = ScoringMetrics(
            is_scored=True,
            true_positives=80,
            false_positives=10,  # TP+FP=90, not 100
            gold_count=85,
            false_negatives=5,
        )

        errors = metrics.validate_invariants()
        assert any("scoring" in e.lower() or "export" in e.lower() for e in errors)

    def test_summary(self):
        metrics = PipelineMetrics(run_id="RUN_001", doc_id="test.pdf")
        metrics.generation.generated_candidates = 100
        metrics.heuristics.auto_approved = 30
        metrics.validation.llm_approved = 40
        metrics.export.validated = 70

        summary = metrics.summary()
        assert summary["generated"] == 100
        assert summary["auto_approved"] == 30
        assert summary["llm_approved"] == 40
        assert summary["exported_validated"] == 70

    def test_summary_with_scoring(self):
        metrics = PipelineMetrics(run_id="RUN_001", doc_id="test.pdf")
        metrics.scoring = ScoringMetrics(
            is_scored=True,
            gold_count=80,
            true_positives=60,
            false_positives=10,
            false_negatives=20,
        )

        summary = metrics.summary()
        assert summary["gold_count"] == 80
        assert summary["true_positives"] == 60
