# corpus_metadata/K_tests/K41_test_scorer.py
"""
Tests for F_evaluation.F02_scorer module.

Tests precision/recall/F1 scoring for abbreviation extraction.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import pytest

from F_evaluation.F02_scorer import (
    Scorer,
    ScorerConfig,
    ScoreReport,
    CorpusScoreReport,
)
from F_evaluation.F01_gold_loader import GoldAnnotation
from A_core.A01_domain_models import (
    ExtractedEntity,
    ValidationStatus,
    FieldType,
    EvidenceSpan,
    Coordinate,
    ProvenanceMetadata,
    GeneratorType,
)


def make_entity(
    sf: str,
    lf: Optional[str] = None,
    doc_id: str = "test.pdf",
    status: ValidationStatus = ValidationStatus.VALIDATED,
    field_type: FieldType = FieldType.DEFINITION_PAIR,
) -> ExtractedEntity:
    """Helper to create test entities."""
    return ExtractedEntity(
        candidate_id=uuid.uuid4(),
        doc_id=doc_id,
        short_form=sf,
        long_form=lf,
        field_type=field_type,
        primary_evidence=EvidenceSpan(
            text="test context",
            location=Coordinate(page_num=1),
            scope_ref="abc",
            start_char_offset=0,
            end_char_offset=12,
        ),
        supporting_evidence=[],
        status=status,
        confidence_score=0.9,
        provenance=ProvenanceMetadata(
            run_id="TEST",
            pipeline_version="1.0",
            doc_fingerprint="abc",
            generator_name=GeneratorType.SYNTAX_PATTERN,
        ),
    )


def make_gold(sf: str, lf: Optional[str] = None, doc_id: str = "test.pdf") -> GoldAnnotation:
    """Helper to create gold annotations."""
    return GoldAnnotation(doc_id=doc_id, short_form=sf, long_form=lf)


class TestScoreReport:
    """Tests for ScoreReport model."""

    def test_scored_report(self):
        report = ScoreReport(
            is_scored=True,
            precision=0.85,
            recall=0.90,
            f1=0.87,
            true_positives=17,
            false_positives=3,
            false_negatives=2,
            gold_count=19,
        )
        assert report.is_scored
        assert report.precision == 0.85

    def test_unscored_report(self):
        report = ScoreReport(
            is_scored=False,
            unscored_reason="no gold annotations",
            precision=None,
            recall=None,
            f1=None,
        )
        assert not report.is_scored
        assert report.precision is None


class TestScorerConfig:
    """Tests for ScorerConfig."""

    def test_default_config(self):
        config = ScorerConfig()
        assert config.require_long_form_match is True
        assert config.only_validated is True
        assert config.fuzzy_long_form_match is True

    def test_custom_config(self):
        config = ScorerConfig(
            require_long_form_match=False,
            fuzzy_threshold=0.9,
        )
        assert config.require_long_form_match is False
        assert config.fuzzy_threshold == 0.9


class TestScorerNormalization:
    """Tests for normalization methods."""

    @pytest.fixture
    def scorer(self):
        return Scorer()

    def test_norm_sf_uppercase(self, scorer):
        assert scorer._norm_sf("tnf") == "TNF"

    def test_norm_sf_removes_dashes(self, scorer):
        assert scorer._norm_sf("SC5B-9") == "SC5B9"

    def test_norm_lf_lowercase(self, scorer):
        assert scorer._norm_lf("Tumor Necrosis Factor") == "tumor necrosis factor"

    def test_norm_lf_whitespace(self, scorer):
        assert scorer._norm_lf("Tumor   Necrosis   Factor") == "tumor necrosis factor"

    def test_norm_lf_none(self, scorer):
        assert scorer._norm_lf(None) is None

    def test_is_unknown_lf(self, scorer):
        assert scorer._is_unknown_lf(None)
        assert scorer._is_unknown_lf("UNKNOWN")
        assert scorer._is_unknown_lf("n/a")
        assert not scorer._is_unknown_lf("Tumor Necrosis Factor")


class TestScorerBasicEvaluation:
    """Tests for basic evaluation scenarios."""

    @pytest.fixture
    def scorer(self):
        return Scorer()

    def test_perfect_match(self, scorer):
        system = [make_entity("TNF", "Tumor Necrosis Factor")]
        gold = [make_gold("TNF", "Tumor Necrosis Factor")]

        report = scorer.evaluate_doc(system, gold)

        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1 == 1.0
        assert report.true_positives == 1

    def test_false_positive(self, scorer):
        system = [
            make_entity("TNF", "Tumor Necrosis Factor"),
            make_entity("XYZ", "Not In Gold"),
        ]
        gold = [make_gold("TNF", "Tumor Necrosis Factor")]

        report = scorer.evaluate_doc(system, gold)

        assert report.true_positives == 1
        assert report.false_positives == 1
        assert report.false_negatives == 0

    def test_false_negative(self, scorer):
        system = [make_entity("TNF", "Tumor Necrosis Factor")]
        gold = [
            make_gold("TNF", "Tumor Necrosis Factor"),
            make_gold("IL6", "Interleukin 6"),
        ]

        report = scorer.evaluate_doc(system, gold)

        assert report.true_positives == 1
        assert report.false_positives == 0
        assert report.false_negatives == 1

    def test_no_gold_unscored(self, scorer):
        system = [make_entity("TNF", "Tumor Necrosis Factor")]
        gold: list[Any] = []

        report = scorer.evaluate_doc(system, gold)

        assert not report.is_scored
        assert report.unscored_reason == "no gold annotations"


class TestScorerFuzzyMatching:
    """Tests for fuzzy long form matching."""

    @pytest.fixture
    def scorer(self):
        return Scorer(ScorerConfig(fuzzy_long_form_match=True, fuzzy_threshold=0.8))

    def test_substring_match(self, scorer):
        system = [make_entity("FDA", "Food and Drug Administration")]
        gold = [make_gold("FDA", "US Food and Drug Administration")]

        report = scorer.evaluate_doc(system, gold)

        # Substring match should count as TP
        assert report.true_positives == 1

    def test_similar_match(self, scorer):
        system = [make_entity("TNF", "Tumour Necrosis Factor")]
        gold = [make_gold("TNF", "Tumor Necrosis Factor")]

        report = scorer.evaluate_doc(system, gold)

        # High similarity should match
        assert report.true_positives == 1

    def test_no_fuzzy_when_disabled(self):
        scorer = Scorer(ScorerConfig(fuzzy_long_form_match=False))
        system = [make_entity("TNF", "Tumour Necrosis Factor")]
        gold = [make_gold("TNF", "Tumor Necrosis Factor")]

        report = scorer.evaluate_doc(system, gold)

        # Without fuzzy, different spellings don't match
        assert report.true_positives == 0
        assert report.false_positives == 1
        assert report.false_negatives == 1


class TestScorerSFOnlyMatching:
    """Tests for SF-only gold entries."""

    @pytest.fixture
    def scorer(self):
        return Scorer(ScorerConfig(allow_sf_only_gold=True))

    def test_sf_only_gold_matches_any_lf(self, scorer):
        system = [make_entity("TNF", "Any Long Form")]
        gold = [make_gold("TNF", None)]  # SF-only gold

        report = scorer.evaluate_doc(system, gold)

        assert report.true_positives == 1

    def test_sf_only_gold_missing_system(self, scorer):
        system: list[Any] = []
        gold = [make_gold("TNF", None)]

        report = scorer.evaluate_doc(system, gold)

        assert report.false_negatives == 1


class TestScorerFieldTypes:
    """Tests for field type filtering."""

    def test_excludes_wrong_field_types(self):
        scorer = Scorer(
            ScorerConfig(include_field_types={FieldType.DEFINITION_PAIR})
        )

        # SHORT_FORM_ONLY should be excluded
        system = [
            make_entity("TNF", "Tumor Necrosis Factor", field_type=FieldType.DEFINITION_PAIR),
            make_entity("IL6", None, field_type=FieldType.SHORT_FORM_ONLY),
        ]
        gold = [
            make_gold("TNF", "Tumor Necrosis Factor"),
            make_gold("IL6", "Interleukin 6"),
        ]

        report = scorer.evaluate_doc(system, gold)

        # IL6 from system excluded due to field type
        assert report.true_positives == 1


class TestScorerCorpus:
    """Tests for corpus-level evaluation."""

    @pytest.fixture
    def scorer(self):
        return Scorer()

    def test_corpus_evaluation(self, scorer):
        system = [
            make_entity("TNF", "Tumor Necrosis Factor", doc_id="doc1.pdf"),
            make_entity("IL6", "Interleukin 6", doc_id="doc2.pdf"),
        ]
        gold = [
            make_gold("TNF", "Tumor Necrosis Factor", doc_id="doc1.pdf"),
            make_gold("IL6", "Interleukin 6", doc_id="doc2.pdf"),
        ]

        report = scorer.evaluate_corpus(system, gold)

        assert isinstance(report, CorpusScoreReport)
        assert report.micro.precision == 1.0
        assert report.macro.precision == 1.0
        assert len(report.per_doc) == 2

    def test_corpus_micro_vs_macro(self, scorer):
        # Doc1: 1 TP, Doc2: 0 TP, 1 FP
        system = [
            make_entity("TNF", "Tumor Necrosis Factor", doc_id="doc1.pdf"),
            make_entity("WRONG", "Wrong", doc_id="doc2.pdf"),
        ]
        gold = [
            make_gold("TNF", "Tumor Necrosis Factor", doc_id="doc1.pdf"),
            make_gold("IL6", "Interleukin 6", doc_id="doc2.pdf"),
        ]

        report = scorer.evaluate_corpus(system, gold)

        # Micro: 1 TP, 1 FP, 1 FN -> P=0.5, R=0.5
        assert report.micro.true_positives == 1
        assert report.micro.false_positives == 1
        assert report.micro.false_negatives == 1
        assert report.micro.precision == pytest.approx(0.5, rel=0.01)
        assert report.micro.recall == pytest.approx(0.5, rel=0.01)


class TestScorerMetrics:
    """Tests for metric calculations."""

    @pytest.fixture
    def scorer(self):
        return Scorer()

    def test_precision_calculation(self, scorer):
        # 2 TP, 1 FP -> P = 2/3
        system = [
            make_entity("A", "Alpha"),
            make_entity("B", "Beta"),
            make_entity("C", "Wrong"),
        ]
        gold = [
            make_gold("A", "Alpha"),
            make_gold("B", "Beta"),
        ]

        report = scorer.evaluate_doc(system, gold)

        assert report.precision == pytest.approx(2 / 3, rel=0.01)

    def test_recall_calculation(self, scorer):
        # 2 TP, 1 FN -> R = 2/3
        system = [
            make_entity("A", "Alpha"),
            make_entity("B", "Beta"),
        ]
        gold = [
            make_gold("A", "Alpha"),
            make_gold("B", "Beta"),
            make_gold("C", "Gamma"),
        ]

        report = scorer.evaluate_doc(system, gold)

        assert report.recall == pytest.approx(2 / 3, rel=0.01)

    def test_f1_calculation(self, scorer):
        # P = 0.5, R = 0.5 -> F1 = 0.5
        system = [
            make_entity("A", "Alpha"),
            make_entity("X", "Wrong"),
        ]
        gold = [
            make_gold("A", "Alpha"),
            make_gold("B", "Beta"),
        ]

        report = scorer.evaluate_doc(system, gold)

        assert report.f1 == pytest.approx(0.5, rel=0.01)
