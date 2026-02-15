# corpus_metadata/K_tests/K42_test_d_e_f_imports.py
"""
Tests for D_validation, E_normalization, and F_evaluation module imports.

Ensures all modules can be imported and have proper exports.
"""

from __future__ import annotations

import importlib
from enum import Enum

import pytest


# D_validation modules
D_VALIDATION_MODULES = [
    "D01_prompt_registry",
    "D02_llm_engine",
    "D03_validation_logger",
]

# E_normalization modules
E_NORMALIZATION_MODULES = [
    "E01_term_mapper",
    "E02_disambiguator",
    "E03_disease_normalizer",
    "E04_pubtator_enricher",
    "E05_drug_enricher",
    "E06_nct_enricher",
    "E07_deduplicator",
    "E08_epi_extract_enricher",
    "E09_zeroshot_bioner",
    "E10_biomedical_ner_all",
    "E11_span_deduplicator",
    "E12_patient_journey_enricher",
    "E13_registry_enricher",
    "E14_citation_validator",
    "E15_genetic_enricher",
    "E16_drug_combination_parser",
    "E17_entity_deduplicator",
]

# F_evaluation modules
F_EVALUATION_MODULES = [
    "F01_gold_loader",
    "F02_scorer",
    "F03_evaluation_runner",
]


class TestDValidationImports:
    """Tests that all D_validation modules can be imported."""

    @pytest.mark.parametrize("module_name", D_VALIDATION_MODULES)
    def test_module_imports(self, module_name):
        try:
            module = importlib.import_module(f"D_validation.{module_name}")
            assert module is not None
        except ImportError as e:
            pytest.fail(f"Failed to import D_validation.{module_name}: {e}")

    @pytest.mark.parametrize("module_name", D_VALIDATION_MODULES)
    def test_module_has_all(self, module_name):
        try:
            module = importlib.import_module(f"D_validation.{module_name}")
            if hasattr(module, "__all__"):
                assert isinstance(module.__all__, (list, tuple))
        except ImportError as e:
            pytest.fail(f"Module {module_name} not importable: {e}")


class TestENormalizationImports:
    """Tests that all E_normalization modules can be imported."""

    @pytest.mark.parametrize("module_name", E_NORMALIZATION_MODULES)
    def test_module_imports(self, module_name):
        try:
            module = importlib.import_module(f"E_normalization.{module_name}")
            assert module is not None
        except ImportError as e:
            pytest.fail(f"Failed to import E_normalization.{module_name}: {e}")

    @pytest.mark.parametrize("module_name", E_NORMALIZATION_MODULES)
    def test_module_has_all(self, module_name):
        try:
            module = importlib.import_module(f"E_normalization.{module_name}")
            if hasattr(module, "__all__"):
                assert isinstance(module.__all__, (list, tuple))
        except ImportError as e:
            pytest.fail(f"Module {module_name} not importable: {e}")


class TestFEvaluationImports:
    """Tests that all F_evaluation modules can be imported."""

    @pytest.mark.parametrize("module_name", F_EVALUATION_MODULES)
    def test_module_imports(self, module_name):
        try:
            module = importlib.import_module(f"F_evaluation.{module_name}")
            assert module is not None
        except ImportError as e:
            pytest.fail(f"Failed to import F_evaluation.{module_name}: {e}")

    @pytest.mark.parametrize("module_name", F_EVALUATION_MODULES)
    def test_module_has_all(self, module_name):
        try:
            module = importlib.import_module(f"F_evaluation.{module_name}")
            if hasattr(module, "__all__"):
                assert isinstance(module.__all__, (list, tuple))
        except ImportError as e:
            pytest.fail(f"Module {module_name} not importable: {e}")


class TestDValidationExports:
    """Tests for specific D_validation module exports."""

    def test_prompt_registry_exports(self):
        from D_validation.D01_prompt_registry import (
            PromptTask,
            PromptBundle,
            PromptRegistry,
        )
        assert issubclass(PromptTask, Enum)
        assert hasattr(PromptBundle, "model_fields")
        assert callable(PromptRegistry.get_bundle)

    def test_llm_engine_exports(self):
        from D_validation.D02_llm_engine import (
            ClaudeClient,
            LLMEngine,
            VerificationResult,
        )
        assert hasattr(ClaudeClient, "complete_json")
        assert hasattr(LLMEngine, "verify_candidate")
        # VerificationResult has status as a Pydantic field
        assert "status" in VerificationResult.model_fields

    def test_validation_logger_exports(self):
        from D_validation.D03_validation_logger import ValidationLogger
        assert hasattr(ValidationLogger, "log_validation")
        assert hasattr(ValidationLogger, "log_error")

    def test_quote_verifier_exports(self):
        from Z_utils.Z14_quote_verifier import (
            verify_quote,
            verify_number,
        )
        assert callable(verify_quote)
        assert callable(verify_number)


class TestENormalizationExports:
    """Tests for specific E_normalization module exports."""

    def test_term_mapper_exports(self):
        from E_normalization.E01_term_mapper import TermMapper
        assert hasattr(TermMapper, "normalize")

    def test_deduplicator_exports(self):
        from E_normalization.E07_deduplicator import Deduplicator
        assert hasattr(Deduplicator, "deduplicate")

    def test_span_deduplicator_exports(self):
        from E_normalization.E11_span_deduplicator import (
            NERSpan,
            SpanDeduplicator,
        )
        assert hasattr(SpanDeduplicator, "deduplicate")
        assert hasattr(NERSpan, "overlaps_with")

    def test_entity_deduplicator_exports(self):
        from E_normalization.E17_entity_deduplicator import EntityDeduplicator
        assert hasattr(EntityDeduplicator, "deduplicate_diseases")
        assert hasattr(EntityDeduplicator, "deduplicate_drugs")
        assert hasattr(EntityDeduplicator, "deduplicate_genes")


class TestFEvaluationExports:
    """Tests for specific F_evaluation module exports."""

    def test_gold_loader_exports(self):
        from F_evaluation.F01_gold_loader import (
            GoldLoader,
        )
        assert hasattr(GoldLoader, "load_json")
        assert hasattr(GoldLoader, "load_csv")

    def test_scorer_exports(self):
        from F_evaluation.F02_scorer import (
            Scorer,
        )
        assert hasattr(Scorer, "evaluate_doc")
        assert hasattr(Scorer, "evaluate_corpus")


# =========================================================================
# Behavioral tests
# =========================================================================


class TestPromptRegistryBehavioral:
    """Behavioral tests for PromptRegistry.get_bundle()."""

    def test_get_bundle_disease_batch_returns_populated_bundle(self):
        from D_validation.D01_prompt_registry import PromptRegistry, PromptTask

        bundle = PromptRegistry.get_bundle(PromptTask.VERIFY_DISEASE_BATCH)

        assert bundle.task == PromptTask.VERIFY_DISEASE_BATCH
        assert len(bundle.system_prompt) > 0
        assert len(bundle.user_template) > 0
        assert len(bundle.prompt_bundle_hash) > 0
        assert bundle.version == "v1.0"

    def test_get_bundle_abbreviation_batch_returns_different_prompts(self):
        from D_validation.D01_prompt_registry import PromptRegistry, PromptTask

        disease_bundle = PromptRegistry.get_bundle(PromptTask.VERIFY_DISEASE_BATCH)
        abbrev_bundle = PromptRegistry.get_bundle(PromptTask.VERIFY_BATCH)

        assert disease_bundle.system_prompt != abbrev_bundle.system_prompt
        assert disease_bundle.user_template != abbrev_bundle.user_template
        assert disease_bundle.prompt_bundle_hash != abbrev_bundle.prompt_bundle_hash

    def test_get_bundle_invalid_version_raises_value_error(self):
        from D_validation.D01_prompt_registry import PromptRegistry, PromptTask

        with pytest.raises(ValueError, match="Prompt template not found"):
            PromptRegistry.get_bundle(PromptTask.VERIFY_DISEASE, version="v99.0")


class TestQuoteVerifierBehavioral:
    """Behavioral tests for QuoteVerifier and NumericalVerifier."""

    def test_verify_quote_exact_match_in_clinical_text(self):
        from Z_utils.Z14_quote_verifier import verify_quote

        source_text = (
            "Patients with pulmonary arterial hypertension (PAH) "
            "were randomized to receive sildenafil 20 mg three times daily."
        )
        assert verify_quote("pulmonary arterial hypertension", source_text) is True

    def test_verify_quote_missing_substring_returns_false(self):
        from Z_utils.Z14_quote_verifier import verify_quote

        source_text = (
            "The study enrolled 350 patients with confirmed "
            "Duchenne muscular dystrophy across 12 clinical sites."
        )
        assert verify_quote("Becker muscular dystrophy", source_text) is False

    def test_verify_number_present_in_text(self):
        from Z_utils.Z14_quote_verifier import verify_number

        source_text = (
            "The primary endpoint was met with a response rate of 42.5% "
            "in the treatment arm versus 18.3% in the placebo arm."
        )
        assert verify_number(42.5, source_text) is True

    def test_verify_number_absent_from_text(self):
        from Z_utils.Z14_quote_verifier import verify_number

        source_text = (
            "The primary endpoint was met with a response rate of 42.5% "
            "in the treatment arm versus 18.3% in the placebo arm."
        )
        assert verify_number(99.9, source_text) is False

    def test_quote_verifier_class_provides_match_details(self):
        from Z_utils.Z14_quote_verifier import QuoteVerifier

        verifier = QuoteVerifier(fuzzy_threshold=0.90)
        source_text = (
            "Eligibility required eGFR above 30 mL/min/1.73m2 "
            "and hemoglobin A1c below 10%."
        )
        result = verifier.verify("eGFR above 30 mL/min/1.73m2", source_text)

        assert result.verified is True
        assert result.match_ratio == 1.0
        assert result.is_exact_match is True
        assert result.position is not None


class TestSpanDeduplicatorBehavioral:
    """Behavioral tests for NERSpan overlap and SpanDeduplicator."""

    def test_longer_span_wins_over_overlapping_shorter_span(self):
        from E_normalization.E11_span_deduplicator import NERSpan, SpanDeduplicator

        spans = [
            NERSpan(
                text="Type 2 Diabetes Mellitus",
                category="symptom",
                confidence=0.92,
                source="BiomedicalNER",
                start=10,
                end=34,
            ),
            NERSpan(
                text="Diabetes",
                category="symptom",
                confidence=0.85,
                source="EpiExtract4GARD-v2",
                start=17,
                end=25,
            ),
        ]

        deduplicator = SpanDeduplicator()
        result = deduplicator.deduplicate(spans)

        assert len(result.unique_spans) == 1
        assert result.unique_spans[0].text == "Type 2 Diabetes Mellitus"
        assert result.merged_count == 1
        assert result.total_input == 2

    def test_non_overlapping_spans_both_kept(self):
        from E_normalization.E11_span_deduplicator import NERSpan, SpanDeduplicator

        spans = [
            NERSpan(
                text="hypertension",
                category="symptom",
                confidence=0.90,
                source="BiomedicalNER",
                start=0,
                end=12,
            ),
            NERSpan(
                text="renal failure",
                category="symptom",
                confidence=0.88,
                source="BiomedicalNER",
                start=50,
                end=63,
            ),
        ]

        deduplicator = SpanDeduplicator()
        result = deduplicator.deduplicate(spans)

        assert len(result.unique_spans) == 2
        assert result.merged_count == 0
        texts = {s.text for s in result.unique_spans}
        assert texts == {"hypertension", "renal failure"}

    def test_empty_input_returns_empty_result(self):
        from E_normalization.E11_span_deduplicator import SpanDeduplicator

        deduplicator = SpanDeduplicator()
        result = deduplicator.deduplicate([])

        assert len(result.unique_spans) == 0
        assert result.merged_count == 0
        assert result.total_input == 0


class TestGoldLoaderBehavioral:
    """Behavioral tests for GoldLoader JSON loading."""

    def test_load_json_with_valid_annotations(self, tmp_path):
        import json

        from F_evaluation.F01_gold_loader import GoldLoader

        gold_data = {
            "annotations": [
                {
                    "doc_id": "rare_disease_protocol.pdf",
                    "short_form": "PAH",
                    "long_form": "Pulmonary Arterial Hypertension",
                },
                {
                    "doc_id": "rare_disease_protocol.pdf",
                    "short_form": "eGFR",
                    "long_form": "estimated Glomerular Filtration Rate",
                },
            ]
        }
        gold_file = tmp_path / "test_gold.json"
        gold_file.write_text(json.dumps(gold_data))

        loader = GoldLoader(strict=True)
        gold_standard, index = loader.load_json(str(gold_file))

        assert len(gold_standard.annotations) == 2
        assert "rare_disease_protocol.pdf" in index
        assert len(index["rare_disease_protocol.pdf"]) == 2

        short_forms = {a.short_form for a in gold_standard.annotations}
        assert short_forms == {"PAH", "EGFR"}

    def test_load_json_with_empty_annotations(self, tmp_path):
        import json

        from F_evaluation.F01_gold_loader import GoldLoader

        gold_data: dict[str, list[dict[str, str]]] = {"annotations": []}
        gold_file = tmp_path / "empty_gold.json"
        gold_file.write_text(json.dumps(gold_data))

        loader = GoldLoader(strict=True)
        gold_standard, index = loader.load_json(str(gold_file))

        assert len(gold_standard.annotations) == 0
        assert len(index) == 0

    def test_load_json_normalizes_doc_id_to_filename(self, tmp_path):
        import json

        from F_evaluation.F01_gold_loader import GoldLoader

        gold_data = {
            "annotations": [
                {
                    "doc_id": "/data/corpus/trials/NCT00123456.pdf",
                    "short_form": "AE",
                    "long_form": "Adverse Event",
                },
            ]
        }
        gold_file = tmp_path / "path_gold.json"
        gold_file.write_text(json.dumps(gold_data))

        loader = GoldLoader(strict=True)
        gold_standard, index = loader.load_json(str(gold_file))

        assert gold_standard.annotations[0].doc_id == "NCT00123456.pdf"
        assert "NCT00123456.pdf" in index


class TestScorerBehavioral:
    """Behavioral tests for Scorer precision/recall/F1 computation."""

    def _make_entity(self, doc_id, short_form, long_form):
        """Helper to create a minimal ExtractedEntity for scoring."""
        import uuid

        from A_core.A01_domain_models import (
            Coordinate,
            EvidenceSpan,
            ExtractedEntity,
            FieldType,
            GeneratorType,
            ProvenanceMetadata,
            ValidationStatus,
        )

        provenance = ProvenanceMetadata(
            pipeline_version="test",
            run_id="RUN_TEST",
            doc_fingerprint="abc123",
            generator_name=GeneratorType.LEXICON_MATCH,
        )
        evidence = EvidenceSpan(
            text=f"{short_form} ({long_form})",
            location=Coordinate(page_num=1),
            scope_ref="test_scope",
            start_char_offset=0,
            end_char_offset=len(f"{short_form} ({long_form})"),
        )
        return ExtractedEntity(
            candidate_id=uuid.uuid4(),
            doc_id=doc_id,
            field_type=FieldType.DEFINITION_PAIR,
            short_form=short_form,
            long_form=long_form,
            primary_evidence=evidence,
            status=ValidationStatus.VALIDATED,
            confidence_score=0.95,
            provenance=provenance,
        )

    def test_perfect_match_yields_full_precision_recall(self):
        from F_evaluation.F01_gold_loader import GoldAnnotation
        from F_evaluation.F02_scorer import Scorer, ScorerConfig

        doc_id = "cystic_fibrosis_trial.pdf"
        entities = [
            self._make_entity(doc_id, "CFTR", "cystic fibrosis transmembrane conductance regulator"),
            self._make_entity(doc_id, "FEV1", "forced expiratory volume in 1 second"),
        ]
        gold = [
            GoldAnnotation(
                doc_id=doc_id,
                short_form="CFTR",
                long_form="cystic fibrosis transmembrane conductance regulator",
            ),
            GoldAnnotation(
                doc_id=doc_id,
                short_form="FEV1",
                long_form="forced expiratory volume in 1 second",
            ),
        ]

        scorer = Scorer(ScorerConfig(require_long_form_match=True))
        report = scorer.evaluate_doc(entities, gold)

        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1 == 1.0
        assert report.true_positives == 2
        assert report.false_positives == 0
        assert report.false_negatives == 0

    def test_partial_match_computes_correct_metrics(self):
        from F_evaluation.F01_gold_loader import GoldAnnotation
        from F_evaluation.F02_scorer import Scorer, ScorerConfig

        doc_id = "oncology_protocol.pdf"
        entities = [
            self._make_entity(doc_id, "OS", "overall survival"),
            self._make_entity(doc_id, "PFS", "progression-free survival"),
            self._make_entity(doc_id, "DLT", "dose-limiting toxicity"),
        ]
        gold = [
            GoldAnnotation(doc_id=doc_id, short_form="OS", long_form="overall survival"),
            GoldAnnotation(doc_id=doc_id, short_form="PFS", long_form="progression-free survival"),
            GoldAnnotation(doc_id=doc_id, short_form="ORR", long_form="overall response rate"),
        ]

        scorer = Scorer(ScorerConfig(require_long_form_match=True))
        report = scorer.evaluate_doc(entities, gold)

        # TP=2 (OS, PFS), FP=1 (DLT), FN=1 (ORR)
        assert report.true_positives == 2
        assert report.false_positives == 1
        assert report.false_negatives == 1
        expected_precision = round(2 / 3, 4)
        expected_recall = round(2 / 3, 4)
        expected_f1 = round(2 * expected_precision * expected_recall / (expected_precision + expected_recall), 4)
        assert report.precision == expected_precision
        assert report.recall == expected_recall
        assert report.f1 == expected_f1

    def test_zero_predictions_handles_gracefully(self):
        from F_evaluation.F01_gold_loader import GoldAnnotation
        from F_evaluation.F02_scorer import Scorer, ScorerConfig

        doc_id = "empty_extraction.pdf"
        gold = [
            GoldAnnotation(doc_id=doc_id, short_form="TNF", long_form="tumor necrosis factor"),
            GoldAnnotation(doc_id=doc_id, short_form="IL6", long_form="interleukin 6"),
        ]

        scorer = Scorer(ScorerConfig(require_long_form_match=True))
        report = scorer.evaluate_doc([], gold)

        assert report.precision == 0.0
        assert report.recall == 0.0
        assert report.f1 == 0.0
        assert report.true_positives == 0
        assert report.false_positives == 0
        assert report.false_negatives == 2
