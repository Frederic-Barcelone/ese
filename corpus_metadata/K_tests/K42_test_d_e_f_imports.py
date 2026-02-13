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
    "D04_quote_verifier",
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
        from D_validation.D04_quote_verifier import (
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
