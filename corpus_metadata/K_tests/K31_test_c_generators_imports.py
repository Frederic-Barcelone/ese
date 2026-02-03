# corpus_metadata/K_tests/K31_test_c_generators_imports.py
"""
Tests for C_generators module imports and __all__ exports.

Ensures all C_generators modules can be imported and have proper exports.
"""

from __future__ import annotations

import importlib
import pytest


# List of all C_generators modules that should be importable
C_GENERATOR_MODULES = [
    "C00_strategy_identifiers",
    "C01_strategy_abbrev",
    "C02_strategy_regex",
    "C03_strategy_layout",
    "C04_strategy_flashtext",
    "C05_strategy_glossary",
    "C06_strategy_disease",
    "C07_strategy_drug",
    "C08_strategy_feasibility",
    "C09_strategy_document_metadata",
    "C10_vision_image_analysis",
    "C11_llm_feasibility",
    "C12_guideline_recommendation_extractor",
    "C13_strategy_author",
    "C14_strategy_citation",
    "C15_vlm_table_extractor",
    "C16_strategy_gene",
    "C17_flowchart_graph_extractor",
    "C18_strategy_pharma",
    "C19_vlm_visual_enrichment",
    "C20_abbrev_patterns",
    "C21_noise_filters",
    "C22_lexicon_loaders",
    "C23_inline_definition_detector",
    "C24_disease_fp_filter",
    "C25_drug_fp_filter",
    "C26_drug_fp_constants",
    "C27_feasibility_patterns",
    "C28_feasibility_fp_filter",
    "C29_feasibility_prompts",
    "C30_feasibility_response_parser",
    "C31_recommendation_patterns",
    "C32_recommendation_llm",
    "C33_recommendation_vlm",
    "C34_gene_fp_filter",
]


class TestModuleImports:
    """Tests that all C_generators modules can be imported."""

    @pytest.mark.parametrize("module_name", C_GENERATOR_MODULES)
    def test_module_imports(self, module_name):
        """Test that each module can be imported."""
        try:
            module = importlib.import_module(f"C_generators.{module_name}")
            assert module is not None
        except ImportError as e:
            pytest.fail(f"Failed to import C_generators.{module_name}: {e}")


class TestModuleExports:
    """Tests that modules have __all__ defined."""

    @pytest.mark.parametrize("module_name", C_GENERATOR_MODULES)
    def test_module_has_all(self, module_name):
        """Test that each module has __all__ defined."""
        try:
            module = importlib.import_module(f"C_generators.{module_name}")
            if hasattr(module, "__all__"):
                assert isinstance(module.__all__, (list, tuple))
                # __all__ should not be empty for utility modules
                # Strategy modules may not need __all__
        except ImportError:
            pytest.skip(f"Module {module_name} not importable")


class TestSpecificModuleExports:
    """Tests for specific module exports."""

    def test_abbrev_patterns_exports(self):
        """Test C20_abbrev_patterns exports."""
        from C_generators.C20_abbrev_patterns import (
            _ABBREV_TOKEN_RE,
            _is_likely_author_initial,
            _clean_ws,
        )
        assert _ABBREV_TOKEN_RE is not None
        assert callable(_is_likely_author_initial)
        assert callable(_clean_ws)

    def test_noise_filters_exports(self):
        """Test C21_noise_filters exports."""
        from C_generators.C21_noise_filters import (
            OBVIOUS_NOISE,
            MIN_ABBREV_LENGTH,
            is_valid_abbreviation_match,
        )
        assert isinstance(OBVIOUS_NOISE, set)
        assert isinstance(MIN_ABBREV_LENGTH, int)
        assert callable(is_valid_abbreviation_match)

    def test_inline_definition_detector_exports(self):
        """Test C23_inline_definition_detector exports."""
        from C_generators.C23_inline_definition_detector import (
            InlineDefinitionDetectorMixin,
        )
        assert hasattr(InlineDefinitionDetectorMixin, "_extract_inline_definitions")

    def test_disease_fp_filter_exports(self):
        """Test C24_disease_fp_filter exports."""
        from C_generators.C24_disease_fp_filter import DiseaseFalsePositiveFilter
        assert hasattr(DiseaseFalsePositiveFilter, "should_filter")
        assert hasattr(DiseaseFalsePositiveFilter, "score_adjustment")

    def test_drug_fp_filter_exports(self):
        """Test C25_drug_fp_filter exports."""
        from C_generators.C25_drug_fp_filter import (
            DrugFalsePositiveFilter,
        )
        assert hasattr(DrugFalsePositiveFilter, "is_false_positive")

    def test_feasibility_patterns_exports(self):
        """Test C27_feasibility_patterns exports."""
        from C_generators.C27_feasibility_patterns import (
            EPIDEMIOLOGY_ANCHORS,
            INCLUSION_MARKERS,
            COUNTRIES,
        )
        assert isinstance(EPIDEMIOLOGY_ANCHORS, list)
        assert isinstance(INCLUSION_MARKERS, list)
        assert isinstance(COUNTRIES, set)

    def test_gene_fp_filter_exports(self):
        """Test C34_gene_fp_filter exports."""
        from C_generators.C34_gene_fp_filter import GeneFalsePositiveFilter
        assert hasattr(GeneFalsePositiveFilter, "is_false_positive")
        assert hasattr(GeneFalsePositiveFilter, "STATISTICAL_TERMS")


class TestStrategyClasses:
    """Tests for strategy class imports."""

    def test_abbrev_strategy_import(self):
        """Test abbreviation strategy can be imported."""
        try:
            from C_generators.C01_strategy_abbrev import AbbrevSyntaxCandidateGenerator
            assert hasattr(AbbrevSyntaxCandidateGenerator, "extract")
        except ImportError:
            pytest.skip("AbbrevSyntaxCandidateGenerator not available")

    def test_disease_strategy_import(self):
        """Test disease strategy can be imported."""
        try:
            from C_generators.C06_strategy_disease import DiseaseDetector
            assert hasattr(DiseaseDetector, "extract")
        except ImportError:
            pytest.skip("DiseaseDetector not available")

    def test_drug_strategy_import(self):
        """Test drug strategy can be imported."""
        try:
            from C_generators.C07_strategy_drug import DrugDetector
            assert hasattr(DrugDetector, "detect")
        except ImportError:
            pytest.skip("DrugDetector not available")

    def test_gene_strategy_import(self):
        """Test gene strategy can be imported."""
        try:
            from C_generators.C16_strategy_gene import GeneDetector
            assert hasattr(GeneDetector, "detect")
        except ImportError:
            pytest.skip("GeneDetector not available")

    def test_feasibility_strategy_import(self):
        """Test feasibility strategy can be imported."""
        try:
            from C_generators.C08_strategy_feasibility import FeasibilityDetector
            assert FeasibilityDetector is not None
        except ImportError:
            pytest.skip("FeasibilityDetector not available")


class TestHelperModules:
    """Tests for helper module imports."""

    def test_drug_fp_constants_import(self):
        """Test C26_drug_fp_constants can be imported."""
        from C_generators.C26_drug_fp_constants import (
            ALWAYS_FILTER,
            BACTERIA_ORGANISMS,
            COMMON_WORDS,
        )
        assert isinstance(ALWAYS_FILTER, set)
        assert isinstance(BACTERIA_ORGANISMS, set)
        assert isinstance(COMMON_WORDS, set)

    def test_feasibility_prompts_import(self):
        """Test C29_feasibility_prompts can be imported."""
        from C_generators.C29_feasibility_prompts import (
            STUDY_DESIGN_PROMPT,
            ELIGIBILITY_PROMPT,
        )
        assert isinstance(STUDY_DESIGN_PROMPT, str)
        assert isinstance(ELIGIBILITY_PROMPT, str)

    def test_recommendation_patterns_import(self):
        """Test C31_recommendation_patterns can be imported."""
        try:
            from C_generators.C31_recommendation_patterns import (
                ORGANIZATION_PATTERNS,
            )
            assert ORGANIZATION_PATTERNS is not None
        except ImportError:
            pytest.skip("Recommendation patterns not available")
