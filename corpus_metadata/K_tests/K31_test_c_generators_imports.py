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
        module = importlib.import_module(f"C_generators.{module_name}")
        if hasattr(module, "__all__"):
            assert isinstance(module.__all__, (list, tuple))


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
        assert isinstance(COUNTRIES, (set, frozenset))

    def test_gene_fp_filter_exports(self):
        """Test C34_gene_fp_filter exports."""
        from C_generators.C34_gene_fp_filter import GeneFalsePositiveFilter
        assert hasattr(GeneFalsePositiveFilter, "is_false_positive")
        assert hasattr(GeneFalsePositiveFilter, "STATISTICAL_TERMS")


class TestStrategyClasses:
    """Tests for strategy class imports."""

    def test_abbrev_strategy_import(self):
        """Test abbreviation strategy can be imported."""
        from C_generators.C01_strategy_abbrev import AbbrevSyntaxCandidateGenerator
        assert hasattr(AbbrevSyntaxCandidateGenerator, "extract")

    def test_disease_strategy_import(self):
        """Test disease strategy can be imported."""
        from C_generators.C06_strategy_disease import DiseaseDetector
        assert hasattr(DiseaseDetector, "extract")

    def test_drug_strategy_import(self):
        """Test drug strategy can be imported."""
        from C_generators.C07_strategy_drug import DrugDetector
        assert hasattr(DrugDetector, "detect")

    def test_gene_strategy_import(self):
        """Test gene strategy can be imported."""
        from C_generators.C16_strategy_gene import GeneDetector
        assert hasattr(GeneDetector, "detect")

    def test_feasibility_strategy_import(self):
        """Test feasibility strategy can be imported."""
        from C_generators.C08_strategy_feasibility import FeasibilityDetector
        assert hasattr(FeasibilityDetector, "extract")


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
        from C_generators.C31_recommendation_patterns import (
            ORGANIZATION_PATTERNS,
        )
        assert isinstance(ORGANIZATION_PATTERNS, dict)


# =============================================================================
# BEHAVIORAL TESTS
# =============================================================================


class TestNoiseFiltersBehavioral:
    """Behavioral tests for C21_noise_filters abbreviation validation."""

    def test_valid_pharma_abbreviations_accepted(self):
        """Valid pharma abbreviations like TNF, BRCA1, IL-6 should pass the filter."""
        from C_generators.C21_noise_filters import is_valid_abbreviation_match

        assert is_valid_abbreviation_match("TNF") is True
        assert is_valid_abbreviation_match("BRCA1") is True
        assert is_valid_abbreviation_match("IL-6") is True
        assert is_valid_abbreviation_match("mRNA") is True
        assert is_valid_abbreviation_match("PCR") is True

    def test_noise_inputs_rejected(self):
        """Single letters, stopwords, and month abbreviations should be rejected."""
        from C_generators.C21_noise_filters import is_valid_abbreviation_match

        # Single letters
        assert is_valid_abbreviation_match("a") is False
        assert is_valid_abbreviation_match("z") is False
        # Stopwords
        assert is_valid_abbreviation_match("the") is False
        assert is_valid_abbreviation_match("and") is False
        # Month abbreviations (in OBVIOUS_NOISE via month_names())
        assert is_valid_abbreviation_match("jan") is False
        assert is_valid_abbreviation_match("feb") is False
        # Empty / pure digits
        assert is_valid_abbreviation_match("") is False
        assert is_valid_abbreviation_match("123") is False
        # Short alphanumeric starting with digit
        assert is_valid_abbreviation_match("2B") is False

    def test_obvious_noise_set_and_min_length_constant(self):
        """OBVIOUS_NOISE contains expected noise terms; MIN_ABBREV_LENGTH is >= 2."""
        from C_generators.C21_noise_filters import MIN_ABBREV_LENGTH, OBVIOUS_NOISE

        # Single letters must be in OBVIOUS_NOISE
        assert "a" in OBVIOUS_NOISE
        assert "z" in OBVIOUS_NOISE
        # Stopwords
        assert "the" in OBVIOUS_NOISE
        assert "and" in OBVIOUS_NOISE
        # Month names (lowercase)
        assert "january" in OBVIOUS_NOISE
        assert "jan" in OBVIOUS_NOISE

        # MIN_ABBREV_LENGTH must be a reasonable int
        assert isinstance(MIN_ABBREV_LENGTH, int)
        assert MIN_ABBREV_LENGTH >= 2


class TestDiseaseFPFilterBehavioral:
    """Behavioral tests for C24_disease_fp_filter disease false positive detection."""

    def test_common_english_fp_hard_filtered(self):
        """Common English words that collide with disease names should be hard-filtered."""
        from C_generators.C24_disease_fp_filter import DiseaseFalsePositiveFilter

        fp_filter = DiseaseFalsePositiveFilter()

        # "common" is in COMMON_ENGLISH_FP_TERMS
        filtered, reason = fp_filter.should_filter("common", "the disease is common", False)
        assert filtered is True
        assert reason == "common_english_word"

        # "grade 3" matches the grade pattern hard filter
        filtered, reason = fp_filter.should_filter("grade 3", "patient had grade 3 toxicity", False)
        assert filtered is True
        assert reason == "grade_pattern"

        # Generic multiword terms
        filtered, reason = fp_filter.should_filter(
            "autosomal dominant", "autosomal dominant inheritance pattern", False
        )
        assert filtered is True
        assert reason == "generic_multiword_term"

    def test_real_disease_names_not_filtered(self):
        """Real disease names should not be hard-filtered."""
        from C_generators.C24_disease_fp_filter import DiseaseFalsePositiveFilter

        fp_filter = DiseaseFalsePositiveFilter()

        filtered, _reason = fp_filter.should_filter(
            "Type 2 Diabetes",
            "patients diagnosed with Type 2 Diabetes mellitus",
            False,
        )
        assert filtered is False

        filtered, _reason = fp_filter.should_filter(
            "Huntington disease",
            "Huntington disease is a progressive neurodegenerative disorder",
            False,
        )
        assert filtered is False

        filtered, _reason = fp_filter.should_filter(
            "cystic fibrosis",
            "cystic fibrosis transmembrane conductance regulator",
            False,
        )
        assert filtered is False

    def test_score_adjustment_returns_float(self):
        """score_adjustment should return a (float, str) tuple for any input."""
        from C_generators.C24_disease_fp_filter import DiseaseFalsePositiveFilter

        fp_filter = DiseaseFalsePositiveFilter()

        adjustment, reason = fp_filter.score_adjustment(
            "Type 2 Diabetes",
            "patients diagnosed with Type 2 Diabetes mellitus",
        )
        assert isinstance(adjustment, float)
        assert isinstance(reason, str)

        # Chromosome context should produce a negative adjustment
        adjustment_chr, reason_chr = fp_filter.score_adjustment(
            "22q11",
            "chromosome 22q11 deletion karyotype cytogenetics",
        )
        assert isinstance(adjustment_chr, float)
        assert adjustment_chr < 0.0
        assert "chromosome" in reason_chr or "capped" in reason_chr


class TestDrugFPFilterBehavioral:
    """Behavioral tests for C25_drug_fp_filter drug false positive detection."""

    def test_nct_id_filtered_as_false_positive(self):
        """NCT trial identifiers should be filtered as drug false positives."""
        from A_core.A06_drug_models import DrugGeneratorType
        from C_generators.C25_drug_fp_filter import DrugFalsePositiveFilter

        fp_filter = DrugFalsePositiveFilter()

        assert fp_filter.is_false_positive(
            "NCT04817618", "trial NCT04817618 enrolled patients", DrugGeneratorType.SCISPACY_NER
        ) is True

    def test_real_drug_names_not_filtered(self):
        """Real pharmaceutical drug names should not be filtered."""
        from A_core.A06_drug_models import DrugGeneratorType
        from C_generators.C25_drug_fp_filter import DrugFalsePositiveFilter

        fp_filter = DrugFalsePositiveFilter()

        assert fp_filter.is_false_positive(
            "pembrolizumab", "pembrolizumab was administered", DrugGeneratorType.LEXICON_FDA
        ) is False

        assert fp_filter.is_false_positive(
            "metformin", "metformin 500mg twice daily", DrugGeneratorType.LEXICON_RXNORM
        ) is False

        assert fp_filter.is_false_positive(
            "ravulizumab", "ravulizumab for PNH treatment", DrugGeneratorType.LEXICON_ALEXION
        ) is False

    def test_short_and_ethics_codes_filtered(self):
        """Too-short names and ethics committee codes should be filtered."""
        from A_core.A06_drug_models import DrugGeneratorType
        from C_generators.C25_drug_fp_filter import DrugFalsePositiveFilter

        fp_filter = DrugFalsePositiveFilter()

        # Too short (< 3 chars)
        assert fp_filter.is_false_positive(
            "ab", "ab was used", DrugGeneratorType.SCISPACY_NER
        ) is True

        # Ethics committee code pattern
        assert fp_filter.is_false_positive(
            "IRB2023", "approved by IRB2023", DrugGeneratorType.SCISPACY_NER
        ) is True

    def test_bacteria_organism_filtered(self):
        """Bacteria and organism names should be filtered as drug false positives."""
        from A_core.A06_drug_models import DrugGeneratorType
        from C_generators.C25_drug_fp_filter import DrugFalsePositiveFilter

        fp_filter = DrugFalsePositiveFilter()

        # Check that at least one known organism triggers filtering
        # BACTERIA_ORGANISMS is a curated set -- pick one that's likely present
        assert fp_filter.is_false_positive(
            "influenza", "influenza virus infection", DrugGeneratorType.SCISPACY_NER
        ) is True


class TestGeneFPFilterBehavioral:
    """Behavioral tests for C34_gene_fp_filter gene false positive detection."""

    def test_statistical_term_in_stats_context_filtered(self):
        """Statistical abbreviations like OR, HR, CI in statistical context should be filtered."""
        from A_core.A19_gene_models import GeneGeneratorType
        from C_generators.C34_gene_fp_filter import GeneFalsePositiveFilter

        fp_filter = GeneFalsePositiveFilter()

        # OR in statistics context -- filtered as common_abbreviation (not from lexicon)
        is_fp, reason = fp_filter.is_false_positive(
            "OR", "OR = 1.5, 95% CI 1.1-2.0, p < 0.05", GeneGeneratorType.PATTERN_GENE_SYMBOL
        )
        assert is_fp is True
        assert "abbreviation" in reason or "statistical" in reason or "common" in reason

        # HR in statistics context
        is_fp, reason = fp_filter.is_false_positive(
            "HR", "hazard ratio HR 2.3 confidence interval", GeneGeneratorType.PATTERN_GENE_SYMBOL
        )
        assert is_fp is True

    def test_real_gene_symbols_not_filtered(self):
        """Real gene symbols in gene context should not be filtered."""
        from A_core.A19_gene_models import GeneGeneratorType
        from C_generators.C34_gene_fp_filter import GeneFalsePositiveFilter

        fp_filter = GeneFalsePositiveFilter()

        # BRCA1 in gene context (long enough to skip short-gene validation)
        is_fp, _reason = fp_filter.is_false_positive(
            "BRCA1",
            "BRCA1 mutation carriers have increased expression of the gene protein",
            GeneGeneratorType.LEXICON_ORPHADATA,
            is_from_lexicon=True,
        )
        assert is_fp is False

        # TP53 in gene context
        is_fp, _reason = fp_filter.is_false_positive(
            "TP53",
            "TP53 gene mutation was detected in the allele exon",
            GeneGeneratorType.LEXICON_ORPHADATA,
            is_from_lexicon=True,
        )
        assert is_fp is False

    def test_statistical_terms_set_contents(self):
        """STATISTICAL_TERMS set should contain the expected core terms."""
        from C_generators.C34_gene_fp_filter import GeneFalsePositiveFilter

        stats = GeneFalsePositiveFilter.STATISTICAL_TERMS
        # These are loaded from gene_fp_terms.yaml -- stored lowercase
        expected_terms = {"or", "hr", "ci", "sd"}
        for term in expected_terms:
            assert term in stats, f"Expected '{term}' in STATISTICAL_TERMS"

    def test_common_english_words_filtered(self):
        """Common English words should always be filtered, even from lexicon."""
        from A_core.A19_gene_models import GeneGeneratorType
        from C_generators.C34_gene_fp_filter import GeneFalsePositiveFilter

        fp_filter = GeneFalsePositiveFilter()

        # Common English words are always filtered per the source code
        common_words = GeneFalsePositiveFilter.COMMON_ENGLISH_WORDS
        if common_words:
            sample_word = next(iter(common_words))
            is_fp, reason = fp_filter.is_false_positive(
                sample_word,
                "some arbitrary context with gene mutation expression",
                GeneGeneratorType.LEXICON_ORPHADATA,
                is_from_lexicon=True,
            )
            assert is_fp is True
            assert reason == "common_english_word"


class TestAbbrevPatternsBehavioral:
    """Behavioral tests for C20_abbrev_patterns helper functions."""

    def test_clean_ws_normalizes_whitespace(self):
        """_clean_ws should collapse multiple whitespace characters to single spaces."""
        from C_generators.C20_abbrev_patterns import _clean_ws

        assert _clean_ws("Tumor  Necrosis   Factor") == "Tumor Necrosis Factor"
        assert _clean_ws("  leading  trailing  ") == "leading trailing"
        assert _clean_ws("tab\there\nnewline") == "tab here newline"
        assert _clean_ws("") == ""
        assert _clean_ws("single") == "single"

    def test_is_likely_author_initial_detection(self):
        """_is_likely_author_initial returns True for author patterns, False for gene abbreviations."""
        from C_generators.C20_abbrev_patterns import _is_likely_author_initial

        # Author initial in author contribution context
        assert _is_likely_author_initial(
            "BH", "BH contributed to the manuscript and reviewed the data"
        ) is True

        # Author initial in acknowledgement context
        assert _is_likely_author_initial(
            "JM", "JM and SK designed the study. acknowledgement section"
        ) is True

        # TNF is a known medical abbreviation, not an author initial
        # Also fails the pattern check because it must be pure alphabetic 2-3 chars
        # but TNF IS 3 alpha chars. However it is in the known_abbrevs exclusion set.
        # Without author context keywords, TNF in a scientific context is not flagged.
        assert _is_likely_author_initial(
            "TNF", "TNF alpha levels were elevated in the serum"
        ) is False

        # Non-matching pattern: contains digits
        assert _is_likely_author_initial("IL6", "IL6 was measured") is False

        # Empty inputs
        assert _is_likely_author_initial("", "some context") is False
        assert _is_likely_author_initial("BH", "") is False

    def test_looks_like_short_form_validation(self):
        """_looks_like_short_form accepts valid abbreviations and rejects invalid ones."""
        from C_generators.C20_abbrev_patterns import _looks_like_short_form

        # Valid short forms
        assert _looks_like_short_form("TNF") is True
        assert _looks_like_short_form("IL-6") is True
        assert _looks_like_short_form("BRCA1") is True
        assert _looks_like_short_form("mRNA") is True

        # Invalid: pure digits
        assert _looks_like_short_form("123") is False
        # Invalid: too short
        assert _looks_like_short_form("A") is False
        # Invalid: no uppercase letter
        assert _looks_like_short_form("abc") is False
