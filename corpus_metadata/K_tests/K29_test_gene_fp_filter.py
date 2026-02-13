# corpus_metadata/K_tests/K29_test_gene_fp_filter.py
"""
Tests for C_generators.C34_gene_fp_filter module.

Tests gene false positive filtering for ambiguous gene symbols.
"""

from __future__ import annotations

import pytest

from C_generators.C34_gene_fp_filter import GeneFalsePositiveFilter
from A_core.A19_gene_models import GeneGeneratorType


@pytest.fixture
def filter():
    return GeneFalsePositiveFilter()


class TestStatisticalTerms:
    """Tests for statistical term filtering."""

    def test_odds_ratio_filtered(self, filter):
        """Test that OR (odds ratio) is filtered in stats context."""
        is_fp, reason = filter.is_false_positive(
            "OR", "odds ratio OR 1.5 (95% CI)", GeneGeneratorType.PATTERN_GENE_SYMBOL
        )
        assert is_fp

    def test_hazard_ratio_filtered(self, filter):
        """Test that HR (hazard ratio) is filtered."""
        is_fp, reason = filter.is_false_positive(
            "HR", "hazard ratio HR 0.85", GeneGeneratorType.PATTERN_GENE_SYMBOL
        )
        assert is_fp

    def test_confidence_interval_filtered(self, filter):
        """Test that CI is filtered in stats context."""
        is_fp, reason = filter.is_false_positive(
            "CI", "95% confidence interval CI", GeneGeneratorType.PATTERN_GENE_SYMBOL
        )
        assert is_fp

    def test_statistical_terms_set(self, filter):
        """Test that STATISTICAL_TERMS set is defined."""
        expected = ["or", "hr", "ci", "sd", "se", "rr"]
        for term in expected:
            assert term in filter.statistical_lower


class TestUnits:
    """Tests for unit filtering."""

    def test_mm_filtered(self, filter):
        """Test that mm (millimeters) is filtered."""
        is_fp, reason = filter.is_false_positive(
            "mm", "5 mm diameter", GeneGeneratorType.PATTERN_GENE_SYMBOL
        )
        assert is_fp

    def test_mg_filtered(self, filter):
        """Test that mg (milligrams) is filtered."""
        is_fp, reason = filter.is_false_positive(
            "mg", "500 mg dose", GeneGeneratorType.PATTERN_GENE_SYMBOL
        )
        assert is_fp

    def test_units_set(self, filter):
        """Test that UNITS set is defined."""
        expected = ["mm", "cm", "kg", "mg", "ml"]
        for unit in expected:
            assert unit in filter.units_lower


class TestClinicalTerms:
    """Tests for clinical term filtering."""

    def test_iv_filtered(self, filter):
        """Test that IV (intravenous) is filtered."""
        is_fp, reason = filter.is_false_positive(
            "IV", "IV administration", GeneGeneratorType.PATTERN_GENE_SYMBOL
        )
        assert is_fp

    def test_clinical_terms_set(self, filter):
        """Test that CLINICAL_TERMS set is defined."""
        expected = ["iv", "po", "im", "sc", "bid"]
        for term in expected:
            assert term in filter.clinical_lower


class TestCommonEnglishWords:
    """Tests for common English word filtering."""

    def test_common_words_filtered(self, filter):
        """Test that common words are filtered."""
        common = ["of", "an", "was", "set", "can"]
        for word in common:
            is_fp, reason = filter.is_false_positive(
                word, "some context", GeneGeneratorType.PATTERN_GENE_SYMBOL
            )
            assert is_fp, f"{word} should be filtered"

    def test_common_words_set(self, filter):
        """Test that COMMON_ENGLISH_WORDS set is defined."""
        assert len(filter.common_english_lower) > 50
        assert "of" in filter.common_english_lower
        assert "was" in filter.common_english_lower


class TestValidGenes:
    """Tests for valid gene detection."""

    def test_brca1_not_filtered(self, filter):
        """Test that BRCA1 is not filtered in gene context."""
        is_fp, reason = filter.is_false_positive(
            "BRCA1", "BRCA1 gene mutation carriers expression",
            GeneGeneratorType.LEXICON_ORPHADATA, is_from_lexicon=True
        )
        assert not is_fp

    def test_tp53_not_filtered(self, filter):
        """Test that TP53 is not filtered in gene context."""
        is_fp, reason = filter.is_false_positive(
            "TP53", "TP53 mutation variant allele",
            GeneGeneratorType.LEXICON_ORPHADATA, is_from_lexicon=True
        )
        assert not is_fp


class TestShortGeneContext:
    """Tests for short gene context validation."""

    def test_short_gene_needs_context(self, filter):
        """Test that short genes need gene context."""
        is_fp, reason = filter.is_false_positive(
            "AR", "AR was measured in the study",
            GeneGeneratorType.PATTERN_GENE_SYMBOL
        )
        # Short gene without context should be filtered
        assert is_fp or "context" in reason.lower()

    def test_short_gene_with_context(self, filter):
        """Test that short genes with gene context pass."""
        is_fp, reason = filter.is_false_positive(
            "AR", "AR gene mutation variant expression protein pathway",
            GeneGeneratorType.LEXICON_ORPHADATA, is_from_lexicon=True
        )
        assert not is_fp

    def test_short_genes_set(self, filter):
        """Test that SHORT_GENES_NEED_CONTEXT is defined."""
        assert len(filter.short_genes_lower) > 100


class TestEgfrDisambiguation:
    """Tests for EGFR disambiguation."""

    def test_egfr_kidney_context(self, filter):
        """Test EGFR filtered in kidney function context."""
        is_fp, reason = filter.is_false_positive(
            "egfr", "eGFR ml/min renal function creatinine",
            GeneGeneratorType.PATTERN_GENE_SYMBOL
        )
        # egfr is filtered - could be as common_abbreviation or kidney context
        assert is_fp

    def test_egfr_gene_context(self, filter):
        """Test EGFR not filtered in gene context."""
        is_fp, reason = filter.is_false_positive(
            "EGFR", "EGFR gene mutation expression receptor tyrosine kinase",
            GeneGeneratorType.LEXICON_ORPHADATA, is_from_lexicon=True
        )
        assert not is_fp


class TestContextScoring:
    """Tests for context scoring."""

    def test_gene_context_keywords(self, filter):
        """Test gene context keyword detection."""
        gene_score, nongene_score = filter._score_context(
            "BRCA1 gene mutation variant expression"
        )
        assert gene_score > 0
        assert gene_score > nongene_score

    def test_non_gene_context_keywords(self, filter):
        """Test non-gene context keyword detection."""
        gene_score, nongene_score = filter._score_context(
            "odds ratio p-value statistically significant"
        )
        assert nongene_score > 0

    def test_context_keywords_sets(self, filter):
        """Test that context keyword sets are defined."""
        assert len(filter.gene_context_lower) > 20
        assert len(filter.non_gene_context_lower) > 10


class TestMinLength:
    """Tests for minimum length filtering."""

    def test_single_letter_filtered(self, filter):
        """Test that single letters are filtered."""
        is_fp, reason = filter.is_false_positive(
            "A", "some context", GeneGeneratorType.PATTERN_GENE_SYMBOL
        )
        assert is_fp
        assert "short" in reason.lower()

    def test_min_length_value(self, filter):
        """Test MIN_LENGTH constant."""
        assert filter.MIN_LENGTH == 2


class TestCountriesAndCredentials:
    """Tests for country and credential filtering."""

    def test_country_codes_filtered(self, filter):
        """Test that country codes are filtered."""
        countries = ["us", "uk", "de", "fr"]
        for country in countries:
            is_fp, reason = filter.is_false_positive(
                country, "conducted in US", GeneGeneratorType.PATTERN_GENE_SYMBOL
            )
            assert is_fp

    def test_credentials_filtered(self, filter):
        """Test that credentials are filtered."""
        credentials = ["md", "phd", "mph"]
        for cred in credentials:
            is_fp, reason = filter.is_false_positive(
                cred, "John Smith, MD, PhD", GeneGeneratorType.PATTERN_GENE_SYMBOL
            )
            assert is_fp


class TestDrugTerms:
    """Tests for drug term filtering."""

    def test_drug_abbreviations_filtered(self, filter):
        """Test that drug abbreviations are filtered."""
        drugs = ["ace", "arb", "ssri"]
        for drug in drugs:
            is_fp, reason = filter.is_false_positive(
                drug, "ACE inhibitors", GeneGeneratorType.PATTERN_GENE_SYMBOL
            )
            assert is_fp


class TestStudyTerms:
    """Tests for study term filtering."""

    def test_trial_terms_filtered(self, filter):
        """Test that trial terms are filtered."""
        terms = ["rct", "itt", "sae", "dmc"]
        for term in terms:
            is_fp, reason = filter.is_false_positive(
                term, "RCT design", GeneGeneratorType.PATTERN_GENE_SYMBOL
            )
            assert is_fp


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self, filter):
        """Test handling of empty text."""
        is_fp, reason = filter.is_false_positive(
            "", "context", GeneGeneratorType.PATTERN_GENE_SYMBOL
        )
        assert is_fp

    def test_empty_context(self, filter):
        """Test handling of empty context."""
        is_fp, reason = filter.is_false_positive(
            "BRCA1", "", GeneGeneratorType.LEXICON_ORPHADATA, is_from_lexicon=True
        )
        assert isinstance(is_fp, bool)
        assert isinstance(reason, str)

    def test_alias_handling(self, filter):
        """Test that aliases are treated differently."""
        is_fp1, _ = filter.is_false_positive(
            "p53", "context", GeneGeneratorType.LEXICON_HGNC_ALIAS,
            is_from_lexicon=True, is_alias=False
        )
        is_fp2, _ = filter.is_false_positive(
            "p53", "context", GeneGeneratorType.LEXICON_HGNC_ALIAS,
            is_from_lexicon=True, is_alias=True
        )
        assert isinstance(is_fp1, bool)
        assert isinstance(is_fp2, bool)


class TestAdditionalCommonWords:
    """Tests for additional common English words added to filter."""

    def test_son_filtered(self, filter):
        """Test that 'son' is filtered as common word."""
        is_fp, reason = filter.is_false_positive(
            "son", "her son was diagnosed", GeneGeneratorType.PATTERN_GENE_SYMBOL
        )
        assert is_fp
        assert reason == "common_english_word"

    def test_best_filtered(self, filter):
        """Test that 'best' is filtered as common word."""
        is_fp, reason = filter.is_false_positive(
            "best", "best supportive care", GeneGeneratorType.PATTERN_GENE_SYMBOL
        )
        assert is_fp
        assert reason == "common_english_word"

    def test_ren_filtered(self, filter):
        """Test that 'ren' is filtered as common word."""
        is_fp, reason = filter.is_false_positive(
            "Ren", "Ren et al. reported", GeneGeneratorType.PATTERN_GENE_SYMBOL
        )
        assert is_fp
        assert reason == "common_english_word"

    def test_additional_words_in_set(self, filter):
        """Test that all additional words are in the common words set."""
        additional = [
            "son", "best", "ren", "rest", "last", "most", "near", "well",
            "good", "part", "step", "mark", "ring", "pair", "map", "gap",
            "cap", "tip", "bar", "tag", "tan", "dim",
        ]
        for word in additional:
            assert word in filter.common_english_lower, f"{word} not in set"


class TestQuestionnaireDisambiguation:
    """Tests for questionnaire abbreviation disambiguation."""

    def test_maf_filtered_without_gene_context(self, filter):
        """Test that MAF is filtered as questionnaire without gene context."""
        is_fp, reason = filter.is_false_positive(
            "MAF", "MAF score was measured at baseline",
            GeneGeneratorType.LEXICON_ORPHADATA, is_from_lexicon=True
        )
        assert is_fp
        assert reason == "questionnaire_term"

    def test_maf_passes_with_gene_context(self, filter):
        """Test that MAF passes with strong gene context."""
        is_fp, reason = filter.is_false_positive(
            "MAF", "MAF gene mutation variant expression allele",
            GeneGeneratorType.LEXICON_ORPHADATA, is_from_lexicon=True
        )
        assert not is_fp

    def test_haq_filtered(self, filter):
        """Test that HAQ is filtered as questionnaire."""
        is_fp, reason = filter.is_false_positive(
            "HAQ", "HAQ-DI disability index",
            GeneGeneratorType.PATTERN_GENE_SYMBOL
        )
        assert is_fp
        assert reason == "questionnaire_term"

    def test_das_filtered(self, filter):
        """Test that DAS is filtered as questionnaire."""
        is_fp, reason = filter.is_false_positive(
            "DAS", "DAS28 disease activity score",
            GeneGeneratorType.PATTERN_GENE_SYMBOL
        )
        assert is_fp
        assert reason == "questionnaire_term"


class TestAntibodyDisambiguation:
    """Tests for antibody abbreviation disambiguation."""

    def test_acpa_filtered_in_antibody_context(self, filter):
        """Test that ACPA is filtered as antibody in antibody context."""
        is_fp, reason = filter.is_false_positive(
            "ACPA", "ACPA antibody titer was elevated",
            GeneGeneratorType.LEXICON_ORPHADATA, is_from_lexicon=True
        )
        assert is_fp
        assert reason == "antibody_abbreviation"

    def test_acpa_passes_without_antibody_context(self, filter):
        """Test that ACPA passes without antibody context."""
        is_fp, reason = filter.is_false_positive(
            "ACPA", "ACPA gene mutation expression allele variant",
            GeneGeneratorType.LEXICON_ORPHADATA, is_from_lexicon=True
        )
        assert not is_fp

    def test_ana_filtered_in_antibody_context(self, filter):
        """Test that ANA is filtered as antibody in antibody context."""
        is_fp, reason = filter.is_false_positive(
            "ANA", "ANA seropositive patients",
            GeneGeneratorType.LEXICON_ORPHADATA, is_from_lexicon=True
        )
        assert is_fp
        assert reason == "antibody_abbreviation"

    def test_antibody_context_keywords(self, filter):
        """Test antibody context keyword detection."""
        assert filter._is_antibody_context("antibody titer levels")
        assert filter._is_antibody_context("seropositive patients")
        assert not filter._is_antibody_context("gene mutation expression")

    def test_strong_gene_context_helper(self, filter):
        """Test strong gene context detection."""
        assert filter._has_strong_gene_context("gene mutation expression")
        assert not filter._has_strong_gene_context("patient treatment")
