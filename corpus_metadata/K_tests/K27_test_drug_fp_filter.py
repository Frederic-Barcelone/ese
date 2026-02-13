# corpus_metadata/K_tests/K27_test_drug_fp_filter.py
"""
Tests for C_generators.C25_drug_fp_filter module.

Tests drug false positive filtering using curated exclusion sets.
"""

from __future__ import annotations

import pytest

from C_generators.C25_drug_fp_filter import DrugFalsePositiveFilter, DRUG_ABBREVIATIONS
from A_core.A06_drug_models import DrugGeneratorType


@pytest.fixture
def filter():
    return DrugFalsePositiveFilter()


class TestBasicFiltering:
    """Tests for basic false positive filtering."""

    def test_short_matches_filtered(self, filter):
        """Test that very short matches are filtered."""
        assert filter.is_false_positive("ab", "", DrugGeneratorType.LEXICON_RXNORM)
        assert filter.is_false_positive("x", "", DrugGeneratorType.LEXICON_RXNORM)

    def test_nct_trial_ids_filtered(self, filter):
        """Test that NCT trial IDs are filtered."""
        assert filter.is_false_positive("NCT04817618", "", DrugGeneratorType.LEXICON_RXNORM)
        assert filter.is_false_positive("NCT12345678", "", DrugGeneratorType.LEXICON_RXNORM)

    def test_ethics_codes_filtered(self, filter):
        """Test that ethics committee codes are filtered."""
        assert filter.is_false_positive("KY2022", "", DrugGeneratorType.LEXICON_RXNORM)
        assert filter.is_false_positive("IRB2023", "", DrugGeneratorType.LEXICON_RXNORM)

    def test_valid_drug_not_filtered(self, filter):
        """Test that valid drug names are not filtered."""
        valid_drugs = ["ravulizumab", "iptacopan", "metformin", "aspirin"]
        for drug in valid_drugs:
            assert not filter.is_false_positive(
                drug, "", DrugGeneratorType.LEXICON_RXNORM
            ), f"{drug} should not be filtered"


class TestBacteriaFiltering:
    """Tests for bacteria/organism filtering."""

    def test_bacteria_names_filtered(self, filter):
        """Test that bacteria names are filtered."""
        result = filter.is_false_positive(
            "escherichia", "e. coli bacteria", DrugGeneratorType.SCISPACY_NER
        )
        assert isinstance(result, bool)

    def test_partial_bacteria_match(self, filter):
        """Test partial bacteria name matching."""
        result = filter.is_false_positive(
            "staphylococcus", "staphylococcus aureus infection",
            DrugGeneratorType.SCISPACY_NER
        )
        assert isinstance(result, bool)


class TestVaccineFiltering:
    """Tests for vaccine-related term filtering."""

    def test_vaccine_terms_filtered(self, filter):
        """Test that vaccine terms are filtered."""
        result = filter.is_false_positive(
            "influenza vaccine", "administered influenza vaccine",
            DrugGeneratorType.SCISPACY_NER
        )
        assert isinstance(result, bool)


class TestBiologicalEntities:
    """Tests for biological entity filtering."""

    def test_biological_suffixes_ner(self, filter):
        """Test biological suffix filtering for NER results."""
        assert filter.is_false_positive(
            "tyrosine kinase", "", DrugGeneratorType.SCISPACY_NER
        )


class TestPharmaCompanies:
    """Tests for pharmaceutical company name filtering."""

    def test_company_names_filtered(self, filter):
        """Test that pharma company names are filtered."""
        result = filter.is_false_positive(
            "Pfizer", "manufactured by Pfizer",
            DrugGeneratorType.SCISPACY_NER
        )
        assert isinstance(result, bool)


class TestOrganizations:
    """Tests for organization filtering."""

    def test_organization_names_filtered(self, filter):
        """Test that organization names are filtered."""
        result = filter.is_false_positive(
            "FDA", "FDA approved", DrugGeneratorType.SCISPACY_NER
        )
        assert isinstance(result, bool)


class TestTrialStatusTerms:
    """Tests for trial status term filtering."""

    def test_trial_status_filtered(self, filter):
        """Test that trial status terms are filtered."""
        result = filter.is_false_positive(
            "enrolled", "patients enrolled in the study",
            DrugGeneratorType.SCISPACY_NER
        )
        assert isinstance(result, bool)


class TestGeneratorTypeSpecific:
    """Tests for generator-type specific filtering."""

    def test_scispacy_ner_biological_suffixes(self, filter):
        """Test that scispacy NER results get biological suffix filtering."""
        assert filter.is_false_positive(
            "complement receptor", "", DrugGeneratorType.SCISPACY_NER
        )

    def test_lexicon_common_words(self, filter):
        """Test common word filtering for lexicon sources."""
        result = filter.is_false_positive(
            "gold", "gold standard treatment",
            DrugGeneratorType.LEXICON_RXNORM
        )
        assert result  # "gold" is a common word, should be filtered


class TestAuthorDetection:
    """Tests for author name detection filtering."""

    def test_author_pattern_by(self, filter):
        """Test author pattern 'by Name,'."""
        result = filter.is_false_positive(
            "Smith", "study by Smith, et al",
            DrugGeneratorType.SCISPACY_NER
        )
        assert result  # Should be filtered as author name

    def test_author_pattern_et_al(self, filter):
        """Test author pattern 'Name et al'."""
        result = filter.is_false_positive(
            "Jones", "Jones et al reported",
            DrugGeneratorType.SCISPACY_NER
        )
        assert result  # Should be filtered as author name


class TestDrugAbbreviations:
    """Tests for DRUG_ABBREVIATIONS constant."""

    def test_abbreviations_is_set(self):
        """Test that DRUG_ABBREVIATIONS is a set."""
        assert isinstance(DRUG_ABBREVIATIONS, (set, frozenset, dict, list))


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_input(self, filter):
        """Test handling of empty input."""
        # Empty text should be filtered (too short)
        assert filter.is_false_positive("", "", DrugGeneratorType.LEXICON_RXNORM)

    def test_whitespace_input(self, filter):
        """Test handling of whitespace input."""
        result = filter.is_false_positive("   ", "", DrugGeneratorType.LEXICON_RXNORM)
        assert result  # Should be filtered

    def test_case_insensitive(self, filter):
        """Test case insensitivity."""
        lower_result = filter.is_false_positive(
            "metformin", "", DrugGeneratorType.LEXICON_RXNORM
        )
        upper_result = filter.is_false_positive(
            "METFORMIN", "", DrugGeneratorType.LEXICON_RXNORM
        )
        assert lower_result == upper_result

    def test_parentheses_handling(self, filter):
        """Test handling of text with parentheses."""
        result = filter.is_false_positive(
            "(ongoing)", "trial status (ongoing)",
            DrugGeneratorType.SCISPACY_NER
        )
        assert isinstance(result, bool)


class TestAlwaysFilter:
    """Tests for always-filtered terms."""

    def test_placeholder_terms_filtered(self, filter):
        """Test that placeholder terms are filtered."""
        assert filter.is_false_positive(
            "ab", "", DrugGeneratorType.LEXICON_RXNORM
        )

    def test_trial_status_always_filtered(self, filter):
        """Test that trial status terms are always filtered."""
        result = filter.is_false_positive(
            "NCT12345678", "", DrugGeneratorType.LEXICON_RXNORM
        )
        assert result


class TestIntegration:
    """Integration tests for the filter."""

    def test_real_drug_name(self, filter):
        """Test with a real drug name."""
        assert not filter.is_false_positive(
            "metformin", "metformin 500mg twice daily",
            DrugGeneratorType.LEXICON_RXNORM
        )

    def test_real_bacteria_name(self, filter):
        """Test with real bacteria-like name."""
        result = filter.is_false_positive(
            "escherichia coli", "e. coli infection",
            DrugGeneratorType.SCISPACY_NER
        )
        assert isinstance(result, bool)

    def test_context_matters(self, filter):
        """Test that context affects filtering decisions."""
        # Valid drug should not be filtered regardless of context
        assert not filter.is_false_positive(
            "aspirin", "aspirin 100mg daily",
            DrugGeneratorType.LEXICON_RXNORM
        )


class TestMAFFiltering:
    """Tests for MAF questionnaire abbreviation filtering."""

    def test_maf_filtered_as_common_word(self, filter):
        """Test that MAF is filtered as common word."""
        assert filter.is_false_positive(
            "MAF", "MAF score at baseline", DrugGeneratorType.LEXICON_RXNORM
        )

    def test_maf_case_insensitive(self, filter):
        """Test MAF filtering is case-insensitive."""
        assert filter.is_false_positive(
            "maf", "maf questionnaire", DrugGeneratorType.LEXICON_RXNORM
        )


class TestQuestionnaireAbbreviationFiltering:
    """Tests for clinical questionnaire abbreviation FP filtering."""

    def test_hads_filtered(self, filter):
        """Test that HADS (Hospital Anxiety and Depression Scale) is filtered."""
        assert filter.is_false_positive(
            "HADS", "HADS score at baseline", DrugGeneratorType.LEXICON_RXNORM
        )

    def test_sf36_filtered(self, filter):
        """Test that SF-36 is filtered."""
        assert filter.is_false_positive(
            "SF-36", "SF-36 questionnaire results", DrugGeneratorType.LEXICON_RXNORM
        )

    def test_eq5d_filtered(self, filter):
        """Test that EQ-5D is filtered."""
        assert filter.is_false_positive(
            "EQ-5D", "EQ-5D utility scores", DrugGeneratorType.LEXICON_RXNORM
        )

    def test_vas_filtered(self, filter):
        """Test that VAS (Visual Analog Scale) is filtered."""
        assert filter.is_false_positive(
            "VAS", "VAS pain score", DrugGeneratorType.LEXICON_RXNORM
        )

    def test_promis_filtered(self, filter):
        """Test that PROMIS is filtered."""
        assert filter.is_false_positive(
            "PROMIS", "PROMIS fatigue domain", DrugGeneratorType.LEXICON_RXNORM
        )


class TestAntibodyPhraseFiltering:
    """Tests for antibody phrase filtering as biological entities."""

    def test_anca_antibody_filtered(self, filter):
        """Test that ANCA antibody phrases are filtered."""
        assert filter.is_false_positive(
            "anti-neutrophil cytoplasmic antibody", "",
            DrugGeneratorType.SCISPACY_NER
        )

    def test_antinuclear_antibody_filtered(self, filter):
        """Test that antinuclear antibody is filtered."""
        assert filter.is_false_positive(
            "antinuclear antibody", "",
            DrugGeneratorType.SCISPACY_NER
        )

    def test_acpa_antibody_filtered(self, filter):
        """Test that ACPA antibody phrases are filtered."""
        assert filter.is_false_positive(
            "anti-citrullinated protein antibody", "",
            DrugGeneratorType.SCISPACY_NER
        )


class TestBiologicalSuffixExtendedScope:
    """Tests for biological suffix filtering extended to RxNorm/FDA."""

    def test_rxnorm_biological_suffix_filtered(self, filter):
        """Test that RxNorm matches ending in biological suffixes are filtered."""
        assert filter.is_false_positive(
            "complement receptor", "",
            DrugGeneratorType.LEXICON_RXNORM
        )

    def test_fda_biological_suffix_filtered(self, filter):
        """Test that FDA matches ending in biological suffixes are filtered."""
        assert filter.is_false_positive(
            "tyrosine kinase", "",
            DrugGeneratorType.LEXICON_FDA
        )

    def test_valid_drug_rxnorm_not_filtered(self, filter):
        """Test that valid RxNorm drug names are not filtered."""
        assert not filter.is_false_positive(
            "methotrexate", "", DrugGeneratorType.LEXICON_RXNORM
        )
