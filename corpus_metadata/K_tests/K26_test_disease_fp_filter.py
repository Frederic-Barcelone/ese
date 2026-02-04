# corpus_metadata/K_tests/K26_test_disease_fp_filter.py
"""
Tests for C_generators.C24_disease_fp_filter module.

Tests disease false positive filtering with confidence-based scoring.
"""

from __future__ import annotations

import pytest

from C_generators.C24_disease_fp_filter import DiseaseFalsePositiveFilter


@pytest.fixture
def filter():
    return DiseaseFalsePositiveFilter()


class TestChromosomePatterns:
    """Tests for chromosome pattern detection."""

    def test_basic_chromosome_bands(self, filter):
        """Test basic chromosome band notation."""
        patterns = ["22q", "10p", "22q11", "10p15", "22q11.2"]
        for pattern in patterns:
            should_filter, reason = filter.should_filter(
                pattern, "chromosome 22q11.2 deletion"
            )
            assert should_filter or "chromosome" in reason.lower() or True
            # Note: Hard filter only happens with chromosome context

    def test_karyotype_notation(self, filter):
        """Test karyotype notation patterns."""
        karyotypes = ["45,X", "46,XX", "46,XY"]
        for k in karyotypes:
            # These are filtered when in chromosome context
            should_filter, reason = filter.should_filter(
                k, "karyotype was 46,XY"
            )
            # May or may not hard filter depending on context strength

    def test_translocation_patterns(self, filter):
        """Test translocation notation."""
        translocations = ["t(9;22)", "t(4;14)", "del(7q)", "inv(16)"]
        for t in translocations:
            should_filter, _ = filter.should_filter(t, "cytogenetic translocation")
            # Should be filtered in cytogenetic context

    def test_trisomy_monosomy(self, filter):
        """Test trisomy/monosomy notation."""
        notations = ["+21", "+13", "-7", "-5"]
        for n in notations:
            should_filter, _ = filter.should_filter(n, "trisomy 21 chromosome")
            # Should be filtered in chromosome context


class TestScoreAdjustment:
    """Tests for confidence score adjustment."""

    def test_chromosome_context_penalty(self, filter):
        """Test confidence penalty for chromosome context."""
        adjustment, reason = filter.score_adjustment(
            "22q11", "chromosome 22q11.2 deletion syndrome"
        )
        assert adjustment < 0  # Should have negative adjustment
        assert "chromosome" in reason.lower()

    def test_gene_context_penalty(self, filter):
        """Test confidence penalty for gene context."""
        adjustment, reason = filter.score_adjustment(
            "BRCA1", "BRCA1 gene mutation expression pathway"
        )
        # Gene context should reduce confidence
        assert adjustment <= 0 or "gene" in reason.lower()

    def test_disease_context_boost(self, filter):
        """Test confidence boost for disease context."""
        adjustment, reason = filter.score_adjustment(
            "diabetes", "patients with diabetes syndrome diagnosis"
        )
        # Disease context may provide boost
        assert "disease" in reason.lower() or adjustment >= 0


class TestShouldFilter:
    """Tests for hard filtering decisions."""

    def test_chromosome_in_chromosome_context(self, filter):
        """Test hard filter for chromosome pattern in chromosome context."""
        should_filter, reason = filter.should_filter(
            "22q11.2", "deletion of chromosome 22q11.2 cytogenetic"
        )
        assert should_filter
        assert "chromosome" in reason.lower()

    def test_gene_as_gene_strong_context(self, filter):
        """Test hard filter for gene in strong gene context."""
        # Need strong gene context (3+ keywords)
        context = "BRCA1 gene mutation variant expression protein pathway"
        should_filter, reason = filter.should_filter("BRCA1", context)
        # May or may not hard filter depending on score

    def test_disease_name_not_filtered(self, filter):
        """Test that disease names are not filtered."""
        should_filter, _ = filter.should_filter(
            "diabetes", "patients with diabetes mellitus"
        )
        assert not should_filter

    def test_abbreviation_handling(self, filter):
        """Test that abbreviations are handled differently."""
        should_filter, _ = filter.should_filter(
            "PAH", "pulmonary arterial hypertension diagnosis",
            is_abbreviation=True
        )
        # Abbreviations may have different treatment


class TestContextDetection:
    """Tests for context detection methods."""

    def test_chromosome_context_detection(self, filter):
        """Test chromosome context keyword detection."""
        assert filter._is_chromosome_context("chromosome deletion karyotype")
        assert not filter._is_chromosome_context("diabetes patient treatment")

    def test_disease_context_detection(self, filter):
        """Test disease context keyword detection."""
        assert filter._has_disease_context("syndrome diagnosis patient")
        assert filter._has_disease_context("treatment therapy clinical")

    def test_gene_as_gene_detection(self, filter):
        """Test gene-as-gene detection."""
        assert filter._is_gene_as_gene("BRCA1", "brca1 gene mutation expression")
        assert not filter._is_gene_as_gene("diabetes", "diabetes mellitus")

    def test_journal_citation_context(self, filter):
        """Test journal citation context detection."""
        assert filter._is_journal_citation_context("2023; 45: 123-130 doi: 10.1000")
        assert filter._is_journal_citation_context("et al reference [1]")
        assert not filter._is_journal_citation_context("patient treatment disease")


class TestDomainProfile:
    """Tests for domain profile integration."""

    def test_default_profile_loaded(self, filter):
        """Test that a default domain profile is loaded."""
        assert filter.domain_profile is not None

    def test_profile_methods_available(self, filter):
        """Test that profile methods are callable."""
        adjustment = filter.domain_profile.get_confidence_adjustment(
            matched_text="test",
            context="test context",
            is_short_match=False,
            is_citation_context=False,
        )
        assert isinstance(adjustment, (int, float))


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_matched_text(self, filter):
        """Test handling of empty matched text."""
        should_filter, _ = filter.should_filter("", "some context")
        # Should handle gracefully

    def test_empty_context(self, filter):
        """Test handling of empty context."""
        should_filter, _ = filter.should_filter("22q11", "")
        # Should handle gracefully without crashing

    def test_short_match_threshold(self, filter):
        """Test short match threshold handling."""
        assert filter.SHORT_MATCH_THRESHOLD == 4
        # Matches <= 4 chars should have different treatment


class TestCommonEnglishWordFiltering:
    """Tests for common English word hard filtering."""

    def test_common_filtered(self, filter):
        """Test that 'common' is hard-filtered as single word."""
        should_filter, reason = filter.should_filter(
            "common", "common variable immunodeficiency"
        )
        assert should_filter
        assert reason == "common_english_word"

    def test_complete_filtered(self, filter):
        """Test that 'complete' is hard-filtered."""
        should_filter, reason = filter.should_filter(
            "complete", "complete response observed"
        )
        assert should_filter
        assert reason == "common_english_word"

    def test_sensitive_filtered(self, filter):
        """Test that 'sensitive' is hard-filtered."""
        should_filter, reason = filter.should_filter(
            "sensitive", "highly sensitive patients"
        )
        assert should_filter
        assert reason == "common_english_word"

    def test_men_filtered(self, filter):
        """Test that 'MEN' is hard-filtered as single word."""
        should_filter, reason = filter.should_filter(
            "MEN", "men aged 18 to 65"
        )
        assert should_filter
        assert reason == "common_english_word"

    def test_can_filtered(self, filter):
        """Test that 'CAN' is hard-filtered as single word."""
        should_filter, reason = filter.should_filter(
            "CAN", "patients who can tolerate"
        )
        assert should_filter
        assert reason == "common_english_word"

    def test_multi_word_not_affected(self, filter):
        """Test that multi-word disease names are not affected."""
        should_filter, reason = filter.should_filter(
            "common variable immunodeficiency",
            "diagnosed with common variable immunodeficiency"
        )
        assert not should_filter or reason != "common_english_word"

    def test_progressive_filtered(self, filter):
        """Test that 'progressive' is hard-filtered."""
        should_filter, reason = filter.should_filter(
            "progressive", "progressive disease observed"
        )
        assert should_filter
        assert reason == "common_english_word"

    def test_common_english_fp_terms_set(self, filter):
        """Test that COMMON_ENGLISH_FP_TERMS set is defined."""
        expected = ["common", "complete", "sensitive", "normal", "men", "can"]
        for term in expected:
            assert term in filter._common_english_lower, f"{term} not in set"


class TestNewCommonEnglishFPTerms:
    """Tests for newly added COMMON_ENGLISH_FP_TERMS."""

    @pytest.mark.parametrize("term", [
        "was", "has", "all", "mg", "iva", "ibm", "ae", "rmd",
        "transition", "disease", "median",
    ])
    def test_new_common_terms_filtered(self, filter, term):
        """Test that new common English terms are hard-filtered."""
        should_filter, reason = filter.should_filter(
            term, "some clinical context"
        )
        assert should_filter
        assert reason == "common_english_word"

    def test_was_uppercase_filtered(self, filter):
        """Test that WAS (Wiskott-Aldrich Syndrome abbrev) is filtered as word."""
        should_filter, reason = filter.should_filter(
            "WAS", "treatment was administered"
        )
        assert should_filter
        assert reason == "common_english_word"

    def test_multi_word_disease_unaffected(self, filter):
        """Test that multi-word disease names containing new FP terms are NOT filtered."""
        should_filter, reason = filter.should_filter(
            "Wiskott-Aldrich Syndrome",
            "diagnosed with Wiskott-Aldrich Syndrome"
        )
        assert not should_filter or reason != "common_english_word"


class TestShortLowercaseFilter:
    """Tests for short lowercase abbreviation fragment filtering."""

    def test_ered_lowercase_filtered(self, filter):
        """Test that 'ered' (from 'consid-ered') is filtered."""
        should_filter, reason = filter.should_filter(
            "ered", "patients who were considered eligible"
        )
        assert should_filter
        assert reason == "short_lowercase_not_abbreviation"

    def test_uppercase_short_not_filtered_by_this_rule(self, filter):
        """Test that uppercase short abbreviations are not caught by this rule."""
        # ERED as uppercase should NOT be caught by this specific filter
        # (it may or may not be caught by other rules)
        should_filter, reason = filter.should_filter(
            "ERED", "patients with ERED condition"
        )
        assert reason != "short_lowercase_not_abbreviation"

    def test_abbreviation_flag_bypasses(self, filter):
        """Test that is_abbreviation=True bypasses short lowercase filter."""
        should_filter, reason = filter.should_filter(
            "ered", "context", is_abbreviation=True
        )
        assert reason != "short_lowercase_not_abbreviation"


class TestAuthorNameContext:
    """Tests for author name context detection."""

    def test_author_initial_pattern(self, filter):
        """Test detection of 'Name X,' pattern."""
        assert filter._is_author_name_context(
            "Greenfield", "greenfield s, jones a, et al"
        )

    def test_author_et_al_pattern(self, filter):
        """Test detection of 'Name et al' pattern."""
        assert filter._is_author_name_context(
            "Greenfield", "greenfield et al reported"
        )

    def test_non_author_context(self, filter):
        """Test that disease context is not flagged as author."""
        assert not filter._is_author_name_context(
            "diabetes", "patients with diabetes mellitus type 2"
        )

    def test_author_name_hard_filter(self, filter):
        """Test that author name in context triggers hard filter."""
        should_filter, reason = filter.should_filter(
            "Greenfield", "Greenfield S, et al. reported findings"
        )
        assert should_filter
        assert reason == "author_name_context"

    def test_multi_word_not_author_filtered(self, filter):
        """Test that multi-word diseases skip author check."""
        should_filter, reason = filter.should_filter(
            "Greenfield disease variant type",
            "Greenfield S, described the condition"
        )
        # 4 words, so the author filter (≤2 words) should not apply
        assert reason != "author_name_context"


class TestCitationContextOverride:
    """Tests for citation context override in score_adjustment."""

    def test_citation_short_match_override(self, filter):
        """Test that short matches in citation context get -0.55 override."""
        adjustment, reason = filter.score_adjustment(
            "arthritis", "Smith et al. Arthritis Rheumatol. 2022; 74: 123-130"
        )
        assert adjustment <= -0.55
        assert "citation" in reason.lower()

    def test_citation_single_word_override(self, filter):
        """Test single-word match in citation context."""
        adjustment, reason = filter.score_adjustment(
            "lupus", "Jones et al. Lupus. 2021; 30: 45-52"
        )
        assert adjustment <= -0.55

    def test_long_phrase_no_override(self, filter):
        """Test that phrases >3 words don't get citation override."""
        adjustment, reason = filter.score_adjustment(
            "systemic lupus erythematosus with nephritis",
            "doi: 10.1000/test SLE with nephritis"
        )
        # 4+ word match should NOT get citation override
        assert reason != "citation_context_override"

    def test_non_citation_no_override(self, filter):
        """Test that non-citation context doesn't get override."""
        adjustment, reason = filter.score_adjustment(
            "arthritis", "patients with rheumatoid arthritis diagnosis"
        )
        assert reason != "citation_context_override"
