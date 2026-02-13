# corpus_metadata/K_tests/K58_test_text_helpers.py
"""
Tests for Z_utils.Z02_text_helpers module.

Tests text processing helpers for abbreviation extraction.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from Z_utils.Z02_text_helpers import (
    extract_context_snippet,
    normalize_lf_for_dedup,
    has_numeric_evidence,
    is_valid_sf_form,
    score_lf_quality,
)


class TestExtractContextSnippet:
    """Tests for extract_context_snippet function."""

    def test_basic_extraction(self):
        text = "The tumor necrosis factor (TNF) is a cytokine involved in inflammation."
        result = extract_context_snippet(text, match_start=27, match_end=30, window=20)
        assert "TNF" in result
        assert len(result) <= 43  # 3 + 20 + 20

    def test_start_of_document(self):
        text = "TNF is important for the immune system."
        result = extract_context_snippet(text, match_start=0, match_end=3, window=10)
        assert result.startswith("TNF")

    def test_end_of_document(self):
        text = "The abbreviation is TNF"
        result = extract_context_snippet(text, match_start=20, match_end=23, window=10)
        assert result.endswith("TNF")

    def test_custom_window_size(self):
        text = "A" * 50 + "TNF" + "B" * 50
        result = extract_context_snippet(text, match_start=50, match_end=53, window=25)
        assert len(result) == 53  # 25 + 3 + 25


class TestNormalizeLfForDedup:
    """Tests for normalize_lf_for_dedup function."""

    def test_empty_string(self):
        assert normalize_lf_for_dedup("") == ""

    def test_none_input(self):
        assert normalize_lf_for_dedup(None) == ""  # type: ignore[arg-type]

    def test_lowercase_conversion(self):
        assert normalize_lf_for_dedup("Tumor Necrosis Factor") == "tumor necrosis factor"

    def test_whitespace_normalization(self):
        assert normalize_lf_for_dedup("tumor   necrosis    factor") == "tumor necrosis factor"

    def test_en_dash_normalization(self):
        # En-dash (\u2013) to hyphen
        assert normalize_lf_for_dedup("renin\u2013angiotensin") == "renin-angiotensin"

    def test_em_dash_normalization(self):
        # Em-dash (\u2014) to hyphen
        assert normalize_lf_for_dedup("protein\u2014creatinine") == "protein-creatinine"

    def test_hyphen_whitespace_normalization(self):
        assert normalize_lf_for_dedup("urine protein \u2013 creatinine") == "urine protein-creatinine"

    def test_strip(self):
        assert normalize_lf_for_dedup("  tumor necrosis factor  ") == "tumor necrosis factor"


class TestHasNumericEvidence:
    """Tests for has_numeric_evidence function."""

    def test_empty_context(self):
        assert has_numeric_evidence("", "CI") is False

    def test_sf_not_in_context(self):
        assert has_numeric_evidence("some random text", "TNF") is False

    def test_with_digits(self):
        assert has_numeric_evidence("95% CI for HR 1.5", "CI") is True

    def test_with_percent(self):
        assert has_numeric_evidence("CI 95%", "CI") is True

    def test_with_equals(self):
        assert has_numeric_evidence("HR = 1.5", "HR") is True

    def test_with_colon_number(self):
        assert has_numeric_evidence("SD: 0.5", "SD") is True

    def test_with_range(self):
        assert has_numeric_evidence("CI: 1.2-1.8", "CI") is True

    def test_without_numeric(self):
        # Just text, no numbers
        assert has_numeric_evidence("the confidence interval was calculated", "CI") is False


class TestIsValidSfForm:
    """Tests for is_valid_sf_form function."""

    @pytest.fixture
    def allowed_sets(self):
        return {
            "allowed_2letter": {"CI", "OR", "HR", "SD", "SE"},
            "allowed_mixed": {"TNF", "IgA", "VEGF"},
        }

    def test_allowed_2letter(self, allowed_sets):
        assert is_valid_sf_form("CI", "", **allowed_sets) is True
        assert is_valid_sf_form("OR", "", **allowed_sets) is True

    def test_allowed_mixed(self, allowed_sets):
        assert is_valid_sf_form("TNF", "", **allowed_sets) is True

    def test_reject_author_initials(self, allowed_sets):
        assert is_valid_sf_form("A.B.", "", **allowed_sets) is False

    def test_reject_figure_reference(self, allowed_sets):
        assert is_valid_sf_form("Figure 3B", "", **allowed_sets) is False
        assert is_valid_sf_form("Table S1", "", **allowed_sets) is False
        assert is_valid_sf_form("Fig 2", "", **allowed_sets) is False

    def test_reject_et_al(self, allowed_sets):
        assert is_valid_sf_form("al", "", **allowed_sets) is False

    def test_reject_doi_patterns(self, allowed_sets):
        assert is_valid_sf_form("10.1056/NEJMoa2024816", "", **allowed_sets) is False
        assert is_valid_sf_form("doi:10.1056", "", **allowed_sets) is False

    def test_reject_statistical_names(self, allowed_sets):
        assert is_valid_sf_form("COX", "", **allowed_sets) is False
        assert is_valid_sf_form("KAPLAN", "", **allowed_sets) is False

    def test_reject_plural_forms(self, allowed_sets):
        assert is_valid_sf_form("CIs", "", **allowed_sets) is False
        assert is_valid_sf_form("HRs", "", **allowed_sets) is False

    def test_reject_author_pattern_in_citation(self, allowed_sets):
        context = "Hoffman EA, doi: 10.1056/NEJMoa2024816"
        assert is_valid_sf_form("EA", context, **allowed_sets) is False

    def test_ig_only_with_immunoglobulin_context(self, allowed_sets):
        assert is_valid_sf_form("IG", "IgG levels were measured", **allowed_sets) is True
        assert is_valid_sf_form("IG", "some random text", **allowed_sets) is False

    def test_reject_lowercase_words(self, allowed_sets):
        assert is_valid_sf_form("medication", "", **allowed_sets) is False
        assert is_valid_sf_form("patients", "", **allowed_sets) is False

    def test_reject_capitalized_words(self, allowed_sets):
        assert is_valid_sf_form("Medications", "", **allowed_sets) is False
        assert is_valid_sf_form("Crucially", "", **allowed_sets) is False


class TestScoreLfQuality:
    """Tests for score_lf_quality function."""

    @pytest.fixture
    def mock_candidate(self):
        candidate = MagicMock()
        candidate.short_form = "CI"
        candidate.long_form = "confidence interval"
        candidate.provenance = MagicMock()
        candidate.provenance.lexicon_source = None
        return candidate

    def test_stats_canonical_boost(self, mock_candidate):
        mock_candidate.short_form = "CI"
        mock_candidate.long_form = "confidence interval"
        full_text = "The 95% confidence interval (CI) was calculated."

        score = score_lf_quality(mock_candidate, full_text)
        assert score > 0  # Should get boost for canonical stats term

    def test_stats_non_canonical_penalty(self, mock_candidate):
        mock_candidate.short_form = "CI"
        mock_candidate.long_form = "curie"
        full_text = "CI units are used."

        score = score_lf_quality(mock_candidate, full_text)
        # Should get penalty for non-canonical stats term
        assert score < 100  # Assuming other boosts could be +100

    def test_lf_in_text_boost(self, mock_candidate):
        mock_candidate.short_form = "TNF"
        mock_candidate.long_form = "tumor necrosis factor"
        full_text = "The tumor necrosis factor (TNF) plays a role in inflammation."

        score = score_lf_quality(mock_candidate, full_text)
        assert score > 0

    def test_long_lf_penalty(self, mock_candidate):
        mock_candidate.short_form = "TEST"
        mock_candidate.long_form = "this is a very long long form that should be penalized" * 2
        full_text = "TEST"

        score = score_lf_quality(mock_candidate, full_text)
        assert score < 0  # Long forms should be penalized

    def test_stopword_penalty(self, mock_candidate):
        mock_candidate.short_form = "TEST"
        mock_candidate.long_form = "the test is a thing"
        full_text = "TEST"

        score = score_lf_quality(mock_candidate, full_text)
        assert score < 50  # Stopword-heavy LF should score low

    def test_partial_extraction_penalty(self, mock_candidate):
        mock_candidate.short_form = "TEST"
        mock_candidate.long_form = "and this is partial"
        full_text = "TEST"

        score = score_lf_quality(mock_candidate, full_text)
        assert score < 50  # Starting with connector should be penalized

    def test_umls_boost(self, mock_candidate):
        mock_candidate.short_form = "TNF"
        mock_candidate.long_form = "tumor necrosis factor"
        mock_candidate.provenance.lexicon_source = "umls-metathesaurus"
        full_text = "TNF"

        score_with_umls = score_lf_quality(mock_candidate, full_text)
        mock_candidate.provenance.lexicon_source = None
        score_without = score_lf_quality(mock_candidate, full_text)
        assert score_with_umls >= score_without  # UMLS source should boost or equal

    def test_cached_lowercase(self, mock_candidate):
        """Test that cached lowercase text is used correctly."""
        mock_candidate.short_form = "TNF"
        mock_candidate.long_form = "tumor necrosis factor"
        full_text = "The tumor necrosis factor (TNF) plays a role."
        full_text_lower = full_text.lower()

        score1 = score_lf_quality(mock_candidate, full_text, full_text_lower)
        score2 = score_lf_quality(mock_candidate, full_text)
        assert score1 == score2
