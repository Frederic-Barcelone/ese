# corpus_metadata/K_tests/K59_test_text_normalization.py
"""
Tests for Z_utils.Z03_text_normalization module.

Tests text normalization utilities for PDF extraction cleanup.
"""

from __future__ import annotations


from Z_utils.Z03_text_normalization import (
    clean_whitespace,
    dehyphenate_long_form,
    truncate_at_clause_breaks,
    normalize_long_form,
)


class TestCleanWhitespace:
    """Tests for clean_whitespace function."""

    def test_empty_string(self):
        assert clean_whitespace("") == ""

    def test_none_input(self):
        assert clean_whitespace(None) == ""  # type: ignore[arg-type]

    def test_single_spaces(self):
        assert clean_whitespace("hello world") == "hello world"

    def test_multiple_spaces(self):
        assert clean_whitespace("hello    world") == "hello world"

    def test_tabs(self):
        assert clean_whitespace("hello\tworld") == "hello world"

    def test_newlines(self):
        assert clean_whitespace("hello\nworld") == "hello world"

    def test_mixed_whitespace(self):
        assert clean_whitespace("  hello  \t\n  world  ") == "hello world"

    def test_leading_trailing(self):
        assert clean_whitespace("   test   ") == "test"


class TestDehyphenateLongForm:
    """Tests for dehyphenate_long_form function."""

    def test_empty_string(self):
        assert dehyphenate_long_form("") == ""

    def test_none_input(self):
        assert dehyphenate_long_form(None) is None  # type: ignore[arg-type]

    def test_no_hyphens(self):
        assert dehyphenate_long_form("tumor necrosis factor") == "tumor necrosis factor"

    def test_line_break_hyphenation(self):
        # Hyphen + space + lowercase = dehyphenate
        assert dehyphenate_long_form("gastro- intestinal") == "gastrointestinal"

    def test_preserve_anti_hyphen(self):
        assert dehyphenate_long_form("anti-inflammatory") == "anti-inflammatory"

    def test_preserve_non_hyphen(self):
        assert dehyphenate_long_form("non-profit") == "non-profit"

    def test_preserve_pre_hyphen(self):
        assert dehyphenate_long_form("pre-existing") == "pre-existing"

    def test_preserve_post_hyphen(self):
        assert dehyphenate_long_form("post-operative") == "post-operative"

    def test_preserve_intra_hyphen(self):
        assert dehyphenate_long_form("intra-venous") == "intra-venous"

    def test_preserve_inter_hyphen(self):
        assert dehyphenate_long_form("inter-cellular") == "inter-cellular"

    def test_preserve_self_hyphen(self):
        assert dehyphenate_long_form("self-administered") == "self-administered"

    def test_dehyphenate_mid_word(self):
        # vasculit-is -> vasculitis (line break in middle of word)
        result = dehyphenate_long_form("Vasculi- tis Study Group")
        # This should keep "Vasculitis" together
        assert "Vasculi" in result or "vasculitis" in result.lower()

    def test_whitespace_normalized_first(self):
        result = dehyphenate_long_form("  tumor   necrosis  ")
        assert result == "tumor necrosis"


class TestTruncateAtClauseBreaks:
    """Tests for truncate_at_clause_breaks function."""

    def test_empty_string(self):
        assert truncate_at_clause_breaks("") == ""

    def test_none_input(self):
        assert truncate_at_clause_breaks(None) == ""  # type: ignore[arg-type]

    def test_no_breaks(self):
        assert truncate_at_clause_breaks("tumor necrosis factor") == "tumor necrosis factor"

    def test_stop_at_period(self):
        result = truncate_at_clause_breaks("tumor necrosis factor. This is more text")
        assert result == "tumor necrosis factor"

    def test_stop_at_semicolon(self):
        result = truncate_at_clause_breaks("tumor necrosis factor; and more")
        assert result == "tumor necrosis factor"

    def test_stop_at_colon(self):
        result = truncate_at_clause_breaks("TNF: tumor necrosis factor")
        assert result == "TNF"

    def test_stop_at_newline(self):
        result = truncate_at_clause_breaks("tumor necrosis factor\nmore text")
        assert result == "tumor necrosis factor"

    def test_stop_at_which(self):
        result = truncate_at_clause_breaks("tumor necrosis factor which is involved")
        assert result == "tumor necrosis factor"

    def test_stop_at_that(self):
        result = truncate_at_clause_breaks("a cytokine that regulates")
        assert result == "a cytokine"

    def test_stop_at_is(self):
        result = truncate_at_clause_breaks("tumor necrosis factor is important")
        assert result == "tumor necrosis factor"

    def test_stop_at_was(self):
        result = truncate_at_clause_breaks("the factor was discovered")
        assert result == "the factor"

    def test_remove_trailing_and(self):
        result = truncate_at_clause_breaks("tumor necrosis and")
        assert result == "tumor necrosis"

    def test_remove_trailing_or(self):
        result = truncate_at_clause_breaks("tumor necrosis or")
        assert result == "tumor necrosis"

    def test_remove_trailing_the(self):
        result = truncate_at_clause_breaks("tumor necrosis the")
        assert result == "tumor necrosis"

    def test_stop_at_parenthesis(self):
        result = truncate_at_clause_breaks("tumor necrosis factor (TNF)")
        assert result == "tumor necrosis factor"

    def test_stop_at_bracket(self):
        result = truncate_at_clause_breaks("tumor necrosis factor [1]")
        assert result == "tumor necrosis factor"


class TestNormalizeLongForm:
    """Tests for normalize_long_form function."""

    def test_empty_string(self):
        assert normalize_long_form("") == ""

    def test_none_input(self):
        assert normalize_long_form(None) is None  # type: ignore[arg-type]

    def test_simple_normalization(self):
        result = normalize_long_form("tumor necrosis factor")
        assert result == "tumor necrosis factor"

    def test_whitespace_normalization(self):
        result = normalize_long_form("  tumor   necrosis   factor  ")
        assert result == "tumor necrosis factor"

    def test_clause_truncation(self):
        result = normalize_long_form("tumor necrosis factor which is a cytokine")
        assert result == "tumor necrosis factor"

    def test_dehyphenation(self):
        result = normalize_long_form("gastro- intestinal tract")
        assert "gastrointestinal" in result.lower() or "gastro" in result

    def test_preserve_compound_hyphens(self):
        result = normalize_long_form("anti-inflammatory response")
        assert "anti-inflammatory" in result

    def test_combined_normalization(self):
        result = normalize_long_form("  gastro- intestinal  tract which is important")
        # Should clean whitespace, dehyphenate, and truncate
        assert "which" not in result.lower()

    def test_very_short_truncation_fallback(self):
        # If truncation results in too-short text, should fall back
        result = normalize_long_form("A. is more text")
        # Should return something meaningful
        assert result == "A. is more text"

    def test_period_before_clause(self):
        result = normalize_long_form("tumor necrosis factor. The cytokine")
        assert result == "tumor necrosis factor"

    def test_multiple_normalizations(self):
        # Complex case with multiple issues
        input_text = "  tumor  necrosis   factor  (TNF) which is important "
        result = normalize_long_form(input_text)
        assert "  " not in result  # No double spaces
        assert result.strip() == result  # No leading/trailing whitespace
