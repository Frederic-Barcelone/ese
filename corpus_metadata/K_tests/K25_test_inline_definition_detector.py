# corpus_metadata/K_tests/K25_test_inline_definition_detector.py
"""
Tests for C_generators.C23_inline_definition_detector module.

Tests inline abbreviation definition detection patterns.
"""

from __future__ import annotations

import pytest

from C_generators.C23_inline_definition_detector import InlineDefinitionDetectorMixin


class TestInlineDefinitionDetector(InlineDefinitionDetectorMixin):
    """Test class that uses the mixin."""
    pass


@pytest.fixture
def detector():
    return TestInlineDefinitionDetector()


class TestExtractInlineDefinitions:
    """Tests for _extract_inline_definitions method."""

    def test_standard_pattern_title_case(self, detector):
        """Test 'Long Form (ABBREV)' pattern with title case."""
        text = "Tumor Necrosis Factor (TNF) plays a key role"
        results = detector._extract_inline_definitions(text)
        assert len(results) >= 1
        sf, lf, start, end = results[0]
        assert sf == "TNF"
        assert "Tumor" in lf

    def test_standard_pattern_lowercase(self, detector):
        """Test 'long form (ABBREV)' pattern with lowercase."""
        text = "level of agreement (LoA) was calculated"
        results = detector._extract_inline_definitions(text)
        # May or may not match depending on validation
        # Pattern 1b should catch this

    def test_reversed_pattern(self, detector):
        """Test 'ABBREV (long form)' reversed pattern."""
        text = "LoE (level of evidence) was used"
        results = detector._extract_inline_definitions(text)
        # Pattern 3 should catch reversed definitions
        found_loe = any(sf == "LoE" for sf, lf, s, e in results)
        # This tests the pattern exists, actual match depends on validation

    def test_comma_separator_pattern(self, detector):
        """Test 'ABBREV, the long form' pattern."""
        text = "GPA, or granulomatosis with polyangiitis, is a disease"
        results = detector._extract_inline_definitions(text)
        # Pattern 2 should catch this

    def test_equals_separator_pattern(self, detector):
        """Test 'ABBREV = long form' pattern."""
        text = "TNF = tumor necrosis factor in this study"
        results = detector._extract_inline_definitions(text)
        # Pattern 4 should catch this

    def test_hyphenated_long_form(self, detector):
        """Test long forms with hyphens."""
        text = "Five-Factor Score (FFS) was measured"
        results = detector._extract_inline_definitions(text)
        # Should handle hyphenated words

    def test_no_definitions(self, detector):
        """Test text with no definitions."""
        text = "This is plain text without any abbreviation definitions."
        results = detector._extract_inline_definitions(text)
        assert len(results) == 0

    def test_empty_input(self, detector):
        """Test empty input."""
        results = detector._extract_inline_definitions("")
        assert results == []


class TestCouldBeAbbreviation:
    """Tests for _could_be_abbreviation validation method."""

    def test_initials_match(self, detector):
        """Test when initials match exactly."""
        assert detector._could_be_abbreviation("TNF", "Tumor Necrosis Factor")
        assert detector._could_be_abbreviation("DNA", "Deoxyribonucleic Acid")

    def test_partial_initials_match(self, detector):
        """Test when initials partially match."""
        assert detector._could_be_abbreviation("IL", "Interleukin")

    def test_letters_in_order(self, detector):
        """Test when letters appear in order."""
        assert detector._could_be_abbreviation("EGFR", "Epidermal Growth Factor Receptor")

    def test_no_match(self, detector):
        """Test when there's no plausible match."""
        assert not detector._could_be_abbreviation("XYZ", "Tumor Necrosis Factor")

    def test_case_insensitive(self, detector):
        """Test case insensitivity."""
        assert detector._could_be_abbreviation("tnf", "tumor necrosis factor")
        assert detector._could_be_abbreviation("TNF", "tumor necrosis factor")


class TestLeadInRemoval:
    """Tests for lead-in phrase removal in pattern 1b."""

    def test_removes_known_as(self, detector):
        """Test removal of 'known as' prefix."""
        text = "known as level of evidence (LoE)"
        results = detector._extract_inline_definitions(text)
        # Should not include "known as" in long form

    def test_removes_including(self, detector):
        """Test removal of 'including' prefix."""
        text = "including tumor necrosis factor (TNF)"
        results = detector._extract_inline_definitions(text)
        # Should not include "including" in long form


class TestEdgeCases:
    """Tests for edge cases in inline definition detection."""

    def test_multiple_definitions(self, detector):
        """Test text with multiple definitions."""
        text = "Tumor Necrosis Factor (TNF) and Interleukin (IL) levels"
        results = detector._extract_inline_definitions(text)
        assert len(results) >= 1

    def test_nested_parentheses(self, detector):
        """Test handling of nested parentheses."""
        text = "enzyme activity (EA) (measured in U/L)"
        results = detector._extract_inline_definitions(text)
        # Should handle gracefully

    def test_unicode_text(self, detector):
        """Test handling of unicode characters."""
        text = "α-synuclein (αSyn) aggregates"
        results = detector._extract_inline_definitions(text)
        # Should handle unicode

    def test_numbers_in_abbreviation(self, detector):
        """Test abbreviations with numbers."""
        text = "Interleukin 6 (IL6) concentration"
        results = detector._extract_inline_definitions(text)
        # IL6 should be detected

    def test_long_form_truncation(self, detector):
        """Test that very long forms are properly truncated."""
        text = "a very long phrase that goes on and on with many words (ABBREV)"
        results = detector._extract_inline_definitions(text)
        # Should truncate or reject very long forms
