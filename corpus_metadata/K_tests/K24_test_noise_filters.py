# corpus_metadata/K_tests/K24_test_noise_filters.py
"""
Tests for C_generators.C21_noise_filters module.

Tests noise filtering constants and validation functions.
"""

from __future__ import annotations

import pytest

from C_generators.C21_noise_filters import (
    OBVIOUS_NOISE,
    MIN_ABBREV_LENGTH,
    WRONG_EXPANSION_BLACKLIST,
    BAD_LONG_FORMS,
    LexiconEntry,
    is_valid_abbreviation_match,
    is_wrong_expansion,
)


class TestObviousNoise:
    """Tests for OBVIOUS_NOISE constant set."""

    def test_single_letters_filtered(self):
        for letter in "abcdefghijklmnopqrstuvwxyz":
            assert letter in OBVIOUS_NOISE

    def test_function_words_filtered(self):
        function_words = ["the", "and", "for", "but", "not", "are", "was"]
        for word in function_words:
            assert word in OBVIOUS_NOISE

    def test_months_filtered(self):
        months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        for month in months:
            assert month in OBVIOUS_NOISE

    def test_units_filtered(self):
        units = ["ml", "mg", "kg", "mm", "cm"]
        for unit in units:
            assert unit in OBVIOUS_NOISE

    def test_company_names_filtered(self):
        companies = ["roche", "novartis", "pfizer", "merck"]
        for company in companies:
            assert company in OBVIOUS_NOISE

    def test_valid_terms_not_filtered(self):
        valid_terms = ["tnf", "dna", "rna", "covid"]
        for term in valid_terms:
            assert term not in OBVIOUS_NOISE


class TestMinAbbrevLength:
    """Tests for MIN_ABBREV_LENGTH constant."""

    def test_minimum_length_value(self):
        assert MIN_ABBREV_LENGTH == 2


class TestWrongExpansionBlacklist:
    """Tests for WRONG_EXPANSION_BLACKLIST constant."""

    def test_known_wrong_pairs(self):
        assert ("task", "product") in WRONG_EXPANSION_BLACKLIST
        assert ("musk", "musk secretion from musk deer") in WRONG_EXPANSION_BLACKLIST
        assert ("et", "essential thrombocythemia") in WRONG_EXPANSION_BLACKLIST

    def test_blacklist_is_set_of_tuples(self):
        assert isinstance(WRONG_EXPANSION_BLACKLIST, set)
        for item in WRONG_EXPANSION_BLACKLIST:
            assert isinstance(item, tuple)
            assert len(item) == 2


class TestBadLongForms:
    """Tests for BAD_LONG_FORMS constant."""

    def test_known_bad_forms(self):
        bad_forms = [
            "product",
            "musk secretion from musk deer",
            "ambulatory care facilities",
        ]
        for form in bad_forms:
            assert form in BAD_LONG_FORMS

    def test_bad_forms_is_set(self):
        assert isinstance(BAD_LONG_FORMS, set)


class TestLexiconEntry:
    """Tests for LexiconEntry class."""

    def test_basic_creation(self):
        import re
        pattern = re.compile(r"\bTNF\b")
        entry = LexiconEntry(
            sf="TNF",
            lf="Tumor Necrosis Factor",
            pattern=pattern,
            source="test_lexicon.json",
        )
        assert entry.sf == "TNF"
        assert entry.lf == "Tumor Necrosis Factor"
        assert entry.source == "test_lexicon.json"
        assert entry.lexicon_ids == []
        assert entry.preserve_case is True

    def test_with_lexicon_ids(self):
        import re
        entry = LexiconEntry(
            sf="IL6",
            lf="Interleukin 6",
            pattern=re.compile(r"\bIL6\b"),
            source="test.json",
            lexicon_ids=[{"source": "UMLS", "id": "C0021760"}],
        )
        assert len(entry.lexicon_ids) == 1
        assert entry.lexicon_ids[0]["source"] == "UMLS"

    def test_slots_optimization(self):
        assert hasattr(LexiconEntry, "__slots__")


class TestIsValidAbbreviationMatch:
    """Tests for is_valid_abbreviation_match function."""

    def test_valid_abbreviations(self):
        valid = ["TNF", "IL6", "COVID", "DNA", "RNA", "mRNA"]
        for term in valid:
            assert is_valid_abbreviation_match(term), f"{term} should be valid"

    def test_obvious_noise_filtered(self):
        noise = ["the", "and", "a", "jan", "feb", "mg"]
        for term in noise:
            assert not is_valid_abbreviation_match(term), f"{term} should be filtered"

    def test_too_short_filtered(self):
        assert not is_valid_abbreviation_match("a")
        assert not is_valid_abbreviation_match("")

    def test_pure_numbers_filtered(self):
        assert not is_valid_abbreviation_match("123")
        assert not is_valid_abbreviation_match("456789")

    def test_short_alphanumeric_starting_with_digit(self):
        assert not is_valid_abbreviation_match("1A")
        assert not is_valid_abbreviation_match("2B")
        assert not is_valid_abbreviation_match("45")

    def test_empty_input(self):
        assert not is_valid_abbreviation_match("")
        assert not is_valid_abbreviation_match(None)


class TestIsWrongExpansion:
    """Tests for is_wrong_expansion function."""

    def test_blacklisted_pairs(self):
        assert is_wrong_expansion("TASK", "product")
        assert is_wrong_expansion("task", "Product")  # Case insensitive
        assert is_wrong_expansion("MUSK", "musk secretion from musk deer")

    def test_bad_long_forms(self):
        assert is_wrong_expansion("ANY", "product")
        assert is_wrong_expansion("X", "ambulatory care facilities")

    def test_valid_expansions(self):
        assert not is_wrong_expansion("TNF", "Tumor Necrosis Factor")
        assert not is_wrong_expansion("DNA", "Deoxyribonucleic Acid")

    def test_case_insensitive(self):
        assert is_wrong_expansion("TASK", "PRODUCT")
        assert is_wrong_expansion("task", "product")
