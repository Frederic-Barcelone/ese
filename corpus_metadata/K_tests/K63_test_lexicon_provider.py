"""
Tests for Z_utils.Z15_lexicon_provider.

Validates:
    - Library-backed lexicon functions return correct types and sizes
    - ABBREVIATION_EXCLUSIONS are properly excluded from stopwords
    - build_obvious_noise() is superset of old YAML entries
    - build_garbage_tokens() is superset of old YAML entries
    - CREDENTIALS contains all entries from both old drug + gene lists
    - month_names() covers all 12 months
    - single_letters() covers a-z
"""

from __future__ import annotations

import calendar
import string


class TestStopwordsBase:
    """Tests for stopwords_base()."""

    def test_returns_frozenset(self) -> None:
        from Z_utils.Z15_lexicon_provider import stopwords_base
        result = stopwords_base()
        assert isinstance(result, frozenset)

    def test_has_many_entries(self) -> None:
        from Z_utils.Z15_lexicon_provider import stopwords_base
        result = stopwords_base()
        # spaCy English has 300+ stopwords
        assert len(result) > 300

    def test_common_stopwords_present(self) -> None:
        from Z_utils.Z15_lexicon_provider import stopwords_base
        result = stopwords_base()
        for word in ["the", "and", "is", "in", "of", "to", "a"]:
            assert word in result, f"Expected '{word}' in stopwords"

    def test_abbreviation_exclusions_absent(self) -> None:
        from Z_utils.Z15_lexicon_provider import ABBREVIATION_EXCLUSIONS, stopwords_base
        result = stopwords_base()
        for word in ABBREVIATION_EXCLUSIONS:
            assert word not in result, f"'{word}' should be excluded from stopwords"


class TestSingleLetters:
    """Tests for single_letters()."""

    def test_returns_frozenset(self) -> None:
        from Z_utils.Z15_lexicon_provider import single_letters
        result = single_letters()
        assert isinstance(result, frozenset)

    def test_covers_all_26_letters(self) -> None:
        from Z_utils.Z15_lexicon_provider import single_letters
        result = single_letters()
        assert len(result) == 26
        for c in string.ascii_lowercase:
            assert c in result


class TestMonthNames:
    """Tests for month_names()."""

    def test_returns_frozenset(self) -> None:
        from Z_utils.Z15_lexicon_provider import month_names
        result = month_names()
        assert isinstance(result, frozenset)

    def test_covers_all_12_full_months(self) -> None:
        from Z_utils.Z15_lexicon_provider import month_names
        result = month_names()
        for i in range(1, 13):
            name = calendar.month_name[i].lower()
            assert name in result, f"Expected '{name}' in month_names"

    def test_covers_all_12_abbreviated_months(self) -> None:
        from Z_utils.Z15_lexicon_provider import month_names
        result = month_names()
        for i in range(1, 13):
            abbr = calendar.month_abbr[i].lower()
            assert abbr in result, f"Expected '{abbr}' in month_names"

    def test_sept_included(self) -> None:
        from Z_utils.Z15_lexicon_provider import month_names
        assert "sept" in month_names()


class TestBuildObviousNoise:
    """Tests for build_obvious_noise()."""

    def test_returns_frozenset(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_obvious_noise
        result = build_obvious_noise()
        assert isinstance(result, frozenset)

    def test_contains_single_letters(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_obvious_noise
        result = build_obvious_noise()
        assert "a" in result
        assert "z" in result

    def test_contains_stopwords(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_obvious_noise
        result = build_obvious_noise()
        assert "the" in result
        assert "and" in result

    def test_contains_months(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_obvious_noise
        result = build_obvious_noise()
        assert "january" in result
        assert "sept" in result

    def test_contains_domain_entries(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_obvious_noise
        result = build_obvious_noise()
        # Domain terms from noise_filters.yaml -> obvious_noise_domain
        assert "mmhg" in result
        assert "roche" in result
        assert "investigator" in result

    def test_excludes_abbreviation_exclusions(self) -> None:
        from Z_utils.Z15_lexicon_provider import ABBREVIATION_EXCLUSIONS, build_obvious_noise
        result = build_obvious_noise()
        for word in ABBREVIATION_EXCLUSIONS:
            assert word not in result, f"'{word}' should NOT be in obvious noise"

    def test_superset_of_original_entries(self) -> None:
        """All entries from the old obvious_noise YAML should be present."""
        from Z_utils.Z15_lexicon_provider import build_obvious_noise
        result = build_obvious_noise()
        # Spot-check entries that were in the original YAML
        old_entries = [
            "a", "b", "z",         # single letters
            "the", "and", "for",   # function words
            "jan", "february",     # months
            "mmhg", "kpa",         # units
            "roche", "pfizer",     # companies
            "et", "al",            # citation artifacts
        ]
        for entry in old_entries:
            assert entry in result, f"Old entry '{entry}' missing from build_obvious_noise()"


class TestBuildGarbageTokens:
    """Tests for build_garbage_tokens()."""

    def test_returns_frozenset(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_garbage_tokens
        result = build_garbage_tokens()
        assert isinstance(result, frozenset)

    def test_contains_stopwords(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_garbage_tokens
        result = build_garbage_tokens()
        assert "the" in result
        assert "was" in result

    def test_contains_domain_entries(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_garbage_tokens
        result = build_garbage_tokens()
        assert "rate" in result
        assert "mg" in result
        assert "higher" in result

    def test_superset_of_original_entries(self) -> None:
        """All entries from the old garbage_tokens YAML should be present."""
        from Z_utils.Z15_lexicon_provider import build_garbage_tokens
        result = build_garbage_tokens()
        old_entries = [
            "rate", "reduced", "crea",
            "the", "and", "for", "with",
            "mg", "ml", "dl", "kg",
            "higher", "lower", "normal",
        ]
        for entry in old_entries:
            assert entry in result, f"Old entry '{entry}' missing from build_garbage_tokens()"


class TestCredentials:
    """Tests for CREDENTIALS constant."""

    def test_is_frozenset(self) -> None:
        from Z_utils.Z15_lexicon_provider import CREDENTIALS
        assert isinstance(CREDENTIALS, frozenset)

    def test_contains_all_old_drug_credentials(self) -> None:
        from Z_utils.Z15_lexicon_provider import CREDENTIALS
        old_drug = {"md", "phd", "mph", "mbbs", "frcp", "do", "rn", "np", "pa",
                    "pharmd", "dnp", "dpt", "ms", "ma", "msc", "bsc", "ba"}
        for cred in old_drug:
            assert cred in CREDENTIALS, f"Drug credential '{cred}' missing"

    def test_contains_all_old_gene_credentials(self) -> None:
        from Z_utils.Z15_lexicon_provider import CREDENTIALS
        old_gene = {"md", "phd", "mph", "do", "rn", "np", "pa", "pharmd",
                    "dds", "dmd", "dpt", "od", "dvm", "dc", "ms", "ma",
                    "msc", "bsc", "ba", "mba", "jd", "llm"}
        for cred in old_gene:
            assert cred in CREDENTIALS, f"Gene credential '{cred}' missing"

    def test_unified_count(self) -> None:
        from Z_utils.Z15_lexicon_provider import CREDENTIALS
        # Union of drug (17) + gene (22) = 25 unique entries
        assert len(CREDENTIALS) == 25

    def test_drug_fp_constants_uses_same_credentials(self) -> None:
        from C_generators.C26_drug_fp_constants import CREDENTIALS as DRUG_CREDS
        from Z_utils.Z15_lexicon_provider import CREDENTIALS
        # Drug FP constants should now use the unified set
        assert DRUG_CREDS == set(CREDENTIALS)

    def test_gene_fp_filter_uses_same_credentials(self) -> None:
        from C_generators.C34_gene_fp_filter import GeneFalsePositiveFilter
        from Z_utils.Z15_lexicon_provider import CREDENTIALS
        assert GeneFalsePositiveFilter.CREDENTIALS == set(CREDENTIALS)
