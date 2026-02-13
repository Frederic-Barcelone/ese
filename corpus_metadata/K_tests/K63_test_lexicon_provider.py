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
    - build_country_names() covers pycountry + aliases
    - build_country_code_mapping() maps names to ISO alpha-2
    - build_country_alpha2_codes() covers all 249 codes + eu
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
        assert "the" in result

    def test_has_many_entries(self) -> None:
        from Z_utils.Z15_lexicon_provider import stopwords_base
        result = stopwords_base()
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
        assert "a" in result

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
        assert "january" in result

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


class TestBuildCountryNames:
    """Tests for build_country_names()."""

    def test_returns_frozenset(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_country_names
        result = build_country_names()
        assert isinstance(result, frozenset)

    def test_has_many_entries(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_country_names
        result = build_country_names()
        # pycountry has 249 countries, plus common_name/official_name variants + aliases
        assert len(result) > 250

    def test_contains_common_countries(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_country_names
        result = build_country_names()
        for name in ["france", "germany", "japan", "brazil", "india", "australia"]:
            assert name in result, f"Expected '{name}' in country names"

    def test_contains_aliases(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_country_names
        result = build_country_names()
        for alias in ["usa", "us", "uk", "korea", "hong kong"]:
            assert alias in result, f"Expected alias '{alias}' in country names"

    def test_all_old_yaml_countries_present(self) -> None:
        """All 50 countries from old feasibility_data.yaml are still covered."""
        from Z_utils.Z15_lexicon_provider import build_country_names
        result = build_country_names()
        old_countries = [
            "united states", "usa", "us", "germany", "france", "uk",
            "united kingdom", "italy", "spain", "canada", "australia",
            "japan", "china", "brazil", "netherlands", "belgium",
            "switzerland", "austria", "sweden", "norway", "denmark",
            "finland", "poland", "czech republic", "hungary", "israel",
            "south korea", "korea", "taiwan", "india", "russia",
            "mexico", "argentina", "turkey", "greece", "portugal",
            "ireland", "new zealand", "singapore", "hong kong",
            "thailand", "malaysia", "south africa", "egypt", "chile",
            "colombia", "peru", "ukraine", "romania", "bulgaria",
        ]
        for country in old_countries:
            assert country in result, f"Old YAML country '{country}' missing"

    def test_all_lowercase(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_country_names
        result = build_country_names()
        for name in result:
            assert name == name.lower(), f"Country name '{name}' is not lowercase"


class TestBuildCountryCodeMapping:
    """Tests for build_country_code_mapping()."""

    def test_returns_dict(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_country_code_mapping
        result = build_country_code_mapping()
        assert isinstance(result, dict)

    def test_has_many_entries(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_country_code_mapping
        result = build_country_code_mapping()
        assert len(result) > 250

    def test_common_mappings(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_country_code_mapping
        result = build_country_code_mapping()
        assert result["france"] == "FR"
        assert result["germany"] == "DE"
        assert result["japan"] == "JP"
        assert result["norway"] == "NO"

    def test_alias_mappings(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_country_code_mapping
        result = build_country_code_mapping()
        assert result["usa"] == "US"
        assert result["us"] == "US"
        assert result["uk"] == "GB"
        assert result["korea"] == "KR"

    def test_all_old_yaml_mappings_present(self) -> None:
        """All 34 mappings from old feasibility_data.yaml country_codes are covered."""
        from Z_utils.Z15_lexicon_provider import build_country_code_mapping
        result = build_country_code_mapping()
        old_mappings = {
            "united states": "US", "usa": "US", "us": "US",
            "germany": "DE", "france": "FR", "uk": "GB",
            "united kingdom": "GB", "italy": "IT", "spain": "ES",
            "canada": "CA", "australia": "AU", "japan": "JP",
            "china": "CN", "brazil": "BR", "netherlands": "NL",
            "belgium": "BE", "switzerland": "CH", "austria": "AT",
            "sweden": "SE", "norway": "NO", "denmark": "DK",
            "finland": "FI", "poland": "PL", "czech republic": "CZ",
            "hungary": "HU", "israel": "IL", "south korea": "KR",
            "korea": "KR", "taiwan": "TW", "india": "IN",
            "russia": "RU", "mexico": "MX", "argentina": "AR",
            "turkey": "TR",
        }
        for name, code in old_mappings.items():
            assert result[name] == code, f"Old mapping '{name}' → '{code}' missing or wrong"

    def test_values_are_uppercase_alpha2(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_country_code_mapping
        result = build_country_code_mapping()
        for name, code in result.items():
            assert code == code.upper(), f"Code '{code}' for '{name}' not uppercase"
            assert len(code) == 2, f"Code '{code}' for '{name}' not 2 chars"


class TestBuildCountryAlpha2Codes:
    """Tests for build_country_alpha2_codes()."""

    def test_returns_frozenset(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_country_alpha2_codes
        result = build_country_alpha2_codes()
        assert isinstance(result, frozenset)

    def test_has_250_entries(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_country_alpha2_codes
        result = build_country_alpha2_codes()
        # 249 ISO codes + "eu"
        assert len(result) == 250

    def test_contains_common_codes(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_country_alpha2_codes
        result = build_country_alpha2_codes()
        for code in ["us", "gb", "de", "fr", "jp", "cn", "in", "br", "au"]:
            assert code in result, f"Expected '{code}' in alpha-2 codes"

    def test_contains_eu(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_country_alpha2_codes
        result = build_country_alpha2_codes()
        assert "eu" in result

    def test_all_old_yaml_codes_present(self) -> None:
        """All valid ISO codes from old gene_fp_terms.yaml countries are covered.

        Note: "uk" was in the old YAML but is not a valid ISO 3166-1 alpha-2 code
        (the correct code is "gb"). pycountry correctly uses "gb" instead.
        """
        from Z_utils.Z15_lexicon_provider import build_country_alpha2_codes
        result = build_country_alpha2_codes()
        old_codes = [
            "us", "eu", "ca", "au", "de", "fr", "jp", "cn", "in",
            "it", "es", "nl", "be", "ch", "at", "se", "no", "dk", "fi",
            "pl", "cz", "hu", "ro", "bg", "gr", "pt", "ie", "nz", "sg",
            "hk", "tw", "kr", "mx", "br", "ar", "cl", "co", "za", "eg",
        ]
        for code in old_codes:
            assert code in result, f"Old YAML code '{code}' missing"
        # "uk" replaced by correct ISO code "gb"
        assert "gb" in result

    def test_all_lowercase(self) -> None:
        from Z_utils.Z15_lexicon_provider import build_country_alpha2_codes
        result = build_country_alpha2_codes()
        for code in result:
            assert code == code.lower(), f"Code '{code}' is not lowercase"
