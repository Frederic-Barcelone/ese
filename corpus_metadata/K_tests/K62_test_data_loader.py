"""
Tests for Z_utils.Z12_data_loader and YAML data file integrity.

Validates:
    - All 9 YAML files load without error
    - Type-safety assertions catch YAML boolean coercion
    - Spot-checks known values from each data file
    - Pair list loading works for WRONG_EXPANSION_BLACKLIST
    - List mapping loading works for ENTITY_CATEGORIES
    - Refactored modules still export the same values
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


class TestDataLoaderFunctions:
    """Test the loader utility functions."""

    def test_load_term_set_returns_set(self) -> None:
        from Z_utils.Z12_data_loader import load_term_set
        result = load_term_set("drug_fp_terms.yaml", "bacteria_organisms")
        assert isinstance(result, set)
        assert len(result) > 0

    def test_load_term_list_returns_list(self) -> None:
        from Z_utils.Z12_data_loader import load_term_list
        result = load_term_list("drug_fp_terms.yaml", "biological_suffixes")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_load_mapping_returns_dict(self) -> None:
        from Z_utils.Z12_data_loader import load_mapping
        result = load_mapping("drug_mappings.yaml", "drug_abbreviations")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_load_pair_list_returns_set_of_tuples(self) -> None:
        from Z_utils.Z12_data_loader import load_pair_list
        result = load_pair_list("noise_filters.yaml", "wrong_expansion_blacklist")
        assert isinstance(result, set)
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in result)

    def test_load_list_mapping_returns_dict_of_lists(self) -> None:
        from Z_utils.Z12_data_loader import load_list_mapping
        result = load_list_mapping("biomedical_ner_data.yaml", "entity_categories")
        assert isinstance(result, dict)
        assert len(result) > 0
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, list)
            assert all(isinstance(item, str) for item in v)

    def test_load_list_mapping_non_string_key_guard(self) -> None:
        """Verify that non-string keys in list mappings are caught."""
        from Z_utils.Z12_data_loader import load_list_mapping, _load_yaml
        bad_yaml = {"test_mapping": {123: ["a", "b"]}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(bad_yaml, f)
            temp_path = Path(f.name)
        try:
            with patch("Z_utils.Z12_data_loader._DATA_DIR", temp_path.parent):
                _load_yaml.cache_clear()
                with pytest.raises(TypeError, match="non-string key"):
                    load_list_mapping(temp_path.name, "test_mapping")
        finally:
            temp_path.unlink()
            _load_yaml.cache_clear()

    def test_load_list_mapping_non_list_value_guard(self) -> None:
        """Verify that non-list values in list mappings are caught."""
        from Z_utils.Z12_data_loader import load_list_mapping, _load_yaml
        bad_yaml = {"test_mapping": {"key": "not_a_list"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(bad_yaml, f)
            temp_path = Path(f.name)
        try:
            with patch("Z_utils.Z12_data_loader._DATA_DIR", temp_path.parent):
                _load_yaml.cache_clear()
                with pytest.raises(TypeError, match="not a list"):
                    load_list_mapping(temp_path.name, "test_mapping")
        finally:
            temp_path.unlink()
            _load_yaml.cache_clear()

    def test_boolean_coercion_guard(self) -> None:
        """Verify that unquoted YAML booleans are caught by the loader."""
        from Z_utils.Z12_data_loader import _check_strings
        # Unquoted 'no' in YAML becomes Python False
        with pytest.raises(TypeError, match="non-string"):
            _check_strings([False], "test.yaml", "test_key")
        # Unquoted 'yes' in YAML becomes Python True
        with pytest.raises(TypeError, match="non-string"):
            _check_strings([True], "test.yaml", "test_key")

    def test_mapping_boolean_guard(self) -> None:
        """Verify that non-string keys/values in mappings are caught."""
        from Z_utils.Z12_data_loader import load_mapping, _load_yaml
        # Create a temp YAML with a boolean value
        bad_yaml = {"test_mapping": {"key": True}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(bad_yaml, f)
            temp_path = Path(f.name)
        try:
            with patch("Z_utils.Z12_data_loader._DATA_DIR", temp_path.parent):
                # Clear the cache so our temp file is loaded
                _load_yaml.cache_clear()
                with pytest.raises(TypeError, match="non-string"):
                    load_mapping(temp_path.name, "test_mapping")
        finally:
            temp_path.unlink()
            _load_yaml.cache_clear()


class TestDrugFpTermsYaml:
    """Spot-check drug_fp_terms.yaml values."""

    def test_bacteria_organisms(self) -> None:
        from C_generators.C26_drug_fp_constants import BACTERIA_ORGANISMS
        assert "escherichia coli" in BACTERIA_ORGANISMS
        assert "influenza" in BACTERIA_ORGANISMS

    def test_common_words(self) -> None:
        from C_generators.C26_drug_fp_constants import COMMON_WORDS
        assert "gold" in COMMON_WORDS
        assert "lancet" in COMMON_WORDS

    def test_ner_false_positives(self) -> None:
        from C_generators.C26_drug_fp_constants import NER_FALSE_POSITIVES
        assert "pharmaceutical preparations" in NER_FALSE_POSITIVES
        assert "cocaine" in NER_FALSE_POSITIVES

    def test_biological_suffixes_is_list(self) -> None:
        from C_generators.C26_drug_fp_constants import BIOLOGICAL_SUFFIXES
        assert isinstance(BIOLOGICAL_SUFFIXES, list)
        assert " protein" in BIOLOGICAL_SUFFIXES


class TestDrugMappingsYaml:
    """Spot-check drug_mappings.yaml values."""

    def test_drug_abbreviations(self) -> None:
        from C_generators.C26_drug_fp_constants import DRUG_ABBREVIATIONS
        assert DRUG_ABBREVIATIONS["5-fu"] == "fluorouracil"
        assert DRUG_ABBREVIATIONS["mtx"] == "methotrexate"

    def test_consumer_drug_variants(self) -> None:
        from C_generators.C26_drug_fp_constants import CONSUMER_DRUG_VARIANTS
        assert CONSUMER_DRUG_VARIANTS["sinivastatin"] == "simvastatin"


class TestNoiseFiltersYaml:
    """Spot-check noise_filters.yaml values."""

    def test_obvious_noise(self) -> None:
        from C_generators.C21_noise_filters import OBVIOUS_NOISE
        assert "the" in OBVIOUS_NOISE  # from spaCy stopwords
        assert "a" in OBVIOUS_NOISE    # from single_letters()
        assert "no" in OBVIOUS_NOISE   # from spaCy stopwords
        assert "on" in OBVIOUS_NOISE   # from spaCy stopwords
        # Domain-specific entries still present
        assert "mmhg" in OBVIOUS_NOISE
        assert "roche" in OBVIOUS_NOISE

    def test_wrong_expansion_blacklist(self) -> None:
        from C_generators.C21_noise_filters import WRONG_EXPANSION_BLACKLIST
        assert ("task", "product") in WRONG_EXPANSION_BLACKLIST
        assert ("musk", "musk secretion from musk deer") in WRONG_EXPANSION_BLACKLIST

    def test_bad_long_forms(self) -> None:
        from C_generators.C21_noise_filters import BAD_LONG_FORMS
        assert "product" in BAD_LONG_FORMS

    def test_filter_functions_still_work(self) -> None:
        from C_generators.C21_noise_filters import is_valid_abbreviation_match, is_wrong_expansion
        assert not is_valid_abbreviation_match("the")
        assert is_valid_abbreviation_match("CT")
        assert is_wrong_expansion("TASK", "product")
        assert not is_wrong_expansion("CT", "computed tomography")


class TestGeneFpTermsYaml:
    """Spot-check gene_fp_terms.yaml values."""

    def test_statistical_terms(self) -> None:
        from C_generators.C34_gene_fp_filter import GeneFalsePositiveFilter
        assert "hr" in GeneFalsePositiveFilter.STATISTICAL_TERMS
        assert "or" in GeneFalsePositiveFilter.STATISTICAL_TERMS

    def test_countries_boolean_trap(self) -> None:
        from C_generators.C34_gene_fp_filter import GeneFalsePositiveFilter
        # "no" (Norway code) must be string, not boolean
        assert "no" in GeneFalsePositiveFilter.COUNTRIES

    def test_common_english_words_boolean_trap(self) -> None:
        from C_generators.C34_gene_fp_filter import GeneFalsePositiveFilter
        assert "no" in GeneFalsePositiveFilter.COMMON_ENGLISH_WORDS
        assert "on" in GeneFalsePositiveFilter.COMMON_ENGLISH_WORDS

    def test_filter_instance_works(self) -> None:
        from C_generators.C34_gene_fp_filter import GeneFalsePositiveFilter
        from A_core.A19_gene_models import GeneGeneratorType
        f = GeneFalsePositiveFilter()
        # always_filter should be populated from the union of multiple sets
        assert len(f.always_filter) > 100
        # Common English word should be filtered
        is_fp, reason = f.is_false_positive("set", "some context", GeneGeneratorType.PATTERN_GENE_SYMBOL)
        assert is_fp
        assert reason == "common_english_word"


class TestDiseaseFpTermsYaml:
    """Spot-check disease_fp_terms.yaml values."""

    def test_common_english_fp_terms(self) -> None:
        from C_generators.C24_disease_fp_filter import DiseaseFalsePositiveFilter
        terms = DiseaseFalsePositiveFilter.COMMON_ENGLISH_FP_TERMS
        assert isinstance(terms, set)
        assert "complex" in terms
        assert "syndrome" in terms
        assert "ige" in terms

    def test_generic_multiword_fp_terms(self) -> None:
        from C_generators.C24_disease_fp_filter import DiseaseFalsePositiveFilter
        terms = DiseaseFalsePositiveFilter.GENERIC_MULTIWORD_FP_TERMS
        assert isinstance(terms, set)
        assert "rare disease" in terms
        # "hearing loss" moved to symptom_disease_fp_terms (dataset-aware filtering)
        assert "hearing loss" not in terms

    def test_symptom_disease_fp_terms(self) -> None:
        from C_generators.C24_disease_fp_filter import DiseaseFalsePositiveFilter
        terms = DiseaseFalsePositiveFilter.SYMPTOM_DISEASE_FP_TERMS
        assert isinstance(terms, set)
        assert "hearing loss" in terms
        assert "skin rash" in terms
        assert "bacterial infections" in terms

    def test_chromosome_context_keywords(self) -> None:
        from C_generators.C24_disease_fp_filter import DiseaseFalsePositiveFilter
        kw = DiseaseFalsePositiveFilter.CHROMOSOME_CONTEXT_KEYWORDS
        assert isinstance(kw, list)
        assert "chromosome" in kw
        assert "karyotype" in kw

    def test_disease_context_keywords(self) -> None:
        from C_generators.C24_disease_fp_filter import DiseaseFalsePositiveFilter
        kw = DiseaseFalsePositiveFilter.DISEASE_CONTEXT_KEYWORDS
        assert isinstance(kw, list)
        assert "syndrome" in kw
        assert "patient" in kw

    def test_gene_context_keywords(self) -> None:
        from C_generators.C24_disease_fp_filter import DiseaseFalsePositiveFilter
        kw = DiseaseFalsePositiveFilter.GENE_CONTEXT_KEYWORDS
        assert isinstance(kw, list)
        assert "mutation" in kw
        assert "gene" in kw

    def test_filter_instance_works(self) -> None:
        from C_generators.C24_disease_fp_filter import DiseaseFalsePositiveFilter
        f = DiseaseFalsePositiveFilter()
        # Common English word should be hard-filtered
        should_filter, reason = f.should_filter("complex", "some context")
        assert should_filter
        assert reason == "common_english_word"


class TestLexiconLoadBlacklistYaml:
    """Spot-check lexicon_load_blacklist in drug_fp_terms.yaml."""

    def test_lexicon_load_blacklist(self) -> None:
        from C_generators.C07_strategy_drug import DrugDetector
        bl = DrugDetector.LEXICON_LOAD_BLACKLIST
        assert isinstance(bl, set)
        assert "turkey" in bl
        assert "lasso" in bl
        assert "alkaline phosphatase" in bl


class TestIdentifierDataYaml:
    """Spot-check identifier_data.yaml values."""

    def test_known_genes(self) -> None:
        from C_generators.C00_strategy_identifiers import KNOWN_GENES
        assert isinstance(KNOWN_GENES, set)
        assert "BRCA1" in KNOWN_GENES
        assert "JAG1" in KNOWN_GENES
        assert "GDF2" in KNOWN_GENES
        assert len(KNOWN_GENES) == 39


class TestFeasibilityDataYaml:
    """Spot-check feasibility_data.yaml values."""

    def test_vaccine_types(self) -> None:
        from C_generators.C27_feasibility_patterns import VACCINE_TYPES
        assert isinstance(VACCINE_TYPES, list)
        assert "covid-19" in VACCINE_TYPES
        assert "bcg" in VACCINE_TYPES

    def test_countries(self) -> None:
        from C_generators.C27_feasibility_patterns import COUNTRIES
        assert isinstance(COUNTRIES, set)
        assert "united states" in COUNTRIES
        assert "japan" in COUNTRIES

    def test_country_codes_no_boolean(self) -> None:
        """Verify 'NO' for Norway is a string, not a boolean."""
        from C_generators.C27_feasibility_patterns import COUNTRY_CODES
        assert isinstance(COUNTRY_CODES, dict)
        assert COUNTRY_CODES["norway"] == "NO"
        assert isinstance(COUNTRY_CODES["norway"], str)

    def test_ambiguous_countries(self) -> None:
        from C_generators.C27_feasibility_patterns import AMBIGUOUS_COUNTRIES
        assert isinstance(AMBIGUOUS_COUNTRIES, set)
        assert "georgia" in AMBIGUOUS_COUNTRIES

    def test_country_context_cues(self) -> None:
        from C_generators.C27_feasibility_patterns import COUNTRY_CONTEXT_CUES
        assert isinstance(COUNTRY_CONTEXT_CUES, set)
        assert "sites" in COUNTRY_CONTEXT_CUES


class TestBiomedicalNerDataYaml:
    """Spot-check biomedical_ner_data.yaml values."""

    def test_entity_categories(self) -> None:
        from E_normalization.E10_biomedical_ner_all import ENTITY_CATEGORIES
        assert isinstance(ENTITY_CATEGORIES, dict)
        assert "clinical" in ENTITY_CATEGORIES
        assert "Disease_disorder" in ENTITY_CATEGORIES["clinical"]
        assert "temporal" in ENTITY_CATEGORIES
        assert "Date" in ENTITY_CATEGORIES["temporal"]

    def test_all_entity_types_computed(self) -> None:
        from E_normalization.E10_biomedical_ner_all import ALL_ENTITY_TYPES
        assert isinstance(ALL_ENTITY_TYPES, list)
        assert "Disease_disorder" in ALL_ENTITY_TYPES
        assert "Age" in ALL_ENTITY_TYPES

    def test_entity_to_category_computed(self) -> None:
        from E_normalization.E10_biomedical_ner_all import ENTITY_TO_CATEGORY
        assert isinstance(ENTITY_TO_CATEGORY, dict)
        assert ENTITY_TO_CATEGORY["Disease_disorder"] == "clinical"
        assert ENTITY_TO_CATEGORY["Age"] == "demographics"

    def test_garbage_tokens(self) -> None:
        from E_normalization.E10_biomedical_ner_all import GARBAGE_TOKENS
        assert isinstance(GARBAGE_TOKENS, set)
        assert "the" in GARBAGE_TOKENS   # from spaCy stopwords
        assert "mg" in GARBAGE_TOKENS    # from domain YAML
        assert "rate" in GARBAGE_TOKENS  # from domain YAML


class TestAllYamlFilesLoad:
    """Verify all 9 YAML files load without error."""

    @pytest.mark.parametrize("filename", [
        "drug_fp_terms.yaml",
        "drug_mappings.yaml",
        "noise_filters.yaml",
        "gene_fp_terms.yaml",
        "disease_fp_terms.yaml",
        "identifier_data.yaml",
        "feasibility_data.yaml",
        "biomedical_ner_data.yaml",
    ])
    def test_yaml_file_loads(self, filename: str) -> None:
        from Z_utils.Z12_data_loader import _load_yaml
        data = _load_yaml(filename)
        assert isinstance(data, dict)
        assert len(data) > 0
