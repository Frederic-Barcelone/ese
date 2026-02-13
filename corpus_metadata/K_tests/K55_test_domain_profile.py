# corpus_metadata/K_tests/K55_test_domain_profile.py
"""
Tests for A_core.A15_domain_profile module.

Tests domain profiles, confidence adjustments, and profile loading.
"""

from __future__ import annotations

import tempfile

import pytest

from A_core.A15_domain_profile import (
    ConfidenceAdjustments,
    DomainProfile,
    load_domain_profile,
    get_available_profiles,
)


class TestConfidenceAdjustments:
    """Tests for ConfidenceAdjustments dataclass."""

    def test_default_values(self):
        adj = ConfidenceAdjustments()
        assert adj.generic_disease_term == -0.30
        assert adj.physiological_system == -0.40
        assert adj.short_match_no_context == -0.25
        assert adj.chromosome_pattern == -0.50
        assert adj.journal_citation == -0.40
        assert adj.priority_disease_boost == 0.15
        assert adj.priority_journal_boost == 0.10
        assert adj.domain_noise_penalty == -0.20

    def test_custom_values(self):
        adj = ConfidenceAdjustments(
            generic_disease_term=-0.50,
            priority_disease_boost=0.25,
        )
        assert adj.generic_disease_term == -0.50
        assert adj.priority_disease_boost == 0.25


class TestDomainProfile:
    """Tests for DomainProfile class."""

    def test_create_minimal(self):
        profile = DomainProfile(name="test")
        assert profile.name == "test"
        assert profile.description == ""
        assert len(profile.priority_diseases) == 0

    def test_create_with_data(self):
        profile = DomainProfile(
            name="nephrology",
            description="Kidney disease focus",
            priority_diseases={"iga nephropathy", "fsgs"},
            noise_terms={"kidney disease"},
        )
        assert "iga nephropathy" in profile.priority_diseases
        assert "kidney disease" in profile.noise_terms

    def test_is_priority_disease(self):
        profile = DomainProfile(
            name="test",
            priority_diseases={"iga nephropathy", "fsgs"},
        )
        assert profile.is_priority_disease("IgA Nephropathy") is True
        assert profile.is_priority_disease("iga nephropathy") is True
        assert profile.is_priority_disease("diabetes") is False

    def test_is_noise_term(self):
        profile = DomainProfile(
            name="test",
            noise_terms={"kidney disease", "renal disease"},
        )
        assert profile.is_noise_term("kidney disease") is True
        assert profile.is_noise_term("KIDNEY DISEASE") is True
        assert profile.is_noise_term("IgA nephropathy") is False


class TestGetConfidenceAdjustment:
    """Tests for DomainProfile.get_confidence_adjustment method."""

    @pytest.fixture
    def test_profile(self):
        return DomainProfile(
            name="test",
            priority_diseases={"iga nephropathy"},
            noise_terms={"kidney disease"},
            generic_terms={"disease", "syndrome"},
            physiological_systems={"cns", "immune system"},
        )

    def test_generic_term_penalty(self, test_profile):
        adj = test_profile.get_confidence_adjustment("disease")
        assert adj < 0

    def test_physiological_system_penalty(self, test_profile):
        adj = test_profile.get_confidence_adjustment("CNS")
        assert adj < 0

    def test_priority_disease_boost(self, test_profile):
        adj = test_profile.get_confidence_adjustment(
            "IgA Nephropathy",
            context="patient with IgA nephropathy treatment",
        )
        assert adj > 0

    def test_noise_term_penalty(self, test_profile):
        adj = test_profile.get_confidence_adjustment("kidney disease")
        assert adj < 0

    def test_short_match_no_context_penalty(self, test_profile):
        adj = test_profile.get_confidence_adjustment(
            "CKD",
            context="",  # No disease context
            is_short_match=True,
        )
        assert adj < 0

    def test_citation_context_penalty(self, test_profile):
        adj = test_profile.get_confidence_adjustment(
            "nephropathy",
            is_citation_context=True,
        )
        assert adj < 0

    def test_adjustment_clamping(self):
        # Create profile with many penalties
        profile = DomainProfile(
            name="test",
            generic_terms={"test"},
            physiological_systems={"test"},
            noise_terms={"test"},
        )
        adj = profile.get_confidence_adjustment(
            "test",
            is_short_match=True,
            is_citation_context=True,
        )
        assert adj >= -1.0
        assert adj <= 0.3


class TestShouldHardFilter:
    """Tests for DomainProfile.should_hard_filter method."""

    @pytest.fixture
    def test_profile(self):
        return DomainProfile(
            name="test",
            generic_terms={"disease", "syndrome"},
        )

    def test_generic_term_no_context(self, test_profile):
        should_filter, reason = test_profile.should_hard_filter(
            "disease",
            context="",
        )
        assert should_filter is True
        assert reason == "generic_term_no_context"

    def test_generic_term_with_disease_context(self, test_profile):
        should_filter, reason = test_profile.should_hard_filter(
            "disease",
            context="the patient was diagnosed with a rare disease",
        )
        assert should_filter is False

    def test_non_generic_term(self, test_profile):
        should_filter, reason = test_profile.should_hard_filter(
            "IgA nephropathy",
            context="",
        )
        assert should_filter is False

    def test_multi_word_generic_term(self, test_profile):
        # Multi-word terms are less likely to be filtered
        profile = DomainProfile(
            name="test",
            generic_terms={"kidney disease"},
        )
        should_filter, reason = profile.should_hard_filter(
            "kidney disease",
            context="",
        )
        assert should_filter is False  # Multi-word, so not filtered


class TestHasDiseaseContext:
    """Tests for DomainProfile._has_disease_context method."""

    def test_with_disease_keywords(self):
        profile = DomainProfile(name="test")
        assert profile._has_disease_context("patient diagnosed with syndrome") is True
        assert profile._has_disease_context("clinical trial for treatment") is True
        assert profile._has_disease_context("rare disease prevalence") is True

    def test_without_disease_keywords(self):
        profile = DomainProfile(name="test")
        assert profile._has_disease_context("hello world") is False
        assert profile._has_disease_context("") is False


class TestLoadDomainProfile:
    """Tests for load_domain_profile function."""

    def test_load_generic(self):
        profile = load_domain_profile("generic")
        assert profile.name == "generic"
        assert "disease" in profile.generic_terms

    def test_load_nephrology(self):
        profile = load_domain_profile("nephrology")
        assert profile.name == "nephrology"
        assert "iga nephropathy" in profile.priority_diseases

    def test_load_oncology(self):
        profile = load_domain_profile("oncology")
        assert profile.name == "oncology"
        assert "nsclc" in profile.priority_diseases

    def test_load_pulmonology(self):
        profile = load_domain_profile("pulmonology")
        assert profile.name == "pulmonology"
        assert "pah" in profile.priority_diseases

    def test_load_unknown_falls_back_to_generic(self):
        profile = load_domain_profile("unknown_domain")
        assert profile.name == "generic"

    def test_load_case_insensitive(self):
        profile = load_domain_profile("NEPHROLOGY")
        assert profile.name == "nephrology"

    def test_load_from_yaml(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("""
name: custom
description: Custom profile
priority_diseases:
  - custom disease
  - another disease
noise_terms:
  - noise
adjustments:
  priority_disease_boost: 0.20
""")
            f.flush()
            profile = load_domain_profile(f.name)

        assert profile.name == "custom"
        assert "custom disease" in profile.priority_diseases
        assert profile.adjustments.priority_disease_boost == 0.20

    def test_load_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_domain_profile("/nonexistent/path.yaml")

    def test_load_with_config_overrides(self):
        config = {
            "domain_profile": {
                "priority_diseases": ["override disease"],
            }
        }
        profile = load_domain_profile("generic", config=config)
        assert "override disease" in profile.priority_diseases


class TestGetAvailableProfiles:
    """Tests for get_available_profiles function."""

    def test_returns_list(self):
        profiles = get_available_profiles()
        assert isinstance(profiles, list)
        assert len(profiles) >= 4
        assert "generic" in profiles

    def test_contains_builtin_profiles(self):
        profiles = get_available_profiles()
        assert "generic" in profiles
        assert "nephrology" in profiles
        assert "oncology" in profiles
        assert "pulmonology" in profiles
