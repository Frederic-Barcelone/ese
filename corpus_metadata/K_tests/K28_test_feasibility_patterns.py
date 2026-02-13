# corpus_metadata/K_tests/K28_test_feasibility_patterns.py
"""
Tests for C_generators.C27_feasibility_patterns module.

Tests pattern constants for clinical trial feasibility extraction.
"""

from __future__ import annotations

import re

from C_generators.C27_feasibility_patterns import (
    EPIDEMIOLOGY_ANCHORS,
    INCLUSION_MARKERS,
    EXCLUSION_MARKERS,
    CRITERION_CATEGORIES,
    PREVALENCE_PATTERNS,
    INCIDENCE_PATTERNS,
    DEMOGRAPHICS_PATTERNS,
    JOURNEY_PHASE_KEYWORDS,
    DURATION_PATTERN,
    ENDPOINT_PATTERNS,
    SCREENING_YIELD_PATTERNS,
    VACCINATION_PATTERNS,
    SITE_COUNT_PATTERNS,
    COUNTRIES,
    COUNTRY_CODES,
    AMBIGUOUS_COUNTRIES,
    EXPECTED_SECTIONS,
)
from A_core.A07_feasibility_models import (
    EndpointType,
    PatientJourneyPhaseType,
    FeasibilityFieldType,
)


class TestEpidemiologyAnchors:
    """Tests for EPIDEMIOLOGY_ANCHORS patterns."""

    def test_prevalence_anchor(self):
        text = "prevalence of the disease"
        assert any(re.search(p, text, re.I) for p in EPIDEMIOLOGY_ANCHORS)

    def test_incidence_anchor(self):
        text = "incidence rate was 5 per million"
        assert any(re.search(p, text, re.I) for p in EPIDEMIOLOGY_ANCHORS)

    def test_per_million_anchor(self):
        text = "5.2 per million population"
        assert any(re.search(p, text, re.I) for p in EPIDEMIOLOGY_ANCHORS)

    def test_registry_anchor(self):
        text = "registry-based study"
        assert any(re.search(p, text, re.I) for p in EPIDEMIOLOGY_ANCHORS)


class TestInclusionMarkers:
    """Tests for INCLUSION_MARKERS patterns."""

    def test_inclusion_criteria(self):
        text = "Inclusion criteria: Age >= 18"
        assert any(re.search(p, text, re.I) for p in INCLUSION_MARKERS)

    def test_patients_eligible(self):
        text = "Patients were eligible if they met"
        assert any(re.search(p, text, re.I) for p in INCLUSION_MARKERS)

    def test_must_have(self):
        text = "Participants must have documented diagnosis"
        assert any(re.search(p, text, re.I) for p in INCLUSION_MARKERS)


class TestExclusionMarkers:
    """Tests for EXCLUSION_MARKERS patterns."""

    def test_exclusion_criteria(self):
        text = "Exclusion criteria: Prior treatment"
        assert any(re.search(p, text, re.I) for p in EXCLUSION_MARKERS)

    def test_patients_excluded(self):
        text = "Patients were excluded if they had"
        assert any(re.search(p, text, re.I) for p in EXCLUSION_MARKERS)

    def test_will_be_excluded(self):
        text = "Subjects will be excluded from the study"
        assert any(re.search(p, text, re.I) for p in EXCLUSION_MARKERS)


class TestCriterionCategories:
    """Tests for CRITERION_CATEGORIES patterns."""

    def test_age_category(self):
        pattern = CRITERION_CATEGORIES["age"]
        assert re.search(pattern, "age 18 to 65 years", re.I)
        assert re.search(pattern, "18 years old", re.I)

    def test_disease_definition_category(self):
        pattern = CRITERION_CATEGORIES["disease_definition"]
        assert re.search(pattern, "genetically confirmed disease", re.I)
        assert re.search(pattern, "pathogenic variant detected", re.I)

    def test_prior_treatment_category(self):
        pattern = CRITERION_CATEGORIES["prior_treatment"]
        assert re.search(pattern, "prior treatment with", re.I)
        assert re.search(pattern, "treatment-naive patients", re.I)

    def test_biomarker_category(self):
        pattern = CRITERION_CATEGORIES["biomarker"]
        assert re.search(pattern, "positive for HLA-B27", re.I)
        assert re.search(pattern, "mutation in BRCA1", re.I)


class TestPrevalencePatterns:
    """Tests for PREVALENCE_PATTERNS."""

    def test_per_million(self):
        # Test various prevalence patterns
        # Pattern expects: prevalence of/: NUMBER in/per NUMBER [million|...]
        text = "prevalence: 5.2 per 100000"
        assert any(re.search(p, text, re.I) for p in PREVALENCE_PATTERNS)

    def test_percentage(self):
        text = "1.5% of patients"
        # Check if any pattern matches percentage format
        assert any(re.search(p, text, re.I) for p in PREVALENCE_PATTERNS)


class TestIncidencePatterns:
    """Tests for INCIDENCE_PATTERNS."""

    def test_cases_per_year(self):
        text = "2.5 new cases per million per year"
        assert any(re.search(p, text, re.I) for p in INCIDENCE_PATTERNS)


class TestDemographicsPatterns:
    """Tests for DEMOGRAPHICS_PATTERNS."""

    def test_median_age(self):
        text = "median age of 45 years"
        assert any(re.search(p, text, re.I) for p in DEMOGRAPHICS_PATTERNS)

    def test_gender_percentage(self):
        text = "52% female participants"
        assert any(re.search(p, text, re.I) for p in DEMOGRAPHICS_PATTERNS)


class TestJourneyPhaseKeywords:
    """Tests for JOURNEY_PHASE_KEYWORDS."""

    def test_screening_phase(self):
        keywords = JOURNEY_PHASE_KEYWORDS[PatientJourneyPhaseType.SCREENING]
        assert "screening period" in keywords
        assert "baseline assessment" in keywords

    def test_treatment_phase(self):
        keywords = JOURNEY_PHASE_KEYWORDS[PatientJourneyPhaseType.TREATMENT]
        assert "treatment period" in keywords
        assert "active treatment" in keywords

    def test_follow_up_phase(self):
        keywords = JOURNEY_PHASE_KEYWORDS[PatientJourneyPhaseType.FOLLOW_UP]
        assert "follow-up period" in keywords
        assert "post-treatment" in keywords


class TestDurationPattern:
    """Tests for DURATION_PATTERN."""

    def test_weeks(self):
        match = DURATION_PATTERN.search("12 weeks of treatment")
        assert match
        assert match.group(1) == "12"
        assert "week" in match.group(2).lower()

    def test_months(self):
        match = DURATION_PATTERN.search("6 months follow-up")
        assert match
        assert match.group(1) == "6"

    def test_range(self):
        match = DURATION_PATTERN.search("12-24 weeks duration")
        assert match


class TestEndpointPatterns:
    """Tests for ENDPOINT_PATTERNS."""

    def test_primary_endpoint(self):
        patterns = ENDPOINT_PATTERNS[EndpointType.PRIMARY]
        text = "primary endpoint was overall survival"
        assert any(re.search(p, text, re.I) for p in patterns)

    def test_secondary_endpoint(self):
        patterns = ENDPOINT_PATTERNS[EndpointType.SECONDARY]
        text = "secondary endpoints included"
        assert any(re.search(p, text, re.I) for p in patterns)

    def test_safety_endpoint(self):
        patterns = ENDPOINT_PATTERNS[EndpointType.SAFETY]
        text = "safety endpoints adverse events"
        assert any(re.search(p, text, re.I) for p in patterns)


class TestScreeningYieldPatterns:
    """Tests for SCREENING_YIELD_PATTERNS."""

    def test_patients_screened(self):
        text = "500 patients were screened"
        assert any(re.search(p, text, re.I) for p in SCREENING_YIELD_PATTERNS)

    def test_patients_randomized(self):
        text = "200 patients randomized"
        assert any(re.search(p, text, re.I) for p in SCREENING_YIELD_PATTERNS)

    def test_screen_failures(self):
        text = "120 screen failures"
        assert any(re.search(p, text, re.I) for p in SCREENING_YIELD_PATTERNS)


class TestVaccinationPatterns:
    """Tests for VACCINATION_PATTERNS."""

    def test_covid_vaccination(self):
        text = "COVID-19 vaccination required"
        assert any(re.search(p, text, re.I) for p in VACCINATION_PATTERNS)

    def test_live_vaccine_restriction(self):
        text = "no live vaccine within 4 weeks"
        assert any(re.search(p, text, re.I) for p in VACCINATION_PATTERNS)


class TestSiteCountPatterns:
    """Tests for SITE_COUNT_PATTERNS."""

    def test_sites_in_countries(self):
        text = "50 sites in 12 countries"
        assert any(re.search(p, text, re.I) for p in SITE_COUNT_PATTERNS)

    def test_multinational(self):
        text = "multinational study in 15 countries"
        assert any(re.search(p, text, re.I) for p in SITE_COUNT_PATTERNS)


class TestCountries:
    """Tests for COUNTRIES set."""

    def test_common_countries(self):
        expected = ["united states", "usa", "germany", "france", "japan", "china"]
        for country in expected:
            assert country in COUNTRIES

    def test_is_set(self):
        assert isinstance(COUNTRIES, (set, frozenset))
        assert len(COUNTRIES) > 200


class TestCountryCodes:
    """Tests for COUNTRY_CODES mapping."""

    def test_us_codes(self):
        assert COUNTRY_CODES["united states"] == "US"
        assert COUNTRY_CODES["usa"] == "US"
        assert COUNTRY_CODES["us"] == "US"

    def test_european_codes(self):
        assert COUNTRY_CODES["germany"] == "DE"
        assert COUNTRY_CODES["france"] == "FR"
        assert COUNTRY_CODES["uk"] == "GB"


class TestAmbiguousCountries:
    """Tests for AMBIGUOUS_COUNTRIES set."""

    def test_known_ambiguous(self):
        expected = ["georgia", "jordan", "turkey", "chad"]
        for country in expected:
            assert country in AMBIGUOUS_COUNTRIES

    def test_is_set(self):
        assert isinstance(AMBIGUOUS_COUNTRIES, set)
        assert len(AMBIGUOUS_COUNTRIES) >= 4


class TestExpectedSections:
    """Tests for EXPECTED_SECTIONS mapping."""

    def test_eligibility_sections(self):
        sections = EXPECTED_SECTIONS[FeasibilityFieldType.ELIGIBILITY_INCLUSION]
        assert "eligibility" in sections
        assert "methods" in sections

    def test_epidemiology_sections(self):
        sections = EXPECTED_SECTIONS[FeasibilityFieldType.EPIDEMIOLOGY_PREVALENCE]
        assert "epidemiology" in sections

    def test_endpoint_sections(self):
        sections = EXPECTED_SECTIONS[FeasibilityFieldType.STUDY_ENDPOINT]
        assert "endpoints" in sections or "methods" in sections
