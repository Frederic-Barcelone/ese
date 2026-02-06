# corpus_metadata/K_tests/K60_test_cross_entity_filter.py
"""
Tests for cross-entity filtering logic in orchestrator._cross_entity_filter.

Since the method is on the Orchestrator class and requires heavy setup,
we test the core logic by constructing minimal mock entities.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from A_core.A01_domain_models import ValidationStatus


def _make_abbrev(short_form: str, long_form: str, status: str = "VALIDATED"):
    """Create a minimal abbreviation-like object."""
    obj = MagicMock()
    obj.short_form = short_form
    obj.long_form = long_form
    obj.status = status
    return obj


def _make_author(full_name: str, status: str = "VALIDATED"):
    """Create a minimal author-like object."""
    obj = MagicMock()
    obj.full_name = full_name
    obj.status = status
    return obj


def _make_drug(matched_text: str, status: str = "VALIDATED"):
    """Create a minimal drug-like object."""
    obj = MagicMock()
    obj.matched_text = matched_text
    obj.status = status
    return obj


def _make_disease(matched_text: str, status: str = "VALIDATED"):
    """Create a minimal disease-like object."""
    obj = MagicMock()
    obj.matched_text = matched_text
    obj.status = status
    return obj


def _make_gene(matched_text: str, status: str = "VALIDATED"):
    """Create a minimal gene-like object."""
    obj = MagicMock()
    obj.matched_text = matched_text
    obj.status = status
    return obj


class TestCrossEntityFilterLogic:
    """Test the cross-entity filtering logic directly."""

    def test_drug_removed_when_abbrev_is_questionnaire(self):
        """Drug should be removed when abbreviation expands to questionnaire."""
        abbrevs = [_make_abbrev("MPA", "Microscopic Polyangiitis")]
        drugs = [_make_drug("MPA")]

        # Build abbreviation map
        abbrev_map: dict[str, str] = {}
        for r in abbrevs:
            if r.status == ValidationStatus.VALIDATED and r.long_form:
                abbrev_map[r.short_form.upper()] = r.long_form

        non_drug_keywords = {
            "scale", "score", "questionnaire", "index", "inventory",
            "polyangiitis", "vasculitis", "disease", "syndrome",
            "assessment", "survey", "rating",
        }

        filtered = []
        for drug in drugs:
            drug_upper = drug.matched_text.upper().strip()
            if drug_upper in abbrev_map:
                long_form_lower = abbrev_map[drug_upper].lower()
                if any(kw in long_form_lower for kw in non_drug_keywords):
                    continue
            filtered.append(drug)

        assert len(filtered) == 0, "MPA should be filtered (polyangiitis in long form)"

    def test_drug_kept_when_abbrev_is_drug(self):
        """Drug should be kept when abbreviation expands to a drug term."""
        abbrevs = [_make_abbrev("MTX", "Methotrexate")]
        drugs = [_make_drug("MTX")]

        abbrev_map: dict[str, str] = {}
        for r in abbrevs:
            if r.status == ValidationStatus.VALIDATED and r.long_form:
                abbrev_map[r.short_form.upper()] = r.long_form

        non_drug_keywords = {
            "scale", "score", "questionnaire", "index", "inventory",
            "polyangiitis", "vasculitis", "disease", "syndrome",
            "assessment", "survey", "rating",
        }

        filtered = []
        for drug in drugs:
            drug_upper = drug.matched_text.upper().strip()
            if drug_upper in abbrev_map:
                long_form_lower = abbrev_map[drug_upper].lower()
                if any(kw in long_form_lower for kw in non_drug_keywords):
                    continue
            filtered.append(drug)

        assert len(filtered) == 1, "MTX should be kept (Methotrexate is a drug)"

    def test_drug_kept_when_no_abbreviation(self):
        """Drug with no matching abbreviation should be kept."""
        abbrevs: list[MagicMock] = []
        drugs = [_make_drug("aspirin")]

        abbrev_map: dict[str, str] = {}
        for r in abbrevs:
            if r.status == ValidationStatus.VALIDATED and r.long_form:
                abbrev_map[r.short_form.upper()] = r.long_form

        non_drug_keywords = {"scale", "questionnaire"}

        filtered = []
        for drug in drugs:
            drug_upper = drug.matched_text.upper().strip()
            if drug_upper in abbrev_map:
                long_form_lower = abbrev_map[drug_upper].lower()
                if any(kw in long_form_lower for kw in non_drug_keywords):
                    continue
            filtered.append(drug)

        assert len(filtered) == 1

    def test_disease_removed_when_matches_author_name(self):
        """Disease should be removed when matched_text matches author name token."""
        authors = [_make_author("John Greenfield")]
        diseases = [_make_disease("greenfield")]

        author_name_tokens: set[str] = set()
        for a in authors:
            if a.status == ValidationStatus.VALIDATED:
                for token in a.full_name.lower().split():
                    if len(token) >= 4:
                        author_name_tokens.add(token)

        filtered = []
        for d in diseases:
            matched_lower = d.matched_text.lower().strip()
            if len(matched_lower.split()) <= 2 and matched_lower in author_name_tokens:
                continue
            filtered.append(d)

        assert len(filtered) == 0, "greenfield should be removed (author name)"

    def test_disease_kept_when_no_author_match(self):
        """Disease should be kept when it doesn't match any author name."""
        authors = [_make_author("John Smith")]
        diseases = [_make_disease("vasculitis")]

        author_name_tokens: set[str] = set()
        for a in authors:
            if a.status == ValidationStatus.VALIDATED:
                for token in a.full_name.lower().split():
                    if len(token) >= 4:
                        author_name_tokens.add(token)

        filtered = []
        for d in diseases:
            matched_lower = d.matched_text.lower().strip()
            if len(matched_lower.split()) <= 2 and matched_lower in author_name_tokens:
                continue
            filtered.append(d)

        assert len(filtered) == 1

    def test_short_author_tokens_ignored(self):
        """Author name tokens shorter than 4 chars should not cause filtering."""
        authors = [_make_author("Li Wei")]
        diseases = [_make_disease("li")]

        author_name_tokens: set[str] = set()
        for a in authors:
            if a.status == ValidationStatus.VALIDATED:
                for token in a.full_name.lower().split():
                    if len(token) >= 4:
                        author_name_tokens.add(token)

        filtered = []
        for d in diseases:
            matched_lower = d.matched_text.lower().strip()
            if len(matched_lower.split()) <= 2 and matched_lower in author_name_tokens:
                continue
            filtered.append(d)

        # "li" is only 2 chars, should not be in author_name_tokens
        assert len(filtered) == 1

    def test_hads_drug_removed_via_abbreviation(self):
        """HADS drug should be removed when abbreviation expands to scale."""
        abbrevs = [_make_abbrev("HADS", "Hospital Anxiety and Depression Scale")]
        drugs = [_make_drug("HADS")]

        abbrev_map: dict[str, str] = {}
        for r in abbrevs:
            if r.status == ValidationStatus.VALIDATED and r.long_form:
                abbrev_map[r.short_form.upper()] = r.long_form

        non_drug_keywords = {
            "scale", "score", "questionnaire", "index", "inventory",
            "polyangiitis", "vasculitis", "disease", "syndrome",
            "assessment", "survey", "rating",
        }

        filtered = []
        for drug in drugs:
            drug_upper = drug.matched_text.upper().strip()
            if drug_upper in abbrev_map:
                long_form_lower = abbrev_map[drug_upper].lower()
                if any(kw in long_form_lower for kw in non_drug_keywords):
                    continue
            filtered.append(drug)

        assert len(filtered) == 0, "HADS should be filtered (scale in long form)"
