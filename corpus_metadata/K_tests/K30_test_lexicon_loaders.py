# corpus_metadata/K_tests/K30_test_lexicon_loaders.py
"""
Tests for C_generators.C22_lexicon_loaders module.

Tests lexicon loading mixin methods.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from C_generators.C22_lexicon_loaders import LexiconLoaderMixin
from C_generators.C21_noise_filters import LexiconEntry


class MockLexiconLoader(LexiconLoaderMixin):
    """Mock class for testing the mixin."""

    def __init__(self):
        self.abbrev_entries: list[LexiconEntry] = []
        self.entity_kp = MagicMock()
        self.entity_canonical: dict[str, str] = {}
        self.entity_source: dict[str, str] = {}
        self.entity_ids: dict[str, list[dict[str, str]]] = {}
        self._lexicon_stats: list[tuple] = []


@pytest.fixture
def loader():
    return MockLexiconLoader()


class TestLexiconLoaderMixin:
    """Tests for LexiconLoaderMixin class."""

    def test_has_expected_methods(self, loader):
        """Test that mixin has expected methods."""
        expected_methods = [
            "_load_abbrev_lexicon",
            "_load_disease_lexicon",
            "_load_orphanet_lexicon",
            "_load_rare_disease_acronyms",
            "_load_umls_tsv",
            "_load_anca_lexicon",
            "_load_igan_lexicon",
            "_load_pah_lexicon",
            "_load_trial_acronyms",
            "_load_pro_scales",
            "_load_pharma_companies",
            "_load_meta_inventory",
            "_load_mondo_lexicon",
            "_load_chembl_lexicon",
            "_print_lexicon_summary",
            "_extract_identifiers",
        ]
        for method in expected_methods:
            assert hasattr(loader, method), f"Missing method: {method}"

    def test_load_abbrev_nonexistent(self, loader):
        """Test loading nonexistent abbreviation lexicon."""
        loader._load_abbrev_lexicon(Path("/nonexistent/path.json"))
        assert len(loader.abbrev_entries) == 0

    def test_load_disease_nonexistent(self, loader):
        """Test loading nonexistent disease lexicon."""
        loader._load_disease_lexicon(Path("/nonexistent/path.json"))
        # No terms added since file doesn't exist
        assert loader.entity_kp.add_keyword.call_count == 0

    def test_load_orphanet_nonexistent(self, loader):
        """Test loading nonexistent Orphanet lexicon."""
        loader._load_orphanet_lexicon(Path("/nonexistent/path.json"))
        assert loader.entity_kp.add_keyword.call_count == 0

    def test_print_lexicon_summary_empty(self, loader):
        """Test printing summary with no lexicons."""
        loader._print_lexicon_summary()
        assert len(loader._lexicon_stats) == 0

    def test_print_lexicon_summary_with_stats(self, loader):
        """Test printing summary with some lexicons."""
        loader._lexicon_stats = [
            ("Test Lexicon", 100, "test.json"),
            ("Another Lexicon", 200, "another.json"),
        ]
        loader._print_lexicon_summary()
        assert len(loader._lexicon_stats) == 2


class TestExtractIdentifiers:
    """Tests for _extract_identifiers method."""

    def test_empty_identifiers(self, loader):
        """Test with empty identifiers."""
        result = loader._extract_identifiers({})
        assert result == []

    def test_none_identifiers(self, loader):
        """Test with None identifiers."""
        result = loader._extract_identifiers(None)
        assert result == []

    def test_orpha_identifier(self, loader):
        """Test ORPHA identifier extraction."""
        identifiers = {"ORPHA": "123"}
        result = loader._extract_identifiers(identifiers)
        assert len(result) == 1
        assert result[0]["source"] == "Orphanet"
        assert result[0]["id"] == "ORPHA:123"

    def test_mesh_identifier(self, loader):
        """Test MESH identifier extraction."""
        identifiers = {"MESH": "D012345"}
        result = loader._extract_identifiers(identifiers)
        assert len(result) == 1
        assert result[0]["source"] == "MeSH"
        assert result[0]["id"] == "MESH:D012345"

    def test_multiple_identifiers(self, loader):
        """Test multiple identifier extraction."""
        identifiers = {
            "ORPHA": "123",
            "MONDO": "MONDO:0000456",
            "ICD10": "E10.9",
        }
        result = loader._extract_identifiers(identifiers)
        assert len(result) == 3
        sources = [r["source"] for r in result]
        assert "Orphanet" in sources
        assert "MONDO" in sources
        assert "ICD-10" in sources


class TestLexiconStats:
    """Tests for lexicon statistics tracking."""

    def test_stats_empty_initially(self, loader):
        """Test that stats are empty initially."""
        assert len(loader._lexicon_stats) == 0

    def test_stats_after_load_attempt(self, loader):
        """Test that stats are updated after load attempts."""
        # Load nonexistent files - should not add stats
        loader._load_abbrev_lexicon(Path("/nonexistent.json"))
        assert len(loader._lexicon_stats) == 0


class TestLexiconCategorization:
    """Tests for lexicon categorization in summary."""

    def test_category_mapping(self, loader):
        """Test that lexicons are mapped to correct categories."""
        # The category mapping is defined in _print_lexicon_summary
        # This tests that the method doesn't crash with various categories
        loader._lexicon_stats = [
            ("Abbreviations", 100, "abbrev.json"),
            ("Meta-Inventory", 65000, "meta_inventory.json"),
            ("ChEMBL drugs", 23000, "chembl.json"),
            ("MONDO diseases", 97000, "mondo.json"),
            ("Trial acronyms", 125000, "trials.json"),
        ]
        loader._print_lexicon_summary()
        # Should not raise


class TestLexiconEntryCreation:
    """Tests for LexiconEntry creation during loading."""

    def test_lexicon_entry_attributes(self):
        """Test LexiconEntry has expected attributes."""
        import re
        entry = LexiconEntry(
            sf="TNF",
            lf="Tumor Necrosis Factor",
            pattern=re.compile(r"\bTNF\b"),
            source="test.json",
            lexicon_ids=[{"source": "UMLS", "id": "C0021760"}],
            preserve_case=True,
        )
        assert entry.sf == "TNF"
        assert entry.lf == "Tumor Necrosis Factor"
        assert entry.source == "test.json"
        assert len(entry.lexicon_ids) == 1
        assert entry.preserve_case is True
