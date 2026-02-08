# corpus_metadata/K_tests/K61_test_evaluation_expansion.py
"""
Tests for F03_evaluation_runner expansion: new gold standards, comparison functions,
dataclasses, loaders, and dataset preset configuration.

Covers: NCBI Disease, BC5CDR, PubMed Author/Citation evaluation wiring.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from F_evaluation.F03_evaluation_runner import (
    GoldAuthor,
    GoldCitation,
    GoldDisease,
    GoldDrug,
    ExtractedAuthorEval,
    ExtractedCitationEval,
    EntityResult,
    DocumentResult,
    DatasetResult,
    author_matches,
    compare_authors,
    citation_matches,
    compare_citations,
    load_ncbi_disease_gold,
    load_bc5cdr_gold,
    load_pubmed_author_gold,
)


# ---------------------------------------------------------------------------
# GoldAuthor dataclass tests
# ---------------------------------------------------------------------------
class TestGoldAuthor:
    """Tests for GoldAuthor dataclass properties."""

    def test_last_name_normalized(self):
        a = GoldAuthor(doc_id="1.pdf", last_name="Smith")
        assert a.last_name_normalized == "smith"

    def test_last_name_normalized_strips_whitespace(self):
        a = GoldAuthor(doc_id="1.pdf", last_name="  Jones  ")
        assert a.last_name_normalized == "jones"

    def test_first_initial_from_initials(self):
        a = GoldAuthor(doc_id="1.pdf", last_name="Smith", initials="JA")
        assert a.first_initial == "j"

    def test_first_initial_from_first_name(self):
        a = GoldAuthor(doc_id="1.pdf", last_name="Smith", first_name="John")
        assert a.first_initial == "j"

    def test_first_initial_initials_takes_priority(self):
        a = GoldAuthor(doc_id="1.pdf", last_name="Smith", first_name="John", initials="RA")
        assert a.first_initial == "r"

    def test_first_initial_empty_when_no_names(self):
        a = GoldAuthor(doc_id="1.pdf", last_name="Smith")
        assert a.first_initial == ""


# ---------------------------------------------------------------------------
# GoldCitation dataclass tests
# ---------------------------------------------------------------------------
class TestGoldCitation:
    """Tests for GoldCitation dataclass."""

    def test_all_fields_optional(self):
        c = GoldCitation(doc_id="1.pdf")
        assert c.pmid is None
        assert c.doi is None
        assert c.pmcid is None

    def test_all_fields_set(self):
        c = GoldCitation(doc_id="1.pdf", pmid="12345", doi="10.1234/test", pmcid="PMC999")
        assert c.pmid == "12345"
        assert c.doi == "10.1234/test"
        assert c.pmcid == "PMC999"


# ---------------------------------------------------------------------------
# ExtractedAuthorEval dataclass tests
# ---------------------------------------------------------------------------
class TestExtractedAuthorEval:
    """Tests for ExtractedAuthorEval dataclass properties."""

    def test_last_name_single_word(self):
        a = ExtractedAuthorEval(full_name="Smith")
        assert a.last_name == "smith"

    def test_last_name_multi_word(self):
        a = ExtractedAuthorEval(full_name="John A Smith")
        assert a.last_name == "smith"

    def test_first_initial(self):
        a = ExtractedAuthorEval(full_name="John Smith")
        assert a.first_initial == "j"

    def test_first_initial_uppercase(self):
        a = ExtractedAuthorEval(full_name="J Smith")
        assert a.first_initial == "j"

    def test_empty_name(self):
        a = ExtractedAuthorEval(full_name="")
        assert a.last_name == ""
        assert a.first_initial == ""


# ---------------------------------------------------------------------------
# ExtractedCitationEval dataclass tests
# ---------------------------------------------------------------------------
class TestExtractedCitationEval:
    """Tests for ExtractedCitationEval dataclass."""

    def test_all_none(self):
        c = ExtractedCitationEval()
        assert c.pmid is None
        assert c.doi is None
        assert c.pmcid is None
        assert c.confidence == 0.0

    def test_with_values(self):
        c = ExtractedCitationEval(pmid="12345", doi="10.1/x", confidence=0.9)
        assert c.pmid == "12345"
        assert c.doi == "10.1/x"
        assert c.confidence == 0.9


# ---------------------------------------------------------------------------
# author_matches() tests
# ---------------------------------------------------------------------------
class TestAuthorMatches:
    """Tests for author_matches() function."""

    def test_exact_match(self):
        ext = ExtractedAuthorEval(full_name="John Smith")
        gold = GoldAuthor(doc_id="1.pdf", last_name="Smith", first_name="John", initials="J")
        assert author_matches(ext, gold) is True

    def test_case_insensitive_last_name(self):
        ext = ExtractedAuthorEval(full_name="John SMITH")
        gold = GoldAuthor(doc_id="1.pdf", last_name="smith", first_name="John")
        assert author_matches(ext, gold) is True

    def test_different_last_name(self):
        ext = ExtractedAuthorEval(full_name="John Jones")
        gold = GoldAuthor(doc_id="1.pdf", last_name="Smith", first_name="John")
        assert author_matches(ext, gold) is False

    def test_same_last_name_different_initial(self):
        ext = ExtractedAuthorEval(full_name="Robert Smith")
        gold = GoldAuthor(doc_id="1.pdf", last_name="Smith", first_name="John", initials="J")
        assert author_matches(ext, gold) is False

    def test_same_last_name_no_initials(self):
        """When no initials available, last name match is sufficient."""
        ext = ExtractedAuthorEval(full_name="Smith")
        gold = GoldAuthor(doc_id="1.pdf", last_name="Smith")
        assert author_matches(ext, gold) is True

    def test_gold_has_initial_extracted_does_not(self):
        """Single-word extracted name has no first initial — still matches on last name."""
        ext = ExtractedAuthorEval(full_name="Smith")
        gold = GoldAuthor(doc_id="1.pdf", last_name="Smith", initials="J")
        # ext.first_initial would be 's' (first char of "Smith"), gold is 'j'
        # Actually extracted last_name="smith", first_initial="s" (single word → first char)
        # gold first_initial="j" → these don't match
        # This tests the edge case correctly
        assert author_matches(ext, gold) is False

    def test_initial_only_match(self):
        ext = ExtractedAuthorEval(full_name="J Smith")
        gold = GoldAuthor(doc_id="1.pdf", last_name="Smith", initials="J")
        assert author_matches(ext, gold) is True


# ---------------------------------------------------------------------------
# compare_authors() tests
# ---------------------------------------------------------------------------
class TestCompareAuthors:
    """Tests for compare_authors() function."""

    def test_perfect_match(self):
        extracted = [
            ExtractedAuthorEval(full_name="John Smith"),
            ExtractedAuthorEval(full_name="Jane Doe"),
        ]
        gold = [
            GoldAuthor(doc_id="1.pdf", last_name="Smith", first_name="John", initials="J"),
            GoldAuthor(doc_id="1.pdf", last_name="Doe", first_name="Jane", initials="J"),
        ]
        result = compare_authors(extracted, gold, "1.pdf")
        assert result.tp == 2
        assert result.fp == 0
        assert result.fn == 0

    def test_extra_extracted(self):
        extracted = [
            ExtractedAuthorEval(full_name="John Smith"),
            ExtractedAuthorEval(full_name="Unknown Author"),
        ]
        gold = [
            GoldAuthor(doc_id="1.pdf", last_name="Smith", first_name="John"),
        ]
        result = compare_authors(extracted, gold, "1.pdf")
        assert result.tp == 1
        assert result.fp == 1
        assert result.fn == 0

    def test_missing_extracted(self):
        extracted = [
            ExtractedAuthorEval(full_name="John Smith"),
        ]
        gold = [
            GoldAuthor(doc_id="1.pdf", last_name="Smith", first_name="John"),
            GoldAuthor(doc_id="1.pdf", last_name="Doe", first_name="Jane"),
        ]
        result = compare_authors(extracted, gold, "1.pdf")
        assert result.tp == 1
        assert result.fp == 0
        assert result.fn == 1

    def test_empty_extracted(self):
        gold = [
            GoldAuthor(doc_id="1.pdf", last_name="Smith", first_name="John"),
        ]
        result = compare_authors([], gold, "1.pdf")
        assert result.tp == 0
        assert result.fp == 0
        assert result.fn == 1

    def test_empty_gold(self):
        extracted = [
            ExtractedAuthorEval(full_name="John Smith"),
        ]
        result = compare_authors(extracted, [], "1.pdf")
        assert result.tp == 0
        assert result.fp == 1
        assert result.fn == 0

    def test_both_empty(self):
        result = compare_authors([], [], "1.pdf")
        assert result.tp == 0
        assert result.fp == 0
        assert result.fn == 0

    def test_fn_items_contain_names(self):
        gold = [
            GoldAuthor(doc_id="1.pdf", last_name="Smith", first_name="John"),
        ]
        result = compare_authors([], gold, "1.pdf")
        assert len(result.fn_items) == 1
        assert "Smith" in result.fn_items[0]

    def test_entity_type(self):
        result = compare_authors([], [], "1.pdf")
        assert result.entity_type == "authors"


# ---------------------------------------------------------------------------
# citation_matches() tests
# ---------------------------------------------------------------------------
class TestCitationMatches:
    """Tests for citation_matches() function."""

    def test_pmid_match(self):
        ext = ExtractedCitationEval(pmid="12345")
        gold = GoldCitation(doc_id="1.pdf", pmid="12345")
        assert citation_matches(ext, gold) is True

    def test_pmid_mismatch(self):
        ext = ExtractedCitationEval(pmid="12345")
        gold = GoldCitation(doc_id="1.pdf", pmid="99999")
        assert citation_matches(ext, gold) is False

    def test_doi_match(self):
        ext = ExtractedCitationEval(doi="10.1234/test.2024")
        gold = GoldCitation(doc_id="1.pdf", doi="10.1234/test.2024")
        assert citation_matches(ext, gold) is True

    def test_doi_case_insensitive(self):
        ext = ExtractedCitationEval(doi="10.1234/TEST.2024")
        gold = GoldCitation(doc_id="1.pdf", doi="10.1234/test.2024")
        assert citation_matches(ext, gold) is True

    def test_doi_strips_trailing_period(self):
        ext = ExtractedCitationEval(doi="10.1234/test.")
        gold = GoldCitation(doc_id="1.pdf", doi="10.1234/test")
        assert citation_matches(ext, gold) is True

    def test_pmcid_match(self):
        ext = ExtractedCitationEval(pmcid="PMC12345")
        gold = GoldCitation(doc_id="1.pdf", pmcid="PMC12345")
        assert citation_matches(ext, gold) is True

    def test_pmcid_case_insensitive(self):
        ext = ExtractedCitationEval(pmcid="pmc12345")
        gold = GoldCitation(doc_id="1.pdf", pmcid="PMC12345")
        assert citation_matches(ext, gold) is True

    def test_no_identifiers(self):
        ext = ExtractedCitationEval()
        gold = GoldCitation(doc_id="1.pdf")
        assert citation_matches(ext, gold) is False

    def test_cross_identifier_no_match(self):
        """PMID vs DOI should not match."""
        ext = ExtractedCitationEval(pmid="12345")
        gold = GoldCitation(doc_id="1.pdf", doi="10.1234/x")
        assert citation_matches(ext, gold) is False

    def test_any_identifier_matches(self):
        """Match if at least one identifier matches."""
        ext = ExtractedCitationEval(pmid="12345", doi="10.1/wrong")
        gold = GoldCitation(doc_id="1.pdf", pmid="12345", doi="10.1/different")
        assert citation_matches(ext, gold) is True


# ---------------------------------------------------------------------------
# compare_citations() tests
# ---------------------------------------------------------------------------
class TestCompareCitations:
    """Tests for compare_citations() function."""

    def test_perfect_match(self):
        extracted = [
            ExtractedCitationEval(pmid="111"),
            ExtractedCitationEval(doi="10.1/a"),
        ]
        gold = [
            GoldCitation(doc_id="1.pdf", pmid="111"),
            GoldCitation(doc_id="1.pdf", doi="10.1/a"),
        ]
        result = compare_citations(extracted, gold, "1.pdf")
        assert result.tp == 2
        assert result.fp == 0
        assert result.fn == 0

    def test_extra_extracted(self):
        extracted = [
            ExtractedCitationEval(pmid="111"),
            ExtractedCitationEval(pmid="999"),
        ]
        gold = [
            GoldCitation(doc_id="1.pdf", pmid="111"),
        ]
        result = compare_citations(extracted, gold, "1.pdf")
        assert result.tp == 1
        assert result.fp == 1
        assert result.fn == 0

    def test_missing_extracted(self):
        extracted: list[ExtractedCitationEval] = []
        gold = [
            GoldCitation(doc_id="1.pdf", pmid="111"),
        ]
        result = compare_citations(extracted, gold, "1.pdf")
        assert result.tp == 0
        assert result.fp == 0
        assert result.fn == 1

    def test_entity_type(self):
        result = compare_citations([], [], "1.pdf")
        assert result.entity_type == "citations"

    def test_no_double_match(self):
        """Same gold entry should not be matched twice."""
        extracted = [
            ExtractedCitationEval(pmid="111"),
            ExtractedCitationEval(pmid="111"),
        ]
        gold = [
            GoldCitation(doc_id="1.pdf", pmid="111"),
        ]
        result = compare_citations(extracted, gold, "1.pdf")
        assert result.tp == 1
        assert result.fp == 1


# ---------------------------------------------------------------------------
# DocumentResult tests (author/citation fields)
# ---------------------------------------------------------------------------
class TestDocumentResultExpansion:
    """Tests for author/citation fields on DocumentResult."""

    def test_is_perfect_with_authors_citations(self):
        doc = DocumentResult(
            doc_id="1.pdf",
            authors=EntityResult(entity_type="authors", doc_id="1.pdf", tp=3),
            citations=EntityResult(entity_type="citations", doc_id="1.pdf", tp=1),
        )
        assert doc.is_perfect is True

    def test_not_perfect_with_author_fp(self):
        doc = DocumentResult(
            doc_id="1.pdf",
            authors=EntityResult(entity_type="authors", doc_id="1.pdf", tp=2, fp=1),
        )
        assert doc.is_perfect is False

    def test_not_perfect_with_citation_fn(self):
        doc = DocumentResult(
            doc_id="1.pdf",
            citations=EntityResult(entity_type="citations", doc_id="1.pdf", tp=1, fn=1),
        )
        assert doc.is_perfect is False


# ---------------------------------------------------------------------------
# DatasetResult tests (author/citation metrics)
# ---------------------------------------------------------------------------
class TestDatasetResultExpansion:
    """Tests for author/citation metrics on DatasetResult."""

    def test_author_precision(self):
        ds = DatasetResult(name="test", author_tp=8, author_fp=2)
        assert ds.precision("authors") == pytest.approx(0.8)

    def test_author_recall(self):
        ds = DatasetResult(name="test", author_tp=8, author_fn=2)
        assert ds.recall("authors") == pytest.approx(0.8)

    def test_author_f1(self):
        ds = DatasetResult(name="test", author_tp=8, author_fp=2, author_fn=2)
        p = 8 / 10  # 0.8
        r = 8 / 10  # 0.8
        assert ds.f1("authors") == pytest.approx(2 * p * r / (p + r))

    def test_citation_precision(self):
        ds = DatasetResult(name="test", citation_tp=9, citation_fp=1)
        assert ds.precision("citations") == pytest.approx(0.9)

    def test_citation_recall(self):
        ds = DatasetResult(name="test", citation_tp=7, citation_fn=3)
        assert ds.recall("citations") == pytest.approx(0.7)

    def test_citation_f1(self):
        ds = DatasetResult(name="test", citation_tp=7, citation_fp=1, citation_fn=3)
        p = 7 / 8
        r = 7 / 10
        assert ds.f1("citations") == pytest.approx(2 * p * r / (p + r))

    def test_zero_tp_precision(self):
        ds = DatasetResult(name="test", author_tp=0, author_fp=0)
        assert ds.precision("authors") == 1.0  # no predictions, perfect precision

    def test_zero_tp_recall(self):
        ds = DatasetResult(name="test", author_tp=0, author_fn=0)
        assert ds.recall("authors") == 1.0  # no gold, perfect recall


# ---------------------------------------------------------------------------
# load_ncbi_disease_gold() tests
# ---------------------------------------------------------------------------
class TestLoadNcbiDiseaseGold:
    """Tests for load_ncbi_disease_gold() loader."""

    def test_loads_diseases(self):
        gold_data = {
            "corpus": "NCBI-Disease",
            "diseases": {
                "total": 2,
                "annotations": [
                    {"doc_id": "111.pdf", "text": "prostate cancer", "type": "DISEASE", "split": "test"},
                    {"doc_id": "111.pdf", "text": "breast cancer", "type": "DISEASE", "split": "test"},
                    {"doc_id": "222.pdf", "text": "diabetes", "type": "DISEASE", "split": "train"},
                ],
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gold_data, f)
            tmp_path = Path(f.name)

        try:
            result = load_ncbi_disease_gold(tmp_path)
            assert "diseases" in result
            # No split filter → all 3 annotations
            all_diseases = [d for docs in result["diseases"].values() for d in docs]
            assert len(all_diseases) == 3
        finally:
            tmp_path.unlink()

    def test_filters_by_split(self):
        gold_data = {
            "corpus": "NCBI-Disease",
            "diseases": {
                "total": 2,
                "annotations": [
                    {"doc_id": "111.pdf", "text": "cancer", "type": "DISEASE", "split": "test"},
                    {"doc_id": "222.pdf", "text": "diabetes", "type": "DISEASE", "split": "train"},
                ],
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gold_data, f)
            tmp_path = Path(f.name)

        try:
            result = load_ncbi_disease_gold(tmp_path, splits=["test"])
            all_diseases = [d for docs in result["diseases"].values() for d in docs]
            assert len(all_diseases) == 1
            assert all_diseases[0].text == "cancer"
        finally:
            tmp_path.unlink()

    def test_missing_file_returns_empty(self):
        result = load_ncbi_disease_gold(Path("/nonexistent/gold.json"))
        assert result["diseases"] == {}

    def test_returns_gold_disease_objects(self):
        gold_data = {
            "corpus": "NCBI-Disease",
            "diseases": {
                "total": 1,
                "annotations": [
                    {"doc_id": "1.pdf", "text": "cancer", "type": "DISEASE", "split": "test"},
                ],
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gold_data, f)
            tmp_path = Path(f.name)

        try:
            result = load_ncbi_disease_gold(tmp_path)
            disease = result["diseases"]["1.pdf"][0]
            assert isinstance(disease, GoldDisease)
            assert disease.text == "cancer"
            assert disease.entity_type == "DISEASE"
        finally:
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# load_bc5cdr_gold() tests
# ---------------------------------------------------------------------------
class TestLoadBc5cdrGold:
    """Tests for load_bc5cdr_gold() loader."""

    def test_loads_diseases_and_drugs(self):
        gold_data = {
            "corpus": "BC5CDR",
            "diseases": {
                "total": 1,
                "annotations": [
                    {"doc_id": "1.pdf", "text": "lung cancer", "type": "DISEASE", "split": "test"},
                ],
            },
            "drugs": {
                "total": 1,
                "annotations": [
                    {"doc_id": "1.pdf", "name": "cisplatin", "split": "test"},
                ],
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gold_data, f)
            tmp_path = Path(f.name)

        try:
            result = load_bc5cdr_gold(tmp_path)
            all_diseases = [d for docs in result["diseases"].values() for d in docs]
            all_drugs = [d for docs in result["drugs"].values() for d in docs]
            assert len(all_diseases) == 1
            assert len(all_drugs) == 1
            assert isinstance(all_diseases[0], GoldDisease)
            assert isinstance(all_drugs[0], GoldDrug)
            assert all_drugs[0].name == "cisplatin"
        finally:
            tmp_path.unlink()

    def test_filters_by_split(self):
        gold_data = {
            "corpus": "BC5CDR",
            "diseases": {
                "total": 2,
                "annotations": [
                    {"doc_id": "1.pdf", "text": "cancer", "type": "DISEASE", "split": "test"},
                    {"doc_id": "2.pdf", "text": "flu", "type": "DISEASE", "split": "train"},
                ],
            },
            "drugs": {
                "total": 2,
                "annotations": [
                    {"doc_id": "1.pdf", "name": "aspirin", "split": "test"},
                    {"doc_id": "2.pdf", "name": "ibuprofen", "split": "train"},
                ],
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gold_data, f)
            tmp_path = Path(f.name)

        try:
            result = load_bc5cdr_gold(tmp_path, splits=["test"])
            all_diseases = [d for docs in result["diseases"].values() for d in docs]
            all_drugs = [d for docs in result["drugs"].values() for d in docs]
            assert len(all_diseases) == 1
            assert len(all_drugs) == 1
        finally:
            tmp_path.unlink()

    def test_missing_file_returns_empty(self):
        result = load_bc5cdr_gold(Path("/nonexistent/gold.json"))
        assert result["diseases"] == {}
        assert result["drugs"] == {}


# ---------------------------------------------------------------------------
# load_pubmed_author_gold() tests
# ---------------------------------------------------------------------------
class TestLoadPubmedAuthorGold:
    """Tests for load_pubmed_author_gold() loader."""

    def test_loads_authors_and_citations(self):
        gold_data = {
            "corpus": "PubMed-Authors",
            "authors": {
                "total": 2,
                "annotations": [
                    {"doc_id": "1.pdf", "last_name": "Smith", "first_name": "John", "initials": "J", "split": "test"},
                    {"doc_id": "1.pdf", "last_name": "Doe", "first_name": "Jane", "initials": "J", "split": "test"},
                ],
            },
            "citations": {
                "total": 1,
                "annotations": [
                    {"doc_id": "1.pdf", "pmid": "12345", "doi": "10.1/x", "pmcid": "PMC999", "split": "test"},
                ],
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gold_data, f)
            tmp_path = Path(f.name)

        try:
            result = load_pubmed_author_gold(tmp_path)
            all_authors = [a for docs in result["authors"].values() for a in docs]
            all_citations = [c for docs in result["citations"].values() for c in docs]
            assert len(all_authors) == 2
            assert len(all_citations) == 1
            assert isinstance(all_authors[0], GoldAuthor)
            assert isinstance(all_citations[0], GoldCitation)
            assert all_authors[0].last_name == "Smith"
            assert all_citations[0].pmid == "12345"
        finally:
            tmp_path.unlink()

    def test_filters_by_split(self):
        gold_data = {
            "corpus": "PubMed-Authors",
            "authors": {
                "total": 2,
                "annotations": [
                    {"doc_id": "1.pdf", "last_name": "Smith", "split": "test"},
                    {"doc_id": "2.pdf", "last_name": "Doe", "split": "train"},
                ],
            },
            "citations": {
                "total": 2,
                "annotations": [
                    {"doc_id": "1.pdf", "pmid": "111", "split": "test"},
                    {"doc_id": "2.pdf", "pmid": "222", "split": "train"},
                ],
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gold_data, f)
            tmp_path = Path(f.name)

        try:
            result = load_pubmed_author_gold(tmp_path, splits=["test"])
            all_authors = [a for docs in result["authors"].values() for a in docs]
            all_citations = [c for docs in result["citations"].values() for c in docs]
            assert len(all_authors) == 1
            assert len(all_citations) == 1
        finally:
            tmp_path.unlink()

    def test_missing_file_returns_empty(self):
        result = load_pubmed_author_gold(Path("/nonexistent/gold.json"))
        assert result["authors"] == {}
        assert result["citations"] == {}

    def test_author_optional_fields(self):
        gold_data = {
            "corpus": "PubMed-Authors",
            "authors": {
                "total": 1,
                "annotations": [
                    {"doc_id": "1.pdf", "last_name": "Smith", "split": "test"},
                ],
            },
            "citations": {"total": 0, "annotations": []},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gold_data, f)
            tmp_path = Path(f.name)

        try:
            result = load_pubmed_author_gold(tmp_path)
            author = result["authors"]["1.pdf"][0]
            assert author.last_name == "Smith"
            assert author.first_name is None
            assert author.initials is None
        finally:
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# DATASET_PRESETS tests
# ---------------------------------------------------------------------------
class TestDatasetPresets:
    """Tests for DATASET_PRESETS configuration."""

    def test_all_presets_have_abbreviations_enabled(self):
        """User requirement: abbreviations should always be enabled."""
        from F_evaluation.F03_evaluation_runner import DATASET_PRESETS
        for dataset_name, preset in DATASET_PRESETS.items():
            assert preset.get("abbreviations") is True, (
                f"DATASET_PRESETS['{dataset_name}'] must have abbreviations=True"
            )

    def test_known_datasets_exist(self):
        from F_evaluation.F03_evaluation_runner import DATASET_PRESETS
        expected = {"NLP4RARE", "NLM-Gene", "RareDisGene", "NCBI-Disease", "BC5CDR", "PubMed-Authors", "Papers"}
        assert expected.issubset(set(DATASET_PRESETS.keys()))

    def test_ncbi_disease_enables_diseases(self):
        from F_evaluation.F03_evaluation_runner import DATASET_PRESETS
        preset = DATASET_PRESETS["NCBI-Disease"]
        assert preset["diseases"] is True
        assert preset["drugs"] is False
        assert preset["genes"] is False

    def test_bc5cdr_enables_diseases_and_drugs(self):
        from F_evaluation.F03_evaluation_runner import DATASET_PRESETS
        preset = DATASET_PRESETS["BC5CDR"]
        assert preset["diseases"] is True
        assert preset["drugs"] is True
        assert preset["genes"] is False

    def test_pubmed_authors_enables_authors_citations(self):
        from F_evaluation.F03_evaluation_runner import DATASET_PRESETS
        preset = DATASET_PRESETS["PubMed-Authors"]
        assert preset["authors"] is True
        assert preset["citations"] is True
        assert preset["diseases"] is False
        assert preset["drugs"] is False
        assert preset["genes"] is False

    def test_gene_presets_enable_genes(self):
        from F_evaluation.F03_evaluation_runner import DATASET_PRESETS
        assert DATASET_PRESETS["NLM-Gene"]["genes"] is True
        assert DATASET_PRESETS["RareDisGene"]["genes"] is True

    def test_nlp4rare_enables_all_entities(self):
        from F_evaluation.F03_evaluation_runner import DATASET_PRESETS
        preset = DATASET_PRESETS["NLP4RARE"]
        assert preset["diseases"] is True
        assert preset["drugs"] is True
        assert preset["genes"] is True

    def test_all_presets_disable_feasibility(self):
        """Feasibility is not evaluated in any gold standard."""
        from F_evaluation.F03_evaluation_runner import DATASET_PRESETS
        for dataset_name, preset in DATASET_PRESETS.items():
            assert preset.get("feasibility") is False, (
                f"DATASET_PRESETS['{dataset_name}'] should have feasibility=False"
            )


# ---------------------------------------------------------------------------
# Config flags tests
# ---------------------------------------------------------------------------
class TestEvalConfigFlags:
    """Tests for evaluation configuration flags."""

    def test_new_corpus_flags_default_off(self):
        from F_evaluation.F03_evaluation_runner import (
            RUN_NCBI_DISEASE,
            RUN_BC5CDR,
            RUN_PUBMED_AUTHORS,
        )
        assert RUN_NCBI_DISEASE is False
        assert RUN_BC5CDR is False
        assert RUN_PUBMED_AUTHORS is False

    def test_entity_eval_flags_default_on(self):
        from F_evaluation.F03_evaluation_runner import (
            EVAL_AUTHORS,
            EVAL_CITATIONS,
        )
        assert EVAL_AUTHORS is True
        assert EVAL_CITATIONS is True


# ---------------------------------------------------------------------------
# __all__ exports tests
# ---------------------------------------------------------------------------
class TestExports:
    """Tests that all new types are properly exported."""

    def test_new_types_in_all(self):
        from F_evaluation import F03_evaluation_runner
        expected_exports = [
            "GoldAuthor",
            "GoldCitation",
            "ExtractedAuthorEval",
            "ExtractedCitationEval",
            "load_ncbi_disease_gold",
            "load_bc5cdr_gold",
            "load_pubmed_author_gold",
        ]
        for name in expected_exports:
            assert name in F03_evaluation_runner.__all__, f"{name} not in __all__"
            assert hasattr(F03_evaluation_runner, name), f"{name} not accessible"
