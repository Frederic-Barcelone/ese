# corpus_metadata/K_tests/K40_test_gold_loader.py
"""
Tests for F_evaluation.F01_gold_loader module.

Tests gold standard loading and annotation parsing.
"""

from __future__ import annotations

import json
import tempfile

import pytest

from F_evaluation.F01_gold_loader import (
    GoldAnnotation,
    GoldStandard,
    GoldLoader,
)


class TestGoldAnnotation:
    """Tests for GoldAnnotation model."""

    def test_basic_annotation(self):
        anno = GoldAnnotation(
            doc_id="test.pdf",
            short_form="TNF",
            long_form="Tumor Necrosis Factor",
        )
        assert anno.doc_id == "test.pdf"
        assert anno.short_form == "TNF"
        assert anno.long_form == "Tumor Necrosis Factor"

    def test_annotation_with_category(self):
        anno = GoldAnnotation(
            doc_id="test.pdf",
            short_form="BRCA1",
            long_form="Breast Cancer Gene 1",
            category="Gene",
        )
        assert anno.category == "Gene"

    def test_annotation_without_long_form(self):
        anno = GoldAnnotation(
            doc_id="test.pdf",
            short_form="TNF",
            long_form=None,
        )
        assert anno.long_form is None


class TestGoldStandard:
    """Tests for GoldStandard model."""

    def test_empty_standard(self):
        gold = GoldStandard()
        assert len(gold.annotations) == 0

    def test_with_annotations(self):
        gold = GoldStandard(
            annotations=[
                GoldAnnotation(doc_id="a.pdf", short_form="A", long_form="Alpha"),
                GoldAnnotation(doc_id="b.pdf", short_form="B", long_form="Beta"),
            ]
        )
        assert len(gold.annotations) == 2


class TestGoldLoaderJson:
    """Tests for JSON loading."""

    @pytest.fixture
    def loader(self):
        return GoldLoader(strict=False)

    def test_load_list_format(self, loader):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [
                {"doc_id": "test.pdf", "short_form": "TNF", "long_form": "Tumor Necrosis Factor"},
                {"doc_id": "test.pdf", "short_form": "IL6", "long_form": "Interleukin 6"},
            ]
            json.dump(data, f)
            path = f.name

        gold, index = loader.load_json(path)

        assert len(gold.annotations) == 2
        assert "test.pdf" in index

    def test_load_dict_with_annotations_key(self, loader):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = {
                "annotations": [
                    {"doc_id": "test.pdf", "short_form": "TNF", "long_form": "TNF Alpha"},
                ]
            }
            json.dump(data, f)
            path = f.name

        gold, index = loader.load_json(path)

        assert len(gold.annotations) == 1

    def test_load_v2_format(self, loader):
        """Test loading defined_annotations key (v2 format)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = {
                "defined_annotations": [
                    {"doc_id": "test.pdf", "short_form": "TNF", "long_form": "TNF Alpha"},
                ]
            }
            json.dump(data, f)
            path = f.name

        gold, index = loader.load_json(path)

        assert len(gold.annotations) == 1

    def test_alternative_field_names(self, loader):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [
                {"filename": "doc.pdf", "sf": "TNF", "lf": "TNF Alpha"},
                {"doc": "doc.pdf", "abbr": "IL6", "expansion": "Interleukin 6"},
            ]
            json.dump(data, f)
            path = f.name

        gold, index = loader.load_json(path)

        assert len(gold.annotations) == 2


class TestGoldLoaderCsv:
    """Tests for CSV loading."""

    @pytest.fixture
    def loader(self):
        return GoldLoader(strict=False)

    def test_load_csv(self, loader):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("doc_id,short_form,long_form\n")
            f.write("test.pdf,TNF,Tumor Necrosis Factor\n")
            f.write("test.pdf,IL6,Interleukin 6\n")
            path = f.name

        gold, index = loader.load_csv(path)

        assert len(gold.annotations) == 2
        assert "test.pdf" in index


class TestGoldLoaderNormalization:
    """Tests for normalization during loading."""

    @pytest.fixture
    def loader(self):
        return GoldLoader(strict=False)

    def test_doc_id_path_stripped(self, loader):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [
                {"doc_id": "/path/to/test.pdf", "short_form": "TNF", "long_form": "TNF Alpha"},
            ]
            json.dump(data, f)
            path = f.name

        gold, index = loader.load_json(path)

        # Should only keep filename
        assert gold.annotations[0].doc_id == "test.pdf"
        assert "test.pdf" in index

    def test_short_form_uppercased(self, loader):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [
                {"doc_id": "test.pdf", "short_form": "tnf", "long_form": "TNF Alpha"},
            ]
            json.dump(data, f)
            path = f.name

        gold, _ = loader.load_json(path)

        assert gold.annotations[0].short_form == "TNF"

    def test_long_form_whitespace_normalized(self, loader):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [
                {"doc_id": "test.pdf", "short_form": "TNF", "long_form": "Tumor   Necrosis   Factor"},
            ]
            json.dump(data, f)
            path = f.name

        gold, _ = loader.load_json(path)

        assert gold.annotations[0].long_form == "Tumor Necrosis Factor"

    def test_duplicate_removal(self, loader):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [
                {"doc_id": "test.pdf", "short_form": "TNF", "long_form": "TNF Alpha"},
                {"doc_id": "test.pdf", "short_form": "TNF", "long_form": "TNF Alpha"},  # Duplicate
            ]
            json.dump(data, f)
            path = f.name

        gold, _ = loader.load_json(path)

        assert len(gold.annotations) == 1


class TestGoldLoaderStrict:
    """Tests for strict mode."""

    def test_strict_requires_long_form(self):
        loader = GoldLoader(strict=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [
                {"doc_id": "test.pdf", "short_form": "TNF"},  # Missing long_form
            ]
            json.dump(data, f)
            path = f.name

        with pytest.raises(ValueError, match="long_form"):
            loader.load_json(path)

    def test_strict_requires_doc_id(self):
        loader = GoldLoader(strict=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [
                {"short_form": "TNF", "long_form": "TNF Alpha"},  # Missing doc_id
            ]
            json.dump(data, f)
            path = f.name

        with pytest.raises(ValueError, match="doc_id"):
            loader.load_json(path)

    def test_non_strict_allows_missing_lf(self):
        loader = GoldLoader(strict=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [
                {"doc_id": "test.pdf", "short_form": "TNF"},
            ]
            json.dump(data, f)
            path = f.name

        gold, _ = loader.load_json(path)

        assert len(gold.annotations) == 1
        assert gold.annotations[0].long_form is None


class TestGoldLoaderIndexing:
    """Tests for document indexing."""

    @pytest.fixture
    def loader(self):
        return GoldLoader(strict=False)

    def test_index_by_doc(self, loader):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [
                {"doc_id": "doc1.pdf", "short_form": "A", "long_form": "Alpha"},
                {"doc_id": "doc1.pdf", "short_form": "B", "long_form": "Beta"},
                {"doc_id": "doc2.pdf", "short_form": "C", "long_form": "Gamma"},
            ]
            json.dump(data, f)
            path = f.name

        _, index = loader.load_json(path)

        assert len(index["doc1.pdf"]) == 2
        assert len(index["doc2.pdf"]) == 1
