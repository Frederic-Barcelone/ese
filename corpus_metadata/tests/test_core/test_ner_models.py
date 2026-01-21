# corpus_metadata/tests/test_core/test_ner_models.py
"""Tests for A_core/A13_ner_models.py - Unified NER data models."""

import pytest

from A_core.A13_ner_models import (
    EntityCategory,
    NEREntity,
    NERResult,
    create_entity,
    merge_results,
    DEFAULT_TYPE_TO_CATEGORY,
)


class TestEntityCategory:
    """Tests for EntityCategory constants."""

    def test_category_constants(self):
        """Test that category constants are defined."""
        assert EntityCategory.CLINICAL == "clinical"
        assert EntityCategory.DISEASE == "disease"
        assert EntityCategory.GENETIC == "genetic"
        assert EntityCategory.UNKNOWN == "unknown"


class TestNEREntity:
    """Tests for NEREntity dataclass."""

    def test_basic_creation(self):
        """Test basic entity creation."""
        entity = NEREntity(
            text="diabetes",
            entity_type="Disease",
            score=0.95,
            start=10,
            end=18,
        )
        assert entity.text == "diabetes"
        assert entity.entity_type == "Disease"
        assert entity.score == 0.95
        assert entity.start == 10
        assert entity.end == 18

    def test_auto_category_assignment(self):
        """Test that category is auto-assigned from entity type."""
        entity = NEREntity(text="diabetes", entity_type="Disease_disorder")
        assert entity.category == EntityCategory.CLINICAL

        entity2 = NEREntity(text="BRCA1", entity_type="Gene")
        assert entity2.category == EntityCategory.GENE

    def test_explicit_category(self):
        """Test explicit category assignment."""
        entity = NEREntity(
            text="test",
            entity_type="CustomType",
            category="custom_category",
        )
        assert entity.category == "custom_category"

    def test_unknown_type_gets_unknown_category(self):
        """Test that unknown types get unknown category."""
        entity = NEREntity(text="test", entity_type="UnknownType")
        assert entity.category == EntityCategory.UNKNOWN

    def test_normalized_field(self):
        """Test normalized form storage."""
        entity = NEREntity(
            text="Diabetes Mellitus Type 2",
            entity_type="Disease",
            normalized="type 2 diabetes mellitus",
        )
        assert entity.normalized == "type 2 diabetes mellitus"

    def test_metadata_field(self):
        """Test metadata storage."""
        entity = NEREntity(
            text="BRCA1",
            entity_type="Gene",
            metadata={"hgnc_id": "1100", "symbol": "BRCA1"},
        )
        assert entity.metadata["hgnc_id"] == "1100"

    def test_to_dict(self):
        """Test dictionary conversion."""
        entity = NEREntity(
            text="diabetes",
            entity_type="Disease",
            category="clinical",
            score=0.9,
            start=0,
            end=8,
            normalized="diabetes mellitus",
            metadata={"source": "test"},
        )
        d = entity.to_dict()
        assert d["text"] == "diabetes"
        assert d["type"] == "Disease"
        assert d["normalized"] == "diabetes mellitus"
        assert d["metadata"]["source"] == "test"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "text": "diabetes",
            "type": "Disease",
            "category": "clinical",
            "score": 0.9,
            "start": 0,
            "end": 8,
        }
        entity = NEREntity.from_dict(data)
        assert entity.text == "diabetes"
        assert entity.entity_type == "Disease"
        assert entity.score == 0.9

    def test_repr(self):
        """Test string representation."""
        entity = NEREntity(text="test", entity_type="Disease", score=0.85)
        repr_str = repr(entity)
        assert "Disease" in repr_str
        assert "test" in repr_str
        assert "0.85" in repr_str


class TestNERResult:
    """Tests for NERResult container."""

    def test_empty_result(self):
        """Test empty result creation."""
        result = NERResult()
        assert result.total_entities == 0
        assert result.source == "unknown"

    def test_add_entity(self):
        """Test adding single entity."""
        result = NERResult(source="test")
        entity = NEREntity(text="diabetes", entity_type="Disease")
        result.add_entity(entity)
        assert result.total_entities == 1
        assert result.entities[0].text == "diabetes"

    def test_add_entities(self):
        """Test adding multiple entities."""
        result = NERResult()
        entities = [
            NEREntity(text="diabetes", entity_type="Disease"),
            NEREntity(text="BRCA1", entity_type="Gene"),
        ]
        result.add_entities(entities)
        assert result.total_entities == 2

    def test_get_by_type(self):
        """Test filtering by entity type."""
        result = NERResult()
        result.add_entities([
            NEREntity(text="diabetes", entity_type="Disease"),
            NEREntity(text="cancer", entity_type="Disease"),
            NEREntity(text="BRCA1", entity_type="Gene"),
        ])
        diseases = result.get_by_type("Disease")
        assert len(diseases) == 2
        genes = result.get_by_type("Gene")
        assert len(genes) == 1

    def test_get_by_category(self):
        """Test filtering by category."""
        result = NERResult()
        result.add_entities([
            NEREntity(text="diabetes", entity_type="Disease_disorder"),
            NEREntity(text="100mg", entity_type="Dosage"),
            NEREntity(text="BRCA1", entity_type="Gene"),
        ])
        clinical = result.get_by_category(EntityCategory.CLINICAL)
        drug_admin = result.get_by_category(EntityCategory.DRUG_ADMIN)
        genetic = result.get_by_category(EntityCategory.GENE)

        assert len(clinical) == 1
        assert len(drug_admin) == 1
        assert len(genetic) == 1

    def test_get_by_types(self):
        """Test filtering by multiple types."""
        result = NERResult()
        result.add_entities([
            NEREntity(text="diabetes", entity_type="Disease"),
            NEREntity(text="headache", entity_type="Symptom"),
            NEREntity(text="BRCA1", entity_type="Gene"),
        ])
        filtered = result.get_by_types(["Disease", "Symptom"])
        assert len(filtered) == 2

    def test_get_above_threshold(self):
        """Test filtering by confidence threshold."""
        result = NERResult()
        result.add_entities([
            NEREntity(text="high", entity_type="Disease", score=0.9),
            NEREntity(text="medium", entity_type="Disease", score=0.6),
            NEREntity(text="low", entity_type="Disease", score=0.3),
        ])
        high_conf = result.get_above_threshold(0.7)
        assert len(high_conf) == 1
        assert high_conf[0].text == "high"

    def test_get_unique_texts(self):
        """Test getting unique text values."""
        result = NERResult()
        result.add_entities([
            NEREntity(text="diabetes", entity_type="Disease"),
            NEREntity(text="diabetes", entity_type="Disease"),  # Duplicate
            NEREntity(text="cancer", entity_type="Disease"),
        ])
        unique = result.get_unique_texts()
        assert len(unique) == 2
        assert "diabetes" in unique
        assert "cancer" in unique

    def test_group_by_type(self):
        """Test grouping by entity type."""
        result = NERResult()
        result.add_entities([
            NEREntity(text="diabetes", entity_type="Disease"),
            NEREntity(text="cancer", entity_type="Disease"),
            NEREntity(text="BRCA1", entity_type="Gene"),
        ])
        groups = result.group_by_type()
        assert len(groups["Disease"]) == 2
        assert len(groups["Gene"]) == 1

    def test_count_by_type(self):
        """Test counting by type."""
        result = NERResult()
        result.add_entities([
            NEREntity(text="diabetes", entity_type="Disease"),
            NEREntity(text="cancer", entity_type="Disease"),
            NEREntity(text="BRCA1", entity_type="Gene"),
        ])
        counts = result.count_by_type()
        assert counts["Disease"] == 2
        assert counts["Gene"] == 1

    def test_to_summary(self):
        """Test summary generation."""
        result = NERResult(source="biomedical_ner")
        result.add_entity(NEREntity(text="diabetes", entity_type="Disease"))
        result.extraction_time_seconds = 1.5

        summary = result.to_summary()
        assert summary["source"] == "biomedical_ner"
        assert summary["total_entities"] == 1
        assert summary["extraction_time_seconds"] == 1.5
        assert "Disease" in summary["type_counts"]

    def test_to_dict_and_from_dict(self):
        """Test full serialization round-trip."""
        result = NERResult(source="test")
        result.add_entity(NEREntity(text="diabetes", entity_type="Disease", score=0.9))
        result.extraction_time_seconds = 1.0
        result.metadata = {"version": "1.0"}

        # Convert to dict and back
        data = result.to_dict()
        restored = NERResult.from_dict(data)

        assert restored.source == "test"
        assert restored.total_entities == 1
        assert restored.entities[0].text == "diabetes"
        assert restored.extraction_time_seconds == 1.0

    def test_merge(self):
        """Test merging two results."""
        result1 = NERResult(source="ner1")
        result1.add_entity(NEREntity(text="diabetes", entity_type="Disease"))
        result1.extraction_time_seconds = 1.0

        result2 = NERResult(source="ner2")
        result2.add_entity(NEREntity(text="BRCA1", entity_type="Gene"))
        result2.extraction_time_seconds = 0.5

        merged = result1.merge(result2)
        assert merged.total_entities == 2
        assert "ner1" in merged.source
        assert "ner2" in merged.source
        assert merged.extraction_time_seconds == 1.5


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_create_entity(self):
        """Test create_entity helper."""
        entity = create_entity(
            text="diabetes",
            entity_type="Disease",
            score=0.9,
            start=0,
            end=8,
            source="test",  # Goes to metadata
        )
        assert entity.text == "diabetes"
        assert entity.score == 0.9
        assert entity.metadata["source"] == "test"

    def test_merge_results(self):
        """Test merge_results helper."""
        r1 = NERResult(source="a")
        r1.add_entity(NEREntity(text="e1", entity_type="T1"))

        r2 = NERResult(source="b")
        r2.add_entity(NEREntity(text="e2", entity_type="T2"))

        r3 = NERResult(source="c")
        r3.add_entity(NEREntity(text="e3", entity_type="T3"))

        merged = merge_results(r1, r2, r3)
        assert merged.total_entities == 3

    def test_merge_empty_results(self):
        """Test merging with empty input."""
        result = merge_results()
        assert result.total_entities == 0


class TestDefaultTypeToCategoryMapping:
    """Tests for the default type-to-category mapping."""

    def test_clinical_types(self):
        """Test clinical entity type mappings."""
        assert DEFAULT_TYPE_TO_CATEGORY["Disease"] == EntityCategory.CLINICAL
        assert DEFAULT_TYPE_TO_CATEGORY["Disease_disorder"] == EntityCategory.CLINICAL
        assert DEFAULT_TYPE_TO_CATEGORY["Medication"] == EntityCategory.MEDICATION

    def test_drug_admin_types(self):
        """Test drug administration type mappings."""
        assert DEFAULT_TYPE_TO_CATEGORY["Dosage"] == EntityCategory.DRUG_ADMIN
        assert DEFAULT_TYPE_TO_CATEGORY["Frequency"] == EntityCategory.DRUG_ADMIN
        assert DEFAULT_TYPE_TO_CATEGORY["Route"] == EntityCategory.DRUG_ADMIN

    def test_genetic_types(self):
        """Test genetic type mappings."""
        assert DEFAULT_TYPE_TO_CATEGORY["gene_symbol"] == EntityCategory.GENE
        assert DEFAULT_TYPE_TO_CATEGORY["variant_hgvs"] == EntityCategory.VARIANT
        assert DEFAULT_TYPE_TO_CATEGORY["hpo_term"] == EntityCategory.PHENOTYPE
