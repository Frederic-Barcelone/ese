# corpus_metadata/A_core/A13_ner_models.py
"""
Unified data models for Named Entity Recognition (NER) results.

This module provides standardized data structures that unify NER outputs from
multiple extraction backends (ZeroShotBioNER, BiomedicalNER, scispaCy, etc.).
Use these models when building or consuming NER pipelines to ensure consistent
entity representation, filtering, grouping, and serialization.

Key Components:
    - EntityCategory: Constants for entity categorization (clinical, genetic, etc.)
    - NEREntity: Single extracted entity with type, score, position, and metadata
    - NERResult: Container for multiple entities with filtering and grouping methods
    - create_entity: Factory function for convenient entity creation
    - merge_results: Combine multiple NERResult instances
    - DEFAULT_TYPE_TO_CATEGORY: Mapping from entity types to categories

Example:
    >>> from A_core.A13_ner_models import NEREntity, NERResult
    >>> entity = NEREntity(text="diabetes", entity_type="Disease", score=0.95)
    >>> result = NERResult(source="biomedical_ner")
    >>> result.add_entity(entity)
    >>> diseases = result.get_by_type("Disease")
    >>> summary = result.to_summary()

Dependencies:
    - None (standalone module using only dataclasses and typing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# ENTITY CATEGORIES
# =============================================================================

class EntityCategory:
    """
    Standard entity category constants.

    Categories group related entity types for easier filtering.
    """

    # Clinical/Medical
    CLINICAL = "clinical"
    DISEASE = "disease"
    SYMPTOM = "symptom"
    PROCEDURE = "procedure"
    MEDICATION = "medication"

    # Drug administration
    DRUG_ADMIN = "drug_admin"

    # Genetic/Molecular
    GENETIC = "genetic"
    GENE = "gene"
    VARIANT = "variant"
    PHENOTYPE = "phenotype"

    # Demographics
    DEMOGRAPHICS = "demographics"

    # Anatomical
    ANATOMICAL = "anatomical"

    # Temporal
    TEMPORAL = "temporal"

    # Adverse events
    ADVERSE_EVENT = "adverse_event"

    # Other
    OTHER = "other"
    UNKNOWN = "unknown"


# Default category mappings for common entity types
DEFAULT_TYPE_TO_CATEGORY: Dict[str, str] = {
    # Clinical
    "Disease": EntityCategory.CLINICAL,
    "Disease_disorder": EntityCategory.CLINICAL,
    "Sign_symptom": EntityCategory.SYMPTOM,
    "Symptom": EntityCategory.SYMPTOM,
    "Diagnostic_procedure": EntityCategory.PROCEDURE,
    "Therapeutic_procedure": EntityCategory.PROCEDURE,
    "Procedure": EntityCategory.PROCEDURE,
    "Medication": EntityCategory.MEDICATION,
    "Drug": EntityCategory.MEDICATION,
    "Chemical": EntityCategory.MEDICATION,
    "Lab_value": EntityCategory.CLINICAL,
    "Clinical_event": EntityCategory.CLINICAL,
    "Outcome": EntityCategory.CLINICAL,

    # Drug administration
    "Dosage": EntityCategory.DRUG_ADMIN,
    "Frequency": EntityCategory.DRUG_ADMIN,
    "Strength": EntityCategory.DRUG_ADMIN,
    "Form": EntityCategory.DRUG_ADMIN,
    "Route": EntityCategory.DRUG_ADMIN,
    "Duration": EntityCategory.DRUG_ADMIN,
    "Administration": EntityCategory.DRUG_ADMIN,

    # Adverse events
    "ADE": EntityCategory.ADVERSE_EVENT,
    "Adverse_event": EntityCategory.ADVERSE_EVENT,

    # Genetic
    "gene_symbol": EntityCategory.GENE,
    "Gene": EntityCategory.GENE,
    "variant_hgvs": EntityCategory.VARIANT,
    "variant_rsid": EntityCategory.VARIANT,
    "hpo_term": EntityCategory.PHENOTYPE,
    "disease_ordo": EntityCategory.DISEASE,

    # Demographics
    "Age": EntityCategory.DEMOGRAPHICS,
    "Sex": EntityCategory.DEMOGRAPHICS,
    "Height": EntityCategory.DEMOGRAPHICS,
    "Weight": EntityCategory.DEMOGRAPHICS,
    "Family_history": EntityCategory.DEMOGRAPHICS,
    "Occupation": EntityCategory.DEMOGRAPHICS,
    "Personal_background": EntityCategory.DEMOGRAPHICS,

    # Anatomical
    "Biological_structure": EntityCategory.ANATOMICAL,
    "Biological_attribute": EntityCategory.ANATOMICAL,
    "Body_part": EntityCategory.ANATOMICAL,

    # Temporal
    "Date": EntityCategory.TEMPORAL,
    "Time": EntityCategory.TEMPORAL,
    "History": EntityCategory.TEMPORAL,
    "Reason": EntityCategory.OTHER,
}


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class NEREntity:
    """
    Single named entity extracted by any NER system.

    Provides a unified representation for entities from different backends
    (ZeroShotBioNER, BiomedicalNER, GeneticEnricher, etc.).

    Attributes:
        text: The extracted text span.
        entity_type: Specific entity type (e.g., "Disease", "Gene", "ADE").
        category: Broader category (e.g., "clinical", "genetic").
        score: Confidence score from the model (0.0-1.0).
        start: Character start position in source text.
        end: Character end position in source text.
        normalized: Normalized/canonical form of the entity.
        metadata: Additional entity-specific metadata.
    """

    text: str
    entity_type: str
    category: str = EntityCategory.UNKNOWN
    score: float = 1.0
    start: int = 0
    end: int = 0
    normalized: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Auto-assign category if not provided."""
        if self.category == EntityCategory.UNKNOWN:
            self.category = DEFAULT_TYPE_TO_CATEGORY.get(
                self.entity_type, EntityCategory.UNKNOWN
            )

    def __repr__(self) -> str:
        """Return concise string representation."""
        return f"NEREntity({self.entity_type}: '{self.text}' [{self.score:.2f}])"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "text": self.text,
            "type": self.entity_type,
            "category": self.category,
            "score": self.score,
            "start": self.start,
            "end": self.end,
        }
        if self.normalized:
            result["normalized"] = self.normalized
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NEREntity":
        """Create entity from dictionary."""
        return cls(
            text=data["text"],
            entity_type=data.get("type", data.get("entity_type", "unknown")),
            category=data.get("category", EntityCategory.UNKNOWN),
            score=data.get("score", 1.0),
            start=data.get("start", 0),
            end=data.get("end", 0),
            normalized=data.get("normalized"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class NERResult:
    """
    Container for NER extraction results.

    Provides methods for filtering, grouping, and exporting entities
    extracted by any NER backend.

    Attributes:
        source: Name of the NER system that produced results.
        entities: List of all extracted entities.
        extraction_time_seconds: Processing time.
        metadata: Additional result-level metadata.
    """

    source: str = "unknown"
    entities: List[NEREntity] = field(default_factory=list)
    extraction_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_entity(self, entity: NEREntity) -> None:
        """Add an entity to the result."""
        self.entities.append(entity)

    def add_entities(self, entities: List[NEREntity]) -> None:
        """Add multiple entities to the result."""
        self.entities.extend(entities)

    @property
    def total_entities(self) -> int:
        """Total number of extracted entities."""
        return len(self.entities)

    def get_by_type(self, entity_type: str) -> List[NEREntity]:
        """
        Get all entities of a specific type.

        Args:
            entity_type: Entity type to filter by.

        Returns:
            List of matching entities.
        """
        return [e for e in self.entities if e.entity_type == entity_type]

    def get_by_category(self, category: str) -> List[NEREntity]:
        """
        Get all entities in a category.

        Args:
            category: Category to filter by.

        Returns:
            List of matching entities.
        """
        return [e for e in self.entities if e.category == category]

    def get_by_types(self, entity_types: List[str]) -> List[NEREntity]:
        """
        Get all entities matching any of the specified types.

        Args:
            entity_types: List of entity types to include.

        Returns:
            List of matching entities.
        """
        type_set = set(entity_types)
        return [e for e in self.entities if e.entity_type in type_set]

    def get_above_threshold(self, threshold: float = 0.5) -> List[NEREntity]:
        """
        Get entities above a confidence threshold.

        Args:
            threshold: Minimum confidence score.

        Returns:
            List of entities with score >= threshold.
        """
        return [e for e in self.entities if e.score >= threshold]

    def get_unique_texts(self, entity_type: Optional[str] = None) -> Set[str]:
        """
        Get unique entity texts.

        Args:
            entity_type: Optional type to filter by.

        Returns:
            Set of unique text values.
        """
        entities = self.get_by_type(entity_type) if entity_type else self.entities
        return {e.text for e in entities}

    def group_by_type(self) -> Dict[str, List[NEREntity]]:
        """
        Group entities by their type.

        Returns:
            Dictionary mapping entity types to entity lists.
        """
        groups: Dict[str, List[NEREntity]] = {}
        for entity in self.entities:
            if entity.entity_type not in groups:
                groups[entity.entity_type] = []
            groups[entity.entity_type].append(entity)
        return groups

    def group_by_category(self) -> Dict[str, List[NEREntity]]:
        """
        Group entities by their category.

        Returns:
            Dictionary mapping categories to entity lists.
        """
        groups: Dict[str, List[NEREntity]] = {}
        for entity in self.entities:
            if entity.category not in groups:
                groups[entity.category] = []
            groups[entity.category].append(entity)
        return groups

    def count_by_type(self) -> Dict[str, int]:
        """
        Count entities by type.

        Returns:
            Dictionary mapping entity types to counts.
        """
        counts: Dict[str, int] = {}
        for entity in self.entities:
            counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1
        return counts

    def count_by_category(self) -> Dict[str, int]:
        """
        Count entities by category.

        Returns:
            Dictionary mapping categories to counts.
        """
        counts: Dict[str, int] = {}
        for entity in self.entities:
            counts[entity.category] = counts.get(entity.category, 0) + 1
        return counts

    def to_summary(self) -> Dict[str, Any]:
        """
        Convert to summary dictionary for logging/export.

        Returns:
            Dictionary with source, counts, and timing.
        """
        return {
            "source": self.source,
            "total_entities": self.total_entities,
            "type_counts": self.count_by_type(),
            "category_counts": self.count_by_category(),
            "extraction_time_seconds": self.extraction_time_seconds,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to full dictionary for serialization.

        Returns:
            Complete result as dictionary.
        """
        return {
            "source": self.source,
            "entities": [e.to_dict() for e in self.entities],
            "extraction_time_seconds": self.extraction_time_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NERResult":
        """Create result from dictionary."""
        result = cls(
            source=data.get("source", "unknown"),
            extraction_time_seconds=data.get("extraction_time_seconds", 0.0),
            metadata=data.get("metadata", {}),
        )
        for entity_data in data.get("entities", []):
            result.add_entity(NEREntity.from_dict(entity_data))
        return result

    def merge(self, other: "NERResult") -> "NERResult":
        """
        Merge another result into this one.

        Args:
            other: NERResult to merge.

        Returns:
            New NERResult with combined entities.
        """
        merged = NERResult(
            source=f"{self.source}+{other.source}",
            entities=self.entities + other.entities,
            extraction_time_seconds=self.extraction_time_seconds + other.extraction_time_seconds,
        )
        merged.metadata = {**self.metadata, **other.metadata}
        return merged

    def __repr__(self) -> str:
        """Return string representation."""
        return f"NERResult(source={self.source}, entities={self.total_entities})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_entity(
    text: str,
    entity_type: str,
    score: float = 1.0,
    start: int = 0,
    end: int = 0,
    normalized: Optional[str] = None,
    **metadata: Any,
) -> NEREntity:
    """
    Convenience function to create an NEREntity.

    Args:
        text: Extracted text span.
        entity_type: Entity type.
        score: Confidence score.
        start: Start position.
        end: End position.
        normalized: Normalized form.
        **metadata: Additional metadata as keyword arguments.

    Returns:
        NEREntity instance.
    """
    return NEREntity(
        text=text,
        entity_type=entity_type,
        score=score,
        start=start,
        end=end,
        normalized=normalized,
        metadata=dict(metadata) if metadata else {},
    )


def merge_results(*results: NERResult) -> NERResult:
    """
    Merge multiple NERResults into one.

    Args:
        *results: NERResult instances to merge.

    Returns:
        Combined NERResult.
    """
    if not results:
        return NERResult()

    merged = results[0]
    for result in results[1:]:
        merged = merged.merge(result)
    return merged


# Export public API
__all__ = [
    "EntityCategory",
    "NEREntity",
    "NERResult",
    "create_entity",
    "merge_results",
    "DEFAULT_TYPE_TO_CATEGORY",
]
