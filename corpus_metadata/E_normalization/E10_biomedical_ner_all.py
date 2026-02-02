"""
Comprehensive biomedical NER using d4data/biomedical-ner-all model.

This module provides broad biomedical entity extraction using a DistilBERT
model trained on the Maccrobat dataset, covering 84 entity types across
clinical, demographic, temporal, and anatomical categories.

Key Components:
    - BiomedicalNEREnricher: Main enricher using d4data/biomedical-ner-all
    - ENTITY_CATEGORIES: Groupings for extracted entity types
    - Entity categories:
        - Clinical: Disease, Symptom, Procedure, Medication, Lab values
        - Demographics: Age, Sex, Family history
        - Temporal: Date, Duration, History
        - Anatomical: Biological structure, Body part
        - Drug administration: Dosage, Frequency, Administration

Example:
    >>> from E_normalization.E10_biomedical_ner_all import BiomedicalNEREnricher
    >>> enricher = BiomedicalNEREnricher()
    >>> result = enricher.extract("45-year-old male with diabetes on metformin")
    >>> for entity in result.entities:
    ...     print(f"{entity.category}/{entity.entity_type}: {entity.text}")
    demographics/Age: 45-year-old
    clinical/Disease_disorder: diabetes
    clinical/Medication: metformin

Model: https://huggingface.co/d4data/biomedical-ner-all

Dependencies:
    - transformers: HuggingFace Pipeline (lazy loading)
    - torch: PyTorch backend (lazy loading)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy load transformers
_pipeline = None
_model_loaded = False


# Entity categories for grouping results
ENTITY_CATEGORIES = {
    # Clinical entities
    "clinical": [
        "Disease_disorder",
        "Sign_symptom",
        "Diagnostic_procedure",
        "Therapeutic_procedure",
        "Medication",
        "Lab_value",
        "Clinical_event",
        "Outcome",
    ],
    # Drug administration (may overlap with ZeroShotBioNER)
    "drug_admin": [
        "Dosage",
        "Administration",
        "Frequency",
    ],
    # Demographics
    "demographics": [
        "Age",
        "Sex",
        "Height",
        "Weight",
        "Mass",
        "Occupation",
        "Personal_background",
        "Family_history",
    ],
    # Anatomical
    "anatomical": [
        "Biological_structure",
        "Biological_attribute",
        "Body_part",
    ],
    # Descriptive
    "descriptive": [
        "Color",
        "Shape",
        "Texture",
        "Severity",
        "Qualitative_concept",
        "Quantitative_concept",
        "Detailed_description",
    ],
    # Temporal
    "temporal": [
        "Date",
        "Time",
        "Duration",
        "History",
    ],
    # Spatial/Other
    "other": [
        "Area",
        "Distance",
        "Volume",
        "Nonbiological_location",
        "Activity",
        "Subject",
        "Coreference",
        "Other_entity",
        "Other_event",
    ],
}

# Flatten for quick lookup
ALL_ENTITY_TYPES = []
ENTITY_TO_CATEGORY = {}
for category, entities in ENTITY_CATEGORIES.items():
    ALL_ENTITY_TYPES.extend(entities)
    for entity in entities:
        ENTITY_TO_CATEGORY[entity] = category


# Stopwords/garbage tokens to filter out
# These are common BERT artifacts or non-informative fragments
GARBAGE_TOKENS = {
    # Common fragments
    "rate", "reduced", "crea", "min", "per", "the", "and", "for", "with",
    "from", "that", "this", "were", "was", "are", "has", "had", "have",
    "been", "being", "will", "would", "could", "should", "may", "might",
    # Partial units/measurements
    "mg", "ml", "dl", "kg", "cm", "mm", "hr", "hrs", "day", "days",
    # Common partial lab values
    "higher", "lower", "normal", "above", "below", "within",
}


@dataclass
class BiomedicalEntity:
    """Single entity extracted by biomedical-ner-all."""

    text: str
    entity_type: str  # e.g., "Disease_disorder", "Sign_symptom"
    category: str  # e.g., "clinical", "demographics"
    score: float
    start: int
    end: int


@dataclass
class BiomedicalNERResult:
    """Structured extraction result from biomedical-ner-all."""

    # Grouped by category
    clinical: List[BiomedicalEntity] = field(default_factory=list)
    drug_admin: List[BiomedicalEntity] = field(default_factory=list)
    demographics: List[BiomedicalEntity] = field(default_factory=list)
    anatomical: List[BiomedicalEntity] = field(default_factory=list)
    descriptive: List[BiomedicalEntity] = field(default_factory=list)
    temporal: List[BiomedicalEntity] = field(default_factory=list)
    other: List[BiomedicalEntity] = field(default_factory=list)

    # All raw entities
    raw_entities: List[Dict[str, Any]] = field(default_factory=list)

    def get_by_type(self, entity_type: str) -> List[BiomedicalEntity]:
        """Get all entities of a specific type."""
        all_entities = (
            self.clinical
            + self.drug_admin
            + self.demographics
            + self.anatomical
            + self.descriptive
            + self.temporal
            + self.other
        )
        return [e for e in all_entities if e.entity_type == entity_type]

    def to_summary(self) -> Dict[str, Any]:
        """Convert to summary dict for export."""
        return {
            "clinical": {
                "diseases": [e.text for e in self.clinical if e.entity_type == "Disease_disorder"],
                "symptoms": [e.text for e in self.clinical if e.entity_type == "Sign_symptom"],
                "diagnostic_procedures": [e.text for e in self.clinical if e.entity_type == "Diagnostic_procedure"],
                "therapeutic_procedures": [e.text for e in self.clinical if e.entity_type == "Therapeutic_procedure"],
                "medications": [e.text for e in self.clinical if e.entity_type == "Medication"],
                "lab_values": [e.text for e in self.clinical if e.entity_type == "Lab_value"],
                "outcomes": [e.text for e in self.clinical if e.entity_type == "Outcome"],
            },
            "demographics": {
                "ages": [e.text for e in self.demographics if e.entity_type == "Age"],
                "sex": [e.text for e in self.demographics if e.entity_type == "Sex"],
                "family_history": [e.text for e in self.demographics if e.entity_type == "Family_history"],
            },
            "temporal": {
                "dates": [e.text for e in self.temporal if e.entity_type == "Date"],
                "durations": [e.text for e in self.temporal if e.entity_type == "Duration"],
            },
            "entity_counts": self._count_by_type(),
            "category_counts": {
                "clinical": len(self.clinical),
                "drug_admin": len(self.drug_admin),
                "demographics": len(self.demographics),
                "anatomical": len(self.anatomical),
                "descriptive": len(self.descriptive),
                "temporal": len(self.temporal),
                "other": len(self.other),
            },
        }

    def _count_by_type(self) -> Dict[str, int]:
        """Count entities by type."""
        counts: Dict[str, int] = {}
        all_entities = (
            self.clinical
            + self.drug_admin
            + self.demographics
            + self.anatomical
            + self.descriptive
            + self.temporal
            + self.other
        )
        for entity in all_entities:
            counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1
        return counts


class BiomedicalNEREnricher:
    """
    Enricher using d4data/biomedical-ner-all for comprehensive biomedical NER.

    This model extracts 84 biomedical entity types from text using a
    DistilBERT model fine-tuned on the Maccrobat dataset.
    """

    MODEL_NAME = "d4data/biomedical-ner-all"
    MAX_SEQUENCE_LENGTH = 512

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.device = config.get("device", -1)  # -1 for CPU, 0+ for GPU
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self._pipeline = None

    def _load_model(self) -> bool:
        """Lazy load the transformers pipeline."""
        global _pipeline, _model_loaded

        if _model_loaded and _pipeline is not None:
            self._pipeline = _pipeline
            return True

        try:
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )

            logger.info("Loading d4data/biomedical-ner-all model...")
            tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            model = AutoModelForTokenClassification.from_pretrained(self.MODEL_NAME)

            self._pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=self.device,
            )
            _pipeline = self._pipeline
            _model_loaded = True
            logger.info("d4data/biomedical-ner-all model loaded successfully")
            return True

        except ImportError:
            logger.warning(
                "transformers library not installed. "
                "Install with: pip install transformers torch"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load biomedical-ner-all model: {e}")
            return False

    def extract(self, text: str) -> BiomedicalNERResult:
        """
        Extract biomedical entities from text.

        Args:
            text: Input text (abstract, section, or document)

        Returns:
            BiomedicalNERResult with categorized entities
        """
        if not self._load_model():
            return BiomedicalNERResult()

        assert self._pipeline is not None  # Guaranteed after successful _load_model()

        if not text or not text.strip():
            return BiomedicalNERResult()

        # Process text in chunks if too long
        chunks = self._split_into_chunks(text)
        all_entities = []
        offset = 0

        for chunk in chunks:
            if len(chunk.strip()) < 10:
                offset += len(chunk)
                continue

            try:
                entities = self._pipeline(chunk)

                # Adjust offsets for full text position
                for ent in entities:
                    ent["start"] += offset
                    ent["end"] += offset
                    all_entities.append(ent)

            except Exception as e:
                logger.warning(f"Error processing chunk: {e}")

            offset += len(chunk)

        return self._categorize_entities(all_entities)

    def _split_into_chunks(self, text: str, chunk_size: int = 1500) -> List[str]:
        """Split text into chunks for processing."""
        # Split by sentences/paragraphs to avoid breaking mid-entity
        chunks = []
        current_chunk = ""

        # Split by double newlines (paragraphs) first
        paragraphs = text.split("\n\n")

        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    def _categorize_entities(
        self, entities: List[Dict[str, Any]]
    ) -> BiomedicalNERResult:
        """Categorize raw NER entities into typed lists."""
        result = BiomedicalNERResult(raw_entities=entities)

        for ent in entities:
            score = ent.get("score", 0)
            if score < self.confidence_threshold:
                continue

            # Get entity text
            text = ent.get("word", "")

            # Filter out garbage/subword tokens
            # - Skip tokens with ## (BERT subword continuation)
            # - Skip tokens that are too short (< 4 chars for most, allow >= 3 for specific valid entities)
            # - Skip tokens starting with punctuation (partial units like "· 73 m²")
            # - Skip known garbage tokens
            if not text:
                continue

            text_stripped = text.strip()
            text_lower = text_stripped.lower()

            # Skip BERT subword tokens
            if "##" in text:
                continue

            # Skip tokens starting with punctuation (partial extractions)
            if text_stripped and text_stripped[0] in "·•.,;:/\\|<>=-+*":
                continue

            # Skip very short tokens (less than 4 chars)
            if len(text_stripped) < 4:
                continue

            # Skip known garbage tokens
            if text_lower in GARBAGE_TOKENS:
                continue

            # Skip tokens that are just numbers or units
            if text_stripped.replace(".", "").replace(",", "").replace(" ", "").isdigit():
                continue

            # Get entity type (remove B- or I- prefix if present)
            raw_label = ent.get("entity_group", ent.get("entity", ""))
            entity_type = raw_label.replace("B-", "").replace("I-", "")

            # Determine category
            category = ENTITY_TO_CATEGORY.get(entity_type, "other")

            entity = BiomedicalEntity(
                text=text,
                entity_type=entity_type,
                category=category,
                score=score,
                start=ent.get("start", 0),
                end=ent.get("end", 0),
            )

            # Add to appropriate category list
            if category == "clinical":
                result.clinical.append(entity)
            elif category == "drug_admin":
                result.drug_admin.append(entity)
            elif category == "demographics":
                result.demographics.append(entity)
            elif category == "anatomical":
                result.anatomical.append(entity)
            elif category == "descriptive":
                result.descriptive.append(entity)
            elif category == "temporal":
                result.temporal.append(entity)
            else:
                result.other.append(entity)

        return result


# Convenience function for quick extraction
def extract_biomedical_entities(
    text: str, config: Optional[Dict[str, Any]] = None
) -> BiomedicalNERResult:
    """
    Quick extraction of biomedical entities from text.

    Args:
        text: Input text
        config: Optional configuration

    Returns:
        BiomedicalNERResult
    """
    enricher = BiomedicalNEREnricher(config)
    return enricher.extract(text)


__all__ = [
    "ENTITY_CATEGORIES",
    "ALL_ENTITY_TYPES",
    "ENTITY_TO_CATEGORY",
    "BiomedicalEntity",
    "BiomedicalNERResult",
    "BiomedicalNEREnricher",
    "extract_biomedical_entities",
]
