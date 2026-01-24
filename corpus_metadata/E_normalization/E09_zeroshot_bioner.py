# corpus_metadata/E_normalization/E09_zeroshot_bioner.py
"""
ZeroShotBioNER integration for flexible biomedical entity extraction.

Uses Bayer/Serbian AI Institute BioBERT-based model for zero-shot NER.
Extracts entities not covered by other extractors:
- ADE: Adverse Drug Events
- Dosage, Frequency, Strength, Form, Route, Duration: Drug administration
- Reason: Treatment rationale

Model: https://huggingface.co/ProdicusII/ZeroShotBioNER
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy load transformers to avoid import overhead
_model = None
_tokenizer = None
_model_loaded = False


# Entity types that ADD VALUE (not already covered by existing extractors)
# Drugs/Chemicals/Diseases are already handled by lexicons + scispaCy
ENTITY_TYPES_TO_EXTRACT = [
    # Adverse events
    "ADE",  # Adverse Drug Event
    # Drug administration details
    "Dosage",
    "Frequency",
    "Strength",
    "Form",
    "Route",
    "Duration",
    # Treatment context
    "Reason",
]


@dataclass
class BioNEREntity:
    """Single entity extracted by ZeroShotBioNER."""

    text: str
    entity_type: str
    score: float
    start: int
    end: int


@dataclass
class ZeroShotExtractionResult:
    """Structured extraction result from ZeroShotBioNER."""

    # Adverse events
    adverse_events: List[BioNEREntity] = field(default_factory=list)
    # Drug administration
    dosages: List[BioNEREntity] = field(default_factory=list)
    frequencies: List[BioNEREntity] = field(default_factory=list)
    strengths: List[BioNEREntity] = field(default_factory=list)
    forms: List[BioNEREntity] = field(default_factory=list)
    routes: List[BioNEREntity] = field(default_factory=list)
    durations: List[BioNEREntity] = field(default_factory=list)
    # Treatment context
    reasons: List[BioNEREntity] = field(default_factory=list)
    # All raw entities
    raw_entities: List[Dict[str, Any]] = field(default_factory=list)

    def to_summary(self) -> Dict[str, Any]:
        """Convert to summary dict for export."""
        return {
            "adverse_events": [
                {"text": e.text, "score": e.score} for e in self.adverse_events
            ],
            "drug_administration": {
                "dosages": [e.text for e in self.dosages],
                "frequencies": [e.text for e in self.frequencies],
                "strengths": [e.text for e in self.strengths],
                "forms": [e.text for e in self.forms],
                "routes": [e.text for e in self.routes],
                "durations": [e.text for e in self.durations],
            },
            "treatment_reasons": [e.text for e in self.reasons],
            "entity_counts": {
                "ADE": len(self.adverse_events),
                "dosage": len(self.dosages),
                "frequency": len(self.frequencies),
                "strength": len(self.strengths),
                "form": len(self.forms),
                "route": len(self.routes),
                "duration": len(self.durations),
                "reason": len(self.reasons),
            },
        }


class ZeroShotBioNEREnricher:
    """
    Enricher using ZeroShotBioNER for flexible biomedical entity extraction.

    This model uses a unique approach: it takes TWO inputs:
    1. The entity type to search for (e.g., "ADE", "Dosage")
    2. The text to search in

    This allows extraction of any entity type without retraining.
    """

    MODEL_NAME = "ProdicusII/ZeroShotBioNER"
    MAX_SEQUENCE_LENGTH = 512

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.device = config.get("device", -1)  # -1 for CPU, 0+ for GPU
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.entity_types = config.get("entity_types", ENTITY_TYPES_TO_EXTRACT)
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> bool:
        """Lazy load the model and tokenizer."""
        global _model, _tokenizer, _model_loaded

        if _model_loaded and _model is not None:
            self._model = _model
            self._tokenizer = _tokenizer
            return True

        try:
            from transformers import AutoTokenizer, BertForTokenClassification

            logger.info("Loading ZeroShotBioNER model...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = BertForTokenClassification.from_pretrained(
                self.MODEL_NAME, num_labels=2
            )
            assert self._model is not None  # Type narrowing for mypy

            # Move to device if GPU
            if self.device >= 0:
                import torch

                if torch.cuda.is_available():
                    self._model = self._model.to(f"cuda:{self.device}")

            self._model.eval()

            _model = self._model
            _tokenizer = self._tokenizer
            _model_loaded = True
            logger.info("ZeroShotBioNER model loaded successfully")
            return True

        except ImportError:
            logger.warning(
                "transformers library not installed. "
                "Install with: pip install transformers torch"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load ZeroShotBioNER model: {e}")
            return False

    def _extract_entity_type(
        self, entity_type: str, text: str
    ) -> List[BioNEREntity]:
        """Extract all occurrences of a specific entity type from text."""
        import torch

        if not text or len(text.strip()) < 5:
            return []

        # Ensure model and tokenizer are loaded
        assert self._model is not None and self._tokenizer is not None

        try:
            # Truncate text if needed
            truncated_text = text[: self.MAX_SEQUENCE_LENGTH * 4]

            # Encode: [CLS] entity_type [SEP] text [SEP]
            encodings = self._tokenizer(
                entity_type,
                truncated_text,
                is_split_into_words=False,
                padding=True,
                truncation=True,
                add_special_tokens=True,
                max_length=self.MAX_SEQUENCE_LENGTH,
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            # Get offset mapping before moving to device
            offset_mapping = encodings.pop("offset_mapping")[0].tolist()

            # Move to device if needed
            if self.device >= 0 and torch.cuda.is_available():
                encodings = {k: v.to(f"cuda:{self.device}") for k, v in encodings.items()}

            # Run inference
            with torch.no_grad():
                outputs = self._model(**encodings)

            # Get predictions (0 = O, 1 = entity)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
            scores = torch.softmax(outputs.logits, dim=-1)[0, :, 1].cpu().tolist()

            # Convert token predictions to entities
            entities = self._tokens_to_entities(
                predictions, scores, offset_mapping, truncated_text, entity_type
            )

            return entities

        except Exception as e:
            logger.warning(f"Error extracting {entity_type}: {e}")
            return []

    def _tokens_to_entities(
        self,
        predictions: List[int],
        scores: List[float],
        offset_mapping: List[tuple],
        text: str,
        entity_type: str,
    ) -> List[BioNEREntity]:
        """Convert token-level predictions to entity spans."""
        entities = []
        current_entity_start = None
        current_entity_scores = []

        # Skip special tokens and first segment (entity type)
        # Find where the second segment starts (after first [SEP])
        in_second_segment = False

        for idx, (pred, score, offsets) in enumerate(
            zip(predictions, scores, offset_mapping)
        ):
            # Skip special tokens (offset is (0, 0))
            if offsets == (0, 0):
                if current_entity_start is not None:
                    # Close current entity
                    in_second_segment = True
                continue

            # Only process second segment (the actual text)
            if not in_second_segment:
                # Check if we've passed the entity type segment
                if idx > 0 and offset_mapping[idx - 1] == (0, 0):
                    in_second_segment = True
                else:
                    continue

            start, end = offsets

            if pred == 1:  # Entity token
                if current_entity_start is None:
                    current_entity_start = start
                current_entity_scores.append(score)
            else:
                if current_entity_start is not None:
                    # Close entity
                    avg_score = sum(current_entity_scores) / len(current_entity_scores)
                    if avg_score >= self.confidence_threshold:
                        entity_text = text[current_entity_start:start].strip()
                        if entity_text:
                            entities.append(
                                BioNEREntity(
                                    text=entity_text,
                                    entity_type=entity_type,
                                    score=avg_score,
                                    start=current_entity_start,
                                    end=start,
                                )
                            )
                    current_entity_start = None
                    current_entity_scores = []

        # Handle entity at end of text
        if current_entity_start is not None and current_entity_scores:
            avg_score = sum(current_entity_scores) / len(current_entity_scores)
            if avg_score >= self.confidence_threshold:
                entity_text = text[current_entity_start:].strip()
                if entity_text:
                    entities.append(
                        BioNEREntity(
                            text=entity_text,
                            entity_type=entity_type,
                            score=avg_score,
                            start=current_entity_start,
                            end=len(text),
                        )
                    )

        return entities

    def extract(self, text: str) -> ZeroShotExtractionResult:
        """
        Extract all configured entity types from text.

        Args:
            text: Input text (abstract, section, or document)

        Returns:
            ZeroShotExtractionResult with categorized entities
        """
        if not self._load_model():
            return ZeroShotExtractionResult()

        if not text or not text.strip():
            return ZeroShotExtractionResult()

        result = ZeroShotExtractionResult()

        # Extract each entity type
        for entity_type in self.entity_types:
            entities = self._extract_entity_type(entity_type, text)

            # Categorize into result
            for entity in entities:
                result.raw_entities.append(
                    {
                        "text": entity.text,
                        "type": entity.entity_type,
                        "score": entity.score,
                        "start": entity.start,
                        "end": entity.end,
                    }
                )

                if entity_type == "ADE":
                    result.adverse_events.append(entity)
                elif entity_type == "Dosage":
                    result.dosages.append(entity)
                elif entity_type == "Frequency":
                    result.frequencies.append(entity)
                elif entity_type == "Strength":
                    result.strengths.append(entity)
                elif entity_type == "Form":
                    result.forms.append(entity)
                elif entity_type == "Route":
                    result.routes.append(entity)
                elif entity_type == "Duration":
                    result.durations.append(entity)
                elif entity_type == "Reason":
                    result.reasons.append(entity)

        return result

    def extract_custom(
        self, text: str, entity_types: List[str]
    ) -> Dict[str, List[BioNEREntity]]:
        """
        Extract custom entity types (zero-shot capability).

        Args:
            text: Input text
            entity_types: List of entity type names to extract

        Returns:
            Dict mapping entity type to list of extracted entities
        """
        if not self._load_model():
            return {}

        results = {}
        for entity_type in entity_types:
            results[entity_type] = self._extract_entity_type(entity_type, text)

        return results


# Convenience function for quick extraction
def extract_biomedical_entities(
    text: str, config: Optional[Dict[str, Any]] = None
) -> ZeroShotExtractionResult:
    """
    Quick extraction of biomedical entities from text.

    Args:
        text: Input text
        config: Optional configuration

    Returns:
        ZeroShotExtractionResult
    """
    enricher = ZeroShotBioNEREnricher(config)
    return enricher.extract(text)
