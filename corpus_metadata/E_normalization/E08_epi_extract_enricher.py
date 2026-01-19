# corpus_metadata/E_normalization/E08_epi_extract_enricher.py
"""
EpiExtract4GARD-v2 integration for rare disease epidemiology NER.

Uses NCATS BioBERT-based model to extract:
- LOC: Geographic locations
- EPI: Epidemiology type indicators (prevalence, incidence, etc.)
- STAT: Statistical rates and values

Model: https://huggingface.co/ncats/EpiExtract4GARD-v2
Paper: https://translational-medicine.biomedcentral.com/articles/10.1186/s12967-023-04011-y
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from A_core.A07_feasibility_models import EpidemiologyData

logger = logging.getLogger(__name__)

# Lazy load transformers to avoid import overhead
_pipeline = None
_model_loaded = False


@dataclass
class EpiEntity:
    """Single epidemiology entity extracted by the model."""

    text: str
    label: str  # LOC, EPI, STAT
    score: float
    start: int
    end: int


@dataclass
class EpiExtractionResult:
    """Structured extraction result from EpiExtract4GARD."""

    locations: List[EpiEntity] = field(default_factory=list)
    epi_types: List[EpiEntity] = field(default_factory=list)
    statistics: List[EpiEntity] = field(default_factory=list)
    raw_entities: List[Dict[str, Any]] = field(default_factory=list)

    def to_epidemiology_data(self) -> List[EpidemiologyData]:
        """Convert to EpidemiologyData models for integration with feasibility pipeline."""
        results = []

        # Group statistics with their associated epi types and locations
        for stat in self.statistics:
            # Find closest epi_type (prevalence/incidence indicator)
            epi_type = self._find_nearest_entity(stat, self.epi_types)
            location = self._find_nearest_entity(stat, self.locations)

            # Determine data_type from epi_type entity
            data_type = self._classify_epi_type(epi_type.text if epi_type else "")

            # Normalize the statistical value
            normalized = self._normalize_stat_value(stat.text)

            results.append(
                EpidemiologyData(
                    data_type=data_type,
                    value=stat.text,
                    normalized_value=normalized,
                    geography=location.text if location else None,
                    source="EpiExtract4GARD-v2",
                )
            )

        return results

    def _find_nearest_entity(
        self, target: EpiEntity, candidates: List[EpiEntity], max_distance: int = 200
    ) -> Optional[EpiEntity]:
        """Find the nearest entity to target within max_distance characters."""
        if not candidates:
            return None

        nearest = None
        min_dist = float("inf")

        for cand in candidates:
            # Distance from end of candidate to start of target (or vice versa)
            dist = min(
                abs(target.start - cand.end),
                abs(cand.start - target.end),
            )
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                nearest = cand

        return nearest

    def _classify_epi_type(self, epi_text: str) -> str:
        """Classify epidemiology type from extracted text."""
        text_lower = epi_text.lower()

        if any(
            kw in text_lower
            for kw in ["prevalence", "prevalent", "affected", "living with"]
        ):
            return "prevalence"
        elif any(
            kw in text_lower
            for kw in ["incidence", "incident", "new cases", "diagnosed", "annual"]
        ):
            return "incidence"
        elif any(kw in text_lower for kw in ["mortality", "death", "survival", "fatal"]):
            return "mortality"
        elif any(kw in text_lower for kw in ["age", "male", "female", "sex", "gender"]):
            return "demographics"
        else:
            return "epidemiology"

    def _normalize_stat_value(self, stat_text: str) -> Optional[float]:
        """Normalize statistical value to per-million rate."""
        text = stat_text.lower().strip()

        # Pattern: X per Y (e.g., "1.5 per 100,000")
        per_match = re.search(
            r"([\d.,]+)\s*(?:per|/|in)\s*([\d.,]+(?:\s*(?:million|thousand|hundred))?)",
            text,
        )
        if per_match:
            try:
                numerator = float(per_match.group(1).replace(",", ""))
                denom_str = per_match.group(2).replace(",", "")

                # Handle word-based denominators
                if "million" in denom_str:
                    denom_num = re.search(r"[\d.]+", denom_str)
                    denominator = float(denom_num.group()) * 1_000_000 if denom_num else 1_000_000
                elif "thousand" in denom_str:
                    denom_num = re.search(r"[\d.]+", denom_str)
                    denominator = float(denom_num.group()) * 1_000 if denom_num else 1_000
                elif "hundred" in denom_str:
                    denom_num = re.search(r"[\d.]+", denom_str)
                    denominator = float(denom_num.group()) * 100 if denom_num else 100
                else:
                    denominator = float(denom_str) if denom_str else 1

                # Normalize to per million
                if denominator > 0:
                    return (numerator / denominator) * 1_000_000
            except (ValueError, AttributeError):
                pass

        # Pattern: X% (percentage)
        pct_match = re.search(r"([\d.,]+)\s*%", text)
        if pct_match:
            try:
                pct = float(pct_match.group(1).replace(",", ""))
                return pct * 10_000  # Convert % to per million
            except ValueError:
                pass

        # Pattern: 1:X or 1 in X
        ratio_match = re.search(r"1\s*(?::|in)\s*([\d,]+)", text)
        if ratio_match:
            try:
                denominator = float(ratio_match.group(1).replace(",", ""))
                if denominator > 0:
                    return 1_000_000 / denominator
            except ValueError:
                pass

        return None


class EpiExtractEnricher:
    """
    Enricher using EpiExtract4GARD-v2 BioBERT model for epidemiology NER.

    Extracts structured epidemiology information from text using a fine-tuned
    BioBERT model trained on rare disease literature.
    """

    MODEL_NAME = "ncats/EpiExtract4GARD-v2"
    MAX_SEQUENCE_LENGTH = 192  # Model limit

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.device = config.get("device", -1)  # -1 for CPU, 0+ for GPU
        self.batch_size = config.get("batch_size", 8)
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

            logger.info("Loading EpiExtract4GARD-v2 model...")
            model = AutoModelForTokenClassification.from_pretrained(self.MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

            self._pipeline = pipeline(
                "token-classification",  # NER is token-classification
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=self.device,
            )
            _pipeline = self._pipeline
            _model_loaded = True
            logger.info("EpiExtract4GARD-v2 model loaded successfully")
            return True

        except ImportError:
            logger.warning(
                "transformers library not installed. "
                "Install with: pip install transformers torch"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load EpiExtract4GARD-v2 model: {e}")
            return False

    def extract(self, text: str) -> EpiExtractionResult:
        """
        Extract epidemiology entities from text.

        Args:
            text: Input text (abstract, section, or document)

        Returns:
            EpiExtractionResult with categorized entities
        """
        if not self._load_model():
            return EpiExtractionResult()

        if not text or not text.strip():
            return EpiExtractionResult()

        # Split into sentences for better handling of long texts
        sentences = self._split_sentences(text)
        all_entities = []
        offset = 0

        for sentence in sentences:
            if len(sentence.strip()) < 10:
                offset += len(sentence)
                continue

            try:
                # Truncate if needed (model limit is 192 tokens)
                truncated = sentence[: self.MAX_SEQUENCE_LENGTH * 4]  # Rough char limit
                entities = self._pipeline(truncated)

                # Adjust offsets for full text position
                for ent in entities:
                    ent["start"] += offset
                    ent["end"] += offset
                    all_entities.append(ent)

            except Exception as e:
                logger.warning(f"Error processing sentence: {e}")

            offset += len(sentence)

        return self._categorize_entities(all_entities)

    def extract_batch(self, texts: List[str]) -> List[EpiExtractionResult]:
        """
        Extract epidemiology entities from multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of EpiExtractionResult for each input
        """
        if not self._load_model():
            return [EpiExtractionResult() for _ in texts]

        results = []
        for text in texts:
            results.append(self.extract(text))

        return results

    def enrich_epidemiology_data(
        self, text: str, existing_data: Optional[List[EpidemiologyData]] = None
    ) -> List[EpidemiologyData]:
        """
        Enrich existing epidemiology data or extract new data from text.

        Args:
            text: Source text
            existing_data: Existing EpidemiologyData to merge with

        Returns:
            Merged list of EpidemiologyData
        """
        extraction = self.extract(text)
        new_data = extraction.to_epidemiology_data()

        if not existing_data:
            return new_data

        # Merge: add new data that doesn't duplicate existing
        existing_values = {d.value.lower().strip() for d in existing_data}
        merged = list(existing_data)

        for item in new_data:
            if item.value.lower().strip() not in existing_values:
                merged.append(item)

        return merged

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences for processing."""
        # Simple sentence splitting - handles common patterns
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return sentences

    def _categorize_entities(
        self, entities: List[Dict[str, Any]]
    ) -> EpiExtractionResult:
        """Categorize raw NER entities into typed lists."""
        result = EpiExtractionResult(raw_entities=entities)

        for ent in entities:
            if ent.get("score", 0) < self.confidence_threshold:
                continue

            entity = EpiEntity(
                text=ent.get("word", ""),
                label=ent.get("entity_group", ""),
                score=ent.get("score", 0),
                start=ent.get("start", 0),
                end=ent.get("end", 0),
            )

            if entity.label == "LOC":
                result.locations.append(entity)
            elif entity.label == "EPI":
                result.epi_types.append(entity)
            elif entity.label == "STAT":
                result.statistics.append(entity)

        return result


# Convenience function for quick extraction
def extract_epidemiology(
    text: str, config: Optional[Dict[str, Any]] = None
) -> List[EpidemiologyData]:
    """
    Quick extraction of epidemiology data from text.

    Args:
        text: Input text
        config: Optional configuration

    Returns:
        List of EpidemiologyData
    """
    enricher = EpiExtractEnricher(config)
    result = enricher.extract(text)
    return result.to_epidemiology_data()
