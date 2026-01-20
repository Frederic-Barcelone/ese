# corpus_metadata/E_normalization/E08_epi_extract_enricher.py
"""
EpiExtract4GARD-v2 integration for rare disease epidemiology NER.

This module provides Named Entity Recognition (NER) for epidemiology data
using the NCATS BioBERT-based model fine-tuned on rare disease literature.

Entity Types Extracted:
    - LOC: Geographic locations (countries, regions, populations)
    - EPI: Epidemiology type indicators (prevalence, incidence, mortality)
    - STAT: Statistical rates and values (e.g., "1 per 100,000")

The module handles:
    - Lazy model loading to minimize startup overhead
    - Sentence-level processing for long documents
    - Statistical value normalization to per-million rates
    - Entity proximity analysis to associate statistics with their context

Model Information:
    - Source: https://huggingface.co/ncats/EpiExtract4GARD-v2
    - Paper: https://doi.org/10.1186/s12967-023-04011-y
    - Architecture: BioBERT fine-tuned for token classification

Example:
    >>> from E_normalization.E08_epi_extract_enricher import EpiExtractEnricher
    >>> enricher = EpiExtractEnricher()
    >>> result = enricher.extract("The prevalence is 1 per 100,000 in Europe.")
    >>> print(result.statistics[0].text)
    '1 per 100,000'

Dependencies:
    - transformers (HuggingFace)
    - torch
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from A_core.A00_logging import get_logger
from A_core.A07_feasibility_models import EpidemiologyData

if TYPE_CHECKING:
    from transformers import Pipeline

# Module logger
logger = get_logger(__name__)

# Global state for lazy model loading (singleton pattern)
_pipeline: Optional["Pipeline"] = None
_model_loaded: bool = False


@dataclass
class EpiEntity:
    """
    Single epidemiology entity extracted by the NER model.

    Attributes:
        text: The extracted text span.
        label: Entity type (LOC, EPI, or STAT).
        score: Model confidence score (0.0-1.0).
        start: Character start position in source text.
        end: Character end position in source text.
    """

    text: str
    label: str  # LOC, EPI, STAT
    score: float
    start: int
    end: int

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"EpiEntity({self.label}: '{self.text}' @{self.start}-{self.end}, conf={self.score:.2f})"


@dataclass
class EpiExtractionResult:
    """
    Structured extraction result from EpiExtract4GARD model.

    Groups extracted entities by type and provides methods for
    converting to pipeline-compatible data structures.

    Attributes:
        locations: Geographic location entities (LOC).
        epi_types: Epidemiology type indicators (EPI).
        statistics: Statistical values and rates (STAT).
        raw_entities: Original model output for debugging.
    """

    locations: List[EpiEntity] = field(default_factory=list)
    epi_types: List[EpiEntity] = field(default_factory=list)
    statistics: List[EpiEntity] = field(default_factory=list)
    raw_entities: List[Dict[str, Any]] = field(default_factory=list)

    def to_epidemiology_data(self) -> List[EpidemiologyData]:
        """
        Convert extraction results to EpidemiologyData models.

        Associates each statistic with its nearest epidemiology type
        and location entities using proximity analysis.

        Returns:
            List of EpidemiologyData objects suitable for the feasibility pipeline.
        """
        results: List[EpidemiologyData] = []

        for stat in self.statistics:
            # Find closest epi_type (prevalence/incidence indicator)
            epi_type = self._find_nearest_entity(stat, self.epi_types)
            location = self._find_nearest_entity(stat, self.locations)

            # Determine data_type from epi_type entity
            data_type = self._classify_epi_type(epi_type.text if epi_type else "")

            # Normalize the statistical value to per-million
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
        self,
        target: EpiEntity,
        candidates: List[EpiEntity],
        max_distance: int = 200,
    ) -> Optional[EpiEntity]:
        """
        Find the nearest entity to target within max_distance characters.

        Uses character-level proximity to associate entities that appear
        close together in the source text.

        Args:
            target: The entity to find neighbors for.
            candidates: List of candidate entities to search.
            max_distance: Maximum character distance to consider.

        Returns:
            The nearest candidate entity, or None if none within range.
        """
        if not candidates:
            return None

        nearest: Optional[EpiEntity] = None
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
        """
        Classify epidemiology type from extracted indicator text.

        Args:
            epi_text: Text from an EPI entity.

        Returns:
            Classification string: "prevalence", "incidence", "mortality",
            "demographics", or "epidemiology" (default).
        """
        text_lower = epi_text.lower()

        # Prevalence indicators
        if any(kw in text_lower for kw in ["prevalence", "prevalent", "affected", "living with"]):
            return "prevalence"
        # Incidence indicators
        if any(kw in text_lower for kw in ["incidence", "incident", "new cases", "diagnosed", "annual"]):
            return "incidence"
        # Mortality indicators
        if any(kw in text_lower for kw in ["mortality", "death", "survival", "fatal"]):
            return "mortality"
        # Demographic indicators
        if any(kw in text_lower for kw in ["age", "male", "female", "sex", "gender"]):
            return "demographics"

        return "epidemiology"

    def _normalize_stat_value(self, stat_text: str) -> Optional[float]:
        """
        Normalize statistical value to per-million rate.

        Handles various formats:
            - "X per Y" (e.g., "1.5 per 100,000")
            - "X%" (percentages)
            - "1:X" or "1 in X" (ratios)

        Args:
            stat_text: Raw statistical text from STAT entity.

        Returns:
            Normalized per-million rate, or None if parsing fails.
        """
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

    Extracts structured epidemiology information from clinical text using
    a fine-tuned BioBERT model trained on rare disease literature.

    The model is loaded lazily on first use and cached globally to avoid
    repeated loading overhead.

    Attributes:
        MODEL_NAME: HuggingFace model identifier.
        MAX_SEQUENCE_LENGTH: Maximum tokens per inference (model limit).
        device: Compute device (-1 for CPU, 0+ for GPU).
        batch_size: Batch size for inference.
        confidence_threshold: Minimum score to accept an entity.

    Example:
        >>> enricher = EpiExtractEnricher({"confidence_threshold": 0.7})
        >>> result = enricher.extract(abstract_text)
        >>> for stat in result.statistics:
        ...     print(f"{stat.text} (confidence: {stat.score:.2f})")
    """

    MODEL_NAME: str = "ncats/EpiExtract4GARD-v2"
    MAX_SEQUENCE_LENGTH: int = 192  # Model token limit

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the enricher with configuration.

        Args:
            config: Optional configuration dictionary with keys:
                - device: Compute device (-1 for CPU, 0+ for GPU)
                - batch_size: Inference batch size
                - confidence_threshold: Minimum entity confidence (0.0-1.0)
        """
        config = config or {}
        self.device: int = config.get("device", -1)
        self.batch_size: int = config.get("batch_size", 8)
        self.confidence_threshold: float = config.get("confidence_threshold", 0.5)
        self._pipeline: Optional["Pipeline"] = None

    def _load_model(self) -> bool:
        """
        Lazy load the transformers pipeline.

        Uses global singleton pattern to avoid loading the model multiple
        times across different enricher instances.

        Returns:
            True if model loaded successfully, False otherwise.
        """
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

            # Suppress "Invalid model-index" warning from HF Hub
            # (model metadata issue, not functional)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Invalid model-index")
                model = AutoModelForTokenClassification.from_pretrained(self.MODEL_NAME)
                tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

            self._pipeline = pipeline(
                "token-classification",
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

        Processes text sentence-by-sentence to handle documents longer
        than the model's maximum sequence length.

        Args:
            text: Input text (abstract, section, or full document).

        Returns:
            EpiExtractionResult with categorized entities.
        """
        if not self._load_model():
            return EpiExtractionResult()

        if not text or not text.strip():
            return EpiExtractionResult()

        # Split into sentences for better handling of long texts
        sentences = self._split_sentences(text)
        all_entities: List[Dict[str, Any]] = []
        offset = 0

        for sentence in sentences:
            if len(sentence.strip()) < 10:
                offset += len(sentence)
                continue

            try:
                # Truncate if needed (model limit is 192 tokens, ~4 chars/token)
                truncated = sentence[: self.MAX_SEQUENCE_LENGTH * 4]
                entities = self._pipeline(truncated)  # type: ignore[misc]

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
            texts: List of input texts.

        Returns:
            List of EpiExtractionResult, one per input text.
        """
        if not self._load_model():
            return [EpiExtractionResult() for _ in texts]

        return [self.extract(text) for text in texts]

    def enrich_epidemiology_data(
        self,
        text: str,
        existing_data: Optional[List[EpidemiologyData]] = None,
    ) -> List[EpidemiologyData]:
        """
        Enrich existing epidemiology data or extract new data from text.

        Merges newly extracted data with existing data, avoiding duplicates
        based on value text matching.

        Args:
            text: Source text to extract from.
            existing_data: Existing EpidemiologyData to merge with.

        Returns:
            Merged list of EpidemiologyData (existing + new unique items).
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
        """
        Split text into sentences for processing.

        Uses a simple regex pattern that handles common sentence boundaries.

        Args:
            text: Input text to split.

        Returns:
            List of sentence strings.
        """
        return re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    def _categorize_entities(
        self,
        entities: List[Dict[str, Any]],
    ) -> EpiExtractionResult:
        """
        Categorize raw NER output into typed entity lists.

        Filters entities by confidence threshold and routes them
        to appropriate lists based on entity label.

        Args:
            entities: Raw model output dictionaries.

        Returns:
            EpiExtractionResult with categorized entities.
        """
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


def extract_epidemiology(
    text: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[EpidemiologyData]:
    """
    Quick extraction of epidemiology data from text.

    Convenience function for one-off extractions without
    managing an enricher instance.

    Args:
        text: Input text to process.
        config: Optional enricher configuration.

    Returns:
        List of EpidemiologyData extracted from the text.

    Example:
        >>> data = extract_epidemiology("Prevalence is 1 in 50,000.")
        >>> print(data[0].normalized_value)
        20.0
    """
    enricher = EpiExtractEnricher(config)
    result = enricher.extract(text)
    return result.to_epidemiology_data()
