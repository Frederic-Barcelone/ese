"""
VLM visual enrichment for extracted figures and tables.

This module uses Claude Vision to enrich extracted visual elements with
classification, reference parsing, and validation. Enhances raw visual
extraction with intelligent type detection and multi-page handling.

Key Components:
    - VLMVisualEnricher: Main enricher using Claude Vision API
    - VLMConfig: Configuration for model and confidence thresholds
    - VLMClassificationResult: Type classification output
    - VLMTableValidation: Table structure validation results
    - Enrichment features:
        - Type classification (table vs figure)
        - Reference parsing ("Table 1" -> structured ref)
        - Caption validation/extraction
        - Table structure validation
        - Multi-page continuation detection

Example:
    >>> from C_generators.C19_vlm_visual_enrichment import VLMVisualEnricher
    >>> enricher = VLMVisualEnricher(config=VLMConfig())
    >>> result = enricher.enrich(visual_element)
    >>> print(f"Type: {result.visual_type}, Ref: {result.reference}")
    Type: TABLE, Ref: VisualReference(type='Table', number=1)

Dependencies:
    - A_core.A13_visual_models: VisualType, VisualReference, VLMEnrichmentResult
    - anthropic: Claude Vision API client
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from A_core.A13_visual_models import (
    ReferenceSource,
    TableStructure,
    VisualReference,
    VisualType,
    VLMClassificationResult,
    VLMEnrichmentResult,
    VLMTableValidation,
)

from Z_utils.Z13_llm_tracking import record_api_usage, resolve_model_tier

logger = logging.getLogger(__name__)


# -------------------------
# Configuration
# -------------------------


@dataclass
class VLMConfig:
    """Configuration for VLM enrichment."""

    model: str = ""
    max_tokens: int = 2000
    temperature: float = 0.0

    def __post_init__(self):
        if not self.model:
            self.model = resolve_model_tier("vlm_visual_enrichment")

    # Classification thresholds
    min_classification_confidence: float = 0.7
    min_validation_confidence: float = 0.8


# -------------------------
# Prompts
# -------------------------


CLASSIFICATION_PROMPT = """Analyze this image from a clinical/scientific document.

Determine:
1. Is this a TABLE or FIGURE?
   - TABLE: Has structured rows/columns with data
   - FIGURE: Chart, graph, diagram, flowchart, image, photo

2. What is the reference label if visible? (e.g., "Table 1", "Figure 2-4", "Exhibit A")

3. What is the full caption text if visible?

4. If this is a continuation of a previous table/figure, indicate which one.

Respond in JSON format:
{
    "visual_type": "table" or "figure",
    "confidence": 0.0-1.0,
    "reference": {
        "raw_string": "Table 1" or null,
        "type_label": "Table" or "Figure" or null,
        "numbers": [1] or [2, 3, 4] or null
    },
    "caption_text": "Full caption text" or null,
    "is_continuation": true/false,
    "continuation_of": "Table 1" or null if not continuation,
    "reasoning": "Brief explanation"
}"""


TABLE_VALIDATION_PROMPT = """Analyze this table image from a clinical document.

The document extraction system parsed this table with the following structure:
Headers: {headers}
Sample rows: {sample_rows}

Check:
1. Is this structure CORRECT? Are rows/columns properly aligned?
2. Are there merged cells that were missed?
3. Are header rows correctly identified?
4. Is any data truncated or misaligned?

If there are issues, provide the corrected structure.

Respond in JSON format:
{
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of issues found"] or [],
    "corrected_headers": [["header1", "header2"]] or null,
    "corrected_rows": [["val1", "val2"]] or null
}"""


REFERENCE_PARSING_PROMPT = """Extract the reference label from this caption text:

Caption: "{caption_text}"

Parse the reference (e.g., "Table 1", "Figure 2-4", "Exhibit A.1").

Respond in JSON format:
{
    "raw_string": "Table 1" or null,
    "type_label": "Table" or "Figure" or "Exhibit" or null,
    "numbers": [1] or [2, 3, 4] or null,
    "suffix": "A" or null,
    "is_range": true/false
}"""


# -------------------------
# VLM Client Interface
# -------------------------


class VLMClient:
    """
    Client for calling Vision LLM.

    This is an interface that can be implemented with different backends
    (Claude API, local VLM, etc.).
    """

    def __init__(self, config: VLMConfig = VLMConfig()):
        self.config = config
        self._client: Optional[Any] = None

    def _get_client(self):
        """Lazy initialization of Claude client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
                logger.info(f"VLM client initialized with model: {self.config.model}")
            except ImportError:
                raise ImportError(
                    "anthropic package required for VLM enrichment. "
                    "Install with: pip install anthropic. "
                    "Without this, visual classification will use heuristics only."
                )
        return self._client

    def call_vision(
        self,
        image_base64: str,
        prompt: str,
        image_media_type: str = "image/png",
    ) -> Optional[str]:
        """
        Call VLM with an image and prompt.

        Args:
            image_base64: Base64-encoded image
            prompt: Text prompt
            image_media_type: MIME type of image

        Returns:
            Model response text, or None if failed
        """
        try:
            client = self._get_client()

            message = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image_media_type,
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )

            record_api_usage(message, self.config.model, "vlm_visual_enrichment")
            return message.content[0].text

        except Exception as e:
            logger.error(f"VLM call failed: {e}")
            return None

    def call_text(self, prompt: str) -> Optional[str]:
        """
        Call LLM with text-only prompt.

        Args:
            prompt: Text prompt

        Returns:
            Model response text, or None if failed
        """
        try:
            client = self._get_client()

            message = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            record_api_usage(message, self.config.model, "vlm_visual_enrichment")
            return message.content[0].text

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None


# -------------------------
# Response Parsing
# -------------------------


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from VLM response.

    Handles cases where JSON is wrapped in markdown code blocks.

    Args:
        response: Raw VLM response

    Returns:
        Parsed JSON dict, or None if parsing fails
    """
    if not response:
        return None

    # Try to extract JSON from code blocks
    text = response.strip()

    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON from VLM response: {text[:100]}...")
        return None


def parse_classification_response(
    response_data: Dict[str, Any],
) -> Optional[VLMClassificationResult]:
    """Parse classification result from VLM response."""
    try:
        visual_type_str = response_data.get("visual_type", "").lower()

        if visual_type_str == "table":
            visual_type = VisualType.TABLE
        elif visual_type_str == "figure":
            visual_type = VisualType.FIGURE
        else:
            visual_type = VisualType.OTHER

        return VLMClassificationResult(
            visual_type=visual_type,
            confidence=float(response_data.get("confidence", 0.8)),
            reasoning=response_data.get("reasoning"),
        )

    except Exception as e:
        logger.warning(f"Failed to parse classification response: {e}")
        return None


def parse_reference_response(
    response_data: Dict[str, Any],
    source: ReferenceSource = ReferenceSource.VLM,
) -> Optional[VisualReference]:
    """Parse reference from VLM response."""
    try:
        ref_data = response_data.get("reference", response_data)

        raw_string = ref_data.get("raw_string")
        type_label = ref_data.get("type_label")
        numbers = ref_data.get("numbers")

        if not raw_string or not type_label or not numbers:
            return None

        return VisualReference(
            raw_string=raw_string,
            type_label=type_label,
            numbers=numbers,
            is_range=ref_data.get("is_range", False),
            suffix=ref_data.get("suffix"),
            source=source,
        )

    except Exception as e:
        logger.warning(f"Failed to parse reference response: {e}")
        return None


def parse_table_validation_response(
    response_data: Dict[str, Any],
) -> Optional[VLMTableValidation]:
    """Parse table validation result from VLM response."""
    try:
        is_misparsed = not response_data.get("is_correct", True)
        confidence = float(response_data.get("confidence", 0.8))
        issues = response_data.get("issues", [])

        corrected_structure = None
        if is_misparsed:
            corrected_headers = response_data.get("corrected_headers")
            corrected_rows = response_data.get("corrected_rows")

            if corrected_headers or corrected_rows:
                corrected_structure = TableStructure(
                    headers=corrected_headers or [],
                    rows=corrected_rows or [],
                    structure_confidence=confidence,
                )

        return VLMTableValidation(
            is_misparsed=is_misparsed,
            confidence=confidence,
            issues=issues,
            corrected_structure=corrected_structure,
        )

    except Exception as e:
        logger.warning(f"Failed to parse table validation response: {e}")
        return None


# -------------------------
# Enrichment Functions
# -------------------------


def classify_visual(
    image_base64: str,
    client: VLMClient,
) -> Optional[VLMEnrichmentResult]:
    """
    Classify a visual using VLM.

    Args:
        image_base64: Base64-encoded image
        client: VLM client

    Returns:
        VLMEnrichmentResult with classification and parsed reference
    """
    response = client.call_vision(image_base64, CLASSIFICATION_PROMPT)

    if not response:
        return None

    response_data = parse_json_response(response)
    if not response_data:
        return None

    # Parse classification
    classification = parse_classification_response(response_data)
    if not classification:
        return None

    # Parse reference
    parsed_reference = parse_reference_response(response_data)

    # Get caption text
    caption_text = response_data.get("caption_text")

    # Check continuation
    is_continuation = response_data.get("is_continuation", False)
    continuation_of = response_data.get("continuation_of")

    return VLMEnrichmentResult(
        classification=classification,
        parsed_reference=parsed_reference,
        extracted_caption=caption_text,
        is_continuation=is_continuation,
        continuation_of_reference=continuation_of,
    )


def validate_table_structure(
    image_base64: str,
    docling_structure: TableStructure,
    client: VLMClient,
) -> Optional[VLMTableValidation]:
    """
    Validate table structure using VLM.

    Args:
        image_base64: Base64-encoded table image
        docling_structure: Table structure from Docling
        client: VLM client

    Returns:
        VLMTableValidation result
    """
    # Prepare prompt with current structure
    headers_str = json.dumps(docling_structure.headers[:3])  # First 3 header rows
    sample_rows = docling_structure.rows[:5]  # First 5 rows
    rows_str = json.dumps(sample_rows)

    prompt = TABLE_VALIDATION_PROMPT.format(
        headers=headers_str,
        sample_rows=rows_str,
    )

    response = client.call_vision(image_base64, prompt)

    if not response:
        return None

    response_data = parse_json_response(response)
    if not response_data:
        return None

    return parse_table_validation_response(response_data)


def parse_reference_from_caption(
    caption_text: str,
    client: VLMClient,
) -> Optional[VisualReference]:
    """
    Parse reference from caption text using LLM.

    Args:
        caption_text: Caption text to parse
        client: VLM client

    Returns:
        VisualReference if parseable
    """
    prompt = REFERENCE_PARSING_PROMPT.format(caption_text=caption_text)

    response = client.call_text(prompt)

    if not response:
        return None

    response_data = parse_json_response(response)
    if not response_data:
        return None

    return parse_reference_response(response_data, ReferenceSource.VLM)


# -------------------------
# Batch Enrichment
# -------------------------


def enrich_visual_batch(
    visuals: List[Dict[str, Any]],
    client: Optional[VLMClient] = None,
    config: VLMConfig = VLMConfig(),
    validate_tables: bool = True,
) -> List[Optional[VLMEnrichmentResult]]:
    """
    Enrich a batch of visuals using VLM.

    Args:
        visuals: List of dicts with 'image_base64' and optional 'docling_structure'
        client: VLM client (created if not provided)
        config: VLM configuration
        validate_tables: Whether to validate table structures

    Returns:
        List of VLMEnrichmentResult or None (one per visual)
    """
    if client is None:
        client = VLMClient(config)

    results: list[Optional[VLMEnrichmentResult]] = []

    for visual in visuals:
        image_base64 = visual.get("image_base64")

        if not image_base64:
            results.append(None)
            continue

        # Step 1: Classify
        enrichment = classify_visual(image_base64, client)

        if enrichment is None:
            results.append(None)
            continue

        # Step 2: Validate table structure if applicable
        if (
            validate_tables
            and enrichment.classification.visual_type == VisualType.TABLE
            and "docling_structure" in visual
        ):
            validation = validate_table_structure(
                image_base64,
                visual["docling_structure"],
                client,
            )

            # Create new result with validation
            enrichment = VLMEnrichmentResult(
                classification=enrichment.classification,
                parsed_reference=enrichment.parsed_reference,
                extracted_caption=enrichment.extracted_caption,
                table_validation=validation,
                is_continuation=enrichment.is_continuation,
                continuation_of_reference=enrichment.continuation_of_reference,
            )

        results.append(enrichment)

    return results


__all__ = [
    # Types
    "VLMConfig",
    "VLMClient",
    # Main functions
    "classify_visual",
    "validate_table_structure",
    "parse_reference_from_caption",
    "enrich_visual_batch",
    # Parsing
    "parse_json_response",
    "parse_classification_response",
    "parse_reference_response",
    "parse_table_validation_response",
]
