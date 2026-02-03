"""
VLM-based table structure extraction using Claude Vision.

This module extracts table structures directly from images using Claude Vision,
replacing HTML parsing from Unstructured with more accurate VLM extraction.
Handles complex table layouts with superior numeric precision.

Key Components:
    - VLMTableExtractor: Main extractor using Claude Vision API
    - resize_image_for_vlm: Image preprocessing for API limits
    - Extraction benefits:
        - Numeric precision (26.1 not 26-1)
        - Complex table layouts with merged cells
        - Multi-row headers and spanning columns

Example:
    >>> from C_generators.C15_vlm_table_extractor import VLMTableExtractor
    >>> extractor = VLMTableExtractor(config={"model": "claude-sonnet-4-20250514"})
    >>> table_data = extractor.extract(image_base64, table_context)
    >>> for row in table_data.rows:
    ...     print(row)
    ['Patient ID', 'Age', 'Dose (mg)']
    ['001', '45', '26.1']

Dependencies:
    - pydantic: Data model definitions
    - anthropic: Claude Vision API client
    - PIL: Image resizing (optional)
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Claude Vision API limits
MAX_IMAGE_DIMENSION = 8000  # Max pixels in any dimension
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5MB


def resize_image_for_vlm(image_base64: str) -> Tuple[str, bool]:
    """
    Resize image if it exceeds Claude Vision API limits.

    Args:
        image_base64: Base64-encoded PNG image

    Returns:
        Tuple of (resized_base64, was_resized)
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("PIL not available for image resizing")
        return image_base64, False

    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_base64)
        original_size = len(image_bytes)

        # Open image
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        was_resized = False

        # Check if resizing needed for dimensions
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            # Calculate scale factor to fit within limits
            scale = min(MAX_IMAGE_DIMENSION / width, MAX_IMAGE_DIMENSION / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)  # type: ignore[assignment]
            was_resized = True
            logger.info(
                "Resized image from %dx%d to %dx%d (dimension limit)",
                width, height, new_width, new_height
            )

        # Convert back to bytes and check size
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        new_bytes = buffer.getvalue()

        # If still too large, reduce quality/size iteratively
        quality_scale = 0.9
        while len(new_bytes) > MAX_IMAGE_BYTES and quality_scale > 0.3:
            new_width = int(img.size[0] * quality_scale)
            new_height = int(img.size[1] * quality_scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)  # type: ignore[assignment]
            buffer = io.BytesIO()
            img.save(buffer, format="PNG", optimize=True)
            new_bytes = buffer.getvalue()
            quality_scale -= 0.1
            was_resized = True
            logger.info(
                "Reduced image to %dx%d (size: %.1fMB)",
                new_width, new_height, len(new_bytes) / (1024 * 1024)
            )

        if was_resized:
            logger.info(
                "Image resized: %.1fMB -> %.1fMB",
                original_size / (1024 * 1024), len(new_bytes) / (1024 * 1024)
            )

        return base64.b64encode(new_bytes).decode("utf-8"), was_resized

    except Exception as e:
        logger.warning("Image resize failed: %s", e)
        return image_base64, False


# =============================================================================
# DATA MODELS
# =============================================================================


class VLMTableResponse(BaseModel):
    """Response from VLM table extraction."""

    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    row_count: int = 0
    col_count: int = 0
    notes: Optional[str] = None

    # Verification fields
    confidence: float = 0.95
    verification_warning: Optional[str] = None


# =============================================================================
# VLM TABLE EXTRACTOR
# =============================================================================


class VLMTableExtractor:
    """
    Extract table structure using Vision LLM.

    Uses Claude Vision to read table images and extract structured data,
    replacing OCR-based extraction from Unstructured which often has errors.
    """

    VLM_PROMPT = """You see an image of a table from a scientific PDF.

Extract ALL visible text and output a machine-readable table.

Return JSON:
{
    "headers": ["Column 1", "Column 2", ...],
    "rows": [
        ["cell1", "cell2", ...],
        ["cell3", "cell4", ...]
    ],
    "row_count": <integer - number of data rows, not including header>,
    "col_count": <integer - number of columns>,
    "notes": "any footnotes or captions visible below the table"
}

CRITICAL RULES:
- Extract EXACTLY what you see - no summarization
- Preserve numeric precision (26.1, not 26-1 or ~26)
- Multi-row headers: concatenate with " | " (e.g. "Treatment | n=38")
- Empty cells: use empty string ""
- Do NOT add or remove rows/columns
- Do NOT include the header row in the "rows" array
- Include ALL data rows, even if the table is long
- For merged cells spanning multiple rows, repeat the value in each row
- Preserve special characters and symbols exactly as shown

Return valid JSON only, no additional text."""

    def __init__(
        self,
        llm_client: Any,
        llm_model: str = "claude-sonnet-4-20250514",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize VLM table extractor.

        Args:
            llm_client: LLM client with complete_vision_json method
            llm_model: Model to use for vision tasks
            config: Optional configuration dict
        """
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.config = config or {}

    def extract(self, image_base64: str) -> Dict[str, Any]:
        """
        Send table image to VLM and parse response.

        Args:
            image_base64: Base64-encoded table image (PNG)

        Returns:
            Dict with headers, rows, row_count, col_count, notes, confidence
        """
        if not image_base64:
            return self._empty_response("No image provided")

        if not self.llm_client:
            return self._empty_response("No LLM client available")

        try:
            # Resize image if needed to fit Claude Vision limits
            resized_image, was_resized = resize_image_for_vlm(image_base64)

            # Call Vision LLM
            response = self.llm_client.complete_vision_json(
                image_base64=resized_image,
                prompt=self.VLM_PROMPT,
                model=self.llm_model,
                max_tokens=4000,  # Tables can be large
                temperature=0.0,  # Deterministic extraction
            )

            if not response:
                return self._empty_response("VLM returned empty response")

            # Ensure response is a dict (not a list or other type)
            if not isinstance(response, dict):
                logger.warning("VLM returned non-dict response: %s", type(response).__name__)
                return self._empty_response(f"VLM returned {type(response).__name__} instead of dict")

            # Parse and verify response
            return self._verify_extraction(response)

        except AttributeError as e:
            logger.warning("VLM client missing vision method: %s", e)
            return self._empty_response(f"VLM not available: {e}")
        except Exception as e:
            logger.warning("VLM table extraction failed: %s", e)
            return self._empty_response(f"Extraction failed: {e}")

    def _verify_extraction(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify VLM extraction for consistency and add confidence score.

        Args:
            response: Raw VLM response dict

        Returns:
            Verified response with confidence score
        """
        headers = response.get("headers", [])
        rows = response.get("rows", [])
        claimed_rows = response.get("row_count", 0)
        claimed_cols = response.get("col_count", 0)
        notes = response.get("notes")

        actual_rows = len(rows)
        actual_cols = len(headers) if headers else (len(rows[0]) if rows else 0)

        result = {
            "headers": headers,
            "rows": rows,
            "row_count": actual_rows,
            "col_count": actual_cols,
            "notes": notes,
            "confidence": 0.95,
            "verification_warning": None,
        }

        # Check row count consistency
        if claimed_rows != actual_rows and claimed_rows > 0:
            result["verification_warning"] = (
                f"Row count mismatch: claimed {claimed_rows}, actual {actual_rows}"
            )
            result["confidence"] = 0.7

        # Check column count consistency
        if claimed_cols != actual_cols and claimed_cols > 0:
            warning = f"Col count mismatch: claimed {claimed_cols}, actual {actual_cols}"
            if result["verification_warning"]:
                result["verification_warning"] += f"; {warning}"
            else:
                result["verification_warning"] = warning
            result["confidence"] = min(result["confidence"], 0.7)

        # Check row consistency (all rows should have same column count)
        if rows:
            row_lengths = [len(row) for row in rows]
            if len(set(row_lengths)) > 1:
                warning = f"Inconsistent row lengths: {row_lengths}"
                if result["verification_warning"]:
                    result["verification_warning"] += f"; {warning}"
                else:
                    result["verification_warning"] = warning
                result["confidence"] = min(result["confidence"], 0.6)

        # Check header/row column alignment
        if headers and rows:
            if len(headers) != actual_cols:
                warning = f"Header/row column mismatch: {len(headers)} vs {actual_cols}"
                if result["verification_warning"]:
                    result["verification_warning"] += f"; {warning}"
                else:
                    result["verification_warning"] = warning
                result["confidence"] = min(result["confidence"], 0.5)

        return result

    def _empty_response(self, reason: str) -> Dict[str, Any]:
        """Return empty response with error reason."""
        return {
            "headers": [],
            "rows": [],
            "row_count": 0,
            "col_count": 0,
            "notes": None,
            "confidence": 0.0,
            "verification_warning": reason,
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def extract_table_with_vlm(
    image_base64: str,
    llm_client: Any,
    llm_model: str = "claude-sonnet-4-20250514",
) -> Dict[str, Any]:
    """
    Extract table structure from image using VLM.

    Args:
        image_base64: Base64-encoded table image
        llm_client: LLM client with complete_vision_json method
        llm_model: Model to use

    Returns:
        Dict with headers, rows, and metadata
    """
    extractor = VLMTableExtractor(llm_client, llm_model)
    return extractor.extract(image_base64)
