"""
Vision LLM-based image analysis for clinical trial figures.

This module uses Claude Vision to extract structured data from clinical trial
figures and diagrams. Analyzes CONSORT flowcharts for patient numbers, interprets
Kaplan-Meier curves, and extracts data from study design diagrams.

Key Components:
    - VisionImageAnalyzer: Main analyzer using Claude Vision API
    - PatientFlowData: Structured patient flow from CONSORT diagrams
    - ExclusionReason: Patient exclusion reasons with counts
    - PatientFlowStage: Individual stages in patient flow

Example:
    >>> from C_generators.C10_vision_image_analysis import VisionImageAnalyzer
    >>> analyzer = VisionImageAnalyzer(config={"model": "claude-sonnet-4-20250514"})
    >>> flow_data = analyzer.analyze_flowchart(image_base64)
    >>> print(f"Screened: {flow_data.screened}, Randomized: {flow_data.randomized}")
    Screened: 500, Randomized: 350

Dependencies:
    - pydantic: Data model definitions
    - anthropic: Claude Vision API client
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================


class PatientFlowStage(BaseModel):
    """A stage in patient flow (screening, randomization, etc.)"""
    stage_name: str  # e.g., "Screened", "Randomized", "Completed"
    count: int  # Number of patients
    details: Optional[str] = None  # Additional context


class ExclusionReason(BaseModel):
    """Reason for patient exclusion with count."""
    reason: str
    count: int


class PatientFlowData(BaseModel):
    """Structured patient flow from CONSORT/screening diagram."""

    # Main flow stages
    screened: Optional[int] = None
    screen_failures: Optional[int] = None
    randomized: Optional[int] = None
    completed: Optional[int] = None
    discontinued: Optional[int] = None

    # Treatment arms
    arms: Dict[str, int] = Field(default_factory=dict)  # {"Iptacopan": 38, "Placebo": 36}

    # Exclusion reasons
    exclusion_reasons: List[ExclusionReason] = Field(default_factory=list)

    # Additional stages
    stages: List[PatientFlowStage] = Field(default_factory=list)

    # Raw extraction notes
    notes: Optional[str] = None


class ChartDataPoint(BaseModel):
    """Data point from a chart."""
    label: str
    value: float
    unit: Optional[str] = None
    group: Optional[str] = None  # Treatment arm or legend label (not color)


class LegendEntry(BaseModel):
    """A legend entry from a chart."""
    color: Optional[str] = None  # "red", "black", "blue", etc.
    marker: Optional[str] = None  # "circle", "square", "triangle", etc.
    label: str  # The actual legend text (e.g., "Protocol target")
    series_id: Optional[str] = None  # Internal ID for binding to data points


class TaperScheduleInfo(BaseModel):
    """Taper schedule information extracted from a dose chart."""
    drug_name: Optional[str] = None
    starting_dose: Optional[str] = None
    target_dose: Optional[str] = None
    target_timepoint: Optional[str] = None


class ChartData(BaseModel):
    """Structured data from a clinical trial chart."""
    chart_type: str  # "bar", "line", "kaplan_meier", "taper_schedule", etc.
    title: Optional[str] = None
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    legend_entries: List[LegendEntry] = Field(default_factory=list)  # Chart legend
    data_points: List[ChartDataPoint] = Field(default_factory=list)
    statistical_results: Dict[str, Any] = Field(default_factory=dict)  # p-values, CIs, etc.
    taper_schedule: Optional[TaperScheduleInfo] = None  # For dose taper charts


# =============================================================================
# PROMPTS
# =============================================================================

FLOWCHART_PROMPT = """Analyze this clinical trial screening/CONSORT flowchart image.

Extract ALL patient flow numbers into this JSON structure:
{
    "screened": <number of patients screened>,
    "screen_failures": <number who failed screening>,
    "randomized": <number randomized>,
    "completed": <number who completed>,
    "discontinued": <number who discontinued>,
    "arms": {
        "<arm1_name>": <count>,
        "<arm2_name>": <count>
    },
    "exclusion_reasons": [
        {"reason": "<reason text>", "count": <number>}
    ],
    "stages": [
        {"stage_name": "<stage>", "count": <number>, "details": "<optional details>"}
    ],
    "notes": "<any other relevant information>"
}

IMPORTANT:
- Extract EXACT numbers from the diagram
- Include ALL exclusion/dropout reasons with their counts
- Capture treatment arm names and sizes
- Include run-in period dropouts if shown

Return JSON only."""

CHART_PROMPT = """Analyze this clinical trial results chart image.

Extract data into this JSON structure:
{
    "chart_type": "bar" | "line" | "kaplan_meier" | "forest_plot" | "taper_schedule" | "other",
    "title": "<chart title if visible>",
    "x_axis": "<x-axis label>",
    "y_axis": "<y-axis label>",
    "legend_entries": [
        {"color": "red", "marker": "circle", "label": "<legend text>", "series_id": "series_1"},
        {"color": "black", "marker": "square", "label": "<legend text>", "series_id": "series_2"}
    ],
    "data_points": [
        {"label": "<timepoint or category>", "value": <number>, "unit": "<unit>", "group": "<series_id or legend label>"}
    ],
    "statistical_results": {
        "p_value": <if shown>,
        "odds_ratio": <if shown>,
        "confidence_interval": "<if shown>",
        "hazard_ratio": <if shown>
    },
    "taper_schedule": {
        "drug_name": "<if this is a dose taper chart>",
        "starting_dose": "<initial dose>",
        "target_dose": "<final target dose>",
        "target_timepoint": "<when target should be reached>"
    }
}

IMPORTANT:
- Extract legend entries with their colors/markers and bind to data series
- For each data_point, use the legend label (not color) as the "group" value
- Extract actual numeric values from the chart
- Identify treatment groups/arms by their legend labels
- Capture statistical annotations (p-values, CIs, ORs)
- Note timepoints (months, weeks)
- For taper/dose charts: extract drug name, starting dose, and target dose with timepoint

Return JSON only."""

TABLE_PROMPT = """Analyze this clinical trial table image.

Extract the complete table structure into this JSON format:
{
    "title": "<table title/caption if visible>",
    "headers": ["Column 1", "Column 2", "Column 3", ...],
    "rows": [
        ["cell1", "cell2", "cell3", ...],
        ["cell1", "cell2", "cell3", ...]
    ],
    "merged_cells": [
        {"text": "<merged cell content>", "row_span": 2, "col_span": 1, "start_row": 0, "start_col": 0}
    ],
    "table_type": "data" | "glossary" | "demographics" | "results" | "endpoints" | "adverse_events",
    "notes": "<any footnotes or additional context>"
}

IMPORTANT:
- Extract ALL rows and columns - do not truncate
- Preserve exact cell values (numbers, text, symbols)
- Note merged/spanning cells in the merged_cells array
- For multi-level headers, include all header rows in 'headers' as nested arrays
- Identify table type based on content:
  - "glossary": abbreviation/definition tables
  - "demographics": baseline characteristics
  - "results": primary/secondary endpoint results
  - "endpoints": endpoint definitions
  - "adverse_events": safety data
  - "data": general data tables

Return JSON only."""

GLOSSARY_TABLE_PROMPT = """Extract abbreviations and definitions from this glossary/abbreviation table.

Return JSON format:
{
    "abbreviations": [
        {"short_form": "AE", "long_form": "Adverse Event"},
        {"short_form": "CI", "long_form": "Confidence Interval"}
    ],
    "table_title": "<title if visible>",
    "notes": "<any footnotes>"
}

IMPORTANT:
- Extract ALL abbreviation-definition pairs
- Short form is typically in the first column
- Long form/definition is in the second column
- Some tables have multiple columns of pairs - extract all

Return JSON only."""

IMAGE_CLASSIFICATION_PROMPT = """Classify this clinical/scientific image and determine the best analysis approach.

Return JSON format:
{
    "image_type": "flowchart" | "chart" | "table" | "diagram" | "photo" | "logo" | "text_page" | "other",
    "confidence": <0.0 to 1.0>,
    "description": "<brief description of what the image shows>",
    "contains_data": true | false,
    "analysis_recommendation": "flowchart_analysis" | "chart_analysis" | "table_extraction" | "skip"
}

Classification guide:
- "flowchart": CONSORT diagrams, treatment algorithms, decision trees with boxes/diamonds/arrows
- "chart": Bar charts, line graphs, Kaplan-Meier curves, forest plots, scatter plots
- "table": Tabular data (even if embedded as image)
- "diagram": Study design diagrams, molecular structures, anatomical illustrations
- "photo": Clinical photos, equipment images
- "logo": Company logos, journal logos, small decorative images
- "text_page": Pages that are primarily text (should not have been extracted as figure)
- "other": Anything that doesn't fit above categories

Return JSON only."""


# =============================================================================
# DATA MODELS FOR TABLES
# =============================================================================


class TableCellData(BaseModel):
    """A cell in a table with potential spanning."""
    text: str
    row_span: int = 1
    col_span: int = 1
    start_row: int = 0
    start_col: int = 0


class VisionTableData(BaseModel):
    """Structured table data extracted by vision LLM."""
    title: Optional[str] = None
    headers: List[Any] = Field(default_factory=list)  # Can be nested for multi-level
    rows: List[List[str]] = Field(default_factory=list)
    merged_cells: List[TableCellData] = Field(default_factory=list)
    table_type: str = "data"
    notes: Optional[str] = None


class GlossaryEntry(BaseModel):
    """An abbreviation-definition pair from a glossary table."""
    short_form: str
    long_form: str


class GlossaryTableData(BaseModel):
    """Extracted glossary/abbreviation table."""
    abbreviations: List[GlossaryEntry] = Field(default_factory=list)
    table_title: Optional[str] = None
    notes: Optional[str] = None


# =============================================================================
# VISION IMAGE ANALYZER
# =============================================================================


class VisionImageAnalyzer:
    """
    Analyze clinical trial images using Vision LLM.

    Extracts structured data from flowcharts, charts, and diagrams.
    """

    def __init__(
        self,
        llm_client: Any,
        llm_model: str = "claude-sonnet-4-20250514",
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Safely convert value to int, handling None and invalid types."""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert value to float, handling None and invalid types."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def analyze_flowchart(
        self,
        image_base64: str,
        ocr_text: Optional[str] = None,
    ) -> Optional[PatientFlowData]:
        """
        Analyze a CONSORT/screening flowchart image.

        Args:
            image_base64: Base64-encoded image
            ocr_text: Optional OCR text already extracted (provides hints)

        Returns:
            PatientFlowData with structured patient flow numbers
        """
        if not image_base64:
            return None

        # Build prompt with OCR context if available
        prompt = FLOWCHART_PROMPT
        if ocr_text:
            prompt += f"\n\nOCR text from image (for reference):\n{ocr_text[:1000]}"

        response = self._call_vision_llm(image_base64, prompt, ocr_fallback_text=ocr_text)
        if not response:
            return None

        try:
            # Parse exclusion reasons
            exclusions = []
            for exc in response.get("exclusion_reasons", []):
                if isinstance(exc, dict) and exc.get("reason"):
                    count = self._safe_int(exc.get("count"), 0)
                    if count > 0:
                        exclusions.append(ExclusionReason(
                            reason=exc["reason"],
                            count=count,
                        ))

            # Parse stages
            stages = []
            for stg in response.get("stages", []):
                if isinstance(stg, dict) and stg.get("stage_name"):
                    count = self._safe_int(stg.get("count"), 0)
                    if count > 0:
                        stages.append(PatientFlowStage(
                            stage_name=stg["stage_name"],
                            count=count,
                            details=stg.get("details"),
                        ))

            return PatientFlowData(
                screened=response.get("screened"),
                screen_failures=response.get("screen_failures"),
                randomized=response.get("randomized"),
                completed=response.get("completed"),
                discontinued=response.get("discontinued"),
                arms=response.get("arms", {}),
                exclusion_reasons=exclusions,
                stages=stages,
                notes=response.get("notes"),
            )
        except Exception as e:
            logger.warning("Failed to parse flowchart response: %s", e)
            return None

    def analyze_chart(
        self,
        image_base64: str,
        caption: Optional[str] = None,
    ) -> Optional[ChartData]:
        """
        Analyze a clinical trial results chart.

        Args:
            image_base64: Base64-encoded image
            caption: Optional figure caption (provides context)

        Returns:
            ChartData with extracted values and statistics
        """
        if not image_base64:
            return None

        prompt = CHART_PROMPT
        if caption:
            prompt += f"\n\nFigure caption: {caption}"

        response = self._call_vision_llm(image_base64, prompt)
        if not response:
            return None

        try:
            # Parse legend entries
            legend_entries = []
            for le in response.get("legend_entries", []):
                if isinstance(le, dict) and le.get("label"):
                    legend_entries.append(LegendEntry(
                        color=le.get("color"),
                        marker=le.get("marker"),
                        label=le["label"],
                        series_id=le.get("series_id"),
                    ))

            # Build legend label lookup for data point binding
            # Maps series_id/color to legend label
            legend_lookup: Dict[str, str] = {}
            for le in legend_entries:
                if le.series_id:
                    legend_lookup[le.series_id] = le.label
                if le.color:
                    # Also map color-based groups (e.g., "red_circles" -> "Protocol target")
                    legend_lookup[le.color] = le.label
                    if le.marker:
                        legend_lookup[f"{le.color}_{le.marker}s"] = le.label

            # Parse data points and bind to legend labels
            data_points = []
            for dp in response.get("data_points", []):
                if isinstance(dp, dict) and dp.get("label") is not None:
                    group = dp.get("group")
                    # Try to resolve group to legend label
                    if group and group in legend_lookup:
                        group = legend_lookup[group]
                    data_points.append(ChartDataPoint(
                        label=str(dp["label"]),
                        value=self._safe_float(dp.get("value"), 0.0),
                        unit=dp.get("unit"),
                        group=group,
                    ))

            # Parse taper schedule info if present
            taper_info = None
            taper_data = response.get("taper_schedule")
            if isinstance(taper_data, dict) and any(taper_data.values()):
                taper_info = TaperScheduleInfo(
                    drug_name=taper_data.get("drug_name"),
                    starting_dose=taper_data.get("starting_dose"),
                    target_dose=taper_data.get("target_dose"),
                    target_timepoint=taper_data.get("target_timepoint"),
                )

            return ChartData(
                chart_type=response.get("chart_type", "unknown"),
                title=response.get("title"),
                x_axis=response.get("x_axis"),
                y_axis=response.get("y_axis"),
                legend_entries=legend_entries,
                data_points=data_points,
                statistical_results=response.get("statistical_results", {}),
                taper_schedule=taper_info,
            )
        except Exception as e:
            logger.warning("Failed to parse chart response: %s", e)
            return None

    def analyze_table(
        self,
        image_base64: str,
        caption: Optional[str] = None,
        is_glossary: bool = False,
    ) -> Optional[VisionTableData]:
        """
        Analyze a table image and extract structured data.

        Args:
            image_base64: Base64-encoded table image
            caption: Optional table caption
            is_glossary: If True, use glossary-specific extraction

        Returns:
            VisionTableData with headers, rows, and metadata
        """
        if not image_base64:
            return None

        prompt = TABLE_PROMPT
        if caption:
            prompt += f"\n\nTable caption: {caption}"

        response = self._call_vision_llm(image_base64, prompt)
        if not response:
            return None

        try:
            # Parse merged cells
            merged_cells = []
            for mc in response.get("merged_cells", []):
                if isinstance(mc, dict) and mc.get("text"):
                    merged_cells.append(TableCellData(
                        text=mc["text"],
                        row_span=self._safe_int(mc.get("row_span"), 1),
                        col_span=self._safe_int(mc.get("col_span"), 1),
                        start_row=self._safe_int(mc.get("start_row"), 0),
                        start_col=self._safe_int(mc.get("start_col"), 0),
                    ))

            return VisionTableData(
                title=response.get("title"),
                headers=response.get("headers", []),
                rows=response.get("rows", []),
                merged_cells=merged_cells,
                table_type=response.get("table_type", "data"),
                notes=response.get("notes"),
            )
        except Exception as e:
            logger.warning("Failed to parse table response: %s", e)
            return None

    def analyze_glossary_table(
        self,
        image_base64: str,
        caption: Optional[str] = None,
    ) -> Optional[GlossaryTableData]:
        """
        Extract abbreviations from a glossary/abbreviation table image.

        Args:
            image_base64: Base64-encoded table image
            caption: Optional table caption

        Returns:
            GlossaryTableData with abbreviation-definition pairs
        """
        if not image_base64:
            return None

        prompt = GLOSSARY_TABLE_PROMPT
        if caption:
            prompt += f"\n\nTable caption: {caption}"

        response = self._call_vision_llm(image_base64, prompt)
        if not response:
            return None

        try:
            # Parse abbreviation entries
            entries = []
            for entry in response.get("abbreviations", []):
                if isinstance(entry, dict):
                    sf = entry.get("short_form", "").strip()
                    lf = entry.get("long_form", "").strip()
                    if sf and lf:
                        entries.append(GlossaryEntry(
                            short_form=sf,
                            long_form=lf,
                        ))

            return GlossaryTableData(
                abbreviations=entries,
                table_title=response.get("table_title"),
                notes=response.get("notes"),
            )
        except Exception as e:
            logger.warning("Failed to parse glossary table response: %s", e)
            return None

    def classify_image(
        self,
        image_base64: str,
        ocr_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Classify an unknown image to determine the best analysis approach.

        This is used for images that weren't classified during extraction
        (orphan images, vector figures without captions).

        Args:
            image_base64: Base64-encoded image
            ocr_text: Optional OCR text for context

        Returns:
            Dict with image_type, confidence, description, analysis_recommendation
        """
        default_result = {
            "image_type": "other",
            "confidence": 0.0,
            "description": "Classification failed",
            "contains_data": False,
            "analysis_recommendation": "skip",
        }

        if not image_base64:
            return default_result

        prompt = IMAGE_CLASSIFICATION_PROMPT
        if ocr_text:
            prompt += f"\n\nOCR text from image (for reference):\n{ocr_text[:500]}"

        response = self._call_vision_llm(image_base64, prompt)
        if not response:
            return default_result

        try:
            return {
                "image_type": response.get("image_type", "other"),
                "confidence": self._safe_float(response.get("confidence"), 0.5),
                "description": response.get("description", ""),
                "contains_data": response.get("contains_data", False),
                "analysis_recommendation": response.get("analysis_recommendation", "skip"),
            }
        except Exception as e:
            logger.warning("Failed to parse classification response: %s", e)
            return default_result

    def analyze_unknown_image(
        self,
        image_base64: str,
        ocr_text: Optional[str] = None,
        caption: Optional[str] = None,
    ) -> Tuple[str, Optional[Any]]:
        """
        Classify and analyze an unknown image using appropriate method.

        This is the main entry point for handling UNKNOWN type images.

        Args:
            image_base64: Base64-encoded image
            ocr_text: Optional OCR text
            caption: Optional caption

        Returns:
            Tuple of (detected_type, analysis_result)
            analysis_result may be PatientFlowData, ChartData, etc.
        """
        if not image_base64:
            return ("error", None)

        # First classify the image
        classification = self.classify_image(image_base64, ocr_text)
        image_type = classification.get("image_type", "other")
        recommendation = classification.get("analysis_recommendation", "skip")

        # Run appropriate analysis based on classification
        if recommendation == "flowchart_analysis" or image_type == "flowchart":
            result = self.analyze_flowchart(image_base64, ocr_text)
            return ("flowchart", result)

        elif recommendation == "chart_analysis" or image_type == "chart":
            result = self.analyze_chart(image_base64, caption)
            return ("chart", result)

        elif recommendation == "table_extraction" or image_type == "table":
            result = self.analyze_table(image_base64, caption)
            return ("table", result)

        elif image_type in ("text_page", "logo"):
            # Skip these - they're false positive figure extractions
            return (image_type, None)

        else:
            # For other types, try chart analysis as a general fallback
            # (it can extract basic data from various visual formats)
            if classification.get("contains_data", False):
                result = self.analyze_chart(image_base64, caption)
                return ("chart", result)
            return (image_type, None)

    def _call_vision_llm(
        self,
        image_base64: str,
        prompt: str,
        ocr_fallback_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Call Vision LLM with image and prompt.

        Args:
            image_base64: Base64-encoded image
            prompt: Analysis prompt
            ocr_fallback_text: Optional OCR text to use if Vision fails

        Returns:
            Parsed response dict or None if failed
        """
        if not self.llm_client:
            return None

        try:
            # Use Claude's vision capability
            # The image should be sent as a base64-encoded media block
            response = self.llm_client.complete_vision_json(
                image_base64=image_base64,
                prompt=prompt,
                model=self.llm_model,
                max_tokens=2000,
            )

            return response if isinstance(response, dict) else None
        except AttributeError:
            # Fallback: try standard completion with image description
            logger.warning("Vision LLM not available, using OCR text only")
            if ocr_fallback_text:
                return self._extract_from_ocr_text(ocr_fallback_text, prompt)
            return None
        except Exception as e:
            error_msg = str(e)
            # Check if it's a size limit error
            if "5 MB" in error_msg or "size" in error_msg.lower() or "exceeds" in error_msg.lower():
                logger.warning("Vision LLM image too large: %s", e)
            else:
                logger.warning("Vision LLM call failed: %s", e)

            # Try OCR fallback if available
            if ocr_fallback_text:
                logger.info("Attempting OCR text fallback...")
                return self._extract_from_ocr_text(ocr_fallback_text, prompt)
            return None

    def _extract_from_ocr_text(
        self,
        ocr_text: str,
        prompt: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract structured data from OCR text using text-only LLM.

        This is a fallback when Vision LLM fails (e.g., image too large).
        Uses Claude's text completion to analyze extracted text.

        Args:
            ocr_text: OCR text extracted from the figure
            prompt: Original prompt (used to determine extraction type)

        Returns:
            Parsed response dict or None if failed
        """
        if not self.llm_client or not ocr_text:
            return None

        # Determine extraction type from prompt
        extraction_type = "generic"
        if "flowchart" in prompt.lower() or "screening" in prompt.lower():
            extraction_type = "flowchart"
        elif "chart" in prompt.lower() or "kaplan" in prompt.lower():
            extraction_type = "chart"
        elif "table" in prompt.lower():
            extraction_type = "table"

        # Build text-only prompt
        text_prompt = f"""Analyze the following text extracted from a clinical trial {extraction_type} image.
Extract any structured data you can find.

Text from image:
{ocr_text[:3000]}

{prompt}

Note: This is OCR-extracted text, so there may be some extraction errors."""

        try:
            if hasattr(self.llm_client, "complete_json_any"):
                response = self.llm_client.complete_json_any(
                    system_prompt="You are analyzing text extracted from a clinical trial figure.",
                    user_prompt=text_prompt,
                    model=self.llm_model,
                    max_tokens=2000,
                    temperature=0.0,
                )
            else:
                return None

            result = response if isinstance(response, dict) else None
            if result:
                result["_extraction_method"] = "ocr_fallback"
            return result
        except Exception as e:
            logger.warning("OCR text fallback failed: %s", e)
            return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def extract_patient_flow_from_image(
    image_base64: str,
    ocr_text: Optional[str] = None,
    llm_client: Any = None,
) -> Optional[PatientFlowData]:
    """
    Extract patient flow data from a screening flowchart image.

    Args:
        image_base64: Base64-encoded flowchart image
        ocr_text: Optional OCR text from image
        llm_client: Claude client instance

    Returns:
        PatientFlowData with structured patient numbers
    """
    if not llm_client:
        return None

    analyzer = VisionImageAnalyzer(llm_client)
    return analyzer.analyze_flowchart(image_base64, ocr_text)


def extract_chart_data_from_image(
    image_base64: str,
    caption: Optional[str] = None,
    llm_client: Any = None,
) -> Optional[ChartData]:
    """
    Extract data from a clinical trial chart image.

    Args:
        image_base64: Base64-encoded chart image
        caption: Optional figure caption
        llm_client: Claude client instance

    Returns:
        ChartData with extracted values
    """
    if not llm_client:
        return None

    analyzer = VisionImageAnalyzer(llm_client)
    return analyzer.analyze_chart(image_base64, caption)
