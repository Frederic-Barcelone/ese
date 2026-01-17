# corpus_metadata/C_generators/C10_vision_image_analysis.py
"""
Vision LLM-based image analysis for clinical trial documents.

Extracts structured data from:
- CONSORT/Screening flowcharts (patient flow numbers)
- Kaplan-Meier curves
- Bar/line charts with results
- Study design diagrams
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
    group: Optional[str] = None  # Treatment arm


class ChartData(BaseModel):
    """Structured data from a clinical trial chart."""
    chart_type: str  # "bar", "line", "kaplan_meier", etc.
    title: Optional[str] = None
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    data_points: List[ChartDataPoint] = Field(default_factory=list)
    statistical_results: Dict[str, Any] = Field(default_factory=dict)  # p-values, CIs, etc.


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
    "chart_type": "bar" | "line" | "kaplan_meier" | "forest_plot" | "other",
    "title": "<chart title if visible>",
    "x_axis": "<x-axis label>",
    "y_axis": "<y-axis label>",
    "data_points": [
        {"label": "<timepoint or category>", "value": <number>, "unit": "<unit>", "group": "<treatment arm>"}
    ],
    "statistical_results": {
        "p_value": <if shown>,
        "odds_ratio": <if shown>,
        "confidence_interval": "<if shown>",
        "hazard_ratio": <if shown>
    }
}

IMPORTANT:
- Extract actual numeric values from the chart
- Identify treatment groups/arms
- Capture statistical annotations (p-values, CIs, ORs)
- Note timepoints (months, weeks)

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

        response = self._call_vision_llm(image_base64, prompt)
        if not response:
            return None

        try:
            # Parse exclusion reasons
            exclusions = []
            for exc in response.get("exclusion_reasons", []):
                if isinstance(exc, dict) and exc.get("reason") and exc.get("count"):
                    exclusions.append(ExclusionReason(
                        reason=exc["reason"],
                        count=int(exc["count"]),
                    ))

            # Parse stages
            stages = []
            for stg in response.get("stages", []):
                if isinstance(stg, dict) and stg.get("stage_name") and stg.get("count"):
                    stages.append(PatientFlowStage(
                        stage_name=stg["stage_name"],
                        count=int(stg["count"]),
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
            print(f"[WARN] Failed to parse flowchart response: {e}")
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
            # Parse data points
            data_points = []
            for dp in response.get("data_points", []):
                if isinstance(dp, dict) and dp.get("label") is not None:
                    data_points.append(ChartDataPoint(
                        label=str(dp["label"]),
                        value=float(dp.get("value", 0)),
                        unit=dp.get("unit"),
                        group=dp.get("group"),
                    ))

            return ChartData(
                chart_type=response.get("chart_type", "unknown"),
                title=response.get("title"),
                x_axis=response.get("x_axis"),
                y_axis=response.get("y_axis"),
                data_points=data_points,
                statistical_results=response.get("statistical_results", {}),
            )
        except Exception as e:
            print(f"[WARN] Failed to parse chart response: {e}")
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
                        row_span=int(mc.get("row_span", 1)),
                        col_span=int(mc.get("col_span", 1)),
                        start_row=int(mc.get("start_row", 0)),
                        start_col=int(mc.get("start_col", 0)),
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
            print(f"[WARN] Failed to parse table response: {e}")
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
            print(f"[WARN] Failed to parse glossary table response: {e}")
            return None

    def _call_vision_llm(
        self,
        image_base64: str,
        prompt: str,
    ) -> Optional[Dict[str, Any]]:
        """Call Vision LLM with image and prompt."""
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
            print("[WARN] Vision LLM not available, using OCR text only")
            return None
        except Exception as e:
            print(f"[WARN] Vision LLM call failed: {e}")
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
