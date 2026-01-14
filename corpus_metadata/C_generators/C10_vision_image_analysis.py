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

import base64
import json
import re
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
