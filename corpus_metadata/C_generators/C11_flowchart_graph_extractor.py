# corpus_metadata/C_generators/C11_flowchart_graph_extractor.py
"""
Flowchart Graph Extractor - Extract treatment algorithm decision logic.

Extracts structured care pathway graphs from clinical algorithm figures,
capturing nodes (actions/decisions), edges (transitions), and conditions.

This goes beyond simple flowchart analysis (patient counts) to extract
the actual decision logic that can be used for clinical decision support.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from A_core.A17_care_pathway_models import (
    CarePathway,
    CarePathwayNode,
    CarePathwayEdge,
    NodeType,
    TaperSchedule,
    TaperSchedulePoint,
)


# =============================================================================
# PROMPTS FOR CARE PATHWAY EXTRACTION
# =============================================================================

CARE_PATHWAY_PROMPT = """Analyze this clinical treatment algorithm/flowchart image.

Extract the COMPLETE decision graph into this JSON structure:
{
    "title": "<algorithm title>",
    "condition": "<target condition, e.g., 'Active GPA/MPA'>",
    "phases": ["<phase 1>", "<phase 2>"],
    "nodes": [
        {
            "id": "n1",
            "type": "start" | "end" | "action" | "decision" | "assessment" | "note",
            "text": "<exact text from box/diamond>",
            "phase": "<which phase this belongs to>",
            "drugs": ["<drug1>", "<drug2>"],
            "dose": "<dosing info if present>",
            "duration": "<duration if specified>"
        }
    ],
    "edges": [
        {
            "source_id": "n1",
            "target_id": "n2",
            "condition": "<edge label like 'Yes', 'No', 'organ/life-threatening'>",
            "condition_type": "positive" | "negative" | "severity" | "other"
        }
    ],
    "entry_criteria": "<who this pathway applies to>",
    "primary_drugs": ["<main drugs>"],
    "alternative_drugs": ["<alternative options>"],
    "decision_points": ["<key decision 1>", "<key decision 2>"],
    "target_dose": "<target dose if specified, e.g., '5 mg/day'>",
    "target_timepoint": "<when target should be reached>",
    "maintenance_duration": "<how long maintenance therapy>",
    "relapse_handling": "<how relapse is managed>"
}

NODE TYPES:
- "start": Entry point to the pathway
- "end": Exit/completion point
- "action": Treatment action (prescribe, administer, perform)
- "decision": Decision point (diamond shape, has Yes/No branches)
- "assessment": Clinical assessment/evaluation
- "note": Informational note or annotation

IMPORTANT:
- Create a UNIQUE id for each node (n1, n2, n3, etc.)
- Extract ALL boxes/diamonds as nodes
- Extract ALL arrows/connections as edges
- Capture edge conditions (Yes/No, severity levels)
- Identify drug names in action nodes
- Extract dosing and duration information
- Identify which phase each node belongs to

Return JSON only."""


# =============================================================================
# FLOWCHART GRAPH EXTRACTOR CLASS
# =============================================================================


class FlowchartGraphExtractor:
    """
    Extracts structured care pathway graphs from clinical algorithm figures.

    Uses Vision LLM to analyze flowchart images and extract:
    - Node structure (actions, decisions, assessments)
    - Edge connections with conditions
    - Clinical parameters (drugs, doses, durations)
    """

    def __init__(
        self,
        llm_client: Any,
        llm_model: str = "claude-sonnet-4-20250514",
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model

        # Drug name normalization patterns
        self.drug_patterns = {
            r"\bRTX\b": "rituximab",
            r"\bCYC\b": "cyclophosphamide",
            r"\bGC[s]?\b": "glucocorticoid",
            r"\bMTX\b": "methotrexate",
            r"\bMMF\b": "mycophenolate mofetil",
            r"\bAZA\b": "azathioprine",
            r"\bavacopan\b": "avacopan",
            r"\bmepolizumab\b": "mepolizumab",
            r"\bbenralizumab\b": "benralizumab",
        }

    def extract_care_pathway(
        self,
        image_base64: str,
        ocr_text: Optional[str] = None,
        caption: Optional[str] = None,
        figure_id: Optional[str] = None,
        page_num: Optional[int] = None,
    ) -> Optional[CarePathway]:
        """
        Extract a care pathway graph from a flowchart image.

        Args:
            image_base64: Base64-encoded flowchart image
            ocr_text: Optional OCR text from the image
            caption: Optional figure caption
            figure_id: Figure identifier (e.g., "Figure 1")
            page_num: Page number in source document

        Returns:
            CarePathway object with full graph structure
        """
        if not image_base64:
            return None

        # Build prompt with context
        prompt = CARE_PATHWAY_PROMPT
        if caption:
            prompt += f"\n\nFigure caption: {caption}"
        if ocr_text:
            prompt += f"\n\nOCR text from image (for reference):\n{ocr_text[:1500]}"

        # Call Vision LLM
        response = self._call_vision_llm(image_base64, prompt)
        if not response:
            return None

        try:
            return self._parse_response(response, figure_id, page_num)
        except Exception as e:
            print(f"[WARN] Failed to parse care pathway response: {e}")
            return None

    def _parse_response(
        self,
        response: Dict[str, Any],
        figure_id: Optional[str],
        page_num: Optional[int],
    ) -> CarePathway:
        """Parse the LLM response into a CarePathway object."""
        # Parse nodes
        nodes = []
        for node_data in response.get("nodes", []):
            if not isinstance(node_data, dict):
                continue

            node_id = node_data.get("id", f"n{len(nodes)+1}")
            node_type_str = node_data.get("type", "action")

            # Map type string to enum
            try:
                node_type = NodeType(node_type_str)
            except ValueError:
                node_type = NodeType.ACTION

            # Extract and normalize drugs
            drugs = node_data.get("drugs", [])
            if not drugs:
                # Try to extract drugs from text
                drugs = self._extract_drugs_from_text(node_data.get("text", ""))

            nodes.append(CarePathwayNode(
                id=node_id,
                type=node_type,
                text=node_data.get("text", ""),
                phase=node_data.get("phase"),
                drugs=drugs,
                dose=node_data.get("dose"),
                duration=node_data.get("duration"),
            ))

        # Parse edges
        edges = []
        for edge_data in response.get("edges", []):
            if not isinstance(edge_data, dict):
                continue

            edges.append(CarePathwayEdge(
                source_id=edge_data.get("source_id", ""),
                target_id=edge_data.get("target_id", ""),
                condition=edge_data.get("condition"),
                condition_type=edge_data.get("condition_type"),
            ))

        # Build pathway
        pathway = CarePathway(
            pathway_id=figure_id or "pathway_1",
            title=response.get("title", "Unknown Pathway"),
            condition=response.get("condition", ""),
            phases=response.get("phases", []),
            nodes=nodes,
            edges=edges,
            entry_criteria=response.get("entry_criteria"),
            primary_drugs=response.get("primary_drugs", []),
            alternative_drugs=response.get("alternative_drugs", []),
            decision_points=response.get("decision_points", []),
            target_dose=response.get("target_dose"),
            target_timepoint=response.get("target_timepoint"),
            maintenance_duration=response.get("maintenance_duration"),
            relapse_handling=response.get("relapse_handling"),
            source_figure=figure_id,
            source_page=page_num,
            extraction_confidence=0.8 if nodes else 0.0,
        )

        return pathway

    def _extract_drugs_from_text(self, text: str) -> List[str]:
        """Extract drug names from node text using patterns."""
        drugs = []
        text_upper = text.upper()

        for pattern, drug_name in self.drug_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                if drug_name not in drugs:
                    drugs.append(drug_name)

        return drugs

    def _call_vision_llm(
        self,
        image_base64: str,
        prompt: str,
    ) -> Optional[Dict[str, Any]]:
        """Call Vision LLM and parse JSON response."""
        if not self.llm_client:
            return None

        try:
            # Prepare message with image
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }

            # Call API
            response = self.llm_client.messages.create(
                model=self.llm_model,
                max_tokens=4096,
                messages=[message],
            )

            # Extract text from response
            if response.content and len(response.content) > 0:
                text = response.content[0].text

                # Parse JSON from response
                import json

                # Try to find JSON in the response
                text = text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]

                return json.loads(text.strip())

        except Exception as e:
            print(f"[WARN] Vision LLM call failed: {e}")
            return None

        return None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def extract_care_pathway_from_figure(
    image_base64: str,
    llm_client: Any,
    llm_model: str = "claude-sonnet-4-20250514",
    ocr_text: Optional[str] = None,
    caption: Optional[str] = None,
    figure_id: Optional[str] = None,
    page_num: Optional[int] = None,
) -> Optional[CarePathway]:
    """
    Convenience function to extract care pathway from a figure.

    Args:
        image_base64: Base64-encoded image
        llm_client: Anthropic client
        llm_model: Model to use
        ocr_text: Optional OCR text
        caption: Optional caption
        figure_id: Figure identifier
        page_num: Page number

    Returns:
        CarePathway or None
    """
    extractor = FlowchartGraphExtractor(llm_client, llm_model)
    return extractor.extract_care_pathway(
        image_base64,
        ocr_text=ocr_text,
        caption=caption,
        figure_id=figure_id,
        page_num=page_num,
    )


__all__ = [
    "FlowchartGraphExtractor",
    "extract_care_pathway_from_figure",
    "CARE_PATHWAY_PROMPT",
]
