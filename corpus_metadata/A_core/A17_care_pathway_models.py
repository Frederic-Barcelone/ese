# corpus_metadata/A_core/A17_care_pathway_models.py
"""
Domain models for clinical care pathways and treatment algorithms.

This module provides Pydantic models for representing structured clinical decision
trees extracted from guideline flowcharts and protocol documents. Use these models
when processing treatment algorithms, drug tapering schedules, or disease management
workflows from sources like EULAR/ACR guidelines.

Key Components:
    - NodeType: Enum for pathway node types (START, END, ACTION, DECISION, etc.)
    - CarePathwayNode: Single step in a treatment algorithm with drugs, dosing, duration
    - CarePathwayEdge: Transition between nodes with optional conditions
    - CarePathway: Complete decision tree with nodes, edges, and clinical metadata
    - TaperSchedulePoint: Single timepoint in a medication taper schedule
    - TaperSchedule: Structured dosing schedule for gradual dose reduction

Example:
    >>> from A_core.A17_care_pathway_models import CarePathway, CarePathwayNode, NodeType
    >>> node = CarePathwayNode(
    ...     id="n1", type=NodeType.ACTION, text="Start RTX 375mg/m2",
    ...     drugs=["rituximab"], phase="induction"
    ... )
    >>> pathway = CarePathway(
    ...     pathway_id="aav_001", title="AAV Induction", condition="Active GPA/MPA",
    ...     nodes=[node]
    ... )
    >>> pathway.get_action_nodes()
    [CarePathwayNode(id='n1', ...)]

Dependencies:
    - pydantic: For model validation and serialization
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Type of node in a care pathway graph."""
    START = "start"
    END = "end"
    ACTION = "action"  # Treatment action (prescribe drug, perform procedure)
    DECISION = "decision"  # Decision point (diamond shape)
    ASSESSMENT = "assessment"  # Clinical assessment
    NOTE = "note"  # Informational note
    CONDITION = "condition"  # Entry condition/prerequisite


class CarePathwayNode(BaseModel):
    """
    A node in a care pathway graph.

    Represents a single step in a treatment algorithm - either an action
    to take, a decision to make, or an assessment to perform.
    """
    id: str  # Unique identifier within the pathway
    type: NodeType = NodeType.ACTION
    text: str  # Raw text from the flowchart box/diamond
    phase: Optional[str] = None  # e.g., "induction", "maintenance", "relapse"

    # Normalized clinical entities extracted from text
    drugs: List[str] = Field(default_factory=list)  # Drug names mentioned
    dose: Optional[str] = None  # Dosing information if present
    duration: Optional[str] = None  # Duration if specified
    frequency: Optional[str] = None  # Frequency if specified

    # Position info (for visualization)
    position: Optional[Dict[str, float]] = None  # {"x": 0.5, "y": 0.2}

    # Provenance
    source_bbox: Optional[List[float]] = None  # [x0, y0, x1, y1] in figure


class CarePathwayEdge(BaseModel):
    """
    An edge connecting two nodes in a care pathway.

    Represents a transition between steps, potentially with a condition
    that must be met for the transition to apply.
    """
    source_id: str
    target_id: str
    condition: Optional[str] = None  # "Yes", "No", "organ/life-threatening", etc.
    condition_type: Optional[str] = None  # "positive", "negative", "severity", etc.
    label: Optional[str] = None  # Additional label on the edge


class CarePathway(BaseModel):
    """
    A complete care pathway extracted from a clinical algorithm figure.

    Represents the full decision tree with nodes (actions/decisions) and
    edges (transitions with conditions).
    """
    # Identification
    pathway_id: str
    title: str
    condition: str  # Target condition (e.g., "Active GPA/MPA")
    guideline_source: Optional[str] = None  # e.g., "2022 EULAR recommendations"

    # Structure
    phases: List[str] = Field(default_factory=list)  # ["Induction", "Maintenance"]
    nodes: List[CarePathwayNode] = Field(default_factory=list)
    edges: List[CarePathwayEdge] = Field(default_factory=list)

    # Extracted clinical intelligence
    entry_criteria: Optional[str] = None  # Who this pathway applies to
    primary_drugs: List[str] = Field(default_factory=list)  # Main treatment drugs
    alternative_drugs: List[str] = Field(default_factory=list)  # Alternatives
    decision_points: List[str] = Field(default_factory=list)  # Summary of key decisions

    # Specific clinical parameters
    target_dose: Optional[str] = None  # e.g., "5 mg/day"
    target_timepoint: Optional[str] = None  # e.g., "by 4-5 months"
    maintenance_duration: Optional[str] = None  # e.g., "24-48 months"
    relapse_handling: Optional[str] = None  # Description of relapse management

    # Provenance
    source_figure: Optional[str] = None  # Figure ID (e.g., "Figure 1")
    source_page: Optional[int] = None
    extraction_confidence: float = 0.0

    def get_node(self, node_id: str) -> Optional[CarePathwayNode]:
        """Get a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_outgoing_edges(self, node_id: str) -> List[CarePathwayEdge]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e.source_id == node_id]

    def get_incoming_edges(self, node_id: str) -> List[CarePathwayEdge]:
        """Get all edges pointing to a node."""
        return [e for e in self.edges if e.target_id == node_id]

    def get_decision_nodes(self) -> List[CarePathwayNode]:
        """Get all decision nodes in the pathway."""
        return [n for n in self.nodes if n.type == NodeType.DECISION]

    def get_action_nodes(self) -> List[CarePathwayNode]:
        """Get all action nodes in the pathway."""
        return [n for n in self.nodes if n.type == NodeType.ACTION]

    def to_summary(self) -> Dict[str, Any]:
        """Generate a summary of the pathway for export."""
        return {
            "pathway_id": self.pathway_id,
            "title": self.title,
            "condition": self.condition,
            "phases": self.phases,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "decision_points": self.decision_points,
            "primary_drugs": self.primary_drugs,
            "target_dose": self.target_dose,
            "maintenance_duration": self.maintenance_duration,
        }


class TaperSchedulePoint(BaseModel):
    """A single point in a taper schedule."""
    week: Optional[int] = None
    day: Optional[int] = None
    month: Optional[float] = None
    dose_mg: float
    dose_unit: str = "mg/day"


class TaperSchedule(BaseModel):
    """
    A medication taper schedule extracted from a chart.

    Represents a structured dosing schedule typically for glucocorticoids
    or other medications that require gradual dose reduction.
    """
    schedule_id: str
    drug_name: str  # e.g., "prednisone", "prednisolone"
    regimen_name: str  # e.g., "Protocol target", "Reduced regimen"

    # Schedule data
    schedule: List[TaperSchedulePoint] = Field(default_factory=list)

    # Target parameters
    starting_dose: Optional[str] = None  # e.g., "60 mg/day"
    target_dose: Optional[str] = None  # e.g., "5 mg/day"
    target_timepoint: Optional[str] = None  # e.g., "4-5 months"

    # Metadata
    source_figure: Optional[str] = None
    source_page: Optional[int] = None
    legend_color: Optional[str] = None  # "red", "black", etc.
    legend_marker: Optional[str] = None  # "circle", "square", etc.

    # Linked recommendations
    linked_recommendation_text: Optional[str] = None

    def get_dose_at_week(self, week: int) -> Optional[float]:
        """Get the dose at a specific week."""
        for point in self.schedule:
            if point.week == week:
                return point.dose_mg
        return None

    def to_summary(self) -> Dict[str, Any]:
        """Generate a summary for export."""
        return {
            "schedule_id": self.schedule_id,
            "drug_name": self.drug_name,
            "regimen_name": self.regimen_name,
            "starting_dose": self.starting_dose,
            "target_dose": self.target_dose,
            "target_timepoint": self.target_timepoint,
            "data_points": len(self.schedule),
        }


__all__ = [
    "NodeType",
    "CarePathwayNode",
    "CarePathwayEdge",
    "CarePathway",
    "TaperSchedulePoint",
    "TaperSchedule",
]
