# corpus_metadata/A_core/A07b_logical_expressions.py
"""
Logical expression models for eligibility criteria computability.

Provides a tree-based representation for complex eligibility expressions:
- AND/OR/NOT operators
- Recursive evaluation against patient data
- SQL WHERE clause generation for EHR queries
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "LogicalOperator",
    "CriterionNode",
    "LogicalExpression",
]


class LogicalOperator(str, Enum):
    """Logical operators for combining eligibility criteria."""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class CriterionNode(BaseModel):
    """
    Node in a logical expression tree for eligibility criteria.

    Can be:
    - A leaf node containing a single criterion
    - An internal node with an operator and children
    """

    # For leaf nodes - use Any at runtime to avoid circular import
    criterion: Optional[Any] = None
    criterion_id: Optional[str] = None  # Reference to criterion by ID

    # For internal nodes
    operator: Optional[LogicalOperator] = None
    children: List["CriterionNode"] = Field(default_factory=list)

    # Metadata
    raw_text: Optional[str] = None  # Original text fragment
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.criterion is not None or self.criterion_id is not None

    def evaluate(self, criterion_values: Dict[str, bool]) -> bool:
        """
        Evaluate the logical expression given criterion truth values.

        Args:
            criterion_values: Dict mapping criterion_id to True/False

        Returns:
            Boolean result of evaluating the expression
        """
        if self.is_leaf():
            if self.criterion_id:
                return criterion_values.get(self.criterion_id, False)
            return False

        if self.operator == LogicalOperator.AND:
            return all(child.evaluate(criterion_values) for child in self.children)
        elif self.operator == LogicalOperator.OR:
            return any(child.evaluate(criterion_values) for child in self.children)
        elif self.operator == LogicalOperator.NOT:
            if self.children:
                return not self.children[0].evaluate(criterion_values)
        return False


class LogicalExpression(BaseModel):
    """
    Complete logical expression for eligibility criteria.

    Represents structured eligibility like:
    "(Age >= 18 AND Age <= 75) AND (eGFR >= 30 OR on_dialysis)"
    """

    root: CriterionNode
    raw_text: str  # Original eligibility text
    # Use Any at runtime to avoid circular import
    criteria_refs: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    def evaluate(self, patient_data: Dict[str, Any]) -> bool:
        """
        Evaluate eligibility for a patient.

        Args:
            patient_data: Dict with patient values (age, eGFR, etc.)

        Returns:
            True if patient meets eligibility criteria
        """
        # First evaluate each criterion against patient data
        criterion_values = {}
        for crit_id, criterion in self.criteria_refs.items():
            criterion_values[crit_id] = self._evaluate_criterion(criterion, patient_data)

        # Then evaluate the logical expression
        return self.root.evaluate(criterion_values)

    def _evaluate_criterion(
        self, criterion: Any, patient_data: Dict[str, Any]
    ) -> bool:
        """Evaluate a single criterion against patient data."""
        if criterion.lab_criterion:
            lab = criterion.lab_criterion
            actual = patient_data.get(lab.analyte.lower())
            if actual is not None:
                result = lab.evaluate(actual)
                return not result if criterion.is_negated else result
        return False

    def to_sql_where(self) -> str:
        """Convert expression to SQL WHERE clause (for EHR queries)."""
        return self._node_to_sql(self.root)

    def _node_to_sql(self, node: CriterionNode) -> str:
        """Convert a node to SQL."""
        if node.is_leaf():
            crit = self.criteria_refs.get(node.criterion_id) if node.criterion_id else None
            if crit and crit.lab_criterion:
                lab = crit.lab_criterion
                col = lab.analyte.lower()
                if lab.operator == "between":
                    return f"({col} BETWEEN {lab.min_value} AND {lab.max_value})"
                return f"({col} {lab.operator} {lab.value})"
            return "TRUE"

        if node.operator == LogicalOperator.AND:
            parts = [self._node_to_sql(c) for c in node.children]
            return f"({' AND '.join(parts)})"
        elif node.operator == LogicalOperator.OR:
            parts = [self._node_to_sql(c) for c in node.children]
            return f"({' OR '.join(parts)})"
        elif node.operator == LogicalOperator.NOT:
            if node.children:
                return f"(NOT {self._node_to_sql(node.children[0])})"
        return "TRUE"


# Update forward references for recursive models
CriterionNode.model_rebuild()
LogicalExpression.model_rebuild()
