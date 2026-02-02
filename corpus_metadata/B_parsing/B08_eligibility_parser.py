# corpus_metadata/B_parsing/B08_eligibility_parser.py
"""
Eligibility criteria logical expression parser for clinical trials.

This module parses natural language eligibility criteria text into structured
LogicalExpression trees that can be evaluated programmatically for feasibility
simulation. It handles AND/OR operators, nested parentheses, negation, lab value
ranges (eGFR >= 30), severity grades (NYHA, ECOG, CKD), and age criteria.

Key Components:
    - EligibilityParser: Main parser class for criteria text
    - LogicalExpression: Parsed expression tree with criteria references
    - ParsedCriterion: Intermediate representation with lab/severity data
    - TokenizedCriteria: Tokenized criteria with operator and criterion IDs
    - parse_eligibility: Convenience function for parsing criteria text

Example:
    >>> from B_parsing.B08_eligibility_parser import EligibilityParser
    >>> parser = EligibilityParser()
    >>> expr = parser.parse("Age >= 18 years and eGFR >= 30 mL/min")
    >>> for crit_id, criterion in expr.criteria_refs.items():
    ...     print(f"{crit_id}: {criterion.text}")

Dependencies:
    - A_core.A07_feasibility_models: CriterionNode, CriterionType, EligibilityCriterion,
      ExtractionMethod, LabCriterion, LogicalExpression, LogicalOperator,
      SeverityGrade, SeverityGradeType, SEVERITY_GRADE_MAPPINGS
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from A_core.A07_feasibility_models import (
    CriterionNode,
    CriterionType,
    EligibilityCriterion,
    ExtractionMethod,
    LabCriterion,
    LogicalExpression,
    LogicalOperator,
    SeverityGrade,
    SeverityGradeType,
    SEVERITY_GRADE_MAPPINGS,
)

logger = logging.getLogger(__name__)


@dataclass
class ParsedCriterion:
    """Intermediate representation of a parsed criterion."""

    id: str
    text: str
    criterion_type: CriterionType
    lab_criterion: Optional[LabCriterion] = None
    severity_grade: Optional[SeverityGrade] = None
    is_negated: bool = False


@dataclass
class TokenizedCriteria:
    """Tokenized eligibility criteria for parsing."""

    tokens: List[str] = field(default_factory=list)
    criteria: Dict[str, ParsedCriterion] = field(default_factory=dict)


class EligibilityParser:
    """
    Parser for eligibility criteria text.

    Converts natural language eligibility criteria into structured
    LogicalExpression trees that can be evaluated against patient data.
    """

    # Patterns for detecting logical operators
    AND_PATTERNS = [
        r"\band\b",
        r"\b&\b",
        r";\s*(?=[A-Z])",  # Semicolon followed by capital letter
        r",\s*(?=(?:and|with|who|patients?))",  # Comma before certain words
    ]

    OR_PATTERNS = [
        r"\bor\b",
        r"\|",
        r"\beither\b.*\bor\b",
    ]

    NOT_PATTERNS = [
        r"\bnot\b",
        r"\bno\b",
        r"\bwithout\b",
        r"\bexclude[ds]?\b",
        r"\babsence\s+of\b",
    ]

    # Lab value patterns
    LAB_PATTERNS = [
        # eGFR >= 30 mL/min/1.73m²
        r"(?P<analyte>eGFR|GFR|creatinine|UPCR|UACR|proteinuria|albumin|C3|C4|hemoglobin|Hb|platelet|WBC|ANC)\s*"
        r"(?P<op>[<>=≤≥]+)\s*(?P<value>[\d.]+)\s*(?P<unit>[a-zA-Z²/]+)?",
        # eGFR 30-89 mL/min (range)
        r"(?P<analyte>eGFR|GFR|creatinine|UPCR|age)\s*"
        r"(?P<min>[\d.]+)\s*[-–to]+\s*(?P<max>[\d.]+)\s*(?P<unit>[a-zA-Z²/]+)?",
        # Age >= 18 years
        r"(?:age[d]?\s*)?(?P<op>[<>=≤≥]+)\s*(?P<value>\d+)\s*(?:years?|yrs?)",
        # 18-75 years (age range)
        r"(?:age[d]?\s*)?(?P<min>\d+)\s*[-–to]+\s*(?P<max>\d+)\s*(?:years?|yrs?)",
    ]

    # Severity grade patterns
    SEVERITY_PATTERNS = {
        SeverityGradeType.NYHA: [
            r"NYHA\s*(?:class\s*)?(?P<grade>[IViv]+|\d)",
            r"NYHA\s*(?:class\s*)?(?P<min>[IViv]+|\d)\s*[-–to]+\s*(?P<max>[IViv]+|\d)",
            r"(?:class\s*)?(?P<grade>[IViv]+)\s*(?:heart\s*failure|HF)",
        ],
        SeverityGradeType.ECOG: [
            r"ECOG\s*(?:PS|performance\s*status)?\s*(?P<op>[<>=≤≥]+)?\s*(?P<grade>\d)",
            r"ECOG\s*(?P<min>\d)\s*[-–to]+\s*(?P<max>\d)",
        ],
        SeverityGradeType.CKD: [
            r"CKD\s*(?:stage\s*)?(?P<grade>\d[ab]?)",
            r"(?:stage\s*)(?P<grade>\d[ab]?)\s*CKD",
        ],
        SeverityGradeType.CHILD_PUGH: [
            r"Child[-\s]?Pugh\s*(?:class\s*)?(?P<grade>[ABCabc])",
            r"Child[-\s]?Pugh\s*(?:score\s*)?(?P<op>[<>=≤≥]+)?\s*(?P<grade>\d+)",
        ],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.default_criterion_type = CriterionType.INCLUSION

        # Compile patterns
        self._and_re = re.compile("|".join(self.AND_PATTERNS), re.IGNORECASE)
        self._or_re = re.compile("|".join(self.OR_PATTERNS), re.IGNORECASE)
        self._not_re = re.compile("|".join(self.NOT_PATTERNS), re.IGNORECASE)
        self._lab_patterns = [re.compile(p, re.IGNORECASE) for p in self.LAB_PATTERNS]
        self._severity_patterns = {
            k: [re.compile(p, re.IGNORECASE) for p in v]
            for k, v in self.SEVERITY_PATTERNS.items()
        }

    def parse(
        self, text: str, criterion_type: CriterionType = CriterionType.INCLUSION
    ) -> LogicalExpression:
        """
        Parse eligibility criteria text into a LogicalExpression.

        Args:
            text: Eligibility criteria text
            criterion_type: Whether this is inclusion or exclusion

        Returns:
            LogicalExpression with parsed criteria tree
        """
        # Tokenize and extract individual criteria
        tokenized = self._tokenize(text, criterion_type)

        # Build the logical expression tree
        root = self._build_tree(tokenized.tokens, tokenized.criteria)

        # Create criteria refs for the expression
        criteria_refs = {}
        for crit_id, parsed in tokenized.criteria.items():
            criteria_refs[crit_id] = self._to_eligibility_criterion(parsed)

        return LogicalExpression(
            root=root,
            raw_text=text,
            criteria_refs=criteria_refs,
        )

    def parse_lab_value(self, text: str) -> Optional[LabCriterion]:
        """
        Parse a lab value criterion from text.

        Args:
            text: Text containing lab criterion (e.g., "eGFR >= 30 mL/min")

        Returns:
            LabCriterion if parsed, None otherwise
        """
        for pattern in self._lab_patterns:
            match = pattern.search(text)
            if match:
                groups = match.groupdict()

                # Range pattern (min-max)
                if "min" in groups and "max" in groups and groups["min"] and groups["max"]:
                    analyte = groups.get("analyte", "value")
                    if not analyte:
                        # Check if this is an age pattern
                        if "year" in text.lower() or "age" in text.lower():
                            analyte = "age"
                        else:
                            analyte = "value"

                    return LabCriterion(
                        analyte=analyte,
                        operator="between",
                        value=0,  # Not used for between
                        unit=groups.get("unit", ""),
                        min_value=float(groups["min"]),
                        max_value=float(groups["max"]),
                    )

                # Single value pattern
                if "value" in groups and groups["value"]:
                    analyte = groups.get("analyte", "value")
                    if not analyte:
                        if "year" in text.lower() or "age" in text.lower():
                            analyte = "age"
                        else:
                            analyte = "value"

                    op = groups.get("op", ">=")
                    op = self._normalize_operator(op)

                    return LabCriterion(
                        analyte=analyte,
                        operator=op,
                        value=float(groups["value"]),
                        unit=groups.get("unit", ""),
                    )

        return None

    def parse_severity_grade(self, text: str) -> Optional[SeverityGrade]:
        """
        Parse a severity grade from text.

        Args:
            text: Text containing severity grade (e.g., "NYHA Class II-III")

        Returns:
            SeverityGrade if parsed, None otherwise
        """
        for grade_type, patterns in self._severity_patterns.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    groups = match.groupdict()

                    # Range pattern
                    if "min" in groups and "max" in groups:
                        min_val = self._normalize_grade(grade_type, groups["min"])
                        max_val = self._normalize_grade(grade_type, groups["max"])
                        return SeverityGrade(
                            grade_type=grade_type,
                            raw_value=match.group(0),
                            min_value=min_val,
                            max_value=max_val,
                            operator="between",
                        )

                    # Single value
                    if "grade" in groups:
                        grade_val = self._normalize_grade(grade_type, groups["grade"])
                        op = groups.get("op")
                        if op:
                            op = self._normalize_operator(op)
                        else:
                            op = "=="

                        return SeverityGrade(
                            grade_type=grade_type,
                            raw_value=match.group(0),
                            numeric_value=grade_val,
                            operator=op,
                        )

        return None

    def _tokenize(
        self, text: str, criterion_type: CriterionType
    ) -> TokenizedCriteria:
        """Tokenize text into logical tokens and criteria."""
        result = TokenizedCriteria()

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Split on logical operators while preserving them
        parts = self._split_on_operators(text)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check if this is an operator
            if self._and_re.fullmatch(part):
                result.tokens.append("AND")
            elif self._or_re.fullmatch(part):
                result.tokens.append("OR")
            elif part == "(":
                result.tokens.append("(")
            elif part == ")":
                result.tokens.append(")")
            else:
                # This is a criterion - parse it
                crit_id = f"c_{uuid.uuid4().hex[:8]}"
                parsed = self._parse_criterion(part, criterion_type, crit_id)
                result.criteria[crit_id] = parsed
                result.tokens.append(crit_id)

        return result

    def _split_on_operators(self, text: str) -> List[str]:
        """Split text on logical operators while preserving parentheses."""
        # First, handle parentheses
        text = text.replace("(", " ( ").replace(")", " ) ")

        # Split on AND/OR patterns
        parts: list[str] = []
        current: list[str] = []

        # Simple tokenization - split on spaces and recognize operators
        words = text.split()
        i = 0
        while i < len(words):
            word = words[i]

            if word in "()":
                if current:
                    parts.append(" ".join(current))
                    current = []
                parts.append(word)
            elif word.lower() == "and" or word == "&":
                if current:
                    parts.append(" ".join(current))
                    current = []
                parts.append("AND")
            elif word.lower() == "or" or word == "|":
                if current:
                    parts.append(" ".join(current))
                    current = []
                parts.append("OR")
            elif word == ";":
                if current:
                    parts.append(" ".join(current))
                    current = []
                parts.append("AND")  # Treat semicolon as AND
            else:
                current.append(word)

            i += 1

        if current:
            parts.append(" ".join(current))

        return parts

    def _parse_criterion(
        self, text: str, criterion_type: CriterionType, crit_id: str
    ) -> ParsedCriterion:
        """Parse a single criterion from text."""
        # Check for negation
        is_negated = bool(self._not_re.search(text))

        # Try to parse as lab value
        lab_criterion = self.parse_lab_value(text)

        # Try to parse as severity grade
        severity_grade = self.parse_severity_grade(text)

        return ParsedCriterion(
            id=crit_id,
            text=text,
            criterion_type=criterion_type,
            lab_criterion=lab_criterion,
            severity_grade=severity_grade,
            is_negated=is_negated,
        )

    def _build_tree(
        self, tokens: List[str], criteria: Dict[str, ParsedCriterion]
    ) -> CriterionNode:
        """Build a logical expression tree from tokens."""
        if not tokens:
            return CriterionNode(operator=LogicalOperator.AND, children=[])

        # Handle single criterion
        if len(tokens) == 1:
            token = tokens[0]
            if token in criteria:
                return CriterionNode(criterion_id=token)
            return CriterionNode(operator=LogicalOperator.AND, children=[])

        # Find the lowest precedence operator (OR < AND)
        # Work from right to left to get left-associativity
        paren_depth = 0
        or_pos = -1
        and_pos = -1

        for i in range(len(tokens) - 1, -1, -1):
            token = tokens[i]
            if token == "(":
                paren_depth += 1
            elif token == ")":
                paren_depth -= 1
            elif paren_depth == 0:
                if token == "OR" and or_pos == -1:
                    or_pos = i
                elif token == "AND" and and_pos == -1:
                    and_pos = i

        # Split on OR first (lower precedence)
        if or_pos != -1:
            left = self._build_tree(tokens[:or_pos], criteria)
            right = self._build_tree(tokens[or_pos + 1 :], criteria)
            return CriterionNode(
                operator=LogicalOperator.OR,
                children=[left, right],
            )

        # Split on AND
        if and_pos != -1:
            left = self._build_tree(tokens[:and_pos], criteria)
            right = self._build_tree(tokens[and_pos + 1 :], criteria)
            return CriterionNode(
                operator=LogicalOperator.AND,
                children=[left, right],
            )

        # Handle parentheses
        if tokens[0] == "(" and tokens[-1] == ")":
            return self._build_tree(tokens[1:-1], criteria)

        # Multiple criteria without explicit operators - treat as AND
        children = []
        for token in tokens:
            if token in criteria:
                children.append(CriterionNode(criterion_id=token))
            elif token not in ("(", ")"):
                # Recursively handle
                pass

        if len(children) == 1:
            return children[0]
        elif children:
            return CriterionNode(operator=LogicalOperator.AND, children=children)

        return CriterionNode(operator=LogicalOperator.AND, children=[])

    def _to_eligibility_criterion(self, parsed: ParsedCriterion) -> EligibilityCriterion:
        """Convert ParsedCriterion to EligibilityCriterion."""
        return EligibilityCriterion(
            criterion_type=parsed.criterion_type,
            text=parsed.text,
            lab_criterion=parsed.lab_criterion,
            severity_grade=parsed.severity_grade,
            is_negated=parsed.is_negated,
            extraction_method=ExtractionMethod.REGEX,
        )

    def _normalize_operator(self, op: str) -> str:
        """Normalize comparison operators."""
        op = op.strip()
        mapping = {
            "≥": ">=",
            "≤": "<=",
            "=>": ">=",
            "=<": "<=",
            "=": "==",
        }
        return mapping.get(op, op)

    def _normalize_grade(self, grade_type: SeverityGradeType, value: str) -> Optional[int]:
        """Normalize a severity grade value to integer."""
        value_lower = value.lower().strip()
        mappings = SEVERITY_GRADE_MAPPINGS.get(grade_type, {})

        # Direct lookup
        if value_lower in mappings:
            return mappings[value_lower]

        # Try numeric
        try:
            return int(value)
        except ValueError:
            pass

        # Roman numeral conversion for NYHA
        if grade_type == SeverityGradeType.NYHA:
            roman_map = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
            return roman_map.get(value_lower)

        return None


# Convenience function
def parse_eligibility(
    text: str, criterion_type: CriterionType = CriterionType.INCLUSION
) -> LogicalExpression:
    """
    Parse eligibility criteria text into a LogicalExpression.

    Args:
        text: Eligibility criteria text
        criterion_type: Whether this is inclusion or exclusion

    Returns:
        LogicalExpression with parsed criteria tree
    """
    parser = EligibilityParser()
    return parser.parse(text, criterion_type)
