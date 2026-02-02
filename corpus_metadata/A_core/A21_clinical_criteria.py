# corpus_metadata/A_core/A21_clinical_criteria.py
"""
Domain models for computable clinical eligibility criteria.

This module provides Pydantic models for representing structured eligibility
criteria that can be programmatically evaluated against patient data. Use these
models when extracting lab thresholds, diagnosis confirmation requirements, or
severity grades from clinical trial protocols for feasibility simulation.

Key Components:
    - EntityNormalization: Standard coding for drugs/conditions/labs (LOINC, RxNorm)
    - LabTimepoint: Structured timepoint for lab requirements (day, visit, window)
    - LabCriterion: Computable lab criterion with evaluate() method
    - DiagnosisConfirmation: Structured diagnosis confirmation requirement
    - SeverityGradeType: Enum for clinical severity systems (NYHA, ECOG, CKD, etc.)
    - SeverityGrade: Normalized severity grade with evaluate() method
    - SEVERITY_GRADE_MAPPINGS: Dict mapping raw text to numeric grades

Example:
    >>> from A_core.A21_clinical_criteria import LabCriterion, SeverityGrade
    >>> lab = LabCriterion(
    ...     analyte="eGFR", operator=">=", value=30.0, unit="mL/min/1.73m2"
    ... )
    >>> lab.evaluate(45.0)  # Patient's eGFR is 45
    True
    >>> grade = SeverityGrade(
    ...     grade_type=SeverityGradeType.NYHA, raw_value="Class II",
    ...     numeric_value=2, operator="<="
    ... )
    >>> grade.evaluate(2)  # Patient is NYHA Class II
    True

Dependencies:
    - pydantic: For model validation and serialization
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, ConfigDict, Field


# -------------------------
# Entity Normalization
# -------------------------


class EntityNormalization(BaseModel):
    """Normalized coding for drugs, conditions, labs."""

    system: str  # "LOINC", "RxNorm", "ATC", "Orphanet", "SNOMED", "ICD-10"
    code: str
    label: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Lab Criterion (Computable)
# -------------------------


class LabTimepoint(BaseModel):
    """Structured timepoint for lab requirements."""

    day: Optional[int] = None  # relative to randomization
    visit_name: Optional[str] = None  # "screening", "baseline"
    window_days: Optional[int] = None  # +/- days

    model_config = ConfigDict(extra="forbid")


class LabCriterion(BaseModel):
    """Fully computable lab eligibility criterion."""

    analyte: str  # "UPCR", "eGFR", "C3"
    operator: str  # ">=", "<=", ">", "<", "==", "between"
    value: float
    unit: str
    # Range support: min_value and max_value for "between" operator
    min_value: Optional[float] = None  # For ranges like "eGFR 30-89"
    max_value: Optional[float] = None  # For ranges
    specimen: Optional[str] = None  # "first_morning_void", "serum", "plasma"
    timepoints: List[LabTimepoint] = Field(default_factory=list)
    normalization: Optional[EntityNormalization] = None  # LOINC code

    model_config = ConfigDict(extra="forbid")

    def evaluate(self, actual_value: float) -> bool:
        """Evaluate if actual_value satisfies this criterion."""
        if self.operator == ">=":
            return actual_value >= self.value
        elif self.operator == "<=":
            return actual_value <= self.value
        elif self.operator == ">":
            return actual_value > self.value
        elif self.operator == "<":
            return actual_value < self.value
        elif self.operator == "==":
            return actual_value == self.value
        elif self.operator == "between":
            if self.min_value is not None and self.max_value is not None:
                return self.min_value <= actual_value <= self.max_value
        return False


# -------------------------
# Diagnosis Confirmation
# -------------------------


class DiagnosisConfirmation(BaseModel):
    """Structured diagnosis confirmation requirement."""

    method: str  # "biopsy", "genetic_testing", "clinical_criteria"
    window_months: Optional[int] = None  # within X months of screening
    assessor: Optional[str] = None  # "local histopathologist", "central review"
    findings: Optional[str] = None  # specific pathological findings

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Severity Grade Normalization
# -------------------------


class SeverityGradeType(str, Enum):
    """Standard clinical severity grading systems."""

    NYHA = "nyha"  # Heart failure: I-IV
    ECOG = "ecog"  # Performance status: 0-5
    CKD = "ckd"  # Chronic kidney disease: 1-5
    CHILD_PUGH = "child_pugh"  # Liver function: A, B, C (5-15 points)
    MELD = "meld"  # Liver disease: 6-40
    BCLC = "bclc"  # Liver cancer: 0, A, B, C, D
    TNM = "tnm"  # Cancer staging: T0-4, N0-3, M0-1
    EDSS = "edss"  # MS disability: 0-10


class SeverityGrade(BaseModel):
    """Normalized severity grade for clinical scales."""

    grade_type: SeverityGradeType
    raw_value: str  # Original text (e.g., "NYHA Class II", "ECOG 0-1")
    numeric_value: Optional[int] = None  # Normalized integer (e.g., 2 for NYHA II)
    min_value: Optional[int] = None  # For ranges (e.g., ECOG 0-1 -> min=0)
    max_value: Optional[int] = None  # For ranges (e.g., ECOG 0-1 -> max=1)
    operator: Optional[str] = None  # "<=", ">=", "==", "between"

    model_config = ConfigDict(extra="forbid")

    def evaluate(self, actual_grade: int) -> bool:
        """Evaluate if actual_grade satisfies this criterion."""
        if self.operator == "<=":
            return actual_grade <= (self.numeric_value or self.max_value or 0)
        elif self.operator == ">=":
            return actual_grade >= (self.numeric_value or self.min_value or 0)
        elif self.operator == "==":
            return actual_grade == self.numeric_value
        elif self.operator == "between" and self.min_value is not None and self.max_value is not None:
            return self.min_value <= actual_grade <= self.max_value
        return False


# Severity grade normalization mappings
SEVERITY_GRADE_MAPPINGS = {
    SeverityGradeType.NYHA: {
        "i": 1, "1": 1, "class i": 1, "class 1": 1,
        "ii": 2, "2": 2, "class ii": 2, "class 2": 2,
        "iii": 3, "3": 3, "class iii": 3, "class 3": 3,
        "iv": 4, "4": 4, "class iv": 4, "class 4": 4,
    },
    SeverityGradeType.ECOG: {
        "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    },
    SeverityGradeType.CKD: {
        "1": 1, "2": 2, "3": 3, "3a": 3, "3b": 3, "4": 4, "5": 5,
        "stage 1": 1, "stage 2": 2, "stage 3": 3, "stage 3a": 3,
        "stage 3b": 3, "stage 4": 4, "stage 5": 5,
    },
    SeverityGradeType.CHILD_PUGH: {
        "a": 1, "b": 2, "c": 3,
        "class a": 1, "class b": 2, "class c": 3,
        "5": 1, "6": 1, "7": 2, "8": 2, "9": 2, "10": 3,
        "11": 3, "12": 3, "13": 3, "14": 3, "15": 3,
    },
}
