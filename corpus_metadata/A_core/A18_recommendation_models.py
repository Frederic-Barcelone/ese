# corpus_metadata/A_core/A18_recommendation_models.py
"""
Domain models for clinical guideline recommendations.

This module provides Pydantic models for representing structured treatment
recommendations extracted from clinical guidelines. Use these models when
processing EULAR/ACR recommendations, FDA labeling guidance, or protocol-specific
treatment advice with evidence levels and dosing constraints.

Key Components:
    - EvidenceLevel: Enum for evidence quality (HIGH, MODERATE, LOW, VERY_LOW)
    - RecommendationStrength: Enum for recommendation strength (STRONG, CONDITIONAL)
    - RecommendationType: Enum for recommendation categories (TREATMENT, DOSING, etc.)
    - DrugDosingInfo: Structured dosing information for a drug
    - GuidelineRecommendation: Single actionable recommendation with population/action
    - RecommendationSet: Collection of recommendations from a guideline document

Example:
    >>> from A_core.A18_recommendation_models import (
    ...     GuidelineRecommendation, RecommendationType, EvidenceLevel
    ... )
    >>> rec = GuidelineRecommendation(
    ...     recommendation_id="rec_001",
    ...     recommendation_type=RecommendationType.TREATMENT,
    ...     population="GPA/MPA organ-threatening",
    ...     action="GC + RTX or CYC",
    ...     evidence_level=EvidenceLevel.MODERATE,
    ...     source="Table 2"
    ... )
    >>> rec.to_summary()
    {'recommendation_id': 'rec_001', 'type': 'treatment', ...}

Dependencies:
    - pydantic: For model validation and serialization
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvidenceLevel(str, Enum):
    """Evidence level for recommendations."""
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"
    EXPERT_OPINION = "expert_opinion"
    UNKNOWN = "unknown"


class RecommendationStrength(str, Enum):
    """Strength of recommendation."""
    STRONG = "strong"
    CONDITIONAL = "conditional"
    WEAK = "weak"
    UNKNOWN = "unknown"


class RecommendationType(str, Enum):
    """Type of clinical recommendation."""
    TREATMENT = "treatment"
    DOSING = "dosing"
    DURATION = "duration"
    MONITORING = "monitoring"
    CONTRAINDICATION = "contraindication"
    ALTERNATIVE = "alternative"
    PREFERENCE = "preference"
    OTHER = "other"


class DrugDosingInfo(BaseModel):
    """Dosing information for a drug."""
    drug_name: str
    dose_range: Optional[str] = None  # "50-75 mg/day"
    starting_dose: Optional[str] = None
    maintenance_dose: Optional[str] = None
    max_dose: Optional[str] = None
    route: Optional[str] = None  # "oral", "IV", "SC"
    frequency: Optional[str] = None  # "daily", "weekly", "monthly"


class GuidelineRecommendation(BaseModel):
    """
    A structured clinical recommendation from a guideline.

    Represents a single actionable recommendation with associated
    population, condition, action, and dosing constraints.
    """
    # Identification
    recommendation_id: str
    recommendation_type: RecommendationType = RecommendationType.TREATMENT

    # Target population and condition
    population: str  # "GPA/MPA organ-threatening"
    condition: Optional[str] = None  # "newly diagnosed", "relapsing disease"
    severity: Optional[str] = None  # "severe", "non-severe"

    # Recommended action
    action: str  # "GC + RTX or CYC"
    action_description: Optional[str] = None  # Longer description
    preferred: Optional[str] = None  # "RTX preferred in relapsing disease"
    alternatives: List[str] = Field(default_factory=list)  # ["MTX", "MMF"]

    # Dosing constraints
    dosing: List[DrugDosingInfo] = Field(default_factory=list)
    taper_target: Optional[str] = None  # "5 mg/day by 4-5 months"
    duration: Optional[str] = None  # "24-48 months"
    stop_window: Optional[str] = None  # "6-12 months"

    # Evidence
    evidence_level: EvidenceLevel = EvidenceLevel.UNKNOWN
    strength: RecommendationStrength = RecommendationStrength.UNKNOWN
    references: List[str] = Field(default_factory=list)  # PMID, DOI

    # Provenance
    source: str  # "Table 2", "Figure 1", "Recommendation text"
    source_text: Optional[str] = None  # Original text
    page_num: Optional[int] = None

    def to_summary(self) -> Dict[str, Any]:
        """Generate a summary for export."""
        return {
            "recommendation_id": self.recommendation_id,
            "type": self.recommendation_type.value,
            "population": self.population,
            "action": self.action,
            "preferred": self.preferred,
            "taper_target": self.taper_target,
            "duration": self.duration,
            "evidence_level": self.evidence_level.value,
        }


class RecommendationSet(BaseModel):
    """
    A collection of recommendations from a guideline document.

    Groups related recommendations and provides methods for
    querying and filtering.
    """
    # Identification
    guideline_name: str  # "2022 EULAR recommendations for AAV"
    guideline_year: Optional[int] = None
    organization: Optional[str] = None  # "EULAR", "ACR", "FDA"

    # Target
    target_condition: str  # "ANCA-associated vasculitis"
    target_population: Optional[str] = None

    # Recommendations
    recommendations: List[GuidelineRecommendation] = Field(default_factory=list)

    # Metadata
    source_document: Optional[str] = None
    extraction_confidence: float = 0.0

    def get_by_type(self, rec_type: RecommendationType) -> List[GuidelineRecommendation]:
        """Get recommendations by type."""
        return [r for r in self.recommendations if r.recommendation_type == rec_type]

    def get_by_population(self, population: str) -> List[GuidelineRecommendation]:
        """Get recommendations for a specific population."""
        pop_lower = population.lower()
        return [
            r for r in self.recommendations
            if pop_lower in r.population.lower()
        ]

    def get_treatment_recommendations(self) -> List[GuidelineRecommendation]:
        """Get all treatment recommendations."""
        return self.get_by_type(RecommendationType.TREATMENT)

    def get_dosing_recommendations(self) -> List[GuidelineRecommendation]:
        """Get all dosing recommendations."""
        return self.get_by_type(RecommendationType.DOSING)

    def to_summary(self) -> Dict[str, Any]:
        """Generate a summary for export."""
        return {
            "guideline_name": self.guideline_name,
            "guideline_year": self.guideline_year,
            "target_condition": self.target_condition,
            "recommendation_count": len(self.recommendations),
            "by_type": {
                t.value: len(self.get_by_type(t))
                for t in RecommendationType
                if self.get_by_type(t)
            },
        }


__all__ = [
    "EvidenceLevel",
    "RecommendationStrength",
    "RecommendationType",
    "DrugDosingInfo",
    "GuidelineRecommendation",
    "RecommendationSet",
]
