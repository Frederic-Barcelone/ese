# corpus_metadata/corpus_metadata/A_core/A07_feasibility_models.py
"""
Domain models for clinical trial feasibility information extraction.

Covers:
- Patient journey phases (screening → randomization → treatment → follow-up)
- Epidemiology data (prevalence, incidence, demographics)
- Eligibility criteria (inclusion/exclusion)
- Study design parameters (endpoints, sites, duration)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# -------------------------
# Feasibility-specific enums
# -------------------------


class FeasibilityFieldType(str, Enum):
    """Type of feasibility information extracted."""

    ELIGIBILITY_INCLUSION = "ELIGIBILITY_INCLUSION"
    ELIGIBILITY_EXCLUSION = "ELIGIBILITY_EXCLUSION"
    EPIDEMIOLOGY_PREVALENCE = "EPIDEMIOLOGY_PREVALENCE"
    EPIDEMIOLOGY_INCIDENCE = "EPIDEMIOLOGY_INCIDENCE"
    EPIDEMIOLOGY_DEMOGRAPHICS = "EPIDEMIOLOGY_DEMOGRAPHICS"
    PATIENT_JOURNEY_PHASE = "PATIENT_JOURNEY_PHASE"
    STUDY_ENDPOINT = "STUDY_ENDPOINT"
    STUDY_SITE = "STUDY_SITE"
    STUDY_DESIGN = "STUDY_DESIGN"
    STUDY_DURATION = "STUDY_DURATION"
    TREATMENT_PATHWAY = "TREATMENT_PATHWAY"


class FeasibilityGeneratorType(str, Enum):
    """Tracks which strategy produced the feasibility candidate."""

    PATTERN_MATCH = "gen:feasibility_pattern"
    SECTION_PARSER = "gen:feasibility_section"
    NER_EXTRACTION = "gen:feasibility_ner"
    LLM_EXTRACTION = "gen:feasibility_llm"


class CriterionType(str, Enum):
    """Eligibility criterion type."""

    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"


class PatientJourneyPhaseType(str, Enum):
    """Standard clinical trial phases."""

    SCREENING = "screening"
    RUN_IN = "run_in"
    RANDOMIZATION = "randomization"
    TREATMENT = "treatment"
    FOLLOW_UP = "follow_up"
    EXTENSION = "extension"


class EndpointType(str, Enum):
    """Clinical endpoint types."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    EXPLORATORY = "exploratory"
    SAFETY = "safety"


# -------------------------
# Eligibility Criteria
# -------------------------


class EligibilityCriterion(BaseModel):
    """Single inclusion or exclusion criterion."""

    criterion_type: CriterionType
    text: str  # Full criterion text
    category: Optional[str] = None  # e.g., "age", "disease_definition", "biomarker"
    is_negated: bool = False  # True if criterion contains negation
    parsed_value: Optional[Dict[str, Any]] = None  # Structured extraction
    derived_variables: Optional[Dict[str, Any]] = None  # ML-ready: age_min, pediatric_allowed

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Epidemiology Data
# -------------------------


class EpidemiologyData(BaseModel):
    """Epidemiology statistics for a condition."""

    data_type: str  # "prevalence", "incidence", "mortality", "demographics"
    value: str  # e.g., "1-2 per million", "3.5%", "median age 45"
    normalized_value: Optional[float] = None  # Per-million standardized
    unit: Optional[str] = None  # "per 100,000", "percent"
    geography: Optional[str] = None  # Country/region
    time_period: Optional[str] = None  # Year or range
    setting: Optional[str] = None  # "population-based", "registry"
    population: Optional[str] = None  # e.g., "adults", "children", "US population"
    source: Optional[str] = None  # e.g., "Orphanet", "CDC", study citation
    year: Optional[str] = None  # Year of data

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Patient Journey Phase
# -------------------------


class PatientJourneyPhase(BaseModel):
    """A phase in the patient journey through the trial."""

    phase_type: PatientJourneyPhaseType
    description: str  # What happens in this phase
    duration: Optional[str] = None  # e.g., "4 weeks", "up to 12 weeks"
    visits: Optional[int] = None  # Number of visits
    visit_frequency: Optional[str] = None  # "every 4 weeks"
    procedures: List[str] = Field(default_factory=list)  # Procedures/assessments
    inpatient_days: Optional[int] = None  # Days of hospitalization required

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Study Endpoint
# -------------------------


class StudyEndpoint(BaseModel):
    """Clinical trial endpoint."""

    endpoint_type: EndpointType
    name: str  # e.g., "Proteinuria reduction"
    measure: Optional[str] = None  # What is measured (e.g., "24-hour UPCR")
    timepoint: Optional[str] = None  # e.g., "6 months", "Week 52"
    analysis_method: Optional[str] = None  # e.g., "log-transformed ratio to baseline"

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Study Design (NEW)
# -------------------------


class TreatmentArm(BaseModel):
    """Treatment arm in a clinical trial."""

    name: str  # e.g., "Iptacopan", "Placebo"
    dose: Optional[str] = None  # e.g., "200mg"
    frequency: Optional[str] = None  # e.g., "twice daily"
    route: Optional[str] = None  # e.g., "oral"

    model_config = ConfigDict(extra="forbid")


class StudyDesign(BaseModel):
    """Clinical trial study design parameters."""

    phase: Optional[str] = None  # e.g., "2", "3", "2/3"
    design_type: Optional[str] = None  # e.g., "parallel", "crossover", "single-arm"
    blinding: Optional[str] = None  # e.g., "double-blind", "open-label"
    randomization_ratio: Optional[str] = None  # e.g., "1:1", "2:1"
    sample_size: Optional[int] = None  # Planned enrollment
    actual_enrollment: Optional[int] = None  # Actual enrollment
    duration_months: Optional[int] = None  # Total study duration
    treatment_arms: List[TreatmentArm] = Field(default_factory=list)
    control_type: Optional[str] = None  # e.g., "placebo", "active", "standard of care"

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Study Site Information
# -------------------------


class StudySite(BaseModel):
    """Clinical trial site information."""

    country: str
    country_code: Optional[str] = None  # ISO code
    region: Optional[str] = None  # e.g., "Europe", "North America"
    site_count: Optional[int] = None
    site_type: Optional[str] = None  # "academic", "community"
    enrollment_target: Optional[int] = None
    validation_context: Optional[str] = None  # Why we trust this extraction

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Feasibility Provenance
# -------------------------


class FeasibilityProvenanceMetadata(BaseModel):
    """Provenance metadata for feasibility extraction."""

    pipeline_version: str
    run_id: str
    doc_fingerprint: str
    generator_name: FeasibilityGeneratorType
    rule_version: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True, extra="forbid")


# -------------------------
# FeasibilityCandidate (main extraction unit)
# -------------------------


class FeasibilityCandidate(BaseModel):
    """
    Pre-validation feasibility information candidate.

    Contains extracted information about clinical trial feasibility:
    - Eligibility criteria
    - Epidemiology data
    - Patient journey phases
    - Study design parameters
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    doc_id: str

    # What type of feasibility info
    field_type: FeasibilityFieldType
    generator_type: FeasibilityGeneratorType

    # The extracted text
    matched_text: str
    context_text: str

    # Location in document
    page_number: Optional[int] = None
    section_name: Optional[str] = None  # e.g., "Methods", "Eligibility", "Study Design"

    # Structured data (one of these will be populated based on field_type)
    eligibility_criterion: Optional[EligibilityCriterion] = None
    epidemiology_data: Optional[EpidemiologyData] = None
    patient_journey_phase: Optional[PatientJourneyPhase] = None
    study_endpoint: Optional[StudyEndpoint] = None
    study_site: Optional[StudySite] = None
    study_design: Optional[StudyDesign] = None

    # Confidence
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_features: Optional[Dict[str, float]] = None  # For debugging/calibration

    provenance: FeasibilityProvenanceMetadata

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Export Models
# -------------------------


class FeasibilityExportEntry(BaseModel):
    """Single feasibility entry for export."""

    field_type: str
    text: str
    section: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None
    confidence: float


class FeasibilityExportDocument(BaseModel):
    """All feasibility data for a document."""

    doc_id: str
    doc_filename: str

    # Study design (single structured object)
    study_design: Optional[Dict[str, Any]] = None

    # Grouped by type
    eligibility_inclusion: List[FeasibilityExportEntry] = Field(default_factory=list)
    eligibility_exclusion: List[FeasibilityExportEntry] = Field(default_factory=list)
    epidemiology: List[FeasibilityExportEntry] = Field(default_factory=list)
    patient_journey: List[FeasibilityExportEntry] = Field(default_factory=list)
    endpoints: List[FeasibilityExportEntry] = Field(default_factory=list)
    sites: List[FeasibilityExportEntry] = Field(default_factory=list)

    # Metadata
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
