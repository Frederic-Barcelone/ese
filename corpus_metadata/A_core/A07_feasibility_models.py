# corpus_metadata/A_core/A07_feasibility_models.py
"""
Pydantic domain models for clinical trial feasibility information extraction.

This module provides comprehensive data structures for extracting feasibility-critical
information from clinical trial protocols and publications. It models the full spectrum
of feasibility data: eligibility criteria with logical expressions, CONSORT screening
flows, operational burden (visits, procedures, vaccinations), epidemiology statistics,
study design parameters, and geographic footprint.

Key Components:
    - FeasibilityCandidate: Pre-validation feasibility data with structured sub-types
    - EligibilityCriterion: Inclusion/exclusion criterion with lab values and severity grades
    - ScreeningFlow: CONSORT flow metrics with screen failure reasons
    - OperationalBurden: Visit schedules, invasive procedures, vaccination requirements
    - StudyDesign: Phase, blinding, randomization, treatment arms
    - EpidemiologyData: Prevalence, incidence, demographics statistics
    - PatientJourneyPhase: Screening, run-in, treatment, follow-up phases
    - StudyFootprint: Geographic distribution of study sites
    - TrialIdentifier: NCT, EudraCT, CTIS registry identifiers

Example:
    >>> from A_core.A07_feasibility_models import EligibilityCriterion, CriterionType
    >>> criterion = EligibilityCriterion(
    ...     criterion_type=CriterionType.INCLUSION,
    ...     text="Age >= 18 years",
    ...     category="age",
    ...     parsed_value={"min_age": 18, "unit": "years"},
    ... )

Dependencies:
    - A_core.A21_clinical_criteria: For LabCriterion, SeverityGrade, DiagnosisConfirmation
    - A_core.A22_logical_expressions: For LogicalExpression, CriterionNode, LogicalOperator
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# Re-exported from A07a_clinical_criteria and A07b_logical_expressions for backward compatibility
__all__ = [
    "DiagnosisConfirmation",
    "EntityNormalization",
    "LabCriterion",
    "LabTimepoint",
    "SeverityGrade",
    "SeverityGradeType",
    "SEVERITY_GRADE_MAPPINGS",
    # Logical expressions (from A07b)
    "LogicalOperator",
    "CriterionNode",
    "LogicalExpression",
]


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
    STUDY_FOOTPRINT = "STUDY_FOOTPRINT"
    STUDY_DESIGN = "STUDY_DESIGN"
    STUDY_DURATION = "STUDY_DURATION"
    TREATMENT_PATHWAY = "TREATMENT_PATHWAY"
    SCREENING_YIELD = "SCREENING_YIELD"
    SCREENING_FLOW = "SCREENING_FLOW"
    VACCINATION_REQUIREMENT = "VACCINATION_REQUIREMENT"
    OPERATIONAL_BURDEN = "OPERATIONAL_BURDEN"
    INVASIVE_PROCEDURE = "INVASIVE_PROCEDURE"
    BACKGROUND_THERAPY = "BACKGROUND_THERAPY"
    LAB_CRITERION = "LAB_CRITERION"
    PATIENT_POPULATION = "PATIENT_POPULATION"
    LOCAL_GUIDELINES = "LOCAL_GUIDELINES"
    PATIENT_JOURNEY = "PATIENT_JOURNEY"


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


class ExtractionMethod(str, Enum):
    """How the value was extracted."""

    REGEX = "regex"
    TABLE = "table"
    LLM = "llm"
    HYBRID = "hybrid"


class EligibilityCategory(str, Enum):
    """Refined eligibility criterion categories for computability."""

    AGE = "age"
    DISEASE_DEFINITION = "disease_definition"
    DISEASE_SEVERITY = "disease_severity"
    DISEASE_DURATION = "disease_duration"
    DIAGNOSIS_CONFIRMATION = "diagnosis_confirmation"
    HISTOPATHOLOGY = "histopathology"
    PRIOR_TREATMENT = "prior_treatment"
    CONCOMITANT_MEDICATIONS = "concomitant_medications"
    BACKGROUND_THERAPY = "background_therapy"
    LAB_VALUE = "lab_value"
    BIOMARKER = "biomarker"
    ORGAN_FUNCTION = "organ_function"
    COMORBIDITY = "comorbidity"
    VACCINATION_REQUIREMENT = "vaccination_requirement"
    PROPHYLAXIS_REQUIREMENT = "prophylaxis_requirement"
    PREGNANCY = "pregnancy"
    CONSENT = "consent"
    ADMINISTRATIVE = "administrative"
    HISTOPATHOLOGY_EXCLUSION = "histopathology_exclusion"
    RUN_IN_REQUIREMENT = "run_in_requirement"


# -------------------------
# Evidence Span (Provenance)
# -------------------------


class EvidenceSpan(BaseModel):
    """Evidence supporting an extracted value - critical for auditability."""

    page: Optional[int] = None
    section_header: Optional[str] = None
    quote: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    bbox: Optional[List[float]] = None  # [x0, y0, x1, y1]
    source_node_id: Optional[str] = None  # doc-graph node reference
    source_doc_id: Optional[str] = None  # document ID for cross-paper verification

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Clinical Criteria (imported from A21_clinical_criteria)
# -------------------------

from A_core.A21_clinical_criteria import (
    DiagnosisConfirmation,
    EntityNormalization,
    LabCriterion,
    LabTimepoint,
    SeverityGrade,
    SeverityGradeType,
    SEVERITY_GRADE_MAPPINGS,
)

# -------------------------
# Logical Expression Models (imported from A22_logical_expressions)
# -------------------------

from A_core.A22_logical_expressions import (
    CriterionNode,
    LogicalExpression,
    LogicalOperator,
)


# -------------------------
# Eligibility Criteria
# -------------------------


class EligibilityCriterion(BaseModel):
    """Single inclusion or exclusion criterion with full provenance."""

    criterion_type: CriterionType
    text: str
    category: Optional[str] = None
    is_negated: bool = False
    parsed_value: Optional[Dict[str, Any]] = None
    derived_variables: Optional[Dict[str, Any]] = None

    # Structured sub-types for computability
    lab_criterion: Optional[LabCriterion] = None
    diagnosis_confirmation: Optional[DiagnosisConfirmation] = None
    severity_grade: Optional[SeverityGrade] = None  # NYHA, ECOG, CKD staging

    # Logical structure (for complex criteria with AND/OR)
    logical_expression: Optional[LogicalExpression] = None

    # Evidence/provenance
    evidence: List[EvidenceSpan] = Field(default_factory=list)
    extraction_method: Optional[ExtractionMethod] = None

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
    n: Optional[int] = None  # number randomized to this arm
    dose: Optional[str] = None  # e.g., "200mg"
    frequency: Optional[str] = None  # e.g., "twice daily"
    route: Optional[str] = None  # e.g., "oral"

    model_config = ConfigDict(extra="forbid")


class StudyPeriod(BaseModel):
    """A distinct period/phase within the study design."""

    name: str  # e.g., "double_blind", "open_label", "run_in", "follow_up"
    duration_months: Optional[float] = None
    duration_weeks: Optional[int] = None
    description: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class StudyDesign(BaseModel):
    """Clinical trial study design parameters."""

    phase: Optional[str] = None  # e.g., "2", "3", "2/3"
    design_type: Optional[str] = None  # e.g., "parallel", "crossover", "single-arm"
    blinding: Optional[str] = None  # e.g., "double-blind", "open-label"
    randomization_ratio: Optional[str] = None  # e.g., "1:1", "2:1"
    allocation: Optional[str] = None  # e.g., "iptacopan n=38, placebo n=36"
    sample_size: Optional[int] = None  # Planned enrollment
    actual_enrollment: Optional[int] = None  # Actual enrollment
    duration_months: Optional[int] = None  # Total study duration
    treatment_arms: List[TreatmentArm] = Field(default_factory=list)
    control_type: Optional[str] = None  # e.g., "placebo", "active", "standard of care"

    # Study setting
    setting: Optional[str] = None  # e.g., "35 hospitals/medical centres in 18 countries"
    sites_total: Optional[int] = None
    countries_total: Optional[int] = None

    # Study periods (for complex designs with multiple phases)
    periods: List[StudyPeriod] = Field(default_factory=list)

    # Evidence
    evidence: List[EvidenceSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Screening Yield (CONSORT Flow)
# -------------------------


class ScreenFailReason(BaseModel):
    """Reason for screen failure with count and evidence."""

    reason: str
    count: Optional[int] = None
    # Dual percentage tracking for numeric hygiene
    percentage_reported: Optional[float] = None  # What the document explicitly states
    percentage_computed: Optional[float] = None  # Calculated from counts
    # Overlap flag - critical when participants can fail multiple criteria
    can_overlap: bool = False  # True if this reason can co-occur with others
    evidence: List[EvidenceSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ScreeningFlow(BaseModel):
    """Complete CONSORT flow with screen fail reasons - key for feasibility."""

    # Planned vs actual
    planned_sample_size: Optional[int] = None
    actual_enrollment: Optional[int] = None

    # CONSORT flow numbers
    screened: Optional[int] = None
    screen_failures: Optional[int] = None
    randomized: Optional[int] = None
    treated: Optional[int] = None
    completed: Optional[int] = None
    discontinued: Optional[int] = None

    # Derived metrics (computed)
    screening_yield: Optional[float] = None  # randomized/screened
    screen_failure_rate: Optional[float] = None  # Legacy - computed value
    dropout_rate: Optional[float] = None

    # Numeric hygiene: distinguish reported vs computed
    screen_failure_rate_reported: Optional[float] = None  # What document states (e.g., "44%")
    screen_failure_rate_computed: Optional[float] = None  # Calculated: failures/screened

    # Screen failure breakdown - critical for feasibility
    screen_fail_reasons: List[ScreenFailReason] = Field(default_factory=list)
    # Note: screen fail reasons may overlap (participants can fail multiple criteria)
    reasons_can_overlap: bool = False  # Set True if document indicates overlap

    # Run-in failures (separate from screen failures)
    run_in_failures: Optional[int] = None
    run_in_failure_reasons: List[ScreenFailReason] = Field(default_factory=list)

    # Evidence
    evidence: List[EvidenceSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ScreeningYield(BaseModel):
    """CONSORT flow metrics for screening and enrollment (legacy, use ScreeningFlow)."""

    screened: Optional[int] = None
    screen_failures: Optional[int] = None
    randomized: Optional[int] = None
    enrolled: Optional[int] = None
    completed: Optional[int] = None
    discontinued: Optional[int] = None
    screen_failure_rate: Optional[float] = None
    dropout_rate: Optional[float] = None

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Vaccination Requirement
# -------------------------


class VaccinationRequirement(BaseModel):
    """Vaccination requirement for trial eligibility."""

    vaccine_type: str
    requirement_type: str  # "required", "prohibited", "completed_before"
    timing: Optional[str] = None  # "at least 4 weeks before", "within 6 months"
    agents: List[str] = Field(default_factory=list)  # specific vaccines
    evidence: List[EvidenceSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Operational Burden (Key Feasibility Driver)
# -------------------------


class InvasiveProcedure(BaseModel):
    """Invasive procedure requirement - distinguishes eligibility vs study procedures."""

    name: str  # "renal biopsy", "lumbar puncture", "bone marrow aspirate"
    timing: List[str] = Field(default_factory=list)  # ["screening", "month 6"]
    timing_days: List[int] = Field(default_factory=list)  # [45, 180] - days from randomization
    optional: bool = False
    # Purpose distinction - critical for separating eligibility from operational burden
    purpose: Optional[str] = None  # "diagnosis_confirmation", "efficacy_assessment", "safety_monitoring"
    is_eligibility_requirement: bool = False  # True if this confirms prior diagnosis (not study procedure)
    evidence: List[EvidenceSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ScheduledVisit(BaseModel):
    """A single scheduled visit with phase context."""

    day: int  # relative to randomization (negative = pre-randomization)
    visit_name: Optional[str] = None  # e.g., "Screening", "Week 2", "Month 6"
    phase: Optional[str] = None  # "pre_randomization", "double_blind", "open_label", "follow_up"
    procedures: List[str] = Field(default_factory=list)  # what happens at this visit

    model_config = ConfigDict(extra="forbid")


class VisitSchedule(BaseModel):
    """Trial visit schedule intensity with phase-structured visits."""

    total_visits: Optional[int] = None
    # Legacy flat list - kept for backward compatibility
    visit_days: List[int] = Field(default_factory=list)  # days from randomization
    frequency: Optional[str] = None  # "every 4 weeks", "monthly"
    duration_weeks: Optional[int] = None

    # Structured visits with phase information (preferred)
    scheduled_visits: List[ScheduledVisit] = Field(default_factory=list)

    # Phase-grouped visit days (alternative structure)
    pre_randomization_days: List[int] = Field(default_factory=list)  # e.g., [-75, -15, 1]
    on_treatment_days: List[int] = Field(default_factory=list)  # e.g., [14, 30, 90, 180]
    follow_up_days: List[int] = Field(default_factory=list)

    evidence: List[EvidenceSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class BackgroundTherapy(BaseModel):
    """Background/concomitant therapy requirements - critical for feasibility."""

    therapy_class: str  # "ACEi/ARB", "immunosuppressant", "SGLT2 inhibitors"
    requirement_type: str  # "allowed", "required", "prohibited"
    requirement: str  # "stable dose ≥90 days", "prohibited", "required"
    agents: List[str] = Field(default_factory=list)
    stable_duration_days: Optional[int] = None
    max_dose: Optional[str] = None  # e.g., "≤7.5 mg prednisone equivalent"
    evidence: List[EvidenceSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class CentralLabRequirement(BaseModel):
    """Central laboratory requirement with evidence."""

    required: bool = False
    analytes: List[str] = Field(default_factory=list)  # ["UPCR", "eGFR", "C3", "sC5b-9"]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: List[EvidenceSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class OperationalBurden(BaseModel):
    """Operational burden assessment - major feasibility driver."""

    # Invasive procedures (study-required, not eligibility confirmation)
    invasive_procedures: List[InvasiveProcedure] = Field(default_factory=list)

    # Visit schedule
    visit_schedule: Optional[VisitSchedule] = None

    # Lab/sample handling - enhanced with evidence
    central_lab_required: bool = False  # Legacy boolean for backward compatibility
    central_lab: Optional[CentralLabRequirement] = None  # Rich model with evidence
    special_sample_handling: List[str] = Field(default_factory=list)  # "frozen samples", "timed collection"

    # Vaccinations/prophylaxis
    vaccination_requirements: List[VaccinationRequirement] = Field(default_factory=list)

    # Background/concomitant therapy (legacy field)
    background_therapy: List[BackgroundTherapy] = Field(default_factory=list)

    # Concomitant medications allowed (feasibility-critical)
    concomitant_meds_allowed: List[BackgroundTherapy] = Field(default_factory=list)

    # Run-in requirements
    run_in_duration_days: Optional[int] = None
    run_in_requirements: List[str] = Field(default_factory=list)

    # Derived scores (heuristic)
    burden_score: Optional[float] = None  # 0-10 composite
    site_complexity_score: Optional[float] = None

    # Hard gates (criteria most likely to limit enrollment)
    hard_gates: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Study Footprint (Reshaped Sites)
# -------------------------


class StudyFootprint(BaseModel):
    """Study geographic footprint - consolidated site information."""

    sites_total: Optional[int] = None
    countries_total: Optional[int] = None
    countries: List[str] = Field(default_factory=list)  # ISO codes or names
    regions: List[str] = Field(default_factory=list)  # "North America", "Europe"
    sites_by_country: Optional[Dict[str, int]] = None  # only if counts available
    evidence: List[EvidenceSpan] = Field(default_factory=list)

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
# Trial Identifiers
# -------------------------


class TrialIdentifier(BaseModel):
    """Clinical trial registry identifier."""

    id_type: str  # "NCT", "EudraCT", "CTIS", "ISRCTN", "ACTRN", etc.
    value: str  # e.g., "NCT04817618", "2020-001234-56"
    registry: Optional[str] = None  # e.g., "ClinicalTrials.gov", "EU CTIS"
    url: Optional[str] = None  # Direct link to registry entry
    title: Optional[str] = None  # Trial title if available

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
# Patient Population (Feasibility)
# -------------------------


class PatientPopulation(BaseModel):
    """Patient population data for feasibility assessment."""

    estimated_diagnosed_patients: Optional[int] = None
    estimated_eligible_patients: Optional[int] = None
    eligibility_funnel_ratio: Optional[float] = None  # eligible/diagnosed
    registry_name: Optional[str] = None
    registry_size: Optional[int] = None
    diagnostic_delay_years: Optional[float] = None
    referral_centres: Optional[int] = None
    referral_centre_names: List[str] = Field(default_factory=list)
    geographic_distribution: Optional[str] = None
    recruitment_rate_per_site_month: Optional[float] = None
    evidence: List[EvidenceSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Local Guidelines (Feasibility)
# -------------------------


class LocalGuideline(BaseModel):
    """Local/national clinical guideline reference for feasibility context."""

    guideline_name: str
    issuing_body: Optional[str] = None
    year: Optional[int] = None
    country: Optional[str] = None
    key_recommendations: List[str] = Field(default_factory=list)
    standard_of_care: Optional[str] = None
    impact_on_feasibility: Optional[str] = None
    evidence: List[EvidenceSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Patient Journey (Comprehensive)
# -------------------------


class DiagnosticPathway(BaseModel):
    """Diagnostic pathway for a disease."""

    diagnostic_delay_years: Optional[float] = None
    diagnostic_tests_required: List[str] = Field(default_factory=list)
    specialist_type: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class TreatmentLine(BaseModel):
    """A line of therapy in the treatment pathway."""

    line: int
    therapy: str

    model_config = ConfigDict(extra="forbid")


class TreatmentPathway(BaseModel):
    """Treatment pathway for a disease."""

    current_standard_of_care: Optional[str] = None
    treatment_lines: List[TreatmentLine] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class TrialPhase(BaseModel):
    """A phase in the clinical trial journey."""

    phase: str  # "screening", "treatment", "follow_up"
    duration: Optional[str] = None
    visits: Optional[int] = None
    visit_frequency: Optional[str] = None
    procedures: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class PatientJourney(BaseModel):
    """Comprehensive patient journey through diagnosis, treatment, and trial participation."""

    country: Optional[str] = None  # Country context (e.g., "Germany", "Japan")
    region: Optional[str] = None  # Region context (e.g., "Europe", "Asia-Pacific")
    diagnostic_pathway: Optional[DiagnosticPathway] = None
    treatment_pathway: Optional[TreatmentPathway] = None
    trial_phases: List[TrialPhase] = Field(default_factory=list)
    participation_barriers: List[str] = Field(default_factory=list)
    evidence: List[EvidenceSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Patient Funnel (Computed)
# -------------------------


class PatientFunnel(BaseModel):
    """Complete patient funnel from population to completion (computed post-extraction)."""

    # Population level (from epidemiology)
    disease_prevalence: Optional[str] = None
    prevalence_per_million: Optional[float] = None

    # Diagnosed level (from patient_population)
    estimated_diagnosed: Optional[int] = None
    diagnostic_delay_years: Optional[float] = None

    # Eligible level (from patient_population + eligibility)
    estimated_eligible: Optional[int] = None
    eligibility_funnel_ratio: Optional[float] = None
    key_eligibility_gates: List[str] = Field(default_factory=list)

    # Screening level (from screening_flow)
    screened: Optional[int] = None
    screen_failure_rate: Optional[float] = None
    top_screen_failure_reasons: List[str] = Field(default_factory=list)

    # Enrollment level
    randomized: Optional[int] = None
    enrollment_rate_per_site_month: Optional[float] = None

    # Completion level
    completed: Optional[int] = None
    discontinuation_rate: Optional[float] = None

    # Computed metrics
    overall_yield: Optional[float] = None  # completed/screened
    funnel_completeness: float = 0.0  # fraction of stages populated (0-1)

    model_config = ConfigDict(extra="forbid")


# -------------------------
# FeasibilityCandidate (main extraction unit)
# -------------------------


class FeasibilityCandidate(BaseModel):
    """
    Pre-validation feasibility information candidate.

    Contains extracted information about clinical trial feasibility:
    - Eligibility criteria (with lab criteria, diagnosis confirmation)
    - Epidemiology data
    - Patient journey phases
    - Study design parameters
    - Screening flow (CONSORT with screen fail reasons)
    - Operational burden (procedures, visits, vaccines)
    - Study footprint (sites/countries consolidated)
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
    section_name: Optional[str] = None

    # Evidence/provenance - CRITICAL for auditability
    evidence: List[EvidenceSpan] = Field(default_factory=list)
    extraction_method: Optional[ExtractionMethod] = None

    # Structured data (one of these will be populated based on field_type)
    eligibility_criterion: Optional[EligibilityCriterion] = None
    epidemiology_data: Optional[EpidemiologyData] = None
    patient_journey_phase: Optional[PatientJourneyPhase] = None
    study_endpoint: Optional[StudyEndpoint] = None
    study_site: Optional[StudySite] = None
    study_footprint: Optional[StudyFootprint] = None
    study_design: Optional[StudyDesign] = None
    screening_yield: Optional[ScreeningYield] = None
    screening_flow: Optional[ScreeningFlow] = None
    vaccination_requirement: Optional[VaccinationRequirement] = None
    operational_burden: Optional[OperationalBurden] = None
    lab_criterion: Optional[LabCriterion] = None
    background_therapy: Optional[BackgroundTherapy] = None
    invasive_procedure: Optional[InvasiveProcedure] = None
    patient_population: Optional[PatientPopulation] = None
    local_guideline: Optional[LocalGuideline] = None
    patient_journey: Optional[PatientJourney] = None

    # Confidence
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_features: Optional[Dict[str, float]] = None

    provenance: FeasibilityProvenanceMetadata

    model_config = ConfigDict(extra="forbid")


# -------------------------
# NER Candidate (Simplified)
# -------------------------


class NERCandidate(BaseModel):
    """
    Simplified candidate for NER-extracted entities.

    Used by NER enrichers (EpiExtract4GARD, ZeroShotBioNER, BiomedicalNER,
    PatientJourneyNER) that extract entities without full feasibility context.

    These are later routed to the unified schema via candidates_to_unified_schema().
    """

    category: str  # e.g., "epidemiology", "adverse_event", "diagnostic_delay"
    text: str
    evidence_text: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str  # e.g., "EpiExtract4GARD-v2", "ZeroShotBioNER", "PatientJourneyNER"

    # Optional structured data
    epidemiology_data: Optional[EpidemiologyData] = None
    entity_type: Optional[str] = None  # Original entity type from NER model

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Export Models
# -------------------------


class EvidenceExport(BaseModel):
    """Simplified evidence for export - what reviewers need to trust extraction."""

    page: Optional[int] = None
    quote: Optional[str] = None
    source_node_id: Optional[str] = None
    source_doc_id: Optional[str] = None  # document ID for cross-paper verification

    model_config = ConfigDict(extra="forbid")


class FeasibilityExportEntry(BaseModel):
    """Single feasibility entry for export."""

    field_type: str
    text: str
    section: Optional[str] = None
    page: Optional[int] = None  # Page number for quick reference
    structured_data: Optional[Dict[str, Any]] = None
    confidence: float
    evidence: List[EvidenceExport] = Field(default_factory=list)  # Supporting quotes


class FeasibilityExportDocument(BaseModel):
    """All feasibility data for a document."""

    doc_id: str
    doc_filename: str

    # Trial identifiers (NCT, EudraCT, CTIS, etc.)
    trial_ids: List[TrialIdentifier] = Field(default_factory=list)

    # Study design (single structured object)
    study_design: Optional[Dict[str, Any]] = None

    # Grouped by type
    eligibility_inclusion: List[FeasibilityExportEntry] = Field(default_factory=list)
    eligibility_exclusion: List[FeasibilityExportEntry] = Field(default_factory=list)
    epidemiology: List[FeasibilityExportEntry] = Field(default_factory=list)
    patient_journey: List[FeasibilityExportEntry] = Field(default_factory=list)
    endpoints: List[FeasibilityExportEntry] = Field(default_factory=list)
    sites: List[FeasibilityExportEntry] = Field(default_factory=list)

    # Operational feasibility data (single structured objects)
    operational_burden: Optional[Dict[str, Any]] = None
    screening_flow: Optional[Dict[str, Any]] = None

    # Patient population, journey, and guidelines
    patient_population: Optional[Dict[str, Any]] = None
    patient_journey_data: Optional[Dict[str, Any]] = None
    local_guidelines: List[Dict[str, Any]] = Field(default_factory=list)

    # Patient funnel (computed from other fields)
    patient_funnel: Optional[Dict[str, Any]] = None

    # Metadata
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
