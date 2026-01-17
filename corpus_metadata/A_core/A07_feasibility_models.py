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

    model_config = ConfigDict(extra="forbid")


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
    operator: str  # ">=", "<=", ">", "<", "=="
    value: float
    unit: str
    specimen: Optional[str] = None  # "first_morning_void", "serum", "plasma"
    timepoints: List[LabTimepoint] = Field(default_factory=list)
    normalization: Optional[EntityNormalization] = None  # LOINC code

    model_config = ConfigDict(extra="forbid")


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
# Screening Yield (CONSORT Flow)
# -------------------------


class ScreenFailReason(BaseModel):
    """Reason for screen failure with count and evidence."""

    reason: str
    count: Optional[int] = None
    percentage: Optional[float] = None
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

    # Derived metrics
    screening_yield: Optional[float] = None  # randomized/screened
    screen_failure_rate: Optional[float] = None
    dropout_rate: Optional[float] = None

    # Screen failure breakdown - critical for feasibility
    screen_fail_reasons: List[ScreenFailReason] = Field(default_factory=list)

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
    """Invasive procedure requirement."""

    name: str  # "renal biopsy", "lumbar puncture", "bone marrow aspirate"
    timing: List[str] = Field(default_factory=list)  # ["screening", "month 6"]
    optional: bool = False
    evidence: List[EvidenceSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class VisitSchedule(BaseModel):
    """Trial visit schedule intensity."""

    total_visits: Optional[int] = None
    visit_days: List[int] = Field(default_factory=list)  # days from randomization
    frequency: Optional[str] = None  # "every 4 weeks", "monthly"
    duration_weeks: Optional[int] = None
    evidence: List[EvidenceSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class BackgroundTherapy(BaseModel):
    """Background/concomitant therapy requirements."""

    therapy_class: str  # "ACEi/ARB", "immunosuppressant"
    requirement: str  # "stable dose ≥90 days", "prohibited"
    agents: List[str] = Field(default_factory=list)
    stable_duration_days: Optional[int] = None
    evidence: List[EvidenceSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class OperationalBurden(BaseModel):
    """Operational burden assessment - major feasibility driver."""

    # Invasive procedures
    invasive_procedures: List[InvasiveProcedure] = Field(default_factory=list)

    # Visit schedule
    visit_schedule: Optional[VisitSchedule] = None

    # Lab/sample handling
    central_lab_required: bool = False
    special_sample_handling: List[str] = Field(default_factory=list)  # "frozen samples", "timed collection"

    # Vaccinations/prophylaxis
    vaccination_requirements: List[VaccinationRequirement] = Field(default_factory=list)

    # Background therapy
    background_therapy: List[BackgroundTherapy] = Field(default_factory=list)

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

    # Confidence
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_features: Optional[Dict[str, float]] = None

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

    # Metadata
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
