# corpus_metadata/A_core/A04_feasibility_models.py
"""
Feasibility-first data models for clinical trial extraction.

These models capture the key signals that drive feasibility assessments:
- Recruitment footprint & screening yield
- Computable eligibility criteria
- Operational burden
- Trial design choices
- Derived feasibility metrics
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# -------------------------
# Evidence (reusable)
# -------------------------


class EvidenceSnippet(BaseModel):
    """Lightweight evidence for feasibility extractions."""

    text: str = Field(..., description="Exact quote from document")
    page: int = Field(..., ge=1, description="1-based page number")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    class Config:
        frozen = True


# -------------------------
# Enums
# -------------------------


class DiagnosisConfirmationType(str, Enum):
    BIOPSY = "biopsy"
    GENETIC_TEST = "genetic_test"
    IMAGING = "imaging"
    CLINICAL = "clinical"
    LAB_BASED = "lab_based"
    PHYSICIAN_DIAGNOSIS = "physician_diagnosis"
    OTHER = "other"


class ProcedureType(str, Enum):
    BIOPSY = "biopsy"
    LUMBAR_PUNCTURE = "lumbar_puncture"
    BONE_MARROW = "bone_marrow"
    ENDOSCOPY = "endoscopy"
    BRONCHOSCOPY = "bronchoscopy"
    IMAGING_MRI = "imaging_mri"
    IMAGING_CT = "imaging_ct"
    IMAGING_PET = "imaging_pet"
    ECHOCARDIOGRAM = "echocardiogram"
    ECG = "ecg"
    BLOOD_DRAW = "blood_draw"
    URINE_COLLECTION = "urine_collection"
    OTHER = "other"


class ComparisonOperator(str, Enum):
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "="
    BETWEEN = "between"
    NOT_EQ = "!="


class TrialPhase(str, Enum):
    PHASE_1 = "phase_1"
    PHASE_1_2 = "phase_1_2"
    PHASE_2 = "phase_2"
    PHASE_2_3 = "phase_2_3"
    PHASE_3 = "phase_3"
    PHASE_4 = "phase_4"
    NOT_APPLICABLE = "not_applicable"


class BlindingType(str, Enum):
    OPEN_LABEL = "open_label"
    SINGLE_BLIND = "single_blind"
    DOUBLE_BLIND = "double_blind"
    TRIPLE_BLIND = "triple_blind"


class ControlType(str, Enum):
    PLACEBO = "placebo"
    ACTIVE_COMPARATOR = "active_comparator"
    STANDARD_OF_CARE = "standard_of_care"
    NO_INTERVENTION = "no_intervention"
    DOSE_COMPARISON = "dose_comparison"


# -------------------------
# Trial Identifiers
# -------------------------


class TrialIdentifiers(BaseModel):
    """Trial registry identifiers."""

    nct_id: Optional[str] = Field(None, pattern=r"^NCT\d{8}$")
    eudract_id: Optional[str] = None
    isrctn_id: Optional[str] = None
    who_id: Optional[str] = None
    sponsor_id: Optional[str] = None

    evidence: List[EvidenceSnippet] = Field(default_factory=list)


# -------------------------
# Recruitment Footprint
# -------------------------


class SiteInfo(BaseModel):
    """Individual site information."""

    name: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    site_type: Optional[str] = None  # academic, community, specialty

    evidence: List[EvidenceSnippet] = Field(default_factory=list)


class RecruitmentFootprint(BaseModel):
    """Geographic and temporal recruitment data."""

    num_countries: Optional[int] = Field(None, ge=0)
    countries: List[str] = Field(default_factory=list)

    num_sites: Optional[int] = Field(None, ge=0)
    sites: List[SiteInfo] = Field(default_factory=list)

    enrollment_start: Optional[date] = None
    enrollment_end: Optional[date] = None
    enrollment_duration_months: Optional[float] = None

    evidence: List[EvidenceSnippet] = Field(default_factory=list)


# -------------------------
# Screening Yield
# -------------------------


class ScreenFailReason(BaseModel):
    """Individual screen failure reason with count."""

    reason: str
    count: Optional[int] = None
    percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    category: Optional[str] = None  # e.g., "lab_threshold", "diagnosis", "consent"

    evidence: List[EvidenceSnippet] = Field(default_factory=list)


class ScreeningYield(BaseModel):
    """CONSORT-style screening and enrollment data."""

    screened: Optional[int] = Field(None, ge=0)
    screen_failed: Optional[int] = Field(None, ge=0)
    randomized: Optional[int] = Field(None, ge=0)
    enrolled: Optional[int] = Field(None, ge=0)  # May differ from randomized
    completed: Optional[int] = Field(None, ge=0)
    discontinued: Optional[int] = Field(None, ge=0)

    # Run-in period
    run_in_completers: Optional[int] = Field(None, ge=0)
    run_in_failures: Optional[int] = Field(None, ge=0)

    # Screen failure breakdown
    screen_fail_reasons: List[ScreenFailReason] = Field(default_factory=list)

    # Derived (computed)
    screening_yield_pct: Optional[float] = Field(None, ge=0.0, le=100.0)

    evidence: List[EvidenceSnippet] = Field(default_factory=list)


# -------------------------
# Computable Eligibility
# -------------------------


class AgeRange(BaseModel):
    """Age eligibility criteria."""

    min_age: Optional[int] = Field(None, ge=0)
    max_age: Optional[int] = Field(None, ge=0)
    unit: str = "years"

    evidence: List[EvidenceSnippet] = Field(default_factory=list)


class DiagnosisRequirement(BaseModel):
    """Diagnosis confirmation requirement."""

    condition: str
    condition_code: Optional[str] = None  # SNOMED/ICD/Orphanet code
    subtype: Optional[str] = None
    confirmation_method: Optional[DiagnosisConfirmationType] = None
    timing: Optional[str] = None  # e.g., "within 12 months"

    evidence: List[EvidenceSnippet] = Field(default_factory=list)


class LabThreshold(BaseModel):
    """Laboratory value threshold."""

    analyte: str
    analyte_code: Optional[str] = None  # LOINC code
    operator: ComparisonOperator
    value: float
    value_upper: Optional[float] = None  # For "between" operator
    unit: str
    timepoints: List[str] = Field(default_factory=list)  # e.g., ["screening", "day -15"]
    num_measurements: Optional[int] = Field(None, ge=1)

    evidence: List[EvidenceSnippet] = Field(default_factory=list)


class MedicationRequirement(BaseModel):
    """Prior/concomitant medication requirements."""

    medication: str
    medication_code: Optional[str] = None  # RxNorm code
    requirement_type: str  # "required", "prohibited", "washout"
    duration: Optional[str] = None  # e.g., "at least 3 months"
    washout_period: Optional[str] = None  # e.g., "4 weeks"
    stable_dose: Optional[bool] = None

    evidence: List[EvidenceSnippet] = Field(default_factory=list)


class ComorbidityExclusion(BaseModel):
    """Excluded comorbidities."""

    condition: str
    condition_code: Optional[str] = None  # SNOMED/ICD code
    timing: Optional[str] = None  # e.g., "within 5 years"

    evidence: List[EvidenceSnippet] = Field(default_factory=list)


class ComputableEligibility(BaseModel):
    """Structured eligibility criteria that can be translated to cohort queries."""

    # Demographics
    age: Optional[AgeRange] = None
    sex: Optional[str] = None  # "male", "female", "all"

    # Diagnosis
    diagnosis_requirements: List[DiagnosisRequirement] = Field(default_factory=list)
    disease_duration: Optional[str] = None  # e.g., "at least 6 months"

    # Laboratory
    lab_thresholds: List[LabThreshold] = Field(default_factory=list)

    # Medications
    medication_requirements: List[MedicationRequirement] = Field(default_factory=list)

    # Exclusions
    comorbidity_exclusions: List[ComorbidityExclusion] = Field(default_factory=list)

    # Other key criteria
    pregnancy_exclusion: bool = True

    # Raw text (for reference)
    inclusion_criteria_text: Optional[str] = None
    exclusion_criteria_text: Optional[str] = None

    evidence: List[EvidenceSnippet] = Field(default_factory=list)


# -------------------------
# Operational Burden
# -------------------------


class VisitSchedule(BaseModel):
    """Visit schedule information."""

    total_visits: Optional[int] = Field(None, ge=0)
    study_duration_weeks: Optional[float] = None
    visit_days: List[int] = Field(default_factory=list)  # Days from randomization
    visit_windows: Optional[str] = None  # e.g., "Day 1, 14, 30, 60, 90, 180"

    evidence: List[EvidenceSnippet] = Field(default_factory=list)


class InvasiveProcedure(BaseModel):
    """Required invasive procedure."""

    procedure: str
    procedure_type: ProcedureType
    timing: List[str] = Field(default_factory=list)  # e.g., ["screening", "month 6"]
    mandatory: bool = True

    evidence: List[EvidenceSnippet] = Field(default_factory=list)


class VaccinationRequirement(BaseModel):
    """Required vaccinations."""

    vaccine: str
    pathogen: Optional[str] = None  # e.g., "Neisseria meningitidis"
    timing: Optional[str] = None  # e.g., "at least 2 weeks before first dose"
    mandatory: bool = True

    evidence: List[EvidenceSnippet] = Field(default_factory=list)


class SpecialSampleRequirement(BaseModel):
    """Special sample handling requirements."""

    sample_type: str  # e.g., "24h urine", "CSF", "bone marrow"
    central_lab: bool = False
    special_handling: Optional[str] = None

    evidence: List[EvidenceSnippet] = Field(default_factory=list)


class OperationalBurden(BaseModel):
    """Site and patient burden factors."""

    # Visit schedule
    visit_schedule: Optional[VisitSchedule] = None

    # Run-in period
    run_in_duration_days: Optional[int] = Field(None, ge=0)
    run_in_requirements: Optional[str] = None

    # Invasive procedures
    invasive_procedures: List[InvasiveProcedure] = Field(default_factory=list)

    # Vaccinations
    vaccination_requirements: List[VaccinationRequirement] = Field(default_factory=list)

    # Special samples
    special_samples: List[SpecialSampleRequirement] = Field(default_factory=list)

    # PRO burden
    patient_reported_outcomes: List[str] = Field(default_factory=list)
    diary_required: bool = False
    wearable_required: bool = False

    # SOC constraints
    background_therapy_constraints: Optional[str] = None

    evidence: List[EvidenceSnippet] = Field(default_factory=list)


# -------------------------
# Trial Design
# -------------------------


class TrialDesign(BaseModel):
    """Trial design characteristics."""

    phase: Optional[TrialPhase] = None
    blinding: Optional[BlindingType] = None
    control_type: Optional[ControlType] = None

    randomization_ratio: Optional[str] = None  # e.g., "1:1", "2:1"
    stratification_factors: List[str] = Field(default_factory=list)

    # Treatment
    treatment_duration_weeks: Optional[float] = None
    open_label_extension: bool = False

    # Sample size
    planned_enrollment: Optional[int] = Field(None, ge=0)
    sample_size_rationale: Optional[str] = None

    # Endpoints
    primary_endpoint: Optional[str] = None
    primary_endpoint_timing: Optional[str] = None

    evidence: List[EvidenceSnippet] = Field(default_factory=list)


# -------------------------
# Derived Feasibility Metrics
# -------------------------


class FeasibilityMetrics(BaseModel):
    """Computed feasibility metrics and flags."""

    # Screening
    screening_yield_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    top_screen_fail_drivers: List[str] = Field(default_factory=list)

    # Hard gates (high-friction criteria)
    hard_gates: List[str] = Field(default_factory=list)

    # Burden scores (0-10 scale)
    patient_burden_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    site_complexity_score: Optional[float] = Field(None, ge=0.0, le=10.0)

    # Flags
    biopsy_required: bool = False
    vaccination_required: bool = False
    central_lab_required: bool = False
    rare_biomarker_required: bool = False
    pediatric_population: bool = False

    # Enrollment
    patients_per_site_per_month: Optional[float] = None
    estimated_enrollment_duration_months: Optional[float] = None


# -------------------------
# Complete Feasibility Profile
# -------------------------


class FeasibilityProfile(BaseModel):
    """
    Complete feasibility extraction from a clinical trial document.

    This is the top-level output schema that captures all feasibility-relevant
    signals in a structured, auditable format.
    """

    # Document metadata
    doc_id: str
    doc_type: Optional[str] = None  # "protocol", "csr", "article", "synopsis"
    extraction_timestamp: Optional[str] = None

    # Core components
    trial_id: TrialIdentifiers = Field(default_factory=TrialIdentifiers)
    footprint: RecruitmentFootprint = Field(default_factory=RecruitmentFootprint)
    screening: ScreeningYield = Field(default_factory=ScreeningYield)
    eligibility: ComputableEligibility = Field(default_factory=ComputableEligibility)
    burden: OperationalBurden = Field(default_factory=OperationalBurden)
    design: TrialDesign = Field(default_factory=TrialDesign)

    # Derived metrics
    feasibility_metrics: FeasibilityMetrics = Field(default_factory=FeasibilityMetrics)

    # Quality indicators
    extraction_completeness: Optional[float] = Field(None, ge=0.0, le=1.0)
    fields_with_evidence: Optional[int] = Field(None, ge=0)
    total_fields_extracted: Optional[int] = Field(None, ge=0)

    def compute_metrics(self) -> None:
        """Compute derived feasibility metrics from raw extractions."""
        metrics = self.feasibility_metrics

        # Screening yield
        if self.screening.screened and self.screening.randomized:
            metrics.screening_yield_pct = (
                self.screening.randomized / self.screening.screened * 100
            )

        # Top screen fail drivers
        if self.screening.screen_fail_reasons:
            sorted_reasons = sorted(
                self.screening.screen_fail_reasons,
                key=lambda x: x.count or 0,
                reverse=True
            )
            metrics.top_screen_fail_drivers = [r.reason for r in sorted_reasons[:5]]

        # Hard gates
        hard_gates = []
        for lab in self.eligibility.lab_thresholds:
            hard_gates.append(f"{lab.analyte} {lab.operator.value} {lab.value} {lab.unit}")
        for diag in self.eligibility.diagnosis_requirements:
            if diag.confirmation_method:
                hard_gates.append(f"{diag.confirmation_method.value} confirmation required")
        metrics.hard_gates = hard_gates[:10]

        # Burden flags
        metrics.biopsy_required = any(
            p.procedure_type == ProcedureType.BIOPSY
            for p in self.burden.invasive_procedures
        )
        metrics.vaccination_required = len(self.burden.vaccination_requirements) > 0
        metrics.central_lab_required = any(
            s.central_lab for s in self.burden.special_samples
        )

        # Patient burden score (simplified)
        score = 0.0
        if self.burden.visit_schedule:
            visits = self.burden.visit_schedule.total_visits or 0
            score += min(visits / 20, 2.0)  # Max 2 points for visits
        score += len(self.burden.invasive_procedures) * 1.5
        score += len(self.burden.vaccination_requirements) * 0.5
        if self.burden.diary_required:
            score += 1.0
        if self.burden.wearable_required:
            score += 0.5
        metrics.patient_burden_score = min(score, 10.0)

        # Site complexity score
        site_score = 0.0
        site_score += len(self.burden.special_samples) * 1.0
        if metrics.central_lab_required:
            site_score += 2.0
        site_score += len(self.burden.invasive_procedures) * 1.0
        site_score += len(self.burden.vaccination_requirements) * 0.5
        metrics.site_complexity_score = min(site_score, 10.0)

        # Pediatric flag
        if self.eligibility.age:
            if self.eligibility.age.min_age is not None and self.eligibility.age.min_age < 18:
                metrics.pediatric_population = True
