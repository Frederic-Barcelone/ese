# corpus_metadata/A_core/A09_unified_feasibility_schema.py
"""
Unified feasibility output schema.

Consolidates outputs from multiple NER enrichers into a clean,
structured format for downstream consumption.

Sources:
- LLM/Pattern extraction: Eligibility criteria, study design
- EpiExtract4GARD-v2: Epidemiology (prevalence, incidence, geography)
- ZeroShotBioNER: Drug administration (ADE, dosage, frequency, route)
- BiomedicalNER: Clinical signals (symptoms, procedures, labs, demographics)
- PatientJourneyEnricher: Patient journey (diagnostic delay, treatment lines, care pathway)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# -------------------------
# Span Models
# -------------------------


class ExtractedSpan(BaseModel):
    """Base model for an extracted text span."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str  # e.g., "EpiExtract4GARD-v2", "ZeroShotBioNER"
    entity_type: Optional[str] = None  # Original entity type
    start: Optional[int] = None
    end: Optional[int] = None

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Epidemiology Section
# -------------------------


class EpidemiologyOutput(BaseModel):
    """Epidemiology data from EpiExtract4GARD-v2."""

    prevalence: List[ExtractedSpan] = Field(default_factory=list)
    incidence: List[ExtractedSpan] = Field(default_factory=list)
    mortality: List[ExtractedSpan] = Field(default_factory=list)
    geography: List[ExtractedSpan] = Field(default_factory=list)
    statistics: List[ExtractedSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Drug Administration Section
# -------------------------


class DrugAdminOutput(BaseModel):
    """Drug administration data from ZeroShotBioNER."""

    adverse_events: List[ExtractedSpan] = Field(default_factory=list)
    dosages: List[ExtractedSpan] = Field(default_factory=list)
    frequencies: List[ExtractedSpan] = Field(default_factory=list)
    routes: List[ExtractedSpan] = Field(default_factory=list)
    durations: List[ExtractedSpan] = Field(default_factory=list)
    strengths: List[ExtractedSpan] = Field(default_factory=list)
    forms: List[ExtractedSpan] = Field(default_factory=list)
    reasons: List[ExtractedSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Clinical Signals Section
# -------------------------


class DemographicsOutput(BaseModel):
    """Demographics data from BiomedicalNER."""

    age: List[ExtractedSpan] = Field(default_factory=list)
    sex: List[ExtractedSpan] = Field(default_factory=list)
    family_history: List[ExtractedSpan] = Field(default_factory=list)
    personal_background: List[ExtractedSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ClinicalOutput(BaseModel):
    """Clinical signals from BiomedicalNER."""

    symptoms: List[ExtractedSpan] = Field(default_factory=list)
    diagnostic_procedures: List[ExtractedSpan] = Field(default_factory=list)
    therapeutic_procedures: List[ExtractedSpan] = Field(default_factory=list)
    lab_values: List[ExtractedSpan] = Field(default_factory=list)
    outcomes: List[ExtractedSpan] = Field(default_factory=list)
    diseases: List[ExtractedSpan] = Field(default_factory=list)
    medications: List[ExtractedSpan] = Field(default_factory=list)
    demographics: DemographicsOutput = Field(default_factory=DemographicsOutput)

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Eligibility Criteria Section
# -------------------------


class EligibilityCriterion(BaseModel):
    """Single eligibility criterion."""

    text: str
    criterion_type: str  # "inclusion" or "exclusion"
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    category: Optional[str] = None  # e.g., "age", "diagnosis", "lab_value"
    parsed_logic: Optional[str] = None  # e.g., "age >= 18 AND age <= 65"

    model_config = ConfigDict(extra="forbid")


class EligibilityOutput(BaseModel):
    """Eligibility criteria from LLM extraction."""

    inclusion: List[EligibilityCriterion] = Field(default_factory=list)
    exclusion: List[EligibilityCriterion] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Study Design Section
# -------------------------


class StudyDesignOutput(BaseModel):
    """Study design information from LLM extraction."""

    endpoints: List[ExtractedSpan] = Field(default_factory=list)
    duration: List[ExtractedSpan] = Field(default_factory=list)
    sites: List[ExtractedSpan] = Field(default_factory=list)
    arms: List[ExtractedSpan] = Field(default_factory=list)
    phase: Optional[str] = None
    design_type: Optional[str] = None  # e.g., "randomized", "open-label"

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Patient Journey Section
# -------------------------


class PatientJourneyOutput(BaseModel):
    """
    Patient journey data from PatientJourneyEnricher.

    Captures longitudinal progression and real-world barriers that determine
    how patients reach (or fail to reach) trial eligibility.
    """

    # Diagnostic timeline: Time from symptom onset to diagnosis
    diagnostic_delay: List[ExtractedSpan] = Field(default_factory=list)

    # Care pathway: Healthcare journey milestones (GP -> specialist -> CoE -> trial)
    care_pathway: List[ExtractedSpan] = Field(default_factory=list)

    # Treatment history: Prior therapies (1L/2L/3L), washout considerations
    treatment_history: List[ExtractedSpan] = Field(default_factory=list)

    # Trial burden: Monitoring intervals, visit schedules
    trial_burden: List[ExtractedSpan] = Field(default_factory=list)

    # Retention risks: Patient barriers, travel burden, recruitment touchpoints
    retention_risks: List[ExtractedSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Unified Output Schema
# -------------------------


class UnifiedFeasibilityOutput(BaseModel):
    """
    Unified feasibility extraction output.

    Consolidates all NER enricher outputs into a single, clean schema.
    """

    # Document identification
    document_id: str
    document_name: Optional[str] = None
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    pipeline_version: str

    # Structured sections
    epidemiology: EpidemiologyOutput = Field(default_factory=EpidemiologyOutput)
    drug_admin: DrugAdminOutput = Field(default_factory=DrugAdminOutput)
    clinical: ClinicalOutput = Field(default_factory=ClinicalOutput)
    patient_journey: PatientJourneyOutput = Field(default_factory=PatientJourneyOutput)
    eligibility: EligibilityOutput = Field(default_factory=EligibilityOutput)
    study_design: StudyDesignOutput = Field(default_factory=StudyDesignOutput)

    # Metadata
    source_counts: Dict[str, int] = Field(default_factory=dict)
    total_spans: int = 0
    deduplication_stats: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid")

    def to_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "extraction_timestamp": self.extraction_timestamp.isoformat(),
            "pipeline_version": self.pipeline_version,
            "counts": {
                "epidemiology": {
                    "prevalence": len(self.epidemiology.prevalence),
                    "incidence": len(self.epidemiology.incidence),
                    "geography": len(self.epidemiology.geography),
                },
                "drug_admin": {
                    "adverse_events": len(self.drug_admin.adverse_events),
                    "dosages": len(self.drug_admin.dosages),
                    "frequencies": len(self.drug_admin.frequencies),
                    "routes": len(self.drug_admin.routes),
                },
                "clinical": {
                    "symptoms": len(self.clinical.symptoms),
                    "diagnostic_procedures": len(self.clinical.diagnostic_procedures),
                    "therapeutic_procedures": len(self.clinical.therapeutic_procedures),
                    "lab_values": len(self.clinical.lab_values),
                    "outcomes": len(self.clinical.outcomes),
                },
                "patient_journey": {
                    "diagnostic_delay": len(self.patient_journey.diagnostic_delay),
                    "care_pathway": len(self.patient_journey.care_pathway),
                    "treatment_history": len(self.patient_journey.treatment_history),
                    "trial_burden": len(self.patient_journey.trial_burden),
                    "retention_risks": len(self.patient_journey.retention_risks),
                },
                "eligibility": {
                    "inclusion": len(self.eligibility.inclusion),
                    "exclusion": len(self.eligibility.exclusion),
                },
            },
            "total_spans": self.total_spans,
            "source_counts": self.source_counts,
        }


# -------------------------
# Conversion Functions
# -------------------------


def candidates_to_unified_schema(
    candidates: List[Any],  # List of FeasibilityCandidate
    document_id: str,
    document_name: Optional[str] = None,
    pipeline_version: str = "0.8",
    dedup_stats: Optional[Dict[str, Any]] = None,
) -> UnifiedFeasibilityOutput:
    """
    Convert FeasibilityCandidate list to unified schema.

    Args:
        candidates: List of FeasibilityCandidate objects
        document_id: Unique document identifier
        document_name: Optional document name
        pipeline_version: Pipeline version string
        dedup_stats: Optional deduplication statistics

    Returns:
        UnifiedFeasibilityOutput with categorized spans
    """
    output = UnifiedFeasibilityOutput(
        document_id=document_id,
        document_name=document_name,
        pipeline_version=pipeline_version,
        deduplication_stats=dedup_stats,
    )

    source_counts: Dict[str, int] = {}

    for cand in candidates:
        category = getattr(cand, 'category', 'unknown')
        text = getattr(cand, 'text', '')
        confidence = getattr(cand, 'confidence', 0.5)
        source = getattr(cand, 'source', 'unknown')

        # Track source counts
        source_counts[source] = source_counts.get(source, 0) + 1

        span = ExtractedSpan(
            text=text,
            confidence=confidence,
            source=source,
            entity_type=category,
        )

        # Route to appropriate section based on category
        _route_span_to_section(output, category, span, cand)

    output.source_counts = source_counts
    output.total_spans = len(candidates)

    return output


def _route_span_to_section(
    output: UnifiedFeasibilityOutput,
    category: str,
    span: ExtractedSpan,
    candidate: Any,
) -> None:
    """Route a span to the appropriate section in the unified output."""
    cat_lower = category.lower()

    # Epidemiology
    if cat_lower in ["epidemiology", "prevalence"]:
        output.epidemiology.prevalence.append(span)
    elif cat_lower == "incidence":
        output.epidemiology.incidence.append(span)
    elif cat_lower == "mortality":
        output.epidemiology.mortality.append(span)
    elif cat_lower in ["geography", "location"]:
        output.epidemiology.geography.append(span)

    # Drug administration
    elif cat_lower in ["adverse_event", "ade"]:
        output.drug_admin.adverse_events.append(span)
    elif cat_lower in ["drug_dosage", "dosage"]:
        output.drug_admin.dosages.append(span)
    elif cat_lower in ["drug_frequency", "frequency"]:
        output.drug_admin.frequencies.append(span)
    elif cat_lower in ["drug_route", "route"]:
        output.drug_admin.routes.append(span)
    elif cat_lower in ["treatment_duration", "duration"]:
        output.drug_admin.durations.append(span)
    elif cat_lower == "strength":
        output.drug_admin.strengths.append(span)
    elif cat_lower == "form":
        output.drug_admin.forms.append(span)
    elif cat_lower == "reason":
        output.drug_admin.reasons.append(span)

    # Clinical
    elif cat_lower in ["symptom", "sign_symptom"]:
        output.clinical.symptoms.append(span)
    elif cat_lower == "diagnostic_procedure":
        output.clinical.diagnostic_procedures.append(span)
    elif cat_lower == "therapeutic_procedure":
        output.clinical.therapeutic_procedures.append(span)
    elif cat_lower in ["lab_value", "lab"]:
        output.clinical.lab_values.append(span)
    elif cat_lower == "outcome":
        output.clinical.outcomes.append(span)
    elif cat_lower == "disease":
        output.clinical.diseases.append(span)
    elif cat_lower == "medication":
        output.clinical.medications.append(span)

    # Patient Journey
    elif cat_lower == "diagnostic_delay":
        output.patient_journey.diagnostic_delay.append(span)
    elif cat_lower in ["treatment_line", "prior_therapy"]:
        output.patient_journey.treatment_history.append(span)
    elif cat_lower in ["care_pathway_step", "care_pathway"]:
        output.patient_journey.care_pathway.append(span)
    elif cat_lower in ["surveillance_frequency", "visit_frequency"]:
        output.patient_journey.trial_burden.append(span)
    elif cat_lower in ["pain_point", "recruitment_touchpoint"]:
        output.patient_journey.retention_risks.append(span)

    # Demographics
    elif cat_lower.startswith("demographics_") or cat_lower in ["age", "sex"]:
        demo_type = cat_lower.replace("demographics_", "")
        if demo_type == "age":
            output.clinical.demographics.age.append(span)
        elif demo_type == "sex":
            output.clinical.demographics.sex.append(span)
        elif demo_type == "family_history":
            output.clinical.demographics.family_history.append(span)
        else:
            output.clinical.demographics.personal_background.append(span)

    # Eligibility
    elif cat_lower in ["eligibility_inclusion", "inclusion"]:
        crit = EligibilityCriterion(
            text=span.text,
            criterion_type="inclusion",
            confidence=span.confidence,
            source=span.source,
        )
        output.eligibility.inclusion.append(crit)
    elif cat_lower in ["eligibility_exclusion", "exclusion"]:
        crit = EligibilityCriterion(
            text=span.text,
            criterion_type="exclusion",
            confidence=span.confidence,
            source=span.source,
        )
        output.eligibility.exclusion.append(crit)

    # Study design
    elif cat_lower in ["endpoint", "study_endpoint"]:
        output.study_design.endpoints.append(span)
    elif cat_lower in ["study_duration"]:
        output.study_design.duration.append(span)
    elif cat_lower in ["site", "study_site"]:
        output.study_design.sites.append(span)
