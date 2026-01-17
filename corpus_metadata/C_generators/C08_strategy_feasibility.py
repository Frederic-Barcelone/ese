# corpus_metadata/corpus_metadata/C_generators/C08_strategy_feasibility.py
"""
Clinical trial feasibility information extraction strategy.

Extracts key information needed for clinical trial feasibility assessment:
1. Eligibility criteria (inclusion/exclusion) with negation handling
2. Epidemiology data (prevalence, incidence, demographics) with context
3. Patient journey phases (screening → treatment → follow-up) with burden metrics
4. Study endpoints (primary, secondary) with proper boundaries
5. Site/country information with disambiguation

Uses a combination of:
- Layout-aware section detection
- Pattern matching with negation handling
- Context-gated country extraction
- Feature-based confidence scoring
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from A_core.A07_feasibility_models import (
    BackgroundTherapy,
    CriterionType,
    DiagnosisConfirmation,
    EligibilityCriterion,
    EndpointType,
    EpidemiologyData,
    EvidenceSpan,
    ExtractionMethod,
    FeasibilityCandidate,
    FeasibilityFieldType,
    FeasibilityGeneratorType,
    FeasibilityProvenanceMetadata,
    InvasiveProcedure,
    LabCriterion,
    LabTimepoint,
    OperationalBurden,
    PatientJourneyPhase,
    PatientJourneyPhaseType,
    ScreenFailReason,
    ScreeningFlow,
    ScreeningYield,
    StudyEndpoint,
    StudyFootprint,
    StudySite,
    VaccinationRequirement,
    VisitSchedule,
)
from B_parsing.B02_doc_graph import DocumentGraph
from B_parsing.B05_section_detector import SectionDetector
from B_parsing.B06_confidence import ConfidenceFeatures, ConfidenceCalculator
from B_parsing.B07_negation import NegationDetector, EXCEPTION_CUES


# =============================================================================
# NOTE: Section patterns, negation cues, and exception cues are imported from
# B_parsing.B05_section_detector, B_parsing.B07_negation
# =============================================================================


# =============================================================================
# FALSE POSITIVE FILTER
# =============================================================================


class FeasibilityFalsePositiveFilter:
    """
    Multi-layer false positive filter for feasibility extraction.

    Filters out:
    - Figure/Table/Appendix captions
    - Definitions (X=...)
    - Statistics/results (95% CI, p<, SE, SD)
    - OCR garbage
    """

    # Layer 1: Caption patterns - never eligibility/epidemiology
    CAPTION_PATTERNS = [
        r"^(?:Figure|Fig\.?|Table|Appendix|Panel|Supplementary)\s*\d",
        r"^(?:Figure|Fig\.?|Table)\s+[A-Z]\d?",
    ]

    # Layer 2: Definition patterns (X=...) -> glossary, not eligibility
    DEFINITION_PATTERNS = [
        r"^[A-Z]{2,10}\s*=",  # e.g., "eGFR=..."
        r"^\w+\s*=\s*\w+",     # Variable definitions
    ]

    # Layer 3: Statistics/results patterns
    STATISTICS_PATTERNS = [
        r"95%?\s*CI",
        r"p\s*[<>=]\s*0?\.\d+",
        r"(?:^|\s)(?:SE|SD)\s*[=:]?\s*\d",
        r"model[\s-]?estimated",
        r"hazard\s*ratio",
        r"odds\s*ratio",
        r"relative\s*risk",
        r"confidence\s*interval",
        r"standard\s*(?:error|deviation)",
        r"interquartile\s*range",
        r"IQR",
    ]

    # Layer 4: OCR garbage patterns
    OCR_GARBAGE_PATTERNS = [
        r"[^\w\s]{3,}",  # 3+ consecutive special chars
        r"^\d+\s*[-—–]\s*\w+\s+\d+\s+\d+",  # Table row patterns like "8 — Baseline 14 30"
        r"^[\d\s,.-]+$",  # Only numbers and punctuation
    ]

    # Layer 5: Bare fractions without context (not epidemiology)
    BARE_FRACTION_PATTERN = r"^\d+/\d+$"

    def __init__(self):
        self.caption_re = [re.compile(p, re.IGNORECASE) for p in self.CAPTION_PATTERNS]
        self.definition_re = [re.compile(p) for p in self.DEFINITION_PATTERNS]
        self.statistics_re = [re.compile(p, re.IGNORECASE) for p in self.STATISTICS_PATTERNS]
        self.ocr_re = [re.compile(p) for p in self.OCR_GARBAGE_PATTERNS]
        self.bare_fraction_re = re.compile(self.BARE_FRACTION_PATTERN)

    def is_caption(self, text: str) -> bool:
        """Check if text is a figure/table caption."""
        text = text.strip()
        return any(p.search(text) for p in self.caption_re)

    def is_definition(self, text: str) -> bool:
        """Check if text is a definition (X=...)."""
        text = text.strip()
        return any(p.match(text) for p in self.definition_re)

    def is_statistics(self, text: str) -> bool:
        """Check if text contains statistical results."""
        return any(p.search(text) for p in self.statistics_re)

    def is_ocr_garbage(self, text: str) -> bool:
        """Check if text looks like OCR noise."""
        text = text.strip()
        if len(text) < 5:
            return True
        return any(p.search(text) for p in self.ocr_re)

    def is_bare_fraction(self, text: str) -> bool:
        """Check if text is just a bare fraction like '11/37'."""
        return bool(self.bare_fraction_re.match(text.strip()))

    def filter_eligibility(self, text: str) -> Tuple[bool, str]:
        """
        Filter eligibility candidates.
        Returns (should_keep, rejection_reason).
        """
        if self.is_caption(text):
            return False, "caption"
        if self.is_definition(text):
            return False, "definition"
        if self.is_statistics(text):
            return False, "statistics"
        if self.is_ocr_garbage(text):
            return False, "ocr_garbage"
        return True, ""

    def filter_epidemiology(self, text: str, has_anchor: bool) -> Tuple[bool, str]:
        """
        Filter epidemiology candidates.
        Returns (should_keep, rejection_reason).
        """
        if self.is_caption(text):
            return False, "caption"
        if self.is_ocr_garbage(text):
            return False, "ocr_garbage"
        if self.is_bare_fraction(text) and not has_anchor:
            return False, "bare_fraction_no_anchor"
        if self.is_statistics(text) and not has_anchor:
            return False, "statistics_no_anchor"
        return True, ""


# =============================================================================
# EPIDEMIOLOGY ANCHOR PHRASES (required for bare numbers)
# =============================================================================

EPIDEMIOLOGY_ANCHORS = [
    r"prevalen",
    r"inciden",
    r"per\s*(?:million|100,?000|10,?000)",
    r"population[\s-]?based",
    r"registry",
    r"affects?\s*(?:approximately|about|~)?\s*\d",
    r"estimated\s*(?:at|to\s*be)",
]


# =============================================================================
# ELIGIBILITY CRITERIA PATTERNS (Expanded for rare disease)
# =============================================================================

INCLUSION_MARKERS = [
    r"inclusion\s*criteria",
    r"patients?\s*(?:were|are)\s*eligible\s*if",
    r"eligible\s*(?:patients?|subjects?|participants?)",
    r"key\s*inclusion",
    r"to\s*be\s*eligible",
    r"must\s*(?:have|be|meet)",
]

EXCLUSION_MARKERS = [
    r"exclusion\s*criteria",
    r"patients?\s*(?:were|are)\s*excluded\s*if",
    r"ineligible\s*if",
    r"key\s*exclusion",
    r"not\s*eligible\s*if",
    r"will\s*be\s*excluded",
]

# Expanded criterion categories for rare disease (P1)
CRITERION_CATEGORIES = {
    # Age
    "age": r"(?:age|year[s]?\s*old|\d+\s*(?:to|–|-)\s*\d+\s*years?)",

    # Disease definition (rare disease critical)
    "disease_definition": (
        r"(?:genetically\s*confirmed|pathogenic\s*variant|"
        r"confirmed\s*by\s*(?:biopsy|genetic|molecular)|"
        r"meets\s*diagnostic\s*criteria|"
        r"documented\s*(?:diagnosis|mutation))"
    ),

    # Disease severity
    "disease_severity": (
        r"(?:mild|moderate|severe|advanced|"
        r"(?:early|late)[- ]stage|"
        r"NYHA\s*class|mRS\s*\d|"
        r"FEV1|baseline.*(?:score|grade))"
    ),

    # Disease duration
    "disease_duration": (
        r"(?:disease\s*duration|onset\s*(?:before|after)|"
        r"diagnosed\s*(?:within|for\s*at\s*least))"
    ),

    # Prior treatment (expanded)
    "prior_treatment": (
        r"(?:prior|previous|history\s*of)\s+(?:treatment|therapy|medication)|"
        r"treatment[\s-]?na[ïi]ve|"
        r"(?:failure|intolerance)\s*(?:to|of)|"
        r"≥?\s*\d+\s*(?:prior\s*)?lines?\s*(?:of\s*therapy)?"
    ),

    # Biomarker/genetic
    "biomarker": (
        r"(?:mutation|variant|genotype|biomarker|"
        r"positive\s*for|negative\s*for|"
        r"expression\s*(?:of|level))"
    ),

    # Lab values (expanded)
    "lab_value": (
        r"(?:egfr|gfr|creatinine|hemoglobin|platelet|wbc|"
        r"alt|ast|bilirubin|albumin|inr|"
        r"≥?\s*\d+(?:\.\d+)?\s*(?:mg|g|u|mmol|µmol)/)"
    ),

    # Organ function
    "organ_function": (
        r"(?:hepatic|renal|cardiac|pulmonary)\s*(?:function|impairment)|"
        r"child[\s-]?pugh|"
        r"(?:liver|kidney|heart)\s*(?:disease|failure)"
    ),

    # Comorbidity
    "comorbidity": (
        r"(?:comorbid|concurrent|concomitant|coexisting)\s+(?:disease|condition)|"
        r"history\s*of\s*(?:cancer|malignancy|stroke|MI)"
    ),

    # Concomitant medications
    "concomitant_medications": (
        r"(?:concomitant|concurrent)\s*(?:use|medication)|"
        r"stable\s*dose|"
        r"chronic\s*(?:corticosteroid|immunosuppressant)"
    ),

    # Pregnancy
    "pregnancy": (
        r"(?:pregnan|breast[\s-]?feed|lactat|contracept|"
        r"women?\s*of\s*childbearing)"
    ),

    # Consent
    "consent": (
        r"(?:informed\s*consent|willing\s*to\s*participate|"
        r"able\s*to\s*comply)"
    ),
}


# =============================================================================
# EPIDEMIOLOGY PATTERNS (with context extraction)
# =============================================================================

PREVALENCE_PATTERNS = [
    r"prevalence\s*(?:of|:|\s)\s*([\d.,]+\s*(?:in|per|/)\s*[\d.,]+(?:\s*(?:million|thousand|100,?000|10,?000|1,?000))?)",
    r"affects?\s*([\d.,]+(?:\s*[-–to]+\s*[\d.,]+)?\s*(?:per|in)\s*(?:million|[\d,]+))",
    r"(<?\s*[\d.,]+\s*/\s*[\d.,]+)",
    r"([\d.,]+\s*%)\s*(?:of\s*)?(?:patients?|population|adults?|children)",
]

INCIDENCE_PATTERNS = [
    r"incidence\s*(?:of|:|\s)\s*([\d.,]+(?:\s*[-–to]+\s*[\d.,]+)?\s*(?:per|/)\s*[\d.,]+(?:\s*person[\-\s]?years?)?)",
    r"([\d.,]+(?:\s*[-–to]+\s*[\d.,]+)?\s*(?:new\s*)?cases?\s*(?:per|/)\s*(?:million|[\d,]+)(?:\s*per\s*year)?)",
]

DEMOGRAPHICS_PATTERNS = [
    r"((?:median|mean)\s*age\s*(?:of\s*)?[\d.,]+(?:\s*years?)?)",
    r"(age\s*(?:range|:)\s*[\d]+\s*[-–to]+\s*[\d]+)",
    r"([\d.,]+\s*%\s*(?:male|female|women|men))",
    r"((?:male|female)[:\s]+(?:male|female)\s*ratio\s*[\d.,]+\s*:\s*[\d.,]+)",
]

# Context extraction patterns for epidemiology (P1)
GEOGRAPHY_PATTERNS = [
    r"(?:in|across)\s+((?:the\s+)?(?:United States|Europe|Asia|North America|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?))",
    r"(US|European|Asian|global)\s+population",
]

TIME_PATTERNS = [
    r"(?:as\s+of|in)\s+(\d{4})",
    r"(?:between|from)\s+(\d{4})\s*(?:to|and|-)\s*(\d{4})",
    r"(\d{4})\s*(?:data|estimates?|statistics)",
]

SETTING_PATTERNS = [
    r"(population[\s-]?based|registry[\s-]?based|single[\s-]?center|referral\s*center|community)",
]


# =============================================================================
# PATIENT JOURNEY PATTERNS (with burden metrics)
# =============================================================================

JOURNEY_PHASE_KEYWORDS = {
    PatientJourneyPhaseType.SCREENING: [
        "screening period", "screening phase", "screening visit",
        "pre-treatment", "baseline assessment", "wash-out", "washout period",
    ],
    PatientJourneyPhaseType.RUN_IN: [
        "run-in period", "run in phase", "lead-in period", "stabilization period",
    ],
    PatientJourneyPhaseType.RANDOMIZATION: [
        "randomization", "randomised", "randomized",
        "random assignment", "treatment allocation",
    ],
    PatientJourneyPhaseType.TREATMENT: [
        "treatment period", "treatment phase", "active treatment",
        "intervention period", "dosing period", "induction phase", "maintenance phase",
    ],
    PatientJourneyPhaseType.FOLLOW_UP: [
        "follow-up period", "follow up phase", "post-treatment",
        "safety follow-up", "observation period", "long-term follow-up",
    ],
    PatientJourneyPhaseType.EXTENSION: [
        "extension study", "open-label extension", "OLE period", "continued access",
    ],
}

DURATION_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:[-–to]+\s*\d+(?:\.\d+)?\s*)?(weeks?|months?|days?|years?)",
    re.IGNORECASE,
)

# Burden metrics patterns
VISIT_PATTERNS = [
    r"(\d+)\s*(?:study\s*)?visits?",
    r"every\s*(\d+)\s*weeks?",
    r"(?:weekly|bi[\s-]?weekly|monthly)\s*visits?",
]

PROCEDURE_PATTERNS = [
    r"(?:lumbar\s*puncture|bone\s*marrow|biopsy|infusion|"
    r"MRI|CT\s*scan|PET|echocardiogram|endoscopy)",
]

INPATIENT_PATTERNS = [
    r"(\d+)\s*(?:day|night)s?\s*(?:hospitalization|inpatient)",
    r"(?:overnight|24[\s-]?hour)\s*(?:stay|observation)",
]


# =============================================================================
# ENDPOINT PATTERNS
# =============================================================================

ENDPOINT_PATTERNS = {
    EndpointType.PRIMARY: [
        r"primary\s*(?:end)?point[s]?\s*(?:is|was|were|:)",
        r"primary\s*(?:efficacy\s*)?(?:end)?point",
        r"main\s*(?:end)?point",
    ],
    EndpointType.SECONDARY: [
        r"secondary\s*(?:end)?point[s]?",
        r"key\s*secondary",
    ],
    EndpointType.EXPLORATORY: [
        r"exploratory\s*(?:end)?point[s]?",
        r"tertiary\s*(?:end)?point[s]?",
    ],
    EndpointType.SAFETY: [
        r"safety\s*(?:end)?point[s]?",
        r"adverse\s*event[s]?",
        r"tolerability",
    ],
}


# =============================================================================
# SCREENING YIELD PATTERNS (CONSORT Flow)
# =============================================================================

SCREENING_YIELD_PATTERNS = [
    r"(\d+)\s*(?:patients?|subjects?|participants?)\s*(?:were\s*)?screened",
    r"screened\s*(?:n\s*=\s*)?(\d+)",
    r"(\d+)\s*(?:patients?|subjects?)\s*randomized",
    r"randomized\s*(?:n\s*=\s*)?(\d+)",
    r"(\d+)\s*(?:patients?|subjects?)\s*enrolled",
    r"enrolled\s*(?:n\s*=\s*)?(\d+)",
    r"(\d+)\s*(?:screen(?:ing)?\s*)?failures?",
    r"(\d+)\s*(?:patients?|subjects?)\s*discontinued",
    r"(\d+)\s*(?:patients?|subjects?)\s*completed",
    r"completion\s*rate\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%",
    r"screen\s*failure\s*rate\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%",
    r"dropout\s*rate\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%",
]

CONSORT_FLOW_PATTERNS = [
    r"(?:assessed\s*for\s*eligibility|screened)\s*\(?n\s*=\s*(\d+)\)?",
    r"(?:excluded|screen\s*failures?)\s*\(?n\s*=\s*(\d+)\)?",
    r"(?:randomized|allocated)\s*\(?n\s*=\s*(\d+)\)?",
    r"(?:analysed|analyzed|completed)\s*\(?n\s*=\s*(\d+)\)?",
    r"(?:lost\s*to\s*follow[\s-]?up|discontinued)\s*\(?n\s*=\s*(\d+)\)?",
]


# =============================================================================
# VACCINATION REQUIREMENT PATTERNS
# =============================================================================

VACCINATION_PATTERNS = [
    r"(?:covid[\s-]?19|sars[\s-]?cov[\s-]?2)\s*vaccin(?:e|ation)\s*(?:required|completed|received)",
    r"(?:fully\s*)?vaccinated\s*(?:against|for)\s*([\w\s-]+)",
    r"(?:prior|previous)\s*(?:covid[\s-]?19|influenza|hepatitis\s*[ab])\s*vaccination",
    r"vaccin(?:e|ation)\s*(?:at\s*least\s*)?(\d+)\s*(?:weeks?|days?|months?)\s*(?:before|prior)",
    r"(?:live|attenuated)\s*vaccine\s*(?:within|in\s*the\s*past)\s*(\d+)\s*(?:weeks?|days?|months?)",
    r"(?:no|without)\s*(?:live|attenuated)\s*vaccine",
]

VACCINE_TYPES = [
    "covid-19", "sars-cov-2", "influenza", "hepatitis a", "hepatitis b",
    "pneumococcal", "meningococcal", "mmr", "varicella", "zoster",
    "yellow fever", "bcg", "live attenuated",
]


# =============================================================================
# SITE/COUNTRY COUNT PATTERNS
# =============================================================================

SITE_COUNT_PATTERNS = [
    r"(\d+)\s*(?:study\s*)?sites?\s*(?:in|across)\s*(\d+)\s*countries?",
    r"(\d+)\s*(?:investigational?\s*)?centers?\s*(?:in|across)\s*(\d+)\s*countries?",
    r"conducted\s*(?:at|in)\s*(\d+)\s*(?:study\s*)?sites?",
    r"(\d+)\s*(?:participating|active)\s*sites?",
    r"multinational\s*(?:study|trial)\s*(?:in|across)\s*(\d+)\s*countries?",
    r"(\d+)\s*countries?\s*(?:participated|enrolled|recruited)",
]


# =============================================================================
# OPERATIONAL BURDEN PATTERNS
# =============================================================================

INVASIVE_PROCEDURE_PATTERNS = [
    r"(?:renal|kidney)\s*biops(?:y|ies)",
    r"bone\s*marrow\s*(?:aspirat(?:e|ion)|biops(?:y|ies))",
    r"lumbar\s*puncture",
    r"liver\s*biops(?:y|ies)",
    r"skin\s*biops(?:y|ies)",
    r"bronchoscop(?:y|ic)",
    r"colonoscop(?:y|ic)",
    r"endoscop(?:y|ic)",
    r"(?:central\s*)?venous\s*(?:catheter|line)",
]

VISIT_SCHEDULE_PATTERNS = [
    r"(?:day|week)\s*(\d+)(?:\s*[,/]\s*(?:day|week)\s*(\d+))+",
    r"visits?\s*(?:at|on)\s*(?:day|week)s?\s*(\d+(?:\s*[,/and]+\s*\d+)*)",
    r"every\s*(\d+)\s*(?:days?|weeks?|months?)",
    r"(\d+)\s*(?:study\s*)?visits?\s*(?:over|during)\s*(\d+)\s*(?:weeks?|months?)",
    r"follow[\s-]?up\s*(?:at|for)\s*(\d+)\s*(?:weeks?|months?|years?)",
]

BACKGROUND_THERAPY_PATTERNS = [
    r"(?:stable\s*)?(?:dose\s*of\s*)?(?:ACE[\s-]?inhibitor|ARB|ACEi)s?\s*(?:for\s*)?(?:≥|>=|at\s*least)?\s*(\d+)\s*(?:days?|weeks?|months?)",
    r"(?:background|concomitant)\s*(?:therapy|treatment|medication)\s*(?:with|of)\s*([\w\s/-]+)",
    r"(?:immunosuppressant|corticosteroid)s?\s*(?:at\s*)?(?:stable\s*)?(?:dose\s*)?(?:for\s*)?(?:≥|>=)?\s*(\d+)\s*(?:days?|weeks?|months?)",
    r"(?:prohibited|not\s*permitted|excluded).*(?:live\s*)?vaccin(?:e|ation)",
]

LAB_CRITERION_PATTERNS = [
    r"(?:UPCR|proteinuria)\s*(?:≥|>=|>|≤|<=|<)\s*(\d+(?:\.\d+)?)\s*(?:g/g|mg/mmol|mg/g)",
    r"(?:eGFR|GFR)\s*(?:≥|>=|>|≤|<=|<)\s*(\d+)\s*(?:mL/min|ml/min)",
    r"(?:C3|C4|complement)\s*(?:level)?\s*(?:≥|>=|>|≤|<=|<)\s*(\d+(?:\.\d+)?)\s*(?:mg/dL|g/L|mg/L)",
    r"(?:hemoglobin|Hgb|Hb)\s*(?:≥|>=|>|≤|<=|<)\s*(\d+(?:\.\d+)?)\s*(?:g/dL|g/L)",
    r"(?:platelet|PLT)\s*(?:count)?\s*(?:≥|>=|>|≤|<=|<)\s*(\d+)\s*(?:×\s*10\^?9/L|/μL|K/μL)",
]


# =============================================================================
# SITE/COUNTRY PATTERNS (with disambiguation)
# =============================================================================

# Countries that need context validation (P0)
AMBIGUOUS_COUNTRIES = {"georgia", "jordan", "turkey", "chad", "china", "guinea", "mali"}

# Context cues that indicate site/country discussion
COUNTRY_CONTEXT_CUES = {
    "sites", "centers", "countries", "enrolled", "conducted",
    "multicenter", "international", "patients from", "study in",
    "recruitment", "participating", "locations",
}

COUNTRIES = {
    "united states", "usa", "us", "germany", "france", "uk", "united kingdom",
    "italy", "spain", "canada", "australia", "japan", "china", "brazil",
    "netherlands", "belgium", "switzerland", "austria", "sweden", "norway",
    "denmark", "finland", "poland", "czech republic", "hungary", "israel",
    "south korea", "korea", "taiwan", "india", "russia", "mexico", "argentina",
    "turkey", "greece", "portugal", "ireland", "new zealand", "singapore",
    "hong kong", "thailand", "malaysia", "south africa", "egypt", "chile",
    "colombia", "peru", "ukraine", "romania", "bulgaria",
}

# ISO country codes for validation
COUNTRY_CODES = {
    "united states": "US", "usa": "US", "us": "US",
    "germany": "DE", "france": "FR", "uk": "GB", "united kingdom": "GB",
    "italy": "IT", "spain": "ES", "canada": "CA", "australia": "AU",
    "japan": "JP", "china": "CN", "brazil": "BR", "netherlands": "NL",
    "belgium": "BE", "switzerland": "CH", "austria": "AT", "sweden": "SE",
    "norway": "NO", "denmark": "DK", "finland": "FI", "poland": "PL",
    "czech republic": "CZ", "hungary": "HU", "israel": "IL",
    "south korea": "KR", "korea": "KR", "taiwan": "TW", "india": "IN",
    "russia": "RU", "mexico": "MX", "argentina": "AR", "turkey": "TR",
}


# =============================================================================
# CONFIDENCE SCORING - Uses shared B_parsing.B06_confidence
# Expected sections mapping for feasibility fields
# =============================================================================

EXPECTED_SECTIONS = {
    FeasibilityFieldType.ELIGIBILITY_INCLUSION: ["eligibility", "methods"],
    FeasibilityFieldType.ELIGIBILITY_EXCLUSION: ["eligibility", "methods"],
    FeasibilityFieldType.EPIDEMIOLOGY_PREVALENCE: ["epidemiology", "abstract"],
    FeasibilityFieldType.EPIDEMIOLOGY_INCIDENCE: ["epidemiology", "abstract"],
    FeasibilityFieldType.EPIDEMIOLOGY_DEMOGRAPHICS: ["epidemiology", "methods", "results"],
    FeasibilityFieldType.STUDY_ENDPOINT: ["endpoints", "methods"],
    FeasibilityFieldType.PATIENT_JOURNEY_PHASE: ["patient_journey", "methods"],
    FeasibilityFieldType.STUDY_SITE: ["methods"],
    FeasibilityFieldType.SCREENING_YIELD: ["results", "methods", "abstract"],
    FeasibilityFieldType.VACCINATION_REQUIREMENT: ["eligibility", "methods"],
}


# =============================================================================
# FEASIBILITY DETECTOR CLASS
# =============================================================================


class FeasibilityDetector:
    """
    Extracts clinical trial feasibility information from documents.

    Improvements over v1:
    - Negation handling for eligibility criteria
    - Country disambiguation with context validation
    - Expanded eligibility categories for rare disease
    - Epidemiology context extraction (geography, time, setting)
    - Feature-based confidence scoring
    - Patient journey burden metrics
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.run_id = str(self.config.get("run_id") or generate_run_id("FEAS"))
        self.pipeline_version = (
            self.config.get("pipeline_version") or get_git_revision_hash()
        )
        self.doc_fingerprint_default = (
            self.config.get("doc_fingerprint") or "unknown-doc-fingerprint"
        )

        # Context window for evidence extraction
        self.context_window = int(self.config.get("context_window", 300))


        # Shared parsing utilities from B_parsing
        self.section_detector = SectionDetector()
        self.negation_detector = NegationDetector()
        self.confidence_calculator = ConfidenceCalculator()

        # False positive filter
        self.fp_filter = FeasibilityFalsePositiveFilter()

        # Epidemiology anchor patterns
        self.epi_anchor_re = [re.compile(p, re.IGNORECASE) for p in EPIDEMIOLOGY_ANCHORS]

        # Compile patterns
        self._compile_patterns()

        # Stats for summary
        self._extraction_stats: Dict[str, int] = {}
        self._rejection_stats: Dict[str, int] = {}

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        self.inclusion_re = [re.compile(p, re.IGNORECASE) for p in INCLUSION_MARKERS]
        self.exclusion_re = [re.compile(p, re.IGNORECASE) for p in EXCLUSION_MARKERS]

        self.prevalence_re = [re.compile(p, re.IGNORECASE) for p in PREVALENCE_PATTERNS]
        self.incidence_re = [re.compile(p, re.IGNORECASE) for p in INCIDENCE_PATTERNS]
        self.demographics_re = [re.compile(p, re.IGNORECASE) for p in DEMOGRAPHICS_PATTERNS]

        self.geography_re = [re.compile(p, re.IGNORECASE) for p in GEOGRAPHY_PATTERNS]
        self.time_re = [re.compile(p, re.IGNORECASE) for p in TIME_PATTERNS]
        self.setting_re = [re.compile(p, re.IGNORECASE) for p in SETTING_PATTERNS]

        self.endpoint_re = {
            etype: [re.compile(p, re.IGNORECASE) for p in patterns]
            for etype, patterns in ENDPOINT_PATTERNS.items()
        }

        self.criterion_category_re = {
            cat: re.compile(pattern, re.IGNORECASE)
            for cat, pattern in CRITERION_CATEGORIES.items()
        }

        self.visit_re = [re.compile(p, re.IGNORECASE) for p in VISIT_PATTERNS]
        self.procedure_re = re.compile(PROCEDURE_PATTERNS[0], re.IGNORECASE)
        self.inpatient_re = [re.compile(p, re.IGNORECASE) for p in INPATIENT_PATTERNS]

        self.screening_yield_re = [re.compile(p, re.IGNORECASE) for p in SCREENING_YIELD_PATTERNS]
        self.consort_flow_re = [re.compile(p, re.IGNORECASE) for p in CONSORT_FLOW_PATTERNS]
        self.vaccination_re = [re.compile(p, re.IGNORECASE) for p in VACCINATION_PATTERNS]
        self.site_count_re = [re.compile(p, re.IGNORECASE) for p in SITE_COUNT_PATTERNS]

    def extract(self, doc_structure: DocumentGraph) -> List[FeasibilityCandidate]:
        """Extract feasibility information from document."""
        doc = doc_structure
        candidates: List[FeasibilityCandidate] = []
        seen: Set[str] = set()

        doc_fingerprint = getattr(doc, "fingerprint", self.doc_fingerprint_default)
        doc_id = getattr(doc, "doc_id", "unknown")

        current_section = "unknown"

        for block in doc.iter_linear_blocks(skip_header_footer=True):
            text = (block.text or "").strip()
            if not text:
                continue

            page_num = getattr(block, "page_number", None)

            # Layout-aware section detection
            section = self._detect_section(text, block)
            if section:
                current_section = section

            candidates.extend(
                self._extract_eligibility(
                    text, doc_id, doc_fingerprint, page_num, current_section, seen
                )
            )
            candidates.extend(
                self._extract_epidemiology(
                    text, doc_id, doc_fingerprint, page_num, current_section, seen
                )
            )
            candidates.extend(
                self._extract_patient_journey(
                    text, doc_id, doc_fingerprint, page_num, current_section, seen
                )
            )
            candidates.extend(
                self._extract_endpoints(
                    text, doc_id, doc_fingerprint, page_num, current_section, seen
                )
            )
            candidates.extend(
                self._extract_sites(
                    text, doc_id, doc_fingerprint, page_num, current_section, seen
                )
            )
            candidates.extend(
                self._extract_screening_yield(
                    text, doc_id, doc_fingerprint, page_num, current_section, seen
                )
            )
            candidates.extend(
                self._extract_vaccination(
                    text, doc_id, doc_fingerprint, page_num, current_section, seen
                )
            )

        self._update_stats(candidates)
        return candidates

    def _detect_section(self, text: str, block: Any = None) -> Optional[str]:
        """Layout-aware section detection using shared SectionDetector."""
        result = self.section_detector.detect(text, block)
        return result.name if result else None

    # =========================================================================
    # NEGATION DETECTION - Uses shared B_parsing.B07_negation
    # =========================================================================

    def _detect_negation(self, text: str, match_start: int) -> bool:
        """Check if match is negated using shared NegationDetector."""
        result = self.negation_detector.detect(text, match_start)
        return result.is_negated and not result.is_double_negation

    def _has_exception(self, text: str) -> bool:
        """Check if text contains exception cues."""
        text_lower = text.lower()
        return any(cue in text_lower for cue in EXCEPTION_CUES)

    # =========================================================================
    # ELIGIBILITY EXTRACTION (with negation handling)
    # =========================================================================

    def _extract_eligibility(
        self,
        text: str,
        doc_id: str,
        doc_fingerprint: str,
        page_num: Optional[int],
        section: str,
        seen: Set[str],
    ) -> List[FeasibilityCandidate]:
        """Extract eligibility criteria with negation handling and FP filtering."""
        candidates = []

        # Early rejection: block-level FP check
        keep, reason = self.fp_filter.filter_eligibility(text)
        if not keep:
            self._rejection_stats[f"elig_{reason}"] = self._rejection_stats.get(f"elig_{reason}", 0) + 1
            return []

        is_inclusion = any(p.search(text) for p in self.inclusion_re)
        is_exclusion = any(p.search(text) for p in self.exclusion_re)

        if section == "eligibility" or is_inclusion or is_exclusion:
            criteria = self._extract_bullet_items(text)

            for criterion_text in criteria:
                if len(criterion_text) < 10:
                    continue

                # Item-level FP check
                keep, reason = self.fp_filter.filter_eligibility(criterion_text)
                if not keep:
                    self._rejection_stats[f"elig_{reason}"] = self._rejection_stats.get(f"elig_{reason}", 0) + 1
                    continue

                # Determine base type
                criterion_type = CriterionType.EXCLUSION if is_exclusion else CriterionType.INCLUSION

                # Check for negation
                is_negated = self._detect_negation(text, text.find(criterion_text[:20]))

                # Flip type if negated in exclusion context
                # "Excluded if NOT pregnant" → pregnancy allowed (inclusion-like)
                if is_exclusion and is_negated:
                    # Keep as exclusion but flag
                    pass

                dedup_key = f"elig:{criterion_text[:50].lower()}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                category = self._detect_criterion_category(criterion_text)
                derived = self._derive_eligibility_variables(criterion_text, category)

                field_type = (
                    FeasibilityFieldType.ELIGIBILITY_EXCLUSION
                    if criterion_type == CriterionType.EXCLUSION
                    else FeasibilityFieldType.ELIGIBILITY_INCLUSION
                )

                # Calculate confidence
                features = self._calculate_confidence_features(
                    field_type, section, text, criterion_text
                )

                candidate = self._make_candidate(
                    doc_id=doc_id,
                    doc_fingerprint=doc_fingerprint,
                    matched_text=criterion_text,
                    context_text=text[:self.context_window],
                    field_type=field_type,
                    page_num=page_num,
                    section=section,
                    confidence=features.total(),
                    confidence_features=features.to_dict(),
                )
                candidate.eligibility_criterion = EligibilityCriterion(
                    criterion_type=criterion_type,
                    text=criterion_text,
                    category=category,
                    is_negated=is_negated,
                    derived_variables=derived if derived else None,
                )
                candidates.append(candidate)

        return candidates

    def _extract_bullet_items(self, text: str) -> List[str]:
        """Extract bullet point or numbered list items."""
        items = []

        patterns = [
            r"(?:^|\n)\s*[•●○▪▸]\s*(.+?)(?=\n\s*[•●○▪▸]|\n\n|$)",
            r"(?:^|\n)\s*[-–—]\s*(.+?)(?=\n\s*[-–—]|\n\n|$)",
            r"(?:^|\n)\s*\d+[.)]\s*(.+?)(?=\n\s*\d+[.)]|\n\n|$)",
            r"(?:^|\n)\s*[a-z][.)]\s*(.+?)(?=\n\s*[a-z][.)]|\n\n|$)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            items.extend([m.strip() for m in matches if m.strip()])

        if not items and len(text) < 500:
            sentences = re.split(r'[.;](?=\s|$)', text)
            for sent in sentences:
                sent = sent.strip()
                if 20 < len(sent) < 300:
                    if any(p.search(sent) for p in self.criterion_category_re.values()):
                        items.append(sent)

        return items

    def _detect_criterion_category(self, text: str) -> Optional[str]:
        """Detect the category of an eligibility criterion."""
        text_lower = text.lower()
        for category, pattern in self.criterion_category_re.items():
            if pattern.search(text_lower):
                return category
        return None

    def _derive_eligibility_variables(
        self, text: str, category: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Extract ML-ready variables from criterion."""
        text_lower = text.lower()
        derived: Dict[str, Any] = {}

        # Age extraction
        age_match = re.search(r'(\d+)\s*(?:to|–|-)\s*(\d+)\s*years?', text_lower)
        if age_match:
            derived['age_min'] = int(age_match.group(1))
            derived['age_max'] = int(age_match.group(2))
            derived['pediatric_allowed'] = derived['age_min'] < 18
            derived['elderly_allowed'] = derived['age_max'] >= 65

        # Prior lines
        lines_match = re.search(r'(\d+)\s*(?:or\s*more\s*)?(?:prior\s*)?lines?', text_lower)
        if lines_match:
            derived['prior_lines_required'] = int(lines_match.group(1))

        if 'treatment-naive' in text_lower or 'treatment naive' in text_lower:
            derived['prior_lines_required'] = 0
            derived['treatment_naive_required'] = True

        return derived if derived else None

    # =========================================================================
    # EPIDEMIOLOGY EXTRACTION (with context)
    # =========================================================================

    def _has_epi_anchor(self, text: str, start: int, end: int) -> bool:
        """Check if there's an epidemiology anchor phrase near the match."""
        # Check in a window around the match
        window_start = max(0, start - 100)
        window_end = min(len(text), end + 50)
        window = text[window_start:window_end]
        return any(p.search(window) for p in self.epi_anchor_re)

    def _extract_epidemiology(
        self,
        text: str,
        doc_id: str,
        doc_fingerprint: str,
        page_num: Optional[int],
        section: str,
        seen: Set[str],
    ) -> List[FeasibilityCandidate]:
        """Extract epidemiology statistics with context and anchor validation."""
        candidates = []

        # Block-level check for section appropriateness
        valid_sections = {"epidemiology", "abstract", "results", "unknown"}
        if section not in valid_sections:
            # Allow but with lower confidence if in wrong section
            pass

        # Prevalence
        for pattern in self.prevalence_re:
            for match in pattern.finditer(text):
                value = match.group(1) if match.lastindex else match.group(0)

                # Check for anchor phrase
                has_anchor = self._has_epi_anchor(text, match.start(), match.end())

                # FP filter
                keep, reason = self.fp_filter.filter_epidemiology(value, has_anchor)
                if not keep:
                    self._rejection_stats[f"epi_{reason}"] = self._rejection_stats.get(f"epi_{reason}", 0) + 1
                    continue

                dedup_key = f"prev:{value.lower()}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                # Extract context dimensions
                geography = self._extract_geography(text, match.start(), match.end())
                time_period = self._extract_time_period(text, match.start(), match.end())
                setting = self._extract_setting(text)
                normalized = self._normalize_epi_value(value, match.group(0))

                # Calculate confidence with context bonus
                features = self._calculate_confidence_features(
                    FeasibilityFieldType.EPIDEMIOLOGY_PREVALENCE, section, text, value
                )
                if has_anchor:
                    features.anchor_proximity = 0.2
                if geography:
                    features.context_completeness += 0.1
                if time_period:
                    features.context_completeness += 0.1
                if setting:
                    features.context_completeness += 0.05

                candidate = self._make_candidate(
                    doc_id=doc_id,
                    doc_fingerprint=doc_fingerprint,
                    matched_text=match.group(0),
                    context_text=self._get_context(text, match.start(), match.end()),
                    field_type=FeasibilityFieldType.EPIDEMIOLOGY_PREVALENCE,
                    page_num=page_num,
                    section=section,
                    confidence=features.total(),
                    confidence_features=features.to_dict(),
                )
                candidate.epidemiology_data = EpidemiologyData(
                    data_type="prevalence",
                    value=value.strip(),
                    normalized_value=normalized,
                    geography=geography,
                    time_period=time_period,
                    setting=setting,
                )
                candidates.append(candidate)

        # Incidence
        for pattern in self.incidence_re:
            for match in pattern.finditer(text):
                value = match.group(1) if match.lastindex else match.group(0)

                # Check for anchor phrase
                has_anchor = self._has_epi_anchor(text, match.start(), match.end())

                # FP filter
                keep, reason = self.fp_filter.filter_epidemiology(value, has_anchor)
                if not keep:
                    self._rejection_stats[f"epi_{reason}"] = self._rejection_stats.get(f"epi_{reason}", 0) + 1
                    continue

                dedup_key = f"inc:{value.lower()}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                geography = self._extract_geography(text, match.start(), match.end())
                time_period = self._extract_time_period(text, match.start(), match.end())
                normalized = self._normalize_epi_value(value, match.group(0))

                features = self._calculate_confidence_features(
                    FeasibilityFieldType.EPIDEMIOLOGY_INCIDENCE, section, text, value
                )
                if has_anchor:
                    features.anchor_proximity = 0.2

                candidate = self._make_candidate(
                    doc_id=doc_id,
                    doc_fingerprint=doc_fingerprint,
                    matched_text=match.group(0),
                    context_text=self._get_context(text, match.start(), match.end()),
                    field_type=FeasibilityFieldType.EPIDEMIOLOGY_INCIDENCE,
                    page_num=page_num,
                    section=section,
                    confidence=features.total(),
                    confidence_features=features.to_dict(),
                )
                candidate.epidemiology_data = EpidemiologyData(
                    data_type="incidence",
                    value=value.strip(),
                    normalized_value=normalized,
                    geography=geography,
                    time_period=time_period,
                )
                candidates.append(candidate)

        # Demographics - more lenient (baseline characteristics are OK)
        for pattern in self.demographics_re:
            for match in pattern.finditer(text):
                value = match.group(1) if match.lastindex else match.group(0)
                dedup_key = f"demo:{value.lower()}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                features = self._calculate_confidence_features(
                    FeasibilityFieldType.EPIDEMIOLOGY_DEMOGRAPHICS, section, text, value
                )

                candidate = self._make_candidate(
                    doc_id=doc_id,
                    doc_fingerprint=doc_fingerprint,
                    matched_text=match.group(0),
                    context_text=self._get_context(text, match.start(), match.end()),
                    field_type=FeasibilityFieldType.EPIDEMIOLOGY_DEMOGRAPHICS,
                    page_num=page_num,
                    section=section,
                    confidence=features.total(),
                    confidence_features=features.to_dict(),
                )
                candidate.epidemiology_data = EpidemiologyData(
                    data_type="demographics",
                    value=value.strip(),
                )
                candidates.append(candidate)

        return candidates

    def _extract_geography(self, text: str, start: int, end: int) -> Optional[str]:
        """Extract geography context around a match."""
        window = text[max(0, start - 100):min(len(text), end + 100)]
        for pattern in self.geography_re:
            m = pattern.search(window)
            if m:
                return m.group(1).strip()
        return None

    def _extract_time_period(self, text: str, start: int, end: int) -> Optional[str]:
        """Extract time period context around a match."""
        window = text[max(0, start - 100):min(len(text), end + 100)]
        for pattern in self.time_re:
            m = pattern.search(window)
            if m:
                if m.lastindex and m.lastindex >= 2:
                    return f"{m.group(1)}-{m.group(2)}"
                return m.group(1)
        return None

    def _extract_setting(self, text: str) -> Optional[str]:
        """Extract study setting from text."""
        for pattern in self.setting_re:
            m = pattern.search(text)
            if m:
                return m.group(1).strip()
        return None

    def _normalize_epi_value(self, value: str, full_match: str) -> Optional[float]:
        """Normalize epidemiology value to per-million."""
        try:
            # "1 in 10,000" -> 100 per million
            match = re.search(r'([\d.,]+)\s*(?:in|per|/)\s*([\d,]+)', value)
            if match:
                num = float(match.group(1).replace(',', ''))
                denom = float(match.group(2).replace(',', ''))
                if denom > 0:
                    return (num / denom) * 1_000_000

            # "3.5%" -> 35,000 per million
            if '%' in value:
                pct_match = re.search(r'([\d.,]+)', value)
                if pct_match:
                    pct = float(pct_match.group(1).replace(',', ''))
                    return pct * 10_000

        except (ValueError, ZeroDivisionError):
            pass
        return None

    # =========================================================================
    # PATIENT JOURNEY EXTRACTION (with burden metrics)
    # =========================================================================

    def _extract_patient_journey(
        self,
        text: str,
        doc_id: str,
        doc_fingerprint: str,
        page_num: Optional[int],
        section: str,
        seen: Set[str],
    ) -> List[FeasibilityCandidate]:
        """Extract patient journey phase information with explicit phase+duration patterns."""
        candidates = []
        text_lower = text.lower()

        # Skip if clearly wrong section
        if self.fp_filter.is_caption(text):
            return []

        # Phase patterns that require multi-word context (not single tokens)
        STRONG_PHASE_PATTERNS = {
            PatientJourneyPhaseType.SCREENING: [
                r"screening\s+(?:period|phase|visit)",
                r"(?:pre-?treatment|baseline)\s+(?:assessment|period|phase)",
                r"wash[\s-]?out\s+period",
            ],
            PatientJourneyPhaseType.RUN_IN: [
                r"run[\s-]?in\s+(?:period|phase)",
                r"lead[\s-]?in\s+(?:period|phase)",
                r"stabilization\s+period",
            ],
            PatientJourneyPhaseType.TREATMENT: [
                r"(?:treatment|intervention|dosing|active)\s+(?:period|phase)",
                r"(?:double[\s-]?blind|open[\s-]?label)\s+(?:period|phase|treatment)",
                r"(?:induction|maintenance)\s+(?:period|phase)",
            ],
            PatientJourneyPhaseType.FOLLOW_UP: [
                r"follow[\s-]?up\s+(?:period|phase|visit)",
                r"(?:post[\s-]?treatment|observation)\s+(?:period|phase)",
                r"safety\s+follow[\s-]?up",
            ],
            PatientJourneyPhaseType.EXTENSION: [
                r"(?:open[\s-]?label\s+)?extension\s+(?:study|period|phase)",
                r"(?:long[\s-]?term|continued)\s+(?:access|treatment)",
            ],
        }

        for phase_type, patterns in STRONG_PHASE_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    keyword = match.group(0)
                    dedup_key = f"journey:{phase_type.value}:{keyword[:20]}"
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    # Extract duration NEAR the match (not anywhere in text)
                    match_start = match.start()
                    match_end = match.end()
                    duration_window = text[max(0, match_start - 30):min(len(text), match_end + 80)]
                    duration_match = DURATION_PATTERN.search(duration_window)
                    duration = None
                    if duration_match:
                        duration = f"{duration_match.group(1)} {duration_match.group(2)}"

                    # Extract burden metrics
                    visits = self._extract_visit_count(text)
                    visit_frequency = self._extract_visit_frequency(text)
                    procedures = self._extract_procedures(text)
                    inpatient_days = self._extract_inpatient_days(text)

                    features = self._calculate_confidence_features(
                        FeasibilityFieldType.PATIENT_JOURNEY_PHASE, section, text, keyword
                    )
                    # Boost confidence for explicit patterns
                    features.pattern_strength = 0.2
                    if duration:
                        features.context_completeness = 0.1

                    candidate = self._make_candidate(
                        doc_id=doc_id,
                        doc_fingerprint=doc_fingerprint,
                        matched_text=keyword,
                        context_text=text[:self.context_window],
                        field_type=FeasibilityFieldType.PATIENT_JOURNEY_PHASE,
                        page_num=page_num,
                        section=section,
                        confidence=features.total(),
                        confidence_features=features.to_dict(),
                    )
                    candidate.patient_journey_phase = PatientJourneyPhase(
                        phase_type=phase_type,
                        description=text[:200],
                        duration=duration,
                        visits=visits,
                        visit_frequency=visit_frequency,
                        procedures=procedures,
                        inpatient_days=inpatient_days,
                    )
                    candidates.append(candidate)
                    break  # One match per phase type per block

        return candidates

    def _extract_visit_count(self, text: str) -> Optional[int]:
        """Extract number of visits from text."""
        for pattern in self.visit_re:
            m = pattern.search(text)  # Pattern already compiled with IGNORECASE
            if m and m.group(1):
                try:
                    return int(m.group(1))
                except ValueError:
                    pass
        return None

    def _extract_visit_frequency(self, text: str) -> Optional[str]:
        """Extract visit frequency (e.g., 'every 4 weeks')."""
        m = re.search(r'every\s*(\d+)\s*(weeks?|months?|days?)', text, re.IGNORECASE)
        if m:
            return f"every {m.group(1)} {m.group(2)}"

        for freq in ["weekly", "bi-weekly", "biweekly", "monthly", "quarterly"]:
            if freq in text.lower():
                return freq
        return None

    def _extract_procedures(self, text: str) -> List[str]:
        """Extract procedure/assessment names."""
        procedures = []
        matches = self.procedure_re.findall(text)
        procedures.extend(matches)
        return list(set(procedures))

    def _extract_inpatient_days(self, text: str) -> Optional[int]:
        """Extract inpatient/hospitalization requirement."""
        for pattern in self.inpatient_re:
            m = pattern.search(text)  # Pattern already compiled with IGNORECASE
            if m and m.lastindex:
                try:
                    return int(m.group(1))
                except ValueError:
                    pass
        # Check for overnight stay
        if re.search(r'overnight|24[\s-]?hour', text, re.IGNORECASE):
            return 1
        return None

    # =========================================================================
    # ENDPOINT EXTRACTION (with better boundaries)
    # =========================================================================

    def _normalize_endpoint_fingerprint(self, text: str) -> str:
        """Create normalized fingerprint for endpoint deduplication."""
        # Lowercase, remove stopwords and punctuation
        stopwords = {"was", "is", "the", "a", "an", "were", "are", "to", "at", "of", "in"}
        words = re.sub(r'[^\w\s]', '', text.lower()).split()
        return " ".join(w for w in words if w not in stopwords)[:50]

    def _extract_endpoints(
        self,
        text: str,
        doc_id: str,
        doc_fingerprint: str,
        page_num: Optional[int],
        section: str,
        seen: Set[str],
    ) -> List[FeasibilityCandidate]:
        """Extract study endpoint information with explicit type signals."""
        candidates = []

        # Skip captions
        if self.fp_filter.is_caption(text):
            return []

        # Explicit endpoint type markers (required)
        TYPE_MARKERS = {
            EndpointType.PRIMARY: [
                r"primary\s+(?:efficacy\s+)?(?:end)?point",
                r"primary\s+(?:outcome|measure)",
            ],
            EndpointType.SECONDARY: [
                r"(?:key\s+)?secondary\s+(?:end)?point",
                r"secondary\s+(?:outcome|measure)",
            ],
            EndpointType.SAFETY: [
                r"safety\s+(?:end)?point",
                r"(?:serious\s+)?adverse\s+event",
                r"treatment[\s-]?emergent",
            ],
            EndpointType.EXPLORATORY: [
                r"exploratory\s+(?:end)?point",
            ],
        }

        for endpoint_type, patterns in TYPE_MARKERS.items():
            for pattern_str in patterns:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    endpoint_text = self._extract_endpoint_bounded(text, match)

                    # Reject very short or useless extractions
                    if len(endpoint_text) < 15:
                        continue

                    # Reject if it's just the marker itself
                    if endpoint_text.lower() == match.group(0).lower():
                        continue

                    # Reject fragments like "was rejected", "primary endpoint was"
                    useless_patterns = [
                        r"^(?:was|were|is|are)\s+\w+$",
                        r"^primary\s+endpoint\s+was$",
                        r"rejected",
                    ]
                    if any(re.search(p, endpoint_text, re.IGNORECASE) for p in useless_patterns):
                        continue

                    # Normalized fingerprint for dedup
                    fingerprint = self._normalize_endpoint_fingerprint(endpoint_text)
                    dedup_key = f"endpoint:{endpoint_type.value}:{fingerprint}"
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    features = self._calculate_confidence_features(
                        FeasibilityFieldType.STUDY_ENDPOINT, section, text, endpoint_text
                    )
                    # Boost for explicit type marker
                    features.pattern_strength = 0.2

                    candidate = self._make_candidate(
                        doc_id=doc_id,
                        doc_fingerprint=doc_fingerprint,
                        matched_text=endpoint_text,
                        context_text=text[:self.context_window],
                        field_type=FeasibilityFieldType.STUDY_ENDPOINT,
                        page_num=page_num,
                        section=section,
                        confidence=features.total(),
                        confidence_features=features.to_dict(),
                    )
                    candidate.study_endpoint = StudyEndpoint(
                        endpoint_type=endpoint_type,
                        name=endpoint_text,
                    )
                    candidates.append(candidate)

        return candidates

    def _extract_endpoint_bounded(self, text: str, match: re.Match) -> str:
        """Extract endpoint with proper boundary detection."""
        start = match.end()
        remaining = text[start:].strip()

        # Strategy 1: If followed by colon, take content until section break
        if remaining.startswith(':'):
            end_patterns = [r'\n\s*\n', r'\n[A-Z][a-z]+\s*(?:endpoint|outcome)']
            for pat in end_patterns:
                m = re.search(pat, remaining)
                if m:
                    return remaining[1:m.start()].strip()[:500]

        # Strategy 2: Take first complete sentence
        sentence_end = re.search(r'(?<=[.!?])\s+(?=[A-Z])', remaining)
        if sentence_end and sentence_end.start() < 300:
            return remaining[:sentence_end.start()].strip()

        # Strategy 3: Stop at list boundary
        list_boundary = re.search(r'\n\s*(?:[•\-\d]\.?\s)', remaining)
        if list_boundary and list_boundary.start() > 20:
            return remaining[:list_boundary.start()].strip()

        # Fallback: first 200 chars to period
        return re.split(r'[.;]', remaining)[0][:200].strip()

    # =========================================================================
    # SITE EXTRACTION (with disambiguation)
    # =========================================================================

    def _normalize_country(self, country: str) -> str:
        """Normalize country name to canonical form."""
        CANONICAL = {
            "us": "United States", "usa": "United States", "u.s.": "United States",
            "united states": "United States",
            "uk": "United Kingdom", "u.k.": "United Kingdom",
            "united kingdom": "United Kingdom",
        }
        return CANONICAL.get(country.lower(), country.title())

    def _extract_sites(
        self,
        text: str,
        doc_id: str,
        doc_fingerprint: str,
        page_num: Optional[int],
        section: str,
        seen: Set[str],
    ) -> List[FeasibilityCandidate]:
        """Extract site/country information with better USA/UK handling."""
        candidates = []
        text_lower = text.lower()

        # Check if ANY site context cue exists
        has_site_context = any(cue in text_lower for cue in COUNTRY_CONTEXT_CUES)
        if not has_site_context:
            return []

        found_countries = set()  # Use set for dedup

        # Special handling for USA/UK (case-sensitive patterns)
        usa_patterns = [
            r"\bU\.?S\.?A\.?\b",  # USA, U.S.A.
            r"\bU\.S\.\b",  # U.S.
            r"\bUnited\s+States\b",
        ]
        uk_patterns = [
            r"\bU\.?K\.?\b",  # UK, U.K.
            r"\bUnited\s+Kingdom\b",
        ]

        for pattern in usa_patterns:
            if re.search(pattern, text):
                found_countries.add("united states")
                break

        for pattern in uk_patterns:
            if re.search(pattern, text):
                found_countries.add("united kingdom")
                break

        # Check other countries (but NOT 'us' or 'uk' in lowercase - likely pronouns)
        SAFE_COUNTRIES = COUNTRIES - {"us", "uk"}  # Remove ambiguous short forms
        for country in SAFE_COUNTRIES:
            if country in text_lower:
                # Extra validation for ambiguous countries
                if country in AMBIGUOUS_COUNTRIES:
                    if not self._validate_country_context(text_lower, country):
                        continue
                found_countries.add(country)

        site_count = None
        country_count = None

        for pattern in self.site_count_re:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    site_count = int(groups[0])
                    country_count = int(groups[1])
                elif len(groups) >= 1:
                    site_count = int(groups[0])
                break

        if not site_count:
            site_count_match = re.search(
                r"(\d+)\s*(?:sites?|centers?)", text_lower
            )
            if site_count_match:
                site_count = int(site_count_match.group(1))

        if found_countries:
            normalized_countries = list(set(self._normalize_country(c) for c in found_countries))

            dedup_key = f"sites:{','.join(sorted(normalized_countries)[:3])}"
            if dedup_key not in seen:
                seen.add(dedup_key)

                for country in normalized_countries[:5]:
                    features = self._calculate_confidence_features(
                        FeasibilityFieldType.STUDY_SITE, section, text, country
                    )
                    features.pattern_strength = 0.2

                    candidate = self._make_candidate(
                        doc_id=doc_id,
                        doc_fingerprint=doc_fingerprint,
                        matched_text=country,
                        context_text=text[:self.context_window],
                        field_type=FeasibilityFieldType.STUDY_SITE,
                        page_num=page_num,
                        section=section,
                        confidence=features.total(),
                        confidence_features=features.to_dict(),
                    )
                    candidate.study_site = StudySite(
                        country=country,
                        country_code=COUNTRY_CODES.get(country.lower()),
                        site_count=site_count,
                        validation_context="site_context_cue_present",
                    )
                    candidates.append(candidate)

        elif site_count:
            dedup_key = f"sites:count:{site_count}:{country_count or 0}"
            if dedup_key not in seen:
                seen.add(dedup_key)

                features = self._calculate_confidence_features(
                    FeasibilityFieldType.STUDY_SITE, section, text, f"{site_count} sites"
                )
                features.pattern_strength = 0.2

                candidate = self._make_candidate(
                    doc_id=doc_id,
                    doc_fingerprint=doc_fingerprint,
                    matched_text=f"{site_count} sites" + (f" in {country_count} countries" if country_count else ""),
                    context_text=text[:self.context_window],
                    field_type=FeasibilityFieldType.STUDY_SITE,
                    page_num=page_num,
                    section=section,
                    confidence=features.total(),
                    confidence_features=features.to_dict(),
                )
                candidate.study_site = StudySite(
                    country="multiple" if country_count else "unknown",
                    site_count=site_count,
                    validation_context="site_count_pattern",
                )
                candidates.append(candidate)

        return candidates

    def _validate_country_context(self, text: str, country: str) -> bool:
        """Validate ambiguous country with surrounding context."""
        idx = text.find(country)
        if idx == -1:
            return False

        window_start = max(0, idx - 50)
        window_end = min(len(text), idx + len(country) + 50)
        window = text[window_start:window_end]

        # Negative signals (likely a name or US state)
        if re.search(rf"(?:dr\.?|mr\.?|ms\.?|prof\.?)\s*{country}", window):
            return False
        if re.search(rf"{country}\s*(?:,\s*)?(?:usa?|america|u\.s\.)", window):
            return False  # US state like "Georgia, USA"

        # Positive signals
        return any(cue in window for cue in COUNTRY_CONTEXT_CUES)

    # =========================================================================
    # SCREENING YIELD EXTRACTION (CONSORT Flow)
    # =========================================================================

    def _extract_screening_yield(
        self,
        text: str,
        doc_id: str,
        doc_fingerprint: str,
        page_num: Optional[int],
        section: str,
        seen: Set[str],
    ) -> List[FeasibilityCandidate]:
        """Extract CONSORT flow metrics (screened, randomized, completed)."""
        candidates = []
        text_lower = text.lower()

        if self.fp_filter.is_caption(text):
            return []

        yield_data: Dict[str, Any] = {}

        for pattern in self.screening_yield_re:
            match = pattern.search(text)
            if match:
                value = match.group(1)
                matched_text = match.group(0).lower()

                try:
                    num_value = int(value) if '.' not in value else float(value)
                except ValueError:
                    continue

                if "screened" in matched_text:
                    yield_data["screened"] = num_value
                elif "randomized" in matched_text:
                    yield_data["randomized"] = num_value
                elif "enrolled" in matched_text:
                    yield_data["enrolled"] = num_value
                elif "failure" in matched_text:
                    if "rate" in matched_text:
                        yield_data["screen_failure_rate"] = num_value
                    else:
                        yield_data["screen_failures"] = num_value
                elif "discontinued" in matched_text:
                    yield_data["discontinued"] = num_value
                elif "completed" in matched_text or "completion" in matched_text:
                    if "rate" in matched_text:
                        pass
                    else:
                        yield_data["completed"] = num_value
                elif "dropout" in matched_text:
                    yield_data["dropout_rate"] = num_value

        for pattern in self.consort_flow_re:
            match = pattern.search(text)
            if match:
                value = match.group(1)
                matched_text = match.group(0).lower()

                try:
                    num_value = int(value)
                except ValueError:
                    continue

                if "eligibility" in matched_text or "screened" in matched_text:
                    yield_data["screened"] = num_value
                elif "excluded" in matched_text or "failure" in matched_text:
                    yield_data["screen_failures"] = num_value
                elif "randomized" in matched_text or "allocated" in matched_text:
                    yield_data["randomized"] = num_value
                elif "analysed" in matched_text or "analyzed" in matched_text or "completed" in matched_text:
                    yield_data["completed"] = num_value
                elif "discontinued" in matched_text or "lost" in matched_text:
                    yield_data["discontinued"] = num_value

        if yield_data:
            if yield_data.get("screened") and yield_data.get("screen_failures"):
                yield_data["screen_failure_rate"] = round(
                    yield_data["screen_failures"] / yield_data["screened"] * 100, 1
                )
            if yield_data.get("randomized") and yield_data.get("discontinued"):
                yield_data["dropout_rate"] = round(
                    yield_data["discontinued"] / yield_data["randomized"] * 100, 1
                )

            dedup_key = f"yield:{yield_data.get('screened', 0)}:{yield_data.get('randomized', 0)}"
            if dedup_key in seen:
                return []
            seen.add(dedup_key)

            features = self._calculate_confidence_features(
                FeasibilityFieldType.SCREENING_YIELD, section, text, str(yield_data)
            )
            features.pattern_strength = 0.2

            candidate = self._make_candidate(
                doc_id=doc_id,
                doc_fingerprint=doc_fingerprint,
                matched_text=text[:200],
                context_text=text[:self.context_window],
                field_type=FeasibilityFieldType.SCREENING_YIELD,
                page_num=page_num,
                section=section,
                confidence=features.total(),
                confidence_features=features.to_dict(),
            )
            candidate.screening_yield = ScreeningYield(**yield_data)
            candidates.append(candidate)

        return candidates

    # =========================================================================
    # VACCINATION REQUIREMENT EXTRACTION
    # =========================================================================

    def _extract_vaccination(
        self,
        text: str,
        doc_id: str,
        doc_fingerprint: str,
        page_num: Optional[int],
        section: str,
        seen: Set[str],
    ) -> List[FeasibilityCandidate]:
        """Extract vaccination requirements from eligibility criteria."""
        candidates = []
        text_lower = text.lower()

        if self.fp_filter.is_caption(text):
            return []

        for pattern in self.vaccination_re:
            match = pattern.search(text)
            if match:
                matched_text = match.group(0)

                vaccine_type = "unknown"
                for vt in VACCINE_TYPES:
                    if vt in text_lower:
                        vaccine_type = vt
                        break

                requirement_type = "required"
                if re.search(r"\b(?:no|without|prohibited|excluded)\b", text_lower):
                    requirement_type = "prohibited"
                elif re.search(r"\b(?:completed|received|prior)\b", text_lower):
                    requirement_type = "completed_before"

                timing = None
                timing_match = re.search(
                    r"(?:at\s*least\s*)?(\d+)\s*(weeks?|days?|months?)\s*(?:before|prior|after)",
                    text_lower
                )
                if timing_match:
                    timing = f"{timing_match.group(1)} {timing_match.group(2)}"

                dedup_key = f"vacc:{vaccine_type}:{requirement_type}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                features = self._calculate_confidence_features(
                    FeasibilityFieldType.VACCINATION_REQUIREMENT, section, text, matched_text
                )
                features.pattern_strength = 0.2

                candidate = self._make_candidate(
                    doc_id=doc_id,
                    doc_fingerprint=doc_fingerprint,
                    matched_text=matched_text,
                    context_text=text[:self.context_window],
                    field_type=FeasibilityFieldType.VACCINATION_REQUIREMENT,
                    page_num=page_num,
                    section=section,
                    confidence=features.total(),
                    confidence_features=features.to_dict(),
                )
                candidate.vaccination_requirement = VaccinationRequirement(
                    vaccine_type=vaccine_type,
                    requirement_type=requirement_type,
                    timing=timing,
                )
                candidates.append(candidate)

        return candidates

    # =========================================================================
    # CONFIDENCE CALCULATION (P2)
    # =========================================================================

    def _calculate_confidence_features(
        self,
        field_type: FeasibilityFieldType,
        section: str,
        text: str,
        match_text: str,
    ) -> ConfidenceFeatures:
        """Calculate feature-based confidence score with penalties and bonuses."""
        features = ConfidenceFeatures()

        # Section match bonus using expected sections for this field type
        expected = EXPECTED_SECTIONS.get(field_type, [])
        features.apply_section_bonus(section, expected)

        # Speculation check using shared utility
        features.apply_speculation_check(text)

        # Pattern strength (default moderate)
        features.pattern_strength = 0.1

        # PENALTY: Figure/Table captions (-0.3)
        if self.fp_filter.is_caption(text):
            features.negation_penalty = -0.3

        # PENALTY: OCR garbage / many special chars (-0.2)
        special_char_ratio = len(re.findall(r'[^\w\s]', match_text)) / max(len(match_text), 1)
        if special_char_ratio > 0.2:
            features.negation_penalty = min(features.negation_penalty, -0.2)

        # PENALTY: Wrong section for this field type
        if section not in expected and section != "unknown":
            features.section_match = -0.1

        return features

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _make_candidate(
        self,
        doc_id: str,
        doc_fingerprint: str,
        matched_text: str,
        context_text: str,
        field_type: FeasibilityFieldType,
        page_num: Optional[int],
        section: str,
        confidence: float,
        confidence_features: Optional[Dict[str, float]] = None,
        evidence_quote: Optional[str] = None,
        char_start: Optional[int] = None,
        char_end: Optional[int] = None,
    ) -> FeasibilityCandidate:
        """Create a FeasibilityCandidate with provenance and evidence."""
        provenance = FeasibilityProvenanceMetadata(
            pipeline_version=self.pipeline_version,
            run_id=self.run_id,
            doc_fingerprint=doc_fingerprint,
            generator_name=FeasibilityGeneratorType.PATTERN_MATCH,
        )

        evidence = []
        if evidence_quote or matched_text:
            evidence.append(EvidenceSpan(
                page=page_num,
                section_header=section if section != "unknown" else None,
                quote=evidence_quote or matched_text[:500],
                char_start=char_start,
                char_end=char_end,
            ))

        return FeasibilityCandidate(
            doc_id=doc_id,
            field_type=field_type,
            generator_type=FeasibilityGeneratorType.PATTERN_MATCH,
            matched_text=matched_text,
            context_text=context_text,
            page_number=page_num,
            section_name=section,
            evidence=evidence,
            extraction_method=ExtractionMethod.REGEX,
            confidence=confidence,
            confidence_features=confidence_features,
            provenance=provenance,
        )

    def _get_context(self, text: str, start: int, end: int) -> str:
        """Extract context window around a match."""
        ctx_start = max(0, start - self.context_window // 2)
        ctx_end = min(len(text), end + self.context_window // 2)
        return text[ctx_start:ctx_end]

    def _update_stats(self, candidates: List[FeasibilityCandidate]) -> None:
        """Update extraction statistics."""
        for c in candidates:
            field_type = c.field_type.value
            self._extraction_stats[field_type] = (
                self._extraction_stats.get(field_type, 0) + 1
            )

    def get_stats(self) -> Dict[str, int]:
        """Return extraction statistics."""
        return self._extraction_stats.copy()

    def print_summary(self) -> None:
        """Print extraction summary."""
        if not self._extraction_stats:
            print("\nFeasibility extraction: No items found")
            return

        total = sum(self._extraction_stats.values())
        print(f"\nFeasibility extraction: {total} items found")
        print("─" * 50)
        for field_type, count in sorted(self._extraction_stats.items()):
            print(f"  {field_type:<40} {count:>5}")
