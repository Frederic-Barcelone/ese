# corpus_metadata/C_generators/C08a_feasibility_patterns.py
"""
Pattern constants for clinical trial feasibility extraction.

Contains all regex patterns and constants used by FeasibilityDetector:
- Eligibility criteria patterns
- Epidemiology patterns
- Patient journey patterns
- Endpoint patterns
- Site/country patterns
- Screening yield patterns
"""

from __future__ import annotations

import re
from typing import Dict, List

from A_core.A07_feasibility_models import (
    EndpointType,
    FeasibilityFieldType,
    PatientJourneyPhaseType,
)


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

JOURNEY_PHASE_KEYWORDS: Dict[PatientJourneyPhaseType, List[str]] = {
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

ENDPOINT_PATTERNS: Dict[EndpointType, List[str]] = {
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
# OPERATIONAL BURDEN PATTERNS (Generic - works across disease areas)
# =============================================================================

INVASIVE_PROCEDURE_PATTERNS = [
    r"(?:\w+\s+)?biops(?:y|ies)(?:\s+of\s+\w+)?",
    r"(?:\w+\s+)?aspirat(?:e|ion)(?:\s+of\s+\w+)?",
    r"lumbar\s*puncture",
    r"(?:\w+)?scop(?:y|ic|ies)",
    r"(?:central\s*)?(?:venous|arterial)\s*(?:catheter|line|access)",
    r"(?:surgical|invasive)\s*(?:procedure|intervention)",
    r"(?:blood|tissue)\s*sampl(?:e|ing)",
    r"infusion\s*(?:therapy|treatment)",
]

VISIT_SCHEDULE_PATTERNS = [
    r"(?:day|week|month)\s*(\d+)(?:\s*[,/]\s*(?:day|week|month)\s*(\d+))+",
    r"visits?\s*(?:at|on)\s*(?:day|week|month)s?\s*(\d+(?:\s*[,/and]+\s*\d+)*)",
    r"every\s*(\d+)\s*(?:days?|weeks?|months?)",
    r"(\d+)\s*(?:study\s*)?visits?\s*(?:over|during)\s*(\d+)\s*(?:weeks?|months?|years?)",
    r"follow[\s-]?up\s*(?:at|for|every)\s*(\d+)\s*(?:days?|weeks?|months?|years?)",
    r"(?:weekly|monthly|quarterly|annual)\s*(?:visits?|assessments?)",
]

BACKGROUND_THERAPY_PATTERNS = [
    r"(?:stable\s*)?(?:dose\s*(?:of\s*)?)?(.+?)\s*(?:for\s*)?(?:≥|>=|at\s*least)\s*(\d+)\s*(?:days?|weeks?|months?)",
    r"(?:background|concomitant|prior)\s*(?:therapy|treatment|medication)(?:\s+(?:with|of))?\s*(.+?)(?:\s+for|\s*$)",
    r"(?:stable|unchanged)\s+(?:dose|regimen)\s+(?:of\s+)?(.+?)\s+(?:for\s+)?(\d+)\s*(?:days?|weeks?|months?)",
    r"(?:prohibited|not\s*permitted|excluded|disallowed)(?:\s+.+?)?(?:vaccin|therap|medicat|treatment)",
    r"washout\s*(?:period\s*)?(?:of\s*)?(\d+)\s*(?:days?|weeks?|months?)",
]

LAB_CRITERION_PATTERNS = [
    r"([A-Za-z][A-Za-z0-9/-]{1,20})\s*(?:level\s*)?(?:≥|>=|>|≤|<=|<|=)\s*(\d+(?:[.,]\d+)?)\s*([a-zA-Z/%·×\^0-9/-]+)",
    r"(?:serum|plasma|blood|urine)\s+([A-Za-z][A-Za-z0-9/-]+)\s*(?:≥|>=|>|≤|<=|<)\s*(\d+(?:[.,]\d+)?)",
    r"([A-Za-z]+)\s*(?:of|at)\s*(?:≥|>=|>|≤|<=|<)\s*(\d+(?:[.,]\d+)?)\s*(\w+(?:/\w+)?)",
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

EXPECTED_SECTIONS: Dict[FeasibilityFieldType, List[str]] = {
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


__all__ = [
    # Epidemiology
    "EPIDEMIOLOGY_ANCHORS",
    "PREVALENCE_PATTERNS",
    "INCIDENCE_PATTERNS",
    "DEMOGRAPHICS_PATTERNS",
    "GEOGRAPHY_PATTERNS",
    "TIME_PATTERNS",
    "SETTING_PATTERNS",
    # Eligibility
    "INCLUSION_MARKERS",
    "EXCLUSION_MARKERS",
    "CRITERION_CATEGORIES",
    # Patient Journey
    "JOURNEY_PHASE_KEYWORDS",
    "DURATION_PATTERN",
    "VISIT_PATTERNS",
    "PROCEDURE_PATTERNS",
    "INPATIENT_PATTERNS",
    # Endpoints
    "ENDPOINT_PATTERNS",
    # Screening Yield
    "SCREENING_YIELD_PATTERNS",
    "CONSORT_FLOW_PATTERNS",
    # Vaccination
    "VACCINATION_PATTERNS",
    "VACCINE_TYPES",
    # Sites/Countries
    "SITE_COUNT_PATTERNS",
    "AMBIGUOUS_COUNTRIES",
    "COUNTRY_CONTEXT_CUES",
    "COUNTRIES",
    "COUNTRY_CODES",
    # Operational Burden
    "INVASIVE_PROCEDURE_PATTERNS",
    "VISIT_SCHEDULE_PATTERNS",
    "BACKGROUND_THERAPY_PATTERNS",
    "LAB_CRITERION_PATTERNS",
    # Confidence
    "EXPECTED_SECTIONS",
]
