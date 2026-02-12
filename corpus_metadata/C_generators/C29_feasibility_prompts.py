"""
LLM prompt templates for clinical trial feasibility extraction.

This module contains prompt templates and configuration for LLM-based
feasibility extraction. Provides section targeting, anti-hallucination
safeguards, and structured extraction prompts for all feasibility categories.

Key Components:
    - SECTION_TARGETS: Maps extraction types to relevant document sections
    - MAX_SECTION_CHARS: Maximum characters per section for LLM context
    - MAX_TOTAL_CHARS: Total character limit for LLM prompts
    - ANTI_HALLUCINATION_INSTRUCTIONS: Rules preventing LLM fabrication
    - Extraction prompts:
        - STUDY_DESIGN_PROMPT: Phase, sample size, randomization
        - ELIGIBILITY_PROMPT: Inclusion/exclusion criteria
        - ENDPOINTS_PROMPT: Primary/secondary endpoints
        - SITES_PROMPT: Site and country extraction
        - OPERATIONAL_BURDEN_PROMPT: Procedures, visits, vaccinations
        - SCREENING_FLOW_PROMPT: CONSORT data, screen failures

Example:
    >>> from C_generators.C29_feasibility_prompts import SECTION_TARGETS, ELIGIBILITY_PROMPT
    >>> SECTION_TARGETS["eligibility"]
    ['eligibility', 'methods', 'abstract']

Dependencies:
    None (pure constant definitions)
"""

from __future__ import annotations

from typing import Dict, List


# =============================================================================
# SECTION-TO-PROMPT MAPPING
# =============================================================================

# Maps each extraction type to relevant sections (in priority order)
SECTION_TARGETS: Dict[str, List[str]] = {
    "study_design": ["abstract", "methods", "eligibility"],
    "eligibility": ["eligibility", "methods", "abstract"],
    "endpoints": ["endpoints", "methods", "results", "abstract"],
    "sites": ["methods", "abstract", "results"],
    "operational_burden": ["patient_journey", "methods", "eligibility"],
    "screening_flow": ["results", "patient_journey", "abstract"],
    "epidemiology": ["abstract", "introduction", "methods", "results"],
    "patient_population": ["introduction", "methods", "results", "discussion"],
    "local_guidelines": ["introduction", "discussion", "epidemiology", "methods", "guidelines", "recommendations", "abstract", "patient_journey", "results"],
    "patient_journey": ["patient_journey", "methods", "eligibility", "results", "abstract"],
}

# Maximum chars per section to include
MAX_SECTION_CHARS = 8000

# Total max chars to send to LLM per extraction
MAX_TOTAL_CHARS = 15000


# =============================================================================
# LLM PROMPTS
# =============================================================================

# Anti-hallucination instruction block (appended to prompts)
ANTI_HALLUCINATION_INSTRUCTIONS = """

CRITICAL ANTI-HALLUCINATION RULES:
1. For EVERY extracted value, you MUST provide an exact_quote field with the EXACT text from the document that supports it.
2. The exact_quote must be a verbatim copy-paste from the document - DO NOT paraphrase or summarize.
3. If you cannot find explicit text supporting a value, return null for that field.
4. NEVER invent, guess, or infer values that are not explicitly stated in the document.
5. For numerical values (sample_size, counts, percentages), the exact number MUST appear in the document.
6. If the value is implicit or requires calculation, set the field to null.
7. PAGE NUMBERS: The document contains [PAGE X] markers. Always include the page number with each evidence quote.
   Look for the nearest [PAGE X] marker before the quote to determine the page number.
"""

STUDY_DESIGN_PROMPT = """Extract study design information from this clinical trial document.

Return JSON with these fields (use null if not found):
{
    "phase": "1" or "2" or "3" or "2/3" or "2b" or "3b" or null,
    "design_type": "parallel" or "crossover" or "single-arm" or null,
    "blinding": "double-blind" or "single-blind" or "open-label" or null,
    "randomization_ratio": "1:1" or "2:1" or "3:1" etc (IMPORTANT: look for "randomised 1:1" or "randomized 2:1" - extract the exact ratio),
    "allocation": "iptacopan n=38, placebo n=36" (describe how participants were allocated to arms),
    "sample_size": integer or null (planned enrollment),
    "actual_enrollment": integer or null (actual number randomized),
    "duration_months": integer or null (total study duration),
    "treatment_arms": [
        {"name": "Drug name", "n": 38, "dose": "200mg", "frequency": "twice daily", "route": "oral"},
        {"name": "Placebo", "n": 36, "dose": null}
    ],
    "control_type": "placebo" or "active" or "standard of care" or null,
    "setting": "35 hospitals/medical centres in 18 countries" (study setting description),
    "sites_total": integer or null,
    "countries_total": integer or null,
    "periods": [
        {"name": "double_blind", "duration_months": 6},
        {"name": "open_label", "duration_months": 6}
    ],
    "evidence": [
        {"page": integer or null, "quote": "exact text supporting this extraction"}
    ]
}

IMPORTANT for phase: Look for "phase 2" or "phase 3" in the study title, abstract, or methods.
If the document mentions both phases (e.g., reporting phase 2 results while discussing phase 3 plans),
extract the phase of THIS study being reported, not future planned phases.

IMPORTANT: Extract n (number randomized) for each treatment arm if available.
IMPORTANT: Extract study periods - many trials have distinct phases (e.g., double-blind + open-label extension).
IMPORTANT: Include evidence quotes with page numbers where possible.
IMPORTANT: Each evidence quote MUST be verbatim copy-paste from the document.
""" + ANTI_HALLUCINATION_INSTRUCTIONS + """
Focus on extracting ACTUAL values from the text. Return JSON only."""

ELIGIBILITY_PROMPT = """Extract ALL eligibility criteria from this clinical trial document.

You MUST extract BOTH inclusion AND exclusion criteria. Look for:
- "Inclusion criteria" or "Key inclusion" sections
- "Exclusion criteria" or "Key exclusion" sections
- Eligibility requirements in Methods section

For each criterion, provide:
{
    "criteria": [
        {
            "type": "inclusion" or "exclusion",
            "category": "age" | "diagnosis" | "biomarker" | "lab_value" | "prior_treatment" | "comorbidity" | "organ_function" | "pregnancy" | "consent" | "other",
            "text": "The exact criterion text from the document",
            "exact_quote": "REQUIRED: Copy-paste the exact text from the document",
            "page": integer (from nearest [PAGE X] marker before the quote),
            "operator": ">=" | "<=" | ">" | "<" | "=" | "range" | "boolean" | null,
            "value": numeric value if applicable or null,
            "unit": "years" | "mg/dL" | "mL/min" | etc or null
        }
    ]
}

IMPORTANT:
- Include ALL exclusion criteria (transplant history, disease progression, contraindications, etc.)
- Extract at least 3-5 criteria of each type if present
- Look for numbered lists or bullet points in eligibility sections
- The exact_quote field MUST contain verbatim text from the document - not paraphrased
- ALWAYS include the page number from the nearest [PAGE X] marker
""" + ANTI_HALLUCINATION_INSTRUCTIONS + """
Return JSON only."""

ENDPOINTS_PROMPT = """Extract ALL study endpoints from this clinical trial document.

You MUST extract PRIMARY, SECONDARY, and SAFETY endpoints. Look for:
- "Primary endpoint" or "Primary outcome"
- "Secondary endpoints" or "Key secondary"
- "Safety endpoints" or "Adverse events"

For each endpoint, provide:
{
    "endpoints": [
        {
            "type": "primary" | "secondary" | "exploratory" | "safety",
            "name": "Short descriptive name (e.g., 'Proteinuria reduction')",
            "measure": "What is measured (e.g., '24-hour UPCR')",
            "timepoint": "When measured (e.g., '6 months')",
            "analysis_method": "How analyzed (e.g., 'log-transformed ratio to baseline')" or null,
            "exact_quote": "REQUIRED: Copy-paste the exact text from the document describing this endpoint",
            "page": integer (from nearest [PAGE X] marker before the quote)
        }
    ]
}

IMPORTANT:
- Extract ALL secondary endpoints (typically 3-5 in a clinical trial)
- Common secondary endpoints: eGFR change, composite endpoints, patient-reported outcomes, biomarkers
- Safety endpoints: adverse events, treatment-emergent AEs, serious AEs
- Use concise names, not full sentences
- The exact_quote field MUST contain verbatim text from the document
- ALWAYS include the page number from the nearest [PAGE X] marker
""" + ANTI_HALLUCINATION_INSTRUCTIONS + """
Return JSON only."""

SITES_PROMPT = """Extract study site and country information from this clinical trial document.

Look for:
- "conducted at X sites" or "X centers"
- "in X countries" or lists of countries
- Author affiliations may indicate study sites
- Methods section typically lists sites/countries

Return:
{
    "total_sites": integer or null (e.g., "18 sites" → 18),
    "total_countries": integer or null,
    "countries": ["Country1", "Country2", ...],
    "regions": ["Europe", "North America", "Asia", etc.] or null,
    "exact_quote": "REQUIRED: Copy-paste the exact text mentioning sites/countries"
}

IMPORTANT:
- List ALL countries where the trial was conducted
- Use full country names (e.g., "United States" not "US", "United Kingdom" not "UK")
- Common countries in multinational trials: United States, Germany, France, Italy, Spain, Japan, etc.
- The exact_quote MUST contain the verbatim text listing sites/countries
""" + ANTI_HALLUCINATION_INSTRUCTIONS + """
Return JSON only."""


OPERATIONAL_BURDEN_PROMPT = """Extract operational burden information from this clinical trial document.

This is CRITICAL for feasibility assessment. Look for:
- Invasive procedures (biopsies, aspirations, catheterizations) - ONLY from protocol/methods, with explicit timing
- Visit schedule intensity (look for "Scheduled study visits occurred at..." or similar)
- Vaccination/prophylaxis requirements
- Background therapy requirements - ONLY from explicit protocol statements like "allowed if stable for X days"
- Central laboratory requirements - look for explicit "assessed at central laboratory" statements
- Run-in period requirements

Return:
{
    "invasive_procedures": [
        {
            "name": "renal biopsy" or "bone marrow aspirate" etc,
            "timing": ["screening day 45", "month 6"] or ["baseline"],
            "timing_days": [45, 180] (days relative to randomization, negative for pre-randomization),
            "optional": false,
            "purpose": "diagnosis_confirmation" | "efficacy_assessment" | "safety_monitoring",
            "is_eligibility_requirement": true (if this confirms prior diagnosis, not a study procedure),
            "quote": "exact text from document describing this requirement",
            "page": integer (from nearest [PAGE X] marker)
        }
    ],
    "visit_schedule": {
        "total_visits": integer or null,
        "visit_days": [14, 30, 90, 180, 210, 270, 360] (from explicit "visits occurred at days X, Y, Z" statement),
        "frequency": "every 4 weeks" or "monthly" etc,
        "duration_weeks": integer or null,
        "on_treatment_days": [14, 30, 90, 180] (scheduled on-treatment visits),
        "quote": "exact text listing visit schedule",
        "page": integer
    },
    "vaccination_requirements": [
        {
            "vaccine_type": "meningococcal" or "pneumococcal" etc,
            "requirement_type": "required" or "prohibited",
            "timing": "at least 2 weeks before treatment" or null,
            "quote": "exact text",
            "page": integer
        }
    ],
    "background_therapy": [
        {
            "therapy_class": "ACE inhibitor/ARB" or "immunosuppressant" etc,
            "requirement_type": "required" | "allowed" | "prohibited",
            "requirement": "stable dose ≥90 days" or "prohibited",
            "agents": ["lisinopril", "losartan"] or [],
            "stable_duration_days": 90 or null,
            "max_dose": "≤7.5 mg prednisone equivalent" or null,
            "quote": "exact text FROM PROTOCOL/METHODS stating the requirement - NOT from baseline characteristics",
            "page": integer
        }
    ],
    "concomitant_meds_allowed": [
        {
            "therapy_class": "SGLT2 inhibitors",
            "requirement_type": "allowed",
            "stable_duration_days": 90,
            "quote": "exact text FROM PROTOCOL stating 'allowed' or 'permitted' - NOT from baseline usage",
            "page": integer
        }
    ],
    "run_in_duration_days": integer or null,
    "run_in_requirements": ["list of run-in requirements"],
    "central_lab_required": true/false/null (ONLY set true if you find EXPLICIT text like "assessed at central laboratory" or "central lab"),
    "central_lab": {
        "required": true/false/null (ONLY true if explicit "central laboratory" statement found),
        "analytes": ["UPCR", "eGFR", "serum C3", "sC5b-9"] (only if central lab confirmed),
        "quote": "REQUIRED: exact text containing 'central laboratory' - NOT 'normal range' or 'reference range'",
        "page": integer
    },
    "special_sample_handling": ["frozen samples", "24-h urine collection", "first morning void"] or [],
    "hard_gates": ["biopsy requirement", "vaccination", "rare lab threshold"]
}

CRITICAL EVIDENCE RULES:
- ALWAYS include page number from nearest [PAGE X] marker
- For background_therapy and concomitant_meds_allowed: ONLY use quotes from PROTOCOL/METHODS sections that explicitly state "allowed", "permitted", or "required". Do NOT infer allowed meds from baseline characteristics or screen failure footnotes.
- For central_lab: ONLY set required=true if you find explicit text containing "central laboratory" or "centralized lab". Do NOT use quotes about "normal range" or "reference range" - that doesn't prove central processing. If no explicit central lab statement, set central_lab_required=null.
- For invasive_procedures: Include BOTH eligibility requirements (e.g., "biopsy-confirmed") AND scheduled study procedures (e.g., "renal biopsy at day 45 and month 6"). Mark is_eligibility_requirement=true for the former.
- For visit_schedule: Look for explicit "Scheduled study visits occurred at days X, Y, Z" or similar. If not found, leave visit_days empty. Set duration_weeks=null if you cannot determine from explicit text.
- For special_sample_handling: Look for "24-h urine", "24-hour urine collection", "first morning void", "timed urine", frozen samples, etc.
""" + ANTI_HALLUCINATION_INSTRUCTIONS + """
Return JSON only."""


SCREENING_FLOW_PROMPT = """Extract CONSORT flow and screening information from this clinical trial document.

Look for:
- Patient disposition or CONSORT flow diagram data
- "X patients screened", "Y randomized", "Z completed"
- Screen failure reasons and counts (often in trial profile/CONSORT figures)
- Discontinuation reasons

Return:
{
    "planned_sample_size": integer or null (target enrollment),
    "screened": integer or null,
    "screen_failures": integer or null,
    "randomized": integer or null,
    "treated": integer or null,
    "completed": integer or null,
    "discontinued": integer or null,
    "screen_failure_rate_reported": float or null (percentage stated in document, e.g., 44 for "44%"),
    "screen_failure_rate_computed": float or null (computed: screen_failures / screened * 100),
    "reasons_can_overlap": true/false (set true if document indicates participants could fail multiple criteria),
    "screen_fail_reasons": [
        {
            "reason": "Did not meet serum C3 criterion" or "UPCR threshold not met" etc,
            "count": integer or null,
            "percentage_reported": float or null (what document explicitly states),
            "percentage_computed": float or null (calculated from count/total),
            "can_overlap": true (if this reason can co-occur with others),
            "quote": "exact text if available",
            "page": integer or null
        }
    ],
    "discontinuation_reasons": [
        {
            "reason": "Adverse event" or "Withdrew consent" etc,
            "count": integer or null
        }
    ],
    "run_in_failures": integer or null,
    "run_in_failure_reasons": ["list of reasons"] or [],
    "evidence": [
        {"page": integer or null, "quote": "exact text supporting these numbers"}
    ]
}

IMPORTANT:
- Extract ALL screen failure reasons if listed (often in supplementary tables, figures, or footnotes)
- Screen failure breakdown is critical for feasibility - helps predict enrollment difficulty
- Look for specific gate failures: serum C3, UPCR, eGFR, biopsy confirmation, etc.
- If document says "some participants had >1 reason", set reasons_can_overlap: true and can_overlap: true on each reason
- Distinguish percentage_reported (what document says, e.g., "58 (44%)") from percentage_computed
- ALWAYS include page numbers from nearest [PAGE X] marker for each evidence quote
- Include page numbers in evidence where visible

Return JSON only."""


EPIDEMIOLOGY_PROMPT = """Extract ALL epidemiology and disease burden data from this document.

Look for:
- Prevalence rates (e.g., "1 in 40,000", "25 per million", "0.5%")
- Incidence rates (e.g., "1.2 per 100,000 person-years")
- Mortality data (e.g., "5-year survival 65%", "mortality rate 2.3%")
- Demographics (age at onset, sex ratio, ethnic distribution)
- Geographic variation in disease frequency
- Disease burden statistics (hospitalizations, DALYs)

Return:
{
    "epidemiology": [
        {
            "data_type": "prevalence" | "incidence" | "mortality" | "demographics" | "burden",
            "value": "1 in 40,000" (the exact value as stated in the document),
            "normalized_per_million": 25.0 (normalize to per-million if possible, else null),
            "geography": "Germany" | "worldwide" | "Europe" | null,
            "population": "general population" | "children" | "adults" | "males" | null,
            "time_period": "2023" | "2015-2020" | null,
            "source": "Fabry Registry" | "population screening" | "WHO" | null,
            "exact_quote": "REQUIRED: verbatim text from the document",
            "page": integer (from nearest [PAGE X] marker)
        }
    ]
}

IMPORTANT:
- Extract ALL epidemiology data mentioned, including from introduction, methods, and discussion sections
- Include both disease-specific and population-level statistics
- For prevalence/incidence, always try to compute normalized_per_million:
  * "1 in 40,000" → 25.0 per million
  * "3 per 100,000" → 30.0 per million
  * "0.001%" → 10.0 per million
- Geography is critical - always extract the country or region if mentioned
- The exact_quote MUST be verbatim text from the document
""" + ANTI_HALLUCINATION_INSTRUCTIONS + """
Return JSON only."""


PATIENT_POPULATION_PROMPT = """Extract patient population and recruitment feasibility data from this document.

Look for:
- Estimated number of diagnosed patients (in country/region)
- Estimated number of trial-eligible patients (subset of diagnosed)
- Patient registry information (name, size, geographic coverage)
- Diagnostic delay (time from symptom onset to confirmed diagnosis)
- Referral centres / centres of expertise (count, names, locations)
- Geographic distribution of patient population
- Recruitment projections or historical recruitment rates from similar trials

Return:
{
    "patient_population": {
        "estimated_diagnosed_patients": integer or null,
        "estimated_eligible_patients": integer or null,
        "eligibility_funnel_ratio": float or null (eligible / diagnosed, e.g., 0.28),
        "registry_name": "Fabry Registry (Sanofi Genzyme)" or null,
        "registry_size": integer or null (total patients in registry),
        "diagnostic_delay_years": float or null (e.g., 13.7),
        "referral_centres": integer or null (count of specialist centres),
        "referral_centre_names": ["name1", "name2"] or [],
        "geographic_distribution": "descriptive text about where patients are located" or null,
        "recruitment_rate_per_site_month": float or null (patients per site per month),
        "evidence": [
            {"page": integer, "quote": "exact text supporting this extraction"}
        ]
    }
}

IMPORTANT:
- Diagnostic delay is the time between first symptoms and confirmed diagnosis - look for phrases like "average delay of X years", "time to diagnosis"
- Registry information may include disease registries, national databases, or patient organizations
- Referral centres are specialist hospitals or clinics with expertise in the disease
- Geographic distribution describes where patients are concentrated (e.g., "urban centres", "university hospitals")
- Recruitment rates help predict enrollment feasibility
- The exact_quote in evidence MUST be verbatim text from the document
""" + ANTI_HALLUCINATION_INSTRUCTIONS + """
Return JSON only."""


LOCAL_GUIDELINES_PROMPT = """Extract local or national clinical guideline references and their impact on trial feasibility.

Look for:
- Named clinical practice guidelines (e.g., "NICE guidelines", "German Society for Nephrology Guidelines 2023")
- Treatment recommendations from professional societies or government bodies
- Standard-of-care protocols referenced in the document
- Guideline-mandated testing or monitoring that overlaps with trial procedures
- Impact of guidelines on trial feasibility (e.g., "guideline requires genetic testing, aligning with eligibility criteria")

Return:
{
    "local_guidelines": [
        {
            "guideline_name": "Full name of the guideline",
            "issuing_body": "Organization or society name" or null,
            "year": integer or null,
            "country": "Country where guideline applies" or null,
            "key_recommendations": [
                "First key recommendation relevant to feasibility",
                "Second key recommendation"
            ],
            "standard_of_care": "Description of the standard treatment pathway" or null,
            "impact_on_feasibility": "How this guideline affects trial feasibility" or null,
            "exact_quote": "REQUIRED: verbatim text from the document referencing this guideline",
            "page": integer (from nearest [PAGE X] marker)
        }
    ]
}

IMPORTANT:
- Extract ALL guidelines referenced in the document, not just the primary one
- Include treatment algorithms, diagnostic pathways, and monitoring recommendations
- The impact_on_feasibility field is critical: explain whether the guideline HELPS (e.g., pre-existing testing infrastructure) or HINDERS (e.g., guideline prohibits randomization to placebo) trial feasibility
- The exact_quote MUST be verbatim text from the document
""" + ANTI_HALLUCINATION_INSTRUCTIONS + """
Return JSON only."""


PATIENT_JOURNEY_PROMPT = """Extract the complete patient journey from this clinical trial document.

This includes geographic context (country/region), the diagnostic pathway (how patients are diagnosed),
treatment pathway (current standard of care), trial participation phases (screening, treatment, follow-up),
and barriers to participation.

GEOGRAPHIC CONTEXT IS CRITICAL: Always identify the country and region where this study/treatment takes place.
Patient journeys vary significantly by country (local guidelines, specialist availability, healthcare system).

Return:
{
    "country": "Germany" or "United Kingdom" etc or null (country where the study/care takes place),
    "region": "Europe" or "Asia-Pacific" or "North America" etc or null,
    "diagnostic_pathway": {
        "diagnostic_delay_years": float or null (average time from symptoms to diagnosis),
        "diagnostic_tests_required": ["test1", "test2"] (tests needed for confirmed diagnosis),
        "specialist_type": "nephrologist" or "metabolic disease specialist" etc or null
    },
    "treatment_pathway": {
        "current_standard_of_care": "Description of current standard treatment" or null,
        "treatment_lines": [
            {"line": 1, "therapy": "First-line treatment description"},
            {"line": 2, "therapy": "Second-line treatment description"}
        ]
    },
    "trial_phases": [
        {
            "phase": "screening" | "run_in" | "treatment" | "follow_up" | "extension",
            "duration": "4 weeks" or "18 months" etc,
            "visits": integer or null (number of visits in this phase),
            "visit_frequency": "every 2 weeks" or "monthly" etc or null,
            "procedures": ["blood draw", "ECG", "renal biopsy"] (key procedures in this phase)
        }
    ],
    "participation_barriers": [
        "renal biopsy requirement",
        "biweekly IV infusion schedule",
        "travel to specialist centre"
    ],
    "evidence": [
        {"page": integer, "quote": "exact text supporting this extraction"}
    ]
}

IMPORTANT:
- COUNTRY/REGION: Always extract the geographic context. Look for country names in author affiliations, study sites,
  guideline references, and institution names. For multinational studies, identify the primary country.
  Use full country names (e.g., "United Kingdom" not "UK", "United States" not "US").
- Diagnostic delay: Look for "average delay of X years", "time to diagnosis", "diagnostic odyssey"
- Standard of care: Look for "current treatment", "standard therapy", "first-line treatment"
- Treatment lines: Extract all lines of therapy mentioned (1st line, 2nd line, etc.)
- Trial phases: Extract screening period, run-in, treatment period, follow-up — with visit counts and procedures
- Participation barriers: Identify factors that make trial participation difficult:
  * Invasive procedures (biopsies, IV infusions)
  * Frequent visits or travel burden
  * Washout requirements from current therapy
  * Rare disease logistics (few specialist centres)
  * Restrictive eligibility criteria
  * Country-specific barriers (healthcare system, insurance, regulatory)
- The exact_quote in evidence MUST be verbatim text from the document
""" + ANTI_HALLUCINATION_INSTRUCTIONS + """
Return JSON only."""


__all__ = [
    "SECTION_TARGETS",
    "MAX_SECTION_CHARS",
    "MAX_TOTAL_CHARS",
    "ANTI_HALLUCINATION_INSTRUCTIONS",
    "STUDY_DESIGN_PROMPT",
    "ELIGIBILITY_PROMPT",
    "ENDPOINTS_PROMPT",
    "SITES_PROMPT",
    "OPERATIONAL_BURDEN_PROMPT",
    "SCREENING_FLOW_PROMPT",
    "EPIDEMIOLOGY_PROMPT",
    "PATIENT_POPULATION_PROMPT",
    "LOCAL_GUIDELINES_PROMPT",
    "PATIENT_JOURNEY_PROMPT",
]
