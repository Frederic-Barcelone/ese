# corpus_metadata/C_generators/C29_feasibility_prompts.py
"""
LLM prompt templates for clinical trial feasibility extraction.

Contains:
- Section targeting configuration
- Anti-hallucination instructions
- Extraction prompts for study design, eligibility, endpoints, sites,
  operational burden, and screening flow
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
]
