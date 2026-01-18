# corpus_metadata/corpus_metadata/C_generators/C11_llm_feasibility.py
"""
LLM-based clinical trial feasibility extraction.

Uses Claude to extract structured feasibility data:
1. Study design (phase, sample size, randomization, arms)
2. Eligibility criteria (structured inclusion/exclusion)
3. Endpoints (primary, secondary with measures/timepoints)
4. Sites/Countries (full country list)

This replaces the noisy regex-based extraction with high-quality structured output.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from A_core.A07_feasibility_models import (
    BackgroundTherapy,
    CriterionType,
    EligibilityCriterion,
    EndpointType,
    EvidenceSpan,
    ExtractionMethod,
    FeasibilityCandidate,
    FeasibilityFieldType,
    FeasibilityGeneratorType,
    FeasibilityProvenanceMetadata,
    InvasiveProcedure,
    OperationalBurden,
    ScreenFailReason,
    ScreeningFlow,
    StudyDesign,
    StudyEndpoint,
    StudyFootprint,
    StudySite,
    TreatmentArm,
    VaccinationRequirement,
    VisitSchedule,
)
from B_parsing.B02_doc_graph import DocumentGraph
from B_parsing.B05_section_detector import SectionDetector
from D_validation.D05_quote_verifier import (
    ExtractionVerifier,
    QuoteVerifier,
    NumericalVerifier,
)


# =============================================================================
# SECTION-TO-PROMPT MAPPING
# =============================================================================

# Maps each extraction type to relevant sections (in priority order)
SECTION_TARGETS = {
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
"""

STUDY_DESIGN_PROMPT = """Extract study design information from this clinical trial document.

Return JSON with these fields (use null if not found):
{
    "phase": "1" or "2" or "3" or "2/3" or "2b" or "3b" or null,
    "design_type": "parallel" or "crossover" or "single-arm" or null,
    "blinding": "double-blind" or "single-blind" or "open-label" or null,
    "randomization_ratio": "1:1" or "2:1" or null,
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
            "exact_quote": "REQUIRED: Copy-paste the exact text from the document describing this endpoint"
        }
    ]
}

IMPORTANT:
- Extract ALL secondary endpoints (typically 3-5 in a clinical trial)
- Common secondary endpoints: eGFR change, composite endpoints, patient-reported outcomes, biomarkers
- Safety endpoints: adverse events, treatment-emergent AEs, serious AEs
- Use concise names, not full sentences
- The exact_quote field MUST contain verbatim text from the document
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
- Invasive procedures (biopsies, aspirations, catheterizations)
- Visit schedule intensity (number of visits, frequency)
- Vaccination/prophylaxis requirements
- Background therapy requirements (stable dose requirements)
- Concomitant medications allowed (SGLT2 inhibitors, immunosuppressants, etc.)
- Run-in period requirements
- Special sample handling requirements
- Central laboratory requirements

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
            "quote": "exact text from document describing this requirement"
        }
    ],
    "visit_schedule": {
        "total_visits": integer or null,
        "visit_days": [1, 14, 28, 56, 84, ...] or null (legacy flat list),
        "frequency": "every 4 weeks" or "monthly" etc,
        "duration_weeks": integer or null,
        "pre_randomization_days": [-75, -15, 1] (days before/at randomization),
        "on_treatment_days": [14, 30, 90, 180] (scheduled on-treatment visits),
        "follow_up_days": [210, 270, 360] (post-treatment or open-label visits),
        "scheduled_visits": [
            {"day": 14, "visit_name": "Week 2", "phase": "double_blind"},
            {"day": 180, "visit_name": "Month 6", "phase": "double_blind"},
            {"day": 360, "visit_name": "Month 12", "phase": "open_label"}
        ]
    },
    "vaccination_requirements": [
        {
            "vaccine_type": "meningococcal" or "pneumococcal" etc,
            "requirement_type": "required" or "prohibited",
            "timing": "at least 2 weeks before treatment" or null,
            "quote": "exact text"
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
            "quote": "exact text"
        }
    ],
    "concomitant_meds_allowed": [
        {
            "therapy_class": "SGLT2 inhibitors",
            "requirement_type": "allowed",
            "stable_duration_days": 90,
            "quote": "exact text"
        }
    ],
    "run_in_duration_days": integer or null,
    "run_in_requirements": ["list of run-in requirements"],
    "central_lab_required": true/false,
    "central_lab": {
        "required": true/false,
        "analytes": ["UPCR", "eGFR", "serum C3", "sC5b-9"],
        "quote": "exact text mentioning central laboratory"
    },
    "special_sample_handling": ["frozen samples", "timed urine collection"] or [],
    "hard_gates": ["biopsy requirement", "vaccination", "rare lab threshold"] - criteria most likely to limit enrollment
}

IMPORTANT:
- Include EXACT quotes from the document for each procedure/requirement WITH PAGE NUMBERS if visible
- Distinguish "diagnosis confirmation" biopsies (eligibility) from "study-required" biopsies (operational burden)
- For visit schedule, distinguish pre-randomization visits from on-treatment visits
- For concomitant meds, extract what's ALLOWED (affects who qualifies and SOC background)
- For central lab, cite evidence if the document mentions central laboratory assessment
- Hard gates are the top 3-5 criteria most likely to exclude patients

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
- Include page numbers in evidence where visible

Return JSON only."""


# =============================================================================
# LLM FEASIBILITY EXTRACTOR
# =============================================================================


class LLMFeasibilityExtractor:
    """
    LLM-based feasibility information extractor.

    Uses Claude to extract structured feasibility data with high precision.
    """

    def __init__(
        self,
        llm_client: Any,
        llm_model: str = "claude-sonnet-4-20250514",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.config = config or {}
        self.run_id = str(self.config.get("run_id") or generate_run_id("FEAS_LLM"))
        self.pipeline_version = (
            self.config.get("pipeline_version") or get_git_revision_hash()
        )

        # Section detector for targeted extraction
        self.section_detector = SectionDetector()

        # Anti-hallucination verification
        self.quote_verifier = QuoteVerifier(fuzzy_threshold=0.90)
        self.numerical_verifier = NumericalVerifier()
        self.extraction_verifier = ExtractionVerifier(
            fuzzy_threshold=0.90,
            numerical_tolerance=0.0,
        )

        # Stats
        self._extraction_stats: Dict[str, int] = {}
        self._verification_stats: Dict[str, int] = {
            "quotes_verified": 0,
            "quotes_failed": 0,
            "numbers_verified": 0,
            "numbers_failed": 0,
            "missing_fields": 0,
        }

    def extract(
        self,
        doc_graph: DocumentGraph,
        doc_id: str,
        doc_fingerprint: str,
        full_text: Optional[str] = None,
    ) -> List[FeasibilityCandidate]:
        """
        Extract feasibility information using LLM with section-targeted extraction.

        Runs all 6 extraction types in parallel for ~6x speedup.

        Args:
            doc_graph: Document graph for section-aware extraction
            doc_id: Document identifier
            doc_fingerprint: Document fingerprint
            full_text: Pre-built full text (fallback if doc_graph unavailable)

        Returns list of FeasibilityCandidate with structured data.
        """
        candidates: List[FeasibilityCandidate] = []

        # Build section map from doc_graph
        section_map = self._build_section_map(doc_graph)

        # Define extraction tasks: (method, extraction_type)
        extraction_tasks = [
            (self._extract_study_design, "study_design"),
            (self._extract_eligibility, "eligibility"),
            (self._extract_endpoints, "endpoints"),
            (self._extract_sites, "sites"),
            (self._extract_operational_burden, "operational_burden"),
            (self._extract_screening_flow, "screening_flow"),
        ]

        # Run all extractions in parallel
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {}
            for method, extraction_type in extraction_tasks:
                content = self._get_targeted_content(extraction_type, section_map, full_text)
                future = executor.submit(method, content, doc_id, doc_fingerprint)
                futures[future] = extraction_type

            # Collect results as they complete
            for future in as_completed(futures):
                extraction_type = futures[future]
                try:
                    result = future.result()
                    candidates.extend(result)
                except Exception as e:
                    print(f"[WARN] LLM extraction failed for {extraction_type}: {e}")

        return candidates

    def _build_section_map(self, doc_graph: DocumentGraph) -> Dict[str, str]:
        """
        Build a map of section names to their text content.

        Returns dict like {"abstract": "text...", "methods": "text...", ...}
        """
        section_map: Dict[str, List[str]] = {}
        current_section = "preamble"

        for block in doc_graph.iter_linear_blocks():
            if not block.text:
                continue

            # Check if this block is a section header
            section_info = self.section_detector.detect(block.text, block)
            if section_info:
                current_section = section_info.name

            # Add text to current section
            if current_section not in section_map:
                section_map[current_section] = []
            section_map[current_section].append(block.text)

        # Join text for each section
        return {
            section: " ".join(texts)[:MAX_SECTION_CHARS]
            for section, texts in section_map.items()
        }

    def _get_targeted_content(
        self,
        extraction_type: str,
        section_map: Dict[str, str],
        fallback_text: Optional[str] = None,
    ) -> str:
        """
        Get content targeted for a specific extraction type.

        Combines relevant sections up to MAX_TOTAL_CHARS.
        """
        target_sections = SECTION_TARGETS.get(extraction_type, ["abstract", "methods"])
        content_parts = []
        total_chars = 0

        for section in target_sections:
            if section in section_map:
                section_text = section_map[section]
                if total_chars + len(section_text) <= MAX_TOTAL_CHARS:
                    content_parts.append(f"[{section.upper()}]\n{section_text}")
                    total_chars += len(section_text)
                else:
                    # Add partial section to fill remaining space
                    remaining = MAX_TOTAL_CHARS - total_chars
                    if remaining > 500:  # Only add if meaningful amount left
                        content_parts.append(f"[{section.upper()}]\n{section_text[:remaining]}")
                    break

        if content_parts:
            return "\n\n".join(content_parts)

        # Fallback to full text if no sections matched
        if fallback_text:
            return fallback_text[:MAX_TOTAL_CHARS]

        return ""

    def _call_llm(self, system_prompt: str, content: str) -> Optional[Dict[str, Any]]:
        """Call LLM and parse JSON response."""
        if not self.llm_client:
            return None

        user_prompt = f"Document content:\n\n{content}\n\n---\nExtract the requested information. Return JSON only."

        try:
            if hasattr(self.llm_client, "complete_json_any"):
                response = self.llm_client.complete_json_any(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.llm_model,
                    temperature=0.0,
                    max_tokens=2000,
                )
            else:
                response = self.llm_client.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.llm_model,
                    temperature=0.0,
                    max_tokens=2000,
                )

            if isinstance(response, list) and len(response) > 0:
                response = response[0]

            return response if isinstance(response, dict) else None
        except Exception as e:
            print(f"[WARN] LLM feasibility extraction failed: {e}")
            return None

    def _make_provenance(self, doc_fingerprint: str) -> FeasibilityProvenanceMetadata:
        """Create provenance metadata."""
        return FeasibilityProvenanceMetadata(
            pipeline_version=self.pipeline_version,
            run_id=self.run_id,
            doc_fingerprint=doc_fingerprint,
            generator_name=FeasibilityGeneratorType.LLM_EXTRACTION,
        )

    def _verify_quote(self, quote: Optional[str], context: str) -> bool:
        """
        Verify that a quote exists in the context.

        Returns True if verified, False otherwise.
        Updates verification stats.
        """
        if not quote or not context:
            return False

        result = self.quote_verifier.verify(quote, context)
        if result.verified:
            self._verification_stats["quotes_verified"] += 1
        else:
            self._verification_stats["quotes_failed"] += 1
        return result.verified

    def _verify_number(self, value: Optional[int | float], context: str) -> bool:
        """
        Verify that a numerical value exists in the context.

        Returns True if verified, False otherwise.
        Updates verification stats.
        """
        if value is None or not context:
            return False

        result = self.numerical_verifier.verify(value, context)
        if result.verified:
            self._verification_stats["numbers_verified"] += 1
        else:
            self._verification_stats["numbers_failed"] += 1
        return result.verified

    def _apply_verification_penalty(
        self,
        base_confidence: float,
        quote: Optional[str],
        context: str,
        numerical_values: Optional[Dict[str, int | float]] = None,
    ) -> float:
        """
        Apply confidence penalties based on verification results.

        NO FALLBACKS: Missing or unverified evidence results in penalties.

        Args:
            base_confidence: Starting confidence score (0.0 - 1.0).
            quote: Quote to verify. If None/empty, penalty applies.
            context: Source document text.
            numerical_values: Dict of field -> value to verify.

        Returns:
            Adjusted confidence score with penalties applied.
            - Missing quote: confidence × 0.5
            - Unverified quote: confidence × 0.5
            - Unverified number: confidence × 0.7 (per unverified number)
        """
        confidence = base_confidence

        # Quote verification: missing or unverified quote = penalty
        if not quote or not quote.strip():
            # No quote provided - apply penalty
            self._verification_stats["quotes_failed"] += 1
            confidence *= 0.5
        elif not self._verify_quote(quote, context):
            # Quote provided but not found in context
            confidence *= 0.5

        # Verify numerical values if provided
        if numerical_values:
            for field_name, value in numerical_values.items():
                if value is not None and not self._verify_number(value, context):
                    confidence *= 0.7

        return round(confidence, 3)

    def _check_required_fields(
        self,
        data: Dict[str, Any],
        required_fields: List[str],
        context_name: str,
    ) -> int:
        """
        Check for missing required fields and track them.

        Args:
            data: The data dict to check
            required_fields: List of field names that should be present
            context_name: Name for logging (e.g., "eligibility_criterion")

        Returns:
            Count of missing required fields
        """
        missing_count = 0
        for field in required_fields:
            value = data.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing_count += 1
                self._verification_stats["missing_fields"] += 1

        return missing_count

    def _apply_completeness_penalty(
        self,
        base_confidence: float,
        missing_field_count: int,
        penalty_per_field: float = 0.05,
    ) -> float:
        """
        Apply confidence penalty for missing fields.

        Args:
            base_confidence: Starting confidence
            missing_field_count: Number of missing required fields
            penalty_per_field: Penalty per missing field (default 5%)

        Returns:
            Adjusted confidence (minimum 0.1)
        """
        if missing_field_count <= 0:
            return base_confidence

        penalty = missing_field_count * penalty_per_field
        return max(0.1, base_confidence - penalty)

    # =========================================================================
    # STUDY DESIGN EXTRACTION
    # =========================================================================

    def _extract_study_design(
        self,
        content: str,
        doc_id: str,
        doc_fingerprint: str,
    ) -> List[FeasibilityCandidate]:
        """Extract study design using LLM."""
        response = self._call_llm(STUDY_DESIGN_PROMPT, content)
        if not response:
            return []

        # Parse treatment arms
        arms = []
        for arm_data in (response.get("treatment_arms") or []):
            if isinstance(arm_data, dict) and arm_data.get("name"):
                arms.append(TreatmentArm(
                    name=arm_data["name"],
                    n=arm_data.get("n"),
                    dose=arm_data.get("dose"),
                    frequency=arm_data.get("frequency"),
                    route=arm_data.get("route"),
                ))

        # Parse study periods
        from A_core.A07_feasibility_models import StudyPeriod
        periods = []
        for period_data in (response.get("periods") or []):
            if isinstance(period_data, dict) and period_data.get("name"):
                periods.append(StudyPeriod(
                    name=period_data["name"],
                    duration_months=period_data.get("duration_months"),
                    duration_weeks=period_data.get("duration_weeks"),
                    description=period_data.get("description"),
                ))

        # Parse evidence
        evidence = []
        for ev_data in (response.get("evidence") or []):
            if isinstance(ev_data, dict) and ev_data.get("quote"):
                evidence.append(EvidenceSpan(
                    page=ev_data.get("page"),
                    quote=ev_data["quote"],
                ))

        study_design = StudyDesign(
            phase=response.get("phase"),
            design_type=response.get("design_type"),
            blinding=response.get("blinding"),
            randomization_ratio=response.get("randomization_ratio"),
            allocation=response.get("allocation"),
            sample_size=response.get("sample_size"),
            actual_enrollment=response.get("actual_enrollment"),
            duration_months=response.get("duration_months"),
            treatment_arms=arms,
            control_type=response.get("control_type"),
            setting=response.get("setting"),
            sites_total=response.get("sites_total"),
            countries_total=response.get("countries_total"),
            periods=periods,
            evidence=evidence,
        )

        # Only return if we have meaningful data
        if not any([
            study_design.phase,
            study_design.sample_size,
            study_design.blinding,
            study_design.treatment_arms,
        ]):
            return []

        self._extraction_stats["study_design"] = 1

        # Create summary text
        summary_parts = []
        if study_design.phase:
            summary_parts.append(f"Phase {study_design.phase}")
        if study_design.blinding:
            summary_parts.append(study_design.blinding)
        if study_design.sample_size:
            summary_parts.append(f"N={study_design.sample_size}")
        summary_text = ", ".join(summary_parts) or "Study design extracted"

        # Apply verification and confidence penalty
        # Collect first evidence quote for verification
        first_quote = evidence[0].quote if evidence else None
        numerical_values = {}
        if study_design.sample_size:
            numerical_values["sample_size"] = study_design.sample_size
        if study_design.actual_enrollment:
            numerical_values["actual_enrollment"] = study_design.actual_enrollment

        confidence = self._apply_verification_penalty(
            base_confidence=0.9,
            quote=first_quote,
            context=content,
            numerical_values=numerical_values,
        )

        return [FeasibilityCandidate(
            doc_id=doc_id,
            field_type=FeasibilityFieldType.STUDY_DESIGN,
            generator_type=FeasibilityGeneratorType.LLM_EXTRACTION,
            matched_text=summary_text,
            context_text="",
            section_name="study_design",
            confidence=confidence,
            study_design=study_design,
            provenance=self._make_provenance(doc_fingerprint),
        )]

    # =========================================================================
    # ELIGIBILITY EXTRACTION
    # =========================================================================

    def _extract_eligibility(
        self,
        content: str,
        doc_id: str,
        doc_fingerprint: str,
    ) -> List[FeasibilityCandidate]:
        """Extract eligibility criteria using LLM."""
        response = self._call_llm(ELIGIBILITY_PROMPT, content)
        if not response:
            return []

        candidates = []
        criteria_list = response.get("criteria") or []

        for crit_data in criteria_list:
            if not isinstance(crit_data, dict):
                continue

            # Check required fields - no silent defaults
            text = crit_data.get("text")
            if not text or not isinstance(text, str) or len(text.strip()) < 10:
                self._verification_stats["missing_fields"] += 1
                continue

            text = text.strip()

            # Check for missing type field (required, no default)
            crit_type_str = crit_data.get("type")
            if not crit_type_str:
                self._verification_stats["missing_fields"] += 1
                continue
            crit_type_str = crit_type_str.lower()
            crit_type = CriterionType.EXCLUSION if crit_type_str == "exclusion" else CriterionType.INCLUSION
            field_type = (
                FeasibilityFieldType.ELIGIBILITY_EXCLUSION
                if crit_type == CriterionType.EXCLUSION
                else FeasibilityFieldType.ELIGIBILITY_INCLUSION
            )

            # Build derived variables from structured extraction
            derived = {}
            if crit_data.get("operator"):
                derived["operator"] = crit_data["operator"]
            if crit_data.get("value") is not None:
                derived["value"] = crit_data["value"]
            if crit_data.get("unit"):
                derived["unit"] = crit_data["unit"]

            criterion = EligibilityCriterion(
                criterion_type=crit_type,
                text=text,
                category=crit_data.get("category"),
                derived_variables=derived if derived else None,
            )

            # Apply verification and confidence penalty
            # No fallback: if exact_quote is missing, penalty will apply
            exact_quote = crit_data.get("exact_quote")
            numerical_values = {}
            if crit_data.get("value") is not None:
                numerical_values["value"] = crit_data["value"]

            confidence = self._apply_verification_penalty(
                base_confidence=0.85,
                quote=exact_quote,
                context=content,
                numerical_values=numerical_values,
            )

            candidates.append(FeasibilityCandidate(
                doc_id=doc_id,
                field_type=field_type,
                generator_type=FeasibilityGeneratorType.LLM_EXTRACTION,
                matched_text=text,
                context_text="",
                section_name="eligibility",
                confidence=confidence,
                eligibility_criterion=criterion,
                provenance=self._make_provenance(doc_fingerprint),
            ))

            self._extraction_stats[field_type.value] = self._extraction_stats.get(field_type.value, 0) + 1

        return candidates

    # =========================================================================
    # ENDPOINTS EXTRACTION
    # =========================================================================

    def _extract_endpoints(
        self,
        content: str,
        doc_id: str,
        doc_fingerprint: str,
    ) -> List[FeasibilityCandidate]:
        """Extract study endpoints using LLM."""
        response = self._call_llm(ENDPOINTS_PROMPT, content)
        if not response:
            return []

        candidates = []
        endpoints_list = response.get("endpoints") or []

        for ep_data in endpoints_list:
            if not isinstance(ep_data, dict):
                continue

            # Check required fields - no silent defaults
            name = ep_data.get("name")
            if not name or not isinstance(name, str) or len(name.strip()) < 5:
                self._verification_stats["missing_fields"] += 1
                continue
            name = name.strip()

            # Check required type field - no default fallback
            type_str = ep_data.get("type")
            if not type_str:
                self._verification_stats["missing_fields"] += 1
                continue

            type_map = {
                "primary": EndpointType.PRIMARY,
                "secondary": EndpointType.SECONDARY,
                "exploratory": EndpointType.EXPLORATORY,
                "safety": EndpointType.SAFETY,
            }
            endpoint_type = type_map.get(type_str.lower())
            if not endpoint_type:
                self._verification_stats["missing_fields"] += 1
                continue

            endpoint = StudyEndpoint(
                endpoint_type=endpoint_type,
                name=name,
                measure=ep_data.get("measure"),
                timepoint=ep_data.get("timepoint"),
                analysis_method=ep_data.get("analysis_method"),
            )

            # Apply verification and confidence penalty
            exact_quote = ep_data.get("exact_quote")
            confidence = self._apply_verification_penalty(
                base_confidence=0.88,
                quote=exact_quote,
                context=content,
            )

            candidates.append(FeasibilityCandidate(
                doc_id=doc_id,
                field_type=FeasibilityFieldType.STUDY_ENDPOINT,
                generator_type=FeasibilityGeneratorType.LLM_EXTRACTION,
                matched_text=name,
                context_text="",
                section_name="endpoints",
                confidence=confidence,
                study_endpoint=endpoint,
                provenance=self._make_provenance(doc_fingerprint),
            ))

            self._extraction_stats["endpoint"] = self._extraction_stats.get("endpoint", 0) + 1

        return candidates

    # =========================================================================
    # SITES EXTRACTION
    # =========================================================================

    def _extract_sites(
        self,
        content: str,
        doc_id: str,
        doc_fingerprint: str,
    ) -> List[FeasibilityCandidate]:
        """Extract site/country information using LLM."""
        response = self._call_llm(SITES_PROMPT, content)
        if not response:
            return []

        candidates = []
        countries = response.get("countries") or []
        total_sites = response.get("total_sites")
        total_countries = response.get("total_countries")
        exact_quote = response.get("exact_quote")

        # Apply verification penalty once for the sites extraction
        numerical_values = {}
        if total_sites:
            numerical_values["total_sites"] = total_sites
        if total_countries:
            numerical_values["total_countries"] = total_countries

        confidence = self._apply_verification_penalty(
            base_confidence=0.85,
            quote=exact_quote,
            context=content,
            numerical_values=numerical_values,
        )

        for country in countries:
            if not country or not isinstance(country, str):
                continue

            site = StudySite(
                country=country,
                site_count=total_sites if len(countries) == 1 else None,
            )

            candidates.append(FeasibilityCandidate(
                doc_id=doc_id,
                field_type=FeasibilityFieldType.STUDY_SITE,
                generator_type=FeasibilityGeneratorType.LLM_EXTRACTION,
                matched_text=country,
                context_text=f"{total_sites or '?'} sites in {total_countries or len(countries)} countries",
                section_name="sites",
                confidence=confidence,
                study_site=site,
                provenance=self._make_provenance(doc_fingerprint),
            ))

            self._extraction_stats["site"] = self._extraction_stats.get("site", 0) + 1

        return candidates

    # =========================================================================
    # OPERATIONAL BURDEN EXTRACTION
    # =========================================================================

    def _extract_operational_burden(
        self,
        content: str,
        doc_id: str,
        doc_fingerprint: str,
    ) -> List[FeasibilityCandidate]:
        """Extract operational burden using LLM."""
        response = self._call_llm(OPERATIONAL_BURDEN_PROMPT, content)
        if not response:
            return []

        candidates = []

        # Parse invasive procedures with enhanced fields
        procedures = []
        for proc_data in (response.get("invasive_procedures") or []):
            if isinstance(proc_data, dict) and proc_data.get("name"):
                evidence = []
                if proc_data.get("quote"):
                    evidence.append(EvidenceSpan(
                        quote=proc_data["quote"],
                        page=proc_data.get("page"),
                    ))
                procedures.append(InvasiveProcedure(
                    name=proc_data["name"],
                    timing=proc_data.get("timing", []),
                    timing_days=proc_data.get("timing_days", []),
                    optional=proc_data.get("optional", False),
                    purpose=proc_data.get("purpose"),
                    is_eligibility_requirement=proc_data.get("is_eligibility_requirement", False),
                    evidence=evidence,
                ))

        # Parse visit schedule with phase-structured visits
        from A_core.A07_feasibility_models import ScheduledVisit
        visit_data = response.get("visit_schedule") or {}
        visit_schedule = None
        if visit_data:
            # Parse scheduled visits with phase information
            scheduled_visits = []
            for sv_data in (visit_data.get("scheduled_visits") or []):
                if isinstance(sv_data, dict) and sv_data.get("day") is not None:
                    scheduled_visits.append(ScheduledVisit(
                        day=sv_data["day"],
                        visit_name=sv_data.get("visit_name"),
                        phase=sv_data.get("phase"),
                        procedures=sv_data.get("procedures", []),
                    ))

            visit_schedule = VisitSchedule(
                total_visits=visit_data.get("total_visits"),
                visit_days=visit_data.get("visit_days", []),
                frequency=visit_data.get("frequency"),
                duration_weeks=visit_data.get("duration_weeks"),
                scheduled_visits=scheduled_visits,
                pre_randomization_days=visit_data.get("pre_randomization_days", []),
                on_treatment_days=visit_data.get("on_treatment_days", []),
                follow_up_days=visit_data.get("follow_up_days", []),
            )

        # Parse vaccination requirements
        vaccinations = []
        for vacc_data in (response.get("vaccination_requirements") or []):
            if isinstance(vacc_data, dict) and vacc_data.get("vaccine_type"):
                evidence = []
                if vacc_data.get("quote"):
                    evidence.append(EvidenceSpan(quote=vacc_data["quote"]))
                vaccinations.append(VaccinationRequirement(
                    vaccine_type=vacc_data["vaccine_type"],
                    requirement_type=vacc_data.get("requirement_type", "required"),
                    timing=vacc_data.get("timing"),
                    evidence=evidence,
                ))

        # Parse background therapy with requirement_type
        bg_therapy = []
        for bg_data in (response.get("background_therapy") or []):
            if isinstance(bg_data, dict) and bg_data.get("therapy_class"):
                evidence = []
                if bg_data.get("quote"):
                    evidence.append(EvidenceSpan(quote=bg_data["quote"]))
                bg_therapy.append(BackgroundTherapy(
                    therapy_class=bg_data["therapy_class"],
                    requirement_type=bg_data.get("requirement_type", "allowed"),
                    requirement=bg_data.get("requirement", ""),
                    agents=bg_data.get("agents", []),
                    stable_duration_days=bg_data.get("stable_duration_days"),
                    max_dose=bg_data.get("max_dose"),
                    evidence=evidence,
                ))

        # Parse concomitant meds allowed (feasibility-critical)
        concomitant_allowed = []
        for cm_data in (response.get("concomitant_meds_allowed") or []):
            if isinstance(cm_data, dict) and cm_data.get("therapy_class"):
                evidence = []
                if cm_data.get("quote"):
                    evidence.append(EvidenceSpan(quote=cm_data["quote"]))
                concomitant_allowed.append(BackgroundTherapy(
                    therapy_class=cm_data["therapy_class"],
                    requirement_type="allowed",
                    requirement=cm_data.get("requirement", "allowed"),
                    agents=cm_data.get("agents", []),
                    stable_duration_days=cm_data.get("stable_duration_days"),
                    max_dose=cm_data.get("max_dose"),
                    evidence=evidence,
                ))

        # Parse central lab requirement with evidence
        from A_core.A07_feasibility_models import CentralLabRequirement
        central_lab = None
        central_lab_data = response.get("central_lab")
        if isinstance(central_lab_data, dict):
            evidence = []
            if central_lab_data.get("quote"):
                evidence.append(EvidenceSpan(quote=central_lab_data["quote"]))
            central_lab = CentralLabRequirement(
                required=central_lab_data.get("required", False),
                analytes=central_lab_data.get("analytes", []),
                confidence=0.8 if evidence else 0.5,
                evidence=evidence,
            )

        # Build OperationalBurden
        burden = OperationalBurden(
            invasive_procedures=procedures,
            visit_schedule=visit_schedule,
            vaccination_requirements=vaccinations,
            background_therapy=bg_therapy,
            concomitant_meds_allowed=concomitant_allowed,
            run_in_duration_days=response.get("run_in_duration_days"),
            run_in_requirements=response.get("run_in_requirements", []),
            central_lab_required=response.get("central_lab_required", False),
            central_lab=central_lab,
            special_sample_handling=response.get("special_sample_handling", []),
            hard_gates=response.get("hard_gates", []),
        )

        # Only return if we have meaningful data
        has_data = (
            procedures or vaccinations or bg_therapy or concomitant_allowed or
            visit_schedule or burden.run_in_duration_days or
            burden.hard_gates or central_lab
        )
        if not has_data:
            return []

        self._extraction_stats["operational_burden"] = 1

        summary_parts = []
        if procedures:
            summary_parts.append(f"{len(procedures)} procedures")
        if vaccinations:
            summary_parts.append(f"{len(vaccinations)} vaccinations")
        if bg_therapy:
            summary_parts.append(f"{len(bg_therapy)} background therapies")
        if burden.hard_gates:
            summary_parts.append(f"{len(burden.hard_gates)} hard gates")

        # Collect quotes from procedures for verification
        first_quote = None
        for proc in procedures:
            if proc.evidence:
                first_quote = proc.evidence[0].quote
                break

        # Collect numerical values
        numerical_values = {}
        if burden.run_in_duration_days:
            numerical_values["run_in_duration_days"] = burden.run_in_duration_days
        if visit_schedule and visit_schedule.total_visits:
            numerical_values["total_visits"] = visit_schedule.total_visits

        confidence = self._apply_verification_penalty(
            base_confidence=0.85,
            quote=first_quote,
            context=content,
            numerical_values=numerical_values,
        )

        candidates.append(FeasibilityCandidate(
            doc_id=doc_id,
            field_type=FeasibilityFieldType.OPERATIONAL_BURDEN,
            generator_type=FeasibilityGeneratorType.LLM_EXTRACTION,
            matched_text=", ".join(summary_parts) or "Operational burden extracted",
            context_text="",
            section_name="operational_burden",
            extraction_method=ExtractionMethod.LLM,
            confidence=confidence,
            operational_burden=burden,
            provenance=self._make_provenance(doc_fingerprint),
        ))

        return candidates

    # =========================================================================
    # SCREENING FLOW EXTRACTION
    # =========================================================================

    def _extract_screening_flow(
        self,
        content: str,
        doc_id: str,
        doc_fingerprint: str,
    ) -> List[FeasibilityCandidate]:
        """Extract CONSORT flow / screening information using LLM."""
        response = self._call_llm(SCREENING_FLOW_PROMPT, content)
        if not response:
            return []

        # Parse screen fail reasons with enhanced fields (overlap, dual percentages)
        screen_fail_reasons = []
        for sfr_data in (response.get("screen_fail_reasons") or []):
            if isinstance(sfr_data, dict) and sfr_data.get("reason"):
                evidence = []
                if sfr_data.get("quote"):
                    evidence.append(EvidenceSpan(
                        quote=sfr_data["quote"],
                        page=sfr_data.get("page"),
                    ))
                screen_fail_reasons.append(ScreenFailReason(
                    reason=sfr_data["reason"],
                    count=sfr_data.get("count"),
                    percentage_reported=sfr_data.get("percentage_reported") or sfr_data.get("percentage"),
                    percentage_computed=sfr_data.get("percentage_computed"),
                    can_overlap=sfr_data.get("can_overlap", False),
                    evidence=evidence,
                ))

        # Calculate derived metrics
        screened = response.get("screened")
        randomized = response.get("randomized")
        screen_failures = response.get("screen_failures")
        discontinued = response.get("discontinued")

        screening_yield = None
        screen_failure_rate_computed = None
        dropout_rate = None

        if screened and randomized and screened > 0:
            screening_yield = round(randomized / screened, 4)
        if screened and screen_failures and screened > 0:
            screen_failure_rate_computed = round(screen_failures / screened * 100, 2)
        if randomized and discontinued and randomized > 0:
            dropout_rate = round(discontinued / randomized, 4)

        # Parse evidence
        evidence = []
        for ev_data in (response.get("evidence") or []):
            if isinstance(ev_data, dict) and ev_data.get("quote"):
                evidence.append(EvidenceSpan(
                    page=ev_data.get("page"),
                    quote=ev_data["quote"],
                ))

        # Build ScreeningFlow with enhanced fields
        flow = ScreeningFlow(
            planned_sample_size=response.get("planned_sample_size"),
            actual_enrollment=randomized,
            screened=screened,
            screen_failures=screen_failures,
            randomized=randomized,
            treated=response.get("treated"),
            completed=response.get("completed"),
            discontinued=discontinued,
            screening_yield=screening_yield,
            screen_failure_rate=screen_failure_rate_computed,  # Legacy field
            screen_failure_rate_reported=response.get("screen_failure_rate_reported"),
            screen_failure_rate_computed=screen_failure_rate_computed,
            dropout_rate=dropout_rate,
            screen_fail_reasons=screen_fail_reasons,
            reasons_can_overlap=response.get("reasons_can_overlap", False),
            run_in_failures=response.get("run_in_failures"),
            run_in_failure_reasons=[
                ScreenFailReason(reason=r)
                for r in (response.get("run_in_failure_reasons") or [])
                if r
            ],
            evidence=evidence,
        )

        # Only return if we have meaningful data
        has_data = (
            flow.screened or flow.randomized or flow.completed or
            screen_fail_reasons
        )
        if not has_data:
            return []

        self._extraction_stats["screening_flow"] = 1

        summary_parts = []
        if flow.screened:
            summary_parts.append(f"{flow.screened} screened")
        if flow.randomized:
            summary_parts.append(f"{flow.randomized} randomized")
        if screening_yield:
            summary_parts.append(f"{screening_yield*100:.1f}% yield")
        if screen_fail_reasons:
            summary_parts.append(f"{len(screen_fail_reasons)} fail reasons")

        # Collect first evidence quote for verification
        first_quote = evidence[0].quote if evidence else None

        # Collect numerical values
        numerical_values = {}
        if screened:
            numerical_values["screened"] = screened
        if randomized:
            numerical_values["randomized"] = randomized
        if screen_failures:
            numerical_values["screen_failures"] = screen_failures

        confidence = self._apply_verification_penalty(
            base_confidence=0.9,
            quote=first_quote,
            context=content,
            numerical_values=numerical_values,
        )

        return [FeasibilityCandidate(
            doc_id=doc_id,
            field_type=FeasibilityFieldType.SCREENING_FLOW,
            generator_type=FeasibilityGeneratorType.LLM_EXTRACTION,
            matched_text=", ".join(summary_parts) or "Screening flow extracted",
            context_text="",
            section_name="screening_flow",
            extraction_method=ExtractionMethod.LLM,
            confidence=confidence,
            screening_flow=flow,
            provenance=self._make_provenance(doc_fingerprint),
        )]

    def get_stats(self) -> Dict[str, int]:
        """Return extraction statistics."""
        return self._extraction_stats.copy()

    def print_summary(self) -> None:
        """Print extraction summary."""
        if not self._extraction_stats:
            print("\nLLM Feasibility extraction: No items found")
            return

        total = sum(self._extraction_stats.values())
        print(f"\nLLM Feasibility extraction: {total} items found")
        print("─" * 50)
        for field_type, count in sorted(self._extraction_stats.items()):
            print(f"  {field_type:<40} {count:>5}")

        # Print verification stats
        if any(self._verification_stats.values()):
            print("\nVerification statistics:")
            print("─" * 50)
            quotes_verified = self._verification_stats.get("quotes_verified", 0)
            quotes_failed = self._verification_stats.get("quotes_failed", 0)
            numbers_verified = self._verification_stats.get("numbers_verified", 0)
            numbers_failed = self._verification_stats.get("numbers_failed", 0)

            total_quotes = quotes_verified + quotes_failed
            total_numbers = numbers_verified + numbers_failed

            if total_quotes > 0:
                quote_rate = quotes_verified / total_quotes * 100
                print(f"  Quotes verified: {quotes_verified}/{total_quotes} ({quote_rate:.1f}%)")
            if total_numbers > 0:
                number_rate = numbers_verified / total_numbers * 100
                print(f"  Numbers verified: {numbers_verified}/{total_numbers} ({number_rate:.1f}%)")

            missing_fields = self._verification_stats.get("missing_fields", 0)
            if missing_fields > 0:
                print(f"  Missing required fields: {missing_fields} (items skipped)")
