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

from typing import Any, Dict, List, Optional

from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from A_core.A07_feasibility_models import (
    CriterionType,
    EligibilityCriterion,
    EndpointType,
    FeasibilityCandidate,
    FeasibilityFieldType,
    FeasibilityGeneratorType,
    FeasibilityProvenanceMetadata,
    StudyDesign,
    StudyEndpoint,
    StudySite,
    TreatmentArm,
)
from B_parsing.B02_doc_graph import DocumentGraph


# =============================================================================
# LLM PROMPTS
# =============================================================================

STUDY_DESIGN_PROMPT = """Extract study design information from this clinical trial document.

Return JSON with these fields (use null if not found):
{
    "phase": "1" or "2" or "3" or "2/3" or "2b" or "3b" or null,
    "design_type": "parallel" or "crossover" or "single-arm" or null,
    "blinding": "double-blind" or "single-blind" or "open-label" or null,
    "randomization_ratio": "1:1" or "2:1" or null,
    "sample_size": integer or null (planned enrollment),
    "actual_enrollment": integer or null (actual number randomized),
    "duration_months": integer or null (total study duration),
    "treatment_arms": [
        {"name": "Drug name", "dose": "200mg", "frequency": "twice daily", "route": "oral"},
        {"name": "Placebo", "dose": null}
    ],
    "control_type": "placebo" or "active" or "standard of care" or null
}

IMPORTANT for phase: Look for "phase 2" or "phase 3" in the study title, abstract, or methods.
If the document mentions both phases (e.g., reporting phase 2 results while discussing phase 3 plans),
extract the phase of THIS study being reported, not future planned phases.

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
            "analysis_method": "How analyzed (e.g., 'log-transformed ratio to baseline')" or null
        }
    ]
}

IMPORTANT:
- Extract ALL secondary endpoints (typically 3-5 in a clinical trial)
- Common secondary endpoints: eGFR change, composite endpoints, patient-reported outcomes, biomarkers
- Safety endpoints: adverse events, treatment-emergent AEs, serious AEs
- Use concise names, not full sentences

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
    "regions": ["Europe", "North America", "Asia", etc.] or null
}

IMPORTANT:
- List ALL countries where the trial was conducted
- Use full country names (e.g., "United States" not "US", "United Kingdom" not "UK")
- Common countries in multinational trials: United States, Germany, France, Italy, Spain, Japan, etc.

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

        # Content limits for LLM context (20k to include full methods section)
        self.max_content_chars = int(self.config.get("max_content_chars", 20000))

        # Stats
        self._extraction_stats: Dict[str, int] = {}

    def extract(
        self,
        doc_graph: DocumentGraph,
        doc_id: str,
        doc_fingerprint: str,
        full_text: Optional[str] = None,
    ) -> List[FeasibilityCandidate]:
        """
        Extract feasibility information using LLM.

        Args:
            doc_graph: Document graph (for future use)
            doc_id: Document identifier
            doc_fingerprint: Document fingerprint
            full_text: Pre-built full text (optional, built from doc_graph if not provided)

        Returns list of FeasibilityCandidate with structured data.
        """
        candidates: List[FeasibilityCandidate] = []

        # Get document text
        if not full_text:
            # Build from doc_graph
            full_text = " ".join(
                block.text for block in doc_graph.iter_linear_blocks() if block.text
            )
        if not full_text:
            return candidates

        # Truncate for LLM context
        content = full_text[:self.max_content_chars]

        # Extract each category
        candidates.extend(self._extract_study_design(content, doc_id, doc_fingerprint))
        candidates.extend(self._extract_eligibility(content, doc_id, doc_fingerprint))
        candidates.extend(self._extract_endpoints(content, doc_id, doc_fingerprint))
        candidates.extend(self._extract_sites(content, doc_id, doc_fingerprint))

        return candidates

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
                    dose=arm_data.get("dose"),
                    frequency=arm_data.get("frequency"),
                    route=arm_data.get("route"),
                ))

        study_design = StudyDesign(
            phase=response.get("phase"),
            design_type=response.get("design_type"),
            blinding=response.get("blinding"),
            randomization_ratio=response.get("randomization_ratio"),
            sample_size=response.get("sample_size"),
            actual_enrollment=response.get("actual_enrollment"),
            duration_months=response.get("duration_months"),
            treatment_arms=arms,
            control_type=response.get("control_type"),
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

        return [FeasibilityCandidate(
            doc_id=doc_id,
            field_type=FeasibilityFieldType.STUDY_DESIGN,
            generator_type=FeasibilityGeneratorType.LLM_EXTRACTION,
            matched_text=summary_text,
            context_text="",
            section_name="study_design",
            confidence=0.9,
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

            text = crit_data.get("text", "").strip()
            if not text or len(text) < 10:
                continue

            crit_type_str = crit_data.get("type", "inclusion").lower()
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

            candidates.append(FeasibilityCandidate(
                doc_id=doc_id,
                field_type=field_type,
                generator_type=FeasibilityGeneratorType.LLM_EXTRACTION,
                matched_text=text,
                context_text="",
                section_name="eligibility",
                confidence=0.85,
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

            name = ep_data.get("name", "").strip()
            if not name or len(name) < 5:
                continue

            # Map type string to enum
            type_str = ep_data.get("type", "primary").lower()
            type_map = {
                "primary": EndpointType.PRIMARY,
                "secondary": EndpointType.SECONDARY,
                "exploratory": EndpointType.EXPLORATORY,
                "safety": EndpointType.SAFETY,
            }
            endpoint_type = type_map.get(type_str, EndpointType.PRIMARY)

            endpoint = StudyEndpoint(
                endpoint_type=endpoint_type,
                name=name,
                measure=ep_data.get("measure"),
                timepoint=ep_data.get("timepoint"),
                analysis_method=ep_data.get("analysis_method"),
            )

            candidates.append(FeasibilityCandidate(
                doc_id=doc_id,
                field_type=FeasibilityFieldType.STUDY_ENDPOINT,
                generator_type=FeasibilityGeneratorType.LLM_EXTRACTION,
                matched_text=name,
                context_text="",
                section_name="endpoints",
                confidence=0.88,
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
                confidence=0.85,
                study_site=site,
                provenance=self._make_provenance(doc_fingerprint),
            ))

            self._extraction_stats["site"] = self._extraction_stats.get("site", 0) + 1

        return candidates

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
