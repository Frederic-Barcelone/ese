"""
Response parsing mixin for LLM feasibility extraction.

This module provides methods for parsing LLM JSON responses into structured
FeasibilityCandidate objects. Handles all feasibility extraction categories
with validation and error recovery for malformed responses.

Key Components:
    - FeasibilityResponseParserMixin: Mixin class for response parsing
    - Parsing methods for each extraction category:
        - _parse_study_design_response: Phase, sample size, arms
        - _parse_eligibility_response: Inclusion/exclusion criteria
        - _parse_endpoints_response: Primary/secondary endpoints
        - _parse_sites_response: Sites and countries
        - _parse_operational_burden_response: Procedures, visits
        - _parse_screening_flow_response: CONSORT data

Example:
    >>> class MyExtractor(FeasibilityResponseParserMixin):
    ...     pass
    >>> extractor = MyExtractor()
    >>> candidates = extractor._parse_study_design_response(llm_json, provenance)
    >>> candidates[0].field_type
    FeasibilityFieldType.STUDY_PHASE

Dependencies:
    - A_core.A07_feasibility_models: FeasibilityCandidate, StudyDesign, EligibilityCriterion
    - C_generators.C29_feasibility_prompts: Prompt templates for context
"""

from __future__ import annotations

from typing import List

from A_core.A07_feasibility_models import (
    BackgroundTherapy,
    CentralLabRequirement,
    CriterionType,
    EligibilityCriterion,
    EndpointType,
    EvidenceSpan,
    ExtractionMethod,
    FeasibilityCandidate,
    FeasibilityFieldType,
    FeasibilityGeneratorType,
    InvasiveProcedure,
    OperationalBurden,
    ScheduledVisit,
    ScreenFailReason,
    ScreeningFlow,
    StudyDesign,
    StudyEndpoint,
    StudyPeriod,
    StudySite,
    TreatmentArm,
    VaccinationRequirement,
    VisitSchedule,
)

from .C29_feasibility_prompts import (
    ELIGIBILITY_PROMPT,
    ENDPOINTS_PROMPT,
    OPERATIONAL_BURDEN_PROMPT,
    SCREENING_FLOW_PROMPT,
    SITES_PROMPT,
    STUDY_DESIGN_PROMPT,
)


class FeasibilityResponseParserMixin:
    """
    Mixin class providing LLM response parsing methods.

    Designed to be mixed into LLMFeasibilityExtractor to provide
    the _extract_* methods that parse LLM JSON responses.

    Assumes the host class has:
    - self._call_llm(prompt, content): LLM call method
    - self._make_provenance(doc_fingerprint): Provenance factory
    - self._extraction_stats: Dict for tracking extraction counts
    - self._verification_stats: Dict for verification tracking
    - self._apply_verification_penalty(...): Verification penalty method
    - self._get_bool(data, key, default): Bool getter helper
    - self._get_list(data, key): List getter helper
    """

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
                    source_doc_id=doc_id,
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
        first_quote = evidence[0].quote if evidence else None
        numerical_values: dict[str, int | float] = {}
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
            exact_quote = crit_data.get("exact_quote")
            page_num = crit_data.get("page")
            numerical_values = {}
            if crit_data.get("value") is not None:
                numerical_values["value"] = crit_data["value"]

            confidence = self._apply_verification_penalty(
                base_confidence=0.85,
                quote=exact_quote,
                context=content,
                numerical_values=numerical_values,
            )

            # Build evidence from verified quote with page number
            evidence = []
            if exact_quote and isinstance(exact_quote, str) and exact_quote.strip():
                evidence.append(EvidenceSpan(
                    quote=exact_quote.strip(),
                    source_doc_id=doc_id,
                    page=page_num if isinstance(page_num, int) else None,
                ))

            candidates.append(FeasibilityCandidate(
                doc_id=doc_id,
                field_type=field_type,
                generator_type=FeasibilityGeneratorType.LLM_EXTRACTION,
                matched_text=text,
                context_text="",
                section_name="eligibility",
                confidence=confidence,
                eligibility_criterion=criterion,
                evidence=evidence,
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
            page_num = ep_data.get("page")
            confidence = self._apply_verification_penalty(
                base_confidence=0.88,
                quote=exact_quote,
                context=content,
            )

            # Build evidence from verified quote with page number
            evidence = []
            if exact_quote and isinstance(exact_quote, str) and exact_quote.strip():
                evidence.append(EvidenceSpan(
                    quote=exact_quote.strip(),
                    source_doc_id=doc_id,
                    page=page_num if isinstance(page_num, int) else None,
                ))

            candidates.append(FeasibilityCandidate(
                doc_id=doc_id,
                field_type=FeasibilityFieldType.STUDY_ENDPOINT,
                generator_type=FeasibilityGeneratorType.LLM_EXTRACTION,
                matched_text=name,
                context_text="",
                section_name="endpoints",
                confidence=confidence,
                study_endpoint=endpoint,
                evidence=evidence,
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
                        source_doc_id=doc_id,
                    ))
                procedures.append(InvasiveProcedure(
                    name=proc_data["name"],
                    timing=self._get_list(proc_data, "timing"),
                    timing_days=self._get_list(proc_data, "timing_days"),
                    optional=self._get_bool(proc_data, "optional", False),
                    purpose=proc_data.get("purpose"),
                    is_eligibility_requirement=self._get_bool(proc_data, "is_eligibility_requirement", False),
                    evidence=evidence,
                ))

        # Parse visit schedule with phase-structured visits
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
                        procedures=self._get_list(sv_data, "procedures"),
                    ))

            visit_schedule = VisitSchedule(
                total_visits=visit_data.get("total_visits"),
                visit_days=visit_data.get("visit_days") or [],
                frequency=visit_data.get("frequency"),
                duration_weeks=visit_data.get("duration_weeks"),
                scheduled_visits=scheduled_visits,
                pre_randomization_days=visit_data.get("pre_randomization_days") or [],
                on_treatment_days=visit_data.get("on_treatment_days") or [],
                follow_up_days=visit_data.get("follow_up_days") or [],
            )

        # Parse vaccination requirements
        vaccinations = []
        for vacc_data in (response.get("vaccination_requirements") or []):
            if isinstance(vacc_data, dict) and vacc_data.get("vaccine_type"):
                evidence = []
                if vacc_data.get("quote"):
                    page_num = vacc_data.get("page")
                    evidence.append(EvidenceSpan(
                        quote=vacc_data["quote"],
                        source_doc_id=doc_id,
                        page=page_num if isinstance(page_num, int) else None,
                    ))
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
                    page_num = bg_data.get("page")
                    evidence.append(EvidenceSpan(
                        quote=bg_data["quote"],
                        source_doc_id=doc_id,
                        page=page_num if isinstance(page_num, int) else None,
                    ))
                bg_therapy.append(BackgroundTherapy(
                    therapy_class=bg_data["therapy_class"],
                    requirement_type=bg_data.get("requirement_type", "allowed"),
                    requirement=bg_data.get("requirement", ""),
                    agents=self._get_list(bg_data, "agents"),
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
                    page_num = cm_data.get("page")
                    evidence.append(EvidenceSpan(
                        quote=cm_data["quote"],
                        source_doc_id=doc_id,
                        page=page_num if isinstance(page_num, int) else None,
                    ))
                concomitant_allowed.append(BackgroundTherapy(
                    therapy_class=cm_data["therapy_class"],
                    requirement_type="allowed",
                    requirement=cm_data.get("requirement", "allowed"),
                    agents=self._get_list(cm_data, "agents"),
                    stable_duration_days=cm_data.get("stable_duration_days"),
                    max_dose=cm_data.get("max_dose"),
                    evidence=evidence,
                ))

        # Parse central lab requirement with evidence
        # IMPORTANT: Only create CentralLabRequirement if we have explicit evidence
        central_lab = None
        central_lab_data = response.get("central_lab")
        central_lab_required_raw = response.get("central_lab_required")
        if isinstance(central_lab_data, dict):
            quote = central_lab_data.get("quote")
            # Only create central_lab if we have a valid quote containing "central"
            if quote and isinstance(quote, str) and "central" in quote.lower():
                evidence = []
                page_num = central_lab_data.get("page")
                evidence.append(EvidenceSpan(
                    quote=quote,
                    source_doc_id=doc_id,
                    page=page_num if isinstance(page_num, int) else None,
                ))
                central_lab = CentralLabRequirement(
                    required=central_lab_data.get("required") is True,
                    analytes=self._get_list(central_lab_data, "analytes"),
                    confidence=0.85,
                    evidence=evidence,
                )
        # Override central_lab_required to null if no evidence
        if central_lab is None:
            central_lab_required_raw = None

        # Build OperationalBurden
        burden = OperationalBurden(
            invasive_procedures=procedures,
            visit_schedule=visit_schedule,
            vaccination_requirements=vaccinations,
            background_therapy=bg_therapy,
            concomitant_meds_allowed=concomitant_allowed,
            run_in_duration_days=response.get("run_in_duration_days"),
            run_in_requirements=self._get_list(response, "run_in_requirements"),
            central_lab_required=central_lab_required_raw is True,
            central_lab=central_lab,
            special_sample_handling=self._get_list(response, "special_sample_handling"),
            hard_gates=self._get_list(response, "hard_gates"),
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
        numerical_values: dict[str, int | float] = {}
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
                        source_doc_id=doc_id,
                    ))
                screen_fail_reasons.append(ScreenFailReason(
                    reason=sfr_data["reason"],
                    count=sfr_data.get("count"),
                    percentage_reported=sfr_data.get("percentage_reported") or sfr_data.get("percentage"),
                    percentage_computed=sfr_data.get("percentage_computed"),
                    can_overlap=self._get_bool(sfr_data, "can_overlap", False),
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
                    source_doc_id=doc_id,
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
            screen_failure_rate=screen_failure_rate_computed,
            screen_failure_rate_reported=response.get("screen_failure_rate_reported"),
            screen_failure_rate_computed=screen_failure_rate_computed,
            dropout_rate=dropout_rate,
            screen_fail_reasons=screen_fail_reasons,
            reasons_can_overlap=self._get_bool(response, "reasons_can_overlap", False),
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


__all__ = ["FeasibilityResponseParserMixin"]
