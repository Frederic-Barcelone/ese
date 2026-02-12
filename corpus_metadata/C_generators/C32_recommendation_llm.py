"""
LLM-based extraction mixin for guideline recommendations.

This module provides mixin methods for LLM-based recommendation extraction
from clinical guidelines. Handles prompt construction, response parsing,
and evidence level inference from recommendation language.

Key Components:
    - LLMExtractionMixin: Mixin class for LLM-based extraction
    - Methods:
        - _extract_with_llm: Main extraction using Claude API
        - _clean_json_response: Response cleaning and parsing
        - _build_context: Context construction for prompts
        - _infer_evidence_level: Evidence level from language cues
        - _infer_strength: Recommendation strength from language

Example:
    >>> class MyExtractor(LLMExtractionMixin):
    ...     def __init__(self):
    ...         self.llm_client = anthropic.Anthropic()
    ...         self.llm_model = "claude-sonnet-4-5-20250929"
    >>> extractor = MyExtractor()
    >>> recommendations = extractor._extract_with_llm(text, "guidelines")

Dependencies:
    - A_core.A18_recommendation_models: GuidelineRecommendation, EvidenceLevel
    - C_generators.C31_recommendation_patterns: RECOMMENDATION_EXTRACTION_PROMPT
    - anthropic: Claude API client (expected on host class)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from A_core.A18_recommendation_models import (
    DrugDosingInfo,
    EvidenceLevel,
    GuidelineRecommendation,
    RecommendationSet,
    RecommendationStrength,
    RecommendationType,
)

from Z_utils.Z13_llm_tracking import record_api_usage

logger = logging.getLogger(__name__)

from .C31_recommendation_patterns import RECOMMENDATION_EXTRACTION_PROMPT


class LLMExtractionMixin:
    """
    Mixin class providing LLM-based extraction methods.

    Designed to be mixed into GuidelineRecommendationExtractor to provide
    LLM extraction, JSON parsing, and language inference capabilities.

    Assumes the host class has:
    - self.llm_client: LLM API client
    - self.llm_model: Model identifier string
    - self._vlm_loe_sor_cache: Dict for VLM cache
    - self.pdf_path: Optional PDF path
    - self.extract_loe_sor_with_vlm(): VLM extraction method
    """

    # Declare expected attributes from host class for type checking
    if TYPE_CHECKING:
        llm_client: Any
        llm_model: str
        pdf_path: Optional[str]
        _apply_vlm_codes_to_recommendations: Any
        extract_loe_sor_with_vlm: Any

    # System prompt for recommendation extraction — expanded to >=1024 tokens
    # so Anthropic prompt caching applies (system prompt reads at 10% input cost).
    _RECOMMENDATION_SYSTEM_PROMPT = (
        "You are a clinical guideline recommendation extraction specialist. "
        "Your task is to extract structured clinical recommendations from guideline documents.\n\n"
        + RECOMMENDATION_EXTRACTION_PROMPT
        + "\n\n"
        "ADDITIONAL FIELD DEFINITIONS:\n"
        "- rec_number: The sequential number of the recommendation (1, 2, 3...)\n"
        "- population: The patient population the recommendation applies to (e.g., 'patients with active AAV')\n"
        "- condition: The specific disease context — 'new-onset', 'relapsing', 'refractory', or null\n"
        "- severity: Disease severity — 'organ-threatening', 'life-threatening', 'severe', 'non-severe', or null\n"
        "- action: The recommended clinical action or treatment\n"
        "- preferred: The preferred treatment option if explicitly stated, otherwise null\n"
        "- alternatives: Array of alternative treatments mentioned, empty array if none\n"
        "- taper_target: Target dose reduction (e.g., '7.5 mg/day by 3 months'), null if not stated\n"
        "- duration: Treatment duration if stated (e.g., '6-12 months'), null if not stated\n"
        "- loe_code: Raw Level of Evidence code from the table (e.g., '1a', '2b', '5', 'na')\n"
        "- sor_code: Raw Strength of Recommendation grade (e.g., 'A', 'B', 'C')\n"
        "- evidence_level: Interpreted evidence level — 'high', 'moderate', 'low', 'very_low', 'expert_opinion'\n"
        "- strength: Interpreted recommendation strength — 'strong', 'conditional', 'weak'\n"
        "- source_text: The exact original text of the recommendation as it appears in the document\n"
        "- dosing: Array of drug dosing information objects, each with:\n"
        "  - drug_name (required): Name of the drug\n"
        "  - dose_range: Dose range (e.g., '500-1000 mg')\n"
        "  - starting_dose: Initial dose if specified\n"
        "  - maintenance_dose: Ongoing dose if specified\n"
        "  - max_dose: Maximum dose if specified\n"
        "  - route: Administration route (e.g., 'IV', 'oral', 'subcutaneous')\n"
        "  - frequency: Dosing frequency (e.g., 'twice weekly', 'every 6 months')\n\n"
        "ORGANIZATION DETECTION:\n"
        "Identify the publishing organization from these known bodies: "
        "EULAR, ACR, NICE, BSR, KDIGO, FDA, EMA, AHA, ESC, WHO, ACCF.\n\n"
        "EVIDENCE LEVEL MAPPING (Oxford Centre for Evidence-Based Medicine):\n"
        "Level 1a: Systematic review of RCTs → high\n"
        "Level 1b: Individual RCT → high\n"
        "Level 2a: Systematic review of cohort studies → moderate\n"
        "Level 2b: Individual cohort study → moderate\n"
        "Level 3a: Systematic review of case-control studies → low\n"
        "Level 3b: Individual case-control study → low\n"
        "Level 4: Case series, poor-quality cohort → very_low\n"
        "Level 5: Expert opinion without critical appraisal → expert_opinion\n\n"
        "STRENGTH OF RECOMMENDATION MAPPING:\n"
        "Grade A: Consistent level 1 studies → strong\n"
        "Grade B: Consistent level 2/3 studies or extrapolations from level 1 → conditional\n"
        "Grade C: Level 4 studies or extrapolations from level 2/3 → weak\n"
        "Grade D: Level 5 evidence or troublingly inconsistent studies → weak\n"
    )

    def _extract_with_llm(
        self,
        text: str,
        source: str,
        page_num: Optional[int],
    ) -> Optional[RecommendationSet]:
        """Extract recommendations using LLM via ClaudeClient (with prompt caching)."""
        if not self.llm_client:
            return None

        try:
            # Build context that includes both header and table sections
            context_text = self._build_llm_context(text)

            user_prompt = f"TEXT TO ANALYZE:\n---\n{context_text}"

            # Use ClaudeClient.complete_json() for model tier routing + prompt caching
            if hasattr(self.llm_client, "complete_json"):
                response = self.llm_client.complete_json(
                    system_prompt=self._RECOMMENDATION_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    max_tokens=8192,
                    temperature=0.0,
                    call_type="recommendation_extraction",
                )
                if response:
                    result = self._parse_llm_response(response, source, page_num)
                    if result:
                        self._apply_vlm_codes_to_recommendations(result)
                    return result
                return None
            else:
                # Fallback for non-ClaudeClient (e.g., test mocks)
                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=8192,
                    messages=[
                        {
                            "role": "user",
                            "content": f"{RECOMMENDATION_EXTRACTION_PROMPT}\n\n---\n{user_prompt}",
                        }
                    ],
                )
                record_api_usage(response, self.llm_model, "recommendation_extraction")

                if response.content and len(response.content) > 0:
                    import json
                    text_response = response.content[0].text.strip()
                    text_response = self._clean_json_response(text_response)
                    try:
                        data = json.loads(text_response)
                        result = self._parse_llm_response(data, source, page_num)
                        if result:
                            self._apply_vlm_codes_to_recommendations(result)
                        return result
                    except json.JSONDecodeError as je:
                        fixed = self._attempt_json_fix(text_response)
                        if fixed:
                            data = json.loads(fixed)
                            result = self._parse_llm_response(data, source, page_num)
                            if result:
                                self._apply_vlm_codes_to_recommendations(result)
                            return result
                        logger.warning("LLM returned invalid JSON: %s", je)
                        return None
                return None

        except Exception as e:
            logger.warning("LLM recommendation extraction failed: %s", e)
            return None

    def _clean_json_response(self, text: str) -> str:
        """Clean up JSON response from LLM."""
        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # Remove any trailing garbage after the closing brace
        last_brace = text.rfind("}")
        if last_brace > 0:
            text = text[:last_brace + 1]

        return text

    def _attempt_json_fix(self, text: str) -> Optional[str]:
        """Attempt to fix common JSON issues."""
        import json

        # Try adding missing closing brackets
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")

        if open_braces > 0 or open_brackets > 0:
            fixed = text
            fixed += "]" * open_brackets
            fixed += "}" * open_braces
            try:
                json.loads(fixed)
                return fixed
            except json.JSONDecodeError:
                pass

        # Try truncating at the last valid point
        for i in range(len(text) - 1, 0, -1):
            if text[i] in "}]":
                try:
                    candidate = text[:i + 1]
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

        return None

    def _build_llm_context(self, text: str) -> str:
        """
        Build optimized context for LLM extraction.

        Focuses on finding the recommendation table with LoE/SoR codes.
        Also pre-extracts embedded LoE/SoR codes to help the LLM.

        Args:
            text: Full document text

        Returns:
            Optimized context string (up to 12000 chars)
        """
        # Find the main recommendation table section
        table_section = self._find_main_recommendation_table(text)

        # Pre-extract embedded LoE/SoR codes from the text
        extracted_codes = self._extract_embedded_loe_sor(text)

        # If text-based extraction found few codes, try VLM extraction
        if len(extracted_codes) < 5 and self.pdf_path and self.llm_client:
            vlm_codes = self.extract_loe_sor_with_vlm(text)
            # Merge VLM codes (VLM takes precedence for conflicts)
            # Only store loe/sor (first 2 elements), text_snippet and keywords are cached separately
            for rec_num, (loe, sor, _text_snippet, _keywords) in vlm_codes.items():
                extracted_codes[rec_num] = (loe, sor)

        # Build context
        header = text[:2000]
        context_parts = [header]

        # Add pre-extracted codes if found
        if extracted_codes:
            codes_text = "\n\n--- PRE-EXTRACTED LoE/SoR MAPPING ---\n"
            codes_text += "Use these LoE/SoR codes for the numbered recommendations:\n"
            for rec_num, (loe, sor) in sorted(extracted_codes.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 99):
                codes_text += f"  Recommendation {rec_num}: LoE={loe}, SoR={sor}\n"
            context_parts.append(codes_text)

        if table_section and len(table_section) > 500:
            context_parts.append("\n\n--- RECOMMENDATION TABLE ---\n" + table_section[:8000])
        else:
            context_parts.append(text[2000:10000])

        return "".join(context_parts)

    def _extract_embedded_loe_sor(self, text: str) -> Dict[str, Tuple[str, str]]:
        """
        Extract LoE/SoR codes embedded in recommendation text.

        Handles format where codes appear in the middle of recommendation text
        due to PDF table column extraction issues:
        "16 In patients with AAV, we recommend serum 1b B 100 9.2 immunoglobulin..."

        Args:
            text: Document text

        Returns:
            Dict mapping recommendation number to (loe_code, sor_code) tuple
        """
        codes: Dict[str, Tuple[str, str]] = {}

        # Pattern for embedded codes: number + text + LoE + SoR + FV% + LoA + more text
        # The LoA can be like "9.241.4" (garbled "9.2±1.4") or "9.2+1.4"
        embedded_pattern = re.compile(
            r'(\d{1,2})\s+'  # Recommendation number
            r'(?:In\s+patients|For\s+patients|For\s+(?:induction|maintenance)|We\s+recommend|A\s+positive|Patients\s+with)'
            r'[^.]{10,200}?'  # Recommendation text (not too greedy)
            r'\s+([1-5][ab]?|n\.?a\.?)\s+'  # LoE code
            r'([ABC*†])\s+'  # SoR grade
            r'(?:\d{2,3}[*†]?)\s+'  # FV% (we don't capture this)
            r'(?:\d+\.?\d*(?:[+±]|\.)\d+\.?\d*)',  # LoA score (garbled or normal)
            re.IGNORECASE
        )

        for match in embedded_pattern.finditer(text):
            rec_num = match.group(1)
            loe_code = match.group(2).lower().replace(".", "")
            sor_code = match.group(3).upper().replace("*", "").replace("†", "")

            # Only store if we don't already have this recommendation
            if rec_num not in codes:
                codes[rec_num] = (loe_code, sor_code)

        # Also try to extract from "Table 3 Continued" style format specifically
        table3_cont = re.search(r'Table\s+3\s+Continued(.*?)(?=\n\n\n|Table\s+\d|$)', text, re.IGNORECASE | re.DOTALL)
        if table3_cont:
            section = table3_cont.group(1)
            for match in embedded_pattern.finditer(section):
                rec_num = match.group(1)
                loe_code = match.group(2).lower().replace(".", "")
                sor_code = match.group(3).upper().replace("*", "").replace("†", "")
                if rec_num not in codes:
                    codes[rec_num] = (loe_code, sor_code)

        return codes

    def _find_main_recommendation_table(self, text: str) -> Optional[str]:
        """
        Find the main recommendation table that contains LoE/SoR codes.

        Looks for "Table X" sections with recommendation content and evidence codes.

        Args:
            text: Full document text

        Returns:
            Table section text or None
        """
        # Look for "Table 3" or similar recommendation tables
        # These typically have "LoE SoR" headers and numbered recommendations
        table_patterns = [
            # "Table 3. EULAR recommendations..." followed by "LoE SoR"
            r"(Table\s+\d+[.\s]+[^\n]*(?:recommendations?|guidelines?)[^\n]*\n.*?LoE\s+SoR.*?)(?=\n\n\n|Table\s+\d+[.\s]|--- Page \d+ ---|$)",
            # "Table X Continued" sections
            r"(Table\s+\d+\s+Continued\s*\n.*?LoE\s+SoR.*?)(?=\n\n\n|Table\s+\d+[.\s]|--- Page \d+ ---|$)",
            # Any table with "LoE" and "SoR" columns
            r"(LoE\s+SoR\s+FV.*?(?:\d{1,2}\s+(?:We\s+recommend|For\s+|In\s+patients|Patients).*?[1-5][ab]?\s+[ABC].*?)+)",
        ]

        best_match = None
        best_length = 0

        for pattern in table_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                section = match.group(1)
                # Prefer longer matches that contain more content
                if len(section) > best_length and len(section) < 15000:
                    # Verify it has actual recommendation content
                    if re.search(r"we\s+recommend|should\s+be|may\s+(?:be\s+)?consider", section, re.IGNORECASE):
                        best_match = section
                        best_length = len(section)

        return best_match

    def _find_evidence_table_sections(self, text: str) -> List[str]:
        """
        Find table sections containing evidence levels and recommendations.

        Looks for:
        - "Table X" headers followed by recommendation content
        - Sections with "LoE" and "SoR" headers
        - Content with evidence codes (1a, 1b, 2a, 2b, etc.)

        Args:
            text: Full document text

        Returns:
            List of table section strings
        """
        sections = []

        # Pattern 1: Find "Table X" sections with LoE/SoR
        table_pattern = re.compile(
            r"(Table\s+\d+[.\s].*?(?:LoE|SoR|Level of [Ee]vidence|Strength of [Rr]ecommendation).*?)(?=Table\s+\d+[.\s]|--- Page|$)",
            re.IGNORECASE | re.DOTALL
        )
        for match in table_pattern.finditer(text):
            section = match.group(1).strip()
            if len(section) > 200:  # Meaningful content
                sections.append(section[:4000])  # Limit each section

        # Pattern 2: Find sections with "LoE SoR" column headers
        loe_sor_pattern = re.compile(
            r"(LoE\s+SoR.*?)(?=--- Page|\n\n\n|$)",
            re.IGNORECASE | re.DOTALL
        )
        for match in loe_sor_pattern.finditer(text):
            section = match.group(1).strip()
            if len(section) > 100 and section not in sections:
                sections.append(section[:4000])

        # Pattern 3: Find sections with recommendation numbers and evidence codes
        # e.g., "1 We recommend... 1b A" or numbered recommendations
        numbered_rec_pattern = re.compile(
            r"(\d{1,2}\s+(?:We\s+recommend|For\s+(?:induction|maintenance|patients)|In\s+patients|A\s+positive).*?(?:[1-5][ab]?\s+[ABC]|n\.?a\.?\s+[ABC]))",
            re.IGNORECASE | re.DOTALL
        )
        for match in numbered_rec_pattern.finditer(text):
            section = match.group(1).strip()
            if len(section) > 50:
                # Extend to capture full row if truncated
                start = match.start()
                end = min(match.end() + 200, len(text))
                extended = text[start:end]
                # Cut at sentence boundary
                period_pos = extended.rfind(". ", len(section))
                if period_pos > 0:
                    extended = extended[:period_pos + 1]
                if extended not in sections:
                    sections.append(extended)

        return sections

    def _find_recommendation_sections(self, text: str) -> List[str]:
        """
        Find text sections containing recommendations.

        Args:
            text: Full document text

        Returns:
            List of recommendation section strings
        """
        sections = []

        # Find paragraphs with recommendation keywords
        rec_pattern = re.compile(
            r"([^.]*(?:we recommend|we suggest|should be|may be considered|is recommended)[^.]*\.)",
            re.IGNORECASE
        )

        for match in rec_pattern.finditer(text):
            section = match.group(1).strip()
            if len(section) > 30:
                sections.append(section)

        return sections[:20]  # Limit to 20 sections

    def _parse_llm_response(
        self,
        data: Dict[str, Any],
        source: str,
        page_num: Optional[int],
    ) -> RecommendationSet:
        """Parse LLM response into RecommendationSet."""
        recommendations = []

        for idx, rec_data in enumerate(data.get("recommendations", [])):
            if not isinstance(rec_data, dict):
                continue

            # Parse dosing - handle both full and partial dosing info
            dosing = []
            for dose_data in rec_data.get("dosing", []):
                if isinstance(dose_data, dict) and dose_data.get("drug_name"):
                    dosing.append(DrugDosingInfo(
                        drug_name=dose_data["drug_name"],
                        dose_range=dose_data.get("dose_range"),
                        starting_dose=dose_data.get("starting_dose"),
                        maintenance_dose=dose_data.get("maintenance_dose"),
                        max_dose=dose_data.get("max_dose"),
                        route=dose_data.get("route"),
                        frequency=dose_data.get("frequency"),
                    ))

            # Parse evidence level - PRIORITIZE raw codes over LLM interpretation
            evidence = EvidenceLevel.UNKNOWN
            ev_str = rec_data.get("evidence_level", "")
            loe_code = rec_data.get("loe_code", "")

            # FIRST: Try to use the raw LoE code if available (most reliable)
            if loe_code:
                evidence = self._loe_code_to_level(loe_code)

            # SECOND: If no code worked, check LLM's descriptive evidence level
            # BUT: treat "expert_opinion" as unknown since that's the LLM's fallback
            if evidence == EvidenceLevel.UNKNOWN and ev_str:
                ev_lower = ev_str.lower().replace(" ", "_")
                # Only accept specific evidence levels, not the default "expert_opinion"
                if ev_lower in ("high", "moderate", "low", "very_low"):
                    try:
                        evidence = EvidenceLevel(ev_lower)
                    except ValueError:
                        pass

            # THIRD: Infer from language patterns when codes/levels unavailable
            if evidence == EvidenceLevel.UNKNOWN:
                source_text = rec_data.get("source_text", "")
                action = rec_data.get("action", "")
                combined = (source_text + " " + action).lower()
                evidence = self._infer_evidence_from_language(combined)

            # Parse strength - PRIORITIZE raw codes over LLM interpretation
            strength = RecommendationStrength.UNKNOWN
            str_str = rec_data.get("strength", "")
            sor_code = rec_data.get("sor_code", "")

            # FIRST: Try to use the raw SoR code if available (most reliable)
            if sor_code:
                strength = self._sor_code_to_strength(sor_code)

            # SECOND: If no code worked, check LLM's descriptive strength
            if strength == RecommendationStrength.UNKNOWN and str_str:
                str_lower = str_str.lower()
                if str_lower in ("strong", "conditional", "weak"):
                    try:
                        strength = RecommendationStrength(str_lower)
                    except ValueError:
                        pass

            # THIRD: Infer from language patterns when codes unavailable
            if strength == RecommendationStrength.UNKNOWN:
                source_text = rec_data.get("source_text", "")
                action = rec_data.get("action", "")
                combined = (source_text + " " + action).lower()
                strength = self._infer_strength_from_language(combined)

            # Build recommendation ID with more context
            rec_id = f"rec_{source}_{idx+1}"

            recommendations.append(GuidelineRecommendation(
                recommendation_id=rec_id,
                recommendation_type=RecommendationType.TREATMENT,
                population=rec_data.get("population", "Unknown"),
                condition=rec_data.get("condition"),
                severity=rec_data.get("severity"),
                action=rec_data.get("action", ""),
                preferred=rec_data.get("preferred"),
                alternatives=rec_data.get("alternatives", []),
                dosing=dosing,
                taper_target=rec_data.get("taper_target"),
                duration=rec_data.get("duration"),
                stop_window=rec_data.get("stop_window"),
                evidence_level=evidence,
                strength=strength,
                source=source,
                source_text=rec_data.get("source_text"),
                page_num=rec_data.get("page_num", page_num),
            ))

        # Parse guideline year
        guideline_year = data.get("guideline_year")
        if isinstance(guideline_year, str) and guideline_year.isdigit():
            guideline_year = int(guideline_year)
        elif not isinstance(guideline_year, int):
            guideline_year = None

        return RecommendationSet(
            guideline_name=data.get("guideline_name") or "Unknown",
            guideline_year=guideline_year,
            organization=data.get("organization"),
            target_condition=data.get("target_condition") or "Unknown",
            recommendations=recommendations,
            source_document=source,
            extraction_confidence=0.85 if recommendations else 0.0,
        )

    def _loe_code_to_level(self, loe_code: str) -> EvidenceLevel:
        """Convert LoE code to EvidenceLevel enum.

        Note: 'na' (not available) returns UNKNOWN to trigger language inference,
        since 'na' means the LLM couldn't find a code, not that it found evidence
        of expert opinion.
        """
        code = loe_code.lower().replace(".", "").strip()

        # 'na' or empty means no code was found - return UNKNOWN to trigger fallback
        if not code or code == "na":
            return EvidenceLevel.UNKNOWN

        mapping = {
            "1a": EvidenceLevel.HIGH,
            "1b": EvidenceLevel.HIGH,
            "la": EvidenceLevel.HIGH,  # Handle OCR errors
            "lb": EvidenceLevel.HIGH,
            "2a": EvidenceLevel.MODERATE,
            "2b": EvidenceLevel.MODERATE,
            "3a": EvidenceLevel.LOW,
            "3b": EvidenceLevel.LOW,
            "4": EvidenceLevel.VERY_LOW,
            "5": EvidenceLevel.EXPERT_OPINION,
        }
        return mapping.get(code, EvidenceLevel.UNKNOWN)

    def _sor_code_to_strength(self, sor_code: str) -> RecommendationStrength:
        """Convert SoR grade to RecommendationStrength enum.

        Note: 'na' (not available) returns UNKNOWN to trigger language inference.
        """
        code = sor_code.upper().strip()

        # 'NA' or empty means no code was found - return UNKNOWN to trigger fallback
        if not code or code == "NA":
            return RecommendationStrength.UNKNOWN

        mapping = {
            "A": RecommendationStrength.STRONG,
            "B": RecommendationStrength.CONDITIONAL,
            "C": RecommendationStrength.WEAK,
        }
        return mapping.get(code, RecommendationStrength.UNKNOWN)

    def _infer_evidence_from_language(self, text: str) -> EvidenceLevel:
        """
        Infer evidence level from recommendation language when codes are unavailable.

        Uses language patterns to estimate evidence quality:
        - References to RCTs, meta-analyses -> HIGH
        - References to cohort studies, clinical experience -> MODERATE
        - "May consider", "based on expert opinion" -> EXPERT_OPINION

        Args:
            text: Recommendation text or source text

        Returns:
            Inferred EvidenceLevel (defaults to EXPERT_OPINION if no indicators)
        """
        text_lower = text.lower()

        # Check for high evidence indicators (RCTs, systematic reviews)
        high_indicators = [
            r"randomized\s+(?:controlled\s+)?trial",
            r"rct",
            r"meta-analysis",
            r"systematic\s+review",
            r"level\s+1[ab]?\s+evidence",
            r"high[\s-]+quality\s+evidence",
        ]
        for pattern in high_indicators:
            if re.search(pattern, text_lower):
                return EvidenceLevel.HIGH

        # Check for moderate evidence indicators
        moderate_indicators = [
            r"cohort\s+stud(?:y|ies)",
            r"observational\s+stud(?:y|ies)",
            r"moderate[\s-]+quality\s+evidence",
            r"level\s+2[ab]?\s+evidence",
        ]
        for pattern in moderate_indicators:
            if re.search(pattern, text_lower):
                return EvidenceLevel.MODERATE

        # Check for low evidence indicators
        low_indicators = [
            r"case[\s-]+control",
            r"case\s+series",
            r"low[\s-]+quality\s+evidence",
            r"level\s+3[ab]?\s+evidence",
            r"limited\s+evidence",
        ]
        for pattern in low_indicators:
            if re.search(pattern, text_lower):
                return EvidenceLevel.LOW

        # Check for explicit expert opinion indicators
        expert_indicators = [
            r"expert\s+(?:opinion|consensus)",
            r"best\s+practice",
            r"clinical\s+experience",
            r"based\s+on\s+(?:clinical\s+)?experience",
            r"n\.?a\.?\s+evidence",
        ]
        for pattern in expert_indicators:
            if re.search(pattern, text_lower):
                return EvidenceLevel.EXPERT_OPINION

        # Default to expert opinion when no clear indicators
        # Most guideline recommendations without explicit RCT references
        # are based on expert consensus
        return EvidenceLevel.EXPERT_OPINION

    def _infer_strength_from_language(self, text: str) -> RecommendationStrength:
        """
        Infer recommendation strength from language patterns.

        Uses verb forms and modal language to estimate recommendation strength:
        - "We recommend", "should" -> STRONG
        - "Should consider", "is recommended" -> CONDITIONAL
        - "May consider", "can be considered" -> WEAK

        Args:
            text: Recommendation text

        Returns:
            Inferred RecommendationStrength
        """
        text_lower = text.lower()

        # Strong indicators
        strong_patterns = [
            r"\bwe\s+recommend\b",
            r"\bstrongly\s+recommend\b",
            r"\bis\s+(?:strongly\s+)?recommended\b",
            r"\bshould\s+(?:be\s+)?(?:used|given|administered|initiated)\b",
            r"\bmust\s+be\b",
        ]
        for pattern in strong_patterns:
            if re.search(pattern, text_lower):
                return RecommendationStrength.STRONG

        # Weak indicators (check before conditional)
        weak_patterns = [
            r"\bmay\s+(?:be\s+)?considered?\b",
            r"\bcan\s+be\s+considered\b",
            r"\bcould\s+be\s+considered\b",
            r"\bmight\s+be\s+considered\b",
            r"\bweak(?:ly)?\s+recommend\b",
            r"\bmay\s+be\s+used\b",
        ]
        for pattern in weak_patterns:
            if re.search(pattern, text_lower):
                return RecommendationStrength.WEAK

        # Conditional indicators
        conditional_patterns = [
            r"\bshould\s+(?:be\s+)?considered?\b",
            r"\bconditional(?:ly)?\s+recommend\b",
            r"\bis\s+suggested\b",
            r"\bwe\s+suggest\b",
            r"\bcan\s+be\s+used\b",
        ]
        for pattern in conditional_patterns:
            if re.search(pattern, text_lower):
                return RecommendationStrength.CONDITIONAL

        # Default based on presence of "recommend" anywhere
        if "recommend" in text_lower:
            return RecommendationStrength.STRONG

        return RecommendationStrength.UNKNOWN


__all__ = ["LLMExtractionMixin"]
