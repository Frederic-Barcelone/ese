# corpus_metadata/C_generators/C12_guideline_recommendation_extractor.py
"""
Guideline Recommendation Extractor.

Extracts structured clinical recommendations from guideline documents,
including treatment recommendations, dosing guidance, and evidence levels.

Works with:
- Recommendation text blocks
- Summary tables
- Treatment algorithm figures
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from A_core.A18_recommendation_models import (
    GuidelineRecommendation,
    RecommendationSet,
    RecommendationType,
    EvidenceLevel,
    RecommendationStrength,
    DrugDosingInfo,
)


# =============================================================================
# EXTRACTION PATTERNS
# =============================================================================

# Drug name patterns for extraction
DRUG_PATTERNS = {
    r"\brituximab\b": "rituximab",
    r"\bRTX\b": "rituximab",
    r"\bcyclophosphamide\b": "cyclophosphamide",
    r"\bCYC\b": "cyclophosphamide",
    r"\bglucocorticoid[s]?\b": "glucocorticoid",
    r"\bGC[s]?\b": "glucocorticoid",
    r"\bpredniso[nl]one\b": "prednisone",
    r"\bmethotrexate\b": "methotrexate",
    r"\bMTX\b": "methotrexate",
    r"\bmycophenolate\b": "mycophenolate",
    r"\bMMF\b": "mycophenolate",
    r"\bazathioprine\b": "azathioprine",
    r"\bAZA\b": "azathioprine",
    r"\bavacopan\b": "avacopan",
    r"\bmepolizumab\b": "mepolizumab",
    r"\bbenralizumab\b": "benralizumab",
}

# Dose patterns
DOSE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*(mg|g|mcg|µg)(?:/(?:day|d|kg|m2))?",
    re.IGNORECASE
)

# Duration patterns
DURATION_PATTERN = re.compile(
    r"(\d+)\s*(?:-\s*(\d+))?\s*(weeks?|months?|years?|days?)",
    re.IGNORECASE
)

# Evidence level patterns
EVIDENCE_PATTERNS = {
    r"high\s*(?:quality)?\s*evidence": EvidenceLevel.HIGH,
    r"moderate\s*(?:quality)?\s*evidence": EvidenceLevel.MODERATE,
    r"low\s*(?:quality)?\s*evidence": EvidenceLevel.LOW,
    r"very\s*low\s*(?:quality)?\s*evidence": EvidenceLevel.VERY_LOW,
    r"expert\s*(?:opinion|consensus)": EvidenceLevel.EXPERT_OPINION,
}

# Recommendation strength patterns
STRENGTH_PATTERNS = {
    r"strong(?:ly)?\s*recommend": RecommendationStrength.STRONG,
    r"conditional(?:ly)?\s*recommend": RecommendationStrength.CONDITIONAL,
    r"weak(?:ly)?\s*recommend": RecommendationStrength.WEAK,
    r"should\s*(?:be\s*)?consider": RecommendationStrength.CONDITIONAL,
    r"may\s*(?:be\s*)?consider": RecommendationStrength.WEAK,
}


# =============================================================================
# LLM PROMPT FOR RECOMMENDATION EXTRACTION
# =============================================================================

RECOMMENDATION_EXTRACTION_PROMPT = """Extract clinical recommendations from this text.

For each distinct recommendation, extract:
{
    "recommendations": [
        {
            "population": "<target population, e.g., 'GPA/MPA organ-threatening'>",
            "condition": "<specific condition, e.g., 'newly diagnosed'>",
            "severity": "<severity if specified>",
            "action": "<recommended action, e.g., 'GC + RTX or CYC'>",
            "preferred": "<preferred option if stated>",
            "alternatives": ["<alternative 1>", "<alternative 2>"],
            "dosing": [
                {
                    "drug_name": "<drug>",
                    "dose_range": "<dose range>",
                    "route": "<IV, oral, SC>",
                    "frequency": "<daily, weekly>"
                }
            ],
            "taper_target": "<target dose and timepoint for tapering>",
            "duration": "<treatment duration>",
            "stop_window": "<when to stop treatment>",
            "evidence_level": "high" | "moderate" | "low" | "very_low" | "expert_opinion",
            "strength": "strong" | "conditional" | "weak",
            "source_text": "<original text snippet>"
        }
    ],
    "guideline_name": "<guideline title if identifiable>",
    "target_condition": "<main condition addressed>"
}

IMPORTANT:
- Extract EACH distinct recommendation separately
- Capture exact dosing information (mg/day, ranges)
- Note taper targets with timepoints (e.g., "5 mg/day by 4-5 months")
- Identify treatment durations
- Capture evidence levels and recommendation strength
- Include the source text for each recommendation

Return JSON only."""


# =============================================================================
# GUIDELINE RECOMMENDATION EXTRACTOR CLASS
# =============================================================================


class GuidelineRecommendationExtractor:
    """
    Extracts structured clinical recommendations from guideline documents.

    Uses both pattern matching and LLM-based extraction to identify
    and structure clinical recommendations.
    """

    def __init__(
        self,
        llm_client: Any = None,
        llm_model: str = "claude-sonnet-4-20250514",
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def extract_from_text(
        self,
        text: str,
        source: str = "text",
        page_num: Optional[int] = None,
        use_llm: bool = True,
    ) -> RecommendationSet:
        """
        Extract recommendations from text.

        Args:
            text: Text to extract from
            source: Source identifier
            page_num: Page number
            use_llm: Whether to use LLM for extraction

        Returns:
            RecommendationSet with extracted recommendations
        """
        recommendations = []

        if use_llm and self.llm_client:
            # Use LLM-based extraction
            llm_results = self._extract_with_llm(text, source, page_num)
            if llm_results:
                return llm_results

        # Fall back to pattern-based extraction
        recommendations = self._extract_with_patterns(text, source, page_num)

        return RecommendationSet(
            guideline_name="Unknown Guideline",
            target_condition="Unknown",
            recommendations=recommendations,
            source_document=source,
            extraction_confidence=0.5 if recommendations else 0.0,
        )

    def extract_from_table(
        self,
        table_data: Dict[str, Any],
        source: str = "table",
        page_num: Optional[int] = None,
    ) -> RecommendationSet:
        """
        Extract recommendations from a parsed table.

        Args:
            table_data: Parsed table with headers and rows
            source: Source identifier
            page_num: Page number

        Returns:
            RecommendationSet with extracted recommendations
        """
        recommendations = []
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        if not headers or not rows:
            return RecommendationSet(
                guideline_name="Unknown",
                target_condition="Unknown",
                recommendations=[],
            )

        # Try to identify column types
        header_lower = [h.lower() if isinstance(h, str) else "" for h in headers]

        # Look for common columns
        pop_col = next((i for i, h in enumerate(header_lower) if "population" in h or "indication" in h), None)
        action_col = next((i for i, h in enumerate(header_lower) if "treatment" in h or "recommendation" in h or "action" in h), None)
        dose_col = next((i for i, h in enumerate(header_lower) if "dose" in h or "dosing" in h), None)
        duration_col = next((i for i, h in enumerate(header_lower) if "duration" in h), None)

        for row_idx, row in enumerate(rows):
            if not row or len(row) < 2:
                continue

            # Extract population
            population = row[pop_col] if pop_col is not None and pop_col < len(row) else row[0]

            # Extract action
            action = row[action_col] if action_col is not None and action_col < len(row) else row[1] if len(row) > 1 else ""

            if not population or not action:
                continue

            # Extract dosing
            dosing = []
            if dose_col is not None and dose_col < len(row):
                dose_text = row[dose_col]
                drugs = self._extract_drugs(action + " " + dose_text)
                for drug in drugs:
                    dose_info = self._extract_dose_info(drug, dose_text)
                    if dose_info:
                        dosing.append(dose_info)

            # Extract duration
            duration = None
            if duration_col is not None and duration_col < len(row):
                duration = row[duration_col]

            recommendations.append(GuidelineRecommendation(
                recommendation_id=f"rec_{source}_{row_idx+1}",
                recommendation_type=RecommendationType.TREATMENT,
                population=str(population),
                action=str(action),
                dosing=dosing,
                duration=duration,
                source=source,
                page_num=page_num,
            ))

        return RecommendationSet(
            guideline_name=table_data.get("title", "Unknown"),
            target_condition="Unknown",
            recommendations=recommendations,
            source_document=source,
            extraction_confidence=0.7 if recommendations else 0.0,
        )

    def _extract_with_llm(
        self,
        text: str,
        source: str,
        page_num: Optional[int],
    ) -> Optional[RecommendationSet]:
        """Extract recommendations using LLM."""
        if not self.llm_client:
            return None

        try:
            response = self.llm_client.messages.create(
                model=self.llm_model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": f"{RECOMMENDATION_EXTRACTION_PROMPT}\n\nText to analyze:\n{text[:8000]}",
                    }
                ],
            )

            if response.content and len(response.content) > 0:
                import json
                text_response = response.content[0].text.strip()

                # Parse JSON
                if text_response.startswith("```json"):
                    text_response = text_response[7:]
                if text_response.startswith("```"):
                    text_response = text_response[3:]
                if text_response.endswith("```"):
                    text_response = text_response[:-3]

                data = json.loads(text_response.strip())
                return self._parse_llm_response(data, source, page_num)

        except Exception as e:
            print(f"[WARN] LLM recommendation extraction failed: {e}")
            return None

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

            # Parse dosing
            dosing = []
            for dose_data in rec_data.get("dosing", []):
                if isinstance(dose_data, dict) and dose_data.get("drug_name"):
                    dosing.append(DrugDosingInfo(
                        drug_name=dose_data["drug_name"],
                        dose_range=dose_data.get("dose_range"),
                        route=dose_data.get("route"),
                        frequency=dose_data.get("frequency"),
                    ))

            # Parse evidence level
            evidence = EvidenceLevel.UNKNOWN
            ev_str = rec_data.get("evidence_level", "")
            if ev_str:
                try:
                    evidence = EvidenceLevel(ev_str.lower())
                except ValueError:
                    pass

            # Parse strength
            strength = RecommendationStrength.UNKNOWN
            str_str = rec_data.get("strength", "")
            if str_str:
                try:
                    strength = RecommendationStrength(str_str.lower())
                except ValueError:
                    pass

            recommendations.append(GuidelineRecommendation(
                recommendation_id=f"rec_{source}_{idx+1}",
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
                page_num=page_num,
            ))

        return RecommendationSet(
            guideline_name=data.get("guideline_name", "Unknown"),
            target_condition=data.get("target_condition", "Unknown"),
            recommendations=recommendations,
            source_document=source,
            extraction_confidence=0.8 if recommendations else 0.0,
        )

    def _extract_with_patterns(
        self,
        text: str,
        source: str,
        page_num: Optional[int],
    ) -> List[GuidelineRecommendation]:
        """Extract recommendations using pattern matching."""
        recommendations = []

        # Split into sentences/clauses
        sentences = re.split(r'[.;]\s+', text)

        rec_idx = 0
        for sentence in sentences:
            # Look for recommendation indicators
            if not re.search(r'recommend|should|may\s+(?:be\s+)?consider|prefer', sentence, re.IGNORECASE):
                continue

            # Extract drugs mentioned
            drugs = self._extract_drugs(sentence)
            if not drugs:
                continue

            # Extract dose info for each drug
            dosing = []
            for drug in drugs:
                dose_info = self._extract_dose_info(drug, sentence)
                if dose_info:
                    dosing.append(dose_info)

            # Extract duration
            duration_match = DURATION_PATTERN.search(sentence)
            duration = None
            if duration_match:
                start = duration_match.group(1)
                end = duration_match.group(2)
                unit = duration_match.group(3)
                if end:
                    duration = f"{start}-{end} {unit}"
                else:
                    duration = f"{start} {unit}"

            # Extract evidence level
            evidence = EvidenceLevel.UNKNOWN
            for pattern, level in EVIDENCE_PATTERNS.items():
                if re.search(pattern, sentence, re.IGNORECASE):
                    evidence = level
                    break

            # Extract strength
            strength = RecommendationStrength.UNKNOWN
            for pattern, s in STRENGTH_PATTERNS.items():
                if re.search(pattern, sentence, re.IGNORECASE):
                    strength = s
                    break

            rec_idx += 1
            recommendations.append(GuidelineRecommendation(
                recommendation_id=f"rec_{source}_pattern_{rec_idx}",
                recommendation_type=RecommendationType.TREATMENT,
                population="General",
                action=sentence[:200],
                dosing=dosing,
                duration=duration,
                evidence_level=evidence,
                strength=strength,
                source=source,
                source_text=sentence,
                page_num=page_num,
            ))

        return recommendations

    def _extract_drugs(self, text: str) -> List[str]:
        """Extract drug names from text."""
        drugs = []
        for pattern, drug_name in DRUG_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                if drug_name not in drugs:
                    drugs.append(drug_name)
        return drugs

    def _extract_dose_info(self, drug_name: str, text: str) -> Optional[DrugDosingInfo]:
        """Extract dosing information for a drug from text."""
        dose_match = DOSE_PATTERN.search(text)
        if not dose_match:
            return DrugDosingInfo(drug_name=drug_name)

        dose_start = dose_match.group(1)
        dose_end = dose_match.group(2)
        unit = dose_match.group(3)

        if dose_end:
            dose_range = f"{dose_start}-{dose_end} {unit}"
        else:
            dose_range = f"{dose_start} {unit}"

        # Try to find route
        route = None
        if re.search(r'\bIV\b|\bintravenous\b', text, re.IGNORECASE):
            route = "IV"
        elif re.search(r'\boral\b|\bPO\b', text, re.IGNORECASE):
            route = "oral"
        elif re.search(r'\bSC\b|\bsubcutaneous\b', text, re.IGNORECASE):
            route = "SC"

        return DrugDosingInfo(
            drug_name=drug_name,
            dose_range=dose_range,
            route=route,
        )


__all__ = [
    "GuidelineRecommendationExtractor",
    "RECOMMENDATION_EXTRACTION_PROMPT",
]
