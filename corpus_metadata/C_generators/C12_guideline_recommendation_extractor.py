# corpus_metadata/C_generators/C12_guideline_recommendation_extractor.py
"""
Guideline Recommendation Extractor.

Extracts structured clinical recommendations from guideline documents,
including treatment recommendations, dosing guidance, and evidence levels.

Works with:
- Recommendation text blocks
- Summary tables (with VLM fallback for garbled PDF extraction)
- Treatment algorithm figures

This module uses mixin classes for LLM and VLM extraction:
- LLMExtractionMixin: LLM-based extraction and parsing
- VLMExtractionMixin: Vision-based table extraction
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from A_core.A18_recommendation_models import (
    GuidelineRecommendation,
    RecommendationSet,
    RecommendationType,
    EvidenceLevel,
    RecommendationStrength,
    DrugDosingInfo,
)

from .C12a_recommendation_patterns import (
    ORGANIZATION_PATTERNS,
    GUIDELINE_TITLE_PATTERNS,
    CONDITION_PATTERNS,
    DRUG_PATTERNS,
    DOSE_PATTERN,
    DURATION_PATTERN,
    EVIDENCE_PATTERNS,
    STRENGTH_PATTERNS,
    SEVERITY_PATTERNS,
    SPECIFIC_CONDITION_PATTERNS,
    RECOMMENDATION_EXTRACTION_PROMPT,
)
from .C12b_recommendation_llm import LLMExtractionMixin
from .C12c_recommendation_vlm import VLMExtractionMixin


class GuidelineRecommendationExtractor(LLMExtractionMixin, VLMExtractionMixin):
    """
    Extracts structured clinical recommendations from guideline documents.

    Uses both pattern matching and LLM-based extraction to identify
    and structure clinical recommendations. Supports VLM fallback for
    extracting LoE/SoR codes from table images when text is garbled.
    """

    def __init__(
        self,
        llm_client: Any = None,
        llm_model: str = "claude-sonnet-4-20250514",
        pdf_path: Optional[str] = None,
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self._pdf_path = pdf_path
        # Cache: rec_num -> (loe_code, sor_code, text_snippet, keywords)
        self._vlm_loe_sor_cache: Dict[str, Tuple[str, str, str, List[str]]] = {}

    @property
    def pdf_path(self) -> Optional[str]:
        """Get the current PDF path."""
        return self._pdf_path

    @pdf_path.setter
    def pdf_path(self, value: Optional[str]) -> None:
        """Set PDF path and clear VLM cache if path changed."""
        if value != self._pdf_path:
            # Clear cache when switching to a new document
            self._vlm_loe_sor_cache = {}
        self._pdf_path = value

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
        # First, extract guideline metadata from the document
        guideline_metadata = self._extract_guideline_metadata(text)

        if use_llm and self.llm_client:
            # Use LLM-based extraction
            llm_results = self._extract_with_llm(text, source, page_num)
            if llm_results:
                # Merge any missing metadata from pattern extraction
                if llm_results.guideline_name in ("Unknown", "Unknown Guideline", None):
                    llm_results.guideline_name = guideline_metadata.get("guideline_name", "Unknown Guideline")
                if llm_results.guideline_year is None:
                    llm_results.guideline_year = guideline_metadata.get("guideline_year")
                if llm_results.organization is None:
                    llm_results.organization = guideline_metadata.get("organization")
                if llm_results.target_condition in ("Unknown", None):
                    llm_results.target_condition = guideline_metadata.get("target_condition", "Unknown")
                return llm_results

        # Fall back to pattern-based extraction
        recommendations = self._extract_with_patterns(text, source, page_num)

        return RecommendationSet(
            guideline_name=guideline_metadata.get("guideline_name") or "Unknown Guideline",
            guideline_year=guideline_metadata.get("guideline_year"),
            organization=guideline_metadata.get("organization"),
            target_condition=guideline_metadata.get("target_condition") or "Unknown",
            recommendations=recommendations,
            source_document=source,
            extraction_confidence=0.5 if recommendations else 0.0,
        )

    def _extract_guideline_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract guideline metadata from document text.

        Parses the document header/title to extract:
        - Guideline name
        - Organization (EULAR, ACR, etc.)
        - Publication year
        - Target condition

        Args:
            text: Full document text

        Returns:
            Dictionary with extracted metadata
        """
        metadata: Dict[str, Any] = {}

        # Work with first 2000 chars (title/abstract area)
        header_text = text[:2000]

        # Extract organization
        for pattern, org in ORGANIZATION_PATTERNS.items():
            if re.search(pattern, header_text, re.IGNORECASE):
                metadata["organization"] = org
                break

        # Extract guideline title and year
        for pattern in GUIDELINE_TITLE_PATTERNS:
            match = re.search(pattern, header_text, re.IGNORECASE)
            if match:
                groups = match.groups()
                # Handle different pattern formats
                if groups[0] and len(groups[0]) == 4 and groups[0].isdigit():
                    # Pattern: "2022 EULAR recommendations..."
                    metadata["guideline_year"] = int(groups[0])
                    if len(groups) > 1 and groups[1]:
                        metadata["guideline_name"] = groups[1].strip()
                elif len(groups) >= 2 and groups[1] and len(groups[1]) == 4 and groups[1].isdigit():
                    # Pattern: "EULAR recommendations...: 2022 update" (year in group 2)
                    # Full title is in group 1
                    metadata["guideline_name"] = groups[0].strip()
                    metadata["guideline_year"] = int(groups[1])
                else:
                    # Pattern: just title, no year
                    metadata["guideline_name"] = groups[0].strip()
                break

        # If no title found, try to extract from first line
        if "guideline_name" not in metadata:
            first_lines = header_text.split("\n")[:5]
            for line in first_lines:
                line = line.strip()
                if len(line) > 20 and any(kw in line.lower() for kw in ["recommendation", "guideline", "management", "treatment"]):
                    metadata["guideline_name"] = line[:200]
                    break

        # Extract year if not found yet
        if "guideline_year" not in metadata:
            year_match = re.search(r"\b(20[0-2]\d)\b", header_text)
            if year_match:
                metadata["guideline_year"] = int(year_match.group(1))

        # Extract target condition
        for pattern in CONDITION_PATTERNS:
            match = re.search(pattern, header_text, re.IGNORECASE)
            if match:
                condition = match.group(1).strip()
                # Clean up common artifacts
                condition = re.sub(r"\s+", " ", condition)
                if len(condition) > 5 and len(condition) < 100:
                    metadata["target_condition"] = condition
                    break

        return metadata

    def _get_page_number(self, text: str, position: int) -> Optional[int]:
        """
        Determine the page number for a position in the text.

        Looks for page markers like "--- Page X ---" that precede the position.

        Args:
            text: Full document text
            position: Character position in text

        Returns:
            Page number or None if not determinable
        """
        # Find all page markers before this position
        page_pattern = re.compile(r"---\s*Page\s+(\d+)\s*---", re.IGNORECASE)
        last_page = None

        for match in page_pattern.finditer(text[:position]):
            last_page = int(match.group(1))

        return last_page

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

    def _extract_with_patterns(
        self,
        text: str,
        source: str,
        default_page_num: Optional[int],
    ) -> List[GuidelineRecommendation]:
        """Extract recommendations using pattern matching."""
        recommendations = []

        # First, try to extract LoE/SoR mapping from table format
        loe_sor_map = self._extract_loe_sor_from_table(text)

        # Split into sentences/clauses while tracking position
        sentence_pattern = re.compile(r'[^.;]+[.;]')

        rec_idx = 0
        for match in sentence_pattern.finditer(text):
            sentence = match.group().strip()
            position = match.start()

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

            # Extract evidence level - first check table mapping, then inline patterns
            evidence = EvidenceLevel.UNKNOWN
            loe_code = None
            sor_code = None

            # Try to match recommendation to table data
            table_match = self._match_rec_to_table(sentence, loe_sor_map)
            if table_match:
                loe_code, sor_code = table_match
                evidence = self._loe_code_to_level(loe_code)

            # Fall back to inline patterns
            if evidence == EvidenceLevel.UNKNOWN:
                for pattern, level in EVIDENCE_PATTERNS.items():
                    if re.search(pattern, sentence, re.IGNORECASE):
                        evidence = level
                        break

            # Extract strength - first check table mapping
            strength = RecommendationStrength.UNKNOWN
            if sor_code:
                strength = self._sor_code_to_strength(sor_code)

            # Fall back to inline patterns
            if strength == RecommendationStrength.UNKNOWN:
                for pattern, s in STRENGTH_PATTERNS.items():
                    if re.search(pattern, sentence, re.IGNORECASE):
                        strength = s
                        break

            # Extract severity
            severity = None
            for pattern in SEVERITY_PATTERNS:
                sev_match = re.search(pattern, sentence, re.IGNORECASE)
                if sev_match:
                    severity = sev_match.group(1).lower().replace("-", " ")
                    break

            # Extract specific condition
            condition = None
            for pattern in SPECIFIC_CONDITION_PATTERNS:
                cond_match = re.search(pattern, sentence, re.IGNORECASE)
                if cond_match:
                    condition = cond_match.group(1)
                    break

            # Build population string from condition and severity
            population_parts = []
            if condition:
                population_parts.append(f"patients with {condition}")
            if severity:
                population_parts.append(severity)
            population = " ".join(population_parts) if population_parts else "General"

            # Get page number for this recommendation
            page_num = self._get_page_number(text, position) or default_page_num

            rec_idx += 1
            recommendations.append(GuidelineRecommendation(
                recommendation_id=f"rec_{source}_pattern_{rec_idx}",
                recommendation_type=RecommendationType.TREATMENT,
                population=population,
                condition=condition,
                severity=severity,
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

    def _extract_loe_sor_from_table(self, text: str) -> Dict[str, Tuple[str, str]]:
        """
        Extract LoE/SoR codes from table format.

        Looks for patterns like:
        - "16 In patients with AAV... 1b B 100 9.2"
        - Recommendation number followed by text and trailing codes

        Returns:
            Dict mapping recommendation key phrases to (loe_code, sor_code) tuples
        """
        loe_sor_map: Dict[str, Tuple[str, str]] = {}

        # Pattern to find table rows with recommendation number, text, and trailing codes
        # Format: [number] [recommendation text] [LoE code] [SoR grade] [FV%] [LoA]
        table_row_pattern = re.compile(
            r"(\d{1,2})\s+"  # Recommendation number
            r"((?:We\s+recommend|For\s+(?:induction|maintenance|patients)|In\s+patients|A\s+positive|Patients\s+with)[^0-9]{20,200}?)"  # Rec text
            r"\s+([1-5][ab]?|n\.?a\.?)\s+"  # LoE code
            r"([ABC*†])\s*"  # SoR grade
            r"(?:\d{2,3}|\*|†)",  # FV% or symbols
            re.IGNORECASE | re.DOTALL
        )

        for match in table_row_pattern.finditer(text):
            rec_num = match.group(1)
            rec_text = match.group(2).strip()
            loe_code = match.group(3).lower().replace(".", "")
            sor_code = match.group(4).upper().replace("*", "").replace("†", "")

            # Create key from first few meaningful words
            key_words = re.sub(r'\s+', ' ', rec_text)[:100].lower()
            loe_sor_map[key_words] = (loe_code, sor_code)

            # Also store by recommendation number
            loe_sor_map[f"rec_{rec_num}"] = (loe_code, sor_code)

        return loe_sor_map

    def _match_rec_to_table(
        self,
        sentence: str,
        loe_sor_map: Dict[str, Tuple[str, str]]
    ) -> Optional[Tuple[str, str]]:
        """
        Match a recommendation sentence to table LoE/SoR data.

        Args:
            sentence: Recommendation text
            loe_sor_map: Mapping from key phrases to (loe, sor) tuples

        Returns:
            Tuple of (loe_code, sor_code) or None
        """
        sentence_lower = sentence.lower()[:100]

        # Try direct match
        for key, codes in loe_sor_map.items():
            if not key.startswith("rec_"):
                # Check if key phrase is in sentence
                if key[:50] in sentence_lower or sentence_lower[:50] in key:
                    return codes

        return None

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
