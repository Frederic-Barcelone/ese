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

from A_core.A18_recommendation_models import (
    GuidelineRecommendation,
    RecommendationSet,
    RecommendationType,
    EvidenceLevel,
    RecommendationStrength,
    DrugDosingInfo,
)


# =============================================================================
# GUIDELINE METADATA PATTERNS
# =============================================================================

# Known guideline organizations
ORGANIZATION_PATTERNS = {
    r"\bEULAR\b": "EULAR",
    r"\bACR\b": "ACR",
    r"\bAmerican College of Rheumatology\b": "ACR",
    r"\bFDA\b": "FDA",
    r"\bEMA\b": "EMA",
    r"\bNICE\b": "NICE",
    r"\bBSR\b": "BSR",
    r"\bBritish Society for Rheumatology\b": "BSR",
    r"\bKDIGO\b": "KDIGO",
    r"\bACCF\b": "ACCF",
    r"\bAHA\b": "AHA",
    r"\bESC\b": "ESC",
    r"\bWHO\b": "WHO",
}

# Guideline title patterns
GUIDELINE_TITLE_PATTERNS = [
    # "EULAR recommendations for the management of X: 2022 update" - capture full title
    r"((?:EULAR|ACR|NICE|BSR|KDIGO|FDA|EMA|AHA|ESC)\s+(?:recommendations?|guidelines?)\s+(?:for\s+)?(?:the\s+)?(?:management|treatment|diagnosis)\s+of\s+[^:]+:\s*(\d{4})\s*(?:update)?)",
    # Same pattern without colon/year suffix
    r"((?:EULAR|ACR|NICE|BSR|KDIGO|FDA|EMA|AHA|ESC)\s+(?:recommendations?|guidelines?)\s+(?:for\s+)?(?:the\s+)?(?:management|treatment|diagnosis)\s+of\s+[A-Za-z][A-Za-z0-9\s-]+(?:vasculitis|disease|syndrome|disorder|arthritis|lupus))",
    # "2022 EULAR/ACR recommendations for X"
    r"(\d{4})\s+((?:EULAR|ACR|NICE|BSR|KDIGO|FDA|EMA|AHA|ESC)(?:/(?:EULAR|ACR|NICE|BSR|KDIGO|FDA|EMA|AHA|ESC))?\s+(?:recommendations?|guidelines?)\s+(?:for\s+)?[^.\n]+)",
    # "Guidelines for X management (2022)"
    r"((?:Guidelines?|Recommendations?)\s+(?:for\s+)?(?:the\s+)?(?:management|treatment|diagnosis)\s+of\s+[^(\n]+)\s*\((\d{4})\)",
]

# Target condition patterns
CONDITION_PATTERNS = [
    r"(?:management|treatment|diagnosis)\s+of\s+([A-Z][^,.]+?)(?:\s*[,:.]|\s+in\s+|\s+with\s+|\s*$)",
    r"patients?\s+with\s+([A-Z][a-zA-Z\s-]+(?:vasculitis|disease|syndrome|disorder))",
]

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

# Evidence level patterns - includes GRADE/Oxford levels
EVIDENCE_PATTERNS = {
    # Descriptive evidence levels
    r"high\s*(?:quality)?\s*evidence": EvidenceLevel.HIGH,
    r"moderate\s*(?:quality)?\s*evidence": EvidenceLevel.MODERATE,
    r"low\s*(?:quality)?\s*evidence": EvidenceLevel.LOW,
    r"very\s*low\s*(?:quality)?\s*evidence": EvidenceLevel.VERY_LOW,
    r"expert\s*(?:opinion|consensus)": EvidenceLevel.EXPERT_OPINION,
    # Oxford/GRADE numeric levels (1a = high, 1b = high, 2a/2b = moderate, etc.)
    r"\bLoE\s*[:\s]*1[aA]\b": EvidenceLevel.HIGH,
    r"\bLoE\s*[:\s]*1[bB]\b": EvidenceLevel.HIGH,
    r"\b1[aA]\b": EvidenceLevel.HIGH,
    r"\b1[bB]\b": EvidenceLevel.HIGH,
    r"\bLoE\s*[:\s]*2[aA]\b": EvidenceLevel.MODERATE,
    r"\bLoE\s*[:\s]*2[bB]\b": EvidenceLevel.MODERATE,
    r"\b2[aA]\b": EvidenceLevel.MODERATE,
    r"\b2[bB]\b": EvidenceLevel.MODERATE,
    r"\bLoE\s*[:\s]*3[aAbB]?\b": EvidenceLevel.LOW,
    r"\b3[aAbB]\b": EvidenceLevel.LOW,
    r"\bLoE\s*[:\s]*4\b": EvidenceLevel.VERY_LOW,
    r"\bLoE\s*[:\s]*5\b": EvidenceLevel.EXPERT_OPINION,
}

# Recommendation strength patterns - includes letter grades
STRENGTH_PATTERNS = {
    # Descriptive strength
    r"strong(?:ly)?\s*recommend": RecommendationStrength.STRONG,
    r"conditional(?:ly)?\s*recommend": RecommendationStrength.CONDITIONAL,
    r"weak(?:ly)?\s*recommend": RecommendationStrength.WEAK,
    r"should\s*(?:be\s*)?consider": RecommendationStrength.CONDITIONAL,
    r"may\s*(?:be\s*)?consider": RecommendationStrength.WEAK,
    # Letter grades (SoR = Strength of Recommendation)
    r"\bSoR\s*[:\s]*A\b": RecommendationStrength.STRONG,
    r"\bgrade\s*A\b": RecommendationStrength.STRONG,
    r"\bSoR\s*[:\s]*B\b": RecommendationStrength.CONDITIONAL,
    r"\bgrade\s*B\b": RecommendationStrength.CONDITIONAL,
    r"\bSoR\s*[:\s]*C\b": RecommendationStrength.WEAK,
    r"\bgrade\s*C\b": RecommendationStrength.WEAK,
}

# Population/severity patterns
SEVERITY_PATTERNS = [
    r"(organ[- ]threatening)",
    r"(life[- ]threatening)",
    r"(severe)",
    r"(non[- ]?severe)",
    r"(mild(?:\s+to\s+moderate)?)",
    r"(moderate(?:\s+to\s+severe)?)",
    r"(refractory)",
    r"(relapsing)",
    r"(new[- ]onset)",
    r"(active)",
]

# Specific condition patterns within recommendations
SPECIFIC_CONDITION_PATTERNS = [
    r"\b(GPA|granulomatosis with polyangiitis)\b",
    r"\b(MPA|microscopic polyangiitis)\b",
    r"\b(EGPA|eosinophilic GPA)\b",
    r"\b(AAV|ANCA[- ]associated vasculitis)\b",
    r"\b(lupus nephritis)\b",
    r"\b(rheumatoid arthritis)\b",
    r"\b(systemic sclerosis)\b",
]


# =============================================================================
# LLM PROMPT FOR RECOMMENDATION EXTRACTION
# =============================================================================

RECOMMENDATION_EXTRACTION_PROMPT = """Extract clinical guideline recommendations from this text. Return ONLY valid JSON.

EXTRACT:
1. Guideline metadata (name, year, organization, target condition)
2. Each numbered recommendation with its evidence level and strength

OUTPUT FORMAT (JSON only, no markdown):
{
    "guideline_name": "full title with year",
    "guideline_year": 2022,
    "organization": "EULAR",
    "target_condition": "condition name",
    "recommendations": [
        {
            "rec_number": 1,
            "population": "patients with X",
            "condition": "new-onset/relapsing/refractory",
            "severity": "organ-threatening/life-threatening/severe/non-severe",
            "action": "recommended treatment/action",
            "preferred": "preferred treatment if stated",
            "alternatives": ["alt1", "alt2"],
            "taper_target": "target dose by timepoint",
            "duration": "treatment duration",
            "loe_code": "1a/1b/2a/2b/3b/4/5/na",
            "sor_code": "A/B/C",
            "evidence_level": "high/moderate/low/very_low/expert_opinion",
            "strength": "strong/conditional/weak",
            "source_text": "exact recommendation text"
        }
    ]
}

EVIDENCE MAPPING:
- 1a, 1b → high
- 2a, 2b → moderate
- 3a, 3b → low
- 4 → very_low
- 5, na, consensus → expert_opinion

STRENGTH MAPPING:
- Grade A, "we recommend" → strong
- Grade B, "should consider" → conditional
- Grade C, "may consider" → weak

FINDING LoE/SoR CODES:
Look in tables for patterns like:
- "LoE SoR FV LoA" column headers
- Codes at END of recommendation rows: "...we recommend rituximab. 1b A 100 9.2"
- Numbered recommendations with trailing codes

CRITICAL:
- Only extract ACTUAL recommendations (not commentary or rationale)
- Each numbered recommendation is ONE entry
- Look for the LoE/SoR table - codes are there
- Keep responses concise - max 20 recommendations
- Return VALID JSON only"""


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
            guideline_name=guideline_metadata.get("guideline_name", "Unknown Guideline"),
            guideline_year=guideline_metadata.get("guideline_year"),
            organization=guideline_metadata.get("organization"),
            target_condition=guideline_metadata.get("target_condition", "Unknown"),
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
            # Build context that includes both header and table sections
            context_text = self._build_llm_context(text)

            response = self.llm_client.messages.create(
                model=self.llm_model,
                max_tokens=8192,
                messages=[
                    {
                        "role": "user",
                        "content": f"{RECOMMENDATION_EXTRACTION_PROMPT}\n\n---\nTEXT TO ANALYZE:\n---\n{context_text}",
                    }
                ],
            )

            if response.content and len(response.content) > 0:
                import json
                text_response = response.content[0].text.strip()

                # Parse JSON - handle common issues
                text_response = self._clean_json_response(text_response)

                try:
                    data = json.loads(text_response)
                    return self._parse_llm_response(data, source, page_num)
                except json.JSONDecodeError as je:
                    # Try to fix common JSON issues
                    fixed = self._attempt_json_fix(text_response)
                    if fixed:
                        data = json.loads(fixed)
                        return self._parse_llm_response(data, source, page_num)
                    print(f"[WARN] LLM returned invalid JSON: {je}")
                    return None

            # No content in response
            return None

        except Exception as e:
            print(f"[WARN] LLM recommendation extraction failed: {e}")
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

        Args:
            text: Full document text

        Returns:
            Optimized context string (up to 12000 chars)
        """
        # Find the main recommendation table section
        table_section = self._find_main_recommendation_table(text)

        if table_section and len(table_section) > 500:
            # We found a good table section - use it with some header context
            header = text[:2000]
            return header + "\n\n--- RECOMMENDATION TABLE WITH LoE/SoR ---\n" + table_section[:10000]

        # Fallback: use header + first 10000 chars
        return text[:12000]

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

            # Parse evidence level - handle both descriptive and code formats
            evidence = EvidenceLevel.UNKNOWN
            ev_str = rec_data.get("evidence_level", "")
            loe_code = rec_data.get("loe_code", "")

            if ev_str:
                try:
                    evidence = EvidenceLevel(ev_str.lower().replace(" ", "_"))
                except ValueError:
                    pass

            # If we have a LoE code, try to map it
            if evidence == EvidenceLevel.UNKNOWN and loe_code:
                loe_mapping = {
                    "1a": EvidenceLevel.HIGH,
                    "1b": EvidenceLevel.HIGH,
                    "2a": EvidenceLevel.MODERATE,
                    "2b": EvidenceLevel.MODERATE,
                    "3a": EvidenceLevel.LOW,
                    "3b": EvidenceLevel.LOW,
                    "4": EvidenceLevel.VERY_LOW,
                    "5": EvidenceLevel.EXPERT_OPINION,
                }
                evidence = loe_mapping.get(loe_code.lower(), EvidenceLevel.UNKNOWN)

            # Parse strength - handle both descriptive and code formats
            strength = RecommendationStrength.UNKNOWN
            str_str = rec_data.get("strength", "")
            sor_code = rec_data.get("sor_code", "")

            if str_str:
                try:
                    strength = RecommendationStrength(str_str.lower())
                except ValueError:
                    pass

            # If we have a SoR code, try to map it
            if strength == RecommendationStrength.UNKNOWN and sor_code:
                sor_mapping = {
                    "a": RecommendationStrength.STRONG,
                    "b": RecommendationStrength.CONDITIONAL,
                    "c": RecommendationStrength.WEAK,
                }
                strength = sor_mapping.get(sor_code.lower(), RecommendationStrength.UNKNOWN)

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
            guideline_name=data.get("guideline_name", "Unknown"),
            guideline_year=guideline_year,
            organization=data.get("organization"),
            target_condition=data.get("target_condition", "Unknown"),
            recommendations=recommendations,
            source_document=source,
            extraction_confidence=0.85 if recommendations else 0.0,
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

    def _loe_code_to_level(self, loe_code: str) -> EvidenceLevel:
        """Convert LoE code to EvidenceLevel enum."""
        code = loe_code.lower().replace(".", "")
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
            "na": EvidenceLevel.EXPERT_OPINION,
        }
        return mapping.get(code, EvidenceLevel.UNKNOWN)

    def _sor_code_to_strength(self, sor_code: str) -> RecommendationStrength:
        """Convert SoR grade to RecommendationStrength enum."""
        code = sor_code.upper()
        mapping = {
            "A": RecommendationStrength.STRONG,
            "B": RecommendationStrength.CONDITIONAL,
            "C": RecommendationStrength.WEAK,
        }
        return mapping.get(code, RecommendationStrength.UNKNOWN)

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
