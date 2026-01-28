# corpus_metadata/C_generators/C12_guideline_recommendation_extractor.py
"""
Guideline Recommendation Extractor.

Extracts structured clinical recommendations from guideline documents,
including treatment recommendations, dosing guidance, and evidence levels.

Works with:
- Recommendation text blocks
- Summary tables (with VLM fallback for garbled PDF extraction)
- Treatment algorithm figures
"""

from __future__ import annotations

import base64
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

# Optional PyMuPDF for page image rendering
try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False


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

EVIDENCE MAPPING (Oxford/CEBM levels):
- 1a, 1b → high (meta-analyses, RCTs)
- 2a, 2b → moderate (cohort studies)
- 3a, 3b → low (case-control, case series)
- 4 → very_low (expert opinion without appraisal)
- 5, na, consensus → expert_opinion

STRENGTH MAPPING:
- Grade A, "we recommend" → strong
- Grade B, "should consider" → conditional
- Grade C, "may consider" → weak

IMPORTANT - TABLE FORMAT ISSUES:
PDF table extraction can garble table data. Look for LoE/SoR codes in these patterns:

1. EMBEDDED IN TEXT: Codes appear IN THE MIDDLE of recommendation text due to column extraction:
   "16 In patients with AAV, we recommend serum 1b B 100 9.2 immunoglobulin testing"
   → The "1b B 100 9.2" is LoE=1b, SoR=B, FV%=100, LoA=9.2

2. BUNCHED AT END: All codes may be grouped together after all recommendation text:
   "...recommend rituximab. na. na. 3b 1a 1b na. A B A B na. 90 100 100..."
   → Codes are in order matching the recommendations

3. If you find a PRE-EXTRACTED MAPPING section, use those mappings directly.

CRITICAL:
- Only extract ACTUAL recommendations (numbered items, not overarching principles)
- Each numbered recommendation (1, 2, 3...) is ONE entry
- Use the LoE/SoR codes if you can find them, otherwise infer from language
- Keep responses concise - max 20 recommendations
- Return VALID JSON only"""


# VLM prompt for extracting LoE/SoR from recommendation table images
VLM_LOE_SOR_EXTRACTION_PROMPT = """You are looking at a guideline recommendation table image.

Extract the Level of Evidence (LoE), Strength of Recommendation (SoR), and key identifying text for each numbered recommendation.

The table typically has columns: Recommendation Number, Recommendation Text, LoE, SoR, FV%, LoA

Return JSON with recommendation data:
{
    "recommendations": [
        {"rec_num": 1, "loe": "1b", "sor": "A", "text": "A positive biopsy is strongly supportive of a diagnosis of vasculitis", "keywords": ["biopsy", "diagnosis", "vasculitis"]},
        {"rec_num": 2, "loe": "2a", "sor": "B", "text": "In patients with signs and/or symptoms raising suspicion of AAV, test for PR3-ANCA and MPO-ANCA", "keywords": ["PR3-ANCA", "MPO-ANCA", "testing"]},
        {"rec_num": 3, "loe": "na", "sor": "C", "text": "For induction of remission in patients with new-onset or relapsing GPA or MPA with organ-threatening disease", "keywords": ["induction", "remission", "organ-threatening", "rituximab", "cyclophosphamide"]}
    ]
}

IMPORTANT:
- Only include NUMBERED recommendations (1, 2, 3...), not overarching principles
- LoE codes are typically: 1a, 1b, 2a, 2b, 3a, 3b, 4, 5, or na
- SoR codes are typically: A, B, C, D, or na
- If a code has a footnote marker (*, †), ignore the marker
- For "text", extract ~80-100 characters of the recommendation text (enough to identify it)
- For "keywords", extract 3-5 distinctive medical terms/drug names that uniquely identify this recommendation
- Return valid JSON only"""


# =============================================================================
# GUIDELINE RECOMMENDATION EXTRACTOR CLASS
# =============================================================================


class GuidelineRecommendationExtractor:
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
        self.pdf_path = pdf_path
        # Cache: rec_num -> (loe_code, sor_code, text_snippet, keywords)
        self._vlm_loe_sor_cache: Dict[str, Tuple[str, str, str, List[str]]] = {}

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
                    result = self._parse_llm_response(data, source, page_num)
                    # Apply VLM-extracted codes to recommendations missing evidence/strength
                    if result:
                        self._apply_vlm_codes_to_recommendations(result)
                    return result
                except json.JSONDecodeError as je:
                    # Try to fix common JSON issues
                    fixed = self._attempt_json_fix(text_response)
                    if fixed:
                        data = json.loads(fixed)
                        result = self._parse_llm_response(data, source, page_num)
                        # Apply VLM-extracted codes to recommendations missing evidence/strength
                        if result:
                            self._apply_vlm_codes_to_recommendations(result)
                        return result
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

    def extract_loe_sor_with_vlm(self, text: str) -> Dict[str, Tuple[str, str, str, List[str]]]:
        """
        Extract LoE/SoR codes from recommendation table using VLM.

        When text-based extraction fails due to garbled PDF parsing,
        this method renders the relevant pages as images and uses
        vision LLM to extract the table structure.

        Args:
            text: Document text (used to identify table pages)

        Returns:
            Dict mapping recommendation number to (loe_code, sor_code, text_snippet, keywords) tuple
        """
        if not self.pdf_path or not self.llm_client:
            return {}

        if not PYMUPDF_AVAILABLE or fitz is None:
            return {}

        # Return cached results if available
        if self._vlm_loe_sor_cache:
            return self._vlm_loe_sor_cache

        # Find pages containing recommendation tables
        table_pages = self._find_recommendation_table_pages(text)
        if not table_pages:
            return {}

        codes: Dict[str, Tuple[str, str, str, List[str]]] = {}

        for page_num in table_pages:
            # Render page as image
            img_base64 = self._render_page_as_image(page_num)
            if not img_base64:
                continue

            # Extract LoE/SoR using VLM
            page_codes = self._extract_loe_sor_from_image(img_base64)
            for rec_num, (loe, sor, text_snippet, keywords) in page_codes.items():
                if rec_num not in codes:
                    codes[rec_num] = (loe, sor, text_snippet, keywords)

        self._vlm_loe_sor_cache = codes
        if codes:
            print(f"    [VLM] Extracted {len(codes)} LoE/SoR codes with text snippets:")
            for rec_num, (loe, sor, text_snippet, keywords) in sorted(codes.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
                snippet_preview = text_snippet[:50] + "..." if text_snippet else "(no text)"
                kw_str = ", ".join(keywords[:3]) if keywords else "(no keywords)"
                print(f"      Rec {rec_num}: LoE={loe}, SoR={sor}, text=\"{snippet_preview}\", kw=[{kw_str}]")
        return codes

    def _find_recommendation_table_pages(self, text: str) -> List[int]:
        """
        Find page numbers containing recommendation tables with LoE/SoR.

        Args:
            text: Document text with page markers

        Returns:
            List of 1-indexed page numbers
        """
        pages = []

        # Split text by page markers
        page_pattern = re.compile(r'---\s*Page\s+(\d+)\s*---', re.IGNORECASE)
        page_matches = list(page_pattern.finditer(text))

        for i, match in enumerate(page_matches):
            page_num = int(match.group(1))
            start = match.end()
            end = page_matches[i + 1].start() if i + 1 < len(page_matches) else len(text)
            page_text = text[start:end]

            # Check if this page has recommendation table content
            # Match: "Table X recommendation", "Table X ... LoE SoR", "Table X Continued",
            # or pages with numbered recommendations that have LoE/SoR codes
            if re.search(r'Table\s+\d+.*(?:recommendation|LoE\s+SoR|Continued)', page_text, re.IGNORECASE):
                pages.append(page_num)
            # Also check for numbered recommendations with embedded LoE/SoR codes
            elif re.search(r'\d{1,2}\s+(?:In\s+patients|For\s+patients|We\s+recommend).*?[1-5][ab]?\s+[ABCD]\s+\d{2,3}', page_text, re.IGNORECASE):
                pages.append(page_num)

        return pages

    def _render_page_as_image(self, page_num: int, dpi: int = 150) -> Optional[str]:
        """
        Render a PDF page as base64-encoded PNG image.

        Args:
            page_num: 1-indexed page number
            dpi: Resolution for rendering

        Returns:
            Base64-encoded PNG string, or None if fails
        """
        if not PYMUPDF_AVAILABLE or fitz is None or not self.pdf_path:
            return None

        try:
            doc = fitz.open(self.pdf_path)
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return None

            page = doc[page_num - 1]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            doc.close()
            return img_base64

        except Exception as e:
            print(f"[WARN] Failed to render page {page_num}: {e}")
            return None

    def _extract_loe_sor_from_image(self, img_base64: str) -> Dict[str, Tuple[str, str, str, List[str]]]:
        """
        Extract LoE/SoR codes, text snippets, and keywords from a table image using VLM.

        Args:
            img_base64: Base64-encoded PNG image

        Returns:
            Dict mapping recommendation number to (loe_code, sor_code, text_snippet, keywords) tuple
        """
        if not self.llm_client or not img_base64:
            return {}

        try:
            # Call VLM with the image
            response = self.llm_client.messages.create(
                model=self.llm_model,
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": VLM_LOE_SOR_EXTRACTION_PROMPT,
                            },
                        ],
                    }
                ],
            )

            if not response.content:
                return {}

            # Parse JSON response
            import json
            text_response = response.content[0].text.strip()
            text_response = self._clean_json_response(text_response)
            data = json.loads(text_response)

            codes: Dict[str, Tuple[str, str, str, List[str]]] = {}
            for rec in data.get("recommendations", []):
                rec_num = str(rec.get("rec_num", ""))
                loe = rec.get("loe", "").lower()
                sor = rec.get("sor", "").upper()
                text_snippet = rec.get("text", "")
                keywords = rec.get("keywords", [])
                if isinstance(keywords, str):
                    keywords = [keywords]
                if rec_num and (loe or sor):
                    codes[rec_num] = (loe, sor, text_snippet, keywords)

            return codes

        except Exception as e:
            print(f"[WARN] VLM LoE/SoR extraction failed: {e}")
            return {}

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

    def _apply_vlm_codes_to_recommendations(
        self,
        rec_set: RecommendationSet,
    ) -> None:
        """
        Apply VLM-extracted LoE/SoR codes to recommendations using text and keyword matching.

        This post-processing step matches recommendations by text similarity
        and keyword overlap, handling cases where LLM extraction creates
        different recommendation groupings than the PDF table.

        Args:
            rec_set: RecommendationSet to update (modified in place)
        """
        if not self._vlm_loe_sor_cache:
            return

        def normalize_text(text: str) -> str:
            """Normalize text for comparison."""
            if not text:
                return ""
            text = text.lower()
            text = re.sub(r'[\s\-–—]+', ' ', text)
            text = re.sub(r'[^\w\s]', '', text)
            return text.strip()

        def get_content_words(text: str) -> set:
            """Extract meaningful content words, filtering out common words."""
            if not text:
                return set()
            # Common words to ignore in matching
            stopwords = {
                'the', 'a', 'an', 'in', 'of', 'for', 'with', 'and', 'or', 'to',
                'is', 'are', 'be', 'we', 'recommend', 'should', 'may', 'can',
                'patients', 'patient', 'treatment', 'disease', 'therapy',
                'new', 'onset', 'relapsing', 'that', 'this', 'as', 'by', 'on'
            }
            words = set(normalize_text(text).split())
            return words - stopwords

        def text_similarity(text1: str, text2: str) -> float:
            """Calculate word overlap similarity using content words."""
            words1 = get_content_words(text1)
            words2 = get_content_words(text2)
            if not words1 or not words2:
                return 0.0
            intersection = len(words1 & words2)
            return intersection / min(len(words1), len(words2))

        def keyword_match_score(keywords: List[str], text: str) -> float:
            """Calculate how many keywords appear in the text."""
            if not keywords or not text:
                return 0.0
            text_lower = text.lower()
            matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            return matches / len(keywords) if keywords else 0.0

        # Build list of VLM entries for matching
        vlm_entries = []
        for rec_num, (loe_code, sor_code, text_snippet, keywords) in self._vlm_loe_sor_cache.items():
            vlm_entries.append({
                'rec_num': rec_num,
                'loe_code': loe_code,
                'sor_code': sor_code,
                'text_snippet': text_snippet,
                'keywords': keywords or [],
                'matched': False,
            })

        # Match each recommendation to best VLM entry by combined text + keyword similarity
        for rec in rec_set.recommendations:
            rec_text = rec.source_text or rec.action or ""
            if not rec_text:
                continue

            best_match = None
            best_score = 0.0
            best_text_score = 0.0
            best_kw_score = 0.0

            for entry in vlm_entries:
                if entry['matched']:
                    continue

                snippet = entry['text_snippet']
                keywords = entry['keywords']

                # Calculate text similarity (using more of the text)
                text_score = text_similarity(snippet, rec_text[:150]) if snippet else 0.0

                # Calculate keyword match score
                kw_score = keyword_match_score(keywords, rec_text)

                # Combined score: weighted average favoring keywords when available
                if keywords:
                    combined_score = 0.4 * text_score + 0.6 * kw_score
                else:
                    combined_score = text_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_text_score = text_score
                    best_kw_score = kw_score
                    best_match = entry

            # Apply match if score is above threshold
            # Lower threshold (0.25) since we now use keyword matching
            if best_match and best_score >= 0.25:
                best_match['matched'] = True
                loe_code = best_match['loe_code']
                sor_code = best_match['sor_code']

                # Debug output
                match_detail = f"txt={best_text_score:.2f}"
                if best_match['keywords']:
                    match_detail += f", kw={best_kw_score:.2f}"
                print(f"    [VLM-MATCH] Rec '{rec_text[:40]}...' -> PDF rec {best_match['rec_num']} (score={best_score:.2f}, {match_detail})")

                # Apply LoE from VLM - VLM reads actual PDF table, so it's authoritative
                # For matches above 0.5, always apply VLM codes
                # For lower scores (0.25-0.5), only apply if current level is uncertain
                new_evidence = self._loe_code_to_level(loe_code)
                should_update_evidence = (
                    new_evidence != EvidenceLevel.UNKNOWN and (
                        best_score >= 0.5 or  # Good confidence - trust VLM
                        rec.evidence_level in (EvidenceLevel.UNKNOWN, EvidenceLevel.EXPERT_OPINION)
                    )
                )
                if should_update_evidence and new_evidence != rec.evidence_level:
                    print(f"      Updated evidence: {rec.evidence_level.value} -> {new_evidence.value} (from LoE={loe_code})")
                    rec.evidence_level = new_evidence

                # Apply SoR from VLM with same logic
                new_strength = self._sor_code_to_strength(sor_code)
                should_update_strength = (
                    new_strength != RecommendationStrength.UNKNOWN and (
                        best_score >= 0.5 or
                        rec.strength == RecommendationStrength.UNKNOWN
                    )
                )
                if should_update_strength and new_strength != rec.strength:
                    print(f"      Updated strength: {rec.strength.value} -> {new_strength.value} (from SoR={sor_code})")
                    rec.strength = new_strength

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
        - References to RCTs, meta-analyses → HIGH
        - References to cohort studies, clinical experience → MODERATE
        - "May consider", "based on expert opinion" → EXPERT_OPINION

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
        - "We recommend", "should" → STRONG
        - "Should consider", "is recommended" → CONDITIONAL
        - "May consider", "can be considered" → WEAK

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
