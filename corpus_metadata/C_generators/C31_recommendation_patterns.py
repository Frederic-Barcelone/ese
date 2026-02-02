# corpus_metadata/C_generators/C31_recommendation_patterns.py
"""
Pattern constants and prompts for guideline recommendation extraction.

Contains:
- Organization patterns (EULAR, ACR, etc.)
- Guideline title patterns
- Drug, dose, and duration patterns
- Evidence level and strength patterns
- LLM and VLM prompt templates
"""

from __future__ import annotations

import re
from typing import Dict, List, Pattern

from A_core.A18_recommendation_models import (
    EvidenceLevel,
    RecommendationStrength,
)


# =============================================================================
# GUIDELINE METADATA PATTERNS
# =============================================================================

# Known guideline organizations
ORGANIZATION_PATTERNS: Dict[str, str] = {
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
GUIDELINE_TITLE_PATTERNS: List[str] = [
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
CONDITION_PATTERNS: List[str] = [
    r"(?:management|treatment|diagnosis)\s+of\s+([A-Z][^,.]+?)(?:\s*[,:.]|\s+in\s+|\s+with\s+|\s*$)",
    r"patients?\s+with\s+([A-Z][a-zA-Z\s-]+(?:vasculitis|disease|syndrome|disorder))",
]


# =============================================================================
# EXTRACTION PATTERNS
# =============================================================================

# Drug name patterns for extraction
DRUG_PATTERNS: Dict[str, str] = {
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
DOSE_PATTERN: Pattern = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*(mg|g|mcg|µg)(?:/(?:day|d|kg|m2))?",
    re.IGNORECASE
)

# Duration patterns
DURATION_PATTERN: Pattern = re.compile(
    r"(\d+)\s*(?:-\s*(\d+))?\s*(weeks?|months?|years?|days?)",
    re.IGNORECASE
)

# Evidence level patterns - includes GRADE/Oxford levels
EVIDENCE_PATTERNS: Dict[str, EvidenceLevel] = {
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
STRENGTH_PATTERNS: Dict[str, RecommendationStrength] = {
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
SEVERITY_PATTERNS: List[str] = [
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
SPECIFIC_CONDITION_PATTERNS: List[str] = [
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


# =============================================================================
# VLM PROMPT FOR LOE/SOR EXTRACTION
# =============================================================================

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


__all__ = [
    "ORGANIZATION_PATTERNS",
    "GUIDELINE_TITLE_PATTERNS",
    "CONDITION_PATTERNS",
    "DRUG_PATTERNS",
    "DOSE_PATTERN",
    "DURATION_PATTERN",
    "EVIDENCE_PATTERNS",
    "STRENGTH_PATTERNS",
    "SEVERITY_PATTERNS",
    "SPECIFIC_CONDITION_PATTERNS",
    "RECOMMENDATION_EXTRACTION_PROMPT",
    "VLM_LOE_SOR_EXTRACTION_PROMPT",
]
