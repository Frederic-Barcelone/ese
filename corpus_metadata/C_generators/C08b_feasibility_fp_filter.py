# corpus_metadata/C_generators/C08b_feasibility_fp_filter.py
"""
False positive filter for clinical trial feasibility extraction.

Multi-layer filter that rejects:
- Figure/Table/Appendix captions
- Definitions (X=...)
- Statistics/results (95% CI, p<, SE, SD)
- OCR garbage
- Bare fractions without epidemiology context
"""

from __future__ import annotations

import re
from typing import List, Tuple


class FeasibilityFalsePositiveFilter:
    """
    Multi-layer false positive filter for feasibility extraction.

    Filters out:
    - Figure/Table/Appendix captions
    - Definitions (X=...)
    - Statistics/results (95% CI, p<, SE, SD)
    - OCR garbage
    """

    # Layer 1: Caption patterns - never eligibility/epidemiology
    CAPTION_PATTERNS = [
        r"^(?:Figure|Fig\.?|Table|Appendix|Panel|Supplementary)\s*\d",
        r"^(?:Figure|Fig\.?|Table)\s+[A-Z]\d?",
    ]

    # Layer 2: Definition patterns (X=...) -> glossary, not eligibility
    DEFINITION_PATTERNS = [
        r"^[A-Z]{2,10}\s*=",  # e.g., "eGFR=..."
        r"^\w+\s*=\s*\w+",     # Variable definitions
    ]

    # Layer 3: Statistics/results patterns
    STATISTICS_PATTERNS = [
        r"95%?\s*CI",
        r"p\s*[<>=]\s*0?\.\d+",
        r"(?:^|\s)(?:SE|SD)\s*[=:]?\s*\d",
        r"model[\s-]?estimated",
        r"hazard\s*ratio",
        r"odds\s*ratio",
        r"relative\s*risk",
        r"confidence\s*interval",
        r"standard\s*(?:error|deviation)",
        r"interquartile\s*range",
        r"IQR",
    ]

    # Layer 4: OCR garbage patterns
    OCR_GARBAGE_PATTERNS = [
        r"[^\w\s]{3,}",  # 3+ consecutive special chars
        r"^\d+\s*[-—–]\s*\w+\s+\d+\s+\d+",  # Table row patterns like "8 — Baseline 14 30"
        r"^[\d\s,.-]+$",  # Only numbers and punctuation
    ]

    # Layer 5: Bare fractions without context (not epidemiology)
    BARE_FRACTION_PATTERN = r"^\d+/\d+$"

    def __init__(self) -> None:
        self.caption_re: List[re.Pattern] = [
            re.compile(p, re.IGNORECASE) for p in self.CAPTION_PATTERNS
        ]
        self.definition_re: List[re.Pattern] = [
            re.compile(p) for p in self.DEFINITION_PATTERNS
        ]
        self.statistics_re: List[re.Pattern] = [
            re.compile(p, re.IGNORECASE) for p in self.STATISTICS_PATTERNS
        ]
        self.ocr_re: List[re.Pattern] = [
            re.compile(p) for p in self.OCR_GARBAGE_PATTERNS
        ]
        self.bare_fraction_re: re.Pattern = re.compile(self.BARE_FRACTION_PATTERN)

    def is_caption(self, text: str) -> bool:
        """Check if text is a figure/table caption."""
        text = text.strip()
        return any(p.search(text) for p in self.caption_re)

    def is_definition(self, text: str) -> bool:
        """Check if text is a definition (X=...)."""
        text = text.strip()
        return any(p.match(text) for p in self.definition_re)

    def is_statistics(self, text: str) -> bool:
        """Check if text contains statistical results."""
        return any(p.search(text) for p in self.statistics_re)

    def is_ocr_garbage(self, text: str) -> bool:
        """Check if text looks like OCR noise."""
        text = text.strip()
        if len(text) < 5:
            return True
        return any(p.search(text) for p in self.ocr_re)

    def is_bare_fraction(self, text: str) -> bool:
        """Check if text is just a bare fraction like '11/37'."""
        return bool(self.bare_fraction_re.match(text.strip()))

    def filter_eligibility(self, text: str) -> Tuple[bool, str]:
        """
        Filter eligibility candidates.
        Returns (should_keep, rejection_reason).
        """
        if self.is_caption(text):
            return False, "caption"
        if self.is_definition(text):
            return False, "definition"
        if self.is_statistics(text):
            return False, "statistics"
        if self.is_ocr_garbage(text):
            return False, "ocr_garbage"
        return True, ""

    def filter_epidemiology(self, text: str, has_anchor: bool) -> Tuple[bool, str]:
        """
        Filter epidemiology candidates.
        Returns (should_keep, rejection_reason).
        """
        if self.is_caption(text):
            return False, "caption"
        if self.is_ocr_garbage(text):
            return False, "ocr_garbage"
        if self.is_bare_fraction(text) and not has_anchor:
            return False, "bare_fraction_no_anchor"
        if self.is_statistics(text) and not has_anchor:
            return False, "statistics_no_anchor"
        return True, ""


__all__ = ["FeasibilityFalsePositiveFilter"]
