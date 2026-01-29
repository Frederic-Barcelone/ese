# corpus_metadata/C_generators/C06a_disease_fp_filter.py
"""
Disease false positive filtering with confidence-based scoring.

Provides confidence adjustments instead of hard filtering for most cases,
preserving recall while allowing downstream components to use confidence scores.

Only hard-filters truly catastrophic false positives:
- Chromosome patterns in chromosome context
- Gene names with very strong gene context
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from A_core.A15_domain_profile import DomainProfile, load_domain_profile


class DiseaseFalsePositiveFilter:
    """
    Confidence-based scoring for disease matches.

    CHANGED: Converted from hard filtering to confidence adjustment.
    Most cases now adjust confidence instead of hard-rejecting.
    This prevents catastrophic recall loss on out-of-domain corpora.

    The filter uses:
    1. Domain profiles for domain-specific adjustments
    2. Generic patterns for truly universal filters
    3. Context analysis for disambiguation

    Only hard-filters truly catastrophic false positives (chromosomes, genes).
    """

    # Layer 1: Chromosome/karyotype patterns - HARD FILTER (universal)
    CHROMOSOME_PATTERNS = [
        r"^\d{1,2}[pq]$",  # 10p, 22q, etc.
        r"^\d{1,2}[pq]\d+",  # 10p15, 22q11
        r"^\d{1,2}[pq]\d+\.\d+",  # 22q11.2
        r"^4[0-9],X{1,2}Y?$",  # 45,X, 46,XX, 46,XY
        r"^del\(\d+[pq]?\)",  # del(7q), del(5q)
        r"^t\(\d+;\d+\)",  # t(9;22), t(4;14)
        r"^inv\(\d+\)",  # inv(16)
        r"^dup\(\d+\)",  # dup(7)
        r"^\+\d{1,2}$",  # +21, +13 (trisomy notation)
        r"^-\d{1,2}$",  # -7, -5 (monosomy notation)
    ]

    # Context keywords for disambiguation
    CHROMOSOME_CONTEXT_KEYWORDS = [
        "chromosome", "karyotype", "cytogenetic", "translocation",
        "deletion", "duplication", "trisomy", "monosomy", "band",
        "breakpoint", "FISH", "CGH", "array", "copy number",
        "ploidy", "aneuploidy", "mosaicism",
    ]

    DISEASE_CONTEXT_KEYWORDS = [
        "syndrome", "disease", "disorder", "condition", "patient",
        "diagnosis", "diagnosed", "treatment", "therapy", "symptom",
        "clinical", "prognosis", "affected", "prevalence", "incidence",
        "rare", "orphan", "trial", "study",
    ]

    # Gene context (for disambiguation)
    GENE_PATTERN = r"^[A-Z][A-Z0-9]{1,6}$"  # BRCA1, TP53, EGFR, etc.
    GENE_CONTEXT_KEYWORDS = [
        "mutation", "variant", "expression", "gene", "protein",
        "encoded", "pathway", "receptor", "kinase", "transcription",
        "allele", "polymorphism", "genotype",
    ]

    # Short match threshold
    SHORT_MATCH_THRESHOLD = 4

    def __init__(self, domain_profile: Optional[DomainProfile] = None):
        """
        Initialize filter with optional domain profile.

        Args:
            domain_profile: Domain-specific configuration. If None, uses generic.
        """
        self._compiled_chr_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.CHROMOSOME_PATTERNS
        ]
        self._gene_pattern = re.compile(self.GENE_PATTERN)

        # Load domain profile (defaults to generic if not provided)
        self.domain_profile = domain_profile or load_domain_profile("generic")

    def score_adjustment(
        self,
        matched_text: str,
        context: str,
        is_abbreviation: bool = False,
    ) -> Tuple[float, str]:
        """
        Calculate confidence adjustment for a match.

        CHANGED: Primary method - returns adjustment instead of filter decision.

        Returns:
            (adjustment, reason) where adjustment is -1.0 to +0.3
            Negative = likely FP, Positive = domain-relevant boost
        """
        matched_clean = matched_text.strip()
        ctx_lower = context.lower()

        # Start with domain profile adjustment
        is_short = len(matched_clean) <= self.SHORT_MATCH_THRESHOLD and not is_abbreviation
        is_citation = self._is_journal_citation_context(ctx_lower)

        adjustment = self.domain_profile.get_confidence_adjustment(
            matched_text=matched_text,
            context=context,
            is_short_match=is_short,
            is_citation_context=is_citation,
        )

        reason = ""

        # Additional context-based adjustments
        if self._is_chromosome_context(ctx_lower):
            adjustment -= 0.3
            reason = "chromosome_context"

        if self._is_gene_as_gene(matched_clean, ctx_lower):
            adjustment -= 0.25
            reason = "gene_context"

        # Positive adjustment if strong disease context
        if self._has_disease_context(ctx_lower):
            adjustment += 0.1
            if not reason:
                reason = "disease_context_boost"

        return adjustment, reason

    def should_filter(
        self, matched_text: str, context: str, is_abbreviation: bool = False
    ) -> Tuple[bool, str]:
        """
        Determine if a match should be HARD filtered.

        CHANGED: Now only hard-filters truly catastrophic FPs.
        Most filtering is done via score_adjustment() instead.

        Returns:
            (should_filter, reason)
        """
        matched_clean = matched_text.strip()
        ctx_lower = context.lower()

        # Hard filter 1: Chromosome patterns in chromosome context
        for pattern in self._compiled_chr_patterns:
            if pattern.match(matched_clean):
                if self._is_chromosome_context(ctx_lower):
                    return True, "chromosome_pattern_in_chromosome_context"

        # Hard filter 2: Gene names used clearly as genes
        if self._is_gene_as_gene(matched_clean, ctx_lower):
            # Only hard filter if very strong gene context
            gene_score = sum(1 for kw in self.GENE_CONTEXT_KEYWORDS if kw in ctx_lower)
            if gene_score >= 3:
                return True, "strong_gene_context"

        # Hard filter 3: Domain profile catastrophic FPs
        should_filter, reason = self.domain_profile.should_hard_filter(
            matched_text, context
        )
        if should_filter:
            return True, reason

        # Everything else: use confidence adjustment, not hard filter
        return False, ""

    def _is_chromosome_context(self, ctx_lower: str) -> bool:
        """Check if context suggests chromosome/cytogenetic usage."""
        chr_score = sum(1 for kw in self.CHROMOSOME_CONTEXT_KEYWORDS if kw in ctx_lower)
        dis_score = sum(1 for kw in self.DISEASE_CONTEXT_KEYWORDS if kw in ctx_lower)
        return chr_score > dis_score

    def _has_disease_context(self, ctx_lower: str) -> bool:
        """Check if context contains disease-related keywords."""
        return any(kw in ctx_lower for kw in self.DISEASE_CONTEXT_KEYWORDS)

    def _is_gene_as_gene(self, matched_text: str, ctx_lower: str) -> bool:
        """Check if text is a gene name being used as a gene (not disease)."""
        if not self._gene_pattern.match(matched_text):
            return False

        # Count gene vs disease context keywords
        gene_score = sum(1 for kw in self.GENE_CONTEXT_KEYWORDS if kw in ctx_lower)
        dis_score = sum(1 for kw in self.DISEASE_CONTEXT_KEYWORDS if kw in ctx_lower)

        # If strong gene context and weak disease context, it's a gene
        return gene_score >= 2 and gene_score > dis_score

    def _is_journal_citation_context(self, ctx_lower: str) -> bool:
        """Check if context is a journal citation."""
        citation_indicators = [
            r"\d{4};\s*\d+",  # year; volume
            r"vol\.\s*\d+",
            r"pp?\.\s*\d+",
            "doi:", "pmid:",
            r"\[\d+\]",
            "et al", "reference", "citation",
        ]

        for indicator in citation_indicators:
            if indicator in ctx_lower:
                return True

        if re.search(r"\d{4}\s*;\s*\d+\s*:\s*\d+", ctx_lower):
            return True

        return False


__all__ = ["DiseaseFalsePositiveFilter"]
