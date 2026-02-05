"""
Drug false positive filtering using curated exclusion sets.

This module filters false positive drug matches using extensive curated sets
of non-drug terms. Identifies bacteria, biological entities, common words,
and other categories that frequently trigger false positives in drug detection.

Key Components:
    - DrugFalsePositiveFilter: Main filter class for drug FP detection
    - DRUG_ABBREVIATIONS: Valid drug abbreviations to preserve
    - Pattern-based filtering for:
        - NCT trial identifiers
        - Ethics committee codes
        - Minimum length validation
    - Constants imported from C26_drug_fp_constants

Example:
    >>> from C_generators.C25_drug_fp_filter import DrugFalsePositiveFilter
    >>> filter = DrugFalsePositiveFilter()
    >>> filter.is_false_positive("influenza", "scispacy")
    True  # Filtered as virus name
    >>> filter.is_false_positive("ravulizumab", "lexicon")
    False  # Valid drug name

Dependencies:
    - C_generators.C26_drug_fp_constants: All constant sets (ALWAYS_FILTER, etc.)
    - A_core.A06_drug_models: DrugGeneratorType (TYPE_CHECKING)
    - re: Regular expression matching
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Set

from .C26_drug_fp_constants import (
    ALWAYS_FILTER,
    BACTERIA_ORGANISMS,
    BIOLOGICAL_ENTITIES,
    BIOLOGICAL_SUFFIXES,
    BODY_PARTS,
    COMMON_WORDS,
    CREDENTIALS,
    DRUG_ABBREVIATIONS,
    EQUIPMENT_PROCEDURES,
    FP_SUBSTRINGS,
    NER_FALSE_POSITIVES,
    NON_DRUG_ALLCAPS,
    ORGANIZATIONS,
    PHARMA_COMPANY_NAMES,
    TRIAL_STATUS_TERMS,
    VACCINE_TERMS,
)

if TYPE_CHECKING:
    from A_core.A06_drug_models import DrugGeneratorType


class DrugFalsePositiveFilter:
    """Filter false positive drug matches."""

    # NCT trial ID pattern (clinical trial identifiers, not drugs)
    NCT_PATTERN = re.compile(r"^NCT\d+$", re.IGNORECASE)

    # Ethics committee approval code patterns (not drugs)
    ETHICS_CODE_PATTERN = re.compile(
        r"^(?:KY|IRB|EC|REC|IEC|ERB|REB)\d{4}$", re.IGNORECASE
    )

    # Minimum drug name length
    MIN_LENGTH = 3

    def __init__(self) -> None:
        self.common_words_lower = {w.lower() for w in COMMON_WORDS}
        self.body_parts_lower = {w.lower() for w in BODY_PARTS}
        self.trial_status_lower = {w.lower() for w in TRIAL_STATUS_TERMS}
        self.equipment_lower = {w.lower() for w in EQUIPMENT_PROCEDURES}
        self.non_drug_allcaps_lower = {w.lower() for w in NON_DRUG_ALLCAPS}
        self.always_filter_lower = {w.lower() for w in ALWAYS_FILTER}
        self.bacteria_organisms_lower = {w.lower() for w in BACTERIA_ORGANISMS}
        self.vaccine_terms_lower = {w.lower() for w in VACCINE_TERMS}
        self.credentials_lower = {w.lower() for w in CREDENTIALS}
        self.biological_entities_lower = {w.lower() for w in BIOLOGICAL_ENTITIES}
        self.ner_false_positives_lower = {w.lower() for w in NER_FALSE_POSITIVES}
        self.pharma_company_names_lower = {w.lower() for w in PHARMA_COMPANY_NAMES}
        self.organizations_lower = {w.lower() for w in ORGANIZATIONS}
        self.fp_substrings_lower = [s.lower() for s in FP_SUBSTRINGS]

        # Load gene symbols from lexicon (genes are not drugs)
        self.gene_symbols_lower: Set[str] = set()
        self._load_gene_lexicon()

    def _load_gene_lexicon(self) -> None:
        """Load gene symbols to filter from drug matches."""
        gene_path = Path("ouput_datasources/2025_08_orphadata_genes.json")
        if not gene_path.exists():
            return
        try:
            with open(gene_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Only load primary gene symbols (not aliases) for drug filtering
            for entry in data:
                if entry.get("source") == "orphadata_hgnc":
                    symbol = entry.get("term", "").lower()
                    if symbol and len(symbol) >= 3:
                        self.gene_symbols_lower.add(symbol)
        except Exception:
            pass  # Silently fail - gene filtering is optional enhancement

    def is_false_positive(
        self, matched_text: str, context: str, generator_type: "DrugGeneratorType"
    ) -> bool:
        """
        Check if a drug match is likely a false positive.

        Returns True if the match should be filtered out.
        """
        # Import here to avoid circular imports
        from A_core.A06_drug_models import DrugGeneratorType as DGT

        text_lower = matched_text.lower().strip()
        text_stripped = matched_text.strip()

        # Skip very short matches
        if len(text_lower) < self.MIN_LENGTH:
            return True

        # Filter NCT trial IDs (e.g., NCT04817618)
        if self.NCT_PATTERN.match(text_stripped):
            return True

        # Filter ethics committee approval codes (e.g., KY2022, IRB2023)
        if self.ETHICS_CODE_PATTERN.match(text_stripped):
            return True

        # Always filter generic placeholder terms
        if text_lower in self.always_filter_lower:
            return True

        # Always filter trial status terms
        if text_lower in self.trial_status_lower:
            return True

        # Check if text contains any trial status term
        text_normalized = text_lower.replace("_", " ")
        for status in self.trial_status_lower:
            if status in text_normalized:
                return True

        # Filter text containing trial status in parentheses
        if "(" in text_stripped and ")" in text_stripped:
            paren_content = text_stripped[text_stripped.find("(")+1:text_stripped.find(")")]
            paren_lower = paren_content.lower().replace("_", " ")
            for status in self.trial_status_lower:
                if status in paren_lower:
                    return True
            base_word = text_stripped[:text_stripped.find("(")].strip().lower()
            if base_word in self.common_words_lower:
                return True

        # Always filter bacteria/organism names
        if text_lower in self.bacteria_organisms_lower:
            return True

        # Check for partial bacteria matches
        for organism in self.bacteria_organisms_lower:
            if organism in text_lower:
                return True

        # Always filter vaccine-related terms
        if text_lower in self.vaccine_terms_lower:
            return True

        # Always filter credentials
        if text_lower in self.credentials_lower:
            return True

        # Always filter biological entities
        if text_lower in self.biological_entities_lower:
            return True

        # Always filter pharmaceutical company names
        if text_lower in self.pharma_company_names_lower:
            return True
        for company in self.pharma_company_names_lower:
            if company in text_lower or text_lower in company:
                return True

        # Always filter organizations and agencies
        if text_lower in self.organizations_lower:
            return True
        for org in self.organizations_lower:
            if org in text_lower:
                return True

        # Filter NER-specific false positives
        if text_lower in self.ner_false_positives_lower:
            return True

        # Substring-based filtering
        for fp_substr in self.fp_substrings_lower:
            if fp_substr in text_lower:
                return True

        # Pattern-based filtering for NER and lexicon results
        if generator_type in {DGT.SCISPACY_NER, DGT.LEXICON_RXNORM, DGT.LEXICON_FDA}:
            for suffix in BIOLOGICAL_SUFFIXES:
                if text_lower.endswith(suffix):
                    return True

        # Always filter body parts
        if text_lower in self.body_parts_lower:
            return True

        # Always filter equipment/procedures
        if text_lower in self.equipment_lower:
            return True

        # Filter gene symbols
        if generator_type in {
            DGT.SCISPACY_NER,
            DGT.LEXICON_RXNORM,
            DGT.LEXICON_FDA,
        }:
            if text_lower in self.gene_symbols_lower:
                return True

        # Skip common words (unless from specialized lexicon)
        if generator_type not in {
            DGT.LEXICON_ALEXION,
            DGT.LEXICON_INVESTIGATIONAL,
        }:
            if text_lower in self.common_words_lower:
                return True
            if text_lower in self.non_drug_allcaps_lower:
                return True

        # Context-based author name detection
        if context:
            ctx_lower = context.lower()
            author_indicators = [
                f"by {text_lower},",
                f"by {text_lower}.",
                f"{text_lower} et al",
                f", {text_lower},",
                f", {text_lower}.",
            ]
            for indicator in author_indicators:
                if indicator in ctx_lower:
                    return True

            # Check for author initials pattern (uppercase only — not IGNORECASE
            # to avoid matching drug + common 1-2 letter words like "to", "at", "i")
            author_pattern = re.compile(
                rf"\b{re.escape(text_stripped)}\s+[A-Z]{{1,2}}[,\.\s]"
            )
            if author_pattern.search(context):
                return True

        return False


# Re-export for backward compatibility
__all__ = ["DrugFalsePositiveFilter", "DRUG_ABBREVIATIONS"]
