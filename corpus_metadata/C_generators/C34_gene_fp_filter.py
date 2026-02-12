"""
Gene false positive filtering for ambiguous gene symbols.

This module filters false positive gene matches, handling the high ambiguity
of gene symbols that clash with common abbreviations, statistics terms, units,
and other non-gene entities. Provides context-aware filtering to preserve recall.

Key Components:
    - GeneFalsePositiveFilter: Main filter for gene false positives
    - STATISTICAL_TERMS: Terms like OR, HR, CI that look like gene symbols
    - MIN_LENGTH: Minimum gene symbol length
    - Context-aware filtering using surrounding text
    - Generator-specific filtering rules

Example:
    >>> from C_generators.C34_gene_fp_filter import GeneFalsePositiveFilter
    >>> filter = GeneFalsePositiveFilter()
    >>> filter.is_false_positive("OR", "hazard ratio OR 1.5", "pattern")
    True  # Filtered as statistics term
    >>> filter.is_false_positive("BRCA1", "BRCA1 mutation carriers", "lexicon")
    False  # Valid gene symbol

Dependencies:
    - A_core.A19_gene_models: GeneGeneratorType
    - Z_utils.Z12_data_loader: YAML data loading
    - json, pathlib: For loading external gene data
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from A_core.A19_gene_models import GeneGeneratorType
from Z_utils.Z12_data_loader import load_term_set
from Z_utils.Z15_lexicon_provider import CREDENTIALS as _CREDENTIALS, build_country_alpha2_codes


class GeneFalsePositiveFilter:
    """
    Filter false positive gene matches.

    Gene symbols are highly ambiguous - many clash with common abbreviations,
    statistics terms, units, and other non-gene entities.
    """

    MIN_LENGTH = 2  # Minimum gene symbol length

    # All term sets loaded from G_config/data/gene_fp_terms.yaml
    STATISTICAL_TERMS: Set[str] = load_term_set("gene_fp_terms.yaml", "statistical_terms")
    UNITS: Set[str] = load_term_set("gene_fp_terms.yaml", "units")
    CLINICAL_TERMS: Set[str] = load_term_set("gene_fp_terms.yaml", "clinical_terms")
    COUNTRIES: Set[str] = set(build_country_alpha2_codes())
    CREDENTIALS: Set[str] = set(_CREDENTIALS)
    DRUG_TERMS: Set[str] = load_term_set("gene_fp_terms.yaml", "drug_terms")
    STUDY_TERMS: Set[str] = load_term_set("gene_fp_terms.yaml", "study_terms")
    COMMON_ENGLISH_WORDS: Set[str] = load_term_set("gene_fp_terms.yaml", "common_english_words")
    GENE_CONTEXT_KEYWORDS: Set[str] = load_term_set("gene_fp_terms.yaml", "gene_context_keywords")
    NON_GENE_CONTEXT_KEYWORDS: Set[str] = load_term_set("gene_fp_terms.yaml", "non_gene_context_keywords")
    QUESTIONNAIRE_TERMS: Set[str] = load_term_set("gene_fp_terms.yaml", "questionnaire_terms")
    ANTIBODY_ABBREVIATIONS: Set[str] = load_term_set("gene_fp_terms.yaml", "antibody_abbreviations")
    ANTIBODY_CONTEXT_KEYWORDS: Set[str] = load_term_set("gene_fp_terms.yaml", "antibody_context_keywords")
    SHORT_GENES_NEED_CONTEXT: Set[str] = load_term_set("gene_fp_terms.yaml", "short_genes_need_context")

    def __init__(self, lexicon_base_path: Optional[Path] = None):
        self.statistical_lower = {w.lower() for w in self.STATISTICAL_TERMS}
        self.units_lower = {w.lower() for w in self.UNITS}
        self.clinical_lower = {w.lower() for w in self.CLINICAL_TERMS}
        self.countries_lower = {w.lower() for w in self.COUNTRIES}
        self.credentials_lower = {w.lower() for w in self.CREDENTIALS}
        self.drug_terms_lower = {w.lower() for w in self.DRUG_TERMS}
        self.study_terms_lower = {w.lower() for w in self.STUDY_TERMS}
        self.common_english_lower = {w.lower() for w in self.COMMON_ENGLISH_WORDS}
        self.short_genes_lower = {w.lower() for w in self.SHORT_GENES_NEED_CONTEXT}
        self.gene_context_lower = {w.lower() for w in self.GENE_CONTEXT_KEYWORDS}
        self.non_gene_context_lower = {w.lower() for w in self.NON_GENE_CONTEXT_KEYWORDS}
        self.questionnaire_lower = {w.lower() for w in self.QUESTIONNAIRE_TERMS}
        self.antibody_abbrev_lower = {w.lower() for w in self.ANTIBODY_ABBREVIATIONS}
        self.antibody_context_lower = {w.lower() for w in self.ANTIBODY_CONTEXT_KEYWORDS}

        # Combined always-filter set
        self.always_filter = (
            self.statistical_lower |
            self.units_lower |
            self.clinical_lower |
            self.countries_lower |
            self.credentials_lower |
            self.drug_terms_lower |
            self.study_terms_lower |
            self.common_english_lower
        )

        # Load disease lexicons for disambiguation
        self.disease_abbreviations: Dict[str, Dict[str, Any]] = {}
        if lexicon_base_path:
            self._load_disease_lexicons(lexicon_base_path)

    def _load_disease_lexicons(self, base_path: Path) -> None:
        """Load disease lexicons for gene-disease disambiguation."""
        lexicon_files = [
            "disease_lexicon_pah.json",
            "disease_lexicon_anca.json",
            "disease_lexicon_c3g.json",
            "disease_lexicon_igan.json",
        ]

        for filename in lexicon_files:
            lexicon_path = base_path / filename
            if not lexicon_path.exists():
                continue

            try:
                with open(lexicon_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract abbreviations and their context keywords
                if "abbreviation_expansions" in data:
                    for abbrev, info in data["abbreviation_expansions"].items():
                        abbrev_lower = abbrev.lower()
                        if abbrev_lower not in self.disease_abbreviations:
                            self.disease_abbreviations[abbrev_lower] = {
                                "preferred": info.get("preferred", ""),
                                "context_keywords": [],
                                "exclude_keywords": [],
                            }
                        # Add context keywords from alternatives
                        if "alternatives" in info:
                            for alt_name, alt_info in info.get("alternatives", {}).items():
                                # These are contexts where the abbrev is NOT the disease
                                self.disease_abbreviations[abbrev_lower]["exclude_keywords"].extend(
                                    alt_info.get("context_keywords", [])
                                )

                # Also get context_keywords from diseases
                if "diseases" in data:
                    for disease_key, disease_info in data["diseases"].items():
                        abbrev = disease_info.get("abbreviation", "")
                        if abbrev:
                            abbrev_lower = abbrev.lower()
                            if abbrev_lower not in self.disease_abbreviations:
                                self.disease_abbreviations[abbrev_lower] = {
                                    "preferred": disease_info.get("preferred_label", ""),
                                    "context_keywords": [],
                                    "exclude_keywords": [],
                                }
                            self.disease_abbreviations[abbrev_lower]["context_keywords"].extend(
                                disease_info.get("context_keywords", [])
                            )
            except Exception:
                pass  # Silently ignore malformed lexicons

    def _is_disease_abbreviation_context(self, abbrev: str, context: str) -> bool:
        """Check if abbreviation is used as a disease (not gene) in this context."""
        abbrev_lower = abbrev.lower()
        if abbrev_lower not in self.disease_abbreviations:
            return False

        info = self.disease_abbreviations[abbrev_lower]
        ctx_lower = context.lower()

        # Check if disease context keywords are present
        disease_keywords = info.get("context_keywords", [])
        disease_score = sum(1 for kw in disease_keywords if kw.lower() in ctx_lower)

        # Check if gene context keywords are present (from exclude list)
        gene_keywords = info.get("exclude_keywords", [])
        gene_score = sum(1 for kw in gene_keywords if kw.lower() in ctx_lower)

        # If disease context is present, it's being used as disease
        # Lower threshold: just 1 disease keyword is enough if no gene keywords
        return disease_score > gene_score or disease_score >= 1

    def is_false_positive(
        self,
        matched_text: str,
        context: str,
        generator_type: GeneGeneratorType,
        is_from_lexicon: bool = False,
        is_alias: bool = False,
    ) -> Tuple[bool, str]:
        """
        Check if a gene match is likely a false positive.

        Returns (is_fp, reason) tuple.
        """
        text_lower = matched_text.lower().strip()

        # Skip very short matches
        if len(text_lower) < self.MIN_LENGTH:
            return True, "too_short"

        # ALWAYS filter common English words - even from lexicon
        # These are problematic aliases that cause too many false positives
        if text_lower in self.common_english_lower:
            return True, "common_english_word"

        # Filter questionnaire abbreviations unless strong gene context
        if text_lower in self.questionnaire_lower:
            if not self._has_strong_gene_context(context):
                return True, "questionnaire_term"

        # Filter antibody abbreviations when antibody context is present
        if text_lower in self.antibody_abbrev_lower:
            if self._is_antibody_context(context):
                return True, "antibody_abbreviation"

        # Check other always-filter terms (unless from specialized lexicon)
        if not is_from_lexicon or is_alias:
            other_filters = (
                self.statistical_lower |
                self.units_lower |
                self.clinical_lower |
                self.countries_lower |
                self.credentials_lower |
                self.drug_terms_lower |
                self.study_terms_lower
            )
            if text_lower in other_filters:
                return True, "common_abbreviation"

        # For short gene symbols (2-3 chars), require context validation
        # Lexicon aliases (is_from_lexicon=True, is_alias=True) are curated HGNC
        # aliases so they're trusted — only validate if in SHORT_GENES_NEED_CONTEXT
        if len(text_lower) <= 3:
            needs_context = (
                text_lower in self.short_genes_lower
                or not is_from_lexicon
                or (is_alias and not is_from_lexicon)
            )
            if needs_context:
                is_valid, reason = self._validate_short_gene_context(text_lower, context)
                if not is_valid:
                    return True, reason

        # Special handling for EGFR - disambiguate gene vs kidney function
        if text_lower == "egfr":
            if self._is_kidney_egfr_context(context):
                return True, "egfr_kidney_function"

        # Disambiguate gene vs disease abbreviations using disease lexicons
        if text_lower in self.disease_abbreviations:
            if self._is_disease_abbreviation_context(text_lower, context):
                return True, f"disease_abbreviation_{text_lower}"

        # Context-based validation for pattern matches
        if generator_type == GeneGeneratorType.PATTERN_GENE_SYMBOL:
            gene_score, nongene_score = self._score_context(context)
            if nongene_score > gene_score:
                return True, "non_gene_context"
            if gene_score < 1:
                return True, "insufficient_gene_context"

        return False, ""

    def _validate_short_gene_context(self, gene: str, context: str) -> Tuple[bool, str]:
        """Validate short gene symbols require gene context."""
        ctx_lower = context.lower()

        # Count gene context keywords
        gene_score = sum(1 for kw in self.gene_context_lower if kw in ctx_lower)

        # Count non-gene context
        nongene_score = 0
        for phrase in self.non_gene_context_lower:
            if phrase in ctx_lower:
                nongene_score += 1

        # Short genes need at least 2 gene context keywords and more than non-gene
        if gene_score < 2:
            return False, "short_gene_no_context"
        if nongene_score >= gene_score:
            return False, "short_gene_statistical_context"

        return True, ""

    _KIDNEY_GFR_RE = re.compile(r"\b(?:e?gfr)\b", re.IGNORECASE)
    _GENE_SYMBOL_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,6}\b")

    def _is_kidney_egfr_context(self, context: str) -> bool:
        """Check if EGFR is being used as kidney function marker (eGFR)."""
        ctx_lower = context.lower()
        # Definitive kidney context markers (not ambiguous)
        kidney_keywords = [
            "ml/min", "renal", "kidney", "creatinine", "ckd",
            "glomerular", "filtration",
            "chronic kidney", "kidney disease", "renal function",
        ]
        if any(kw in ctx_lower for kw in kidney_keywords):
            return True
        # Check for standalone "gfr" or "egfr" not part of gene symbols
        # Remove all gene-like symbols (e.g. EGFR, FGFR2) before checking
        ctx_no_genes = self._GENE_SYMBOL_RE.sub("", context)
        return bool(self._KIDNEY_GFR_RE.search(ctx_no_genes))

    def _has_strong_gene_context(self, context: str) -> bool:
        """Check if context has strong gene evidence (2+ gene keywords)."""
        ctx_lower = context.lower()
        gene_score = sum(1 for kw in self.gene_context_lower if kw in ctx_lower)
        return gene_score >= 2

    def _is_antibody_context(self, context: str) -> bool:
        """Check if context indicates antibody usage (not gene)."""
        ctx_lower = context.lower()
        return any(kw in ctx_lower for kw in self.antibody_context_lower)

    def _score_context(self, context: str) -> Tuple[int, int]:
        """Score context for gene vs non-gene usage."""
        ctx_lower = context.lower()

        gene_score = sum(1 for kw in self.gene_context_lower if kw in ctx_lower)
        nongene_score = sum(1 for phrase in self.non_gene_context_lower if phrase in ctx_lower)

        return gene_score, nongene_score


__all__ = ["GeneFalsePositiveFilter"]
