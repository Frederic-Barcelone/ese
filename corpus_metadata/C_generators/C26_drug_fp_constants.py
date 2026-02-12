"""
Drug false positive filtering constant sets.

This module contains curated constant sets used by DrugFalsePositiveFilter to
identify common false positives in drug name detection. Includes bacteria,
biological entities, common words, and other non-drug terms.

Data is loaded from G_config/data/drug_fp_terms.yaml and drug_mappings.yaml.

Key Components:
    - BACTERIA_ORGANISMS: Bacteria and virus names (common in vaccine trials)
    - BIOLOGICAL_ENTITIES: Proteins, enzymes, cell types
    - BIOLOGICAL_SUFFIXES: Suffixes indicating non-drug biological terms
    - BODY_PARTS: Anatomical terms
    - COMMON_WORDS: Generic terms that aren't drugs
    - CREDENTIALS: Academic/medical credentials
    - DRUG_ABBREVIATIONS: Valid drug abbreviations to preserve
    - EQUIPMENT_PROCEDURES: Medical equipment and procedure names
    - FP_SUBSTRINGS: Substrings indicating false positives
    - NER_FALSE_POSITIVES: Terms causing NER false positives
    - NON_DRUG_ALLCAPS: All-caps terms that aren't drugs
    - ORGANIZATIONS: Organization names
    - PHARMA_COMPANY_NAMES: Company names (not drugs themselves)
    - TRIAL_STATUS_TERMS: Clinical trial status terminology
    - VACCINE_TERMS: Vaccine-related terms
    - ALWAYS_FILTER: Terms to always filter regardless of context

Example:
    >>> from C_generators.C26_drug_fp_constants import BACTERIA_ORGANISMS
    >>> "influenza" in BACTERIA_ORGANISMS
    True

Dependencies:
    - Z_utils.Z12_data_loader: YAML data loading
"""

from __future__ import annotations

from typing import Dict, List, Set

from Z_utils.Z12_data_loader import load_mapping, load_term_list, load_term_set
from Z_utils.Z15_lexicon_provider import CREDENTIALS

# Word lists (Set[str]) from drug_fp_terms.yaml
BACTERIA_ORGANISMS: Set[str] = load_term_set("drug_fp_terms.yaml", "bacteria_organisms")
VACCINE_TERMS: Set[str] = load_term_set("drug_fp_terms.yaml", "vaccine_terms")
BIOLOGICAL_ENTITIES: Set[str] = load_term_set("drug_fp_terms.yaml", "biological_entities")
PHARMA_COMPANY_NAMES: Set[str] = load_term_set("drug_fp_terms.yaml", "pharma_company_names")
COMMON_WORDS: Set[str] = load_term_set("drug_fp_terms.yaml", "common_words")
ORGANIZATIONS: Set[str] = load_term_set("drug_fp_terms.yaml", "organizations")
BODY_PARTS: Set[str] = load_term_set("drug_fp_terms.yaml", "body_parts")
TRIAL_STATUS_TERMS: Set[str] = load_term_set("drug_fp_terms.yaml", "trial_status_terms")
EQUIPMENT_PROCEDURES: Set[str] = load_term_set("drug_fp_terms.yaml", "equipment_procedures")
ALWAYS_FILTER: Set[str] = load_term_set("drug_fp_terms.yaml", "always_filter")
NER_FALSE_POSITIVES: Set[str] = load_term_set("drug_fp_terms.yaml", "ner_false_positives")
NON_DRUG_ALLCAPS: Set[str] = load_term_set("drug_fp_terms.yaml", "non_drug_allcaps")

# Ordered lists (List[str]) from drug_fp_terms.yaml
BIOLOGICAL_SUFFIXES: List[str] = load_term_list("drug_fp_terms.yaml", "biological_suffixes")
FP_SUBSTRINGS: List[str] = load_term_list("drug_fp_terms.yaml", "fp_substrings")
CONSUMER_DRUG_PATTERNS: List[str] = load_term_list("drug_fp_terms.yaml", "consumer_drug_patterns")

# Mappings (Dict[str, str]) from drug_mappings.yaml
DRUG_ABBREVIATIONS: Dict[str, str] = load_mapping("drug_mappings.yaml", "drug_abbreviations")
CONSUMER_DRUG_VARIANTS: Dict[str, str] = load_mapping("drug_mappings.yaml", "consumer_drug_variants")


__all__ = [
    "BACTERIA_ORGANISMS",
    "VACCINE_TERMS",
    "CREDENTIALS",
    "BIOLOGICAL_ENTITIES",
    "PHARMA_COMPANY_NAMES",
    "COMMON_WORDS",
    "ORGANIZATIONS",
    "BODY_PARTS",
    "TRIAL_STATUS_TERMS",
    "EQUIPMENT_PROCEDURES",
    "ALWAYS_FILTER",
    "NER_FALSE_POSITIVES",
    "NON_DRUG_ALLCAPS",
    "BIOLOGICAL_SUFFIXES",
    "FP_SUBSTRINGS",
    "DRUG_ABBREVIATIONS",
    "CONSUMER_DRUG_VARIANTS",
    "CONSUMER_DRUG_PATTERNS",
]
