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
    - json, pathlib: For loading external gene data
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from A_core.A19_gene_models import GeneGeneratorType


class GeneFalsePositiveFilter:
    """
    Filter false positive gene matches.

    Gene symbols are highly ambiguous - many clash with common abbreviations,
    statistics terms, units, and other non-gene entities.
    """

    MIN_LENGTH = 2  # Minimum gene symbol length

    # Statistical terms that look like gene symbols
    STATISTICAL_TERMS: Set[str] = {
        "or",   # Odds ratio (also olfactory receptor genes)
        "hr",   # Hazard ratio
        "ci",   # Confidence interval
        "sd",   # Standard deviation
        "se",   # Standard error
        "rr",   # Relative risk
        "mr",   # Mendelian randomization / MR imaging
        "md",   # Mean difference
        "smd",  # Standardized mean difference
        "bmi",  # Body mass index
        "auc",  # Area under curve
        "roc",  # ROC curve
        "icc",  # Intraclass correlation
        "cv",   # Coefficient of variation
        "iqr",  # Interquartile range
        "hr",   # Heart rate (also hazard ratio)
        "bp",   # Blood pressure / base pairs
        "ns",   # Not significant
        "na",   # Not available / Not applicable
        "nd",   # Not detected
        "vs",   # Versus
    }

    # Units and measurements
    UNITS: Set[str] = {
        "mm",   # Millimeters
        "cm",   # Centimeters
        "kg",   # Kilograms
        "mg",   # Milligrams
        "ml",   # Milliliters
        "dl",   # Deciliters
        "ul",   # Microliters
        "ng",   # Nanograms
        "pg",   # Picograms
        "hz",   # Hertz
        "kd",   # Kilodaltons
        "da",   # Daltons
        "ph",   # pH
        "min",  # Minutes
        "sec",  # Seconds
        "hr",   # Hours
        "mo",   # Months
        "yr",   # Years
        "wk",   # Weeks
    }

    # Medical/clinical abbreviations
    CLINICAL_TERMS: Set[str] = {
        "iv",   # Intravenous
        "po",   # Per os (oral)
        "im",   # Intramuscular
        "sc",   # Subcutaneous
        "bid",  # Twice daily
        "tid",  # Three times daily
        "qd",   # Once daily
        "prn",  # As needed
        "er",   # Emergency room / Extended release
        "icu",  # Intensive care unit
        "ed",   # Emergency department
        "or",   # Operating room
        "ct",   # CT scan
        "mri",  # MRI
        "ecg",  # Electrocardiogram
        "ekg",  # Electrocardiogram
        "eeg",  # Electroencephalogram
        "gfr",  # Glomerular filtration rate
        "egfr", # eGFR (kidney function) - conflicts with EGFR gene
        "hba1c",
        "ldl",  # LDL cholesterol
        "hdl",  # HDL cholesterol
        "ast",  # Liver enzyme
        "alt",  # Liver enzyme
        "bnp",  # B-type natriuretic peptide
        "crp",  # C-reactive protein
        "esr",  # Erythrocyte sedimentation rate
        "wbc",  # White blood cells
        "rbc",  # Red blood cells
        "hgb",  # Hemoglobin
        "hct",  # Hematocrit
        "plt",  # Platelets
        "inr",  # International normalized ratio
        "ptt",  # Partial thromboplastin time
        "pt",   # Prothrombin time / Physical therapy
    }

    # Countries and regions
    COUNTRIES: Set[str] = {
        "us", "uk", "eu", "ca", "au", "de", "fr", "jp", "cn", "in",
        "it", "es", "nl", "be", "ch", "at", "se", "no", "dk", "fi",
        "pl", "cz", "hu", "ro", "bg", "gr", "pt", "ie", "nz", "sg",
        "hk", "tw", "kr", "mx", "br", "ar", "cl", "co", "za", "eg",
    }

    # Credentials and titles
    CREDENTIALS: Set[str] = {
        "md", "phd", "mph", "do", "rn", "np", "pa", "pharmd",
        "dds", "dmd", "dpt", "od", "dvm", "dc", "ms", "ma",
        "msc", "bsc", "ba", "mba", "jd", "llm",
    }

    # Drug-related terms that might look like genes
    DRUG_TERMS: Set[str] = {
        "ace",  # ACE inhibitors (also ACE gene)
        "arb",  # ARB drugs
        "nsaid",
        "ssri",
        "snri",
        "maoi",
        "ppi",  # Proton pump inhibitors
        "h2",   # H2 blockers
        "bb",   # Beta blockers
        "ccb",  # Calcium channel blockers
    }

    # Trial and study terms
    STUDY_TERMS: Set[str] = {
        "rct",  # Randomized controlled trial
        "itt",  # Intention to treat
        "pp",   # Per protocol
        "sae",  # Serious adverse event
        "ae",   # Adverse event
        "dmc",  # Data monitoring committee
        "dsmb", # Data safety monitoring board
        "irb",  # Institutional review board
        "fda",  # Food and Drug Administration
        "ema",  # European Medicines Agency
        "ich",  # International Council for Harmonisation
        "gcp",  # Good Clinical Practice
    }

    # Common English words that happen to be gene aliases
    # These should ALWAYS be filtered unless there's strong gene context
    COMMON_ENGLISH_WORDS: Set[str] = {
        # Articles, prepositions, conjunctions
        "of", "an", "as", "on", "at", "by", "to", "in", "is", "it",
        "be", "we", "me", "he", "or", "so", "do", "go", "no", "up",
        # Common verbs/nouns
        "was", "set", "can", "not", "for", "had", "has", "get", "let",
        "put", "run", "use", "see", "may", "day", "way", "say", "new",
        "now", "old", "man", "men", "one", "two", "few", "all", "any",
        "end", "big", "bad", "red", "hot", "cut", "hit", "bit", "fit",
        "sit", "got", "put", "yet", "met", "net", "wet", "bet", "pet",
        # Common words that are gene aliases
        "large", "small", "long", "short", "high", "low", "fast", "slow",
        "simple", "fix", "max", "min", "med", "per", "pre", "pro",
        "cox", "age", "lar", "van", "lee", "kim", "li", "wu", "liu",
        "wang", "chen", "yang", "zhang", "lin", "sun", "ma",
        # Abbreviations commonly mistaken
        "ge", "et", "ds", "wr", "uk", "us",
        # Additional common words that are gene aliases
        "son", "best", "ren", "rest", "last", "most", "near", "well", "good",
        "part", "step", "mark", "ring", "pair", "map", "gap", "cap", "tip",
        "bar", "tag", "tan", "dim",
        # Section headers and document structure
        "methods", "results", "discussion", "conclusion", "abstract",
        "introduction", "background", "references", "table", "figure",
    }

    # Context keywords that suggest gene usage
    GENE_CONTEXT_KEYWORDS: Set[str] = {
        "gene", "genes", "genetic", "genomic", "genome",
        "mutation", "mutations", "mutant", "mutated",
        "variant", "variants", "variation", "polymorphism", "snp", "snps",
        "allele", "alleles", "allelic",
        "genotype", "genotypes", "genotyping",
        "expression", "expressed", "overexpression", "downregulation",
        "mrna", "transcript", "transcription", "transcriptional",
        "protein", "proteins", "polypeptide",
        "heterozygous", "homozygous", "carrier", "carriers",
        "pathogenic", "benign", "vus", "likely pathogenic",
        "exon", "exons", "intron", "introns",
        "locus", "loci", "chromosome", "chromosomal",
        "knockout", "knockdown", "transgenic",
        "splicing", "splice", "frameshift", "missense", "nonsense",
        "deletion", "insertion", "duplication",
        "hgnc", "entrez", "ensembl", "omim",
        "encode", "encodes", "encoding",
    }

    # Context keywords that suggest non-gene usage
    NON_GENE_CONTEXT_KEYWORDS: Set[str] = {
        "odds ratio", "hazard ratio", "confidence interval",
        "p-value", "p value", "p =", "p<", "p >",
        "statistically", "significant", "significance",
        "administered", "dosage", "dose", "doses", "dosing",
        "mg/kg", "mg/day", "mg/ml",
        "intravenous", "subcutaneous", "intramuscular", "oral",
        "treatment arm", "placebo", "control group",
        "mmhg", "mm hg", "beats per minute", "bpm",
        "ml/min", "l/min", "kg/m2",
        "median", "mean", "average", "range",
    }

    # Clinical questionnaire/instrument abbreviations confused with genes
    QUESTIONNAIRE_TERMS: Set[str] = {
        "maf",   # Multidimensional Assessment of Fatigue
        "haq",   # Health Assessment Questionnaire
        "das",   # Disease Activity Score
        "bdi",   # Beck Depression Inventory
        "gad",   # Generalized Anxiety Disorder scale
        "phq",   # Patient Health Questionnaire
    }

    # Antibody abbreviations confused with genes
    ANTIBODY_ABBREVIATIONS: Set[str] = {
        "acpa",  # Anti-Citrullinated Protein Antibody
        "ana",   # Antinuclear Antibody
        "asca",  # Anti-Saccharomyces cerevisiae Antibodies
    }

    # Context keywords indicating antibody usage (not gene)
    ANTIBODY_CONTEXT_KEYWORDS: Set[str] = {
        "antibod", "titer", "seropositive", "seronegative",
        "autoantibod", "immunoassay", "elisa", "positivity",
        "serolog", "reactiv",
    }

    # Short genes that ALWAYS need context validation (2-3 chars)
    SHORT_GENES_NEED_CONTEXT: Set[str] = {
        "ar", "vr", "hr", "mr", "or", "nr", "er", "pr", "gr",
        "ca", "cb", "cd", "ce", "cf", "cg", "ch", "ci", "ck", "cl", "cm", "cn", "co", "cp", "cr", "cs", "ct", "cu", "cv", "cx", "cy",
        "il", "in", "ir", "is", "it",
        "no", "np", "nr", "ns", "nt",
        "pa", "pb", "pc", "pd", "pe", "pf", "pg", "ph", "pi", "pk", "pl", "pm", "pn", "po", "pp", "pr", "ps", "pt", "pu", "pv", "px", "py",
        "ra", "rb", "rc", "rd", "re", "rf", "rg", "rh", "ri", "rn", "ro", "rp", "rr", "rs", "rt", "ru", "rv", "rx", "ry",
        "sa", "sb", "sc", "sd", "se", "sf", "sg", "sh", "si", "sk", "sl", "sm", "sn", "so", "sp", "sr", "ss", "st", "su", "sv", "sx", "sy",
        "ta", "tb", "tc", "td", "te", "tf", "tg", "th", "ti", "tk", "tl", "tm", "tn", "to", "tp", "tr", "ts", "tt", "tu", "tv", "tx", "ty",
    }

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
        if len(text_lower) <= 3:
            if text_lower in self.short_genes_lower or not is_from_lexicon or is_alias:
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

    def _is_kidney_egfr_context(self, context: str) -> bool:
        """Check if EGFR is being used as kidney function marker (eGFR)."""
        ctx_lower = context.lower()
        kidney_keywords = [
            "ml/min", "renal", "kidney", "creatinine", "ckd",
            "gfr", "glomerular", "filtration", "stage",
            "chronic kidney", "kidney disease", "renal function",
        ]
        return any(kw in ctx_lower for kw in kidney_keywords)

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
