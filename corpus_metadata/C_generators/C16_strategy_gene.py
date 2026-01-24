# corpus_metadata/C_generators/C16_strategy_gene.py
"""
Gene/protein entity detection strategy for rare diseases.

Multi-layered approach:
1. Orphadata genes (rare disease-associated, highest priority)
2. HGNC aliases (official gene nomenclature)
3. Gene symbol patterns with context validation
4. scispacy NER (GENE semantic type, fallback)

Uses FlashText for fast keyword matching.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from flashtext import KeywordProcessor

from A_core.A01_domain_models import Coordinate
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from A_core.A12_gene_models import (
    GeneCandidate,
    GeneDiseaseLinkage,
    GeneFieldType,
    GeneGeneratorType,
    GeneIdentifier,
    GeneProvenanceMetadata,
)
from B_parsing.B01_pdf_to_docgraph import DocumentGraph
from B_parsing.B06_confidence import ConfidenceCalculator

# Optional scispacy import
try:
    import spacy
    from scispacy.linking import EntityLinker  # noqa: F401

    SCISPACY_AVAILABLE = True
except ImportError:
    spacy = None  # type: ignore[assignment]
    SCISPACY_AVAILABLE = False


# -------------------------
# Gene False Positive Filter
# -------------------------


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
        text_stripped = matched_text.strip()

        # Skip very short matches
        if len(text_lower) < self.MIN_LENGTH:
            return True, "too_short"

        # ALWAYS filter common English words - even from lexicon
        # These are problematic aliases that cause too many false positives
        if text_lower in self.common_english_lower:
            return True, "common_english_word"

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

    def _score_context(self, context: str) -> Tuple[int, int]:
        """Score context for gene vs non-gene usage."""
        ctx_lower = context.lower()

        gene_score = sum(1 for kw in self.gene_context_lower if kw in ctx_lower)
        nongene_score = sum(1 for phrase in self.non_gene_context_lower if phrase in ctx_lower)

        return gene_score, nongene_score


# -------------------------
# Gene Detector
# -------------------------


class GeneDetector:
    """
    Multi-layered gene mention detection for rare diseases.

    Layers (in priority order):
    1. Orphadata genes (rare disease-associated)
    2. HGNC aliases
    3. Gene symbol patterns with context validation
    4. scispacy NER (fallback)
    """

    # UMLS semantic types for genes
    GENE_SEMANTIC_TYPES = {
        "T028",  # Gene or Genome
        "T116",  # Amino Acid, Peptide, or Protein
        "T126",  # Enzyme
    }

    # Gene symbol pattern: 1-6 uppercase letters/numbers, starting with letter
    GENE_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,6})\b")

    # Terms to skip when loading lexicons
    LEXICON_LOAD_BLACKLIST: Set[str] = {
        # Statistical terms
        "or", "hr", "ci", "sd", "se", "rr", "mr", "md", "ns", "na", "nd",
        # Units
        "mm", "cm", "kg", "mg", "ml", "dl", "ng", "pg", "hz", "kd", "da",
        # Clinical
        "iv", "po", "im", "sc", "bid", "tid", "qd", "prn", "er", "icu", "ed",
        # Countries
        "us", "uk", "eu", "ca", "au", "de", "fr", "jp", "cn",
        # Credentials
        "md", "phd", "mph", "do", "rn", "ms", "ma", "mba",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.run_id = str(self.config.get("run_id") or generate_run_id("GENE"))
        self.pipeline_version = (
            self.config.get("pipeline_version") or get_git_revision_hash()
        )
        self.doc_fingerprint_default = (
            self.config.get("doc_fingerprint") or "unknown-doc-fingerprint"
        )

        # Context window for evidence extraction
        self.context_window = int(self.config.get("context_window", 600))

        # Confidence calculator
        self.confidence_calculator = ConfidenceCalculator()

        # Lexicon base path
        self.lexicon_base_path = Path(
            self.config.get("lexicon_base_path", "ouput_datasources")
        )

        # Initialize FlashText processors
        self.orphadata_processor: Optional[KeywordProcessor] = None
        self.alias_processor: Optional[KeywordProcessor] = None

        # Gene metadata dictionaries
        self.orphadata_genes: Dict[str, Dict] = {}
        self.alias_genes: Dict[str, Dict] = {}

        # Lexicon loading stats
        self._lexicon_stats: List[Tuple[str, int, str]] = []

        # Load lexicons
        self._load_lexicons()

        # False positive filter with disease lexicon disambiguation
        self.fp_filter = GeneFalsePositiveFilter(lexicon_base_path=self.lexicon_base_path)

        # scispacy NER model
        self.nlp = None
        if SCISPACY_AVAILABLE:
            self._init_scispacy()

    def _load_lexicons(self) -> None:
        """Load gene lexicons."""
        self._load_orphadata_lexicon()

    def _load_orphadata_lexicon(self) -> None:
        """Load Orphadata gene lexicon (rare disease genes + HGNC aliases)."""
        path = self.lexicon_base_path / "2025_08_orphadata_genes.json"
        if not path.exists():
            print(f"[WARN] Orphadata gene lexicon not found: {path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            orphadata_proc = KeywordProcessor(case_sensitive=False)
            alias_proc = KeywordProcessor(case_sensitive=False)

            self.orphadata_processor = orphadata_proc
            self.alias_processor = alias_proc

            primary_count = 0
            alias_count = 0
            skipped = 0

            for entry in data:
                term = entry.get("term", "").strip()
                if not term or len(term) < 2:
                    continue

                term_key = term.lower()

                # Skip blacklisted terms
                if term_key in self.LEXICON_LOAD_BLACKLIST:
                    skipped += 1
                    continue

                source = entry.get("source", "")

                if source == "orphadata_hgnc":
                    # Primary gene entry
                    self.orphadata_genes[term_key] = {
                        "symbol": entry.get("hgnc_symbol", term),
                        "full_name": entry.get("full_name"),
                        "hgnc_id": entry.get("hgnc_id"),
                        "entrez_id": entry.get("entrez_id"),
                        "ensembl_id": entry.get("ensembl_id"),
                        "omim_id": entry.get("omim_id"),
                        "uniprot_id": entry.get("uniprot_id"),
                        "locus_type": entry.get("locus_type"),
                        "associated_diseases": entry.get("associated_diseases", []),
                        "source": "orphadata",
                    }
                    orphadata_proc.add_keyword(term, term_key)
                    primary_count += 1

                elif source == "hgnc_alias":
                    # Alias entry
                    canonical = entry.get("is_alias_of", entry.get("hgnc_symbol", term))
                    self.alias_genes[term_key] = {
                        "symbol": canonical,
                        "alias_term": term,
                        "full_name": entry.get("full_name"),
                        "hgnc_id": entry.get("hgnc_id"),
                        "entrez_id": entry.get("entrez_id"),
                        "ensembl_id": entry.get("ensembl_id"),
                        "omim_id": entry.get("omim_id"),
                        "uniprot_id": entry.get("uniprot_id"),
                        "locus_type": entry.get("locus_type"),
                        "source": "hgnc_alias",
                    }
                    alias_proc.add_keyword(term, term_key)
                    alias_count += 1

            if skipped > 0:
                print(f"    [INFO] Skipped {skipped} blacklisted gene terms")

            self._lexicon_stats.append(
                ("Orphadata genes", primary_count, "2025_08_orphadata_genes.json")
            )
            self._lexicon_stats.append(
                ("HGNC aliases", alias_count, "2025_08_orphadata_genes.json")
            )

        except Exception as e:
            print(f"[WARN] Failed to load Orphadata gene lexicon: {e}")

    def _init_scispacy(self) -> None:
        """Initialize scispacy NER model."""
        if not SCISPACY_AVAILABLE or spacy is None:
            return

        try:
            try:
                self.nlp = spacy.load("en_core_sci_lg")
            except OSError:
                self.nlp = spacy.load("en_core_sci_sm")

            if "scispacy_linker" not in self.nlp.pipe_names:
                self.nlp.add_pipe(
                    "scispacy_linker",
                    config={
                        "resolve_abbreviations": True,
                        "linker_name": "umls",
                        "threshold": 0.7,
                    },
                )
            self._lexicon_stats.append(("scispacy NER", 1, "en_core_sci_lg"))

        except Exception as e:
            print(f"[WARN] Failed to initialize scispacy for genes: {e}")
            self.nlp = None

    def _print_lexicon_summary(self) -> None:
        """Print compact summary of loaded gene lexicons."""
        if not self._lexicon_stats:
            return

        total = sum(count for _, count, _ in self._lexicon_stats if count > 1)
        print(f"\nGene lexicons: {len(self._lexicon_stats)} sources, {total:,} entries")
        print("─" * 70)

        for name, count, filename in self._lexicon_stats:
            if count > 1:
                print(f"    • {name:<26} {count:>8,}  {filename}")
            else:
                print(f"    • {name:<26} {'enabled':>8}  {filename}")
        print()

    def detect(self, doc_graph: DocumentGraph) -> List[GeneCandidate]:
        """
        Detect gene mentions in document.

        Returns list of GeneCandidate objects.
        """
        candidates: List[GeneCandidate] = []
        doc_fingerprint = getattr(
            doc_graph, "fingerprint", self.doc_fingerprint_default
        )

        # Get full text for detection
        full_text = "\n\n".join(
            block.text for block in doc_graph.iter_linear_blocks(skip_header_footer=True)
            if block.text
        )

        # Layer 1: Orphadata genes (rare disease-associated)
        if self.orphadata_processor:
            candidates.extend(
                self._detect_with_lexicon(
                    full_text,
                    doc_graph,
                    doc_fingerprint,
                    self.orphadata_processor,
                    self.orphadata_genes,
                    GeneGeneratorType.LEXICON_ORPHADATA,
                    "2025_08_orphadata_genes.json",
                    is_primary=True,
                )
            )

        # Layer 2: HGNC aliases
        if self.alias_processor:
            candidates.extend(
                self._detect_with_lexicon(
                    full_text,
                    doc_graph,
                    doc_fingerprint,
                    self.alias_processor,
                    self.alias_genes,
                    GeneGeneratorType.LEXICON_HGNC_ALIAS,
                    "2025_08_orphadata_genes.json",
                    is_primary=False,
                )
            )

        # Layer 3: Gene symbol patterns with context validation
        # DISABLED: Causes too many false positives - the lexicon coverage is sufficient
        # for rare disease genes. Uncomment if you need pattern-based detection.
        # candidates.extend(
        #     self._detect_gene_patterns(full_text, doc_graph, doc_fingerprint)
        # )

        # Layer 4: scispacy NER fallback
        # DISABLED: Causes too many false positives without proper validation.
        # Uncomment if you need NER-based detection.
        # if self.nlp:
        #     candidates.extend(
        #         self._detect_with_ner(full_text, doc_graph, doc_fingerprint)
        #     )

        # Deduplicate
        candidates = self._deduplicate(candidates)

        return candidates

    def _detect_with_lexicon(
        self,
        text: str,
        doc_graph: DocumentGraph,
        doc_fingerprint: str,
        processor: KeywordProcessor,
        gene_dict: Dict[str, Dict],
        generator_type: GeneGeneratorType,
        lexicon_source: str,
        is_primary: bool = True,
    ) -> List[GeneCandidate]:
        """Detect genes using FlashText lexicon matching."""
        candidates = []

        matches = processor.extract_keywords(text, span_info=True)

        for keyword, start, end in matches:
            gene_info = gene_dict.get(keyword, {})
            if not gene_info:
                continue

            matched_text = text[start:end]
            context = self._extract_context(text, start, end)

            # Apply false positive filter
            # Aliases need stricter filtering than primary genes
            is_fp, reason = self.fp_filter.is_false_positive(
                matched_text, context, generator_type,
                is_from_lexicon=True, is_alias=(not is_primary)
            )
            if is_fp:
                continue

            # Build identifiers
            identifiers = self._build_identifiers(gene_info)

            # Build disease linkages
            disease_linkages = []
            for disease in gene_info.get("associated_diseases", []):
                disease_linkages.append(
                    GeneDiseaseLinkage(
                        orphacode=str(disease.get("orphacode", "")),
                        disease_name=disease.get("name", ""),
                        association_type=disease.get("association_type"),
                        association_status=disease.get("association_status"),
                    )
                )

            # Determine confidence
            if generator_type == GeneGeneratorType.LEXICON_ORPHADATA:
                confidence = 0.85
            else:
                confidence = 0.80

            # Build provenance
            provenance = GeneProvenanceMetadata(
                pipeline_version=self.pipeline_version,
                run_id=self.run_id,
                doc_fingerprint=doc_fingerprint,
                generator_name=generator_type,
                lexicon_source=lexicon_source,
                lexicon_ids=identifiers,
            )

            candidate = GeneCandidate(
                doc_id=doc_graph.doc_id,
                matched_text=matched_text,
                hgnc_symbol=gene_info.get("symbol", matched_text),
                full_name=gene_info.get("full_name"),
                is_alias=not is_primary,
                alias_of=gene_info.get("symbol") if not is_primary else None,
                field_type=GeneFieldType.EXACT_MATCH,
                generator_type=generator_type,
                identifiers=identifiers,
                context_text=context,
                context_location=Coordinate(page_num=1),  # TODO: get actual page
                locus_type=gene_info.get("locus_type"),
                associated_diseases=disease_linkages,
                initial_confidence=confidence,
                provenance=provenance,
            )
            candidates.append(candidate)

        return candidates

    def _detect_gene_patterns(
        self,
        text: str,
        doc_graph: DocumentGraph,
        doc_fingerprint: str,
    ) -> List[GeneCandidate]:
        """Detect potential gene symbols using pattern matching with context validation."""
        candidates = []

        # Skip if we already have good lexicon coverage
        # Pattern matching is only for genes NOT in our lexicon

        for match in self.GENE_PATTERN.finditer(text):
            matched_text = match.group(1)
            start, end = match.span(1)

            # Skip if already in our lexicon
            if matched_text.lower() in self.orphadata_genes:
                continue
            if matched_text.lower() in self.alias_genes:
                continue

            context = self._extract_context(text, start, end)

            # Apply strict false positive filter for patterns
            is_fp, reason = self.fp_filter.is_false_positive(
                matched_text, context, GeneGeneratorType.PATTERN_GENE_SYMBOL, is_from_lexicon=False
            )
            if is_fp:
                continue

            # Build provenance
            provenance = GeneProvenanceMetadata(
                pipeline_version=self.pipeline_version,
                run_id=self.run_id,
                doc_fingerprint=doc_fingerprint,
                generator_name=GeneGeneratorType.PATTERN_GENE_SYMBOL,
            )

            candidate = GeneCandidate(
                doc_id=doc_graph.doc_id,
                matched_text=matched_text,
                hgnc_symbol=matched_text,  # Pattern match - use as-is
                field_type=GeneFieldType.PATTERN_MATCH,
                generator_type=GeneGeneratorType.PATTERN_GENE_SYMBOL,
                identifiers=[],
                context_text=context,
                context_location=Coordinate(page_num=1),
                initial_confidence=0.75,
                provenance=provenance,
            )
            candidates.append(candidate)

        return candidates

    def _detect_with_ner(
        self,
        text: str,
        doc_graph: DocumentGraph,
        doc_fingerprint: str,
    ) -> List[GeneCandidate]:
        """Detect genes using scispacy NER as fallback."""
        candidates = []

        if not self.nlp:
            return candidates

        try:
            # Process with scispacy (limit text length for performance)
            max_len = 100000
            if len(text) > max_len:
                text = text[:max_len]

            doc = self.nlp(text)

            for ent in doc.ents:
                # Check UMLS linking
                if not hasattr(ent._, "kb_ents") or not ent._.kb_ents:
                    continue

                # Get best UMLS match
                best_cui, best_score = ent._.kb_ents[0]

                # Check semantic type
                linker = self.nlp.get_pipe("scispacy_linker")
                kb = linker.kb
                if best_cui not in kb.cui_to_entity:
                    continue

                entity = kb.cui_to_entity[best_cui]
                types = entity.types if hasattr(entity, "types") else []

                # Check if it's a gene semantic type
                is_gene = any(t in self.GENE_SEMANTIC_TYPES for t in types)
                if not is_gene:
                    continue

                matched_text = ent.text
                start, end = ent.start_char, ent.end_char

                # Skip if already in lexicon
                if matched_text.lower() in self.orphadata_genes:
                    continue
                if matched_text.lower() in self.alias_genes:
                    continue

                context = self._extract_context(text, start, end)

                # Apply false positive filter
                is_fp, reason = self.fp_filter.is_false_positive(
                    matched_text, context, GeneGeneratorType.SCISPACY_NER, is_from_lexicon=False
                )
                if is_fp:
                    continue

                # Build identifiers
                identifiers = [
                    GeneIdentifier(system="UMLS_CUI", code=best_cui)
                ]

                provenance = GeneProvenanceMetadata(
                    pipeline_version=self.pipeline_version,
                    run_id=self.run_id,
                    doc_fingerprint=doc_fingerprint,
                    generator_name=GeneGeneratorType.SCISPACY_NER,
                )

                candidate = GeneCandidate(
                    doc_id=doc_graph.doc_id,
                    matched_text=matched_text,
                    hgnc_symbol=matched_text,  # NER - use as-is
                    field_type=GeneFieldType.NER_DETECTION,
                    generator_type=GeneGeneratorType.SCISPACY_NER,
                    identifiers=identifiers,
                    context_text=context,
                    context_location=Coordinate(page_num=1),
                    initial_confidence=min(best_score, 0.70),
                    provenance=provenance,
                )
                candidates.append(candidate)

        except Exception as e:
            print(f"[WARN] scispacy NER failed: {e}")

        return candidates

    def _extract_context(self, text: str, start: int, end: int) -> str:
        """Extract context window around match."""
        ctx_start = max(0, start - self.context_window // 2)
        ctx_end = min(len(text), end + self.context_window // 2)
        return text[ctx_start:ctx_end]

    def _build_identifiers(self, gene_info: Dict) -> List[GeneIdentifier]:
        """Build list of gene identifiers from metadata."""
        identifiers = []

        if gene_info.get("hgnc_id"):
            identifiers.append(
                GeneIdentifier(system="HGNC", code=gene_info["hgnc_id"])
            )
        if gene_info.get("entrez_id"):
            identifiers.append(
                GeneIdentifier(system="ENTREZ", code=str(gene_info["entrez_id"]))
            )
        if gene_info.get("ensembl_id"):
            identifiers.append(
                GeneIdentifier(system="ENSEMBL", code=gene_info["ensembl_id"])
            )
        if gene_info.get("omim_id"):
            identifiers.append(
                GeneIdentifier(system="OMIM", code=gene_info["omim_id"])
            )
        if gene_info.get("uniprot_id"):
            identifiers.append(
                GeneIdentifier(system="UNIPROT", code=gene_info["uniprot_id"])
            )

        return identifiers

    def _deduplicate(self, candidates: List[GeneCandidate]) -> List[GeneCandidate]:
        """Deduplicate candidates, preferring higher priority sources."""
        # Priority: ORPHADATA > HGNC_ALIAS > PATTERN > NER
        priority = {
            GeneGeneratorType.LEXICON_ORPHADATA: 0,
            GeneGeneratorType.LEXICON_HGNC_ALIAS: 1,
            GeneGeneratorType.PATTERN_GENE_SYMBOL: 2,
            GeneGeneratorType.SCISPACY_NER: 3,
        }

        seen: Dict[str, GeneCandidate] = {}

        for candidate in candidates:
            key = candidate.matched_text.lower()

            if key not in seen:
                seen[key] = candidate
            else:
                existing = seen[key]
                existing_priority = priority.get(existing.generator_type, 99)
                new_priority = priority.get(candidate.generator_type, 99)

                if new_priority < existing_priority:
                    seen[key] = candidate
                elif new_priority == existing_priority:
                    if candidate.initial_confidence > existing.initial_confidence:
                        seen[key] = candidate

        return list(seen.values())
