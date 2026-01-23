# corpus_metadata/corpus_metadata/orchestrator.py
"""
Orchestrator v0.8: PDF -> Parse -> Generate -> Validate -> Log

All configuration is loaded from config.yaml - no hardcoded parameters.

Usage:
    python orchestrator.py

Requires:
    - ANTHROPIC_API_KEY in .env file (or set in config.yaml)
    - pip install anthropic pyyaml python-dotenv
"""

from __future__ import annotations

# =============================================================================
# SILENCE WARNINGS FIRST (must be before library imports)
# =============================================================================
import os
import warnings

from dotenv import load_dotenv

load_dotenv()  # Load .env file from current directory or parent

# Suppress at environment level for subprocesses
os.environ["PYTHONWARNINGS"] = (
    "ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

_WARNING_FILTERS = [
    (UserWarning, r".*W036.*"),
    (UserWarning, r".*matcher.*does not have any patterns.*"),
    (UserWarning, r".*InconsistentVersionWarning.*"),
    (UserWarning, r".*max_size.*deprecated.*"),
    (FutureWarning, r".*max_size.*deprecated.*"),
    (DeprecationWarning, r".*max_size.*"),
]

for _cat, _pat in _WARNING_FILTERS:
    warnings.filterwarnings("ignore", category=_cat, message=_pat)

warnings.filterwarnings("ignore", module=r"sklearn\.base")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"spacy\.language")
warnings.filterwarnings("ignore", category=UserWarning, module=r"transformers")
# =============================================================================

import heapq
import json
import re
import sys
import base64
import time
import uuid
from collections import Counter
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
# =============================================================================

# Ensure imports work
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Configure logging first (before slow imports)
from A_core.A00_logging import (
    configure_logging,
    get_logger,
    INFO,
)

# Initialize module logger
logger = get_logger(__name__)

# Print startup message before slow imports (scispacy takes ~30s to load)
print("Starting pipeline... (loading NLP models, this may take a moment)")

from A_core.A01_domain_models import (
    Candidate,
    Coordinate,
    EvidenceSpan,
    ExtractedEntity,
    FieldType,
    GeneratorType,
    ProvenanceMetadata,
    ValidationStatus,
)
from A_core.A03_provenance import generate_run_id, hash_string
from A_core.A04_heuristics_config import (
    HeuristicsConfig,
    HeuristicsCounters,
    check_context_match,
    check_trial_id,
)
from A_core.A05_disease_models import (
    DiseaseCandidate,
    DiseaseExportDocument,
    DiseaseExportEntry,
    ExtractedDisease,
)
from A_core.A06_drug_models import (
    DrugCandidate,
    DrugExportDocument,
    DrugExportEntry,
    DrugGeneratorType,
    ExtractedDrug,
)
from A_core.A09_pharma_models import (
    ExtractedPharma,
    PharmaExportDocument,
    PharmaExportEntry,
)
from A_core.A10_author_models import (
    AuthorExportDocument,
    AuthorExportEntry,
    ExtractedAuthor,
)
from A_core.A11_citation_models import (
    CitationExportDocument,
    CitationExportEntry,
    ExtractedCitation,
)
from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser
from B_parsing.B02_doc_graph import DocumentGraph
from B_parsing.B03_table_extractor import TableExtractor
from C_generators.C01_strategy_abbrev import AbbrevSyntaxCandidateGenerator
from C_generators.C02_strategy_regex import RegexCandidateGenerator
from C_generators.C03_strategy_layout import LayoutCandidateGenerator
from C_generators.C04_strategy_flashtext import (
    RegexLexiconGenerator,
    BAD_LONG_FORMS,
    WRONG_EXPANSION_BLACKLIST,
)
from C_generators.C05_strategy_glossary import GlossaryTableCandidateGenerator
from C_generators.C06_strategy_disease import DiseaseDetector
from C_generators.C07_strategy_drug import DrugDetector
from C_generators.C12_strategy_pharma import PharmaCompanyDetector
from C_generators.C13_strategy_author import AuthorDetector
from C_generators.C14_strategy_citation import CitationDetector
from C_generators.C08_strategy_feasibility import FeasibilityDetector
from C_generators.C11_llm_feasibility import LLMFeasibilityExtractor
from C_generators.C09_strategy_document_metadata import DocumentMetadataStrategy
from C_generators.C10_vision_image_analysis import VisionImageAnalyzer
from C_generators.C15_vlm_table_extractor import VLMTableExtractor
from A_core.A07_feasibility_models import FeasibilityCandidate, FeasibilityExportDocument, NERCandidate, TrialIdentifier
from C_generators.C00_strategy_identifiers import IdentifierExtractor, IdentifierType
from A_core.A08_document_metadata_models import DocumentMetadata, DocumentMetadataExport
from D_validation.D02_llm_engine import ClaudeClient, LLMEngine
from D_validation.D03_validation_logger import ValidationLogger
from E_normalization.E01_term_mapper import TermMapper
from E_normalization.E02_disambiguator import Disambiguator
from E_normalization.E03_disease_normalizer import DiseaseNormalizer
from E_normalization.E06_nct_enricher import NCTEnricher, enrich_trial_acronym
from E_normalization.E07_deduplicator import Deduplicator
from E_normalization.E08_epi_extract_enricher import EpiExtractEnricher
from E_normalization.E09_zeroshot_bioner import ZeroShotBioNEREnricher
from E_normalization.E10_biomedical_ner_all import BiomedicalNEREnricher
from E_normalization.E11_span_deduplicator import deduplicate_feasibility_candidates
from E_normalization.E12_patient_journey_enricher import PatientJourneyEnricher
from E_normalization.E13_registry_enricher import RegistryEnricher
from E_normalization.E14_citation_validator import CitationValidator
from E_normalization.E15_genetic_enricher import GeneticEnricher
from F_evaluation.F05_extraction_analysis import run_analysis

PIPELINE_VERSION = "0.8"


# =============================================================================
# HELPER FUNCTIONS (module-level for reuse)
# =============================================================================


def extract_context_snippet(
    full_text: str, match_start: int, match_end: int, window: int = 100
) -> str:
    """Extract context snippet around a match position."""
    start = max(0, match_start - window)
    end = min(len(full_text), match_end + window)
    return full_text[start:end]


# Pattern for normalizing various dash/hyphen characters
_DASH_PATTERN = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212\u00ad\-–—]")


def normalize_lf_for_dedup(lf: str) -> str:
    """
    Normalize a long form for deduplication comparison.

    Normalizes:
    - All dash/hyphen variants to standard hyphen
    - Multiple spaces to single space
    - Case to lowercase
    - Leading/trailing whitespace

    This prevents duplicates like:
    - "renin-angiotensin" vs "renin–angiotensin" (hyphen vs en-dash)
    - "urine protein-creatinine" vs "urine protein– creatinine"
    """
    if not lf:
        return ""
    # Normalize dashes/hyphens to standard hyphen
    normalized = _DASH_PATTERN.sub("-", lf)
    # Normalize whitespace around hyphens
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized)
    # Lowercase and strip
    return normalized.lower().strip()


def has_numeric_evidence(context: str, sf: str) -> bool:
    """Check if SF appears with numeric evidence (digits, %, =, :)."""
    if not context:
        return False
    ctx = context.lower()
    sf_lower = sf.lower()

    idx = ctx.find(sf_lower)
    if idx == -1:
        return False

    window = ctx[max(0, idx - 30) : min(len(ctx), idx + len(sf) + 30)]

    # Check for numeric patterns
    if re.search(r"\d", window) or "%" in window or "=" in window:
        return True
    if re.search(rf"{re.escape(sf_lower)}\s*[:=]?\s*[\d.\-]", window):
        return True
    if re.search(r"[\d.]+\s*-\s*[\d.]+", window):
        return True
    return False


def is_valid_sf_form(
    sf: str, context: str, allowed_2letter: set, allowed_mixed: set
) -> bool:
    """Filter SF by form - reject non-abbreviation patterns."""
    sf_upper = sf.upper()

    if len(sf) == 2 and sf_upper in allowed_2letter:
        return True
    if sf_upper in allowed_mixed:
        return True

    # Reject author initials pattern (X.Y., A.B., M.C., etc.)
    if re.match(r"^[A-Z]\.[A-Z]\.$", sf):
        return False

    # Reject figure/table references (Figure 3B, Table S1, etc.)
    if re.match(r"^(Figure|Table|Fig)\s*\d+[A-Za-z]?$", sf, re.IGNORECASE):
        return False
    if re.match(r"^(Figure|Table|Fig)\s*S\d+$", sf, re.IGNORECASE):
        return False

    # Reject lowercase "al" from "et al."
    if sf == "al":
        return False

    # Reject DOI patterns (10.xxxx/yyyy)
    if re.match(r"^10\.\d{4,}", sf):
        return False

    # Reject statistical method names that look like abbreviations
    if sf_upper in {"COX", "KAPLAN", "MEIER"}:
        return False

    # Reject plural forms of common abbreviations (e.g., "CIs" when "CI" exists)
    # These are just plurals, not separate abbreviations
    if sf_upper.endswith("S") and len(sf) >= 3:
        base = sf_upper[:-1]
        if base in {"CI", "HR", "OR", "RR", "SD", "SE", "AUC", "ROC"}:
            return False

    # Reject author initials pattern in reference/citation context
    # Patterns like "Hoffman EA," or "Celermajer DS," in author lists
    if len(sf) == 2 and sf.isupper():
        # Check if SF appears right after a capitalized name (author pattern)
        author_pattern = re.search(
            rf"[A-Z][a-z]+\s+{re.escape(sf)}[,\.\s]",
            context
        )
        if author_pattern:
            # Check if this appears in reference-like context
            ctx_lower = context.lower()
            if any(ind in ctx_lower for ind in ["doi:", "et al", "10.", "pmid", "j ", "vol", "pp.", "issue"]):
                return False

    # Special case: IG only if near immunoglobulin context
    if sf_upper == "IG":
        ctx_lower = context.lower()
        return any(
            x in ctx_lower for x in ["igg", "iga", "igm", "ige", "immunoglobulin"]
        )

    # Reject lowercase words
    if sf.islower() and len(sf) > 4:
        return False
    if len(sf) > 6 and sf.islower():
        return False
    # Reject capitalized words (e.g., "Medications", "Crucially")
    if len(sf) > 5 and sf[0].isupper() and sf[1:].islower():
        return False
    return True


def score_lf_quality(
    candidate: Candidate, full_text: str, full_text_lower: Optional[str] = None
) -> int:
    """Score LF quality for dedup ranking.

    Args:
        candidate: The candidate to score
        full_text: Original full text
        full_text_lower: Pre-cached lowercase version (optional, avoids repeated .lower() calls)
    """
    score = 0
    lf = (candidate.long_form or "").lower()
    sf = candidate.short_form
    sf_upper = sf.upper()

    # Use cached lowercase or compute once
    text_lower = full_text_lower if full_text_lower is not None else full_text.lower()

    # PRIORITY BOOST: Stats abbreviations prefer canonical forms
    # This ensures CI→"confidence interval" beats CI→"Curie"
    STATS_CANONICAL = {
        "CI": "confidence interval",
        "SD": "standard deviation",
        "SE": "standard error",
        "OR": "odds ratio",
        "RR": "risk ratio",
        "HR": "hazard ratio",
        "IQR": "interquartile range",
        "AUC": "area under the curve",
        "ROC": "receiver operating characteristic",
    }
    if sf_upper in STATS_CANONICAL:
        canonical = STATS_CANONICAL[sf_upper]
        if canonical in lf:
            score += 200  # Strong boost for canonical stats LF
        else:
            score -= 100  # Penalize non-canonical stats LF

    if lf and sf in full_text:
        # Check if LF appears within 200 chars of SF - use islice to avoid materializing all matches
        for m in islice(re.finditer(re.escape(sf), full_text), 5):
            window_start = max(0, m.start() - 200)
            window_end = m.start() + 200
            window = text_lower[window_start:window_end]
            if lf in window:
                score += 100
                break

    if lf and lf in text_lower:
        score += 50

    if lf:
        if len(lf) > 60:
            score -= 20
        lf_words = set(lf.split())
        if lf_words & {"the", "a", "an", "is", "are", "was"}:
            score -= 10
        # Penalize LFs that look like partial text extractions
        if lf.startswith("and ") or lf.startswith("or ") or lf.startswith("the "):
            score -= 50

    if candidate.provenance and candidate.provenance.lexicon_source:
        if "umls" in candidate.provenance.lexicon_source.lower():
            score += 30

    return score


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================


class Orchestrator:
    """
    Main pipeline orchestrator for clinical document metadata extraction.

    This class coordinates the entire extraction pipeline, processing PDF documents
    through multiple stages to extract structured metadata including abbreviations,
    drugs, diseases, feasibility information, and more.

    Pipeline Stages:
        1. Parse PDF -> DocumentGraph (structure extraction)
        2. Generate candidates (syntax patterns + lexicon matching)
        3. Validate candidates with Claude LLM
        4. Normalize and deduplicate entities
        5. Detect domain-specific entities (drugs, diseases, etc.)
        6. Export results to JSON

    Attributes:
        config: Configuration dictionary loaded from YAML.
        run_id: Unique identifier for this pipeline run.
        log_dir: Directory for validation logs.
        pdf_dir: Directory containing input PDFs.
        model: Claude model to use for validation.

    Example:
        >>> orchestrator = Orchestrator(config_path="config.yaml")
        >>> orchestrator.process_folder()
    """

    DEFAULT_CONFIG = (
        "/Users/frederictetard/Projects/ese/corpus_metadata/G_config/config.yaml"
    )

    def __init__(
        self,
        log_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        config_path: Optional[str] = None,
        run_id: Optional[str] = None,
        gold_json: Optional[str] = None,
        heuristics_config: Optional[HeuristicsConfig] = None,
    ) -> None:
        """
        Initialize the orchestrator with configuration.

        Args:
            log_dir: Directory for validation logs. Defaults to config value.
            output_dir: Directory for output files. Defaults to PDF subdirectories.
            model: Claude model for validation. Defaults to config value.
            api_key: Anthropic API key. Defaults to environment variable.
            config_path: Path to config.yaml file.
            run_id: Unique run identifier. Auto-generated if not provided.
            gold_json: Path to gold standard JSON for evaluation.
            heuristics_config: Pre-loaded heuristics configuration.
        """
        self._logger = get_logger(__name__)

        self.config_path = config_path or self.DEFAULT_CONFIG
        self.config = self._load_config(self.config_path)

        # Load extraction pipeline settings directly from config.yaml
        self._load_extraction_settings()

        # Extract paths from config
        paths = self.config.get("paths", {})
        base_path = paths.get("base", "/Users/frederictetard/Projects/ese")

        self.log_dir = Path(
            log_dir or Path(base_path) / paths.get("logs", "corpus_log")
        )
        self.output_dir = Path(output_dir) if output_dir else None
        self.pdf_dir = Path(base_path) / paths.get("pdf_input", "Pdfs")
        self.gold_json = gold_json or str(
            Path(base_path) / paths.get("gold_json", "gold_data/papers_gold_v2.json")
        )

        # API settings
        api_cfg = self.config.get("api", {}).get("claude", {})
        val_cfg = api_cfg.get("validation", {})

        self.model = model or val_cfg.get("model", "claude-sonnet-4-20250514")
        self.batch_delay_ms = api_cfg.get("batch_delay_ms", 100)

        # Heuristics settings
        heur_cfg = self.config.get("heuristics", {})
        self.enable_haiku_screening = heur_cfg.get("enable_haiku_screening", False)

        self.run_id = run_id or generate_run_id("RUN")
        self.heuristics = heuristics_config or HeuristicsConfig.from_yaml(
            self.config_path
        )

        # Initialize components
        self._init_components(paths, base_path, api_key, val_cfg)

        # Log initialization (both console and file)
        self._logger.info(f"Orchestrator v{PIPELINE_VERSION} initialized")
        self._logger.info(f"Run ID: {self.run_id}")
        self._logger.info(f"Config: {self.config_path}")
        self._logger.info(f"Model: {self.model}")

        # Console output for user visibility
        print(f"\nOrchestrator v{PIPELINE_VERSION} initialized")
        print(f"  Run ID: {self.run_id}")
        print(f"  Config: {self.config_path}")
        print(f"  Model:  {self.model}")
        print(f"  Logs:   {self.log_dir}")
        self._print_extraction_config()

    def _load_extraction_settings(self) -> None:
        """Load extraction pipeline settings from config.yaml."""
        pipeline = self.config.get("extraction_pipeline", {})
        preset = pipeline.get("preset")
        extractors = pipeline.get("extractors", {})
        options = pipeline.get("options", {})

        # Define presets (override individual flags if preset is set)
        PRESETS = {
            "drugs_only": {
                "drugs": True, "diseases": False, "abbreviations": False,
                "feasibility": False, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": False,
            },
            "diseases_only": {
                "drugs": False, "diseases": True, "abbreviations": False,
                "feasibility": False, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": False,
            },
            "abbreviations_only": {
                "drugs": False, "diseases": False, "abbreviations": True,
                "feasibility": False, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": False,
            },
            "feasibility_only": {
                "drugs": False, "diseases": False, "abbreviations": False,
                "feasibility": True, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": False,
            },
            "entities_only": {
                "drugs": True, "diseases": True, "abbreviations": True,
                "feasibility": False, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": False,
            },
            "clinical_entities": {
                "drugs": True, "diseases": True, "abbreviations": False,
                "feasibility": False, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": False,
            },
            "metadata_only": {
                "drugs": False, "diseases": False, "abbreviations": False,
                "feasibility": False, "pharma_companies": False, "authors": True,
                "citations": True, "document_metadata": True, "tables": False,
            },
            "standard": {
                "drugs": True, "diseases": True, "abbreviations": True,
                "feasibility": True, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": True,
            },
            "all": {
                "drugs": True, "diseases": True, "abbreviations": True,
                "feasibility": True, "pharma_companies": True, "authors": True,
                "citations": True, "document_metadata": True, "tables": True,
            },
            "minimal": {
                "drugs": False, "diseases": False, "abbreviations": True,
                "feasibility": False, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": False,
            },
        }

        # Apply preset if set, otherwise use individual flags
        self.active_preset = preset
        if preset and preset in PRESETS:
            preset_config = PRESETS[preset]
            self.extract_drugs = preset_config["drugs"]
            self.extract_diseases = preset_config["diseases"]
            self.extract_abbreviations = preset_config["abbreviations"]
            self.extract_feasibility = preset_config["feasibility"]
            self.extract_pharma = preset_config["pharma_companies"]
            self.extract_authors = preset_config["authors"]
            self.extract_citations = preset_config["citations"]
            self.extract_doc_metadata = preset_config["document_metadata"]
            self.extract_tables = preset_config["tables"]
        else:
            # Use individual extractor flags
            self.active_preset = None
            self.extract_drugs = extractors.get("drugs", True)
            self.extract_diseases = extractors.get("diseases", True)
            self.extract_abbreviations = extractors.get("abbreviations", True)
            self.extract_feasibility = extractors.get("feasibility", True)
            self.extract_pharma = extractors.get("pharma_companies", False)
            self.extract_authors = extractors.get("authors", False)
            self.extract_citations = extractors.get("citations", False)
            self.extract_doc_metadata = extractors.get("document_metadata", False)
            self.extract_tables = extractors.get("tables", True)

        # Processing options (always read from options, not affected by preset)
        self.use_llm_validation = options.get("use_llm_validation", True)
        self.use_llm_feasibility = options.get("use_llm_feasibility", True)
        self.use_vlm_tables = options.get("use_vlm_tables", False)
        self.use_normalization = options.get("use_normalization", True)
        self.use_epi_enricher = options.get("use_epi_enricher", True)
        self.use_zeroshot_bioner = options.get("use_zeroshot_bioner", True)
        self.use_biomedical_ner = options.get("use_biomedical_ner", True)
        self.use_patient_journey = options.get("use_patient_journey", True)
        self.use_registry_extraction = options.get("use_registry_extraction", True)
        self.use_genetic_extraction = options.get("use_genetic_extraction", True)

    def _enabled_extractors_str(self) -> str:
        """Return string of enabled extractors."""
        enabled = []
        if self.extract_drugs:
            enabled.append("drugs")
        if self.extract_diseases:
            enabled.append("diseases")
        if self.extract_abbreviations:
            enabled.append("abbreviations")
        if self.extract_feasibility:
            enabled.append("feasibility")
        if self.extract_pharma:
            enabled.append("pharma")
        if self.extract_authors:
            enabled.append("authors")
        if self.extract_citations:
            enabled.append("citations")
        if self.extract_doc_metadata:
            enabled.append("doc_metadata")
        if self.extract_tables:
            enabled.append("tables")
        return ", ".join(enabled) if enabled else "none"

    def _print_extraction_config(self) -> None:
        """Print extraction configuration summary."""
        print("\n  Extraction Pipeline Configuration:")
        print("  " + "-" * 40)

        # Show active preset if set
        if self.active_preset:
            print(f"  PRESET: {self.active_preset}")
        else:
            print("  PRESET: (custom - using individual flags)")

        # Extractors
        print("\n  EXTRACTORS:")
        extractors = [
            ("drugs", self.extract_drugs),
            ("diseases", self.extract_diseases),
            ("abbreviations", self.extract_abbreviations),
            ("feasibility", self.extract_feasibility),
            ("pharma_companies", self.extract_pharma),
            ("authors", self.extract_authors),
            ("citations", self.extract_citations),
            ("document_metadata", self.extract_doc_metadata),
            ("tables", self.extract_tables),
        ]
        for name, enabled in extractors:
            status = "ON" if enabled else "OFF"
            print(f"    {name:<20} {status}")

        # Options
        print("\n  OPTIONS:")
        options = [
            ("use_llm_validation", self.use_llm_validation),
            ("use_llm_feasibility", self.use_llm_feasibility),
            ("use_vlm_tables", self.use_vlm_tables),
            ("use_normalization", self.use_normalization),
            ("use_epi_enricher", self.use_epi_enricher),
            ("use_zeroshot_bioner", self.use_zeroshot_bioner),
            ("use_biomedical_ner", self.use_biomedical_ner),
            ("use_patient_journey", self.use_patient_journey),
            ("use_registry_extraction", self.use_registry_extraction),
            ("use_genetic_extraction", self.use_genetic_extraction),
            ("haiku_screening", self.enable_haiku_screening),
        ]
        for name, enabled in options:
            status = "ON" if enabled else "OFF"
            print(f"    {name:<20} {status}")

        print("  " + "-" * 40)

    def _init_components(
        self, paths: dict, base_path: str, api_key: Optional[str], val_cfg: dict
    ) -> None:
        """Initialize parser, generators, validation, and normalization components."""
        lexicons = self.config.get("lexicons", {})
        dict_path = Path(base_path) / paths.get("dictionaries", "ouput_datasources")

        # Parser
        self.parser = PDFToDocGraphParser()
        self.table_extractor = TableExtractor()

        # Generators
        gen_cfg = self.config.get("generators", {})
        regex_cfg = gen_cfg.get("regex_pattern", {})
        lexicon_cfg = gen_cfg.get("lexicon", {})

        lexicon_gen_config = {
            "run_id": self.run_id,
            "abbrev_lexicon_path": str(
                dict_path
                / lexicons.get(
                    "abbreviation_general", "2025_08_abbreviation_general.json"
                )
            ),
            "disease_lexicon_path": str(
                dict_path
                / lexicons.get("disease_lexicon", "2025_08_lexicon_disease.json")
            ),
            "orphanet_lexicon_path": str(
                dict_path
                / lexicons.get("orphanet_diseases", "2025_08_orphanet_diseases.json")
            ),
            "rare_disease_acronyms_path": str(
                dict_path
                / lexicons.get(
                    "rare_disease_acronyms", "2025_08_rare_disease_acronyms.json"
                )
            ),
            "umls_abbrev_path": str(
                dict_path
                / lexicons.get(
                    "umls_biological", "2025_08_umls_biological_abbreviations_v5.tsv"
                )
            ),
            "umls_clinical_path": str(
                dict_path
                / lexicons.get(
                    "umls_clinical", "2025_08_umls_clinical_abbreviations_v5.tsv"
                )
            ),
            "anca_lexicon_path": str(
                dict_path
                / lexicons.get("disease_lexicon_anca", "disease_lexicon_anca.json")
            ),
            "igan_lexicon_path": str(
                dict_path
                / lexicons.get("disease_lexicon_igan", "disease_lexicon_igan.json")
            ),
            "pah_lexicon_path": str(
                dict_path
                / lexicons.get("disease_lexicon_pah", "disease_lexicon_pah.json")
            ),
            "trial_acronyms_path": str(
                dict_path
                / lexicons.get("trial_acronyms", "trial_acronyms_lexicon.json")
            ),
            "pro_scales_path": str(
                dict_path / lexicons.get("pro_scales", "pro_scales_lexicon.json")
            ),
            "context_window": lexicon_cfg.get("context_window", 300),
        }

        self.generators = [
            AbbrevSyntaxCandidateGenerator(config={"run_id": self.run_id}),
            GlossaryTableCandidateGenerator(config={"run_id": self.run_id}),
            RegexCandidateGenerator(
                config={
                    "run_id": self.run_id,
                    "enabled_types": regex_cfg.get(
                        "enabled_types",
                        ["TRIAL_ID", "COMPOUND_ID", "DOI", "PMID", "PMCID"],
                    ),
                }
            ),
            LayoutCandidateGenerator(config={"run_id": self.run_id}),
            RegexLexiconGenerator(config=lexicon_gen_config),
        ]

        # Validation (controlled by extraction_pipeline.options.use_llm_validation)
        if self.use_llm_validation:
            self.claude_client = ClaudeClient(
                api_key=api_key, model=self.model, config_path=self.config_path
            )
            self.llm_engine = LLMEngine(
                llm_client=self.claude_client,
                model=self.model,
                run_id=self.run_id,
                max_tokens=val_cfg.get("max_tokens", 450),
                temperature=val_cfg.get("temperature", 0.0),
                top_p=val_cfg.get("top_p", 1.0),
            )
        else:
            self.claude_client = None
            self.llm_engine = None

        # VLM Table Extractor (uses Claude Vision for better table extraction)
        # Controlled by extraction_pipeline.options.use_vlm_tables
        if self.use_vlm_tables and self.claude_client:
            self.vlm_table_extractor = VLMTableExtractor(
                llm_client=self.claude_client,
                llm_model=self.config.get("llm", {}).get("model", "claude-sonnet-4-20250514"),
                config={"run_id": self.run_id},
            )
        else:
            self.vlm_table_extractor = None

        # Logger
        self.logger = ValidationLogger(log_dir=str(self.log_dir), run_id=self.run_id)

        # Normalization
        norm_cfg = self.config.get("normalization", {})
        term_mapper_cfg = norm_cfg.get("term_mapper", {})
        disambig_cfg = norm_cfg.get("disambiguator", {})

        self.term_mapper = TermMapper(
            config={
                "mapping_file_path": str(
                    dict_path
                    / lexicons.get(
                        "abbreviation_general", "2025_08_abbreviation_general.json"
                    )
                ),
                "enable_fuzzy_matching": term_mapper_cfg.get(
                    "enable_fuzzy_matching", False
                ),
                "fuzzy_cutoff": term_mapper_cfg.get("fuzzy_cutoff", 0.90),
                "fill_long_form_for_orphans": term_mapper_cfg.get(
                    "fill_long_form_for_orphans", False
                ),
            }
        )

        self.disambiguator = Disambiguator(
            config={
                "min_context_score": disambig_cfg.get("min_context_score", 2),
                "min_margin": disambig_cfg.get("min_margin", 1),
                "fill_long_form_for_orphans": disambig_cfg.get(
                    "fill_long_form_for_orphans", True
                ),
            }
        )

        dedup_cfg = norm_cfg.get("deduplicator", {})
        self.deduplicator = Deduplicator(
            config={
                "only_validated": dedup_cfg.get("only_validated", True),
                "store_alternatives": dedup_cfg.get("store_alternatives", True),
                "max_alternatives": dedup_cfg.get("max_alternatives", 5),
            }
        )

        # Disease detection components
        self.disease_detector = DiseaseDetector(
            config={
                "run_id": self.run_id,
                "lexicon_base_path": str(dict_path),
            }
        )
        self.disease_normalizer = DiseaseNormalizer()
        self.disease_enricher = None

        # Drug detection components
        self.drug_detector = DrugDetector(
            config={
                "run_id": self.run_id,
                "lexicon_base_path": str(dict_path),
            }
        )
        self.drug_enricher = None

        # Pharma company detection
        self.pharma_detector = PharmaCompanyDetector(
            config={
                "run_id": self.run_id,
                "lexicon_base_path": str(dict_path),
            }
        )

        # Author detection
        self.author_detector = AuthorDetector(
            config={
                "run_id": self.run_id,
                "pipeline_version": PIPELINE_VERSION,
            }
        )

        # Citation detection
        self.citation_detector = CitationDetector(
            config={
                "run_id": self.run_id,
                "pipeline_version": PIPELINE_VERSION,
            }
        )

        # Feasibility detection components
        # Controlled by extraction_pipeline.options.use_llm_feasibility
        self.feasibility_detector = FeasibilityDetector(
            config={"run_id": self.run_id}
        )
        if self.use_llm_feasibility and self.claude_client:
            self.llm_feasibility_extractor = LLMFeasibilityExtractor(
                llm_client=self.claude_client,
                llm_model=self.config.get("llm", {}).get("model", "claude-sonnet-4-20250514"),
                config={"run_id": self.run_id},
            )
        else:
            self.llm_feasibility_extractor = None

        # EpiExtract4GARD-v2 enricher for rare disease epidemiology NER
        # Uses BioBERT model to extract LOC, EPI, STAT entities
        if self.use_epi_enricher:
            self.epi_enricher = EpiExtractEnricher(
                config={"run_id": self.run_id}
            )
        else:
            self.epi_enricher = None

        # ZeroShotBioNER enricher for flexible biomedical entity extraction
        # Extracts ADE, dosage, frequency, route, duration, etc.
        if self.use_zeroshot_bioner:
            self.zeroshot_bioner = ZeroShotBioNEREnricher(
                config={"run_id": self.run_id}
            )
        else:
            self.zeroshot_bioner = None

        # d4data/biomedical-ner-all for comprehensive biomedical NER
        # Extracts 84 entity types: symptoms, procedures, lab values, demographics
        if self.use_biomedical_ner:
            self.biomedical_ner = BiomedicalNEREnricher(
                config={"run_id": self.run_id}
            )
        else:
            self.biomedical_ner = None

        # Patient journey enricher for diagnostic delay, treatment lines, care pathway
        # Uses ZeroShotBioNER with custom entity labels
        if self.use_patient_journey:
            self.patient_journey_enricher = PatientJourneyEnricher(
                config={"run_id": self.run_id}
            )
        else:
            self.patient_journey_enricher = None

        # Registry enricher for cohort access, natural history data
        # Uses ZeroShotBioNER with registry-specific entity labels
        if self.use_registry_extraction:
            self.registry_enricher = RegistryEnricher(
                config={"run_id": self.run_id}
            )
        else:
            self.registry_enricher = None

        # Genetic enricher for gene symbols, HGVS variants, HPO, ORDO
        # Uses regex patterns - no model loading required
        if self.use_genetic_extraction:
            self.genetic_enricher = GeneticEnricher(
                config={"run_id": self.run_id}
            )
        else:
            self.genetic_enricher = None

        # Document metadata extraction
        doc_metadata_cfg = self.config.get("document_metadata", {})
        doc_types_path = doc_metadata_cfg.get(
            "document_types_path",
            str(dict_path / "2025_08_document_types.json")
        )
        self.doc_metadata_strategy = DocumentMetadataStrategy(
            document_types_path=doc_types_path if Path(doc_types_path).exists() else None,
            llm_client=self.claude_client,
            llm_model=self.model,
            run_id=self.run_id,
            pipeline_version=PIPELINE_VERSION,
        )

        # Load rare disease acronyms for fallback lookup (SF→LF when LLM has no expansion)
        self.rare_disease_lookup: Dict[str, str] = {}
        rare_disease_path = dict_path / lexicons.get(
            "rare_disease_acronyms", "2025_08_rare_disease_acronyms.json"
        )
        if rare_disease_path.exists():
            try:
                data = json.loads(rare_disease_path.read_text(encoding="utf-8"))
                skipped_bad = 0
                for acronym, entry in data.items():
                    if isinstance(entry, dict) and entry.get("name"):
                        sf_key = acronym.strip().upper()
                        sf_lower = acronym.strip().lower()
                        lf_value = entry["name"].strip()
                        lf_lower = lf_value.lower()
                        # Filter out known wrong SF -> LF pairs and bad long forms
                        if (sf_lower, lf_lower) in WRONG_EXPANSION_BLACKLIST or lf_lower in BAD_LONG_FORMS:
                            skipped_bad += 1
                            continue
                        self.rare_disease_lookup[sf_key] = lf_value
                if skipped_bad > 0:
                    print(f"    [INFO] Skipped {skipped_bad} bad rare disease entries")
            except Exception as e:
                print(f"  [WARN] Failed to load rare disease lexicon: {e}")

        # NCT enricher for clinical trial identifier expansion
        nct_cfg = self.config.get("nct_enricher", {})
        if nct_cfg.get("enabled", True):
            self.nct_enricher = NCTEnricher(nct_cfg)
        else:
            self.nct_enricher = None

        # Print unified lexicon summary
        self._print_unified_lexicon_summary()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            print(f"[WARN] Config file not found: {config_path}, using defaults")
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARN] Failed to load config from {config_path}: {e}")
            return {}

    def _print_unified_lexicon_summary(self) -> None:
        """Print unified summary of all loaded lexicons across all detectors."""
        # Collect stats from all sources, track seen filenames to avoid duplicates
        all_stats: List[
            Tuple[str, int, str, str]
        ] = []  # (name, count, filename, category)
        seen_files: set = set()

        # From C04 (abbreviation generator) - skip specialized disease lexicons (handled by C06)
        for gen in self.generators:
            if hasattr(gen, "_lexicon_stats"):
                for name, count, filename in gen._lexicon_stats:
                    # Skip specialized disease lexicons from C04 - they're handled by C06
                    if name in (
                        "ANCA disease",
                        "IgAN disease",
                        "PAH disease",
                    ):
                        continue
                    # Categorize based on name
                    if name in (
                        "Abbreviations",
                        "Clinical research",
                        "UMLS biological",
                        "UMLS clinical",
                    ):
                        cat = "Abbreviation"
                    elif name == "Rare disease acronyms":
                        cat = "Disease"
                    elif name in ("Trial acronyms", "PRO scales", "Pharma companies"):
                        cat = "Other"
                    else:
                        cat = "Other"
                    if filename not in seen_files:
                        all_stats.append((name, count, filename, cat))
                        seen_files.add(filename)

        # From C07 (drug detector) - add Drug category first for display order
        if self.drug_detector and hasattr(self.drug_detector, "_lexicon_stats"):
            for name, count, filename in self.drug_detector._lexicon_stats:
                # Use category-prefixed key to allow scispacy in multiple categories
                key = f"Drug:{filename}"
                if key not in seen_files:
                    all_stats.append((name, count, filename, "Drug"))
                    seen_files.add(key)

        # From C06 (disease detector)
        if self.disease_detector and hasattr(self.disease_detector, "_lexicon_stats"):
            for name, count, filename in self.disease_detector._lexicon_stats:
                display_name = name.replace("Specialized ", "")
                # Use category-prefixed key to allow scispacy in multiple categories
                key = f"Disease:{filename}"
                if key not in seen_files:
                    all_stats.append((display_name, count, filename, "Disease"))
                    seen_files.add(key)

        if not all_stats:
            return

        # Group by category with explicit order
        categories = [
            ("Abbreviation", []),
            ("Drug", []),
            ("Disease", []),
            ("Other", []),
        ]
        cat_dict = {name: items for name, items in categories}

        for name, count, filename, cat in all_stats:
            if cat in cat_dict:
                cat_dict[cat].append((name, count, filename))

        # Calculate totals
        total = sum(count for _, count, _, _ in all_stats if count > 1)
        file_count = len([s for s in all_stats if s[1] > 0])

        print(f"\nLexicons loaded: {file_count} sources, {total:,} entries")
        print("─" * 70)

        for cat_name, items in categories:
            if not items:
                continue
            cat_total = sum(count for _, count, _ in items if count > 1)
            print(f"  {cat_name} ({cat_total:,} entries)")
            for name, count, filename in items:
                if count > 1:
                    print(f"    • {name:<26} {count:>8,}  {filename}")
                else:
                    print(f"    • {name:<26} {'enabled':>8}  {filename}")
        print()

    # =========================================================================
    # ENTITY CREATION HELPERS
    # =========================================================================

    def _create_entity_from_candidate(
        self,
        candidate: Candidate,
        status: ValidationStatus,
        confidence: float,
        reason: str,
        flags: List[str],
        raw_response: Dict,
        long_form_override: Optional[str] = None,
    ) -> ExtractedEntity:
        """Create ExtractedEntity from a Candidate (for auto-approve/reject)."""
        context = (candidate.context_text or "").strip()
        ctx_hash = hash_string(context) if context else "no_context"
        primary = EvidenceSpan(
            text=context,
            location=candidate.context_location,
            scope_ref=ctx_hash,
            start_char_offset=0,
            end_char_offset=len(context),
        )
        return ExtractedEntity(
            candidate_id=candidate.id,
            doc_id=candidate.doc_id,
            field_type=candidate.field_type,
            short_form=candidate.short_form.strip(),
            long_form=long_form_override
            or (candidate.long_form.strip() if candidate.long_form else None),
            primary_evidence=primary,
            supporting_evidence=[],
            status=status,
            confidence_score=confidence,
            rejection_reason=reason if status == ValidationStatus.REJECTED else None,
            validation_flags=flags,
            provenance=candidate.provenance,
            raw_llm_response=raw_response,
        )

    def _create_entity_from_search(
        self,
        doc_id: str,
        full_text: str,
        match: re.Match,
        long_form: Optional[str],
        field_type: FieldType,
        confidence: float,
        flags: List[str],
        rule_version: str,
        lexicon_source: str,
    ) -> ExtractedEntity:
        """Create ExtractedEntity from a text search match."""
        context_snippet = extract_context_snippet(full_text, match.start(), match.end())
        ctx_hash = hash_string(context_snippet)

        primary = EvidenceSpan(
            text=context_snippet,
            location=Coordinate(page_num=1),
            scope_ref=ctx_hash,
            start_char_offset=match.start() - max(0, match.start() - 100),
            end_char_offset=match.end() - max(0, match.start() - 100),
        )

        prov = ProvenanceMetadata(
            pipeline_version=PIPELINE_VERSION,
            run_id=self.run_id,
            doc_fingerprint=lexicon_source,
            generator_name=GeneratorType.LEXICON_MATCH,
            rule_version=rule_version,
            lexicon_source=f"orchestrator:{lexicon_source}",
        )

        return ExtractedEntity(
            candidate_id=uuid.uuid4(),
            doc_id=doc_id,
            field_type=field_type,
            short_form=match.group(),
            long_form=long_form,
            primary_evidence=primary,
            supporting_evidence=[],
            status=ValidationStatus.VALIDATED,
            confidence_score=confidence,
            rejection_reason=None,
            validation_flags=flags,
            provenance=prov,
            raw_llm_response={"auto": lexicon_source},
        )

    # =========================================================================
    # PIPELINE STAGES
    # =========================================================================

    def _parse_pdf(self, pdf_path: Path):
        """Stage 1: Parse PDF into DocumentGraph."""
        print("\n[1/12] Parsing PDF...")
        start = time.time()

        # Get output directory for images (creates folder if needed)
        output_dir = self._get_output_dir(pdf_path)

        doc = self.parser.parse(str(pdf_path), image_output_dir=str(output_dir))

        # Extract tables with VLM if available (300 DPI for optimal VLM reading)
        doc = self.table_extractor.populate_document_graph(
            doc,
            str(pdf_path),
            render_images=True,
            use_vlm=self.use_vlm_tables and self.vlm_table_extractor is not None,
            vlm_extractor=self.vlm_table_extractor,
        )

        total_blocks = sum(len(p.blocks) for p in doc.pages.values())
        total_tables = sum(len(p.tables) for p in doc.pages.values())

        print(f"  Pages: {len(doc.pages)}")
        print(f"  Blocks: {total_blocks}")
        print(f"  Tables: {total_tables}")
        print(f"  Time: {time.time() - start:.2f}s")

        self._export_extracted_text(pdf_path, doc)
        return doc

    def _generate_candidates(self, doc) -> Tuple[List[Candidate], str]:
        """Stage 2: Generate and deduplicate candidates."""
        print("\n[2/12] Generating candidates...")
        start = time.time()

        all_candidates = []
        for gen in self.generators:
            candidates = gen.extract(doc)
            print(f"  {gen.generator_type.value}: {len(candidates)} candidates")
            all_candidates.extend(candidates)

        # Deduplicate using normalized LF to handle punctuation variants
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            # Use normalize_lf_for_dedup to handle hyphen/dash variants
            key = (c.short_form.upper(), normalize_lf_for_dedup(c.long_form or ""))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        # Build full text
        full_text = " ".join(
            block.text for block in doc.iter_linear_blocks() if block.text
        )

        print(f"  Total unique: {len(unique_candidates)}")
        print(f"  Time: {time.time() - start:.2f}s")

        return unique_candidates, full_text

    def _filter_candidates(
        self,
        candidates: List[Candidate],
        full_text: str,
    ) -> Tuple[List[Candidate], set, Counter, int]:
        """Filter and reduce candidates before validation."""
        # Corroborated SFs (found by non-lexicon generators)
        corroborated_sfs = {
            c.short_form.upper()
            for c in candidates
            if c.generator_type != GeneratorType.LEXICON_MATCH
        }

        # Count SF occurrences
        abbrev_pattern = r"\b[A-Za-z][A-Za-z0-9]{1,11}\b"
        all_tokens = re.findall(abbrev_pattern, full_text)
        word_counts = Counter(
            t.upper() for t in all_tokens if any(c.isupper() for c in t)
        )

        # Keep lexicon matches if corroborated or frequent
        def should_keep(c: Candidate) -> bool:
            if c.generator_type != GeneratorType.LEXICON_MATCH:
                return True
            sf_upper = c.short_form.upper()
            return sf_upper in corroborated_sfs or word_counts.get(sf_upper, 0) >= 2

        needs_validation = [c for c in candidates if should_keep(c)]

        # Reduce lexicon matches (dedup by SF, filter by form)
        allowed_2letter = self.heuristics.allowed_2letter_sfs
        allowed_mixed = self.heuristics.allowed_mixed_case

        lexicon_by_sf: Dict[str, List[Candidate]] = {}
        non_lexicon = []
        sf_form_rejected = 0

        for c in needs_validation:
            if c.generator_type == GeneratorType.LEXICON_MATCH:
                ctx = c.context_text or ""
                if not is_valid_sf_form(
                    c.short_form, ctx, allowed_2letter, allowed_mixed
                ):
                    sf_form_rejected += 1
                    continue
                sf_upper = c.short_form.upper()
                lexicon_by_sf.setdefault(sf_upper, []).append(c)
            else:
                non_lexicon.append(c)

        # Keep top 2 LFs per SF (cache lowercase for performance)
        full_text_lower = full_text.lower()
        deduped_lexicon = []
        for sf, cands in lexicon_by_sf.items():
            if len(cands) <= 2:
                deduped_lexicon.extend(cands)
            else:
                # Use heapq.nlargest instead of full sort for O(n) vs O(n log n)
                top_2 = heapq.nlargest(
                    2,
                    cands,
                    key=lambda c: score_lf_quality(c, full_text, full_text_lower),
                )
                deduped_lexicon.extend(top_2)

        lexicon_before = sum(len(v) for v in lexicon_by_sf.values()) + sf_form_rejected
        print(
            f"  LEXICON_MATCH reduction: {lexicon_before} -> {len(deduped_lexicon)} "
            f"(dedup: {lexicon_before - sf_form_rejected - len(deduped_lexicon)}, form filter: {sf_form_rejected})"
        )

        return (
            non_lexicon + deduped_lexicon,
            corroborated_sfs,
            word_counts,
            len(candidates) - len(non_lexicon + deduped_lexicon),
        )

    def _apply_heuristics(
        self,
        candidates: List[Candidate],
        counters: HeuristicsCounters,
    ) -> Tuple[List[Tuple[Candidate, ExtractedEntity]], List[Candidate]]:
        """Apply auto-approve/reject heuristics. Returns (auto_results, llm_candidates)."""
        auto_results = []
        llm_candidates = []

        for c in candidates:
            sf_upper = c.short_form.upper()
            ctx = c.context_text or ""

            # Auto-reject blacklisted
            if sf_upper in self.heuristics.sf_blacklist:
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.REJECTED,
                    0.95,
                    "Blacklisted: not a valid abbreviation",
                    ["auto_rejected_blacklist"],
                    {"auto": "blacklist"},
                )
                auto_results.append((c, entity))
                counters.blacklisted_fp_count += 1
                continue

            # Auto-reject Figure/Table references (Figure 3B, Table S1, Tables S1-S13, etc.)
            if re.match(r"^(Figure|Table|Fig|Figures|Tables)\s*S?\d*[A-Za-z]?(-S?\d+)?$", c.short_form, re.IGNORECASE):
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.REJECTED,
                    0.95,
                    "Figure/Table reference, not an abbreviation",
                    ["auto_rejected_form"],
                    {"auto": "figure_table_ref"},
                )
                auto_results.append((c, entity))
                counters.form_filter_rejected += 1
                continue

            # Auto-reject author initials pattern (X.Y., A.B., etc.)
            if re.match(r"^[A-Z]\.[A-Z]\.$", c.short_form):
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.REJECTED,
                    0.95,
                    "Author initials pattern, not an abbreviation",
                    ["auto_rejected_form"],
                    {"auto": "author_initials"},
                )
                auto_results.append((c, entity))
                counters.form_filter_rejected += 1
                continue

            # Auto-reject context mismatch
            if not check_context_match(c.short_form, ctx, self.heuristics):
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.REJECTED,
                    0.90,
                    "Rejected: ambiguous SF without required context",
                    ["auto_rejected_context"],
                    {"auto": "context_mismatch"},
                )
                auto_results.append((c, entity))
                counters.context_rejected += 1
                continue

            # Auto-reject trial IDs
            if check_trial_id(c.short_form, self.heuristics):
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.REJECTED,
                    0.90,
                    "Excluded: trial identifier (NCT number)",
                    ["auto_rejected_trial_id"],
                    {"auto": "trial_id_excluded"},
                )
                auto_results.append((c, entity))
                counters.trial_id_excluded += 1
                continue

            # Auto-approve stats with numeric evidence
            if sf_upper in self.heuristics.stats_abbrevs and has_numeric_evidence(
                ctx, c.short_form
            ):
                canonical_lf = self.heuristics.stats_abbrevs.get(sf_upper)
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.VALIDATED,
                    0.90,
                    "Stats abbreviation with numeric evidence",
                    ["auto_approved_stats"],
                    {"auto": "stats_whitelist"},
                    long_form_override=canonical_lf,
                )
                auto_results.append((c, entity))
                counters.recovered_by_stats_whitelist += 1
                continue

            # Auto-approve country codes
            if sf_upper in self.heuristics.country_abbrevs:
                canonical_lf = self.heuristics.country_abbrevs.get(sf_upper)
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.VALIDATED,
                    0.90,
                    "Country code abbreviation",
                    ["auto_approved_country"],
                    {"auto": "country_code"},
                    long_form_override=canonical_lf,
                )
                auto_results.append((c, entity))
                counters.recovered_by_country_code += 1
                continue

            # Auto-reject common words
            if sf_upper in self.heuristics.common_words:
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.REJECTED,
                    0.95,
                    "Common English word, not an abbreviation",
                    ["auto_rejected"],
                    {"auto": "rejected_common_word"},
                )
                auto_results.append((c, entity))
                counters.common_word_rejected += 1
                continue

            # Auto-reject malformed long forms (partial sentences, truncated, etc.)
            lf = c.long_form or ""
            if lf:
                # Reject LFs that look like partial sentences
                lf_lower = lf.lower()
                partial_starters = (
                    "and ", "or ", "our ", "the ", "a ", "an ",
                    "whereas ", "although ", "because ", "while ",
                    "from the ", "in the ", "of the ", "to the ",
                )
                if any(lf_lower.startswith(s) for s in partial_starters):
                    entity = self._create_entity_from_candidate(
                        c,
                        ValidationStatus.REJECTED,
                        0.90,
                        "Malformed LF: starts with article/conjunction",
                        ["auto_rejected_form"],
                        {"auto": "malformed_lf"},
                    )
                    auto_results.append((c, entity))
                    counters.form_filter_rejected += 1
                    continue

                # Reject LFs with unclosed brackets
                if lf.count("[") != lf.count("]") or lf.count("(") != lf.count(")"):
                    entity = self._create_entity_from_candidate(
                        c,
                        ValidationStatus.REJECTED,
                        0.90,
                        "Malformed LF: unclosed brackets",
                        ["auto_rejected_form"],
                        {"auto": "malformed_lf"},
                    )
                    auto_results.append((c, entity))
                    counters.form_filter_rejected += 1
                    continue

            llm_candidates.append(c)

        return auto_results, llm_candidates

    def _validate_with_llm(
        self,
        llm_candidates: List[Candidate],
        batch_delay_ms: float,
    ) -> List[ExtractedEntity]:
        """Validate candidates using LLM."""
        assert self.llm_engine is not None, (
            "LLM engine must be initialized for validation"
        )
        results = []

        # Split by generator type
        explicit_candidates = []
        lexicon_candidates = []

        for c in llm_candidates:
            gen = c.generator_type.value if c.generator_type else ""
            if "syntax" in gen.lower() or "glossary" in gen.lower():
                explicit_candidates.append(c)
            else:
                lexicon_candidates.append(c)

        print(f"  Batch (explicit pairs): {len(explicit_candidates)}")
        print(f"  Individual (lexicon): {len(lexicon_candidates)}")

        # Haiku screening (optional)
        if self.enable_haiku_screening and lexicon_candidates:
            print(
                f"\n  Haiku screening {len(lexicon_candidates)} lexicon candidates..."
            )
            haiku_start = time.time()
            needs_review, haiku_rejected = self.llm_engine.fast_reject_batch(
                lexicon_candidates,
                haiku_model="claude-3-5-haiku-20241022",
                batch_size=20,
            )
            haiku_time = time.time() - haiku_start
            print(
                f"    Haiku rejected: {len(haiku_rejected)}, Needs review: {len(needs_review)}, Time: {haiku_time:.2f}s"
            )

            for entity in haiku_rejected:
                candidate = next(
                    (
                        c
                        for c in lexicon_candidates
                        if str(c.id) == str(entity.candidate_id)
                    ),
                    None,
                )
                if candidate:
                    self.logger.log_validation(
                        candidate,
                        entity,
                        entity.raw_llm_response,
                        haiku_time * 1000 / len(lexicon_candidates),
                    )
                results.append(entity)

            lexicon_candidates = needs_review

        # Batch validate explicit pairs
        if explicit_candidates:
            try:
                val_start = time.time()
                batch_results = self.llm_engine.verify_candidates_batch(
                    explicit_candidates, batch_size=10
                )
                elapsed_ms = (time.time() - val_start) * 1000

                for candidate, entity in zip(explicit_candidates, batch_results):
                    self.logger.log_validation(
                        candidate,
                        entity,
                        entity.raw_llm_response,
                        elapsed_ms / len(explicit_candidates),
                    )
                    results.append(entity)
            except Exception as e:
                print(f"  [WARN] Batch error, falling back to individual: {e}")
                for candidate in explicit_candidates:
                    try:
                        entity = self.llm_engine.verify_candidate(candidate)
                        self.logger.log_validation(
                            candidate, entity, entity.raw_llm_response, 0
                        )
                        results.append(entity)
                    except Exception as e2:
                        self.logger.log_error(candidate, str(e2))

        # Batch validation for lexicon matches (10-15x faster than individual calls)
        if lexicon_candidates:
            try:
                val_start = time.time()
                batch_results = self.llm_engine.verify_candidates_batch(
                    lexicon_candidates, batch_size=15
                )
                elapsed_ms = (time.time() - val_start) * 1000

                for candidate, entity in zip(lexicon_candidates, batch_results):
                    self.logger.log_validation(
                        candidate,
                        entity,
                        entity.raw_llm_response,
                        elapsed_ms / len(lexicon_candidates),
                    )
                    results.append(entity)
            except Exception as e:
                print(f"  [WARN] Batch error, falling back to individual: {e}")
                for candidate in lexicon_candidates:
                    try:
                        entity = self.llm_engine.verify_candidate(candidate)
                        self.logger.log_validation(
                            candidate, entity, entity.raw_llm_response, 0
                        )
                        results.append(entity)
                    except Exception as e2:
                        self.logger.log_error(candidate, str(e2))

        return results

    def _search_missing_abbreviations(
        self,
        doc_id: str,
        full_text: str,
        found_sfs: set,
        counters: HeuristicsCounters,
    ) -> List[ExtractedEntity]:
        """Search for abbreviations missed by generators (PASO C/D)."""
        results = []

        # PASO C: Hyphenated abbreviations
        for hyph_sf, hyph_lf in self.heuristics.hyphenated_abbrevs.items():
            if hyph_sf.upper() in found_sfs:
                continue
            pattern = rf"\b{re.escape(hyph_sf)}\b"
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                # If long form is "trial name", enrich from ClinicalTrials.gov API
                actual_lf = hyph_lf
                if hyph_lf.lower() == "trial name":
                    enriched_title = enrich_trial_acronym(hyph_sf)
                    if enriched_title:
                        actual_lf = enriched_title
                        print(f"    Enriched trial '{hyph_sf}': {enriched_title[:60]}...")

                entity = self._create_entity_from_search(
                    doc_id,
                    full_text,
                    match,
                    actual_lf,
                    FieldType.DEFINITION_PAIR,
                    0.85,
                    ["auto_approved_hyphenated"],
                    "hyphenated_detector:v1.0",
                    "hyphenated",
                )
                results.append(entity)
                found_sfs.add(hyph_sf.upper())
                counters.recovered_by_hyphen += 1

        if counters.recovered_by_hyphen > 0:
            print(f"  Hyphenated detected (PASO C): {counters.recovered_by_hyphen}")

        # Direct search abbreviations
        for direct_sf, direct_lf in self.heuristics.direct_search_abbrevs.items():
            if direct_sf.upper() in found_sfs:
                continue
            pattern = (
                re.escape(direct_sf)
                if " " in direct_sf
                else rf"\b{re.escape(direct_sf)}\b"
            )
            match = re.search(pattern, full_text)
            if match:
                entity = self._create_entity_from_search(
                    doc_id,
                    full_text,
                    match,
                    direct_lf,
                    FieldType.DEFINITION_PAIR,
                    0.85,
                    ["auto_approved_direct_search"],
                    "direct_search:v1.0",
                    "direct_search",
                )
                results.append(entity)
                found_sfs.add(direct_sf.upper())
                counters.recovered_by_direct_search += 1

        if counters.recovered_by_direct_search > 0:
            print(f"  Direct search detected: {counters.recovered_by_direct_search}")

        return results

    def _extract_sf_only_with_llm(
        self,
        doc_id: str,
        full_text: str,
        found_sfs: set,
        counters: HeuristicsCounters,
    ) -> List[ExtractedEntity]:
        """PASO D: Extract SF-only using LLM."""
        print("\n  Running LLM SF-only extractor (PASO D)...")
        if self.claude_client is None:
            return []
        results = []

        # Sample chunks
        chunk_size = 3000
        text_chunks = [
            full_text[i : i + chunk_size]
            for i in range(0, min(len(full_text), 15000), chunk_size)
        ]

        sf_extraction_prompt = """You are an expert at identifying medical/scientific abbreviations and their definitions in text.

Given the following text, identify ALL abbreviations and their long form definitions. An abbreviation is typically:
- 2-10 characters long
- Contains uppercase letters (e.g., "FDA", "eGFR", "IL-6", "C3a", "AN")
- May contain numbers or hyphens
- Represents a longer term/concept

IMPORTANT RULES:
1. Only return abbreviations that EXACTLY appear in the text (case-sensitive)
2. Extract the full long form/definition if it appears in the text (e.g., "Acanthosis nigricans (AN)" → sf: "AN", lf: "Acanthosis nigricans")
3. Do NOT include common words (the, and, for, etc.)
4. Do NOT include author names, months, or country names
5. Include mixed-case abbreviations like "eGFR", "IgA", "C3a"
6. If the long form is not explicitly defined in the text, set lf to null

Already found (DO NOT include these): {already_found}

Text to analyze:
{text}

Return a JSON array of objects with "sf" (short form) and "lf" (long form, or null if not defined).
Example: [{{"sf": "AN", "lf": "Acanthosis nigricans"}}, {{"sf": "FDA", "lf": null}}]
Return ONLY the JSON array, nothing else."""

        llm_sf_candidates: Dict[str, Optional[str]] = {}  # sf -> lf mapping
        llm_errors = 0

        for chunk in text_chunks:
            try:
                prompt = sf_extraction_prompt.format(
                    already_found=", ".join(sorted(found_sfs)[:50]), text=chunk
                )
                response = self.claude_client.complete_json_any(
                    system_prompt="You are an abbreviation extraction assistant. Return only valid JSON arrays of objects.",
                    user_prompt=prompt,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=1000,
                    top_p=1.0,
                )
                if isinstance(response, list):
                    for item in response:
                        if isinstance(item, dict):
                            sf = item.get("sf", "")
                            lf = item.get("lf")
                            if isinstance(sf, str) and sf.strip():
                                sf_clean = sf.strip()
                                # Keep the LF if we found one, or update if new one is better
                                if sf_clean not in llm_sf_candidates or (
                                    lf and not llm_sf_candidates.get(sf_clean)
                                ):
                                    llm_sf_candidates[sf_clean] = (
                                        lf.strip()
                                        if isinstance(lf, str) and lf
                                        else None
                                    )
                        elif isinstance(item, str) and item.strip():
                            # Backward compatibility: handle plain strings
                            sf_clean = item.strip()
                            if sf_clean not in llm_sf_candidates:
                                llm_sf_candidates[sf_clean] = None
            except Exception:
                llm_errors += 1

        print(
            f"    LLM chunks: {len(text_chunks)}, errors: {llm_errors}, candidates: {len(llm_sf_candidates)}"
        )

        # Validate and add LLM-found SFs (now with LFs when available)
        blacklist = self.heuristics.common_words | self.heuristics.sf_blacklist

        for sf_candidate, lf_candidate in llm_sf_candidates.items():
            sf_upper = sf_candidate.upper()

            if sf_upper in found_sfs or sf_upper in blacklist:
                continue

            # Verify SF exists in text
            if sf_candidate not in full_text:
                match = re.search(
                    rf"\b{re.escape(sf_candidate)}\b", full_text, re.IGNORECASE
                )
                if not match:
                    continue
                sf_candidate = match.group()

            # Basic form validation
            if (
                len(sf_candidate) < 2
                or len(sf_candidate) > 15
                or not any(c.isupper() for c in sf_candidate)
            ):
                continue

            # Get context for form validation
            idx = full_text.find(sf_candidate)
            if idx == -1:
                idx = full_text.lower().find(sf_candidate.lower())
            temp_ctx = full_text[max(0, idx - 200):idx + len(sf_candidate) + 200] if idx >= 0 else ""

            # Apply the same form validation used for other candidates
            if not is_valid_sf_form(
                sf_candidate,
                temp_ctx,
                self.heuristics.allowed_2letter_sfs,
                self.heuristics.allowed_mixed_case
            ):
                continue

            # Find context
            idx = full_text.find(sf_candidate)
            if idx == -1:
                idx = full_text.lower().find(sf_candidate.lower())

            context_snippet = extract_context_snippet(
                full_text, idx, idx + len(sf_candidate)
            )
            ctx_hash = hash_string(context_snippet)

            primary = EvidenceSpan(
                text=context_snippet,
                location=Coordinate(page_num=1),
                scope_ref=ctx_hash,
                start_char_offset=max(0, idx - (idx - 100 if idx > 100 else 0)),
                end_char_offset=max(0, idx - (idx - 100 if idx > 100 else 0))
                + len(sf_candidate),
            )

            # Lexicon fallback: if LLM found SF but no LF, check rare disease lexicon
            lexicon_fallback_used = False
            if not lf_candidate:
                sf_upper = sf_candidate.upper()
                sf_lower = sf_candidate.lower()
                if sf_upper in self.rare_disease_lookup:
                    candidate_lf = self.rare_disease_lookup[sf_upper]
                    lf_lower = candidate_lf.lower()
                    # Filter out known wrong SF -> LF pairs and bad long forms
                    if (sf_lower, lf_lower) not in WRONG_EXPANSION_BLACKLIST and lf_lower not in BAD_LONG_FORMS:
                        lf_candidate = candidate_lf
                        lexicon_fallback_used = True

            prov = ProvenanceMetadata(
                pipeline_version=PIPELINE_VERSION,
                run_id=self.run_id,
                doc_fingerprint="llm_sf_extraction",
                generator_name=GeneratorType.LEXICON_MATCH,
                rule_version="llm_sf_lf_extractor:v2.0",
                lexicon_source="orchestrator:llm_extraction"
                + (":lexicon_fallback" if lexicon_fallback_used else ""),
            )

            # Determine field type based on whether we have LF
            has_lf = bool(lf_candidate)
            field_type = (
                FieldType.DEFINITION_PAIR if has_lf else FieldType.SHORT_FORM_ONLY
            )
            confidence = 0.85 if has_lf else 0.75

            entity = ExtractedEntity(
                candidate_id=uuid.uuid4(),
                doc_id=doc_id,
                field_type=field_type,
                short_form=sf_candidate,
                long_form=lf_candidate,
                primary_evidence=primary,
                supporting_evidence=[],
                status=ValidationStatus.VALIDATED,
                confidence_score=confidence,
                rejection_reason=None,
                validation_flags=["llm_extracted"] if has_lf else ["llm_sf_extracted"],
                provenance=prov,
                raw_llm_response={"auto": "llm_sf_lf_extractor"},
            )
            results.append(entity)
            found_sfs.add(sf_upper)
            counters.recovered_by_llm_sf_only += 1

        print(f"  LLM extracted (PASO D): {counters.recovered_by_llm_sf_only}")
        return results

    def _normalize_results(
        self, results: List[ExtractedEntity], full_text: str
    ) -> List[ExtractedEntity]:
        """Stage 3.5: Normalize, disambiguate, and deduplicate results.

        Args:
            results: List of entities to normalize
            full_text: Pre-built full document text (avoids rebuilding)

        Pipeline:
            1. Normalize: Map to canonical forms (term_mapper)
            2. Disambiguate: Resolve ambiguous SF meanings (disambiguator)
            3. Deduplicate: Merge same-SF entries, pick best LF (deduplicator)
        """
        # Check if normalization is disabled in config
        if not self.use_normalization:
            print("\n[4/12] Normalization SKIPPED (disabled in config)")
            return results

        print("\n[4/12] Normalizing, disambiguating & deduplicating...")

        # Step 1: Normalize
        normalized_count = 0
        for i, entity in enumerate(results):
            normalized = self.term_mapper.normalize(entity)
            if "normalized" in (normalized.validation_flags or []):
                normalized_count += 1
            results[i] = normalized

        # Step 1.5: NCT enrichment (fetch trial titles from ClinicalTrials.gov)
        nct_enriched_count = 0
        if self.nct_enricher is not None:
            nct_enriched_count = self._enrich_nct_entities(results)

        # Step 2: Disambiguate
        results = self.disambiguator.resolve(results, full_text)
        disambiguated_count = sum(
            1 for r in results if "disambiguated" in (r.validation_flags or [])
        )

        # Step 3: Deduplicate (merge same SF with different LFs)
        count_before = len(results)
        results = self.deduplicator.deduplicate(results)
        count_after = len(results)
        dedup_removed = count_before - count_after

        print(f"  Normalized: {normalized_count}")
        if nct_enriched_count > 0:
            print(f"  NCT enriched: {nct_enriched_count}")
        print(f"  Disambiguated: {disambiguated_count}")
        print(f"  Deduplicated: {dedup_removed} duplicates merged")

        return results

    def _enrich_nct_entities(self, results: List[ExtractedEntity]) -> int:
        """Enrich NCT identifiers with trial titles from ClinicalTrials.gov.

        Args:
            results: List of entities to check for NCT IDs

        Returns:
            Number of entities enriched
        """
        if self.nct_enricher is None:
            return 0

        import re

        nct_pattern = re.compile(r"^NCT\d{8}$", re.IGNORECASE)
        enriched_count = 0

        for i, entity in enumerate(results):
            # Skip if already has long_form
            if entity.long_form:
                continue

            # Check if short_form is an NCT ID
            sf = (entity.short_form or "").strip().upper()
            if not nct_pattern.match(sf):
                continue

            # Fetch trial info from ClinicalTrials.gov
            info = self.nct_enricher.enrich(sf)
            if info and info.long_form:
                # Update entity with trial title
                new_flags = list(entity.validation_flags or [])
                if "nct_enriched" not in new_flags:
                    new_flags.append("nct_enriched")

                results[i] = entity.model_copy(update={
                    "long_form": info.long_form,
                    "validation_flags": new_flags,
                })
                enriched_count += 1

        return enriched_count

    # =========================================================================
    # MAIN PROCESS METHOD
    # =========================================================================

    def process_pdf(
        self, pdf_path: str | Path, batch_delay_ms: Optional[float] = None
    ) -> List[ExtractedEntity]:
        """Process a single PDF through the full pipeline."""
        delay: float = (
            batch_delay_ms
            if batch_delay_ms is not None
            else (self.batch_delay_ms or 100.0)
        )

        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"\n{'=' * 60}")
        print(f"Processing: {pdf_path_obj.name}")
        print(f"{'=' * 60}")

        # Stage 1: Parse
        doc = self._parse_pdf(pdf_path_obj)

        # Stage 2: Generate candidates (abbreviations conditional on config)
        unique_candidates: List[Candidate] = []
        full_text = ""
        results: List[ExtractedEntity] = []
        counters = HeuristicsCounters()

        # Build full_text (always needed for other extractors)
        full_text = "\n".join(
            block.text for block in doc.iter_linear_blocks()
        )

        if self.extract_abbreviations:
            # Run abbreviation extraction
            unique_candidates, full_text = self._generate_candidates(doc)

            # Check if LLM validation is disabled
            if not self.use_llm_validation:
                print("\n[3/12] Validation SKIPPED")
                self._export_results(pdf_path_obj, [], unique_candidates)
            else:
                # Filter candidates
                needs_validation, corroborated_sfs, word_counts, filtered_count = (
                    self._filter_candidates(unique_candidates, full_text)
                )

                # Apply heuristics
                auto_results, llm_candidates = self._apply_heuristics(
                    needs_validation, counters
                )

                # Print stats
                frequent_sfs = sum(
                    1
                    for c in unique_candidates
                    if c.generator_type == GeneratorType.LEXICON_MATCH
                    and c.short_form.upper() not in corroborated_sfs
                    and word_counts.get(c.short_form.upper(), 0) >= 2
                )

                print("\n[3/12] Validating candidates with Claude...")
                print(f"  Corroborated SFs: {len(corroborated_sfs)}")
                print(f"  Frequent SFs (2+): {frequent_sfs}")
                print(f"  Filtered (lexicon-only, rare): {filtered_count}")
                print(f"  Auto-approved stats: {counters.recovered_by_stats_whitelist}")
                print(f"  Auto-approved country: {counters.recovered_by_country_code}")
                print(f"  Auto-rejected blacklist: {counters.blacklisted_fp_count}")
                print(f"  Auto-rejected context: {counters.context_rejected}")
                print(f"  Auto-rejected trial IDs: {counters.trial_id_excluded}")
                print(f"  Auto-rejected common words: {counters.common_word_rejected}")
                print(f"  Candidates for LLM: {len(llm_candidates)}")

                start = time.time()

                # Log auto results
                for candidate, entity in auto_results:
                    self.logger.log_validation(candidate, entity, entity.raw_llm_response, 0)
                    results.append(entity)

                # LLM validation
                llm_results = self._validate_with_llm(llm_candidates, delay)
                results.extend(llm_results)

                print(f"  Time: {time.time() - start:.2f}s")

                # Search for missing abbreviations
                doc_id = str(pdf_path_obj.stem)
                found_sfs = {
                    r.short_form.upper()
                    for r in results
                    if r.status == ValidationStatus.VALIDATED
                }

                search_results = self._search_missing_abbreviations(
                    doc_id, full_text, found_sfs, counters
                )
                results.extend(search_results)

                # LLM SF-only extraction
                sf_only_results = self._extract_sf_only_with_llm(
                    doc_id, full_text, found_sfs, counters
                )
                results.extend(sf_only_results)
        else:
            print("\n[Abbreviation detection] SKIPPED (disabled in config)")

        # Normalize (pass full_text to avoid rebuilding)
        results = self._normalize_results(results, full_text)

        # Disease detection (conditional on config)
        disease_results: List[ExtractedDisease] = []
        if self.extract_diseases:
            disease_results = self._process_diseases(doc, pdf_path_obj)
        else:
            print("\n[Disease detection] SKIPPED (disabled in config)")

        # Drug detection (conditional on config)
        drug_results: List[ExtractedDrug] = []
        if self.extract_drugs:
            drug_results = self._process_drugs(doc, pdf_path_obj)
        else:
            print("\n[Drug detection] SKIPPED (disabled in config)")

        # Pharma company detection (conditional on config)
        pharma_results: List[ExtractedPharma] = []
        if self.extract_pharma:
            pharma_results = self._process_pharma(doc, pdf_path_obj)
        else:
            print("\n[Pharma detection] SKIPPED (disabled in config)")

        # Author detection (conditional on config)
        author_results: List[ExtractedAuthor] = []
        if self.extract_authors:
            author_results = self._process_authors(doc, pdf_path_obj, full_text)
        else:
            print("\n[Author detection] SKIPPED (disabled in config)")

        # Citation detection (conditional on config)
        citation_results: List[ExtractedCitation] = []
        if self.extract_citations:
            citation_results = self._process_citations(doc, pdf_path_obj, full_text)
        else:
            print("\n[Citation detection] SKIPPED (disabled in config)")

        # Feasibility extraction (conditional on config)
        feasibility_results: List[FeasibilityCandidate] = []
        if self.extract_feasibility:
            feasibility_results = self._process_feasibility(doc, pdf_path_obj, full_text)
        else:
            print("\n[Feasibility extraction] SKIPPED (disabled in config)")

        # Document metadata extraction (conditional on config)
        doc_metadata: Optional[DocumentMetadata] = None
        if self.extract_doc_metadata:
            doc_metadata = self._process_document_metadata(
                doc, pdf_path_obj, full_text[:5000]
            )
        else:
            print("\n[Document metadata] SKIPPED (disabled in config)")

        # Stage 4: Summary & Export
        print("\n[12/12] Writing summary...")
        self.logger.write_summary()
        self.logger.print_summary()
        counters.log_summary()

        self._export_results(
            pdf_path_obj, results, unique_candidates, counters,
            disease_results=disease_results if disease_results else None,
            drug_results=drug_results if drug_results else None,
            pharma_results=pharma_results if pharma_results else None,
        )

        # Export disease results
        if disease_results:
            self._export_disease_results(pdf_path_obj, disease_results)

        # Export drug results
        if drug_results:
            self._export_drug_results(pdf_path_obj, drug_results)

        # Export pharma results
        if pharma_results:
            self._export_pharma_results(pdf_path_obj, pharma_results)

        # Export author results
        if author_results:
            self._export_author_results(pdf_path_obj, author_results)

        # Export citation results
        if citation_results:
            self._export_citation_results(pdf_path_obj, citation_results)

        # Export feasibility results
        if feasibility_results:
            self._export_feasibility_results(pdf_path_obj, feasibility_results, doc)

        # Export images
        if doc is not None:
            self._export_images(pdf_path_obj, doc)

        # Export tables as images (conditional on config)
        if doc is not None and self.extract_tables:
            self._export_tables(pdf_path_obj, doc)

        # Export document metadata
        if doc_metadata:
            self._export_document_metadata(pdf_path_obj, doc_metadata)

        # Print validated abbreviations with provenance
        validated = [r for r in results if r.status == ValidationStatus.VALIDATED]
        if validated:
            print(f"\nValidated abbreviations ({len(validated)}):")
            for v in validated:
                src = ""
                if v.provenance and v.provenance.lexicon_source:
                    # Shorten long lexicon names for display
                    lex = v.provenance.lexicon_source
                    if lex.startswith("2025_08_"):
                        lex = lex.replace("2025_08_", "").replace(".json", "").replace(".tsv", "")
                    src = f" [{lex}]"
                print(f"  * {v.short_form} -> {v.long_form or '(no expansion)'}{src}")

        # Print validated diseases with provenance
        validated_diseases = [
            d for d in disease_results if d.status == ValidationStatus.VALIDATED
        ]
        if validated_diseases:
            print(f"\nValidated diseases ({len(validated_diseases)}):")
            for d in validated_diseases:
                codes = []
                if d.icd10_code:
                    codes.append(f"ICD-10:{d.icd10_code}")
                if d.orpha_code:
                    codes.append(f"ORPHA:{d.orpha_code}")
                code_str = f" [{', '.join(codes)}]" if codes else ""
                src = ""
                if d.provenance and d.provenance.lexicon_source:
                    lex = d.provenance.lexicon_source
                    if lex.startswith("disease_lexicon_"):
                        lex = lex.replace("disease_lexicon_", "").replace(".json", "")
                    src = f" ({lex})"
                print(f"  * {d.preferred_label}{code_str}{src}")

        # Print validated drugs with provenance
        validated_drugs = [
            d for d in drug_results if d.status == ValidationStatus.VALIDATED
        ]
        if validated_drugs:
            print(f"\nValidated drugs ({len(validated_drugs)}):")
            for d in validated_drugs:
                phase = f" ({d.development_phase})" if d.development_phase else ""
                compound = f" [{d.compound_id}]" if d.compound_id else ""
                src = ""
                if d.provenance and d.provenance.lexicon_source:
                    lex = d.provenance.lexicon_source
                    if lex.startswith("2025_08_"):
                        lex = lex.replace("2025_08_", "").replace(".json", "")
                    src = f" <{lex}>"
                print(f"  * {d.preferred_name}{compound}{phase}{src}")

        # Print validated pharma companies with provenance
        validated_pharma = [
            p for p in pharma_results if p.status == ValidationStatus.VALIDATED
        ]
        if validated_pharma:
            print(f"\nValidated pharma companies ({len(validated_pharma)}):")
            for p in validated_pharma:
                hq = f" ({p.headquarters})" if p.headquarters else ""
                parent = f" [parent: {p.parent_company}]" if p.parent_company else ""
                print(f"  * {p.canonical_name}{hq}{parent}")

        # Print feasibility summary
        if feasibility_results:
            print(f"\nFeasibility information ({len(feasibility_results)} items):")
            # Group by type
            by_type: Dict[str, List] = {}
            for f in feasibility_results:
                # Handle both FeasibilityCandidate (has field_type) and NERCandidate (has category)
                if hasattr(f, 'field_type') and f.field_type is not None:
                    key = f.field_type.value.split("_")[0]  # Get main category
                elif hasattr(f, 'category'):
                    key = f.category.upper()
                else:
                    key = "OTHER"
                by_type.setdefault(key, []).append(f)
            for ftype, items in sorted(by_type.items()):
                print(f"  {ftype}: {len(items)} items")

        # Print document metadata summary
        if doc_metadata:
            print("\nDocument metadata:")
            if doc_metadata.classification:
                cls = doc_metadata.classification.primary_type
                print(f"  Type: {cls.code} - {cls.name} (conf: {cls.confidence:.2f})")
            if doc_metadata.description:
                print(f"  Title: {doc_metadata.description.title}")
            if doc_metadata.date_extraction and doc_metadata.date_extraction.primary_date:
                pd = doc_metadata.date_extraction.primary_date
                print(f"  Date: {pd.date.strftime('%Y-%m-%d')} (source: {pd.source.value})")

        return results

    # =========================================================================
    # DISEASE DETECTION METHODS
    # =========================================================================

    def _process_diseases(self, doc, pdf_path: Path) -> List[ExtractedDisease]:
        """
        Process document for disease mentions.

        Returns validated disease entities.
        """
        if self.disease_detector is None:
            return []

        print("\n[5/12] Detecting disease mentions...")
        start = time.time()

        # Generate disease candidates
        candidates = self.disease_detector.extract(doc)
        print(f"  Disease candidates: {len(candidates)}")

        if not candidates:
            print(f"  Time: {time.time() - start:.2f}s")
            return []

        # Auto-validate based on source (specialized lexicons are high trust)
        results: List[ExtractedDisease] = []
        for candidate in candidates:
            # Specialized lexicons (PAH, ANCA, IgAN) are auto-validated
            is_specialized = "specialized" in candidate.generator_type.value

            entity = self._create_disease_entity(
                candidate=candidate,
                status=ValidationStatus.VALIDATED,
                confidence=candidate.initial_confidence + candidate.confidence_boost,
                flags=["auto_validated_lexicon"]
                if is_specialized
                else ["lexicon_match"],
            )
            results.append(entity)

        # Normalize diseases
        if self.disease_normalizer is not None:
            results = self.disease_normalizer.normalize_batch(results)

        # PubTator enrichment
        if self.disease_enricher is not None:
            print("  Enriching with PubTator3...")
            results = self.disease_enricher.enrich_batch(results, verbose=True)

        validated_count = len(
            [r for r in results if r.status == ValidationStatus.VALIDATED]
        )
        print(f"  Validated diseases: {validated_count}")
        print(f"  Time: {time.time() - start:.2f}s")

        return results

    def _create_disease_entity(
        self,
        candidate: DiseaseCandidate,
        status: ValidationStatus,
        confidence: float,
        flags: List[str],
    ) -> ExtractedDisease:
        """Create ExtractedDisease from a DiseaseCandidate."""
        context = (candidate.context_text or "").strip()
        ctx_hash = hash_string(context) if context else "no_context"

        primary = EvidenceSpan(
            text=context,
            location=candidate.context_location,
            scope_ref=ctx_hash,
            start_char_offset=0,
            end_char_offset=len(context),
        )

        # Extract primary codes from identifiers
        icd10 = None
        icd11 = None
        snomed = None
        mondo = None
        orpha = None
        umls = None
        mesh = None

        for ident in candidate.identifiers:
            if ident.system in ("ICD-10", "ICD-10-CM") and not icd10:
                icd10 = ident.code
            elif ident.system == "ICD-11" and not icd11:
                icd11 = ident.code
            elif ident.system in ("SNOMED-CT", "SNOMED") and not snomed:
                snomed = ident.code
            elif ident.system == "MONDO" and not mondo:
                mondo = ident.code
            elif ident.system in ("ORPHA", "Orphanet") and not orpha:
                orpha = ident.code
            elif ident.system == "UMLS" and not umls:
                umls = ident.code
            elif ident.system == "MeSH" and not mesh:
                mesh = ident.code

        return ExtractedDisease(
            candidate_id=candidate.id,
            doc_id=candidate.doc_id,
            matched_text=candidate.matched_text,
            preferred_label=candidate.preferred_label,
            abbreviation=candidate.abbreviation,
            identifiers=candidate.identifiers,
            icd10_code=icd10,
            icd11_code=icd11,
            snomed_code=snomed,
            mondo_id=mondo,
            orpha_code=orpha,
            umls_cui=umls,
            mesh_id=mesh,
            primary_evidence=primary,
            supporting_evidence=[],
            status=status,
            confidence_score=min(1.0, confidence),
            validation_flags=flags,
            is_rare_disease=candidate.is_rare_disease,
            disease_category=candidate.disease_category,
            provenance=candidate.provenance,
        )

    def _get_output_dir(self, pdf_path: Path) -> Path:
        """Get output directory for a PDF file.

        Creates a folder with the same name as the PDF (without extension)
        in the same directory as the PDF file.

        Args:
            pdf_path: Path to the PDF file being processed

        Returns:
            Path to output directory (created if it doesn't exist)
        """
        if self.output_dir:
            out_dir = self.output_dir
        else:
            # Create folder named after PDF in same directory
            out_dir = pdf_path.parent / pdf_path.stem

        # Ensure directory exists
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _export_disease_results(
        self, pdf_path: Path, results: List[ExtractedDisease]
    ) -> None:
        """Export disease detection results to separate JSON file."""
        out_dir = self._get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        validated = [r for r in results if r.status == ValidationStatus.VALIDATED]
        rejected = [r for r in results if r.status == ValidationStatus.REJECTED]
        ambiguous = [r for r in results if r.status == ValidationStatus.AMBIGUOUS]

        # Build export entries
        disease_entries: List[DiseaseExportEntry] = []
        for entity in validated:
            codes = {
                "icd10": entity.icd10_code,
                "icd11": entity.icd11_code,
                "snomed": entity.snomed_code,
                "mondo": entity.mondo_id,
                "orpha": entity.orpha_code,
                "umls": entity.umls_cui,
                "mesh": entity.mesh_id,
            }

            all_identifiers = [
                {"system": i.system, "code": i.code, "display": i.display}
                for i in entity.identifiers
            ]

            entry = DiseaseExportEntry(
                matched_text=entity.matched_text,
                preferred_label=entity.preferred_label,
                abbreviation=entity.abbreviation,
                confidence=entity.confidence_score,
                is_rare_disease=entity.is_rare_disease,
                category=entity.disease_category,
                codes=codes,
                all_identifiers=all_identifiers,
                context=entity.primary_evidence.text
                if entity.primary_evidence
                else None,
                page=entity.primary_evidence.location.page_num
                if entity.primary_evidence
                else None,
                lexicon_source=entity.provenance.lexicon_source
                if entity.provenance
                else None,
                validation_flags=entity.validation_flags,
                mesh_aliases=entity.mesh_aliases,
                pubtator_normalized_name=entity.pubtator_normalized_name,
                enrichment_source=entity.enrichment_source,
            )
            disease_entries.append(entry)

        # Build export document
        export_doc = DiseaseExportDocument(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            document=pdf_path.name,
            document_path=str(pdf_path.absolute()),
            pipeline_version=PIPELINE_VERSION,
            total_candidates=len(results),
            total_validated=len(validated),
            total_rejected=len(rejected),
            total_ambiguous=len(ambiguous),
            diseases=disease_entries,
        )

        # Write to file
        out_file = out_dir / f"diseases_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(export_doc.model_dump_json(indent=2))

        print(f"  Disease export: {out_file.name}")

    # =========================================================================
    # DRUG DETECTION METHODS
    # =========================================================================

    def _process_drugs(self, doc, pdf_path: Path) -> List[ExtractedDrug]:
        """
        Process document for drug mentions.

        Returns validated drug entities.
        """
        if self.drug_detector is None:
            return []

        print("\n[6/12] Detecting drug mentions...")
        start = time.time()

        # Run drug detection
        candidates = self.drug_detector.detect(doc)
        print(f"  Drug candidates: {len(candidates)}")

        # Convert candidates to ExtractedDrug (auto-validated for lexicon matches)
        results: List[ExtractedDrug] = []
        for candidate in candidates:
            # Determine if auto-validated (Alexion, investigational, FDA)
            is_specialized = candidate.generator_type in {
                DrugGeneratorType.LEXICON_ALEXION,
                DrugGeneratorType.LEXICON_INVESTIGATIONAL,
                DrugGeneratorType.PATTERN_COMPOUND_ID,
            }

            entity = self._candidate_to_extracted_drug(
                candidate,
                status=ValidationStatus.VALIDATED,
                confidence=candidate.initial_confidence if is_specialized else 0.7,
                flags=["auto_validated_lexicon"]
                if is_specialized
                else ["lexicon_match"],
            )
            results.append(entity)

        # PubTator enrichment
        if self.drug_enricher is not None:
            print("  Enriching with PubTator3...")
            results = self.drug_enricher.enrich_batch(results, verbose=True)

        validated_count = len(
            [r for r in results if r.status == ValidationStatus.VALIDATED]
        )
        print(f"  Validated drugs: {validated_count}")
        print(f"  Time: {time.time() - start:.2f}s")

        return results

    def _process_pharma(self, doc, pdf_path: Path) -> List[ExtractedPharma]:
        """
        Process document for pharma company mentions.

        Returns validated pharma company entities.
        """
        if self.pharma_detector is None:
            return []

        print("\n[7/12] Detecting pharma company mentions...")
        start = time.time()

        # Build full text for detection
        full_text = " ".join(
            block.text
            for block in doc.iter_linear_blocks()
            if block.text
        )

        doc_fingerprint = hash_string(full_text[:5000])

        # Run pharma detection
        candidates = self.pharma_detector.detect(
            doc_graph=doc,
            doc_id=doc.doc_id,
            doc_fingerprint=doc_fingerprint,
            full_text=full_text,
        )
        print(f"  Pharma candidates: {len(candidates)}")

        # Validate candidates (auto-validated for lexicon matches)
        results = self.pharma_detector.validate_candidates(candidates)

        validated_count = len(
            [r for r in results if r.status == ValidationStatus.VALIDATED]
        )
        print(f"  Validated pharma companies: {validated_count}")
        print(f"  Time: {time.time() - start:.2f}s")

        return results

    def _candidate_to_extracted_drug(
        self,
        candidate: DrugCandidate,
        status: ValidationStatus,
        confidence: float,
        flags: List[str],
    ) -> ExtractedDrug:
        """Convert DrugCandidate to ExtractedDrug."""
        return ExtractedDrug(
            candidate_id=candidate.id,
            doc_id=candidate.doc_id,
            matched_text=candidate.matched_text,
            preferred_name=candidate.preferred_name,
            brand_name=candidate.brand_name,
            compound_id=candidate.compound_id,
            identifiers=candidate.identifiers,
            primary_evidence=EvidenceSpan(
                text=candidate.context_text,
                location=candidate.context_location,
                scope_ref="drug_detection",
                start_char_offset=0,
                end_char_offset=len(candidate.context_text),
            ),
            status=status,
            confidence_score=confidence,
            validation_flags=flags,
            drug_class=candidate.drug_class,
            mechanism=candidate.mechanism,
            development_phase=candidate.development_phase,
            is_investigational=candidate.is_investigational,
            sponsor=candidate.sponsor,
            conditions=candidate.conditions,
            nct_id=candidate.nct_id,
            dosage_form=candidate.dosage_form,
            route=candidate.route,
            marketing_status=candidate.marketing_status,
            provenance=candidate.provenance,
        )

    def _export_drug_results(
        self, pdf_path: Path, results: List[ExtractedDrug]
    ) -> None:
        """Export drug detection results to separate JSON file."""
        out_dir = self._get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        validated = [r for r in results if r.status == ValidationStatus.VALIDATED]
        rejected = [r for r in results if r.status == ValidationStatus.REJECTED]
        investigational = [r for r in validated if r.is_investigational]

        # Build export entries
        drug_entries: List[DrugExportEntry] = []
        for entity in validated:
            codes = {
                "rxcui": entity.rxcui,
                "mesh": entity.mesh_id,
                "ndc": entity.ndc_code,
                "drugbank": entity.drugbank_id,
                "unii": entity.unii,
            }

            all_identifiers = [
                {"system": i.system, "code": i.code, "display": i.display}
                for i in entity.identifiers
            ]

            entry = DrugExportEntry(
                matched_text=entity.matched_text,
                preferred_name=entity.preferred_name,
                brand_name=entity.brand_name,
                compound_id=entity.compound_id,
                confidence=entity.confidence_score,
                is_investigational=entity.is_investigational,
                drug_class=entity.drug_class,
                mechanism=entity.mechanism,
                development_phase=entity.development_phase,
                sponsor=entity.sponsor,
                conditions=entity.conditions,
                nct_id=entity.nct_id,
                dosage_form=entity.dosage_form,
                route=entity.route,
                marketing_status=entity.marketing_status,
                codes=codes,
                all_identifiers=all_identifiers,
                context=entity.primary_evidence.text
                if entity.primary_evidence
                else None,
                page=entity.primary_evidence.location.page_num
                if entity.primary_evidence
                else None,
                lexicon_source=entity.provenance.lexicon_source
                if entity.provenance
                else None,
                validation_flags=entity.validation_flags,
                mesh_aliases=entity.mesh_aliases,
                pubtator_normalized_name=entity.pubtator_normalized_name,
                enrichment_source=entity.enrichment_source,
            )
            drug_entries.append(entry)

        # Build export document
        export_doc = DrugExportDocument(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            document=pdf_path.name,
            document_path=str(pdf_path.absolute()),
            pipeline_version=PIPELINE_VERSION,
            total_candidates=len(results),
            total_validated=len(validated),
            total_rejected=len(rejected),
            total_investigational=len(investigational),
            drugs=drug_entries,
        )

        # Write to file
        out_file = out_dir / f"drugs_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(export_doc.model_dump_json(indent=2))

        print(f"  Drug export: {out_file.name}")

    def _export_pharma_results(
        self, pdf_path: Path, results: List[ExtractedPharma]
    ) -> None:
        """Export pharma company detection results to separate JSON file."""
        out_dir = self._get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        validated = [r for r in results if r.status == ValidationStatus.VALIDATED]

        # Build export entries
        pharma_entries: List[PharmaExportEntry] = []
        for entity in validated:
            entry = PharmaExportEntry(
                matched_text=entity.matched_text,
                canonical_name=entity.canonical_name,
                full_name=entity.full_name,
                headquarters=entity.headquarters,
                parent_company=entity.parent_company,
                subsidiaries=entity.subsidiaries,
                confidence=entity.confidence_score,
                context=entity.primary_evidence.text
                if entity.primary_evidence
                else None,
                page=entity.primary_evidence.location.page_num
                if entity.primary_evidence
                else None,
                lexicon_source=entity.provenance.lexicon_source
                if entity.provenance
                else None,
            )
            pharma_entries.append(entry)

        # Build export document
        unique_companies = set(e.canonical_name for e in pharma_entries)
        export_doc = PharmaExportDocument(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            document=pdf_path.name,
            document_path=str(pdf_path.absolute()),
            pipeline_version=PIPELINE_VERSION,
            total_detected=len(results),
            unique_companies=len(unique_companies),
            companies=pharma_entries,
        )

        # Write to file
        out_file = out_dir / f"pharma_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(export_doc.model_dump_json(indent=2))

        print(f"  Pharma export: {out_file.name}")

    # =========================================================================
    # AUTHOR EXTRACTION METHODS
    # =========================================================================

    def _process_authors(
        self, doc, pdf_path: Path, full_text: str
    ) -> List[ExtractedAuthor]:
        """
        Process document for author/investigator mentions.

        Returns validated author entities.
        """
        if self.author_detector is None:
            return []

        print("\n[8/12] Detecting author/investigator mentions...")
        start = time.time()

        # Build document fingerprint
        doc_fingerprint = pdf_path.stem

        # Run author detection
        candidates = self.author_detector.detect(
            doc_graph=doc,
            doc_id=doc.doc_id if doc else pdf_path.stem,
            doc_fingerprint=doc_fingerprint,
            full_text=full_text,
        )
        print(f"  Author candidates found: {len(candidates)}")

        # Validate candidates
        results = self.author_detector.validate_candidates(candidates)

        validated_count = len(
            [r for r in results if r.status == ValidationStatus.VALIDATED]
        )
        print(f"  Validated authors: {validated_count}")
        print(f"  Author detection took {time.time() - start:.2f}s")

        return results

    def _export_author_results(
        self, pdf_path: Path, results: List[ExtractedAuthor]
    ) -> None:
        """Export author detection results to separate JSON file."""
        out_dir = self._get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        validated = [r for r in results if r.status == ValidationStatus.VALIDATED]

        # Build export entries
        author_entries: List[AuthorExportEntry] = []
        unique_names: set = set()
        for entity in validated:
            unique_names.add(entity.full_name.lower())
            entry = AuthorExportEntry(
                full_name=entity.full_name,
                role=entity.role.value,
                affiliation=entity.affiliation,
                email=entity.email,
                orcid=entity.orcid,
                confidence=entity.confidence_score,
                context=entity.primary_evidence.text
                if entity.primary_evidence
                else None,
                page=entity.primary_evidence.location.page_num
                if entity.primary_evidence
                else None,
            )
            author_entries.append(entry)

        # Build export document
        export_doc = AuthorExportDocument(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            document=pdf_path.name,
            document_path=str(pdf_path.absolute()),
            pipeline_version=PIPELINE_VERSION,
            total_detected=len(results),
            unique_authors=len(unique_names),
            authors=author_entries,
        )

        # Write to file
        out_file = out_dir / f"authors_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(export_doc.model_dump_json(indent=2))

        print(f"  Author export: {out_file.name}")

    # =========================================================================
    # CITATION EXTRACTION METHODS
    # =========================================================================

    def _process_citations(
        self, doc, pdf_path: Path, full_text: str
    ) -> List[ExtractedCitation]:
        """
        Process document for citation/reference mentions.

        Returns validated citation entities.
        """
        if self.citation_detector is None:
            return []

        print("\n[9/12] Detecting citation/reference mentions...")
        start = time.time()

        # Build document fingerprint
        doc_fingerprint = pdf_path.stem

        # Run citation detection
        candidates = self.citation_detector.detect(
            doc_graph=doc,
            doc_id=doc.doc_id if doc else pdf_path.stem,
            doc_fingerprint=doc_fingerprint,
            full_text=full_text,
        )
        print(f"  Citation candidates found: {len(candidates)}")

        # Validate candidates
        results = self.citation_detector.validate_candidates(candidates)

        validated_count = len(
            [r for r in results if r.status == ValidationStatus.VALIDATED]
        )
        print(f"  Validated citations: {validated_count}")
        print(f"  Citation detection took {time.time() - start:.2f}s")

        return results

    def _export_citation_results(
        self, pdf_path: Path, results: List[ExtractedCitation]
    ) -> None:
        """Export citation detection results to separate JSON file with API validation."""
        from A_core.A11_citation_models import CitationValidation, CitationValidationSummary

        out_dir = self._get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        validated = [r for r in results if r.status == ValidationStatus.VALIDATED]

        # Build export entries
        citation_entries: List[CitationExportEntry] = []
        unique_ids: set = set()
        for entity in validated:
            # Track unique identifiers
            if entity.pmid:
                unique_ids.add(f"pmid:{entity.pmid}")
            if entity.pmcid:
                unique_ids.add(f"pmcid:{entity.pmcid}")
            if entity.doi:
                unique_ids.add(f"doi:{entity.doi}")
            if entity.nct:
                unique_ids.add(f"nct:{entity.nct}")

            entry = CitationExportEntry(
                pmid=entity.pmid,
                pmcid=entity.pmcid,
                doi=entity.doi,
                nct=entity.nct,
                url=entity.url,
                citation_text=entity.citation_text,
                citation_number=entity.citation_number,
                confidence=entity.confidence_score,
                page=entity.primary_evidence.location.page_num
                if entity.primary_evidence
                else None,
            )
            citation_entries.append(entry)

        # Run API validation on citations
        validation_summary = None
        if citation_entries:
            print("  Validating citations via API...")
            validator = CitationValidator({"validate_urls": False})  # Skip URL validation for speed
            valid_count = 0
            invalid_count = 0
            error_count = 0

            for entry in citation_entries:
                # Validate primary identifier (prefer DOI > NCT > PMID)
                validation_result = None

                if entry.doi:
                    result = validator.validate_doi(entry.doi)
                    validation_result = CitationValidation(
                        is_valid=result.is_valid,
                        resolved_url=result.resolved_url,
                        title=result.metadata.get("title"),
                        error=result.error_message,
                    )
                elif entry.nct:
                    result = validator.validate_nct(entry.nct)
                    validation_result = CitationValidation(
                        is_valid=result.is_valid,
                        resolved_url=result.resolved_url,
                        title=result.metadata.get("title"),
                        status=result.metadata.get("status"),
                        error=result.error_message,
                    )
                elif entry.pmid:
                    result = validator.validate_pmid(entry.pmid)
                    validation_result = CitationValidation(
                        is_valid=result.is_valid,
                        resolved_url=result.resolved_url,
                        title=result.metadata.get("title"),
                        error=result.error_message,
                    )

                if validation_result:
                    entry.validation = validation_result
                    if validation_result.is_valid:
                        valid_count += 1
                        print(f"    ✓ {entry.doi or entry.nct or entry.pmid}: valid")
                    elif validation_result.error:
                        error_count += 1
                        print(f"    ✗ {entry.doi or entry.nct or entry.pmid}: {validation_result.error}")
                    else:
                        invalid_count += 1
                        print(f"    ✗ {entry.doi or entry.nct or entry.pmid}: not found")

            validation_summary = CitationValidationSummary(
                total_validated=valid_count + invalid_count + error_count,
                valid_count=valid_count,
                invalid_count=invalid_count,
                error_count=error_count,
            )

        # Build export document
        export_doc = CitationExportDocument(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            document=pdf_path.name,
            document_path=str(pdf_path.absolute()),
            pipeline_version=PIPELINE_VERSION,
            total_detected=len(results),
            unique_identifiers=len(unique_ids),
            validation_summary=validation_summary,
            citations=citation_entries,
        )

        # Write to file
        out_file = out_dir / f"citations_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(export_doc.model_dump_json(indent=2))

        print(f"  Citation export: {out_file.name}")

    # =========================================================================
    # FEASIBILITY EXTRACTION METHODS
    # =========================================================================

    def _process_feasibility(
        self, doc, pdf_path: Path, full_text: str
    ) -> List[FeasibilityCandidate]:
        """
        Process document for clinical trial feasibility information.

        Uses LLM-based extraction when available (higher precision),
        falls back to pattern-based extraction otherwise.

        Returns feasibility candidates (eligibility, epidemiology, patient journey, etc.)
        """
        if self.feasibility_detector is None:
            return []

        print("\n[10/12] Extracting feasibility information...")
        start = time.time()

        # Use LLM extraction if available (preferred - more precise structured output)
        if self.llm_feasibility_extractor is not None:
            print("  Using LLM-based extraction...")
            candidates = self.llm_feasibility_extractor.extract(
                doc_graph=doc,
                doc_id=pdf_path.stem,
                doc_fingerprint=pdf_path.stem,
                full_text=full_text,
            )
            self.llm_feasibility_extractor.print_summary()
        else:
            # Fallback to pattern-based extraction
            print("  Using pattern-based extraction...")
            candidates = self.feasibility_detector.extract(doc)
            self.feasibility_detector.print_summary()

        # Enrich with EpiExtract4GARD-v2 (rare disease epidemiology NER)
        if self.epi_enricher is not None:
            print("  Running EpiExtract4GARD-v2 enrichment...")
            epi_start = time.time()
            epi_result = self.epi_enricher.extract(full_text)

            # Convert to EpidemiologyData and add as feasibility candidates
            epi_data_list = epi_result.to_epidemiology_data()
            if epi_data_list:
                for epi_data in epi_data_list:
                    # Create a NERCandidate for each epidemiology finding
                    epi_candidate = NERCandidate(
                        category="epidemiology",
                        text=epi_data.value,
                        evidence_text=epi_data.value,
                        confidence=0.8,
                        source="EpiExtract4GARD-v2",
                        epidemiology_data=epi_data,
                    )
                    candidates.append(epi_candidate)
                print(f"    EpiExtract4GARD: {len(epi_data_list)} epidemiology items")
                print(f"      Locations: {len(epi_result.locations)}")
                print(f"      Epi types: {len(epi_result.epi_types)}")
                print(f"      Statistics: {len(epi_result.statistics)}")
            print(f"    EpiExtract time: {time.time() - epi_start:.2f}s")

        # Enrich with ZeroShotBioNER (ADE, dosage, frequency, route, etc.)
        if self.zeroshot_bioner is not None:
            print("  Running ZeroShotBioNER enrichment...")
            bioner_start = time.time()
            bioner_result = self.zeroshot_bioner.extract(full_text)

            # Add extracted entities as feasibility candidates
            summary = bioner_result.to_summary()
            entity_counts = summary.get("entity_counts", {})
            total_entities = sum(entity_counts.values())

            if total_entities > 0:
                # Add adverse events as candidates
                for ade in bioner_result.adverse_events:
                    ade_candidate = NERCandidate(
                        category="adverse_event",
                        text=ade.text,
                        evidence_text=ade.text,
                        confidence=ade.score,
                        source="ZeroShotBioNER",
                    )
                    candidates.append(ade_candidate)

                # Add drug administration details as candidates
                for dosage in bioner_result.dosages:
                    candidates.append(NERCandidate(
                        category="drug_dosage",
                        text=dosage.text,
                        evidence_text=dosage.text,
                        confidence=dosage.score,
                        source="ZeroShotBioNER",
                    ))
                for freq in bioner_result.frequencies:
                    candidates.append(NERCandidate(
                        category="drug_frequency",
                        text=freq.text,
                        evidence_text=freq.text,
                        confidence=freq.score,
                        source="ZeroShotBioNER",
                    ))
                for route in bioner_result.routes:
                    candidates.append(NERCandidate(
                        category="drug_route",
                        text=route.text,
                        evidence_text=route.text,
                        confidence=route.score,
                        source="ZeroShotBioNER",
                    ))
                for duration in bioner_result.durations:
                    candidates.append(NERCandidate(
                        category="treatment_duration",
                        text=duration.text,
                        evidence_text=duration.text,
                        confidence=duration.score,
                        source="ZeroShotBioNER",
                    ))

                print(f"    ZeroShotBioNER: {total_entities} entities extracted")
                print(f"      ADE: {entity_counts.get('ADE', 0)}")
                print(f"      Dosage: {entity_counts.get('dosage', 0)}")
                print(f"      Frequency: {entity_counts.get('frequency', 0)}")
                print(f"      Route: {entity_counts.get('route', 0)}")
                print(f"      Duration: {entity_counts.get('duration', 0)}")
            print(f"    ZeroShotBioNER time: {time.time() - bioner_start:.2f}s")

        # Enrich with d4data/biomedical-ner-all (84 entity types)
        if self.biomedical_ner is not None:
            print("  Running BiomedicalNER enrichment...")
            biomed_start = time.time()
            biomed_result = self.biomedical_ner.extract(full_text)

            # Add extracted entities as feasibility candidates
            summary = biomed_result.to_summary()
            category_counts = summary.get("category_counts", {})
            total_entities = sum(category_counts.values())

            if total_entities > 0:
                # Add symptoms as candidates
                for entity in biomed_result.clinical:
                    if entity.entity_type == "Sign_symptom":
                        candidates.append(NERCandidate(
                            category="symptom",
                            text=entity.text,
                            evidence_text=entity.text,
                            confidence=entity.score,
                            source="BiomedicalNER",
                        ))
                    elif entity.entity_type == "Diagnostic_procedure":
                        candidates.append(NERCandidate(
                            category="diagnostic_procedure",
                            text=entity.text,
                            evidence_text=entity.text,
                            confidence=entity.score,
                            source="BiomedicalNER",
                        ))
                    elif entity.entity_type == "Therapeutic_procedure":
                        candidates.append(NERCandidate(
                            category="therapeutic_procedure",
                            text=entity.text,
                            evidence_text=entity.text,
                            confidence=entity.score,
                            source="BiomedicalNER",
                        ))
                    elif entity.entity_type == "Lab_value":
                        candidates.append(NERCandidate(
                            category="lab_value",
                            text=entity.text,
                            evidence_text=entity.text,
                            confidence=entity.score,
                            source="BiomedicalNER",
                        ))
                    elif entity.entity_type == "Outcome":
                        candidates.append(NERCandidate(
                            category="outcome",
                            text=entity.text,
                            evidence_text=entity.text,
                            confidence=entity.score,
                            source="BiomedicalNER",
                        ))

                # Add demographics as candidates
                for entity in biomed_result.demographics:
                    candidates.append(NERCandidate(
                        category=f"demographics_{entity.entity_type.lower()}",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="BiomedicalNER",
                    ))

                print(f"    BiomedicalNER: {total_entities} entities extracted")
                print(f"      Clinical: {category_counts.get('clinical', 0)}")
                print(f"      Demographics: {category_counts.get('demographics', 0)}")
                print(f"      Temporal: {category_counts.get('temporal', 0)}")
                print(f"      Anatomical: {category_counts.get('anatomical', 0)}")
            print(f"    BiomedicalNER time: {time.time() - biomed_start:.2f}s")

        # Enrich with PatientJourneyNER (diagnostic delay, treatment lines, care pathway)
        if self.patient_journey_enricher is not None:
            print("  Running PatientJourneyNER enrichment...")
            pj_start = time.time()
            pj_result = self.patient_journey_enricher.extract(full_text)

            # Add extracted entities as feasibility candidates
            summary = pj_result.to_summary()
            total_entities = summary.get("total", 0)

            if total_entities > 0:
                # Add diagnostic delays
                for entity in pj_result.diagnostic_delays:
                    candidates.append(NERCandidate(
                        category="diagnostic_delay",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="PatientJourneyNER",
                    ))

                # Add treatment lines
                for entity in pj_result.treatment_lines:
                    candidates.append(NERCandidate(
                        category="treatment_line",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="PatientJourneyNER",
                    ))

                # Add care pathway steps
                for entity in pj_result.care_pathway_steps:
                    candidates.append(NERCandidate(
                        category="care_pathway_step",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="PatientJourneyNER",
                    ))

                # Add surveillance frequencies
                for entity in pj_result.surveillance_frequencies:
                    candidates.append(NERCandidate(
                        category="surveillance_frequency",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="PatientJourneyNER",
                    ))

                # Add pain points
                for entity in pj_result.pain_points:
                    candidates.append(NERCandidate(
                        category="pain_point",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="PatientJourneyNER",
                    ))

                # Add recruitment touchpoints
                for entity in pj_result.recruitment_touchpoints:
                    candidates.append(NERCandidate(
                        category="recruitment_touchpoint",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="PatientJourneyNER",
                    ))

                print(f"      diagnostic_delay: {summary.get('diagnostic_delay', 0)}")
                print(f"      treatment_line: {summary.get('treatment_line', 0)}")
                print(f"      care_pathway_step: {summary.get('care_pathway_step', 0)}")
                print(f"      surveillance_frequency: {summary.get('surveillance_frequency', 0)}")
                print(f"      pain_point: {summary.get('pain_point', 0)}")
                print(f"      recruitment_touchpoint: {summary.get('recruitment_touchpoint', 0)}")
            print(f"    PatientJourney: {total_entities} entities extracted")
            print(f"    PatientJourney time: {time.time() - pj_start:.2f}s")

        # Enrich with RegistryNER (registry names, sizes, data types, access policies)
        if self.registry_enricher is not None:
            print("  Running RegistryNER enrichment...")
            reg_start = time.time()
            reg_result = self.registry_enricher.extract(full_text)

            # Add extracted entities as feasibility candidates
            summary = reg_result.to_summary()
            total_entities = summary.get("total", 0)

            if total_entities > 0:
                # Add registry names
                for entity in reg_result.registry_names:
                    candidates.append(NERCandidate(
                        category="registry_name",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="RegistryNER",
                    ))

                # Add registry sizes
                for entity in reg_result.registry_sizes:
                    candidates.append(NERCandidate(
                        category="registry_size",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="RegistryNER",
                    ))

                # Add geographic coverage
                for entity in reg_result.geographic_coverages:
                    candidates.append(NERCandidate(
                        category="geographic_coverage",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="RegistryNER",
                    ))

                # Add data types
                for entity in reg_result.data_types:
                    candidates.append(NERCandidate(
                        category="data_types",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="RegistryNER",
                    ))

                # Add access policies
                for entity in reg_result.access_policies:
                    candidates.append(NERCandidate(
                        category="access_policy",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="RegistryNER",
                    ))

                # Add eligibility criteria
                for entity in reg_result.eligibility_criteria:
                    candidates.append(NERCandidate(
                        category="eligibility_criteria",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="RegistryNER",
                    ))

                print(f"      registry_name: {summary.get('registry_name', 0)}")
                print(f"      registry_size: {summary.get('registry_size', 0)}")
                print(f"      geographic_coverage: {summary.get('geographic_coverage', 0)}")
                print(f"      data_types: {summary.get('data_types', 0)}")
                print(f"      access_policy: {summary.get('access_policy', 0)}")
                print(f"      eligibility_criteria: {summary.get('eligibility_criteria', 0)}")

                # Report linked registries
                linked = reg_result.get_linked_registries()
                if linked:
                    print(f"    Linked to known registries: {len(linked)}")
                    for lr in linked[:3]:  # Show first 3
                        print(f"      {lr.get('extracted_text')} -> {lr.get('full_name', 'N/A')}")
            print(f"    RegistryNER: {total_entities} entities extracted")
            print(f"    RegistryNER time: {time.time() - reg_start:.2f}s")

        # Enrich with GeneticNER (gene symbols, HGVS variants, HPO, ORDO)
        if self.genetic_enricher is not None:
            print("  Running GeneticNER enrichment...")
            gen_start = time.time()
            gen_result = self.genetic_enricher.extract(full_text)

            # Add extracted entities as feasibility candidates
            summary = gen_result.to_summary()
            total_entities = summary.get("total", 0)

            if total_entities > 0:
                # Add gene symbols
                for entity in gen_result.gene_symbols:
                    candidates.append(NERCandidate(
                        category="gene_symbol",
                        text=entity.normalized,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="GeneticNER",
                    ))

                # Add HGVS variants
                for entity in gen_result.variants_hgvs:
                    candidates.append(NERCandidate(
                        category="variant_hgvs",
                        text=entity.normalized,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="GeneticNER",
                    ))

                # Add rsID variants
                for entity in gen_result.variants_rsid:
                    candidates.append(NERCandidate(
                        category="variant_rsid",
                        text=entity.normalized,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="GeneticNER",
                    ))

                # Add HPO terms
                for entity in gen_result.hpo_terms:
                    candidates.append(NERCandidate(
                        category="hpo_term",
                        text=entity.normalized,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="GeneticNER",
                    ))

                # Add ORDO disease codes
                for entity in gen_result.disease_ordo:
                    candidates.append(NERCandidate(
                        category="disease_ordo",
                        text=entity.normalized,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="GeneticNER",
                    ))

                print(f"      gene_symbols: {summary.get('gene_symbols', 0)}")
                print(f"      variants_hgvs: {summary.get('variants_hgvs', 0)}")
                print(f"      variants_rsid: {summary.get('variants_rsid', 0)}")
                print(f"      hpo_terms: {summary.get('hpo_terms', 0)}")
                print(f"      disease_ordo: {summary.get('disease_ordo', 0)}")

                # Report gene-variant associations
                genes_with_variants = gen_result.get_genes_with_variants()
                if genes_with_variants:
                    print(f"    Gene-variant associations: {len(genes_with_variants)} genes")
                    for gene, variants in list(genes_with_variants.items())[:3]:  # Show first 3
                        print(f"      {gene}: {len(variants)} variant(s)")
            print(f"    GeneticNER: {total_entities} entities extracted")
            print(f"    GeneticNER time: {time.time() - gen_start:.2f}s")

        # Deduplicate overlapping NER spans (keep highest confidence)
        pre_dedup_count = len(candidates)
        candidates, dedup_result = deduplicate_feasibility_candidates(candidates)
        if dedup_result.merged_count > 0:
            print("  Span deduplication:")
            print(f"    Before: {pre_dedup_count} → After: {len(candidates)}")
            print(f"    Merged: {dedup_result.merged_count} overlapping spans")
            summary = dedup_result.to_summary()
            print(f"    By source: {summary.get('by_source', {})}")

        print(f"  Feasibility items: {len(candidates)}")
        print(f"  Time: {time.time() - start:.2f}s")

        return candidates

    def _export_feasibility_results(
        self, pdf_path: Path, results: List[FeasibilityCandidate], doc: Optional["DocumentGraph"] = None
    ) -> None:
        """Export feasibility extraction results to JSON file."""
        out_dir = self._get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract trial identifiers (NCT, EudraCT, CTIS, etc.)
        trial_ids: List[TrialIdentifier] = []
        if doc is not None:
            try:
                id_extractor = IdentifierExtractor()
                extracted_ids = id_extractor.extract_identifiers(doc)

                # Registry URLs
                registry_urls = {
                    "NCT": "https://clinicaltrials.gov/study/",
                    "EudraCT": "https://www.clinicaltrialsregister.eu/ctr-search/search?query=",
                    "ISRCTN": "https://www.isrctn.com/",
                }

                for ext_id in extracted_ids:
                    # Only include trial-related identifiers
                    if ext_id.id_type in {IdentifierType.NCT, IdentifierType.EUDRACT, IdentifierType.ISRCTN}:
                        url = None
                        registry = None
                        if ext_id.id_type == IdentifierType.NCT:
                            registry = "ClinicalTrials.gov"
                            url = f"{registry_urls['NCT']}NCT{ext_id.normalized}"
                        elif ext_id.id_type == IdentifierType.EUDRACT:
                            registry = "EU Clinical Trials Register"
                            url = f"{registry_urls['EudraCT']}{ext_id.normalized}"
                        elif ext_id.id_type == IdentifierType.ISRCTN:
                            registry = "ISRCTN Registry"
                            url = f"{registry_urls['ISRCTN']}ISRCTN{ext_id.normalized}"

                        trial_ids.append(TrialIdentifier(
                            id_type=ext_id.id_type.value,
                            value=ext_id.value,
                            registry=registry,
                            url=url,
                            title=ext_id.long_form,
                        ))
            except Exception as e:
                print(f"  [WARN] Trial ID extraction failed: {e}")

        # Group by field type
        from A_core.A07_feasibility_models import FeasibilityExportEntry, EvidenceExport

        study_design_data = None
        operational_burden_data = None
        screening_flow_data = None
        eligibility_inclusion = []
        eligibility_exclusion = []
        epidemiology = []
        patient_journey = []
        endpoints = []
        sites = []

        def _convert_evidence(evidence_list) -> List[EvidenceExport]:
            """Convert EvidenceSpan list to EvidenceExport list for export."""
            result = []
            for ev in (evidence_list or []):
                result.append(EvidenceExport(
                    page=ev.page,
                    quote=ev.quote,
                    source_node_id=ev.source_node_id,
                    source_doc_id=ev.source_doc_id,
                ))
            return result

        for r in results:
            # Handle NERCandidate objects - merge epidemiology into main export
            if not hasattr(r, 'field_type') or r.field_type is None:
                # Check if this is an NERCandidate with epidemiology data
                if hasattr(r, 'epidemiology_data') and r.epidemiology_data is not None:
                    epi_entry = FeasibilityExportEntry(
                        field_type="EPIDEMIOLOGY_NER",
                        text=getattr(r, 'text', ''),
                        section="epidemiology",
                        page=None,
                        structured_data={
                            "data_type": r.epidemiology_data.data_type,
                            "value": r.epidemiology_data.value,
                            "population": r.epidemiology_data.population,
                            "source": getattr(r, 'source', 'NER'),
                        },
                        confidence=getattr(r, 'confidence', 0.8),
                        evidence=[],
                    )
                    epidemiology.append(epi_entry)
                continue

            # Handle study design separately (single object, not list)
            if r.field_type.value == "STUDY_DESIGN" and r.study_design:
                study_design_data = r.study_design.model_dump()
                continue

            # Handle operational burden (single object)
            if r.field_type.value == "OPERATIONAL_BURDEN" and r.operational_burden:
                operational_burden_data = r.operational_burden.model_dump()
                continue

            # Handle screening flow (single object)
            if r.field_type.value == "SCREENING_FLOW" and r.screening_flow:
                screening_flow_data = r.screening_flow.model_dump()
                continue

            # Convert evidence from the candidate
            evidence_export = _convert_evidence(r.evidence)

            # Propagate page from evidence if not set on parent
            # Rule: if evidence[0].page exists, use it for parent.page
            page_num = r.page_number
            if page_num is None and evidence_export:
                page_num = evidence_export[0].page

            entry = FeasibilityExportEntry(
                field_type=r.field_type.value,
                text=r.matched_text,
                section=r.section_name,
                page=page_num,
                structured_data=None,
                confidence=r.confidence,
                evidence=evidence_export,
            )

            # Add structured data based on type, including evidence from nested objects
            if r.eligibility_criterion:
                entry.structured_data = {
                    "type": r.eligibility_criterion.criterion_type.value,
                    "category": r.eligibility_criterion.category,
                    "derived": r.eligibility_criterion.derived_variables,
                }
                # Add evidence from eligibility criterion if not already present
                if not entry.evidence and r.eligibility_criterion.evidence:
                    entry.evidence = _convert_evidence(r.eligibility_criterion.evidence)
                    # Also propagate page from this evidence
                    if entry.page is None and entry.evidence:
                        entry.page = entry.evidence[0].page
            elif r.epidemiology_data:
                entry.structured_data = {
                    "data_type": r.epidemiology_data.data_type,
                    "value": r.epidemiology_data.value,
                    "population": r.epidemiology_data.population,
                }
            elif r.patient_journey_phase:
                entry.structured_data = {
                    "phase": r.patient_journey_phase.phase_type.value,
                    "duration": r.patient_journey_phase.duration,
                }
            elif r.study_endpoint:
                entry.structured_data = {
                    "type": r.study_endpoint.endpoint_type.value,
                    "name": r.study_endpoint.name,
                    "measure": r.study_endpoint.measure,
                    "timepoint": r.study_endpoint.timepoint,
                    "analysis_method": r.study_endpoint.analysis_method,
                }
            elif r.study_site:
                entry.structured_data = {
                    "country": r.study_site.country,
                    "site_count": r.study_site.site_count,
                }

            # Categorize
            if "ELIGIBILITY_INCLUSION" in r.field_type.value:
                eligibility_inclusion.append(entry)
            elif "ELIGIBILITY_EXCLUSION" in r.field_type.value:
                eligibility_exclusion.append(entry)
            elif "EPIDEMIOLOGY" in r.field_type.value:
                epidemiology.append(entry)
            elif "PATIENT_JOURNEY" in r.field_type.value:
                patient_journey.append(entry)
            elif "ENDPOINT" in r.field_type.value:
                endpoints.append(entry)
            elif "SITE" in r.field_type.value:
                sites.append(entry)

        # Build export document
        export_doc = FeasibilityExportDocument(
            doc_id=pdf_path.stem,
            doc_filename=pdf_path.name,
            trial_ids=trial_ids,
            study_design=study_design_data,
            eligibility_inclusion=eligibility_inclusion,
            eligibility_exclusion=eligibility_exclusion,
            epidemiology=epidemiology,
            patient_journey=patient_journey,
            endpoints=endpoints,
            sites=sites,
            operational_burden=operational_burden_data,
            screening_flow=screening_flow_data,
        )

        # Write to file
        out_file = out_dir / f"feasibility_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(export_doc.model_dump_json(indent=2))

        print(f"  Feasibility export: {out_file.name}")

    def _render_figure_with_padding(
        self,
        pdf_path: Path,
        page_num: int,
        bbox: Tuple[float, float, float, float],
        dpi: int = 200,
        padding: int = 15,
        bottom_padding: int = 150,
        right_padding: int = 100,
    ) -> Optional[str]:
        """
        Re-render a figure from PDF with extra padding for captions/legends.

        Args:
            pdf_path: Path to PDF file
            page_num: 1-indexed page number
            bbox: (x0, y0, x1, y1) bounding box in PDF points
            dpi: Resolution for rendering
            padding: Extra points around left/top
            bottom_padding: Extra points below figure for captions (default 150pt)
            right_padding: Extra points to right for multi-panel figures (default 100pt)

        Returns:
            Base64-encoded PNG string, or None if rendering fails
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return None

        try:
            doc = fitz.open(str(pdf_path))
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return None

            page = doc[page_num - 1]  # 0-indexed
            page_width = page.rect.width
            page_height = page.rect.height

            x0, y0, x1, y1 = bbox

            # Handle coordinate space issues (Unstructured uses higher DPI)
            if x1 > page_width * 1.1 or y1 > page_height * 1.1:
                # Scale from pixel space to PDF point space
                max_coord = max(x1, y1)
                max_page = max(page_width, page_height)
                dpi_ratio = max_page / max_coord

                x0 = x0 * dpi_ratio
                y0 = y0 * dpi_ratio
                x1 = x1 * dpi_ratio
                y1 = y1 * dpi_ratio

            # For figures that span a significant portion of page, expand to capture full width
            # This handles multi-panel figures where Unstructured may only detect one panel
            # Use aggressive expansion: any figure >15% of page width OR >20% of page height
            # gets expanded to full page width to capture all panels
            figure_width = x1 - x0
            figure_height = y1 - y0
            is_significant_figure = (
                figure_width > page_width * 0.15 or
                figure_height > page_height * 0.20
            )
            if is_significant_figure:
                # Expand to nearly full page width (small margins to avoid edge artifacts)
                margin = 36  # ~0.5 inch margin
                x0 = margin
                x1 = page_width - margin

            # Create clip rectangle with padding
            clip_rect = fitz.Rect(
                max(0, x0 - padding),
                max(0, y0 - padding),
                min(page_width, x1 + right_padding),
                min(page_height, y1 + bottom_padding),
            )

            # Validate clip rect
            if clip_rect.width <= 0 or clip_rect.height <= 0:
                doc.close()
                return None

            # Render to pixmap
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)

            # Convert to base64 PNG
            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            doc.close()
            return img_base64

        except Exception as e:
            print(f"  [WARN] Failed to render figure with padding: {e}")
            return None

    def _export_images(
        self, pdf_path: Path, doc: "DocumentGraph"
    ) -> None:
        """Export extracted images to JSON file with Vision LLM analysis."""
        from B_parsing.B02_doc_graph import ImageType

        # Collect all images
        images = list(doc.iter_images())
        if not images:
            return

        out_dir = self._get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize Vision analyzer if LLM client available
        vision_analyzer = None
        if hasattr(self, 'claude_client') and self.claude_client:
            vision_analyzer = VisionImageAnalyzer(self.claude_client)

        # Build export data
        export_data = {
            "doc_id": pdf_path.stem,
            "doc_filename": pdf_path.name,
            "total_images": len(images),
            "images": []
        }

        for img in images:
            img_data = {
                "page": img.page_num,
                "type": img.image_type.value,
                "caption": img.caption,
                "ocr_text": img.ocr_text,
                "bbox": list(img.bbox.coords) if img.bbox else None,
                "image_base64": img.image_base64,
            }

            # Run Vision LLM analysis based on image type
            if vision_analyzer and img.image_base64:
                if img.image_type == ImageType.FLOWCHART:
                    try:
                        flow_result = vision_analyzer.analyze_flowchart(
                            img.image_base64, img.ocr_text
                        )
                        if flow_result:
                            img_data["vision_analysis"] = {
                                "analysis_type": "patient_flow",
                                "screened": flow_result.screened,
                                "screen_failures": flow_result.screen_failures,
                                "randomized": flow_result.randomized,
                                "completed": flow_result.completed,
                                "discontinued": flow_result.discontinued,
                                "arms": flow_result.arms,
                                "exclusion_reasons": [
                                    {"reason": e.reason, "count": e.count}
                                    for e in flow_result.exclusion_reasons
                                ],
                                "stages": [
                                    {"stage_name": s.stage_name, "count": s.count, "details": s.details}
                                    for s in flow_result.stages
                                ],
                                "notes": flow_result.notes,
                            }
                    except Exception as e:
                        print(f"    [WARN] Flowchart analysis failed: {e}")

                elif img.image_type == ImageType.CHART:
                    try:
                        chart_result = vision_analyzer.analyze_chart(
                            img.image_base64, img.caption
                        )
                        if chart_result:
                            img_data["vision_analysis"] = {
                                "analysis_type": "chart_data",
                                "chart_type": chart_result.chart_type,
                                "title": chart_result.title,
                                "x_axis": chart_result.x_axis,
                                "y_axis": chart_result.y_axis,
                                "data_points": [
                                    {
                                        "label": dp.label,
                                        "value": dp.value,
                                        "unit": dp.unit,
                                        "group": dp.group,
                                    }
                                    for dp in chart_result.data_points
                                ],
                                "statistical_results": chart_result.statistical_results,
                            }
                    except Exception as e:
                        print(f"    [WARN] Chart analysis failed: {e}")

            # Save image as file - re-render with padding if bbox available
            if img.image_base64:
                img_type = img.image_type.value.lower() if img.image_type else "image"
                img_index = len([i for i in export_data['images'] if i.get('page') == img.page_num]) + 1
                img_filename = f"{pdf_path.stem}_{img_type}_page{img.page_num}_{img_index}.png"
                img_path = out_dir / img_filename

                try:
                    # Try to re-render with bottom padding for captions/legends
                    rendered_base64 = None
                    if img.bbox:
                        rendered_base64 = self._render_figure_with_padding(
                            pdf_path=pdf_path,
                            page_num=img.page_num,
                            bbox=img.bbox.coords,
                            bottom_padding=250,  # Extra space for long figure captions/legends
                            right_padding=150,   # Extra space for multi-panel figures
                        )

                    # Use re-rendered image if successful, otherwise use original
                    if rendered_base64:
                        img_bytes = base64.b64decode(rendered_base64)
                        img_data["image_base64"] = rendered_base64  # Update for analysis
                    else:
                        img_bytes = base64.b64decode(img.image_base64)

                    with open(img_path, "wb") as img_file:
                        img_file.write(img_bytes)
                    img_data["saved_file"] = img_filename
                except Exception as e:
                    print(f"    [WARN] Failed to save image {img_filename}: {e}")

            export_data["images"].append(img_data)

        # Write JSON metadata
        out_file = out_dir / f"images_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            import json
            json.dump(export_data, f, indent=2)

        saved_count = sum(1 for img in export_data["images"] if "saved_file" in img)
        analyzed_count = sum(1 for img in export_data["images"] if "vision_analysis" in img)
        print(f"  Images export: {out_file.name} ({len(images)} images, {saved_count} saved, {analyzed_count} analyzed)")

    def _export_tables(
        self, pdf_path: Path, doc: "DocumentGraph"
    ) -> None:
        """Export extracted tables as images to JSON file."""

        # Collect all tables with images
        tables = [t for t in doc.iter_tables() if t.image_base64]
        if not tables:
            return

        out_dir = self._get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build export data
        export_data = {
            "doc_id": pdf_path.stem,
            "doc_filename": pdf_path.name,
            "total_tables": len(tables),
            "tables": []
        }

        for idx, table in enumerate(tables):
            # Determine page number(s)
            if table.is_multipage and table.page_nums:
                page_str = f"pages{table.page_nums[0]}-{table.page_nums[-1]}"
                page_num = table.page_nums[0]
            else:
                page_str = f"page{table.page_num}"
                page_num = table.page_num

            # Calculate row/col counts from cells or logical_rows
            num_rows = len(table.logical_rows) if table.logical_rows else 0
            num_cols = len(table.logical_rows[0]) if table.logical_rows else 0

            # Extract headers from metadata if available
            headers = []
            if table.metadata and "headers" in table.metadata:
                headers_map = table.metadata["headers"]
                if isinstance(headers_map, dict):
                    headers = [headers_map.get(i, f"col_{i}") for i in range(num_cols)]

            table_data = {
                "page": page_num,
                "page_nums": table.page_nums if table.is_multipage else [table.page_num],
                "is_multipage": table.is_multipage,
                "type": table.table_type.value if table.table_type else "UNKNOWN",
                "caption": table.caption,
                "rows": num_rows,
                "cols": num_cols,
                "headers": headers,
                "data": table.logical_rows if table.logical_rows else [],
                "bbox": list(table.bbox.coords) if table.bbox else None,
                "image_base64": table.image_base64,
            }

            # Save table image as file
            if table.image_base64:
                table_type = table.table_type.value.lower() if table.table_type else "table"
                img_filename = f"{pdf_path.stem}_table_{table_type}_{page_str}_{idx + 1}.png"
                img_path = out_dir / img_filename
                try:
                    img_bytes = base64.b64decode(table.image_base64)
                    with open(img_path, "wb") as img_file:
                        img_file.write(img_bytes)
                    table_data["saved_file"] = img_filename
                except Exception as e:
                    print(f"    [WARN] Failed to save table image {img_filename}: {e}")

            export_data["tables"].append(table_data)

        # Write JSON metadata
        out_file = out_dir / f"tables_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            import json
            json.dump(export_data, f, indent=2)

        saved_count = sum(1 for t in export_data["tables"] if "saved_file" in t)
        print(f"  Tables export: {out_file.name} ({len(tables)} tables, {saved_count} saved)")

    # =========================================================================
    # DOCUMENT METADATA METHODS
    # =========================================================================

    def _process_document_metadata(
        self, doc, pdf_path: Path, content_sample: str
    ) -> Optional[DocumentMetadata]:
        """
        Extract document metadata including classification and descriptions.

        Returns DocumentMetadata with file info, PDF metadata, classification, etc.
        """
        if self.doc_metadata_strategy is None:
            return None

        print("\n[11/12] Extracting document metadata...")
        start = time.time()

        try:
            metadata = self.doc_metadata_strategy.extract(
                file_path=str(pdf_path),
                doc_graph=doc,
                doc_id=pdf_path.stem,
                content_sample=content_sample,
            )

            elapsed = time.time() - start
            print(f"  Time: {elapsed:.2f}s")

            return metadata
        except Exception as e:
            print(f"  [WARN] Document metadata extraction failed: {e}")
            return None

    def _export_document_metadata(
        self, pdf_path: Path, metadata: DocumentMetadata
    ) -> None:
        """Export document metadata to JSON file."""
        out_dir = self._get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build simplified export
        export = DocumentMetadataExport(
            doc_id=metadata.doc_id,
            doc_filename=metadata.doc_filename,
            file_size_bytes=metadata.file_metadata.size_bytes if metadata.file_metadata else None,
            file_size_human=metadata.file_metadata.size_human if metadata.file_metadata else None,
            file_extension=metadata.file_metadata.extension if metadata.file_metadata else None,
            pdf_title=metadata.pdf_metadata.title if metadata.pdf_metadata else None,
            pdf_author=metadata.pdf_metadata.author if metadata.pdf_metadata else None,
            pdf_page_count=metadata.pdf_metadata.page_count if metadata.pdf_metadata else None,
            pdf_creation_date=(
                metadata.pdf_metadata.creation_date.isoformat()
                if metadata.pdf_metadata and metadata.pdf_metadata.creation_date
                else None
            ),
            document_type_code=(
                metadata.classification.primary_type.code
                if metadata.classification
                else None
            ),
            document_type_name=(
                metadata.classification.primary_type.name
                if metadata.classification
                else None
            ),
            document_type_group=(
                metadata.classification.primary_type.group
                if metadata.classification
                else None
            ),
            classification_confidence=(
                metadata.classification.primary_type.confidence
                if metadata.classification
                else None
            ),
            title=metadata.description.title if metadata.description else None,
            short_description=(
                metadata.description.short_description if metadata.description else None
            ),
            long_description=(
                metadata.description.long_description if metadata.description else None
            ),
            document_date=(
                metadata.date_extraction.primary_date.date.isoformat()
                if metadata.date_extraction and metadata.date_extraction.primary_date
                else None
            ),
            document_date_source=(
                metadata.date_extraction.primary_date.source.value
                if metadata.date_extraction and metadata.date_extraction.primary_date
                else None
            ),
        )

        # Write to file
        out_file = out_dir / f"metadata_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(export.model_dump_json(indent=2))

        print(f"  Document metadata export: {out_file.name}")

    # =========================================================================
    # EXPORT METHODS
    # =========================================================================

    def _export_extracted_text(self, pdf_path: Path, doc) -> None:
        """Export extracted text to file."""
        out_dir = self._get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = out_dir / f"{pdf_path.stem}_{timestamp}.txt"

        text_lines = []
        current_page = None

        for block in doc.iter_linear_blocks(skip_header_footer=False):
            if block.page_num != current_page:
                if current_page is not None:
                    text_lines.append("")
                text_lines.append(f"--- Page {block.page_num} ---")
                text_lines.append("")
                current_page = block.page_num

            text = (block.text or "").strip()
            if text:
                text_lines.append(text)

        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(text_lines))
            print(f"  Extracted text: {txt_path.name}")
        except Exception as e:
            print(f"  [WARN] Failed to export text: {e}")

    def _export_results(
        self,
        pdf_path: Path,
        results: List[ExtractedEntity],
        candidates: List[Candidate],
        counters: Optional[HeuristicsCounters] = None,
        disease_results: Optional[List[ExtractedDisease]] = None,
        drug_results: Optional[List[ExtractedDrug]] = None,
        pharma_results: Optional[List[ExtractedPharma]] = None,
    ) -> None:
        """Export results to JSON."""
        out_dir = self._get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        export_data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "document": pdf_path.name,
            "document_path": str(pdf_path),
            "total_candidates": len(candidates),
            "total_validated": sum(
                1 for r in results if r.status == ValidationStatus.VALIDATED
            ),
            "total_rejected": sum(
                1 for r in results if r.status == ValidationStatus.REJECTED
            ),
            "total_ambiguous": sum(
                1 for r in results if r.status == ValidationStatus.AMBIGUOUS
            ),
            "heuristics_counters": counters.to_dict() if counters else None,
            "abbreviations": [],
            "diseases": [],
            "drugs": [],
            "pharma": [],
        }

        for entity in results:
            if entity.status == ValidationStatus.VALIDATED:
                lexicon_ids = None
                if entity.provenance.lexicon_ids:
                    lexicon_ids = [
                        {"source": lid.source, "id": lid.id}
                        for lid in entity.provenance.lexicon_ids
                    ]

                export_data["abbreviations"].append(
                    {
                        "short_form": entity.short_form,
                        "long_form": entity.long_form,
                        "confidence": entity.confidence_score,
                        "field_type": entity.field_type.value,
                        "page": entity.primary_evidence.location.page_num
                        if entity.primary_evidence
                        else None,
                        "context_text": entity.primary_evidence.text
                        if entity.primary_evidence
                        else None,
                        "lexicon_source": entity.provenance.lexicon_source,
                        "lexicon_ids": lexicon_ids,
                    }
                )

        # Add diseases to export
        if disease_results:
            for disease in disease_results:
                if disease.status == ValidationStatus.VALIDATED:
                    export_data["diseases"].append(
                        {
                            "name": disease.preferred_label,
                            "matched_text": disease.matched_text,
                            "abbreviation": disease.abbreviation,
                            "confidence": disease.confidence_score,
                            "is_rare": disease.is_rare_disease,
                            "icd10": disease.icd10_code,
                            "orpha": disease.orpha_code,
                        }
                    )

        # Add drugs to export
        if drug_results:
            for drug in drug_results:
                if drug.status == ValidationStatus.VALIDATED:
                    export_data["drugs"].append(
                        {
                            "name": drug.preferred_name,
                            "matched_text": drug.matched_text,
                            "compound_id": drug.compound_id,
                            "confidence": drug.confidence_score,
                            "is_investigational": drug.is_investigational,
                            "phase": drug.development_phase,
                        }
                    )

        # Add pharma companies to export
        if pharma_results:
            for pharma in pharma_results:
                if pharma.status == ValidationStatus.VALIDATED:
                    export_data["pharma"].append(
                        {
                            "name": pharma.canonical_name,
                            "matched_text": pharma.matched_text,
                            "full_name": pharma.full_name,
                            "headquarters": pharma.headquarters,
                            "parent_company": pharma.parent_company,
                            "confidence": pharma.confidence_score,
                        }
                    )

        out_file = out_dir / f"abbreviations_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"\n  Exported: {out_file}")
        run_analysis(export_data, self.gold_json)

    def process_folder(
        self, folder_path: Optional[str] = None, batch_delay_ms: float = 100
    ) -> Dict[str, List[ExtractedEntity]]:
        """Process all PDFs in a folder."""
        folder = Path(folder_path or self.pdf_dir)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        pdf_files = sorted(folder.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {folder}")
            return {}

        print(f"\n{'#' * 60}")
        print(f"BATCH PROCESSING: {len(pdf_files)} PDFs")
        print(f"Folder: {folder}")
        print(f"{'#' * 60}")

        all_results = {}

        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] {pdf_path.name}")
            try:
                all_results[pdf_path.name] = self.process_pdf(
                    str(pdf_path), batch_delay_ms=batch_delay_ms
                )
            except Exception as e:
                print(f"  [WARN] ERROR: {e}")
                all_results[pdf_path.name] = []

        print(f"\n{'#' * 60}")
        print("BATCH COMPLETE")
        print(f"{'#' * 60}")
        print(f"PDFs processed: {len(all_results)}")

        total_validated = sum(
            sum(1 for r in results if r.status == ValidationStatus.VALIDATED)
            for results in all_results.values()
        )
        print(f"Total validated abbreviations: {total_validated}")
        print(f"Log directory: {self.log_dir}")

        return all_results


def main() -> None:
    """
    Process all PDFs in the default folder.

    Initializes the logging system and runs the pipeline orchestrator
    on all PDF files found in the configured input directory.
    """
    # Configure logging for the pipeline run
    configure_logging(
        log_dir="logs",
        log_level=INFO,
        enable_file_logging=True,
        enable_console_logging=False,  # Use print() for console output
    )

    orchestrator = Orchestrator()
    orchestrator.process_folder()
    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
